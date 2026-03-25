from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import Settings, get_settings
from .domain import SpreadPrediction
from .gee_basemap import build_fire_basemap
from .geojson import build_geojson, build_layer_collection, build_model_input_layer_collection
from .model_loader import get_model
from .preprocessing import PreprocessedSample, SamplePreprocessor
from .repository import WildfireCube, WildfireMetadata, WildfireRepository


SCALABLE_ENVIRONMENT_CHANNELS = {
    "viirs_m11": 0,
    "viirs_i2": 1,
    "ndvi": 3,
    "evi2": 4,
    "precip": 5,
    "wind_speed": 6,
}

MODEL_INPUT_CHANNELS = {
    **SCALABLE_ENVIRONMENT_CHANNELS,
    "slope": 12,
    "aspect": 13,
    "elevation": 14,
}


@dataclass(frozen=True)
class InferenceCacheKey:
    fire_id: str
    year: int
    sample_index: int
    scale_signature: Tuple[Tuple[str, float], ...]
    model_fingerprint: str


@dataclass
class CachedInferencePayload:
    metadata: WildfireMetadata
    sample_index: int
    total_samples: int
    probabilities: np.ndarray
    ground_truth: np.ndarray
    observation_dates: Tuple[str, ...]
    target_date: Optional[str]


class InferenceResultCache:
    def __init__(self, ttl_seconds: int, max_entries: int):
        self._ttl_seconds = int(ttl_seconds)
        self._max_entries = int(max_entries)
        self._entries: OrderedDict[InferenceCacheKey, Tuple[float, CachedInferencePayload]] = (
            OrderedDict()
        )
        self._lock = Lock()

    def get(self, key: InferenceCacheKey) -> Optional[CachedInferencePayload]:
        now = self._now()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            expires_at, payload = entry
            if expires_at <= now:
                self._entries.pop(key, None)
                return None
            self._entries.move_to_end(key)
            return payload

    def set(self, key: InferenceCacheKey, payload: CachedInferencePayload) -> None:
        now = self._now()
        expires_at = now + self._ttl_seconds
        with self._lock:
            self._purge_expired(now)
            self._entries[key] = (expires_at, payload)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def stats(self) -> Dict[str, int]:
        now = self._now()
        with self._lock:
            self._purge_expired(now)
            return {
                "size": len(self._entries),
                "ttl_seconds": self._ttl_seconds,
                "max_entries": self._max_entries,
            }

    def _purge_expired(self, now: float) -> None:
        stale_keys = [key for key, (exp, _) in self._entries.items() if exp <= now]
        for key in stale_keys:
            self._entries.pop(key, None)

    @staticmethod
    def _now() -> float:
        from time import monotonic

        return monotonic()


class WildfireService:
    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
        self._repository = WildfireRepository(self._settings)
        self._preprocessor = SamplePreprocessor(self._settings)
        self._model = get_model(self._settings)
        self._model_fingerprint = self._build_model_fingerprint()
        self._inference_cache = InferenceResultCache(
            ttl_seconds=self._settings.cache_ttl_seconds,
            max_entries=self._settings.cache_max_entries,
        )
        self._metrics_lock = Lock()
        self._metrics: Dict[str, float | int] = {
            "rate_limited": 0,
            "cache_hit": 0,
            "cache_miss": 0,
            "inference_requests": 0,
            "in_flight_inference": 0,
            "last_inference_duration_ms": 0.0,
            "last_queue_wait_ms": 0.0,
            "avg_inference_duration_ms": 0.0,
            "avg_queue_wait_ms": 0.0,
        }
        self._inference_duration_total_ms = 0.0
        self._queue_wait_total_ms = 0.0
        self._queue_wait_count = 0

    def health(self) -> Dict[str, object]:
        cache_stats = self._inference_cache.stats()
        with self._metrics_lock:
            metrics = dict(self._metrics)
        return {
            "status": "ok",
            "model_path": str(self._settings.model_path),
            "model_loaded": True,
            "device": str(self._model.device),
            "hdf5_root": str(self._settings.hdf5_root),
            "default_year": self._settings.default_year,
            "infer_max_concurrency": self._settings.infer_max_concurrency,
            "infer_queue_timeout_seconds": self._settings.infer_queue_timeout_seconds,
            "cache_ttl_seconds": self._settings.cache_ttl_seconds,
            "cache_max_entries": self._settings.cache_max_entries,
            "metrics": {
                **metrics,
                "cache_size": cache_stats["size"],
            },
        }

    def record_rate_limited(self) -> None:
        with self._metrics_lock:
            self._metrics["rate_limited"] = int(self._metrics["rate_limited"]) + 1

    def mark_inference_started(self, queue_wait_ms: float) -> None:
        with self._metrics_lock:
            self._metrics["in_flight_inference"] = int(
                self._metrics["in_flight_inference"]
            ) + 1
            self._metrics["last_queue_wait_ms"] = float(queue_wait_ms)
            self._queue_wait_total_ms += float(queue_wait_ms)
            self._queue_wait_count += 1
            self._metrics["avg_queue_wait_ms"] = self._queue_wait_total_ms / max(
                self._queue_wait_count, 1
            )

    def mark_inference_finished(self, inference_duration_ms: float) -> None:
        with self._metrics_lock:
            self._metrics["in_flight_inference"] = max(
                0,
                int(self._metrics["in_flight_inference"]) - 1,
            )
            self._metrics["inference_requests"] = int(
                self._metrics["inference_requests"]
            ) + 1
            self._metrics["last_inference_duration_ms"] = float(inference_duration_ms)
            self._inference_duration_total_ms += float(inference_duration_ms)
            requests = int(self._metrics["inference_requests"])
            self._metrics["avg_inference_duration_ms"] = (
                self._inference_duration_total_ms / max(requests, 1)
            )

    def catalog(
        self, year: Optional[int] = None, limit: int = 50, offset: int = 0
    ) -> List[WildfireMetadata]:
        target_year = year or self._settings.default_year
        entries = [
            entry
            for entry in self._repository.list_year(target_year)
            if entry.has_positive_target
        ]
        start = min(offset, len(entries))
        end = min(start + limit, len(entries))
        return entries[start:end]

    def available_years(self) -> List[int]:
        return self._repository.available_years()

    def overview(
        self, year: Optional[int] = None, limit: int = 200, offset: int = 0
    ) -> List[WildfireMetadata]:
        return self.catalog(year=year, limit=limit, offset=offset)

    def timeline(
        self, fire_id: str, year: Optional[int] = None
    ) -> Tuple[WildfireMetadata, List[Dict[str, object]], int]:
        target_year = year or self._settings.default_year
        metadata = self._repository.get_metadata(fire_id, target_year)
        positive_counts = self._repository.load_target_positive_counts(fire_id, target_year)
        frames: List[Dict[str, object]] = []

        for sample_index, positive_pixels in enumerate(positive_counts):
            if positive_pixels <= 0:
                continue
            observation_dates = self._resolve_observation_dates(
                metadata.img_dates, sample_index
            )
            target_date = self._resolve_target_date(
                metadata.img_dates,
                sample_index,
                self._settings.n_leading_observations,
            )
            label = " -> ".join(filter(None, [*observation_dates, target_date]))
            frames.append(
                {
                    "sampleIndex": sample_index,
                    "sampleOffset": sample_index - metadata.samples,
                    "observationDates": list(observation_dates),
                    "targetDate": target_date,
                    "groundTruthPositivePixels": positive_pixels,
                    "label": label,
                }
            )

        default_sample_index = frames[-1]["sampleIndex"] if frames else 0
        return metadata, frames, default_sample_index

    def find_spread(
        self,
        fire_id: str,
        year: Optional[int] = None,
        sample_offset: int = -1,
        threshold: Optional[float] = None,
        environment_scales: Optional[Dict[str, float]] = None,
    ) -> Tuple[SpreadPrediction, Dict]:
        target_year = year or self._settings.default_year
        cube = self._repository.load_cube(fire_id, target_year)
        leads = self._settings.n_leading_observations
        total_samples = max(cube.cube.shape[0] - leads, 0)
        resolved_sample_index = self._resolve_sample_index(sample_offset, total_samples)

        payload, _ = self._get_or_create_inference_payload(
            cube,
            target_year=target_year,
            sample_index=resolved_sample_index,
            environment_scales=environment_scales,
            include_prepared_cube=False,
        )

        mask_threshold = (
            threshold if threshold is not None else self._settings.probability_threshold
        )
        prediction = self._build_prediction_from_payload(payload, mask_threshold)
        geojson = build_geojson(prediction)
        return prediction, geojson

    def find_layers(
        self,
        fire_id: str,
        year: Optional[int] = None,
        sample_index: Optional[int] = None,
        threshold: Optional[float] = None,
        model_input: Optional[str] = None,
        environment_scales: Optional[Dict[str, float]] = None,
    ) -> Tuple[SpreadPrediction, Dict, Dict]:
        _ = model_input
        requested_sample = -1 if sample_index is None else sample_index
        target_year = year or self._settings.default_year
        cube = self._repository.load_cube(fire_id, target_year)
        leads = self._settings.n_leading_observations
        total_samples = max(cube.cube.shape[0] - leads, 0)
        resolved_sample = self._resolve_sample_index(requested_sample, total_samples)

        payload, prepared_cube = self._get_or_create_inference_payload(
            cube,
            target_year=target_year,
            sample_index=resolved_sample,
            environment_scales=environment_scales,
            include_prepared_cube=True,
        )
        if prepared_cube is None:
            prepared_cube = self._apply_environment_scales(
                cube.cube,
                resolved_sample,
                environment_scales,
            )

        mask_threshold = (
            threshold if threshold is not None else self._settings.probability_threshold
        )
        prediction = self._build_prediction_from_payload(payload, mask_threshold)
        geojson = build_geojson(prediction)
        layers = build_layer_collection(prediction)
        model_input_arrays = self._extract_model_input_arrays(
            prepared_cube,
            prediction.sample_index,
            prediction.probabilities.shape,
            list(MODEL_INPUT_CHANNELS.keys()),
        )
        layers["modelInputs"] = build_model_input_layer_collection(
            prediction,
            model_input_arrays,
        )
        layers["basemap"] = build_fire_basemap(
            prediction.metadata,
            prediction.target_date,
        )
        return prediction, geojson, layers

    def _get_or_create_inference_payload(
        self,
        cube: WildfireCube,
        target_year: int,
        sample_index: int,
        environment_scales: Optional[Dict[str, float]],
        include_prepared_cube: bool,
    ) -> Tuple[CachedInferencePayload, Optional[np.ndarray]]:
        scale_signature = self._normalize_environment_scales(environment_scales)
        cache_key = InferenceCacheKey(
            fire_id=cube.metadata.fire_id,
            year=target_year,
            sample_index=sample_index,
            scale_signature=scale_signature,
            model_fingerprint=self._model_fingerprint,
        )

        cached_payload = self._inference_cache.get(cache_key)
        if cached_payload is not None:
            self._record_cache_hit()
            prepared_cube = (
                self._apply_environment_scales(cube.cube, sample_index, environment_scales)
                if include_prepared_cube
                else None
            )
            return cached_payload, prepared_cube

        self._record_cache_miss()
        prepared_cube = self._apply_environment_scales(
            cube.cube,
            sample_index,
            environment_scales,
        )
        processed = self._preprocessor.prepare(
            prepared_cube,
            sample_offset=sample_index,
        )
        probs = self._infer(processed)
        observation_dates = self._resolve_observation_dates(
            cube.img_dates, processed.sample_index
        )
        target_date = self._resolve_target_date(
            cube.img_dates,
            processed.sample_index,
            self._settings.n_leading_observations,
        )

        payload = CachedInferencePayload(
            metadata=cube.metadata,
            sample_index=processed.sample_index,
            total_samples=processed.total_samples,
            probabilities=np.asarray(probs, dtype=np.float32),
            ground_truth=processed.ground_truth.numpy().astype(np.uint8),
            observation_dates=observation_dates,
            target_date=target_date,
        )
        self._inference_cache.set(cache_key, payload)
        return payload, prepared_cube if include_prepared_cube else None

    def _build_prediction_from_payload(
        self,
        payload: CachedInferencePayload,
        threshold: float,
    ) -> SpreadPrediction:
        probabilities = np.asarray(payload.probabilities, dtype=np.float32)
        mask = (probabilities >= threshold).astype(np.uint8)
        return SpreadPrediction(
            metadata=payload.metadata,
            sample_index=payload.sample_index,
            total_samples=payload.total_samples,
            threshold=threshold,
            probabilities=np.copy(probabilities),
            mask=mask,
            ground_truth=np.copy(payload.ground_truth),
            observation_dates=payload.observation_dates,
            target_date=payload.target_date,
        )

    def _record_cache_hit(self) -> None:
        with self._metrics_lock:
            self._metrics["cache_hit"] = int(self._metrics["cache_hit"]) + 1

    def _record_cache_miss(self) -> None:
        with self._metrics_lock:
            self._metrics["cache_miss"] = int(self._metrics["cache_miss"]) + 1

    def _infer(self, sample: PreprocessedSample) -> np.ndarray:
        tensor = sample.tensor
        predictions = self._model.predict(tensor)
        probs = predictions.squeeze(0).squeeze(0).numpy()
        return probs

    def _resolve_observation_dates(
        self, img_dates: Tuple[str, ...], sample_index: int
    ) -> Tuple[str, ...]:
        leads = self._settings.n_leading_observations
        start = sample_index
        end = min(start + leads, len(img_dates))
        return tuple(img_dates[start:end])

    def _resolve_target_date(
        self, img_dates: Tuple[str, ...], sample_index: int, leads: int
    ) -> Optional[str]:
        idx = sample_index + leads
        if 0 <= idx < len(img_dates):
            return img_dates[idx]
        return None

    def _resolve_sample_index(self, sample_offset: int, total_samples: int) -> int:
        if total_samples <= 0:
            return 0

        offset = sample_offset
        if offset < 0:
            offset = total_samples + offset

        if offset < 0:
            return 0
        if offset >= total_samples:
            return total_samples - 1
        return int(offset)

    def _normalize_environment_scales(
        self, environment_scales: Optional[Dict[str, float]]
    ) -> Tuple[Tuple[str, float], ...]:
        normalized = []
        for key in sorted(SCALABLE_ENVIRONMENT_CHANNELS.keys()):
            factor = 1.0
            if environment_scales is not None:
                factor = float(environment_scales.get(key, 1.0))
            normalized.append((key, round(factor, 6)))
        return tuple(normalized)

    def _apply_environment_scales(
        self,
        cube: np.ndarray,
        sample_offset: int,
        environment_scales: Optional[Dict[str, float]],
    ) -> np.ndarray:
        if not environment_scales:
            return cube

        leads = self._settings.n_leading_observations
        total_samples = cube.shape[0] - leads
        sample_index = self._resolve_sample_index(sample_offset, total_samples)
        start = sample_index
        end = min(start + leads, cube.shape[0])
        scaled_cube = np.copy(cube)

        for key, channel_idx in SCALABLE_ENVIRONMENT_CHANNELS.items():
            factor = float(environment_scales.get(key, 1.0))
            if abs(factor - 1.0) < 1e-6:
                continue
            scaled_cube[start:end, channel_idx, ...] = (
                scaled_cube[start:end, channel_idx, ...] * factor
            )

        return scaled_cube

    def _extract_model_input_arrays(
        self,
        cube: np.ndarray,
        sample_index: int,
        spatial_shape: Tuple[int, int],
        keys: List[str],
    ) -> Dict[str, np.ndarray]:
        leads = self._settings.n_leading_observations
        frame_index = min(sample_index + max(leads - 1, 0), cube.shape[0] - 1)
        raw_frame = cube[frame_index]
        target_h, target_w = spatial_shape
        _, height, width = raw_frame.shape
        h_off = max((height - target_h) // 2, 0)
        w_off = max((width - target_w) // 2, 0)
        cropped = raw_frame[:, h_off:h_off + target_h, w_off:w_off + target_w]

        arrays: Dict[str, np.ndarray] = {}
        for key in keys:
            channel_idx = MODEL_INPUT_CHANNELS.get(key)
            if channel_idx is None:
                continue
            arrays[key] = np.asarray(cropped[channel_idx], dtype=np.float32)
        return arrays

    def _build_model_fingerprint(self) -> str:
        model_path = self._settings.model_path.resolve(strict=False)
        try:
            stat_result = model_path.stat()
            token = (
                f"{model_path}:{stat_result.st_size}:{stat_result.st_mtime_ns}".encode(
                    "utf-8"
                )
            )
        except OSError:
            token = str(model_path).encode("utf-8")
        return hashlib.sha256(token).hexdigest()[:16]
