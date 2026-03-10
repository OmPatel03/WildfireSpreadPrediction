from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import Settings, get_settings
from .domain import SpreadPrediction
from .gee_basemap import build_fire_basemap
from .geojson import build_geojson, build_layer_collection, build_model_input_layer_collection
from .model_loader import get_model
from .preprocessing import PreprocessedSample, SamplePreprocessor
from .repository import WildfireMetadata, WildfireRepository


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


class WildfireService:
    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
        self._repository = WildfireRepository(self._settings)
        self._preprocessor = SamplePreprocessor(self._settings)
        self._model = get_model(self._settings)

    def health(self) -> Dict[str, object]:
        return {
            "status": "ok",
            "model_path": str(self._settings.model_path),
            "model_loaded": True,
            "device": str(self._model.device),
            "hdf5_root": str(self._settings.hdf5_root),
            "default_year": self._settings.default_year,
        }

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
            label = " → ".join(filter(None, [*observation_dates, target_date]))
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
        prepared_cube = self._apply_environment_scales(
            cube.cube,
            sample_offset,
            environment_scales,
        )
        processed = self._preprocessor.prepare(prepared_cube, sample_offset=sample_offset)
        probs = self._infer(processed)
        mask_threshold = threshold if threshold is not None else self._settings.probability_threshold
        mask = (probs >= mask_threshold).astype(np.uint8)
        observation_dates = self._resolve_observation_dates(
            cube.img_dates, processed.sample_index
        )
        target_date = self._resolve_target_date(
            cube.img_dates, processed.sample_index, self._settings.n_leading_observations
        )
        prediction = SpreadPrediction(
            metadata=cube.metadata,
            sample_index=processed.sample_index,
            total_samples=processed.total_samples,
            threshold=mask_threshold,
            probabilities=probs,
            mask=mask,
            ground_truth=processed.ground_truth.numpy().astype(np.uint8),
            observation_dates=observation_dates,
            target_date=target_date,
        )
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
        resolved_sample = -1 if sample_index is None else sample_index
        target_year = year or self._settings.default_year
        cube = self._repository.load_cube(fire_id, target_year)
        prepared_cube = self._apply_environment_scales(
            cube.cube,
            resolved_sample,
            environment_scales,
        )
        prediction, geojson = self.find_spread(
            fire_id=fire_id,
            year=year,
            sample_offset=resolved_sample,
            threshold=threshold,
            environment_scales=environment_scales,
        )
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
        offset = sample_offset
        if offset < 0:
            offset = total_samples + offset
        if offset < 0 or offset >= total_samples:
            raise IndexError(f"Sample offset {offset} outside range [0, {total_samples - 1}]")
        return offset

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
