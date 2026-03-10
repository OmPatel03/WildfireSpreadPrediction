from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import Settings, get_settings
from .domain import SpreadPrediction
from .gee_basemap import build_fire_basemap
from .geojson import build_geojson, build_layer_collection
from .model_loader import get_model
from .preprocessing import PreprocessedSample, SamplePreprocessor
from .repository import WildfireMetadata, WildfireRepository


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
    ) -> Tuple[SpreadPrediction, Dict]:
        target_year = year or self._settings.default_year
        cube = self._repository.load_cube(fire_id, target_year)
        processed = self._preprocessor.prepare(cube.cube, sample_offset=sample_offset)
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
    ) -> Tuple[SpreadPrediction, Dict, Dict]:
        resolved_sample = -1 if sample_index is None else sample_index
        prediction, geojson = self.find_spread(
            fire_id=fire_id,
            year=year,
            sample_offset=resolved_sample,
            threshold=threshold,
        )
        layers = build_layer_collection(prediction)
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
