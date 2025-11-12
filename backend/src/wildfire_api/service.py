from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import Settings, get_settings
from .domain import SpreadPrediction
from .geojson import build_geojson
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
        entries = self._repository.list_year(target_year)
        start = min(offset, len(entries))
        end = min(start + limit, len(entries))
        return entries[start:end]

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
