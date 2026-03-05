from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np

from .config import Settings


def normalize_fire_id(raw_id: str) -> str:
    raw_id = raw_id.strip()
    if not raw_id:
        raise ValueError("Fire id must not be empty.")
    lower = raw_id.lower()
    if lower.startswith("fire_"):
        suffix = raw_id.split("fire_", 1)[1]
    else:
        suffix = "".join(ch for ch in raw_id if ch.isdigit())
        if not suffix:
            suffix = raw_id
    return f"fire_{suffix}"


@dataclass(frozen=True)
class WildfireMetadata:
    fire_id: str
    year: int
    path: Path
    longitude: float
    latitude: float
    time_steps: int
    feature_count: int
    height: int
    width: int
    samples: int


@dataclass
class WildfireCube:
    metadata: WildfireMetadata
    cube: np.ndarray
    img_dates: Tuple[str, ...]


class WildfireRepository:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._cache: Dict[int, Dict[str, WildfireMetadata]] = {}

    def list_year(self, year: int) -> List[WildfireMetadata]:
        catalog = self._catalog_for_year(year)
        return [catalog[key] for key in sorted(catalog.keys())]

    def load_cube(self, fire_id: str, year: Optional[int] = None) -> WildfireCube:
        target_year = year or self._settings.default_year
        normalized_id = normalize_fire_id(fire_id)
        catalog = self._catalog_for_year(target_year)
        if normalized_id not in catalog:
            raise FileNotFoundError(
                f"Could not find fire '{normalized_id}' in {target_year} under {self._settings.hdf5_root}"
            )
        metadata = catalog[normalized_id]
        with h5py.File(metadata.path, "r") as handle:
            dataset = handle["data"]
            cube = np.asarray(dataset[...], dtype=np.float32)
            raw_dates = dataset.attrs.get("img_dates", [])
            img_dates = tuple(str(item) for item in raw_dates)
        return WildfireCube(metadata=metadata, cube=cube, img_dates=img_dates)

    def _catalog_for_year(self, year: int) -> Dict[str, WildfireMetadata]:
        if year in self._cache:
            return self._cache[year]
        year_dir = self._settings.hdf5_root / str(year)
        if not year_dir.is_dir():
            raise FileNotFoundError(f"HDF5 directory '{year_dir}' is missing.")
        catalog: Dict[str, WildfireMetadata] = {}
        for file_path in sorted(year_dir.glob("fire_*.hdf5")):
            metadata = self._read_metadata(file_path, year)
            catalog[metadata.fire_id] = metadata
        if not catalog:
            raise FileNotFoundError(f"No HDF5 files found in {year_dir}.")
        self._cache[year] = catalog
        return catalog

    def _read_metadata(self, file_path: Path, year: int) -> WildfireMetadata:
        with h5py.File(file_path, "r") as handle:
            dataset = handle["data"]
            time_steps, feature_count, height, width = map(int, dataset.shape)
            lnglat = dataset.attrs.get("lnglat", (0.0, 0.0))
            longitude = float(lnglat[0]) if lnglat is not None else 0.0
            latitude = float(lnglat[1]) if lnglat is not None else 0.0
        samples = max(time_steps - self._settings.n_leading_observations, 0)
        return WildfireMetadata(
            fire_id=file_path.stem,
            year=year,
            path=file_path,
            longitude=longitude,
            latitude=latitude,
            time_steps=time_steps,
            feature_count=feature_count,
            height=height,
            width=width,
            samples=samples,
        )
