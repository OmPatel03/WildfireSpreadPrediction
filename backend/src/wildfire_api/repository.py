from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np

from .config import Settings


LAT_STEP = 1 / 296
COLUMN_WIDTH_METERS = 375


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
    img_dates: Tuple[str, ...]
    bbox: Tuple[float, float, float, float]
    has_positive_target: bool
    latest_target_positive_pixels: int


def _decode_attr_value(value: object) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return str(value)


def _compute_bbox(
    height: int, width: int, latitude: float, longitude: float
) -> Tuple[float, float, float, float]:
    safe_cos = max(abs(cos((latitude * pi) / 180.0)), 1e-6)
    long_step = COLUMN_WIDTH_METERS / (111000 * safe_cos)
    zero_lat = latitude - (height // 2) * LAT_STEP - (LAT_STEP / 2) * (height % 2)
    zero_long = longitude - (width // 2) * long_step - (long_step / 2) * (width % 2)
    half_lat = LAT_STEP / 2
    half_long = long_step / 2

    min_lat = zero_lat - half_lat
    max_lat = zero_lat + (max(height - 1, 0) * LAT_STEP) + half_lat
    min_long = zero_long - half_long
    max_long = zero_long + (max(width - 1, 0) * long_step) + half_long
    return (min_long, min_lat, max_long, max_lat)


@dataclass
class WildfireCube:
    metadata: WildfireMetadata
    cube: np.ndarray
    img_dates: Tuple[str, ...]


class WildfireRepository:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._cache: Dict[int, Dict[str, WildfireMetadata]] = {}
        self._target_positive_counts_cache: Dict[Tuple[int, str], Tuple[int, ...]] = {}

    def available_years(self) -> List[int]:
        years: List[int] = []
        for path in sorted(self._settings.hdf5_root.iterdir()):
            if not path.is_dir():
                continue
            try:
                year = int(path.name)
            except ValueError:
                continue
            if next(path.glob("fire_*.hdf5"), None) is not None:
                years.append(year)
        return years

    def list_year(self, year: int) -> List[WildfireMetadata]:
        catalog = self._catalog_for_year(year)
        return [catalog[key] for key in sorted(catalog.keys())]

    def get_metadata(self, fire_id: str, year: Optional[int] = None) -> WildfireMetadata:
        target_year = year or self._settings.default_year
        normalized_id = normalize_fire_id(fire_id)
        catalog = self._catalog_for_year(target_year)
        if normalized_id not in catalog:
            raise FileNotFoundError(
                f"Could not find fire '{normalized_id}' in {target_year} under {self._settings.hdf5_root}"
            )
        return catalog[normalized_id]

    def load_cube(self, fire_id: str, year: Optional[int] = None) -> WildfireCube:
        target_year = year or self._settings.default_year
        metadata = self.get_metadata(fire_id, target_year)
        with h5py.File(metadata.path, "r") as handle:
            dataset = handle["data"]
            cube = np.asarray(dataset[...], dtype=np.float32)
            raw_dates = dataset.attrs.get("img_dates", [])
            img_dates = tuple(_decode_attr_value(item) for item in raw_dates)
        return WildfireCube(metadata=metadata, cube=cube, img_dates=img_dates)

    def load_target_positive_counts(
        self, fire_id: str, year: Optional[int] = None
    ) -> Tuple[int, ...]:
        target_year = year or self._settings.default_year
        metadata = self.get_metadata(fire_id, target_year)
        cache_key = (target_year, metadata.fire_id)
        if cache_key in self._target_positive_counts_cache:
            return self._target_positive_counts_cache[cache_key]

        counts = self._load_target_positive_counts_from_path(metadata.path, target_year)
        self._target_positive_counts_cache[cache_key] = counts
        return counts

    def _load_target_positive_counts_from_path(
        self, file_path: Path, year: int
    ) -> Tuple[int, ...]:
        leads = self._settings.n_leading_observations
        with h5py.File(file_path, "r") as handle:
            dataset = handle["data"]
            if dataset.shape[0] <= leads:
                return tuple()
            targets = np.asarray(dataset[leads:, -1, ...] > 0, dtype=np.uint8)

        counts = np.count_nonzero(targets, axis=(1, 2))
        return tuple(int(value) for value in counts.tolist())

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
            raw_dates = dataset.attrs.get("img_dates", [])
            img_dates = tuple(_decode_attr_value(item) for item in raw_dates)
        positive_counts = self._load_target_positive_counts_from_path(file_path, year)
        self._target_positive_counts_cache[(year, file_path.stem)] = positive_counts
        latest_target_positive_pixels = positive_counts[-1] if positive_counts else 0
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
            img_dates=img_dates,
            bbox=_compute_bbox(height, width, latitude, longitude),
            has_positive_target=any(positive_counts),
            latest_target_positive_pixels=latest_target_positive_pixels,
        )
