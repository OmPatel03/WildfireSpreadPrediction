from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Tuple

from pydantic import BaseModel, Field


def _resolve_path(value: Path) -> Path:
    return value.expanduser().resolve(strict=False)


class Settings(BaseModel):
    model_path: Path = Field(...)
    hdf5_root: Path = Field(...)
    gee_project: str = Field(default="ee-neeljos24")
    stats_years: Tuple[int, int] = Field(default=(2018, 2019))
    default_year: int = Field(default=2021, ge=2018)
    n_leading_observations: int = Field(default=1, ge=1)
    probability_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    infer_max_concurrency: int = Field(default=3, ge=1)
    infer_queue_timeout_seconds: float = Field(default=0.35, ge=0.0)
    cache_ttl_seconds: int = Field(default=600, ge=1)
    cache_max_entries: int = Field(default=300, ge=1)
    flatten_temporal_dimension: bool = True

    class Config:
        arbitrary_types_allowed = True


def _parse_years(raw: str | None) -> Tuple[int, int]:
    if not raw:
        return 2018, 2019
    values = [int(fragment.strip()) for fragment in raw.split(",") if fragment.strip()]
    if len(values) < 2:
        raise ValueError(
            f"WILDFIRE_STATS_YEARS requires at least two comma-separated years, got: {raw}"
        )
    return tuple(values[:2])  # type: ignore[return-value]


def _build_settings() -> Settings:
    backend_root = Path(__file__).resolve().parents[2]

    default_model = backend_root / "resources" / "model.ckpt"
    default_hdf5 = Path("/u50/capstone/cs4zp6g17/data/hdf5")

    stats_years = _parse_years(os.getenv("WILDFIRE_STATS_YEARS"))

    model_path = _resolve_path(Path(os.getenv("MODEL_CHECKPOINT", default_model)))
    hdf5_root = _resolve_path(Path(os.getenv("HDF5_ROOT", default_hdf5)))

    return Settings(
        model_path=model_path,
        hdf5_root=hdf5_root,
        gee_project=os.getenv("EE_PROJECT", "ee-neeljos24"),
        stats_years=stats_years,
        default_year=int(os.getenv("WILDFIRE_DEFAULT_YEAR", 2021)),
        n_leading_observations=int(os.getenv("WILDFIRE_N_LEADS", 1)),
        probability_threshold=float(os.getenv("WILDFIRE_PROB_THRESHOLD", 0.9)),
        infer_max_concurrency=int(os.getenv("WILDFIRE_INFER_MAX_CONCURRENCY", 3)),
        infer_queue_timeout_seconds=float(
            os.getenv("WILDFIRE_INFER_QUEUE_TIMEOUT_SECONDS", 0.35)
        ),
        cache_ttl_seconds=int(os.getenv("WILDFIRE_CACHE_TTL_SECONDS", 600)),
        cache_max_entries=int(os.getenv("WILDFIRE_CACHE_MAX_ENTRIES", 300)),
        flatten_temporal_dimension=os.getenv("WILDFIRE_FLATTEN_TEMP", "true")
        .strip()
        .lower()
        not in {"false", "0"},
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    return _build_settings()
