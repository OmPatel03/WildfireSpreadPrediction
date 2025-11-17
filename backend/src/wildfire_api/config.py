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
    stats_years: Tuple[int, int] = Field(default=(2018, 2019))
    default_year: int = Field(default=2021, ge=2018)
    n_leading_observations: int = Field(default=1, ge=1)
    probability_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
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
    project_root = backend_root.parent
    workspace_root = project_root.parent

    default_model = backend_root / "resources" / "model.ckpt"
    default_hdf5 = workspace_root / "HDF5"

    stats_years = _parse_years(os.getenv("WILDFIRE_STATS_YEARS"))

    model_path = _resolve_path(Path(os.getenv("MODEL_CHECKPOINT", default_model)))
    hdf5_root = _resolve_path(Path(os.getenv("HDF5_ROOT", default_hdf5)))

    return Settings(
        model_path=model_path,
        hdf5_root=hdf5_root,
        stats_years=stats_years,
        default_year=int(os.getenv("WILDFIRE_DEFAULT_YEAR", 2021)),
        n_leading_observations=int(os.getenv("WILDFIRE_N_LEADS", 1)),
        probability_threshold=float(os.getenv("WILDFIRE_PROB_THRESHOLD", 0.5)),
        flatten_temporal_dimension=os.getenv("WILDFIRE_FLATTEN_TEMP", "true")
        .strip()
        .lower()
        not in {"false", "0"},
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    return _build_settings()
