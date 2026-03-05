from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .domain import SpreadPrediction
from .repository import WildfireMetadata


class Coordinate(BaseModel):
    lon: float
    lat: float


class HealthResponse(BaseModel):
    status: str
    model_path: str
    model_loaded: bool
    device: str
    hdf5_root: str
    default_year: int


class WildfireSummary(BaseModel):
    fire_id: str = Field(alias="fireId")
    year: int
    longitude: float
    latitude: float
    time_steps: int = Field(alias="timeSteps")
    samples: int
    height: int
    width: int
    hdf5_path: str = Field(alias="hdf5Path")

    class Config:
        populate_by_name = True

    @classmethod
    def from_metadata(cls, metadata: WildfireMetadata) -> "WildfireSummary":
        return cls(
            fireId=metadata.fire_id,
            year=metadata.year,
            longitude=metadata.longitude,
            latitude=metadata.latitude,
            timeSteps=metadata.time_steps,
            samples=metadata.samples,
            height=metadata.height,
            width=metadata.width,
            hdf5Path=str(metadata.path),
        )


class SpreadRequest(BaseModel):
    fire_id: str = Field(alias="fireId")
    year: Optional[int] = None
    sample_offset: int = Field(default=-1, alias="sampleOffset")
    threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, alias="probabilityThreshold"
    )

    class Config:
        populate_by_name = True


class SpreadResponse(BaseModel):
    fire: WildfireSummary
    sample_index: int = Field(alias="sampleIndex")
    total_samples: int = Field(alias="totalSamples")
    threshold: float
    geojson: Dict[str, Any]

    class Config:
        populate_by_name = True

    @classmethod
    def from_prediction(
        cls, prediction: SpreadPrediction, geojson: Dict[str, Any]
    ) -> "SpreadResponse":
        return cls(
            fire=WildfireSummary.from_metadata(prediction.metadata),
            sampleIndex=prediction.sample_index,
            totalSamples=prediction.total_samples,
            threshold=prediction.threshold,
            geojson=geojson,
        )
