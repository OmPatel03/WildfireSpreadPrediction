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


class GoodPredictionRow(BaseModel):
    fire_id: str = Field(alias="fireId")
    year: int
    sample_index: int = Field(alias="sampleIndex")
    target_date: Optional[str] = Field(alias="targetDate")
    threshold: float
    ground_truth_pixels: int = Field(alias="groundTruthPixels")
    positive_pixels: int = Field(alias="positivePixels")
    precision: float
    recall: float
    f1: float
    accuracy: float

    class Config:
        populate_by_name = True


class BoundingBox(BaseModel):
    min_lon: float = Field(alias="minLon")
    min_lat: float = Field(alias="minLat")
    max_lon: float = Field(alias="maxLon")
    max_lat: float = Field(alias="maxLat")

    class Config:
        populate_by_name = True

    @classmethod
    def from_tuple(cls, bbox: tuple[float, float, float, float]) -> "BoundingBox":
        return cls(
            minLon=bbox[0],
            minLat=bbox[1],
            maxLon=bbox[2],
            maxLat=bbox[3],
        )


class PredictionSummary(BaseModel):
    mean_probability: float = Field(alias="meanProbability")
    max_probability: float = Field(alias="maxProbability")
    min_probability: float = Field(alias="minProbability")
    positive_pixels: int = Field(alias="positivePixels")
    ground_truth_pixels: int = Field(alias="groundTruthPixels")
    total_pixels: int = Field(alias="totalPixels")
    true_positive: int = Field(alias="truePositive")
    false_positive: int = Field(alias="falsePositive")
    false_negative: int = Field(alias="falseNegative")
    true_negative: int = Field(alias="trueNegative")
    precision: float
    recall: float
    f1: float
    accuracy: float

    class Config:
        populate_by_name = True


class TimelineFrame(BaseModel):
    sample_index: int = Field(alias="sampleIndex")
    sample_offset: int = Field(alias="sampleOffset")
    observation_dates: List[str] = Field(alias="observationDates")
    target_date: Optional[str] = Field(alias="targetDate")
    ground_truth_positive_pixels: int = Field(alias="groundTruthPositivePixels")
    label: str

    class Config:
        populate_by_name = True


class WildfireSummary(BaseModel):
    fire_id: str = Field(alias="fireId")
    year: int
    longitude: float
    latitude: float
    time_steps: int = Field(alias="timeSteps")
    samples: int
    height: int
    width: int
    feature_count: int = Field(alias="featureCount")
    hdf5_path: str = Field(alias="hdf5Path")
    first_observation_date: Optional[str] = Field(default=None, alias="firstObservationDate")
    last_observation_date: Optional[str] = Field(default=None, alias="lastObservationDate")
    latest_target_date: Optional[str] = Field(default=None, alias="latestTargetDate")
    latest_target_positive_pixels: int = Field(alias="latestTargetPositivePixels")
    bbox: BoundingBox

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
            featureCount=metadata.feature_count,
            hdf5Path=str(metadata.path),
            firstObservationDate=metadata.img_dates[0] if metadata.img_dates else None,
            lastObservationDate=metadata.img_dates[-1] if metadata.img_dates else None,
            latestTargetDate=(
                metadata.img_dates[-1]
                if len(metadata.img_dates) > 1
                else metadata.img_dates[0]
                if metadata.img_dates
                else None
            ),
            latestTargetPositivePixels=metadata.latest_target_positive_pixels,
            bbox=BoundingBox.from_tuple(metadata.bbox),
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
    observation_dates: List[str] = Field(alias="observationDates")
    target_date: Optional[str] = Field(alias="targetDate")
    summary: PredictionSummary
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
            observationDates=list(prediction.observation_dates),
            targetDate=prediction.target_date,
            summary=geojson["features"][0]["properties"]["summary"],
            geojson=geojson,
        )


class FireLayers(BaseModel):
    prediction_heatmap: Dict[str, Any] = Field(alias="predictionHeatmap")
    prediction_polygons: Dict[str, Any] = Field(alias="predictionPolygons")
    ground_truth_heatmap: Dict[str, Any] = Field(alias="groundTruthHeatmap")
    difference_heatmap: Dict[str, Any] = Field(alias="differenceHeatmap")
    extent: Dict[str, Any]
    origin: Dict[str, Any]
    model_inputs: Dict[str, Any] = Field(default_factory=dict, alias="modelInputs")

    class Config:
        populate_by_name = True


class FireLayersResponse(BaseModel):
    fire: WildfireSummary
    sample_index: int = Field(alias="sampleIndex")
    total_samples: int = Field(alias="totalSamples")
    threshold: float
    observation_dates: List[str] = Field(alias="observationDates")
    target_date: Optional[str] = Field(alias="targetDate")
    summary: PredictionSummary
    layers: FireLayers
    geojson: Dict[str, Any]

    class Config:
        populate_by_name = True

    @classmethod
    def from_prediction(
        cls,
        prediction: SpreadPrediction,
        geojson: Dict[str, Any],
        layers: Dict[str, Any],
    ) -> "FireLayersResponse":
        return cls(
            fire=WildfireSummary.from_metadata(prediction.metadata),
            sampleIndex=prediction.sample_index,
            totalSamples=prediction.total_samples,
            threshold=prediction.threshold,
            observationDates=list(prediction.observation_dates),
            targetDate=prediction.target_date,
            summary=geojson["features"][0]["properties"]["summary"],
            layers=layers,
            geojson=geojson,
        )


class TimelineResponse(BaseModel):
    fire: WildfireSummary
    total_samples: int = Field(alias="totalSamples")
    default_sample_index: int = Field(alias="defaultSampleIndex")
    frames: List[TimelineFrame]

    class Config:
        populate_by_name = True

    @classmethod
    def from_metadata(
        cls,
        metadata: WildfireMetadata,
        frames: List[Dict[str, Any]],
        default_sample_index: int,
    ) -> "TimelineResponse":
        return cls(
            fire=WildfireSummary.from_metadata(metadata),
            totalSamples=metadata.samples,
            defaultSampleIndex=default_sample_index,
            frames=frames,
        )
