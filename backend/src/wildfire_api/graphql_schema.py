from __future__ import annotations

from typing import List, Optional

import strawberry
from strawberry.fastapi import GraphQLRouter
from strawberry.scalars import JSON

from .domain import SpreadPrediction
from .service import WildfireService


@strawberry.type
class HealthType:
    status: str
    model_path: str
    model_loaded: bool
    device: str
    hdf5_root: str
    default_year: int


@strawberry.type
class WildfireType:
    fire_id: str
    year: int
    longitude: float
    latitude: float
    time_steps: int
    samples: int
    height: int
    width: int
    hdf5_path: str


@strawberry.type
class SpreadResultType:
    fire: WildfireType
    sample_index: int
    total_samples: int
    threshold: float
    geojson: JSON


def _to_health_dict(health: dict) -> HealthType:
    return HealthType(**health)


def _to_wildfire_type(metadata) -> WildfireType:
    return WildfireType(
        fire_id=metadata.fire_id,
        year=metadata.year,
        longitude=metadata.longitude,
        latitude=metadata.latitude,
        time_steps=metadata.time_steps,
        samples=metadata.samples,
        height=metadata.height,
        width=metadata.width,
        hdf5_path=str(metadata.path),
    )


def _to_spread_type(prediction: SpreadPrediction, geojson) -> SpreadResultType:
    return SpreadResultType(
        fire=_to_wildfire_type(prediction.metadata),
        sample_index=prediction.sample_index,
        total_samples=prediction.total_samples,
        threshold=prediction.threshold,
        geojson=geojson,
    )


def build_graphql_router(service: WildfireService) -> GraphQLRouter:
    @strawberry.type
    class Query:
        @strawberry.field
        def health(self) -> HealthType:
            return _to_health_dict(service.health())

        @strawberry.field
        def catalog(
            self,
            year: Optional[int] = strawberry.UNSET,
            limit: int = 50,
            offset: int = 0,
        ) -> List[WildfireType]:
            resolved_year = None if year is strawberry.UNSET else year
            metas = service.catalog(year=resolved_year, limit=limit, offset=offset)
            return [_to_wildfire_type(meta) for meta in metas]

        @strawberry.field
        def find_spread(
            self,
            fire_id: str,
            year: Optional[int] = strawberry.UNSET,
            sample_offset: int = -1,
            threshold: Optional[float] = strawberry.UNSET,
        ) -> SpreadResultType:
            resolved_year = None if year is strawberry.UNSET else year
            resolved_threshold = None if threshold is strawberry.UNSET else threshold
            prediction, geojson = service.find_spread(
                fire_id=fire_id,
                year=resolved_year,
                sample_offset=sample_offset,
                threshold=resolved_threshold,
            )
            return _to_spread_type(prediction, geojson)

    schema = strawberry.Schema(query=Query)
    return GraphQLRouter(schema)
