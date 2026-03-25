from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from wildfire_api import WildfireService, get_settings
from wildfire_api.graphql_schema import build_graphql_router
from wildfire_api.schemas import (
    FireLayersResponse,
    HealthResponse,
    SpreadRequest,
    SpreadResponse,
    TimelineResponse,
    WildfireSummary,
)

logger = logging.getLogger("wildfire_api.inference")


def create_app() -> FastAPI:
    settings = get_settings()
    service = WildfireService(settings)
    infer_semaphore = asyncio.Semaphore(settings.infer_max_concurrency)

    app = FastAPI(
        title="Wildfire Spread Service",
        version="0.1.0",
        description="FastAPI + GraphQL gateway for wildfire spread predictions.",
    )
    app.state.service = service
    app.state.infer_semaphore = infer_semaphore

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    graphql_router = build_graphql_router(service)
    app.include_router(graphql_router, prefix="/graphql")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(**service.health())

    @app.get("/catalog", response_model=List[WildfireSummary])
    async def catalog(
        year: int | None = Query(None, description="Year to read from"),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> List[WildfireSummary]:
        entries = await run_in_threadpool(service.catalog, year, limit, offset)
        return [WildfireSummary.from_metadata(meta) for meta in entries]

    @app.get("/overview", response_model=List[WildfireSummary])
    async def overview(
        year: int | None = Query(None, description="Year to read from"),
        limit: int = Query(200, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> List[WildfireSummary]:
        entries = await run_in_threadpool(service.overview, year, limit, offset)
        return [WildfireSummary.from_metadata(meta) for meta in entries]

    @app.get("/years", response_model=List[int])
    async def years() -> List[int]:
        return await run_in_threadpool(service.available_years)

    @app.post("/findSpread", response_model=SpreadResponse)
    async def find_spread(request: SpreadRequest) -> SpreadResponse:
        queue_wait_ms = await _acquire_inference_slot("/findSpread")
        started = time.perf_counter()
        try:
            prediction, geojson = await run_in_threadpool(
                service.find_spread,
                request.fire_id,
                request.year,
                request.sample_offset,
                request.threshold,
            )
            return SpreadResponse.from_prediction(prediction, geojson)
        finally:
            _release_inference_slot(
                route_name="/findSpread",
                queue_wait_ms=queue_wait_ms,
                started=started,
            )

    @app.get("/fires/{fire_id}/timeline", response_model=TimelineResponse)
    async def fire_timeline(
        fire_id: str,
        year: int | None = Query(None, description="Year to read from"),
    ) -> TimelineResponse:
        metadata, frames, default_sample_index = await run_in_threadpool(
            service.timeline,
            fire_id,
            year,
        )
        return TimelineResponse.from_metadata(metadata, frames, default_sample_index)

    @app.get("/fires/{fire_id}/layers", response_model=FireLayersResponse)
    async def fire_layers(
        fire_id: str,
        year: int | None = Query(None, description="Year to read from"),
        sample_index: int | None = Query(None, alias="sampleIndex", ge=0),
        threshold: float | None = Query(None, ge=0.0, le=1.0),
        model_input: str | None = Query(None, alias="modelInput"),
        viirs_m11_scale: float = Query(1.0, alias="viirsM11Scale", ge=0.5, le=2.0),
        viirs_i2_scale: float = Query(1.0, alias="viirsI2Scale", ge=0.5, le=2.0),
        ndvi_scale: float = Query(1.0, alias="ndviScale", ge=0.5, le=2.0),
        evi2_scale: float = Query(1.0, alias="evi2Scale", ge=0.5, le=2.0),
        precip_scale: float = Query(1.0, alias="precipScale", ge=0.5, le=2.0),
        wind_speed_scale: float = Query(1.0, alias="windSpeedScale", ge=0.5, le=2.0),
    ) -> FireLayersResponse:
        queue_wait_ms = await _acquire_inference_slot("/fires/{fire_id}/layers")
        started = time.perf_counter()
        try:
            prediction, geojson, layers = await run_in_threadpool(
                service.find_layers,
                fire_id,
                year,
                sample_index,
                threshold,
                model_input,
                {
                    "viirs_m11": viirs_m11_scale,
                    "viirs_i2": viirs_i2_scale,
                    "ndvi": ndvi_scale,
                    "evi2": evi2_scale,
                    "precip": precip_scale,
                    "wind_speed": wind_speed_scale,
                },
            )
            return FireLayersResponse.from_prediction(prediction, geojson, layers)
        finally:
            _release_inference_slot(
                route_name="/fires/{fire_id}/layers",
                queue_wait_ms=queue_wait_ms,
                started=started,
            )

    async def _acquire_inference_slot(route_name: str) -> float:
        queue_wait_started = time.perf_counter()
        try:
            await asyncio.wait_for(
                infer_semaphore.acquire(),
                timeout=settings.infer_queue_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            service.record_rate_limited()
            retry_after = max(1, math.ceil(settings.infer_queue_timeout_seconds))
            logger.warning(
                "rate_limited route=%s queue_timeout_s=%.3f",
                route_name,
                settings.infer_queue_timeout_seconds,
            )
            raise HTTPException(
                status_code=429,
                detail="Server busy, retry shortly.",
                headers={"Retry-After": str(retry_after)},
            ) from exc

        queue_wait_ms = (time.perf_counter() - queue_wait_started) * 1000.0
        service.mark_inference_started(queue_wait_ms)
        return queue_wait_ms

    def _release_inference_slot(
        route_name: str,
        queue_wait_ms: float,
        started: float,
    ) -> None:
        infer_semaphore.release()
        inference_duration_ms = (time.perf_counter() - started) * 1000.0
        service.mark_inference_finished(inference_duration_ms)
        logger.info(
            "inference_completed route=%s queue_wait_ms=%.2f inference_duration_ms=%.2f",
            route_name,
            queue_wait_ms,
            inference_duration_ms,
        )

    return app


app = create_app()
def _print_endpoint_base_paths(application) -> None:
    prefixes = set()
    for route in application.routes:
        path = getattr(route, "path", "")
        if not path:
            continue
        parts = [p for p in path.split("/") if p]
        prefixes.add("/" + parts[0] if parts else "/")
    print("Endpoint base paths:", ", ".join(sorted(prefixes)))


_print_endpoint_base_paths(app)
