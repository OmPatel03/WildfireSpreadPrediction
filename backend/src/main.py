from __future__ import annotations

from typing import List

from fastapi import Depends, FastAPI, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from wildfire_api import WildfireService, get_settings
from wildfire_api.graphql_schema import build_graphql_router
from wildfire_api.schemas import (
    HealthResponse,
    SpreadRequest,
    SpreadResponse,
    WildfireSummary,
)


def create_app() -> FastAPI:
    settings = get_settings()
    service = WildfireService(settings)

    app = FastAPI(
        title="Wildfire Spread Service",
        version="0.1.0",
        description="FastAPI + GraphQL gateway for wildfire spread predictions.",
    )
    app.state.service = service

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

    @app.post("/findSpread", response_model=SpreadResponse)
    async def find_spread(request: SpreadRequest) -> SpreadResponse:
        prediction, geojson = await run_in_threadpool(
            service.find_spread,
            request.fire_id,
            request.year,
            request.sample_offset,
            request.threshold,
        )
        return SpreadResponse.from_prediction(prediction, geojson)

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