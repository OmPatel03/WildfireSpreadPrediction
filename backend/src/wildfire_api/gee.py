from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional

import ee

from .config import Settings
from .domain import SpreadPrediction


class GEEMapService:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._initialize()

    def _initialize(self) -> None:
        try:
            ee.Initialize(project=self._settings.gee_project)
        except Exception:
            ee.Authenticate()
            ee.Initialize(project=self._settings.gee_project)

    def build_layers(self, prediction: SpreadPrediction) -> Dict[str, Any]:
        bbox = prediction.metadata.bbox
        region = ee.Geometry.BBox(bbox[0], bbox[1], bbox[2], bbox[3])
        target_date = prediction.target_date or (
            prediction.observation_dates[-1] if prediction.observation_dates else None
        )

        satellite_image = self._build_satellite_image(region, target_date)
        terrain_image = self._build_terrain_image(region)

        return {
            "satellite": self._to_tile_layer(
                satellite_image,
                {
                    "bands": ["B4", "B3", "B2"],
                    "min": 0,
                    "max": 3000,
                    "gamma": 1.15,
                },
                label="Satellite",
                attribution="Google Earth Engine · Sentinel-2",
            ),
            "terrain": self._to_tile_layer(
                terrain_image,
                {
                    "min": 0,
                    "max": 255,
                    "palette": ["0b1020", "334155", "94a3b8", "f8fafc"],
                },
                label="Terrain",
                attribution="Google Earth Engine · SRTM",
            ),
            "bounds": {
                "minLon": bbox[0],
                "minLat": bbox[1],
                "maxLon": bbox[2],
                "maxLat": bbox[3],
            },
            "targetDate": target_date,
        }

    def _build_satellite_image(
        self, region: ee.Geometry, target_date: Optional[str]
    ) -> ee.Image:
        if target_date is None:
            target = ee.Date("2021-01-01")
        else:
            target = ee.Date(target_date)

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(target.advance(-21, "day"), target.advance(21, "day"))
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 35))
        )

        fallback = ee.Image.constant([0, 0, 0]).rename(["B4", "B3", "B2"]).toUint16()
        image = ee.Image(
            ee.Algorithms.If(
                collection.size().gt(0),
                collection.sort("CLOUDY_PIXEL_PERCENTAGE").median(),
                fallback,
            )
        )
        return image.clip(region)

    def _build_terrain_image(self, region: ee.Geometry) -> ee.Image:
        elevation = ee.Image("USGS/SRTMGL1_003").clip(region)
        return ee.Terrain.hillshade(elevation)

    def _to_tile_layer(
        self,
        image: ee.Image,
        vis_params: Dict[str, Any],
        *,
        label: str,
        attribution: str,
    ) -> Dict[str, Any]:
        map_id = image.getMapId(vis_params)
        return {
            "label": label,
            "url": map_id["tile_fetcher"].url_format,
            "attribution": attribution,
        }


@lru_cache(maxsize=1)
def get_gee_map_service(settings: Settings) -> GEEMapService:
    return GEEMapService(settings)
