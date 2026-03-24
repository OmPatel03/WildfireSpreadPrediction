from __future__ import annotations

import os
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, Optional

import ee

from .repository import WildfireMetadata


SATELLITE_PALETTE = ["0b1f0b", "1f3b1f", "5c7a29", "a3b18a", "d9e8c4"]
TERRAIN_PALETTE = ["0f172a", "334155", "64748b", "94a3b8", "e2e8f0"]
VALID_BASEMAP_STYLES = {"satellite", "terrain", "outdoors"}


@lru_cache(maxsize=1)
def ensure_gee_initialized() -> None:
    project = (
        os.getenv("EE_PROJECT")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or "ee-neeljos24"
    )
    ee.Initialize(project=project)


def _build_region_from_bbox(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> ee.Geometry:
    lon_padding = max((max_lon - min_lon) * 0.4, 0.05)
    lat_padding = max((max_lat - min_lat) * 0.4, 0.05)
    return ee.Geometry.Rectangle(
        [
            min_lon - lon_padding,
            min_lat - lat_padding,
            max_lon + lon_padding,
            max_lat + lat_padding,
        ]
    )


def _date_window(target_date: Optional[str]) -> tuple[str, str]:
    if not target_date:
        end = datetime.utcnow()
        start = end - timedelta(days=120)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    target = datetime.strptime(target_date, "%Y-%m-%d")
    start = target - timedelta(days=45)
    end = target + timedelta(days=7)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _satellite_visual(region: ee.Geometry, target_date: Optional[str]) -> ee.Image:
    start, end = _date_window(target_date)
    image = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .median()
    )
    return image.visualize(bands=["B4", "B3", "B2"], min=0, max=3000, gamma=1.15)


def _terrain_visual() -> ee.Image:
    elevation = ee.Image("USGS/SRTMGL1_003")
    terrain_rgb = elevation.visualize(min=0, max=4000, palette=TERRAIN_PALETTE)
    hillshade = ee.Terrain.hillshade(elevation).visualize(
        min=0,
        max=255,
        palette=["000000", "ffffff"],
        opacity=0.45,
    )
    return terrain_rgb.blend(hillshade)


def _outdoors_visual(region: ee.Geometry, target_date: Optional[str]) -> ee.Image:
    satellite = _satellite_visual(region, target_date)
    hillshade = ee.Terrain.hillshade(ee.Image("USGS/SRTMGL1_003")).visualize(
        min=0,
        max=255,
        palette=["000000", "ffffff"],
        opacity=0.28,
    )
    vegetation = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(*_date_window(target_date))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .median()
        .normalizedDifference(["B8", "B4"])
        .visualize(min=0.1, max=0.8, palette=SATELLITE_PALETTE, opacity=0.25)
    )
    return satellite.blend(vegetation).blend(hillshade)


def _tile_url(image: ee.Image) -> str:
    map_id = image.getMapId()
    return map_id["tile_fetcher"].url_format


def _build_style_visual(
    style: str,
    region: ee.Geometry,
    target_date: Optional[str],
) -> ee.Image:
    if style == "satellite":
        return _satellite_visual(region, target_date)
    if style == "terrain":
        return _terrain_visual()
    if style == "outdoors":
        return _outdoors_visual(region, target_date)
    raise ValueError(f"Unsupported basemap style '{style}'.")


@lru_cache(maxsize=768)
def _cached_basemap_tile(
    style: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    target_date: Optional[str],
) -> str:
    ensure_gee_initialized()
    region = _build_region_from_bbox(min_lon, min_lat, max_lon, max_lat)
    return _tile_url(_build_style_visual(style, region, target_date))


@lru_cache(maxsize=256)
def _cached_basemap(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    target_date: Optional[str],
) -> tuple[str, str, str]:
    satellite = _cached_basemap_tile(
        "satellite",
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        target_date,
    )
    terrain = _cached_basemap_tile(
        "terrain",
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        target_date,
    )
    outdoors = _cached_basemap_tile(
        "outdoors",
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        target_date,
    )
    return satellite, terrain, outdoors


def build_bbox_basemap_tile(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    target_date: Optional[str],
    style: str,
) -> Dict[str, Any]:
    normalized_style = style.lower()
    if normalized_style not in VALID_BASEMAP_STYLES:
        raise ValueError(f"Unsupported basemap style '{style}'.")

    return {
        "style": normalized_style,
        "url": _cached_basemap_tile(
            normalized_style,
            min_lon,
            min_lat,
            max_lon,
            max_lat,
            target_date,
        ),
        "attribution": "Google Earth Engine",
        "targetDate": target_date,
    }


def build_bbox_basemap(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    target_date: Optional[str],
) -> Dict[str, Any]:
    satellite, terrain, outdoors = _cached_basemap(
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        target_date,
    )

    return {
        "satellite": satellite,
        "terrain": terrain,
        "outdoors": outdoors,
        "attribution": "Google Earth Engine",
        "targetDate": target_date,
    }


def build_fire_basemap(metadata: WildfireMetadata, target_date: Optional[str]) -> Dict[str, Any]:
    min_lon, min_lat, max_lon, max_lat = metadata.bbox
    return build_bbox_basemap(
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        target_date,
    )
