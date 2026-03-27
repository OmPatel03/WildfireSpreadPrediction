import "maplibre-gl/dist/maplibre-gl.css";
import maplibregl from "maplibre-gl";
import { useEffect, useMemo, useState } from "react";
import Map, { Layer, Source } from "react-map-gl/maplibre";
import "./MapView.css";

const OSM_STANDARD_BASEMAP_TILES = [
  "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
  "https://b.tile.openstreetmap.org/{z}/{x}/{y}.png",
  "https://c.tile.openstreetmap.org/{z}/{x}/{y}.png",
];
const OSM_STANDARD_BASEMAP_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';
const OSM_STANDARD_BASEMAP_MAX_ZOOM = 19;
const OSM_TERRAIN_BASEMAP_TILES = [
  "https://a.tile.opentopomap.org/{z}/{x}/{y}.png",
  "https://b.tile.opentopomap.org/{z}/{x}/{y}.png",
  "https://c.tile.opentopomap.org/{z}/{x}/{y}.png",
];
const OSM_TERRAIN_BASEMAP_ATTRIBUTION =
  'Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, ' +
  '<a href="https://viewfinderpanoramas.org">SRTM</a> | ' +
  'Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a> ' +
  '(<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)';
const OSM_TERRAIN_BASEMAP_MAX_ZOOM = 17;

const EMPTY_FEATURE_COLLECTION = {
  type: "FeatureCollection",
  features: [],
};

const OVERVIEW_LAYER_ID = "overview-circles";
const PREDICTION_HEAT_LAYER_ID = "prediction-heat-2d";
const GROUND_TRUTH_HEAT_LAYER_ID = "ground-truth-heat-2d";
const DIFFERENCE_POINT_LAYER_ID = "difference-points-3d";
const DIFFERENCE_TRUE_POSITIVE_LAYER_ID = `${DIFFERENCE_POINT_LAYER_ID}-true-positive`;
const DIFFERENCE_FALSE_POSITIVE_LAYER_ID = `${DIFFERENCE_POINT_LAYER_ID}-false-positive`;
const DIFFERENCE_FALSE_NEGATIVE_LAYER_ID = `${DIFFERENCE_POINT_LAYER_ID}-false-negative`;
const PREDICTION_POLYGON_EXTRUSION_LAYER_ID = "prediction-polygons-3d";
const EXTENT_LAYER_ID = "extent";
const ORIGIN_LAYER_ID = "origin";

const PREDICTION_HEAT_GRADIENT = {
  0.2: "#f59e0b",
  0.45: "#f97316",
  0.7: "#ef4444",
  1.0: "#991b1b",
};
const GROUND_TRUTH_HEAT_GRADIENT = {
  0.2: "#7dd3fc",
  0.45: "#38bdf8",
  0.7: "#0ea5e9",
  1.0: "#075985",
};
const FALSE_POSITIVE_COLOR = "#8b5cf6";
const TRUE_POSITIVE_HEAT_GRADIENT = {
  0.2: "#86efac",
  0.45: "#4ade80",
  0.7: "#22c55e",
  1.0: "#166534",
};
const FALSE_POSITIVE_HEAT_GRADIENT = {
  0.2: "#ddd6fe",
  0.45: "#c4b5fd",
  0.7: FALSE_POSITIVE_COLOR,
  1.0: "#5b21b6",
};
const FALSE_NEGATIVE_HEAT_GRADIENT = {
  0.2: "#fca5a5",
  0.45: "#f87171",
  0.7: "#ef4444",
  1.0: "#991b1b",
};
function buildSelectedExtentGeojson(selectedFire) {
  const bbox = selectedFire?.bbox;
  if (!bbox) return null;

  return {
    type: "FeatureCollection",
    features: [
      {
        type: "Feature",
        geometry: {
          type: "Polygon",
          coordinates: [[
            [bbox.minLon, bbox.minLat],
            [bbox.maxLon, bbox.minLat],
            [bbox.maxLon, bbox.maxLat],
            [bbox.minLon, bbox.maxLat],
            [bbox.minLon, bbox.minLat],
          ]],
        },
        properties: {
          fireId: selectedFire.fireId,
        },
      },
    ],
  };
}

function buildSelectedOriginGeojson(selectedFire) {
  if (
    !selectedFire ||
    !Number.isFinite(selectedFire.longitude) ||
    !Number.isFinite(selectedFire.latitude)
  ) {
    return null;
  }

  return {
    type: "FeatureCollection",
    features: [
      {
        type: "Feature",
        geometry: {
          type: "Point",
          coordinates: [selectedFire.longitude, selectedFire.latitude],
        },
        properties: {
          fireId: selectedFire.fireId,
          lat: selectedFire.latitude,
          lon: selectedFire.longitude,
        },
      },
    ],
  };
}

function buildLoadingExtentPolygon(map, bbox) {
  if (
    !map ||
    !bbox ||
    !Number.isFinite(bbox.minLon) ||
    !Number.isFinite(bbox.minLat) ||
    !Number.isFinite(bbox.maxLon) ||
    !Number.isFinite(bbox.maxLat)
  ) {
    return null;
  }

  const corners = [
    [bbox.minLon, bbox.maxLat],
    [bbox.maxLon, bbox.maxLat],
    [bbox.maxLon, bbox.minLat],
    [bbox.minLon, bbox.minLat],
  ];

  const projected = corners
    .map((corner) => map.project(corner))
    .filter((point) => Number.isFinite(point?.x) && Number.isFinite(point?.y));

  if (projected.length !== 4) {
    return null;
  }

  return projected
    .map((point) => `${point.x}px ${point.y}px`)
    .join(", ");
}

function buildMapLibreStyle(tileUrls, tileAttribution, tileMaxZoom) {
  if (!tileUrls?.length) {
    return {
      version: 8,
      sources: {},
      layers: [
        {
          id: "background",
          type: "background",
          paint: {
            "background-color": "#020617",
          },
        },
      ],
    };
  }

  return {
    version: 8,
    sources: {
      basemap: {
        type: "raster",
        tiles: tileUrls,
        tileSize: 256,
        maxzoom: tileMaxZoom,
        attribution: tileAttribution,
      },
    },
    layers: [
      {
        id: "basemap",
        type: "raster",
        source: "basemap",
      },
    ],
  };
}

function buildProbabilityColorExpression() {
  return [
    "interpolate",
    ["linear"],
    ["coalesce", ["to-number", ["get", "probability"]], 0],
    0,
    "#f59e0b",
    0.45,
    "#f97316",
    0.7,
    "#ef4444",
    1,
    "#991b1b",
  ];
}

function buildHeatmapColorExpression(gradient) {
  const expression = ["interpolate", ["linear"], ["heatmap-density"], 0, "rgba(2, 6, 23, 0)"];
  const stops = Object.entries(gradient)
    .map(([stop, color]) => [Number(stop), color])
    .filter(([stop]) => Number.isFinite(stop))
    .sort((a, b) => a[0] - b[0]);

  stops.forEach(([stop, color]) => {
    expression.push(stop, color);
  });

  return expression;
}

function buildHeatmapOpacityExpression(is3d, opacityScale = 1) {
  const baseStops = is3d
    ? [0, 0.78, 8, 0.88, 12, 0.78, 16, 0.58]
    : [0, 0.72, 8, 0.84, 12, 0.72, 16, 0.5];

  return [
    "interpolate",
    ["linear"],
    ["zoom"],
    ...baseStops.map((value, index) =>
      index % 2 === 0 ? value : Math.min(value * opacityScale, 1),
    ),
  ];
}

function filterFeatureCollectionByOutcome(collection, outcome) {
  const features = (collection?.features ?? []).filter(
    (feature) => feature?.properties?.outcome === outcome,
  );

  return {
    type: "FeatureCollection",
    features,
  };
}

function buildOrderedLayerIds({ is3d, layerVisibility, showExtentLayer }) {
  const orderedLayerIds = [];

  if (layerVisibility.groundTruthHeatmap) {
    orderedLayerIds.push(GROUND_TRUTH_HEAT_LAYER_ID);
  }

  if (layerVisibility.predictionHeatmap) {
    orderedLayerIds.push(PREDICTION_HEAT_LAYER_ID);
  }

  if (layerVisibility.differenceHeatmap) {
    orderedLayerIds.push(
      DIFFERENCE_FALSE_POSITIVE_LAYER_ID,
      DIFFERENCE_FALSE_NEGATIVE_LAYER_ID,
      DIFFERENCE_TRUE_POSITIVE_LAYER_ID,
    );
  }

  if (is3d && layerVisibility.predictionPolygons) {
    orderedLayerIds.push(PREDICTION_POLYGON_EXTRUSION_LAYER_ID);
  }

  if (showExtentLayer) {
    orderedLayerIds.push(EXTENT_LAYER_ID);
  }

  if (layerVisibility.origin) {
    orderedLayerIds.push(ORIGIN_LAYER_ID);
  }

  if (layerVisibility.overview) {
    orderedLayerIds.push(OVERVIEW_LAYER_ID);
  }

  return orderedLayerIds;
}

function reorderLayers(map, orderedLayerIds) {
  if (!map || (typeof map.isStyleLoaded === "function" && !map.isStyleLoaded())) {
    return;
  }

  let nextHigherLayerId;

  for (let index = orderedLayerIds.length - 1; index >= 0; index -= 1) {
    const layerId = orderedLayerIds[index];
    if (!map.getLayer(layerId)) {
      continue;
    }

    map.moveLayer(layerId, nextHigherLayerId);
    nextHigherLayerId = layerId;
  }
}

function createOverviewCircleLayer(selectedId) {
  const hasSelection = Boolean(selectedId);
  const selectedValue = selectedId ?? "";
  const sampleExpression = ["coalesce", ["to-number", ["get", "samples"]], 0];

  return {
    id: OVERVIEW_LAYER_ID,
    type: "circle",
    source: "overview-source",
    paint: {
      "circle-radius": [
        "case",
        ["==", ["get", "fireId"], selectedValue],
        8,
        hasSelection,
        4.5,
        [">=", sampleExpression, 8],
        6,
        [">=", sampleExpression, 3],
        5.5,
        5,
      ],
      "circle-color": [
        "case",
        ["==", ["get", "fireId"], selectedValue],
        "#fff7ed",
        [">=", sampleExpression, 8],
        "#ef4444",
        [">=", sampleExpression, 3],
        "#f59e0b",
        "#3b82f6",
      ],
      "circle-stroke-color": [
        "case",
        ["==", ["get", "fireId"], selectedValue],
        "#fb923c",
        "rgba(255,255,255,0.7)",
      ],
      "circle-stroke-width": [
        "case",
        ["==", ["get", "fireId"], selectedValue],
        2.2,
        1.2,
      ],
      "circle-opacity": [
        "case",
        ["==", ["get", "fireId"], selectedValue],
        1,
        1,
      ],
      "circle-stroke-opacity": [
        "case",
        ["==", ["get", "fireId"], selectedValue],
        1,
        1,
      ],
    },
  };
}

function createHeatmapLayer({
  id,
  gradient,
  filter,
  weightExpression = 1,
  mode = "2d",
  source,
  opacityScale = 1,
}) {
  const is3d = mode === "3d";
  const layer = {
    id,
    type: "heatmap",
    source,
    paint: {
      "heatmap-weight": weightExpression,
      "heatmap-intensity": is3d
        ? ["interpolate", ["linear"], ["zoom"], 0, 0.85, 4, 1.05, 8, 1.25, 12, 1.4]
        : ["interpolate", ["linear"], ["zoom"], 0, 0.75, 4, 0.95, 8, 1.15, 12, 1.3],
      "heatmap-radius": is3d
        ? ["interpolate", ["linear"], ["zoom"], 0, 10, 4, 16, 8, 24, 12, 36]
        : ["interpolate", ["linear"], ["zoom"], 0, 8, 4, 14, 8, 22, 12, 34],
      "heatmap-opacity": buildHeatmapOpacityExpression(is3d, opacityScale),
      "heatmap-color": buildHeatmapColorExpression(gradient),
    },
  };

  if (filter) {
    layer.filter = filter;
  }

  return layer;
}

function createPredictionExtrusionLayer() {
  return {
    id: PREDICTION_POLYGON_EXTRUSION_LAYER_ID,
    type: "fill-extrusion",
    source: "prediction-polygon-source",
    paint: {
      "fill-extrusion-color": buildProbabilityColorExpression(),
      "fill-extrusion-height": ["coalesce", ["to-number", ["get", "height"]], 0],
      "fill-extrusion-base": 0,
      "fill-extrusion-opacity": 0.78,
    },
  };
}

function createExtentLayer() {
  return {
    id: EXTENT_LAYER_ID,
    type: "line",
    source: "extent-source",
    paint: {
      "line-color": "#fb923c",
      "line-width": 3,
      "line-opacity": 0.95,
    },
    layout: {
      "line-join": "round",
      "line-cap": "round",
    },
  };
}

function createOriginLayer() {
  return {
    id: ORIGIN_LAYER_ID,
    type: "circle",
    source: "origin-source",
    paint: {
      "circle-radius": 8,
      "circle-color": "#fff7ed",
      "circle-stroke-color": "#fb923c",
      "circle-stroke-width": 3,
      "circle-opacity": 1,
    },
  };
}

function LoadingExtentOverlay({ map, bbox, active }) {
  const [clipPathPolygon, setClipPathPolygon] = useState(null);

  useEffect(() => {
    if (!active || !map || !bbox) {
      setClipPathPolygon(null);
      return undefined;
    }

    const updatePolygon = () => {
      const polygon = buildLoadingExtentPolygon(map, bbox);
      setClipPathPolygon(polygon ? `polygon(${polygon})` : null);
    };

    updatePolygon();
    map.on("move", updatePolygon);
    map.on("resize", updatePolygon);
    window.addEventListener("resize", updatePolygon);

    return () => {
      map.off("move", updatePolygon);
      map.off("resize", updatePolygon);
      window.removeEventListener("resize", updatePolygon);
    };
  }, [active, bbox, map]);

  if (!active || !clipPathPolygon) {
    return null;
  }

  return (
    <div className="loading-extent-overlay" aria-hidden="true">
      <div
        className="loading-extent-region"
        style={{ clipPath: clipPathPolygon }}
      />
    </div>
  );
}

function MapLibreView({
  mapRef,
  initialViewState,
  viewMode,
  mapProjection,
  maxZoom,
  tileUrls,
  tileAttribution,
  tileMaxZoom,
  overviewData,
  onOverviewSelect,
  selectedId,
  loadingExtentBbox,
  selectedFireLoading,
  predictionHeatmapData,
  groundTruthData,
  differenceData,
  predictionPolygonData,
  extentData,
  originData,
  layerVisibility,
}) {
  const is3d = viewMode === "3d";
  const [mapInstance, setMapInstance] = useState(null);
  const mapStyle = useMemo(
    () => buildMapLibreStyle(tileUrls, tileAttribution, tileMaxZoom),
    [tileAttribution, tileMaxZoom, tileUrls],
  );
  const overviewLayer = useMemo(
    () => createOverviewCircleLayer(selectedId),
    [selectedId],
  );
  const truePositiveDifferenceData = useMemo(
    () => filterFeatureCollectionByOutcome(differenceData, "true_positive"),
    [differenceData],
  );
  const falsePositiveDifferenceData = useMemo(
    () => filterFeatureCollectionByOutcome(differenceData, "false_positive"),
    [differenceData],
  );
  const falseNegativeDifferenceData = useMemo(
    () => filterFeatureCollectionByOutcome(differenceData, "false_negative"),
    [differenceData],
  );
  const interactiveLayerIds = layerVisibility.overview ? [OVERVIEW_LAYER_ID] : [];
  const activeProjection = is3d ? mapProjection : "mercator";
  const showExtentLayer = layerVisibility.extent || selectedFireLoading;
  const orderedLayerIds = useMemo(
    () =>
      buildOrderedLayerIds({
        is3d,
        layerVisibility,
        showExtentLayer,
      }),
    [is3d, layerVisibility, showExtentLayer],
  );

  useEffect(() => {
    if (!mapInstance) {
      return undefined;
    }

    let frameId = null;

    const queueReorder = () => {
      if (frameId !== null) {
        window.cancelAnimationFrame(frameId);
      }

      frameId = window.requestAnimationFrame(() => {
        frameId = null;
        reorderLayers(mapInstance, orderedLayerIds);
      });
    };

    queueReorder();
    mapInstance.on("styledata", queueReorder);

    return () => {
      if (frameId !== null) {
        window.cancelAnimationFrame(frameId);
      }
      mapInstance.off("styledata", queueReorder);
    };
  }, [mapInstance, orderedLayerIds]);

  return (
    <>
      <Map
        ref={(instance) => {
          const nextMap = instance?.getMap?.() ?? null;
          mapRef.current = nextMap;
        }}
        mapLib={maplibregl}
        initialViewState={initialViewState}
        mapStyle={mapStyle}
        projection={activeProjection}
        attributionControl
        interactiveLayerIds={interactiveLayerIds}
        maxZoom={maxZoom}
        maxPitch={is3d ? 75 : 0}
        dragRotate={is3d}
        touchPitch={is3d}
        style={{ position: "fixed", inset: 0, width: "100vw", height: "100vh" }}
        className="map-container"
        onClick={(event) => {
          const selectedFeature = event.features?.find(
            (feature) => feature?.layer?.id === OVERVIEW_LAYER_ID,
          );
          const fireId = selectedFeature?.properties?.fireId;
          if (fireId) {
            onOverviewSelect(fireId);
          }
        }}
        onLoad={(event) => {
          const nextMap = event.target ?? null;
          mapRef.current = nextMap;
          setMapInstance(nextMap);
        }}
        onRemove={() => {
          mapRef.current = null;
          setMapInstance(null);
        }}
      >
        {layerVisibility.predictionHeatmap ? (
          <Source
            id="prediction-source"
            type="geojson"
            data={predictionHeatmapData ?? EMPTY_FEATURE_COLLECTION}
          >
            <Layer
              {...createHeatmapLayer({
                id: PREDICTION_HEAT_LAYER_ID,
                gradient: PREDICTION_HEAT_GRADIENT,
                opacityScale: 0.72,
                weightExpression: [
                  "interpolate",
                  ["linear"],
                  ["coalesce", ["to-number", ["get", "probability"]], 0],
                  0,
                  0,
                  0.35,
                  0.55,
                  1,
                  1,
                ],
                mode: is3d ? "3d" : "2d",
                source: "prediction-source",
              })}
            />
          </Source>
        ) : null}

        {layerVisibility.groundTruthHeatmap ? (
          <Source
            id="ground-truth-source"
            type="geojson"
            data={groundTruthData ?? EMPTY_FEATURE_COLLECTION}
          >
            <Layer
              {...createHeatmapLayer({
                id: GROUND_TRUTH_HEAT_LAYER_ID,
                gradient: GROUND_TRUTH_HEAT_GRADIENT,
                opacityScale: 0.72,
                mode: is3d ? "3d" : "2d",
                source: "ground-truth-source",
              })}
            />
          </Source>
        ) : null}

        {layerVisibility.differenceHeatmap ? (
          <>
            <Source
              id="difference-false-positive-source"
              type="geojson"
              data={falsePositiveDifferenceData}
            >
              <Layer
                {...createHeatmapLayer({
                  id: DIFFERENCE_FALSE_POSITIVE_LAYER_ID,
                  gradient: FALSE_POSITIVE_HEAT_GRADIENT,
                  mode: is3d ? "3d" : "2d",
                  source: "difference-false-positive-source",
                })}
              />
            </Source>
            <Source
              id="difference-false-negative-source"
              type="geojson"
              data={falseNegativeDifferenceData}
            >
              <Layer
                {...createHeatmapLayer({
                  id: DIFFERENCE_FALSE_NEGATIVE_LAYER_ID,
                  gradient: FALSE_NEGATIVE_HEAT_GRADIENT,
                  mode: is3d ? "3d" : "2d",
                  source: "difference-false-negative-source",
                })}
              />
            </Source>
            <Source
              id="difference-true-positive-source"
              type="geojson"
              data={truePositiveDifferenceData}
            >
              <Layer
                {...createHeatmapLayer({
                  id: DIFFERENCE_TRUE_POSITIVE_LAYER_ID,
                  gradient: TRUE_POSITIVE_HEAT_GRADIENT,
                  mode: is3d ? "3d" : "2d",
                  source: "difference-true-positive-source",
                })}
              />
            </Source>
          </>
        ) : null}

        {is3d && layerVisibility.predictionPolygons ? (
          <Source
            id="prediction-polygon-source"
            type="geojson"
            data={predictionPolygonData ?? EMPTY_FEATURE_COLLECTION}
          >
            <Layer {...createPredictionExtrusionLayer()} />
          </Source>
        ) : null}

        {showExtentLayer ? (
          <Source id="extent-source" type="geojson" data={extentData ?? EMPTY_FEATURE_COLLECTION}>
            <Layer {...createExtentLayer()} />
          </Source>
        ) : null}

        {layerVisibility.origin ? (
          <Source id="origin-source" type="geojson" data={originData ?? EMPTY_FEATURE_COLLECTION}>
            <Layer {...createOriginLayer()} />
          </Source>
        ) : null}

        {layerVisibility.overview ? (
          <Source id="overview-source" type="geojson" data={overviewData ?? EMPTY_FEATURE_COLLECTION}>
            <Layer {...overviewLayer} />
          </Source>
        ) : null}
      </Map>

      <LoadingExtentOverlay
        map={mapInstance}
        bbox={loadingExtentBbox}
        active={selectedFireLoading}
      />
    </>
  );
}

export default function MapView({
  mapRef,
  initialViewState,
  viewMode,
  osmMapStyle,
  osmProjection,
  onOverviewSelect,
  overviewData,
  selectedId,
  selectedFire,
  fireLayers,
  selectedFireLoading = false,
  fireDetailMaxZoom,
  layerVisibility,
}) {
  const selectedExtentData = buildSelectedExtentGeojson(selectedFire);
  const selectedOriginData = buildSelectedOriginGeojson(selectedFire);
  const osmTileConfig = osmMapStyle === "terrain"
    ? {
        tiles: OSM_TERRAIN_BASEMAP_TILES,
        attribution: OSM_TERRAIN_BASEMAP_ATTRIBUTION,
        maxZoom: OSM_TERRAIN_BASEMAP_MAX_ZOOM,
      }
    : {
        tiles: OSM_STANDARD_BASEMAP_TILES,
        attribution: OSM_STANDARD_BASEMAP_ATTRIBUTION,
        maxZoom: OSM_STANDARD_BASEMAP_MAX_ZOOM,
      };
  const mapProjection =
    osmProjection === "globe" ? "globe" : "mercator";
  const tileUrls = osmTileConfig.tiles;
  const tileAttribution = osmTileConfig.attribution;
  const tileMaxZoom = osmTileConfig.maxZoom;
  const interactiveMaxZoom = selectedFire ? fireDetailMaxZoom : undefined;
  const predictionHeatmapData = fireLayers?.predictionHeatmap ?? EMPTY_FEATURE_COLLECTION;
  const groundTruthData = fireLayers?.groundTruthHeatmap ?? EMPTY_FEATURE_COLLECTION;
  const differenceData = fireLayers?.differenceHeatmap ?? EMPTY_FEATURE_COLLECTION;
  const predictionPolygonData = fireLayers?.predictionPolygons ?? EMPTY_FEATURE_COLLECTION;
  const extentData = fireLayers?.extent ?? selectedExtentData ?? EMPTY_FEATURE_COLLECTION;
  const originData = fireLayers?.origin ?? selectedOriginData ?? EMPTY_FEATURE_COLLECTION;

  return (
    <MapLibreView
      mapRef={mapRef}
      initialViewState={initialViewState}
      viewMode={viewMode}
      mapProjection={mapProjection}
      maxZoom={interactiveMaxZoom}
      tileUrls={tileUrls}
      tileAttribution={tileAttribution}
      tileMaxZoom={tileMaxZoom}
      overviewData={overviewData}
      onOverviewSelect={onOverviewSelect}
      selectedId={selectedId}
      loadingExtentBbox={selectedFire?.bbox ?? null}
      selectedFireLoading={selectedFireLoading}
      predictionHeatmapData={predictionHeatmapData}
      groundTruthData={groundTruthData}
      differenceData={differenceData}
      predictionPolygonData={predictionPolygonData}
      extentData={extentData}
      originData={originData}
      layerVisibility={layerVisibility}
    />
  );
}
