import L from "leaflet";
import "maplibre-gl/dist/maplibre-gl.css";
import maplibregl from "maplibre-gl";
import { useMemo, useRef } from "react";
import { GeoJSON, MapContainer, TileLayer } from "react-leaflet";
import Map, { Layer, Source } from "react-map-gl/maplibre";
import HeatmapLayer from "./HeatmapLayer";

const FALLBACK_BASEMAP_URL = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
const FALLBACK_BASEMAP_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';
const EMPTY_FEATURE_COLLECTION = {
  type: "FeatureCollection",
  features: [],
};
const OVERVIEW_LAYER_ID = "overview-circles";
const PREDICTION_POLYGON_LAYER_ID = "prediction-polygons-3d";
const PREDICTION_HEAT_LAYER_ID = "prediction-heat-3d";
const GROUND_TRUTH_LAYER_ID = "ground-truth-3d";
const DIFFERENCE_LAYER_ID = "difference-3d";
const EXTENT_LAYER_ID = "extent-3d";
const ORIGIN_LAYER_ID = "origin-3d";
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
const TRUE_POSITIVE_HEAT_GRADIENT = {
  0.2: "#86efac",
  0.45: "#4ade80",
  0.7: "#22c55e",
  1.0: "#166534",
};
const FALSE_POSITIVE_HEAT_GRADIENT = {
  0.2: "#fdba74",
  0.45: "#fb923c",
  0.7: "#f97316",
  1.0: "#9a3412",
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

function getOverviewLayerOptions(selectedId, onOverviewSelect) {
  const hasSelection = Boolean(selectedId);

  return {
    pointToLayer: (feature, latlng) => {
      const isSelected = feature?.properties?.fireId === selectedId;
      const samples = Number(feature?.properties?.samples ?? 0);
      const color = samples >= 8 ? "#ef4444" : samples >= 3 ? "#f59e0b" : "#3b82f6";
      return L.circleMarker(latlng, {
        radius: isSelected ? 7.5 : hasSelection ? 4.5 : 6,
        fillColor: isSelected ? "#fff7ed" : color,
        color: isSelected
          ? "#fb923c"
          : hasSelection
            ? "rgba(148, 163, 184, 0.28)"
            : "rgba(255,255,255,0.7)",
        weight: isSelected ? 2.2 : 1.2,
        opacity: isSelected ? 1 : hasSelection ? 0.42 : 1,
        fillOpacity: isSelected ? 1 : hasSelection ? 0.38 : 0.92,
      });
    },
    onEachFeature: (feature, layer) => {
      if (feature?.properties?.fireId === selectedId) {
        layer.bringToFront?.();
      }
      layer.on("click", () => {
        const fireId = feature?.properties?.fireId;
        if (fireId) onOverviewSelect(fireId);
      });
    },
  };
}

function getPredictionPolygonOptions() {
  return {
    style: (feature) => ({
      color: "#f97316",
      weight: 1,
      opacity: 0.8,
      fillColor: "#ef4444",
      fillOpacity: Math.max(0.15, Math.min(0.7, Number(feature?.properties?.probability ?? 0))),
    }),
  };
}

function getSelectedExtentOptions() {
  return {
    style: {
      color: "#fb923c",
      weight: 3,
      opacity: 0.95,
      fillOpacity: 0.04,
      dashArray: "7 5",
    },
  };
}

function getSelectedOriginOptions() {
  return {
    pointToLayer: (_feature, latlng) =>
      L.circleMarker(latlng, {
        radius: 9,
        fillColor: "#fff7ed",
        color: "#fb923c",
        weight: 3.2,
        opacity: 1,
        fillOpacity: 1,
      }),
  };
}

function buildMapLibreStyle(tileUrl, tileAttribution) {
  if (!tileUrl) {
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
        tiles: [tileUrl],
        tileSize: 256,
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

function createOverviewCircleLayer(selectedId) {
  const hasSelection = Boolean(selectedId);
  const selectedValue = selectedId ?? "";
  const sampleExpression = ["coalesce", ["to-number", ["get", "samples"]], 0];

  return {
    id: OVERVIEW_LAYER_ID,
    type: "circle",
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
        hasSelection,
        "rgba(148, 163, 184, 0.28)",
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
        hasSelection,
        0.42,
        1,
      ],
      "circle-stroke-opacity": [
        "case",
        ["==", ["get", "fireId"], selectedValue],
        1,
        hasSelection,
        0.42,
        1,
      ],
    },
  };
}

function createPredictionHeatLayer() {
  return {
    id: PREDICTION_HEAT_LAYER_ID,
    type: "circle",
    paint: {
      "circle-radius": [
        "interpolate",
        ["linear"],
        ["coalesce", ["to-number", ["get", "probability"]], 0],
        0,
        4,
        0.4,
        7,
        0.75,
        10,
        1,
        12,
      ],
      "circle-color": [
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
      ],
      "circle-opacity": 0.78,
      "circle-stroke-width": 0.6,
      "circle-stroke-color": "rgba(255,255,255,0.18)",
    },
  };
}

function createGroundTruthLayer() {
  return {
    id: GROUND_TRUTH_LAYER_ID,
    type: "circle",
    paint: {
      "circle-radius": 5,
      "circle-color": "#38bdf8",
      "circle-opacity": 0.82,
      "circle-stroke-width": 0.8,
      "circle-stroke-color": "#e0f2fe",
    },
  };
}

function createDifferenceLayer() {
  return {
    id: DIFFERENCE_LAYER_ID,
    type: "circle",
    paint: {
      "circle-radius": 5.5,
      "circle-color": [
        "match",
        ["get", "outcome"],
        "true_positive",
        "#22c55e",
        "false_positive",
        "#f97316",
        "false_negative",
        "#ef4444",
        "#94a3b8",
      ],
      "circle-opacity": 0.84,
      "circle-stroke-width": 0.8,
      "circle-stroke-color": "rgba(255,255,255,0.16)",
    },
  };
}

function createPredictionExtrusionLayer() {
  return {
    id: PREDICTION_POLYGON_LAYER_ID,
    type: "fill-extrusion",
    paint: {
      "fill-extrusion-color": [
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
      ],
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
    paint: {
      "circle-radius": 8,
      "circle-color": "#fff7ed",
      "circle-stroke-color": "#fb923c",
      "circle-stroke-width": 3,
      "circle-opacity": 1,
    },
  };
}

function ThreeDMapView({
  mapRef,
  initialViewState,
  mapProjection,
  tileUrl,
  tileAttribution,
  overviewData,
  onOverviewSelect,
  selectedId,
  predictionHeatmapData,
  groundTruthData,
  differenceData,
  predictionPolygonData,
  extentData,
  originData,
  layerVisibility,
}) {
  const mapInstanceRef = useRef(null);
  const mapStyle = useMemo(
    () => buildMapLibreStyle(tileUrl, tileAttribution),
    [tileAttribution, tileUrl],
  );
  const overviewLayer = useMemo(
    () => createOverviewCircleLayer(selectedId),
    [selectedId],
  );
  const interactiveLayerIds = layerVisibility.overview ? [OVERVIEW_LAYER_ID] : [];

  return (
    <Map
      ref={(instance) => {
        mapInstanceRef.current = instance;
        mapRef.current = instance?.getMap?.() ?? null;
      }}
      mapLib={maplibregl}
      initialViewState={initialViewState}
      mapStyle={mapStyle}
      projection={mapProjection}
      attributionControl
      interactiveLayerIds={interactiveLayerIds}
      maxPitch={75}
      dragRotate
      touchPitch
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
      onLoad={() => {
        mapRef.current = mapInstanceRef.current?.getMap?.() ?? null;
      }}
      onRemove={() => {
        mapRef.current = null;
      }}
    >
      {layerVisibility.overview && overviewData ? (
        <Source id="overview-source" type="geojson" data={overviewData}>
          <Layer {...overviewLayer} />
        </Source>
      ) : null}

      {layerVisibility.predictionHeatmap ? (
        <Source
          id="prediction-heat-source"
          type="geojson"
          data={predictionHeatmapData ?? EMPTY_FEATURE_COLLECTION}
        >
          <Layer {...createPredictionHeatLayer()} />
        </Source>
      ) : null}

      {layerVisibility.groundTruthHeatmap ? (
        <Source
          id="ground-truth-source"
          type="geojson"
          data={groundTruthData ?? EMPTY_FEATURE_COLLECTION}
        >
          <Layer {...createGroundTruthLayer()} />
        </Source>
      ) : null}

      {layerVisibility.differenceHeatmap ? (
        <Source
          id="difference-source"
          type="geojson"
          data={differenceData ?? EMPTY_FEATURE_COLLECTION}
        >
          <Layer {...createDifferenceLayer()} />
        </Source>
      ) : null}

      {layerVisibility.predictionPolygons ? (
        <Source
          id="prediction-polygon-source"
          type="geojson"
          data={predictionPolygonData ?? EMPTY_FEATURE_COLLECTION}
        >
          <Layer {...createPredictionExtrusionLayer()} />
        </Source>
      ) : null}

      {layerVisibility.extent ? (
        <Source id="extent-source" type="geojson" data={extentData ?? EMPTY_FEATURE_COLLECTION}>
          <Layer {...createExtentLayer()} />
        </Source>
      ) : null}

      {layerVisibility.origin ? (
        <Source id="origin-source" type="geojson" data={originData ?? EMPTY_FEATURE_COLLECTION}>
          <Layer {...createOriginLayer()} />
        </Source>
      ) : null}
    </Map>
  );
}

function LeafletMapView({
  mapRef,
  initialViewState,
  tileUrl,
  tileAttribution,
  overviewData,
  onOverviewSelect,
  selectedId,
  fireLayers,
  layerVisibility,
  selectedExtentData,
  selectedOriginData,
}) {
  const predictionHeatPoints = useMemo(
    () =>
      (fireLayers?.predictionHeatmap?.features ?? [])
        .map((feature) => {
          const [lon, lat] = feature?.geometry?.coordinates ?? [];
          const probability = Number(feature?.properties?.probability ?? 0);
          if (!Number.isFinite(lat) || !Number.isFinite(lon) || probability <= 0) {
            return null;
          }
          return [lat, lon, Math.min(1, Math.max(probability, 0.35))];
        })
        .filter(Boolean),
    [fireLayers],
  );
  const groundTruthHeatPoints = useMemo(
    () =>
      (fireLayers?.groundTruthHeatmap?.features ?? [])
        .map((feature) => {
          const [lon, lat] = feature?.geometry?.coordinates ?? [];
          if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
            return null;
          }
          return [lat, lon, 1.0];
        })
        .filter(Boolean),
    [fireLayers],
  );
  const differenceHeatGroups = useMemo(() => {
    const groups = {
      true_positive: [],
      false_positive: [],
      false_negative: [],
    };

    (fireLayers?.differenceHeatmap?.features ?? []).forEach((feature) => {
      const [lon, lat] = feature?.geometry?.coordinates ?? [];
      const outcome = feature?.properties?.outcome;
      if (!Number.isFinite(lat) || !Number.isFinite(lon) || !groups[outcome]) {
        return;
      }
      groups[outcome].push([lat, lon, 1]);
    });

    return groups;
  }, [fireLayers]);
  const overviewOptions = useMemo(
    () => getOverviewLayerOptions(selectedId, onOverviewSelect),
    [onOverviewSelect, selectedId],
  );

  return (
    <MapContainer
      ref={mapRef}
      center={[initialViewState.latitude, initialViewState.longitude]}
      zoom={initialViewState.zoom}
      zoomControl={false}
      preferCanvas
      zoomSnap={0.25}
      zoomDelta={0.5}
      className="map-container"
      style={{ position: "fixed", inset: 0, width: "100vw", height: "100vh" }}
    >
      {tileUrl ? (
        <TileLayer
          key={tileUrl}
          url={tileUrl}
          attribution={tileAttribution}
          maxZoom={18}
        />
      ) : null}

      {layerVisibility.overview && overviewData ? (
        <GeoJSON data={overviewData} {...overviewOptions} />
      ) : null}

      {predictionHeatPoints.length > 0 && layerVisibility.predictionHeatmap ? (
        <HeatmapLayer
          points={predictionHeatPoints}
          radius={24}
          blur={34}
          maxZoom={18}
          minOpacity={0.95}
          gradient={PREDICTION_HEAT_GRADIENT}
        />
      ) : null}

      {groundTruthHeatPoints.length > 0 && layerVisibility.groundTruthHeatmap ? (
        <HeatmapLayer
          points={groundTruthHeatPoints}
          radius={22}
          blur={30}
          maxZoom={18}
          minOpacity={0.95}
          gradient={GROUND_TRUTH_HEAT_GRADIENT}
        />
      ) : null}

      {fireLayers?.predictionPolygons && layerVisibility.predictionPolygons ? (
        <GeoJSON data={fireLayers.predictionPolygons} {...getPredictionPolygonOptions()} />
      ) : null}

      {layerVisibility.differenceHeatmap && differenceHeatGroups.true_positive.length > 0 ? (
        <HeatmapLayer
          points={differenceHeatGroups.true_positive}
          radius={18}
          blur={26}
          maxZoom={18}
          minOpacity={0.9}
          gradient={TRUE_POSITIVE_HEAT_GRADIENT}
        />
      ) : null}

      {layerVisibility.differenceHeatmap && differenceHeatGroups.false_positive.length > 0 ? (
        <HeatmapLayer
          points={differenceHeatGroups.false_positive}
          radius={18}
          blur={26}
          maxZoom={18}
          minOpacity={0.9}
          gradient={FALSE_POSITIVE_HEAT_GRADIENT}
        />
      ) : null}

      {layerVisibility.differenceHeatmap && differenceHeatGroups.false_negative.length > 0 ? (
        <HeatmapLayer
          points={differenceHeatGroups.false_negative}
          radius={18}
          blur={26}
          maxZoom={18}
          minOpacity={0.9}
          gradient={FALSE_NEGATIVE_HEAT_GRADIENT}
        />
      ) : null}

      {fireLayers?.extent && layerVisibility.extent ? (
        <GeoJSON data={fireLayers.extent} {...getSelectedExtentOptions()} />
      ) : null}

      {!fireLayers?.extent && selectedExtentData && layerVisibility.extent ? (
        <GeoJSON data={selectedExtentData} {...getSelectedExtentOptions()} />
      ) : null}

      {fireLayers?.origin && layerVisibility.origin ? (
        <GeoJSON data={fireLayers.origin} {...getSelectedOriginOptions()} />
      ) : null}

      {!fireLayers?.origin && selectedOriginData && layerVisibility.origin ? (
        <GeoJSON data={selectedOriginData} {...getSelectedOriginOptions()} />
      ) : null}
    </MapContainer>
  );
}

export default function MapView({
  mapRef,
  initialViewState,
  viewMode,
  mapProvider,
  mapStyle,
  osmProjection,
  onOverviewSelect,
  overviewData,
  selectedId,
  selectedFire,
  basemap,
  allowFallbackBasemap = false,
  fireLayers,
  layerVisibility,
}) {
  const selectedExtentData = buildSelectedExtentGeojson(selectedFire);
  const selectedOriginData = buildSelectedOriginGeojson(selectedFire);
  const hasFullBasemapSet =
    Boolean(basemap?.satellite) &&
    Boolean(basemap?.terrain) &&
    Boolean(basemap?.outdoors);
  const basemapUrl = mapProvider === "gee"
    ? (basemap?.[mapStyle] ?? (hasFullBasemapSet ? basemap?.satellite : null))
    : null;
  const mapProjection =
    mapProvider === "osm" && osmProjection === "globe" ? "globe" : "mercator";
  const tileUrl = mapProvider === "osm"
    ? FALLBACK_BASEMAP_URL
    : basemapUrl ?? (allowFallbackBasemap ? FALLBACK_BASEMAP_URL : null);
  const tileAttribution = mapProvider === "osm"
    ? FALLBACK_BASEMAP_ATTRIBUTION
    : basemap?.attribution ?? (tileUrl ? FALLBACK_BASEMAP_ATTRIBUTION : undefined);
  const predictionHeatmapData = fireLayers?.predictionHeatmap ?? EMPTY_FEATURE_COLLECTION;
  const groundTruthData = fireLayers?.groundTruthHeatmap ?? EMPTY_FEATURE_COLLECTION;
  const differenceData = fireLayers?.differenceHeatmap ?? EMPTY_FEATURE_COLLECTION;
  const predictionPolygonData = fireLayers?.predictionPolygons ?? EMPTY_FEATURE_COLLECTION;
  const extentData = fireLayers?.extent ?? selectedExtentData ?? EMPTY_FEATURE_COLLECTION;
  const originData = fireLayers?.origin ?? selectedOriginData ?? EMPTY_FEATURE_COLLECTION;

  if (viewMode === "3d") {
    return (
      <ThreeDMapView
        mapRef={mapRef}
        initialViewState={initialViewState}
        mapProjection={mapProjection}
        tileUrl={tileUrl}
        tileAttribution={tileAttribution}
        overviewData={overviewData}
        onOverviewSelect={onOverviewSelect}
        selectedId={selectedId}
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

  return (
    <LeafletMapView
      mapRef={mapRef}
      initialViewState={initialViewState}
      tileUrl={tileUrl}
      tileAttribution={tileAttribution}
      overviewData={overviewData}
      onOverviewSelect={onOverviewSelect}
      selectedId={selectedId}
      fireLayers={fireLayers}
      layerVisibility={layerVisibility}
      selectedExtentData={selectedExtentData}
      selectedOriginData={selectedOriginData}
    />
  );
}
