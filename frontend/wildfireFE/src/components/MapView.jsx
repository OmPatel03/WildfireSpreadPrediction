import L from "leaflet";
import { useMemo } from "react";
import { GeoJSON, MapContainer, TileLayer } from "react-leaflet";
import HeatmapLayer from "./HeatmapLayer";

const FALLBACK_BASEMAP_URL = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
const FALLBACK_BASEMAP_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';
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
  return {
    pointToLayer: (feature, latlng) => {
      const isSelected = feature?.properties?.fireId === selectedId;
      const samples = Number(feature?.properties?.samples ?? 0);
      const color = samples >= 8 ? "#ef4444" : samples >= 3 ? "#f59e0b" : "#3b82f6";
      return L.circleMarker(latlng, {
        radius: isSelected ? 9 : 6,
        fillColor: isSelected ? "#ffffff" : color,
        color: isSelected ? "#2563eb" : "rgba(255,255,255,0.7)",
        weight: isSelected ? 2 : 1.5,
        opacity: 1,
        fillOpacity: 0.92,
      });
    },
    onEachFeature: (feature, layer) => {
      layer.on("click", () => {
        const fireId = feature?.properties?.fireId;
        if (fireId) onOverviewSelect(fireId);
      });
    },
  };
}

function getDifferenceOptions() {
  return {
    pointToLayer: (feature, latlng) => {
      const outcome = feature?.properties?.outcome;
      const color =
        outcome === "true_positive"
          ? "#22c55e"
          : outcome === "false_positive"
            ? "#f97316"
            : outcome === "false_negative"
              ? "#ef4444"
              : "#94a3b8";
      return L.circleMarker(latlng, {
        radius: 4,
        fillColor: color,
        color,
        weight: 1,
        opacity: 0.9,
        fillOpacity: 0.8,
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

function getExtentOptions() {
  return {
    style: {
      color: "#f8fafc",
      weight: 2,
      opacity: 0.9,
      fillOpacity: 0,
      dashArray: "6 4",
    },
  };
}

function getSelectedExtentOptions() {
  return {
    style: {
      color: "#38bdf8",
      weight: 3,
      opacity: 0.95,
      fillOpacity: 0,
      dashArray: "6 4",
    },
  };
}

function getOriginOptions() {
  return {
    pointToLayer: (_feature, latlng) =>
      L.circleMarker(latlng, {
        radius: 6,
        fillColor: "#ffffff",
        color: "#0ea5e9",
        weight: 2,
        opacity: 1,
        fillOpacity: 1,
      }),
  };
}

function getSelectedOriginOptions() {
  return {
    pointToLayer: (_feature, latlng) =>
      L.circleMarker(latlng, {
        radius: 8,
        fillColor: "#ffffff",
        color: "#38bdf8",
        weight: 3,
        opacity: 1,
        fillOpacity: 1,
      }),
  };
}

export default function MapView({
  mapRef,
  initialViewState,
  mapStyle,
  onOverviewSelect,
  overviewData,
  selectedId,
  selectedFire,
  basemap,
  fireLayers,
  layerVisibility,
}) {
  const selectedExtentData = buildSelectedExtentGeojson(selectedFire);
  const selectedOriginData = buildSelectedOriginGeojson(selectedFire);
  const basemapUrl = basemap?.[mapStyle] ?? basemap?.satellite ?? null;
  const tileUrl = basemapUrl ?? FALLBACK_BASEMAP_URL;
  const tileAttribution = basemap?.attribution ?? FALLBACK_BASEMAP_ATTRIBUTION;
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
      <TileLayer
        key={tileUrl}
        url={tileUrl}
        attribution={tileAttribution}
        maxZoom={18}
      />

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

      {fireLayers?.differenceHeatmap && layerVisibility.differenceHeatmap ? (
        <GeoJSON data={fireLayers.differenceHeatmap} {...getDifferenceOptions()} />
      ) : null}

      {fireLayers?.extent && layerVisibility.extent ? (
        <GeoJSON data={fireLayers.extent} {...getExtentOptions()} />
      ) : null}

      {!fireLayers?.extent && selectedExtentData && layerVisibility.extent ? (
        <GeoJSON data={selectedExtentData} {...getSelectedExtentOptions()} />
      ) : null}

      {fireLayers?.origin && layerVisibility.origin ? (
        <GeoJSON data={fireLayers.origin} {...getOriginOptions()} />
      ) : null}

      {!fireLayers?.origin && selectedOriginData && layerVisibility.origin ? (
        <GeoJSON data={selectedOriginData} {...getSelectedOriginOptions()} />
      ) : null}
    </MapContainer>
  );
}
