import React, { useEffect, useMemo, useRef, useState } from "react";
import Map, { Source, Layer } from "react-map-gl/mapbox";
import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";
import { heatmapLayer, probabilityLayer } from "./mapLayers";
import "./app.css";
import { buildCoordinatesArray, probabilityToMagnitude } from "./util/convert.js";
import data from '../test/response.json' with { type: 'json' };

const centerLat = data["data"]["findSpread"]["fire"]["latitude"];
const centerLong = data["data"]["findSpread"]["fire"]["longitude"];
const probabilities =
  data["data"]["findSpread"]["geojson"]["features"][0]["properties"]["prediction"]["probabilities"];

const groundTruth = data["data"]["findSpread"]["geojson"]["features"][0]["properties"]["groundTruthMask"]
const cols = data["data"]["findSpread"]["geojson"]["features"][0]["properties"]["shape"]["width"]
const rows = data["data"]["findSpread"]["geojson"]["features"][0]["properties"]["shape"]["height"]

console.log(probabilities)

const testingFeatures = buildCoordinatesArray(
  rows,
  cols,
  centerLat,
  centerLong,
  probabilities
);
const groundTruthFeatures = buildCoordinatesArray(
  rows,
  cols,
  centerLat,
  centerLong,
  groundTruth
);

export default function App() {
  const [selectedId, setSelectedId] = useState(null); 
  const [dayIndex, setDayIndex] = useState(0);
  const [data, setData] = useState(null);
  const mapRef = useRef(null);

  const token = import.meta.env.VITE_MAPBOX_TOKEN;
  if (!token) console.warn("VITE_MAPBOX_TOKEN not set");

  // Example dataset
  const datasets = useMemo(
    () => ({
      1: {
        1: {
          type: "FeatureCollection",
          features: testingFeatures,
        },
        2: {
          type: "FeatureCollection",
          features: groundTruthFeatures,
        },
      },
      2: {
        1: {
          type: "FeatureCollection",
          features: [
            {
              type: "Feature",
              geometry: { type: "Point", coordinates: [-118, 39.5] },
              properties: { mag: 2.0 },
            },
          ],
        },
        2: {
          type: "FeatureCollection",
          features: [
            {
              type: "Feature",
              geometry: { type: "Point", coordinates: [-118.2, 39.6] },
              properties: { mag: 2.7 },
            },
          ],
        },
      },
    }),
    []
  );

  const dayKeys = useMemo(() => {
    if (!selectedId || !datasets[selectedId]) return [];
    const keys = Object.keys(datasets[selectedId]);
    const numeric = keys.every((k) => !isNaN(Number(k)));
    return numeric
      ? keys
          .map(Number)
          .sort((a, b) => a - b)
          .map(String)
      : keys.sort();
  }, [datasets, selectedId]);

  useEffect(() => {
    if (!selectedId || dayKeys.length === 0) {
      setData(null);
      return;
    }
    const idx = Math.max(0, Math.min(dayIndex, dayKeys.length - 1));
    const currentDayKey = dayKeys[idx];
    setData(datasets[selectedId][currentDayKey]);
  }, [selectedId, dayIndex, dayKeys, datasets]);

  const handleWildfireChange = (newId) => {
    if (newId === "none") {
      setSelectedId(null);
      setData(null);
      return;
    }

    const numericId = Number(newId);
    setSelectedId(numericId);
    setDayIndex(0);

    // fly to the the middle of all coordinates with positive probability
    const wildfire = datasets[numericId];
    if (!wildfire) return;

    const firstDayKey = Object.keys(wildfire)[0];
    const dayData = wildfire[firstDayKey];
    const features = dayData?.features ?? [];
    const positive = features.filter(
      (feature) => (feature?.properties?.mag ?? 0) > 0 && feature?.geometry?.coordinates
    );

    let coords;
    if (positive.length > 0) {
      const total = positive.reduce(
        (acc, feature) => {
          const [lon, lat] = feature.geometry.coordinates;
          return { lat: acc.lat + lat, lon: acc.lon + lon };
        },
        { lat: 0, lon: 0 }
      );
      coords = [total.lon / positive.length, total.lat / positive.length];
    } else {
      const fallback = features[0];
      coords = centerLong && centerLat ? [centerLong, centerLat] : fallback?.geometry?.coordinates;
    }

    if (coords && mapRef.current) {
      mapRef.current.flyTo({
        center: coords,
        zoom: 12,
        speed: 0.8,
        curve: 1.4,
      });
    }
  };

  const prevDay = () => {
    if (dayKeys.length === 0) return;
    setDayIndex((i) => (i - 1 + dayKeys.length) % dayKeys.length);
  };
  const nextDay = () => {
    if (dayKeys.length === 0) return;
    setDayIndex((i) => (i + 1) % dayKeys.length);
  };

  return (
    <>
      <div className="overlay-top-right app-overlay controls-row">
        <div className="control">
          <label htmlFor="wildfire" className="dropdown-label">
            Wildfire
          </label>
          <select
            id="wildfire"
            className="dropdown-select"
            value={selectedId ?? "none"}
            onChange={(e) => handleWildfireChange(e.target.value)}
          >
            <option value="none" disabled>
              — Select a wildfire —
            </option>
            <option value={1}>Yellowstone</option>
            <option value={2}>Nevada</option>
          </select>
        </div>

        <div className="day-nav">
          <button
            type="button"
            className="arrow-btn"
            onClick={prevDay}
            disabled={!selectedId || dayKeys.length <= 1}
          >
            ‹
          </button>
          <span className="day-label">
            {selectedId
              ? dayKeys.length
                ? `Day: ${dayKeys[dayIndex]}`
                : "No data"
              : "No wildfire selected"}
          </span>
          <button
            type="button"
            className="arrow-btn"
            onClick={nextDay}
            disabled={!selectedId || dayKeys.length <= 1}
          >
            ›
          </button>
        </div>
      </div>

      <Map
        ref={mapRef}
        initialViewState={{ longitude: -100, latitude: 40, zoom: 3.5 }}
        mapboxAccessToken={token}
        mapStyle="mapbox://styles/mapbox/dark-v10"
        className="map-container"
        style={{ position: "fixed", inset: 0, width: "100vw", height: "100vh" }}
      >
        {selectedId && data && (
          <Source id="wildfires" type="geojson" data={data}>
            <Layer {...heatmapLayer} />
            {/* <Layer {...probabilityLayer} /> */}
          </Source>
        )}
      </Map>
    </>
  );
}
