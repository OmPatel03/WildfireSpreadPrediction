import React, { useEffect, useMemo, useRef, useState } from "react";
import Map, { Source, Layer } from "react-map-gl/mapbox";
import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";
import { heatmapLayer, probabilityLayer } from "./mapLayers";
import "./app.css";

const testingCoordinates = [
  [43.46385405405405, -80.55334599414483],
  [43.46385405405405, -80.54996759648691],
  [43.46385405405405, -80.54658919882897],
  [43.46385405405405, -80.54321080117104],
  [43.46385405405405, -80.53983240351312],
  [43.46723243243243, -80.55334599414483],
  [43.46723243243243, -80.54996759648691],
  [43.46723243243243, -80.54658919882897],
  [43.46723243243243, -80.54321080117104],
  [43.46723243243243, -80.53983240351312],
  [43.47061081081081, -80.55334599414483],
  [43.47061081081081, -80.54996759648691],
  [43.47061081081081, -80.54658919882897],
  [43.47061081081081, -80.54321080117104],
  [43.47061081081081, -80.53983240351312],
  [43.47398918918919, -80.55334599414483],
  [43.47398918918919, -80.54996759648691],
  [43.47398918918919, -80.54658919882897],
  [43.47398918918919, -80.54321080117104],
  [43.47398918918919, -80.53983240351312],
  [43.47736756756757, -80.55334599414483],
  [43.47736756756757, -80.54996759648691],
  [43.47736756756757, -80.54658919882897],
  [43.47736756756757, -80.54321080117104],
  [43.47736756756757, -80.53983240351312],
];

const testingFeatures = testingCoordinates.map(([lat, lon]) => ({
  type: "Feature",
  geometry: { type: "Point", coordinates: [lon, lat] },
  properties: { probability: 0.75 },
}));

export default function App() {
  const [selectedId, setSelectedId] = useState(null); // ← initially none selected
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
          features: [
            {
              type: "Feature",
              geometry: { type: "Point", coordinates: [-100.3, 40.2] },
              properties: { mag: 3.0 },
            },
          ],
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

    // fly to the first coordinate
    const wildfire = datasets[numericId];
    if (!wildfire) return;
    const firstDayKey = Object.keys(wildfire)[0];
    const firstFeature = wildfire[firstDayKey]?.features?.[0];
    const coords = firstFeature?.geometry?.coordinates;
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
            <Layer {...probabilityLayer} />
          </Source>
        )}
      </Map>
    </>
  );
}
