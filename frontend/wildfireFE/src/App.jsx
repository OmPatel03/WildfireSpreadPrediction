import React, { useEffect, useMemo, useRef, useState } from "react";
import Map, { Source, Layer } from "react-map-gl/mapbox";
import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";
import { heatmapLayer } from "./mapLayers";
import "./App.css";
import { buildCoordinatesArray, computeMetrics } from "./util/convert.js";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://wispr.cas.mcmaster.ca/api";
const DEFAULT_YEAR = 2021;
const MAX_FIRE_COUNT = 60;
const PAGE_SIZE = 10;
const SAMPLE_OFFSET = 19;
const THRESHOLD = 0.5;
const LAYER_SEQUENCE = ["prediction", "groundTruth"];
const LAYER_LABELS = {
  prediction: "Prediction",
  groundTruth: "Ground truth",
};
const INITIAL_VIEW = { longitude: -100, latitude: 40, zoom: 3.5 };

async function fetchJson(url, options = {}) {
  const headers = {
    Accept: "application/json",
    ...options.headers,
  };
  const response = await fetch(url, { ...options, headers });
  if (!response.ok) throw new Error(`Request failed (${response.status})`);
  return response.json();
}

export default function App() {
  const mapRef = useRef(null);
  const pendingSpreadRequestRef = useRef(0);

  const [catalog, setCatalog] = useState([]);
  const [catalogPage, setCatalogPage] = useState(0);
  const [catalogLoading, setCatalogLoading] = useState(false);
  const [catalogError, setCatalogError] = useState(null);

  const [selectedId, setSelectedId] = useState(null);
  const [layerData, setLayerData] = useState(null);
  const [activeLayerIndex, setActiveLayerIndex] = useState(0);
  const [data, setData] = useState(null);

  const [spreadLoading, setSpreadLoading] = useState(false);
  const [spreadError, setSpreadError] = useState(null);
  const [statistics, setStatistics] = useState(null);

  const token = import.meta.env.VITE_MAPBOX_TOKEN;
  if (!token) console.warn("VITE_MAPBOX_TOKEN not set");

  useEffect(() => {
    let ignore = false;
    const geocodeController = new AbortController();

    const fetchCatalog = async () => {
      setCatalogLoading(true);
      setCatalogError(null);

      try {
        const catalogUrl = new URL(`${API_BASE_URL}/catalog`);
        catalogUrl.searchParams.set("year", DEFAULT_YEAR);
        catalogUrl.searchParams.set("limit", MAX_FIRE_COUNT);
        catalogUrl.searchParams.set("offset", 0);

        const payload = await fetchJson(catalogUrl);
        let rows =
          payload?.catalog ??
          payload?.data?.catalog ??
          (Array.isArray(payload) ? payload : []);
        rows = Array.isArray(rows) ? rows : [];
        if (!ignore) {
          setCatalog(rows);
          setCatalogPage(0);
        }

        if (token) {
          try {
            const withLocations = await annotateCatalogWithLocations(
              rows,
              token,
              geocodeController.signal
            );
            if (!ignore) {
              setCatalog(withLocations);
            }
          } catch (geoError) {
            if (geoError?.name !== "AbortError") {
              console.warn("Location lookup failed:", geoError);
            }
          }
        }
      } catch (error) {
        if (!ignore) setCatalogError(error.message ?? "Unable to load catalog");
      } finally {
        if (!ignore) setCatalogLoading(false);
      }
    };

    fetchCatalog();
    return () => {
      ignore = true;
      geocodeController.abort();
    };
  }, [token]);

  const totalPages = Math.max(1, Math.ceil(catalog.length / PAGE_SIZE));
  const visibleCatalog = useMemo(() => {
    const start = catalogPage * PAGE_SIZE;
    return catalog.slice(start, start + PAGE_SIZE);
  }, [catalog, catalogPage]);

  const dayKeys = useMemo(
    () => LAYER_SEQUENCE.filter((key) => layerData?.[key]),
    [layerData]
  );

  useEffect(() => {
    if (!layerData || dayKeys.length === 0) {
      setData(null);
      setActiveLayerIndex(0);
      return;
    }

    const safeIndex = Math.min(activeLayerIndex, dayKeys.length - 1);
    if (safeIndex !== activeLayerIndex) {
      setActiveLayerIndex(safeIndex);
      return;
    }

    setData(layerData[dayKeys[safeIndex]]);
  }, [layerData, dayKeys, activeLayerIndex]);

  const handleCatalogPrev = () => {
    setCatalogPage((prev) => Math.max(prev - 1, 0));
  };

  const handleCatalogNext = () => {
    setCatalogPage((prev) => Math.min(prev + 1, totalPages - 1));
  };

  const fetchSpreadForFire = async (fireId) => {
    const fireMeta = catalog.find((fire) => fire.fireId === fireId);
    if (!fireMeta) {
      setSpreadError("Selected fire is not available in the catalog.");
      return;
    }

    const requestId = ++pendingSpreadRequestRef.current;
    setSpreadLoading(true);
    setSpreadError(null);

    try {
      const body = {
        fireId,
        year: DEFAULT_YEAR,
        sampleOffset: SAMPLE_OFFSET,
        probabilityThreshold: THRESHOLD,
      };

      const payload = await fetchJson(`${API_BASE_URL}/findSpread`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (pendingSpreadRequestRef.current !== requestId) return;

      const spread =
        payload?.findSpread ?? payload?.data?.findSpread ?? payload ?? {};
      const geojson =
        spread?.geojson ??
        spread?.GeoJSON ??
        payload?.geojson ??
        payload?.GeoJSON;

      if (!geojson) {
        throw new Error("Missing GeoJSON in find_spread response");
      }

      const featureProps =
        geojson?.features?.[0]?.properties ?? spread?.properties ?? {};
      const prediction =
        featureProps?.prediction?.probabilities ??
        featureProps?.prediction?.mask ??
        featureProps?.prediction ??
        [];
      const groundTruth =
        featureProps?.groundTruthMask ??
        featureProps?.groundTruth?.mask ??
        featureProps?.groundTruth ??
        [];

      const predictionMask = featureProps?.prediction?.mask ?? [];
      const shape = featureProps?.shape ?? spread?.shape ?? {};
      const meanProbability = featureProps?.summary?.meanProbability ?? null;
      const maxProbability = featureProps?.summary?.maxProbability ?? null;
      const minProbability = featureProps?.summary?.minProbability ?? null;
      const positivePixels = featureProps?.summary?.positivePixels ?? null;
      const groundTruthPixels =
        featureProps?.summary?.groundTruthPixels ?? null;
      const fireCenter =
        spread?.fire ??
        spread?.fireMeta ??
        spread?.metadata?.fire ??
        payload?.fire ??
        payload?.metadata?.fire ??
        {};

      const { precision, recall, f1 } = computeMetrics(
        predictionMask,
        groundTruth
      );

      const rows = shape.height ?? fireMeta.height ?? prediction.length ?? 0; // number of rows
      const cols =
        shape.width ??
        fireMeta.width ??
        prediction[0]?.length ??
        prediction?.[0]?.length ??
        0; // number of columns

      const centerLat = fireCenter?.latitude ?? fireMeta.latitude;
      const centerLong = fireCenter?.longitude ?? fireMeta.longitude;

      const [predictionFeatures, predictionPositive] = buildCoordinatesArray(
        rows,
        cols,
        centerLat,
        centerLong,
        prediction
      );
      const [groundTruthFeatures, groundTruthPositive] = buildCoordinatesArray(
        rows,
        cols,
        centerLat,
        centerLong,
        groundTruth
      );

      if (predictionPositive === 0 && groundTruthPositive === 0) {
        throw new Error("No positive fire spread predictions found");
      }

      const nextLayerData = {
        prediction: {
          type: "FeatureCollection",
          features: predictionFeatures,
        },
        groundTruth: {
          type: "FeatureCollection",
          features: groundTruthFeatures,
        },
      };

      setLayerData(nextLayerData);
      setActiveLayerIndex(0);

      // Calculate statistics including confidence interval
      if (meanProbability !== null) {
        // Calculate standard error and 95% confidence interval
        const totalPixels = rows * cols;
        const variance = meanProbability * (1 - meanProbability);
        const standardError = Math.sqrt(variance / totalPixels);
        const marginOfError = 1.96 * standardError; // 95% CI

        setStatistics({
          meanProbability,
          confidenceInterval: {
            lower: Math.max(0, meanProbability - marginOfError),
            upper: Math.min(1, meanProbability + marginOfError),
          },
          maxProbability,
          minProbability,
          positivePixels,
          groundTruthPixels,
          totalPixels,
          precision,
          recall,
          f1,
        });
      } else {
        setStatistics(null);
      }

      const focusSource = predictionFeatures.length
        ? predictionFeatures
        : groundTruthFeatures;
      const positive = focusSource.filter(
        (feature) =>
          (feature?.properties?.mag ?? 0) > 0 &&
          feature?.geometry?.coordinates?.length === 2
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
      } else if (centerLong && centerLat) {
        coords = [centerLong, centerLat];
      }

      if (coords && mapRef.current) {
        mapRef.current.flyTo({
          center: coords,
          zoom: 9,
          speed: 0.8,
          curve: 1.4,
        });
      }
    } catch (error) {
      if (pendingSpreadRequestRef.current !== requestId) return;
      setLayerData(null);
      setData(null);
      setStatistics(null);
      setSpreadError(error.message ?? "Unable to load fire spread");
      console.error("findSpread error:", error);
    } finally {
      if (pendingSpreadRequestRef.current === requestId) {
        setSpreadLoading(false);
      }
    }
  };

  const handleWildfireChange = (event) => {
    const newId = event.target.value;

    if (newId === "none") {
      setSelectedId(null);
      setLayerData(null);
      setData(null);
      setStatistics(null);
      setSpreadError(null);
      return;
    }

    setSelectedId(newId);
    fetchSpreadForFire(newId);
  };

  const prevDay = () => {
    if (dayKeys.length === 0) return;
    setActiveLayerIndex(
      (index) => (index - 1 + dayKeys.length) % dayKeys.length
    );
  };

  const nextDay = () => {
    if (dayKeys.length === 0) return;
    setActiveLayerIndex((index) => (index + 1) % dayKeys.length);
  };

  return (
    <>
      <div className="overlay-top-right app-overlay controls-column">
        <div className="control">
          <label htmlFor="wildfire" className="dropdown-label">
            Wildfire (showing {PAGE_SIZE} of {MAX_FIRE_COUNT})
          </label>
          <select
            id="wildfire"
            className="dropdown-select"
            value={selectedId ?? "none"}
            onChange={handleWildfireChange}
            disabled={catalogLoading || catalog.length === 0}
          >
            <option value="none" disabled>
              — Select a wildfire —
            </option>
            {visibleCatalog.map((fire) => (
              <option key={fire.fireId} value={fire.fireId}>
                {fire.locationName ??
                  (typeof fire.latitude === "number" &&
                  typeof fire.longitude === "number"
                    ? `${fire.latitude.toFixed(2)}, ${fire.longitude.toFixed(
                        2
                      )}`
                    : fire.fireId)}
              </option>
            ))}
          </select>

          <div className="pagination-controls">
            <button
              type="button"
              className="arrow-btn"
              onClick={handleCatalogPrev}
              disabled={catalogLoading || catalogPage === 0}
            >
              ‹
            </button>
            <span className="day-label">
              Page {Math.min(catalogPage + 1, totalPages)} / {totalPages}
            </span>
            <button
              type="button"
              className="arrow-btn"
              onClick={handleCatalogNext}
              disabled={catalogLoading || catalogPage >= totalPages - 1}
            >
              ›
            </button>
          </div>

          {catalogLoading && (
            <span className="status-text">Loading catalog…</span>
          )}
          {catalogError && (
            <span className="status-text error">{catalogError}</span>
          )}
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
                ? `View: ${
                    LAYER_LABELS[dayKeys[activeLayerIndex]] ??
                    dayKeys[activeLayerIndex]
                  }`
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

        {spreadLoading && (
          <span className="status-text">Loading fire spread…</span>
        )}
        {spreadError && (
          <span className="status-text error">{spreadError}</span>
        )}

        {statistics && (
          <div className="stats-box">
            <div className="stats-title">Prediction Statistics</div>
            <div className="stats-row">
              <span className="stats-label">Mean Probability:</span>
              <span className="stats-value">
                {(statistics.meanProbability * 100).toFixed(2)}%
              </span>
            </div>
            <div className="stats-row">
              <span className="stats-label">95% CI:</span>
              <span className="stats-value">
                [{(statistics.confidenceInterval.lower * 100).toFixed(2)}%,{" "}
                {(statistics.confidenceInterval.upper * 100).toFixed(2)}%]
              </span>
            </div>
            {statistics.maxProbability !== null && (
              <div className="stats-row">
                <span className="stats-label">Max Probability:</span>
                <span className="stats-value">
                  {(statistics.maxProbability * 100).toFixed(2)}%
                </span>
              </div>
            )}
            {statistics.positivePixels !== null && (
              <div className="stats-row">
                <span className="stats-label">Positive Pixels:</span>
                <span className="stats-value">
                  {statistics.positivePixels} / {statistics.totalPixels}
                </span>
              </div>
            )}
            <div className="stats-row">
              <span className="stats-label">Precision:</span>
              <span className="stats-value">
                {(statistics.precision * 100).toFixed(2)}%
              </span>
            </div>
            <div className="stats-row">
              <span className="stats-label">Recall:</span>
              <span className="stats-value">
                {(statistics.recall * 100).toFixed(2)}%
              </span>
            </div>
            <div className="stats-row">
              <span className="stats-label">F1 Score:</span>
              <span className="stats-value">
                {(statistics.f1 * 100).toFixed(2)}%
              </span>
            </div>
          </div>
        )}
      </div>

      <Map
        ref={mapRef}
        initialViewState={INITIAL_VIEW}
        mapboxAccessToken={token}
        mapStyle="mapbox://styles/mapbox/dark-v10"
        className="map-container"
        style={{ position: "fixed", inset: 0, width: "100vw", height: "100vh" }}
      >
        {selectedId && data && (
          <Source id="wildfires" type="geojson" data={data}>
            <Layer {...heatmapLayer} />
          </Source>
        )}
      </Map>
    </>
  );
}
