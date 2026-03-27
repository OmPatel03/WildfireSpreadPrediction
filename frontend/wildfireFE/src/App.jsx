import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import FilterBar from "./components/FilterBar";
import EnvironmentPanel from "./components/EnvironmentPanel";
import IncidentsPanel from "./components/IncidentsPanel";
import MapHud from "./components/MapHud";
import MapView from "./components/MapView";
import ModelInputsPanel from "./components/ModelInputsPanel";
import TimelineDock from "./components/TimelineDock";
import {
  fetchGoodPredictions,
  fetchLayers,
  fetchOverview,
  fetchTimeline,
  fetchYears,
} from "./util/api.js";
import { annotateCatalogWithLocations } from "./util/geocode.js";

const DEFAULT_YEAR = 2021;
const DEFAULT_THRESHOLD = 0.9;
const DEFAULT_CATALOG_LIMIT = 100;
const PAGE_SIZE = 8;
const THREE_D_BEARING = -20;
const THREE_D_PITCH = 55;
const DEFAULT_OSM_MAP_STYLE = "terrain";
const DEFAULT_OSM_PROJECTION = "mercator";
const FIRE_DETAIL_MAX_ZOOM = 12;
const FIRE_LOADING_EXTENT_MAX_ZOOM = 8.3;
const INITIAL_VIEW = {
  longitude: -100,
  latitude: 40,
  zoom: 3.4,
  bearing: 0,
  pitch: 0,
};
const OSM_MAP_STYLES = [
  { value: "standard", label: "Standard" },
  { value: "terrain", label: "Terrain" },
];
const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN ?? "";
const FALLBACK_YEAR_OPTIONS = [DEFAULT_YEAR];
const DEFAULT_ENVIRONMENT_SCALES = {
  viirs_m11: 1,
  viirs_i2: 1,
  ndvi: 1,
  evi2: 1,
  precip: 1,
  wind_speed: 1,
};
const DEFAULT_LAYER_VISIBILITY = {
  overview: true,
  predictionHeatmap: true,
  predictionPolygons: false,
  groundTruthHeatmap: true,
  differenceHeatmap: false,
  extent: true,
  origin: true,
};

function useDebouncedValue(value, delayMs) {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const timeoutId = window.setTimeout(() => {
      setDebouncedValue(value);
    }, delayMs);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [delayMs, value]);

  return debouncedValue;
}

function buildOverviewGeojson(fires) {
  return {
    type: "FeatureCollection",
    features: fires
      .filter(
        (fire) =>
          Number.isFinite(fire?.longitude) && Number.isFinite(fire?.latitude),
      )
      .map((fire) => ({
        type: "Feature",
        geometry: {
          type: "Point",
          coordinates: [fire.longitude, fire.latitude],
        },
        properties: {
          fireId: fire.fireId,
          year: fire.year,
          samples: fire.samples,
          label: fire.locationName ?? fire.fireId,
        },
      })),
  };
}

export default function App() {
  const mapRef = useRef(null);
  const timelineCacheRef = useRef(new Map());
  const layersCacheRef = useRef(new Map());

  const [year, setYear] = useState(DEFAULT_YEAR);
  const [yearOptions, setYearOptions] = useState(FALLBACK_YEAR_OPTIONS);
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);
  const [catalogLimit, setCatalogLimit] = useState(DEFAULT_CATALOG_LIMIT);
  const [searchTerm, setSearchTerm] = useState("");
  const [catalogPage, setCatalogPage] = useState(0);
  const [selectedId, setSelectedId] = useState(null);
  const [sampleIndex, setSampleIndex] = useState(null);
  const [viewMode, setViewMode] = useState("2d");
  const [incidentsView, setIncidentsView] = useState("catalog");
  const [osmMapStyle, setOsmMapStyle] = useState(DEFAULT_OSM_MAP_STYLE);
  const [osmProjection, setOsmProjection] = useState(DEFAULT_OSM_PROJECTION);
  const [modelInputsOpen, setModelInputsOpen] = useState(false);
  const [environmentOpen, setEnvironmentOpen] = useState(false);
  const [collapsedPanels, setCollapsedPanels] = useState({
    incidents: false,
  });
  const [environmentScales, setEnvironmentScales] = useState(DEFAULT_ENVIRONMENT_SCALES);
  const [layerVisibility, setLayerVisibility] = useState(
    DEFAULT_LAYER_VISIBILITY,
  );

  const [catalog, setCatalog] = useState([]);
  const [catalogLoading, setCatalogLoading] = useState(false);
  const [catalogError, setCatalogError] = useState(null);

  const [timeline, setTimeline] = useState(null);
  const [timelineLoading, setTimelineLoading] = useState(false);
  const [timelineError, setTimelineError] = useState(null);

  const [layersResponse, setLayersResponse] = useState(null);
  const [layersLoading, setLayersLoading] = useState(false);
  const [layersError, setLayersError] = useState(null);
  const [selectedFireLoadingId, setSelectedFireLoadingId] = useState(null);

  const debouncedThreshold = useDebouncedValue(threshold, 350);
  const debouncedSampleIndex = useDebouncedValue(sampleIndex, 200);
  const debouncedEnvironmentScales = useDebouncedValue(environmentScales, 300);

  const handleYearChange = (nextYear) => {
    if (nextYear === year) {
      return;
    }

    setCatalog([]);
    setCatalogError(null);
    setSelectedId(null);
    setTimeline(null);
    setTimelineLoading(false);
    setSampleIndex(null);
    setTimelineError(null);
    setLayersResponse(null);
    setLayersLoading(false);
    setLayersError(null);
    setSelectedFireLoadingId(null);
    setIncidentsView("catalog");
    setYear(nextYear);
  };

  useEffect(() => {
    let ignore = false;
    const controller = new AbortController();

    async function loadYears() {
      try {
        const years = await fetchYears({ signal: controller.signal });
        if (ignore || !Array.isArray(years) || years.length === 0) return;
        setYearOptions(years);
        setYear((currentYear) =>
          years.includes(currentYear) ? currentYear : years[years.length - 1],
        );
      } catch (error) {
        if (!ignore && error?.name !== "AbortError") {
          console.warn("Unable to load available years:", error);
        }
      }
    }

    loadYears();
    return () => {
      ignore = true;
      controller.abort();
    };
  }, []);

  useEffect(() => {
    let ignore = false;
    const geocodeController = new AbortController();
    const overviewController = new AbortController();

    async function loadOverview() {
      setCatalogLoading(true);
      setCatalogError(null);

      try {
        const overviewLimit = year === 2021 ? 1000 : catalogLimit;
        const rows = await fetchOverview({
          year,
          limit: overviewLimit,
          offset: 0,
          signal: overviewController.signal,
        });
        let filteredRows = rows;

        if (year === 2021) {
          try {
            const goodPredictions = await fetchGoodPredictions({
              year,
              signal: overviewController.signal,
            });
            const allowedFireIds = new Set(
              goodPredictions
                .map((entry) => entry.fireId)
                .filter(Boolean),
            );
            filteredRows = rows.filter((fire) => allowedFireIds.has(fire.fireId));
          } catch (error) {
            if (error?.name === "AbortError") {
              throw error;
            }
            console.warn("Unable to load good-prediction whitelist for 2021:", error);
          }
        }

        if (year === 2021) {
          filteredRows = filteredRows.slice(0, catalogLimit);
        }

        if (!ignore) {
          setCatalog(filteredRows);
          setCatalogPage(0);
          setSelectedId((currentId) =>
            currentId && !filteredRows.some((fire) => fire.fireId === currentId)
              ? null
              : currentId,
          );
        }

        try {
          const enriched = await annotateCatalogWithLocations(
            filteredRows,
            MAPBOX_TOKEN,
            geocodeController.signal,
          );

          if (!ignore) {
            setCatalog(enriched);
          }
        } catch (error) {
          if (error?.name !== "AbortError") {
            console.warn("Location lookup failed:", error);
          }
        }
      } catch (error) {
        if (!ignore && error?.name !== "AbortError") {
          setCatalogError(error.message ?? "Unable to load overview");
          setCatalog([]);
        }
      } finally {
        if (!ignore) {
          setCatalogLoading(false);
        }
      }
    }

    loadOverview();
    return () => {
      ignore = true;
      overviewController.abort();
      geocodeController.abort();
    };
  }, [catalogLimit, year]);

  const filteredCatalog = useMemo(() => {
    const query = searchTerm.trim().toLowerCase();
    if (!query) return catalog;

    return catalog.filter((fire) => {
      const haystack = [fire.fireId, fire.locationName, fire.latestTargetDate]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return haystack.includes(query);
    });
  }, [catalog, searchTerm]);

  const totalPages = Math.max(1, Math.ceil(filteredCatalog.length / PAGE_SIZE));

  useEffect(() => {
    setCatalogPage((page) => Math.min(page, totalPages - 1));
  }, [totalPages]);

  useEffect(() => {
    if (!selectedId) {
      setIncidentsView("catalog");
    }
  }, [selectedId]);

  const visibleCatalog = useMemo(() => {
    const start = catalogPage * PAGE_SIZE;
    return filteredCatalog.slice(start, start + PAGE_SIZE);
  }, [catalogPage, filteredCatalog]);

  const selectedFire = useMemo(
    () => catalog.find((fire) => fire.fireId === selectedId) ?? null,
    [catalog, selectedId],
  );
  const selectedFireYear = selectedFire?.year ?? null;
  const mapInitialViewState = useMemo(
    () =>
      viewMode === "3d"
        ? {
            ...INITIAL_VIEW,
            bearing: THREE_D_BEARING,
            pitch: THREE_D_PITCH,
          }
        : INITIAL_VIEW,
    [viewMode],
  );

  const overviewData = useMemo(
    () => buildOverviewGeojson(filteredCatalog),
    [filteredCatalog],
  );

  useEffect(() => {
    if (!selectedId || selectedFireYear !== year) {
      setTimeline(null);
      setTimelineLoading(false);
      setSampleIndex(null);
      setTimelineError(null);
      setSelectedFireLoadingId(null);
      return;
    }

    setTimeline(null);
    setSampleIndex(null);
    setTimelineError(null);

    let ignore = false;
    const controller = new AbortController();

    async function loadTimeline() {
      setTimelineLoading(true);
      setTimelineError(null);

      try {
        const timelineCacheKey = `${year}:${selectedId}`;
        const payload = timelineCacheRef.current.get(timelineCacheKey)
          ?? await fetchTimeline({
            fireId: selectedId,
            year,
            signal: controller.signal,
          });
        timelineCacheRef.current.set(timelineCacheKey, payload);
        if (ignore) return;
        setTimeline(payload);
        setSampleIndex((current) => {
          const frameIndices = payload.frames?.map((frame) => frame.sampleIndex) ?? [];
          if (
            Number.isInteger(current) &&
            frameIndices.includes(current)
          ) {
            return current;
          }
          return payload.defaultSampleIndex ?? 0;
        });
      } catch (error) {
        if (!ignore && error?.name !== "AbortError") {
          setTimeline(null);
          setSampleIndex(null);
          setTimelineError(error.message ?? "Unable to load timeline");
        }
      } finally {
        if (!ignore) {
          setTimelineLoading(false);
        }
      }
    }

    loadTimeline();
    return () => {
      ignore = true;
      controller.abort();
    };
  }, [selectedFireYear, selectedId, year]);

  useEffect(() => {
    if (!selectedId) {
      setSelectedFireLoadingId(null);
      return;
    }

    if (layersResponse || layersError || timelineError) {
      setSelectedFireLoadingId((current) =>
        current === selectedId ? null : current,
      );
    }
  }, [layersError, layersResponse, selectedId, timelineError]);

  useEffect(() => {
    if (
      !selectedId ||
      selectedFireYear !== year ||
      sampleIndex === null ||
      sampleIndex === undefined
    ) {
      setLayersResponse(null);
      setLayersLoading(false);
      setLayersError(null);
      return;
    }

    if (
      debouncedSampleIndex === null ||
      debouncedSampleIndex === undefined ||
      debouncedSampleIndex !== sampleIndex
    ) {
      return;
    }

    let ignore = false;
    const controller = new AbortController();

    async function loadLayers() {
      setLayersLoading(true);
      setLayersError(null);

      try {
        const layerCacheKey = [
          year,
          selectedId,
          debouncedSampleIndex,
          debouncedThreshold,
          JSON.stringify(debouncedEnvironmentScales),
        ].join(":");
        const payload = layersCacheRef.current.get(layerCacheKey)
          ?? await fetchLayers({
            fireId: selectedId,
            year,
            sampleIndex: debouncedSampleIndex,
            threshold: debouncedThreshold,
            environmentScales: debouncedEnvironmentScales,
            signal: controller.signal,
          });
        layersCacheRef.current.set(layerCacheKey, payload);
        if (ignore) return;
        setLayersResponse(payload);
      } catch (error) {
        if (!ignore && error?.name !== "AbortError") {
          setLayersResponse(null);
          setLayersError(error.message ?? "Unable to load fire layers");
        }
      } finally {
        if (!ignore) {
          setLayersLoading(false);
        }
      }
    }

    loadLayers();
    return () => {
      ignore = true;
      controller.abort();
    };
  }, [
    debouncedEnvironmentScales,
    debouncedSampleIndex,
    debouncedThreshold,
    sampleIndex,
    selectedFireYear,
    selectedId,
    year,
  ]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const horizontalPadding = Math.min(
      Math.max(Math.round(window.innerWidth * 0.16), 140),
      240,
    );
    const topPadding = Math.min(
      Math.max(Math.round(window.innerHeight * 0.1), 64),
      110,
    );
    const bottomOverlayPadding = window.innerWidth <= 1100 ? 86 : 108;
    const cameraOptions = viewMode === "3d"
      ? {
          bearing: THREE_D_BEARING,
          pitch: THREE_D_PITCH,
        }
      : {
          bearing: 0,
          pitch: 0,
        };
    const fitPadding = {
      top: topPadding,
      right: horizontalPadding,
      bottom: bottomOverlayPadding,
      left: horizontalPadding,
    };

    if (!selectedFire) {
      map.resize?.();
      map.easeTo?.({
        center: [INITIAL_VIEW.longitude, INITIAL_VIEW.latitude],
        zoom: INITIAL_VIEW.zoom,
        duration: 900,
        ...cameraOptions,
      });
      return;
    }

    const bbox = selectedFire.bbox;
    const activeLayers = layersResponse?.layers ?? null;

    map.resize?.();

    const candidateFeatures = [
      ...(activeLayers?.predictionHeatmap?.features ?? []),
      ...(activeLayers?.groundTruthHeatmap?.features ?? []),
    ];

    if (candidateFeatures.length > 0) {
      let minLat = Infinity;
      let minLon = Infinity;
      let maxLat = -Infinity;
      let maxLon = -Infinity;

      candidateFeatures.forEach((feature) => {
        const [lon, lat] = feature?.geometry?.coordinates ?? [];
        if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;
        minLat = Math.min(minLat, lat);
        minLon = Math.min(minLon, lon);
        maxLat = Math.max(maxLat, lat);
        maxLon = Math.max(maxLon, lon);
      });

      if (
        Number.isFinite(minLat) &&
        Number.isFinite(minLon) &&
        Number.isFinite(maxLat) &&
        Number.isFinite(maxLon)
      ) {
        const latPad = Math.max((maxLat - minLat) * 0.35, 0.01);
        const lonPad = Math.max((maxLon - minLon) * 0.35, 0.01);

        map.fitBounds(
          [
            [minLon - lonPad, minLat - latPad],
            [maxLon + lonPad, maxLat + latPad],
          ],
          {
            padding: fitPadding,
            maxZoom: FIRE_DETAIL_MAX_ZOOM,
            duration: viewMode === "3d" ? 1000 : 850,
            ...cameraOptions,
          },
        );
        return;
      }
    }

    if (
      bbox &&
      Number.isFinite(bbox.minLon) &&
      Number.isFinite(bbox.minLat) &&
      Number.isFinite(bbox.maxLon) &&
      Number.isFinite(bbox.maxLat)
    ) {
      map.fitBounds(
        [
          [bbox.minLon, bbox.minLat],
          [bbox.maxLon, bbox.maxLat],
        ],
        {
          padding: fitPadding,
          maxZoom: FIRE_LOADING_EXTENT_MAX_ZOOM,
          duration: viewMode === "3d" ? 1000 : 850,
          ...cameraOptions,
        },
      );
      return;
    }

    map.flyTo({
      center: [selectedFire.longitude, selectedFire.latitude],
      zoom: Math.min(8.2, FIRE_DETAIL_MAX_ZOOM),
      duration: viewMode === "3d" ? 1000 : 850,
      ...cameraOptions,
    });
  }, [layersResponse, selectedFire, viewMode]);

  const currentFrame = useMemo(() => {
    if (!timeline?.frames?.length || sampleIndex === null || sampleIndex === undefined) {
      return null;
    }
    return (
      timeline.frames.find((frame) => frame.sampleIndex === sampleIndex) ??
      timeline.frames[sampleIndex] ??
      null
    );
  }, [sampleIndex, timeline]);

  const currentFramePosition = useMemo(() => {
    if (!timeline?.frames?.length || sampleIndex === null || sampleIndex === undefined) {
      return 0;
    }
    const frameIndex = timeline.frames.findIndex(
      (frame) => frame.sampleIndex === sampleIndex,
    );
    return frameIndex >= 0 ? frameIndex : 0;
  }, [sampleIndex, timeline]);

  const handleToggleLayer = (layerKey) => {
    setLayerVisibility((current) => ({
      ...current,
      [layerKey]: !current[layerKey],
    }));
  };

  const handleSelectFire = (fireId) => {
    setSelectedFireLoadingId(fireId);
    setSelectedId(fireId);
    setIncidentsView("detail");
    setSampleIndex(null);
    setTimeline(null);
    setTimelineError(null);
    setLayersResponse(null);
    setLayersError(null);
  };

  const handleEnvironmentScaleChange = (key, value) => {
    setEnvironmentScales((current) => ({
      ...current,
      [key]: value,
    }));
  };

  const handleResetEnvironment = () => {
    setEnvironmentScales(DEFAULT_ENVIRONMENT_SCALES);
  };

  const handleResetApp = () => {
    const resetYear = yearOptions.includes(DEFAULT_YEAR)
      ? DEFAULT_YEAR
      : yearOptions[yearOptions.length - 1] ?? DEFAULT_YEAR;

    setYear(resetYear);
    setThreshold(DEFAULT_THRESHOLD);
    setCatalogLimit(DEFAULT_CATALOG_LIMIT);
    setSearchTerm("");
    setCatalogPage(0);
    setSelectedId(null);
    setSampleIndex(null);
    setIncidentsView("catalog");
    setOsmMapStyle(DEFAULT_OSM_MAP_STYLE);
    setModelInputsOpen(false);
    setEnvironmentOpen(false);
    setCollapsedPanels({
      incidents: false,
    });
    setEnvironmentScales(DEFAULT_ENVIRONMENT_SCALES);
    setLayerVisibility(DEFAULT_LAYER_VISIBILITY);
    setTimeline(null);
    setTimelineLoading(false);
    setTimelineError(null);
    setLayersResponse(null);
    setLayersLoading(false);
    setLayersError(null);
    setSelectedFireLoadingId(null);
  };

  const handleTogglePanelCollapse = (panelKey) => {
    setCollapsedPanels((current) => ({
      ...current,
      [panelKey]: !current[panelKey],
    }));
  };

  const handleOpenModelInputs = () => {
    setModelInputsOpen((open) => !open);
  };

  const handleOpenEnvironment = () => {
    setEnvironmentOpen((open) => !open);
  };

  const beginFrameTransition = (nextSampleIndex) => {
    if (
      !selectedId ||
      nextSampleIndex === null ||
      nextSampleIndex === undefined ||
      nextSampleIndex === sampleIndex
    ) {
      return;
    }

    setSelectedFireLoadingId(selectedId);
    setLayersLoading(true);
    setLayersResponse(null);
    setLayersError(null);
    setSampleIndex(nextSampleIndex);
  };

  const handleTimelineStep = (step) => {
    const frames = timeline?.frames ?? [];
    if (!frames.length) return;

    const nextPosition = Math.min(
      Math.max(currentFramePosition + step, 0),
      frames.length - 1,
    );
    beginFrameTransition(frames[nextPosition]?.sampleIndex ?? null);
  };

  const handleTimelineChange = (position) => {
    const frames = timeline?.frames ?? [];
    const nextFrame = frames[position];
    if (nextFrame) {
      beginFrameTransition(nextFrame.sampleIndex);
    }
  };

  const fireLayers = layersResponse?.layers ?? null;
  const selectedFireLoading =
    Boolean(selectedId) && selectedFireLoadingId === selectedId;
  const fireSummary = layersResponse?.summary ?? null;
  const insightFire = useMemo(() => {
    if (selectedFire && layersResponse?.fire) {
      return { ...layersResponse.fire, ...selectedFire };
    }
    return layersResponse?.fire ?? selectedFire;
  }, [layersResponse, selectedFire]);

  return (
    <div className="app-shell">
      <MapView
        mapRef={mapRef}
        initialViewState={mapInitialViewState}
        viewMode={viewMode}
        osmMapStyle={osmMapStyle}
        osmProjection={osmProjection}
        onOverviewSelect={handleSelectFire}
        overviewData={overviewData}
        selectedId={selectedId}
        selectedFire={selectedFire}
        fireLayers={fireLayers}
        selectedFireLoading={selectedFireLoading}
        fireDetailMaxZoom={FIRE_DETAIL_MAX_ZOOM}
        layerVisibility={layerVisibility}
      />
      <div className="map-atmosphere" aria-hidden="true">
        <div className="map-glow map-glow-left" />
        <div className="map-glow map-glow-right" />
        <div className="map-vignette" />
      </div>

      <MapHud
        selectedFire={selectedFire}
        currentFrame={currentFrame}
        layerVisibility={layerVisibility}
        layersLoading={layersLoading}
        layersError={layersError}
        timelineLoading={timelineLoading}
      />

      <div className="top-controls-layout app-overlay">
        <FilterBar
          year={year}
          yearOptions={yearOptions}
          onYearChange={handleYearChange}
          threshold={threshold}
          onThresholdChange={setThreshold}
          catalogLimit={catalogLimit}
          onCatalogLimitChange={setCatalogLimit}
          osmMapStyle={osmMapStyle}
          osmMapStyles={OSM_MAP_STYLES}
          onOsmMapStyleChange={setOsmMapStyle}
          viewMode={viewMode}
          layerVisibility={layerVisibility}
          onToggleLayer={handleToggleLayer}
        />

        <div className="control-actions-stack">
          <div className="toolbar-actions">
            <button
              type="button"
              className="control-button control-button-reset"
              onClick={handleResetApp}
              disabled={!selectedFire}
            >
              Reset view
            </button>

            <button
              type="button"
              className={(modelInputsOpen ? "control-button active" : "control-button") + " tooltip-anchor"}
              data-tooltip="Open the panel with the model features for the selected frame."
              onClick={handleOpenModelInputs}
            >
              Model inputs
            </button>

            <button
              type="button"
              className={(environmentOpen ? "control-button active" : "control-button") + " tooltip-anchor"}
              data-tooltip="Open controls for adjusting environmental factors."
              onClick={handleOpenEnvironment}
            >
              Environment
            </button>

            <div
              className="segmented-control"
              style={{ "--active-segment-index": viewMode === "3d" ? 1 : 0 }}
            >
              <button
                type="button"
                className={"tooltip-anchor" + (viewMode === "2d" ? " active" : "")}
                data-tooltip="Use the standard top-down map."
                onClick={() => setViewMode("2d")}
              >
                2D
              </button>
              <button
                type="button"
                className={"tooltip-anchor" + (viewMode === "3d" ? " active" : "")}
                data-tooltip="Use the pitched 3D map."
                onClick={() => setViewMode("3d")}
              >
                3D
              </button>
            </div>

            {viewMode === "3d" ? (
              <div className="action-stack-segment-group">
                <div
                  className="segmented-control"
                  role="group"
                  aria-label="OSM 3D projection"
                  style={{ "--active-segment-index": osmProjection === "globe" ? 1 : 0 }}
                >
                  <button
                    type="button"
                    className={"tooltip-anchor" + (osmProjection === "mercator" ? " active" : "")}
                    data-tooltip="Show the 3D map in the standard flat mercator projection."
                    onClick={() => setOsmProjection("mercator")}
                  >
                    Flat
                  </button>
                  <button
                    type="button"
                    className={"tooltip-anchor tooltip-align-right" + (osmProjection === "globe" ? " active" : "")}
                    data-tooltip="Wrap the 3D map onto a globe projection."
                    onClick={() => setOsmProjection("globe")}
                  >
                    Globe
                  </button>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      </div>

      <ModelInputsPanel
        isOpen={modelInputsOpen}
        onClose={() => setModelInputsOpen(false)}
        selectedFire={selectedFire}
        currentFrame={currentFrame}
        loading={layersLoading}
        error={layersError}
        modelInputs={layersResponse?.layers?.modelInputs}
      />

      <EnvironmentPanel
        isOpen={environmentOpen}
        onClose={() => setEnvironmentOpen(false)}
        selectedFire={selectedFire}
        currentFrame={currentFrame}
        scales={environmentScales}
        onScaleChange={handleEnvironmentScaleChange}
        onReset={handleResetEnvironment}
      />

      <IncidentsPanel
        fires={visibleCatalog}
        totalCount={filteredCatalog.length}
        selectedId={selectedId}
        loading={catalogLoading}
        error={catalogError}
        searchTerm={searchTerm}
        onSearchChange={setSearchTerm}
        page={catalogPage}
        totalPages={totalPages}
        onPrevPage={() => setCatalogPage((page) => Math.max(page - 1, 0))}
        onNextPage={() => setCatalogPage((page) => Math.min(page + 1, totalPages - 1))}
        onSelectFire={handleSelectFire}
        collapsed={collapsedPanels.incidents}
        onToggleCollapse={() => handleTogglePanelCollapse("incidents")}
        view={incidentsView}
        onBackToCatalog={() => setIncidentsView("catalog")}
        fire={insightFire}
        summary={fireSummary}
        frame={currentFrame}
        timelineLoading={timelineLoading}
        timelineError={timelineError}
        layersLoading={layersLoading}
        layerError={layersError}
      />

      {selectedFire ? (
        <TimelineDock
          timeline={timeline}
          currentFrame={currentFrame}
          framePosition={currentFramePosition}
          onChangePosition={handleTimelineChange}
          onStep={handleTimelineStep}
          loading={timelineLoading}
          error={timelineError}
        />
      ) : null}

    </div>
  );
}
