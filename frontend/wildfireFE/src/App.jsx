import { useEffect, useMemo, useRef, useState } from "react";
import "leaflet/dist/leaflet.css";
import "./App.css";
import FilterBar from "./components/FilterBar";
import EnvironmentPanel from "./components/EnvironmentPanel";
import IncidentsPanel from "./components/IncidentsPanel";
import MapHud from "./components/MapHud";
import MapView from "./components/MapView";
import ModelInputsPanel from "./components/ModelInputsPanel";
import TimelineDock from "./components/TimelineDock";
import {
  fetchBasemap,
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
const DEFAULT_MAP_PROVIDER = "gee";
const DEFAULT_OSM_PROJECTION = "mercator";
const INITIAL_VIEW = {
  longitude: -100,
  latitude: 40,
  zoom: 3.4,
  bearing: 0,
  pitch: 0,
};
const MAP_STYLES = [
  { value: "satellite", label: "Satellite" },
  { value: "outdoors", label: "Outdoors" },
  { value: "terrain", label: "Terrain" },
];
const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN ?? "";
const FALLBACK_YEAR_OPTIONS = [DEFAULT_YEAR];
const OVERVIEW_BASEMAP_CACHE_TTL_MS = 1000 * 60 * 60 * 6;
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

function getOverviewBasemapStorageKey(year, style) {
  return `wildfire-overview-basemap:${year}:${style}`;
}

function readStoredOverviewBasemap(year, style) {
  if (typeof window === "undefined") return null;

  try {
    const raw = window.localStorage.getItem(getOverviewBasemapStorageKey(year, style));
    if (!raw) return null;

    const parsed = JSON.parse(raw);
    if (!parsed?.url || parsed?.style !== style) {
      window.localStorage.removeItem(getOverviewBasemapStorageKey(year, style));
      return null;
    }

    if (!Number.isFinite(parsed?.savedAt)) {
      window.localStorage.removeItem(getOverviewBasemapStorageKey(year, style));
      return null;
    }

    if (Date.now() - parsed.savedAt > OVERVIEW_BASEMAP_CACHE_TTL_MS) {
      window.localStorage.removeItem(getOverviewBasemapStorageKey(year, style));
      return null;
    }

    return parsed;
  } catch (error) {
    console.warn("Unable to read cached overview basemap:", error);
    return null;
  }
}

function storeOverviewBasemap(year, payload) {
  if (typeof window === "undefined" || !payload?.style || !payload?.url) return;

  try {
    window.localStorage.setItem(
      getOverviewBasemapStorageKey(year, payload.style),
      JSON.stringify({
        ...payload,
        savedAt: Date.now(),
      }),
    );
  } catch (error) {
    console.warn("Unable to cache overview basemap:", error);
  }
}

function mergeOverviewBasemapTile(current, payload) {
  if (!payload?.style || !payload?.url) return current;

  return {
    ...(current ?? {}),
    [payload.style]: payload.url,
    attribution: payload.attribution,
    targetDate: payload.targetDate,
  };
}

export default function App() {
  const mapRef = useRef(null);
  const basemapCacheRef = useRef(new Map());
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
  const [mapProvider, setMapProvider] = useState(DEFAULT_MAP_PROVIDER);
  const [mapStyle, setMapStyle] = useState(MAP_STYLES[0].value);
  const [osmProjection, setOsmProjection] = useState(DEFAULT_OSM_PROJECTION);
  const [modelInputsOpen, setModelInputsOpen] = useState(false);
  const [environmentOpen, setEnvironmentOpen] = useState(false);
  const [collapsedPanels, setCollapsedPanels] = useState({
    incidents: false,
    modelInputs: false,
    environment: false,
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
  const [overviewBasemap, setOverviewBasemap] = useState(null);
  const [overviewBasemapFailed, setOverviewBasemapFailed] = useState(false);

  const debouncedThreshold = useDebouncedValue(threshold, 350);
  const debouncedSampleIndex = useDebouncedValue(sampleIndex, 200);
  const debouncedEnvironmentScales = useDebouncedValue(environmentScales, 300);

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
        const rows = await fetchOverview({
          year,
          limit: catalogLimit,
          offset: 0,
          signal: overviewController.signal,
        });
        if (!ignore) {
          setCatalog(rows);
          setCatalogPage(0);
          setSelectedId((currentId) =>
            currentId && !rows.some((fire) => fire.fireId === currentId)
              ? null
              : currentId,
          );
        }

        try {
          const enriched = await annotateCatalogWithLocations(
            rows,
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

  useEffect(() => {
    setOverviewBasemap(null);
    setOverviewBasemapFailed(false);
  }, [year]);

  useEffect(() => {
    let ignore = false;
    const controller = new AbortController();
    const cacheKey = `${year}:${mapStyle}`;
    const cachedBasemap = basemapCacheRef.current.get(cacheKey)
      ?? readStoredOverviewBasemap(year, mapStyle);

    setOverviewBasemapFailed(false);

    if (mapProvider !== "gee") {
      return () => {
        ignore = true;
        controller.abort();
      };
    }

    if (cachedBasemap) {
      basemapCacheRef.current.set(cacheKey, cachedBasemap);
      setOverviewBasemap((current) => mergeOverviewBasemapTile(current, cachedBasemap));
      return () => {
        ignore = true;
        controller.abort();
      };
    }

    async function loadBasemap() {
      try {
        const payload = await fetchBasemap({
          year,
          style: mapStyle,
          signal: controller.signal,
        });
        if (ignore) return;
        basemapCacheRef.current.set(cacheKey, payload);
        storeOverviewBasemap(year, payload);
        setOverviewBasemap((current) => mergeOverviewBasemapTile(current, payload));
      } catch (error) {
        if (!ignore && error?.name !== "AbortError") {
          console.warn("Unable to load overview basemap:", error);
          setOverviewBasemapFailed(true);
        }
      }
    }

    loadBasemap();
    return () => {
      ignore = true;
      controller.abort();
    };
  }, [mapProvider, mapStyle, year]);

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
    if (!selectedId) {
      setTimeline(null);
      setSampleIndex(null);
      setTimelineError(null);
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
  }, [selectedId, year]);

  useEffect(() => {
    if (
      !selectedId ||
      sampleIndex === null ||
      sampleIndex === undefined ||
      debouncedSampleIndex === null ||
      debouncedSampleIndex === undefined
    ) {
      setLayersResponse(null);
      setLayersError(null);
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
          mapProvider,
          JSON.stringify(debouncedEnvironmentScales),
        ].join(":");
        const payload = layersCacheRef.current.get(layerCacheKey)
          ?? await fetchLayers({
            fireId: selectedId,
            year,
            sampleIndex: debouncedSampleIndex,
            threshold: debouncedThreshold,
            basemapProvider: mapProvider,
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
  }, [debouncedEnvironmentScales, debouncedSampleIndex, debouncedThreshold, mapProvider, sampleIndex, selectedId, year]);

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

    if (!selectedFire) {
      if (viewMode === "3d") {
        map.resize?.();
        map.easeTo?.({
          center: [INITIAL_VIEW.longitude, INITIAL_VIEW.latitude],
          zoom: INITIAL_VIEW.zoom,
          bearing: THREE_D_BEARING,
          pitch: THREE_D_PITCH,
          duration: 900,
        });
      }
      return;
    }

    const bbox = selectedFire.bbox;
    const activeLayers = layersResponse?.layers ?? null;

    if (viewMode === "2d") {
      map.invalidateSize?.();
    } else {
      map.resize?.();
    }

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

        if (viewMode === "2d") {
          map.fitBounds(
            [
              [minLat - latPad, minLon - lonPad],
              [maxLat + latPad, maxLon + lonPad],
            ],
            {
              paddingTopLeft: [horizontalPadding, topPadding],
              paddingBottomRight: [horizontalPadding, bottomOverlayPadding],
              maxZoom: 13,
              animate: true,
              duration: 1,
            },
          );
        } else {
          map.fitBounds(
            [
              [minLon - lonPad, minLat - latPad],
              [maxLon + lonPad, maxLat + latPad],
            ],
            {
              padding: {
                top: topPadding,
                right: horizontalPadding,
                bottom: bottomOverlayPadding,
                left: horizontalPadding,
              },
              maxZoom: 13,
              duration: 1000,
              bearing: THREE_D_BEARING,
              pitch: THREE_D_PITCH,
            },
          );
        }
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
      if (viewMode === "2d") {
        map.fitBounds(
          [
            [bbox.minLat, bbox.minLon],
            [bbox.maxLat, bbox.maxLon],
          ],
          {
            paddingTopLeft: [horizontalPadding, topPadding],
            paddingBottomRight: [horizontalPadding, bottomOverlayPadding],
            maxZoom: 12.5,
            animate: true,
            duration: 1,
          },
        );
      } else {
        map.fitBounds(
          [
            [bbox.minLon, bbox.minLat],
            [bbox.maxLon, bbox.maxLat],
          ],
          {
            padding: {
              top: topPadding,
              right: horizontalPadding,
              bottom: bottomOverlayPadding,
              left: horizontalPadding,
            },
            maxZoom: 12.5,
            duration: 1000,
            bearing: THREE_D_BEARING,
            pitch: THREE_D_PITCH,
          },
        );
      }
      return;
    }

    if (viewMode === "2d") {
      map.flyTo([selectedFire.latitude, selectedFire.longitude], 8.2, {
        animate: true,
        duration: 1,
      });
    } else {
      map.flyTo({
        center: [selectedFire.longitude, selectedFire.latitude],
        zoom: 8.2,
        bearing: THREE_D_BEARING,
        pitch: THREE_D_PITCH,
        duration: 1000,
      });
    }
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

  const handleTogglePanelCollapse = (panelKey) => {
    setCollapsedPanels((current) => ({
      ...current,
      [panelKey]: !current[panelKey],
    }));
  };

  const handleTimelineStep = (step) => {
    const frames = timeline?.frames ?? [];
    if (!frames.length) return;

    const nextPosition = Math.min(
      Math.max(currentFramePosition + step, 0),
      frames.length - 1,
    );
    setSampleIndex(frames[nextPosition]?.sampleIndex ?? null);
  };

  const handleTimelineChange = (position) => {
    const frames = timeline?.frames ?? [];
    const nextFrame = frames[position];
    if (nextFrame) {
      setSampleIndex(nextFrame.sampleIndex);
    }
  };

  const fireLayers = layersResponse?.layers ?? null;
  const fireBasemap = layersResponse?.basemap ?? null;
  const activeBasemap = mapProvider === "gee" ? (fireBasemap ?? overviewBasemap ?? null) : null;
  const hasActiveBasemapStyle =
    mapProvider === "gee" &&
    (Boolean(fireBasemap?.[mapStyle]) || Boolean(overviewBasemap?.[mapStyle]));
  const allowFallbackBasemap =
    mapProvider === "gee" &&
    !fireBasemap &&
    !hasActiveBasemapStyle &&
    overviewBasemapFailed;
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
        mapProvider={mapProvider}
        mapStyle={mapStyle}
        osmProjection={osmProjection}
        onOverviewSelect={handleSelectFire}
        overviewData={overviewData}
        selectedId={selectedId}
        selectedFire={selectedFire}
        basemap={activeBasemap}
        allowFallbackBasemap={allowFallbackBasemap}
        fireLayers={fireLayers}
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
          onYearChange={setYear}
          threshold={threshold}
          onThresholdChange={setThreshold}
          catalogLimit={catalogLimit}
          onCatalogLimitChange={setCatalogLimit}
          mapProvider={mapProvider}
          onMapProviderChange={setMapProvider}
          mapStyle={mapStyle}
          mapStyles={MAP_STYLES}
          onMapStyleChange={setMapStyle}
          viewMode={viewMode}
          osmProjection={osmProjection}
          onOsmProjectionChange={setOsmProjection}
          layerVisibility={layerVisibility}
          onToggleLayer={handleToggleLayer}
        />

        <div className="control-actions-stack">
          <div className="toolbar-actions">
            <button
              type="button"
              className={modelInputsOpen ? "control-button active" : "control-button"}
              onClick={() => setModelInputsOpen((open) => !open)}
            >
              Model inputs
            </button>

            <button
              type="button"
              className={environmentOpen ? "control-button active" : "control-button"}
              onClick={() => setEnvironmentOpen((open) => !open)}
            >
              Environment
            </button>

            <div className="segmented-control">
              <button
                type="button"
                className={viewMode === "2d" ? "active" : ""}
                onClick={() => setViewMode("2d")}
              >
                2D
              </button>
              <button
                type="button"
                className={viewMode === "3d" ? "active" : ""}
                onClick={() => setViewMode("3d")}
              >
                3D
              </button>
            </div>
          </div>
        </div>
      </div>

      <ModelInputsPanel
        isOpen={modelInputsOpen}
        collapsed={collapsedPanels.modelInputs}
        onToggleCollapse={() => handleTogglePanelCollapse("modelInputs")}
        selectedFire={selectedFire}
        currentFrame={currentFrame}
        loading={layersLoading}
        error={layersError}
        modelInputs={layersResponse?.layers?.modelInputs}
      />

      <EnvironmentPanel
        isOpen={environmentOpen}
        collapsed={collapsedPanels.environment}
        onToggleCollapse={() => handleTogglePanelCollapse("environment")}
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
