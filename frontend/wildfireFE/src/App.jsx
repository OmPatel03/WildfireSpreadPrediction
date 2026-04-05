import { useEffect, useMemo, useRef, useState } from "react";
import "./styles/app-shell.css";
import EnvironmentPanel from "./components/EnvironmentPanel";
import IncidentsPanel from "./components/IncidentsPanel";
import LandingPage from "./components/LandingPage";
import MapAtmosphere from "./components/MapAtmosphere";
import MapHud from "./components/MapHud";
import MapView from "./components/MapView";
import ModelInputsPanel from "./components/ModelInputsPanel";
import TimelineDock from "./components/TimelineDock";
import TopControls from "./components/TopControls";
import useAvailableYears from "./hooks/useAvailableYears";
import useCatalogData from "./hooks/useCatalogData";
import useDebouncedValue from "./hooks/useDebouncedValue";
import useFireLayers from "./hooks/useFireLayers";
import useFireTimeline from "./hooks/useFireTimeline";
import useMapCamera from "./hooks/useMapCamera";
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
const LANDING_SESSION_STORAGE_KEY = "wispr:entered-app";
const TRANSITION_DURATION_MS = 650;

function getInitialScreen() {
  if (typeof window === "undefined") {
    return "landing";
  }

  try {
    return window.sessionStorage.getItem(LANDING_SESSION_STORAGE_KEY) === "true"
      ? "app"
      : "landing";
  } catch {
    return "landing";
  }
}

function joinClasses(...classes) {
  return classes.filter(Boolean).join(" ");
}

function buildLocationLookupKey(fire) {
  if (!fire?.fireId) {
    return null;
  }

  return `${fire.year ?? "unknown"}:${fire.fireId}`;
}

function mergeCatalogLocationNames(catalog, locationNamesByKey) {
  return catalog.map((fire) => {
    const lookupKey = buildLocationLookupKey(fire);
    const locationName = lookupKey ? locationNamesByKey[lookupKey] : null;

    if (!locationName || fire.locationName === locationName) {
      return fire;
    }

    return {
      ...fire,
      locationName,
    };
  });
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

  const [screen, setScreen] = useState(getInitialScreen);
  const [transitionPhase, setTransitionPhase] = useState("idle");
  const [year, setYear] = useState(DEFAULT_YEAR);
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
  const [selectedFireLoadingId, setSelectedFireLoadingId] = useState(null);
  const [locationNamesByKey, setLocationNamesByKey] = useState({});
  const appDataEnabled = screen === "app";

  const {
    yearOptions,
    resolvedInitialYear,
  } = useAvailableYears({
    defaultYear: DEFAULT_YEAR,
    fallbackYearOptions: FALLBACK_YEAR_OPTIONS,
    enabled: appDataEnabled,
  });
  const {
    catalog: rawCatalog,
    loading: catalogLoading,
    error: catalogError,
  } = useCatalogData({
    year,
    catalogLimit,
    enabled: appDataEnabled,
  });
  const catalog = useMemo(
    () => mergeCatalogLocationNames(rawCatalog, locationNamesByKey),
    [locationNamesByKey, rawCatalog],
  );

  const debouncedThreshold = useDebouncedValue(threshold, 350);
  const debouncedSampleIndex = useDebouncedValue(sampleIndex, 200);
  const debouncedEnvironmentScales = useDebouncedValue(environmentScales, 300);

  useEffect(() => {
    if (resolvedInitialYear === null || resolvedInitialYear === undefined) {
      return;
    }

    setYear((currentYear) =>
      yearOptions.includes(currentYear) ? currentYear : resolvedInitialYear,
    );
  }, [resolvedInitialYear, yearOptions]);

  const handleYearChange = (nextYear) => {
    if (nextYear === year) {
      return;
    }

    setCatalogPage(0);
    setSelectedId(null);
    setSampleIndex(null);
    setSelectedFireLoadingId(null);
    setIncidentsView("catalog");
    setYear(nextYear);
  };

  useEffect(() => {
    if (!selectedId) {
      return;
    }

    if (!catalog.some((fire) => fire.fireId === selectedId)) {
      setSelectedId(null);
      setSampleIndex(null);
      setSelectedFireLoadingId(null);
    }
  }, [catalog, selectedId]);

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
  const geocodeTargets = useMemo(() => {
    const pendingTargets = [];
    const seenKeys = new Set();

    const addTarget = (fire) => {
      const lookupKey = buildLocationLookupKey(fire);

      if (
        !fire
        || !lookupKey
        || seenKeys.has(lookupKey)
        || fire.locationName
        || !Number.isFinite(fire.latitude)
        || !Number.isFinite(fire.longitude)
      ) {
        return;
      }

      seenKeys.add(lookupKey);
      pendingTargets.push(fire);
    };

    visibleCatalog.forEach(addTarget);
    addTarget(selectedFire);

    return pendingTargets;
  }, [selectedFire, visibleCatalog]);
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
    if (!appDataEnabled || !MAPBOX_TOKEN || geocodeTargets.length === 0) {
      return undefined;
    }

    let ignore = false;
    const controller = new AbortController();

    async function loadLocationNames() {
      try {
        const enrichedTargets = await annotateCatalogWithLocations(
          geocodeTargets,
          MAPBOX_TOKEN,
          controller.signal,
        );

        if (ignore) {
          return;
        }

        setLocationNamesByKey((current) => {
          let changed = false;
          const next = { ...current };

          enrichedTargets.forEach((fire) => {
            const lookupKey = buildLocationLookupKey(fire);
            if (!lookupKey || !fire.locationName || next[lookupKey] === fire.locationName) {
              return;
            }

            next[lookupKey] = fire.locationName;
            changed = true;
          });

          return changed ? next : current;
        });
      } catch (nextError) {
        if (nextError?.name !== "AbortError") {
          console.warn("Location lookup failed:", nextError);
        }
      }
    }

    loadLocationNames();

    return () => {
      ignore = true;
      controller.abort();
    };
  }, [appDataEnabled, geocodeTargets]);

  const {
    timeline,
    loading: timelineLoading,
    error: timelineError,
  } = useFireTimeline({
    fireId: selectedId,
    year,
    enabled: Boolean(selectedId && selectedFireYear === year),
  });

  useEffect(() => {
    if (!selectedId || selectedFireYear !== year) {
      setSampleIndex(null);
      setSelectedFireLoadingId(null);
      return;
    }

    if (!timeline) {
      return;
    }

    setSampleIndex((current) => {
      const frameIndices = timeline.frames?.map((frame) => frame.sampleIndex) ?? [];
      if (Number.isInteger(current) && frameIndices.includes(current)) {
        return current;
      }
      return timeline.defaultSampleIndex ?? 0;
    });
  }, [selectedFireYear, selectedId, timeline, year]);

  const {
    layersResponse,
    loading: layersLoading,
    error: layersError,
  } = useFireLayers({
    fireId: selectedId,
    year,
    sampleIndex: debouncedSampleIndex,
    threshold: debouncedThreshold,
    environmentScales: debouncedEnvironmentScales,
    enabled: Boolean(
      selectedId
        && selectedFireYear === year
        && debouncedSampleIndex !== null
        && debouncedSampleIndex !== undefined,
    ),
  });

  useMapCamera({
    mapRef,
    selectedFire,
    layersResponse,
    viewMode,
  });

  useEffect(() => {
    if (!selectedId) {
      setSelectedFireLoadingId(null);
      return;
    }

    if (timelineError || layersError) {
      setSelectedFireLoadingId((current) =>
        current === selectedId ? null : current,
      );
      return;
    }

    if (
      layersResponse?.fire?.fireId === selectedId
      && sampleIndex !== null
      && sampleIndex !== undefined
      && layersResponse.sampleIndex === sampleIndex
    ) {
      setSelectedFireLoadingId((current) =>
        current === selectedId ? null : current,
      );
    }
  }, [layersError, layersResponse, sampleIndex, selectedId, timelineError]);

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
    setOsmProjection(DEFAULT_OSM_PROJECTION);
    setModelInputsOpen(false);
    setEnvironmentOpen(false);
    setCollapsedPanels({
      incidents: false,
    });
    setEnvironmentScales(DEFAULT_ENVIRONMENT_SCALES);
    setLayerVisibility(DEFAULT_LAYER_VISIBILITY);
    setSelectedFireLoadingId(null);
  };

  const handleTogglePanelCollapse = (panelKey) => {
    setCollapsedPanels((current) => ({
      ...current,
      [panelKey]: !current[panelKey],
    }));
  };

  const beginFrameTransition = (nextSampleIndex) => {
    if (
      !selectedId
      || nextSampleIndex === null
      || nextSampleIndex === undefined
      || nextSampleIndex === sampleIndex
    ) {
      return;
    }

    setSelectedFireLoadingId(selectedId);
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

  const showLandingOverlay = screen === "landing" || transitionPhase !== "idle";
  const showAppOverlay = screen === "app" || transitionPhase !== "idle";
  const landingTransitionClass = joinClasses(
    transitionPhase === "entering-app" && "landing-screen-exit-to-app",
    transitionPhase === "entering-landing" && "landing-screen-enter-from-app",
  );
  const appOverlayTransitionClass = joinClasses(
    transitionPhase === "entering-app" && "app-overlay-transition app-overlay-enter-from-landing",
    transitionPhase === "entering-landing" && "app-overlay-transition app-overlay-exit-to-landing",
    screen === "app" && transitionPhase === "idle" && "app-overlay-settled",
  );

  useEffect(() => {
    if (transitionPhase === "idle") {
      return undefined;
    }

    const timeoutId = window.setTimeout(() => {
      if (transitionPhase === "entering-app") {
        setScreen("app");
      } else if (transitionPhase === "entering-landing") {
        setScreen("landing");
      }
      setTransitionPhase("idle");
    }, TRANSITION_DURATION_MS);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [transitionPhase]);

  const handleEnterApp = () => {
    if (transitionPhase !== "idle" || screen === "app") {
      return;
    }

    try {
      window.sessionStorage.setItem(LANDING_SESSION_STORAGE_KEY, "true");
    } catch {
      // Ignore storage failures and continue showing the app for this render.
    }

    setTransitionPhase("entering-app");
  };

  const handleReturnToIntro = () => {
    if (transitionPhase !== "idle" || screen === "landing") {
      return;
    }

    try {
      window.sessionStorage.removeItem(LANDING_SESSION_STORAGE_KEY);
    } catch {
      // Ignore storage failures and continue showing the landing screen.
    }

    setTransitionPhase("entering-landing");
  };

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

      <MapAtmosphere />

      {showLandingOverlay ? (
        <LandingPage onEnter={handleEnterApp} className={landingTransitionClass} />
      ) : null}

      {showAppOverlay ? (
        <>
          <MapHud
            cardClassName={joinClasses(
              "app-overlay-left",
              "app-overlay-left-secondary",
              appOverlayTransitionClass,
            )}
            selectedFire={selectedFire}
            layerVisibility={layerVisibility}
          />

          <TopControls
            deckClassName={joinClasses("app-overlay-left", appOverlayTransitionClass)}
            stackClassName={joinClasses("app-overlay-right", appOverlayTransitionClass)}
            onBackToIntro={handleReturnToIntro}
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
            onViewModeChange={setViewMode}
            osmProjection={osmProjection}
            onOsmProjectionChange={setOsmProjection}
            layerVisibility={layerVisibility}
            onToggleLayer={handleToggleLayer}
            selectedFire={selectedFire}
            onResetApp={handleResetApp}
            modelInputsOpen={modelInputsOpen}
            onToggleModelInputs={() => setModelInputsOpen((open) => !open)}
            environmentOpen={environmentOpen}
            onToggleEnvironment={() => setEnvironmentOpen((open) => !open)}
          />

          <ModelInputsPanel
            className={joinClasses("app-overlay-right", appOverlayTransitionClass)}
            isOpen={modelInputsOpen}
            onClose={() => setModelInputsOpen(false)}
            selectedFire={selectedFire}
            currentFrame={currentFrame}
            loading={layersLoading}
            error={layersError}
            modelInputs={layersResponse?.layers?.modelInputs}
          />

          <EnvironmentPanel
            className={joinClasses("app-overlay-right", appOverlayTransitionClass)}
            isOpen={environmentOpen}
            onClose={() => setEnvironmentOpen(false)}
            selectedFire={selectedFire}
            currentFrame={currentFrame}
            scales={environmentScales}
            onScaleChange={handleEnvironmentScaleChange}
            onReset={handleResetEnvironment}
          />

          <IncidentsPanel
            className={joinClasses("app-overlay-left", appOverlayTransitionClass)}
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
              className={joinClasses("app-overlay-right", appOverlayTransitionClass)}
              timeline={timeline}
              currentFrame={currentFrame}
              framePosition={currentFramePosition}
              onChangePosition={handleTimelineChange}
              onStep={handleTimelineStep}
              loading={timelineLoading}
              error={timelineError}
            />
          ) : null}
        </>
      ) : null}
    </div>
  );
}
