import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App.jsx";
import {
  fetchGoodPredictions,
  fetchLayers,
  fetchOverview,
  fetchTimeline,
  fetchYears,
} from "./util/api.js";
import { annotateCatalogWithLocations } from "./util/geocode.js";

const PAGE_SIZE = 8;
const DEFAULT_YEAR = 2021;

vi.mock("./util/api.js", () => ({
  fetchYears: vi.fn(),
  fetchOverview: vi.fn(),
  fetchGoodPredictions: vi.fn(),
  fetchTimeline: vi.fn(),
  fetchLayers: vi.fn(),
}));

vi.mock("./util/geocode.js", () => ({
  annotateCatalogWithLocations: vi.fn(),
}));

vi.mock("./components/MapView", () => ({
  default: () => <div data-testid="map-view">Map View</div>,
}));

vi.mock("./components/MapAtmosphere", () => ({
  default: () => <div data-testid="map-atmosphere">Map Atmosphere</div>,
}));

vi.mock("./components/MapHud", () => ({
  default: () => <div data-testid="map-hud">Map HUD</div>,
}));

vi.mock("./components/ModelInputsPanel", () => ({
  default: () => <div data-testid="model-inputs-panel">Model Inputs Panel</div>,
}));

vi.mock("./components/EnvironmentPanel", () => ({
  default: () => <div data-testid="environment-panel">Environment Panel</div>,
}));

vi.mock("./components/TimelineDock", () => ({
  default: () => <div data-testid="timeline-dock">Timeline Dock</div>,
}));

function createCatalogRows(count) {
  return Array.from({ length: count }, (_, index) => {
    const id = index + 1;

    return {
      fireId: `fire-${id}`,
      year: DEFAULT_YEAR,
      latitude: 34 + (index * 0.1),
      longitude: -118 - (index * 0.1),
      latestTargetDate: `2021-09-${String((index % 28) + 1).padStart(2, "0")}`,
      firstObservationDate: "2021-08-20",
      lastObservationDate: "2021-09-20",
      samples: 3 + (index % 4),
      width: 64,
      height: 64,
      bbox: {
        minLon: -118.5 - (index * 0.1),
        minLat: 33.5 + (index * 0.1),
        maxLon: -117.5 - (index * 0.1),
        maxLat: 34.5 + (index * 0.1),
      },
    };
  });
}

function createGoodPredictions(rows) {
  return rows.map((fire) => ({ fireId: fire.fireId }));
}

function createLocationNameMap(rows) {
  return Object.fromEntries(
    rows.map((fire) => [fire.fireId, `Location ${fire.fireId.replace("fire-", "")}`]),
  );
}

function enrichRows(rows, locationNameMap) {
  return rows.map((fire) => (
    locationNameMap[fire.fireId]
      ? { ...fire, locationName: locationNameMap[fire.fireId] }
      : fire
  ));
}

function createDeferred() {
  let resolve;
  let reject;

  const promise = new Promise((nextResolve, nextReject) => {
    resolve = nextResolve;
    reject = nextReject;
  });

  return {
    promise,
    resolve,
    reject,
  };
}

function setupApiMocks(rows = createCatalogRows(10)) {
  fetchYears.mockResolvedValue([2020, DEFAULT_YEAR]);
  fetchOverview.mockResolvedValue(rows);
  fetchGoodPredictions.mockResolvedValue(createGoodPredictions(rows));
  fetchTimeline.mockImplementation(async ({ fireId, year }) => ({
    defaultSampleIndex: 0,
    frames: [
      {
        sampleIndex: 0,
        targetDate: `${year}-09-15`,
        fireId,
      },
    ],
  }));
  fetchLayers.mockImplementation(async ({ fireId, sampleIndex }) => ({
    fire: { fireId },
    sampleIndex,
    summary: null,
    layers: {
      modelInputs: null,
    },
  }));

  return rows;
}

function renderApp({ enteredApp = false } = {}) {
  window.sessionStorage.clear();

  if (enteredApp) {
    window.sessionStorage.setItem("wispr:entered-app", "true");
  }

  return render(<App />);
}

function getFireCardButton(fireId) {
  const fireIdLabel = screen.getByText(fireId);
  const button = fireIdLabel.closest("button");

  expect(button).toBeTruthy();
  return button;
}

describe("App integration: landing-page data gating", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useRealTimers();
    window.sessionStorage.clear();

    const rows = setupApiMocks();
    annotateCatalogWithLocations.mockImplementation(async (fires) => (
      enrichRows(fires, createLocationNameMap(rows))
    ));
  });

  it("does not start data requests before the user enters the app", () => {
    renderApp();

    expect(screen.getByRole("heading", { name: "WISPR" })).toBeInTheDocument();
    expect(fetchYears).not.toHaveBeenCalled();
    expect(fetchOverview).not.toHaveBeenCalled();
    expect(fetchGoodPredictions).not.toHaveBeenCalled();
    expect(annotateCatalogWithLocations).not.toHaveBeenCalled();
  });

  it("starts year and catalog requests after entering the app", async () => {
    vi.useFakeTimers();

    renderApp();
    fireEvent.click(screen.getByRole("button", { name: "Enter WISPR" }));

    expect(fetchYears).not.toHaveBeenCalled();
    expect(fetchOverview).not.toHaveBeenCalled();

    await act(async () => {
      vi.advanceTimersByTime(650);
    });

    vi.useRealTimers();

    await waitFor(() => {
      expect(fetchYears).toHaveBeenCalledWith({
        signal: expect.any(AbortSignal),
      });
      expect(fetchOverview).toHaveBeenCalledWith({
        year: DEFAULT_YEAR,
        limit: 1000,
        offset: 0,
        signal: expect.any(AbortSignal),
      });
      expect(fetchGoodPredictions).toHaveBeenCalledWith({
        year: DEFAULT_YEAR,
        signal: expect.any(AbortSignal),
      });
    });
  });

  it("geocodes only the first visible catalog page on initial app load", async () => {
    const rows = setupApiMocks(createCatalogRows(10));
    annotateCatalogWithLocations.mockImplementation(async (fires) => (
      enrichRows(fires, createLocationNameMap(rows))
    ));

    renderApp({ enteredApp: true });

    await waitFor(() => {
      expect(annotateCatalogWithLocations).toHaveBeenCalledTimes(1);
    });

    expect(
      annotateCatalogWithLocations.mock.calls[0][0].map((fire) => fire.fireId),
    ).toEqual(rows.slice(0, PAGE_SIZE).map((fire) => fire.fireId));
  });

  it("geocodes only the newly visible page when catalog pagination changes", async () => {
    const rows = setupApiMocks(createCatalogRows(10));
    annotateCatalogWithLocations.mockImplementation(async (fires) => (
      enrichRows(fires, createLocationNameMap(rows))
    ));

    renderApp({ enteredApp: true });

    await waitFor(() => {
      expect(annotateCatalogWithLocations).toHaveBeenCalledTimes(1);
    });

    fireEvent.click(screen.getByRole("button", { name: "›" }));

    await waitFor(() => {
      expect(annotateCatalogWithLocations).toHaveBeenCalledTimes(2);
    });

    expect(
      annotateCatalogWithLocations.mock.calls[1][0].map((fire) => fire.fireId),
    ).toEqual(rows.slice(PAGE_SIZE).map((fire) => fire.fireId));
  });

  it("updates the selected fire header from the fire id to the resolved location name", async () => {
    const rows = setupApiMocks(createCatalogRows(10));
    const deferredGeocode = createDeferred();
    const firstPageRows = rows.slice(0, PAGE_SIZE);
    const firstPageNames = createLocationNameMap(firstPageRows);

    annotateCatalogWithLocations.mockImplementation(() => deferredGeocode.promise);

    renderApp({ enteredApp: true });

    await waitFor(() => {
      expect(annotateCatalogWithLocations).toHaveBeenCalledTimes(1);
    });

    fireEvent.click(getFireCardButton("fire-1"));

    expect(screen.getByRole("heading", { name: "fire-1" })).toBeInTheDocument();

    await act(async () => {
      deferredGeocode.resolve(enrichRows(firstPageRows, firstPageNames));
      await deferredGeocode.promise;
    });

    await waitFor(() => {
      expect(
        screen.getByRole("heading", { name: firstPageNames["fire-1"] }),
      ).toBeInTheDocument();
    });
  });
});
