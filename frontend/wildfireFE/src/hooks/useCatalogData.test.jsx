import { render, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import useCatalogData from "./useCatalogData";
import {
  enrichedOverviewRows,
  goodPredictions,
  overviewRows,
} from "../test/fixtures/fireData.js";
import {
  fetchGoodPredictions,
  fetchOverview,
} from "../util/api.js";
import { annotateCatalogWithLocations } from "../util/geocode.js";

vi.mock("../util/api.js", () => ({
  fetchOverview: vi.fn(),
  fetchGoodPredictions: vi.fn(),
}));

vi.mock("../util/geocode.js", () => ({
  annotateCatalogWithLocations: vi.fn(),
}));

function renderUseCatalogData(props) {
  let latestResult;

  function Harness(hookProps) {
    latestResult = useCatalogData(hookProps);
    return null;
  }

  const renderResult = render(<Harness {...props} />);
  return {
    ...renderResult,
    getLatest: () => latestResult,
  };
}

describe("useCatalogData", () => {
  beforeEach(() => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
  });

  it("returns overview rows directly for non-2021 years", async () => {
    fetchOverview.mockResolvedValueOnce([overviewRows[0]]);
    annotateCatalogWithLocations.mockImplementationOnce(async (rows) => rows);

    const harness = renderUseCatalogData({
      year: 2020,
      catalogLimit: 50,
      mapboxToken: "",
    });

    await waitFor(() => {
      expect(harness.getLatest().loading).toBe(false);
    });

    expect(fetchOverview).toHaveBeenCalledWith({
      year: 2020,
      limit: 50,
      offset: 0,
      signal: expect.any(AbortSignal),
    });
    expect(fetchGoodPredictions).not.toHaveBeenCalled();
    expect(harness.getLatest().catalog).toEqual([overviewRows[0]]);
    expect(harness.getLatest().error).toBeNull();
  });

  it("applies the 2021 good-prediction whitelist before exposing catalog data", async () => {
    fetchOverview.mockResolvedValueOnce(overviewRows);
    fetchGoodPredictions.mockResolvedValueOnce(goodPredictions);
    annotateCatalogWithLocations.mockImplementationOnce(async (rows) => rows);

    const harness = renderUseCatalogData({
      year: 2021,
      catalogLimit: 10,
      mapboxToken: "",
    });

    await waitFor(() => {
      expect(harness.getLatest().loading).toBe(false);
    });

    expect(fetchOverview).toHaveBeenCalledWith({
      year: 2021,
      limit: 1000,
      offset: 0,
      signal: expect.any(AbortSignal),
    });
    expect(fetchGoodPredictions).toHaveBeenCalledTimes(1);
    expect(harness.getLatest().catalog).toEqual([overviewRows[1]]);
  });

  it("caps the 2021 catalog to the requested limit after whitelist filtering", async () => {
    const manyRows = [
      { ...overviewRows[1], fireId: "fire-2" },
      { ...overviewRows[1], fireId: "fire-4" },
      { ...overviewRows[1], fireId: "fire-5" },
    ];
    fetchOverview.mockResolvedValueOnce(manyRows);
    fetchGoodPredictions.mockResolvedValueOnce([
      { fireId: "fire-2" },
      { fireId: "fire-4" },
      { fireId: "fire-5" },
    ]);
    annotateCatalogWithLocations.mockImplementationOnce(async (rows) => rows);

    const harness = renderUseCatalogData({
      year: 2021,
      catalogLimit: 2,
      mapboxToken: "",
    });

    await waitFor(() => {
      expect(harness.getLatest().loading).toBe(false);
    });

    expect(harness.getLatest().catalog).toEqual(manyRows.slice(0, 2));
  });

  it("applies geocode enrichment when reverse geocoding succeeds", async () => {
    fetchOverview.mockResolvedValueOnce(overviewRows);
    fetchGoodPredictions.mockResolvedValueOnce(goodPredictions);
    annotateCatalogWithLocations.mockResolvedValueOnce([
      enrichedOverviewRows[1],
    ]);

    const harness = renderUseCatalogData({
      year: 2021,
      catalogLimit: 10,
      mapboxToken: "token-123",
    });

    await waitFor(() => {
      expect(harness.getLatest().loading).toBe(false);
    });

    expect(annotateCatalogWithLocations).toHaveBeenCalledWith(
      [overviewRows[1]],
      "token-123",
      expect.any(AbortSignal),
    );
    expect(harness.getLatest().catalog).toEqual([enrichedOverviewRows[1]]);
  });

  it("surfaces an error state when overview fetch fails", async () => {
    fetchOverview.mockRejectedValueOnce(new Error("overview failed"));

    const harness = renderUseCatalogData({
      year: 2024,
      catalogLimit: 25,
      mapboxToken: "",
    });

    await waitFor(() => {
      expect(harness.getLatest().loading).toBe(false);
    });

    expect(harness.getLatest().catalog).toEqual([]);
    expect(harness.getLatest().error).toBe("overview failed");
    expect(annotateCatalogWithLocations).not.toHaveBeenCalled();
  });

  it("ignores aborted requests cleanly", async () => {
    fetchOverview.mockRejectedValueOnce(new DOMException("Aborted", "AbortError"));

    const harness = renderUseCatalogData({
      year: 2024,
      catalogLimit: 25,
      mapboxToken: "",
    });

    await waitFor(() => {
      expect(harness.getLatest().loading).toBe(false);
    });

    expect(harness.getLatest().catalog).toEqual([]);
    expect(harness.getLatest().error).toBeNull();
  });

  it("falls back to the overview rows when the 2021 whitelist request fails", async () => {
    fetchOverview.mockResolvedValueOnce(overviewRows);
    fetchGoodPredictions.mockRejectedValueOnce(new Error("whitelist unavailable"));
    annotateCatalogWithLocations.mockImplementationOnce(async (rows) => rows);

    const harness = renderUseCatalogData({
      year: 2021,
      catalogLimit: 2,
      mapboxToken: "",
    });

    await waitFor(() => {
      expect(harness.getLatest().loading).toBe(false);
    });

    expect(console.warn).toHaveBeenCalledWith(
      "Unable to load good-prediction whitelist for 2021:",
      expect.any(Error),
    );
    expect(harness.getLatest().catalog).toEqual(overviewRows.slice(0, 2));
    expect(harness.getLatest().error).toBeNull();
  });
});
