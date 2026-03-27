import { render, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import useAvailableYears from "./useAvailableYears";
import { fetchYears } from "../util/api.js";

vi.mock("../util/api.js", () => ({
  fetchYears: vi.fn(),
}));

function renderUseAvailableYears(props) {
  let latestResult;

  function Harness(hookProps) {
    latestResult = useAvailableYears(hookProps);
    return null;
  }

  const renderResult = render(<Harness {...props} />);
  return {
    ...renderResult,
    getLatest: () => latestResult,
  };
}

describe("useAvailableYears", () => {
  beforeEach(() => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
  });

  it("uses fetched years when available", async () => {
    fetchYears.mockResolvedValueOnce([2019, 2020, 2021]);
    const harness = renderUseAvailableYears({
      defaultYear: 2021,
      fallbackYearOptions: [2021],
    });

    await waitFor(() => {
      expect(harness.getLatest().loading).toBe(false);
    });

    expect(harness.getLatest().yearOptions).toEqual([2019, 2020, 2021]);
    expect(harness.getLatest().resolvedInitialYear).toBe(2021);
    expect(harness.getLatest().error).toBeNull();
  });

  it("returns the fallback-resolved year when the configured default is missing", async () => {
    fetchYears.mockResolvedValueOnce([2018, 2019, 2020]);
    const harness = renderUseAvailableYears({
      defaultYear: 2021,
      fallbackYearOptions: [2021],
    });

    await waitFor(() => {
      expect(harness.getLatest().loading).toBe(false);
    });

    expect(harness.getLatest().yearOptions).toEqual([2018, 2019, 2020]);
    expect(harness.getLatest().resolvedInitialYear).toBe(2020);
  });

  it("preserves fallback options on fetch failure", async () => {
    fetchYears.mockRejectedValueOnce(new Error("years unavailable"));
    const harness = renderUseAvailableYears({
      defaultYear: 2021,
      fallbackYearOptions: [2021, 2020],
    });

    await waitFor(() => {
      expect(harness.getLatest().loading).toBe(false);
    });

    expect(harness.getLatest().yearOptions).toEqual([2021, 2020]);
    expect(harness.getLatest().resolvedInitialYear).toBe(2021);
    expect(harness.getLatest().error).toBeInstanceOf(Error);
    expect(harness.getLatest().error.message).toBe("years unavailable");
  });

  it("keeps fallback options when the API returns an empty list", async () => {
    fetchYears.mockResolvedValueOnce([]);
    const harness = renderUseAvailableYears({
      defaultYear: 2021,
      fallbackYearOptions: [2021, 2020],
    });

    await waitFor(() => {
      expect(harness.getLatest().loading).toBe(false);
    });

    expect(harness.getLatest().yearOptions).toEqual([2021, 2020]);
    expect(harness.getLatest().resolvedInitialYear).toBe(2021);
    expect(harness.getLatest().error).toBeNull();
  });
});
