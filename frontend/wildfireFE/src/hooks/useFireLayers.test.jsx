import { render, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import useFireLayers from "./useFireLayers";
import { createLayersResponse } from "../test/fixtures/fireData.js";
import { fetchLayers } from "../util/api.js";

vi.mock("../util/api.js", () => ({
  fetchLayers: vi.fn(),
}));

function FireLayersHarness({ hookProps, onResult }) {
  const result = useFireLayers(hookProps);
  onResult(result);
  return null;
}

function renderUseFireLayers(props) {
  let latestResult;
  const renderResult = render(
    <FireLayersHarness
      hookProps={props}
      onResult={(result) => {
        latestResult = result;
      }}
    />,
  );

  return {
    ...renderResult,
    getLatest: () => latestResult,
    rerenderWith: (nextProps) =>
      renderResult.rerender(
        <FireLayersHarness
          hookProps={nextProps}
          onResult={(result) => {
            latestResult = result;
          }}
        />,
      ),
  };
}

describe("useFireLayers", () => {
  it("fetches layer data when enabled", async () => {
    const payload = createLayersResponse();
    fetchLayers.mockResolvedValueOnce(payload);
    const harness = renderUseFireLayers({
      fireId: "fire-2",
      year: 2021,
      sampleIndex: 2,
      threshold: 0.9,
      environmentScales: { ndvi: 1 },
      enabled: true,
    });

    await waitFor(() => {
      expect(harness.getLatest().loading).toBe(false);
    });

    expect(fetchLayers).toHaveBeenCalledWith({
      fireId: "fire-2",
      year: 2021,
      sampleIndex: 2,
      threshold: 0.9,
      environmentScales: { ndvi: 1 },
      signal: expect.any(AbortSignal),
    });
    expect(harness.getLatest().layersResponse).toEqual(payload);
    expect(harness.getLatest().error).toBeNull();
  });

  it("clears state when disabled", async () => {
    const payload = createLayersResponse();
    fetchLayers.mockResolvedValueOnce(payload);
    const harness = renderUseFireLayers({
      fireId: "fire-2",
      year: 2021,
      sampleIndex: 2,
      threshold: 0.9,
      environmentScales: { ndvi: 1 },
      enabled: true,
    });

    await waitFor(() => {
      expect(harness.getLatest().layersResponse).toEqual(payload);
    });

    harness.rerenderWith({
      fireId: "fire-2",
      year: 2021,
      sampleIndex: 2,
      threshold: 0.9,
      environmentScales: { ndvi: 1 },
      enabled: false,
    });

    await waitFor(() => {
      expect(harness.getLatest().layersResponse).toBeNull();
    });
    expect(harness.getLatest().error).toBeNull();
    expect(harness.getLatest().loading).toBe(false);
  });

  it("uses a distinct cache key for sampleIndex, threshold, and serialized environment scales", async () => {
    fetchLayers
      .mockResolvedValueOnce(createLayersResponse({ sampleIndex: 2 }))
      .mockResolvedValueOnce(createLayersResponse({ sampleIndex: 3 }))
      .mockResolvedValueOnce(createLayersResponse({ sampleIndex: 3 }))
      .mockResolvedValueOnce(createLayersResponse({ sampleIndex: 3 }));

    const harness = renderUseFireLayers({
      fireId: "fire-2",
      year: 2021,
      sampleIndex: 2,
      threshold: 0.9,
      environmentScales: { ndvi: 1 },
      enabled: true,
    });

    await waitFor(() => {
      expect(fetchLayers).toHaveBeenCalledTimes(1);
    });

    harness.rerenderWith({
      fireId: "fire-2",
      year: 2021,
      sampleIndex: 3,
      threshold: 0.9,
      environmentScales: { ndvi: 1 },
      enabled: true,
    });
    await waitFor(() => {
      expect(fetchLayers).toHaveBeenCalledTimes(2);
    });

    harness.rerenderWith({
      fireId: "fire-2",
      year: 2021,
      sampleIndex: 3,
      threshold: 0.7,
      environmentScales: { ndvi: 1 },
      enabled: true,
    });
    await waitFor(() => {
      expect(fetchLayers).toHaveBeenCalledTimes(3);
    });

    harness.rerenderWith({
      fireId: "fire-2",
      year: 2021,
      sampleIndex: 3,
      threshold: 0.7,
      environmentScales: { ndvi: 1.2 },
      enabled: true,
    });
    await waitFor(() => {
      expect(fetchLayers).toHaveBeenCalledTimes(4);
    });
  });

  it("does not refetch when inputs are unchanged", async () => {
    const payload = createLayersResponse();
    fetchLayers.mockResolvedValueOnce(payload);
    const harness = renderUseFireLayers({
      fireId: "fire-2",
      year: 2021,
      sampleIndex: 2,
      threshold: 0.9,
      environmentScales: { ndvi: 1, precip: 1 },
      enabled: true,
    });

    await waitFor(() => {
      expect(fetchLayers).toHaveBeenCalledTimes(1);
    });

    harness.rerenderWith({
      fireId: "fire-2",
      year: 2021,
      sampleIndex: 2,
      threshold: 0.9,
      environmentScales: { ndvi: 1, precip: 1 },
      enabled: true,
    });

    await waitFor(() => {
      expect(harness.getLatest().layersResponse).toEqual(payload);
    });
    expect(fetchLayers).toHaveBeenCalledTimes(1);
  });
});
