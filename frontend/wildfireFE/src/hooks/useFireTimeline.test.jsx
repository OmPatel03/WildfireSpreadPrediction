import { render, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import useFireTimeline from "./useFireTimeline";
import { timelinePayload } from "../test/fixtures/fireData.js";
import { fetchTimeline } from "../util/api.js";

vi.mock("../util/api.js", () => ({
  fetchTimeline: vi.fn(),
}));

function FireTimelineHarness({ hookProps, onResult }) {
  const result = useFireTimeline(hookProps);
  onResult(result);
  return null;
}

function renderUseFireTimeline(props) {
  let latestResult;
  const renderResult = render(
    <FireTimelineHarness
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
        <FireTimelineHarness
          hookProps={nextProps}
          onResult={(result) => {
            latestResult = result;
          }}
        />,
      ),
  };
}

describe("useFireTimeline", () => {
  it("does not fetch when disabled or missing a fire id", () => {
    const disabledHarness = renderUseFireTimeline({
      fireId: "fire-2",
      year: 2021,
      enabled: false,
    });

    expect(disabledHarness.getLatest().timeline).toBeNull();
    expect(disabledHarness.getLatest().loading).toBe(false);
    expect(disabledHarness.getLatest().error).toBeNull();
    expect(fetchTimeline).not.toHaveBeenCalled();

    const missingFireHarness = renderUseFireTimeline({
      fireId: null,
      year: 2021,
      enabled: true,
    });

    expect(missingFireHarness.getLatest().timeline).toBeNull();
    expect(missingFireHarness.getLatest().loading).toBe(false);
    expect(missingFireHarness.getLatest().error).toBeNull();
    expect(fetchTimeline).not.toHaveBeenCalled();
  });

  it("fetches timeline data when enabled", async () => {
    fetchTimeline.mockResolvedValueOnce(timelinePayload);
    const harness = renderUseFireTimeline({
      fireId: "fire-2",
      year: 2021,
      enabled: true,
    });

    await waitFor(() => {
      expect(harness.getLatest().loading).toBe(false);
    });

    expect(fetchTimeline).toHaveBeenCalledWith({
      fireId: "fire-2",
      year: 2021,
      signal: expect.any(AbortSignal),
    });
    expect(harness.getLatest().timeline).toEqual(timelinePayload);
    expect(harness.getLatest().error).toBeNull();
  });

  it("clears state when disabled", async () => {
    fetchTimeline.mockResolvedValueOnce(timelinePayload);
    const harness = renderUseFireTimeline({
      fireId: "fire-2",
      year: 2021,
      enabled: true,
    });

    await waitFor(() => {
      expect(harness.getLatest().timeline).toEqual(timelinePayload);
    });

    harness.rerenderWith({
      fireId: "fire-2",
      year: 2021,
      enabled: false,
    });

    await waitFor(() => {
      expect(harness.getLatest().timeline).toBeNull();
    });
    expect(harness.getLatest().error).toBeNull();
    expect(harness.getLatest().loading).toBe(false);
  });

  it("reuses cached data for repeated fireId/year loads", async () => {
    fetchTimeline.mockResolvedValueOnce(timelinePayload);
    const harness = renderUseFireTimeline({
      fireId: "fire-2",
      year: 2021,
      enabled: true,
    });

    await waitFor(() => {
      expect(harness.getLatest().timeline).toEqual(timelinePayload);
    });

    harness.rerenderWith({
      fireId: "fire-2",
      year: 2021,
      enabled: false,
    });
    await waitFor(() => {
      expect(harness.getLatest().timeline).toBeNull();
    });

    harness.rerenderWith({
      fireId: "fire-2",
      year: 2021,
      enabled: true,
    });

    await waitFor(() => {
      expect(harness.getLatest().timeline).toEqual(timelinePayload);
    });
    expect(fetchTimeline).toHaveBeenCalledTimes(1);
  });

  it("surfaces non-abort fetch failures", async () => {
    fetchTimeline.mockRejectedValueOnce(new Error("timeline failed"));
    const harness = renderUseFireTimeline({
      fireId: "fire-2",
      year: 2021,
      enabled: true,
    });

    await waitFor(() => {
      expect(harness.getLatest().loading).toBe(false);
    });

    expect(harness.getLatest().timeline).toBeNull();
    expect(harness.getLatest().error).toBe("timeline failed");
  });
});
