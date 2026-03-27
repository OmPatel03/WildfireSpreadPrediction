import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  API_BASE_URL,
  fetchJson,
  fetchLayers,
  fetchOverview,
  fetchTimeline,
  fetchYears,
} from "./api.js";

function jsonResponse(body, init = {}) {
  return new Response(JSON.stringify(body), {
    headers: {
      "Content-Type": "application/json",
    },
    ...init,
  });
}

describe("api utilities", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  it("fetchJson returns parsed JSON on success", async () => {
    fetch.mockResolvedValueOnce(jsonResponse({ ok: true }, { status: 200 }));

    await expect(fetchJson("https://example.com/test")).resolves.toEqual({
      ok: true,
    });
    expect(fetch).toHaveBeenCalledWith("https://example.com/test", {
      headers: {
        Accept: "application/json",
      },
    });
  });

  it("fetchJson retries 429 responses and respects Retry-After", async () => {
    vi.useFakeTimers();
    fetch
      .mockResolvedValueOnce(
        jsonResponse(
          { detail: "Slow down" },
          {
            status: 429,
            headers: {
              "Content-Type": "application/json",
              "Retry-After": "0.5",
            },
          },
        ),
      )
      .mockResolvedValueOnce(jsonResponse({ retried: true }, { status: 200 }));

    const request = fetchJson("https://example.com/retry");
    expect(fetch).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(500);

    await expect(request).resolves.toEqual({ retried: true });
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it("aborting during retry sleep rejects with AbortError", async () => {
    vi.useFakeTimers();
    fetch.mockResolvedValueOnce(
      jsonResponse(
        { detail: "Rate limited" },
        {
          status: 429,
          headers: {
            "Content-Type": "application/json",
            "Retry-After": "10",
          },
        },
      ),
    );
    const controller = new AbortController();

    const request = fetchJson("https://example.com/retry", {
      signal: controller.signal,
    });
    controller.abort();

    await expect(request).rejects.toMatchObject({
      name: "AbortError",
    });
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it("surfaces response detail on non-retryable failures", async () => {
    fetch.mockResolvedValueOnce(
      jsonResponse(
        { detail: "Missing fire" },
        {
          status: 404,
        },
      ),
    );

    await expect(fetchJson("https://example.com/missing")).rejects.toMatchObject({
      message: "Request failed (404): Missing fire",
      status: 404,
    });
  });

  it("uses exponential backoff when Retry-After is not present", async () => {
    vi.useFakeTimers();
    fetch
      .mockResolvedValueOnce(
        jsonResponse(
          { detail: "Busy" },
          {
            status: 429,
          },
        ),
      )
      .mockResolvedValueOnce(jsonResponse({ ok: true }, { status: 200 }));

    const request = fetchJson("https://example.com/backoff");
    expect(fetch).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(249);
    expect(fetch).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(1);
    await expect(request).resolves.toEqual({ ok: true });
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it("fetchLayers maps environment scales to the correct query params", async () => {
    fetch.mockResolvedValueOnce(jsonResponse({ layers: true }, { status: 200 }));

    await fetchLayers({
      fireId: "fire-7",
      year: 2024,
      sampleIndex: 3,
      threshold: 0.7,
      environmentScales: {
        viirs_m11: 1.1,
        viirs_i2: 0.8,
        ndvi: 1.4,
        evi2: 0.6,
        precip: 1.8,
        wind_speed: 1.2,
      },
    });

    const url = new URL(String(fetch.mock.calls[0][0]));
    expect(url.pathname).toBe("/fires/fire-7/layers");
    expect(url.searchParams.get("year")).toBe("2024");
    expect(url.searchParams.get("sampleIndex")).toBe("3");
    expect(url.searchParams.get("threshold")).toBe("0.7");
    expect(url.searchParams.get("viirsM11Scale")).toBe("1.1");
    expect(url.searchParams.get("viirsI2Scale")).toBe("0.8");
    expect(url.searchParams.get("ndviScale")).toBe("1.4");
    expect(url.searchParams.get("evi2Scale")).toBe("0.6");
    expect(url.searchParams.get("precipScale")).toBe("1.8");
    expect(url.searchParams.get("windSpeedScale")).toBe("1.2");
  });

  it("fetchTimeline, fetchOverview, and fetchYears build the expected paths", async () => {
    fetch.mockImplementation(() =>
      Promise.resolve(jsonResponse([], { status: 200 })),
    );

    await fetchTimeline({ fireId: "fire-2", year: 2021 });
    await fetchOverview({ year: 2020, limit: 25, offset: 5 });
    await fetchYears();

    expect(String(fetch.mock.calls[0][0])).toBe(
      `${API_BASE_URL}/fires/fire-2/timeline?year=2021`,
    );
    expect(String(fetch.mock.calls[1][0])).toBe(
      `${API_BASE_URL}/overview?year=2020&limit=25&offset=5`,
    );
    expect(String(fetch.mock.calls[2][0])).toBe(`${API_BASE_URL}/years`);
  });
});
