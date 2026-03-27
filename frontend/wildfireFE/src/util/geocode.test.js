import { beforeEach, describe, expect, it, vi } from "vitest";

function jsonResponse(body, init = {}) {
  return new Response(JSON.stringify(body), {
    headers: {
      "Content-Type": "application/json",
    },
    ...init,
  });
}

describe("geocode utilities", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.stubGlobal("fetch", vi.fn());
    vi.spyOn(console, "warn").mockImplementation(() => {});
  });

  it("lookupLocationName returns null without a token or valid coordinates", async () => {
    const { lookupLocationName } = await import("./geocode.js");

    await expect(lookupLocationName(45, -75, "")).resolves.toBeNull();
    await expect(lookupLocationName("45", -75, "token")).resolves.toBeNull();
    await expect(lookupLocationName(45, null, "token")).resolves.toBeNull();
    expect(fetch).not.toHaveBeenCalled();
  });

  it("lookupLocationName caches successful responses for repeated coordinates", async () => {
    const { lookupLocationName } = await import("./geocode.js");
    fetch.mockResolvedValueOnce(
      jsonResponse(
        {
          features: [{ text: "Kelowna" }],
        },
        { status: 200 },
      ),
    );

    await expect(lookupLocationName(49.888, -119.496, "token")).resolves.toBe("Kelowna");
    await expect(lookupLocationName(49.888, -119.496, "token")).resolves.toBe("Kelowna");

    expect(fetch).toHaveBeenCalledTimes(1);
    expect(String(fetch.mock.calls[0][0])).toContain(
      "/-119.496,49.888.json",
    );
  });

  it("annotateCatalogWithLocations enriches fires and leaves invalid coordinates alone", async () => {
    const { annotateCatalogWithLocations } = await import("./geocode.js");
    fetch.mockResolvedValueOnce(
      jsonResponse(
        {
          features: [{ place_name: "Spokane Valley" }],
        },
        { status: 200 },
      ),
    );

    const fires = [
      { fireId: "fire-1", latitude: 47.6588, longitude: -117.426 },
      { fireId: "fire-2", latitude: null, longitude: -117.4 },
    ];

    await expect(
      annotateCatalogWithLocations(fires, "token"),
    ).resolves.toEqual([
      { fireId: "fire-1", latitude: 47.6588, longitude: -117.426, locationName: "Spokane Valley" },
      { fireId: "fire-2", latitude: null, longitude: -117.4 },
    ]);
  });

  it("annotateCatalogWithLocations swallows lookup failures and warns for non-abort errors", async () => {
    const { annotateCatalogWithLocations } = await import("./geocode.js");
    fetch.mockResolvedValueOnce(
      jsonResponse(
        {
          message: "nope",
        },
        { status: 500 },
      ),
    );

    const fires = [{ fireId: "fire-3", latitude: 33.5, longitude: -112.1 }];
    await expect(annotateCatalogWithLocations(fires, "token")).resolves.toEqual(fires);

    expect(console.warn).toHaveBeenCalledWith(
      "Failed to resolve location for",
      "fire-3",
      expect.any(Error),
    );
  });
});
