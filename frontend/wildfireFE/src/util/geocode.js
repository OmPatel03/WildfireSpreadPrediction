const locationCache = new Map();

async function lookupLocationName(latitude, longitude, token, signal) {
  if (
    !token ||
    typeof latitude !== "number" ||
    typeof longitude !== "number"
  ) {
    return null;
  }

  const cacheKey = `${latitude.toFixed(4)},${longitude.toFixed(4)}`;
  if (locationCache.has(cacheKey)) {
    return locationCache.get(cacheKey);
  }

  const params = new URLSearchParams({
    access_token: token,
    limit: "1",
    types: "place,locality,region",
  });
  const geocodeUrl = `https://api.mapbox.com/geocoding/v5/mapbox.places/${longitude},${latitude}.json?${params.toString()}`;
  const response = await fetch(geocodeUrl, { signal });
  if (!response.ok) {
    throw new Error(`Reverse geocoding failed (${response.status})`);
  }
  const payload = await response.json();
  const feature = payload?.features?.[0];
  const locationName = feature?.text ?? feature?.place_name ?? null;
  locationCache.set(cacheKey, locationName);
  return locationName;
}

async function annotateCatalogWithLocations(fires, token, signal) {
  if (!Array.isArray(fires) || !token) return fires ?? [];

  const enriched = await Promise.all(
    fires.map(async (fire) => {
      if (
        typeof fire?.latitude !== "number" ||
        typeof fire?.longitude !== "number"
      ) {
        return fire;
      }

      try {
        const locationName = await lookupLocationName(
          fire.latitude,
          fire.longitude,
          token,
          signal
        );
        return locationName ? { ...fire, locationName } : fire;
      } catch (error) {
        if (error?.name !== "AbortError") {
          console.warn("Failed to resolve location for", fire?.fireId, error);
        }
        return fire;
      }
    })
  );

  return enriched;
}

export { annotateCatalogWithLocations, lookupLocationName };
