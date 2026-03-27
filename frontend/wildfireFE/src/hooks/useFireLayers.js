import { useEffect, useMemo, useRef, useState } from "react";

import { fetchLayers } from "../util/api.js";

export default function useFireLayers({
  fireId,
  year,
  sampleIndex,
  threshold,
  environmentScales,
  enabled,
}) {
  const cacheRef = useRef(new Map());
  const [layersResponse, setLayersResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const serializedEnvironmentScales = JSON.stringify(environmentScales ?? {});
  const normalizedEnvironmentScales = useMemo(
    () => JSON.parse(serializedEnvironmentScales),
    [serializedEnvironmentScales],
  );

  useEffect(() => {
    if (!enabled || !fireId || sampleIndex === null || sampleIndex === undefined) {
      setLayersResponse(null);
      setLoading(false);
      setError(null);
      return undefined;
    }

    let ignore = false;
    const controller = new AbortController();

    async function loadLayers() {
      setLoading(true);
      setError(null);

      try {
        const cacheKey = [
          year,
          fireId,
          sampleIndex,
          threshold,
          serializedEnvironmentScales,
        ].join(":");

        const payload = cacheRef.current.get(cacheKey)
          ?? await fetchLayers({
            fireId,
            year,
            sampleIndex,
            threshold,
            environmentScales: normalizedEnvironmentScales,
            signal: controller.signal,
          });

        cacheRef.current.set(cacheKey, payload);

        if (!ignore) {
          setLayersResponse(payload);
        }
      } catch (nextError) {
        if (!ignore && nextError?.name !== "AbortError") {
          setLayersResponse(null);
          setError(nextError.message ?? "Unable to load fire layers");
        }
      } finally {
        if (!ignore) {
          setLoading(false);
        }
      }
    }

    loadLayers();
    return () => {
      ignore = true;
      controller.abort();
    };
  }, [
    enabled,
    fireId,
    normalizedEnvironmentScales,
    sampleIndex,
    serializedEnvironmentScales,
    threshold,
    year,
  ]);

  return {
    layersResponse,
    loading,
    error,
  };
}
