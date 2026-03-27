import { useEffect, useRef, useState } from "react";

import { fetchTimeline } from "../util/api.js";

export default function useFireTimeline({
  fireId,
  year,
  enabled,
}) {
  const cacheRef = useRef(new Map());
  const [timeline, setTimeline] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!enabled || !fireId) {
      setTimeline(null);
      setLoading(false);
      setError(null);
      return undefined;
    }

    let ignore = false;
    const controller = new AbortController();

    async function loadTimeline() {
      setTimeline(null);
      setLoading(true);
      setError(null);

      try {
        const cacheKey = `${year}:${fireId}`;
        const payload = cacheRef.current.get(cacheKey)
          ?? await fetchTimeline({
            fireId,
            year,
            signal: controller.signal,
          });
        cacheRef.current.set(cacheKey, payload);

        if (!ignore) {
          setTimeline(payload);
        }
      } catch (nextError) {
        if (!ignore && nextError?.name !== "AbortError") {
          setTimeline(null);
          setError(nextError.message ?? "Unable to load timeline");
        }
      } finally {
        if (!ignore) {
          setLoading(false);
        }
      }
    }

    loadTimeline();
    return () => {
      ignore = true;
      controller.abort();
    };
  }, [enabled, fireId, year]);

  return {
    timeline,
    loading,
    error,
  };
}
