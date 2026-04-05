import { useEffect, useState } from "react";

import {
  fetchGoodPredictions,
  fetchOverview,
} from "../util/api.js";

export default function useCatalogData({
  year,
  catalogLimit,
  enabled = true,
}) {
  const [catalog, setCatalog] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      return undefined;
    }

    let ignore = false;
    const overviewController = new AbortController();

    async function loadOverview() {
      setLoading(true);
      setError(null);

      try {
        const overviewLimit = year === 2021 ? 1000 : catalogLimit;
        const rows = await fetchOverview({
          year,
          limit: overviewLimit,
          offset: 0,
          signal: overviewController.signal,
        });
        let filteredRows = rows;

        if (year === 2021) {
          try {
            const goodPredictions = await fetchGoodPredictions({
              year,
              signal: overviewController.signal,
            });
            const allowedFireIds = new Set(
              goodPredictions
                .map((entry) => entry.fireId)
                .filter(Boolean),
            );
            filteredRows = rows.filter((fire) => allowedFireIds.has(fire.fireId));
          } catch (nextError) {
            if (nextError?.name === "AbortError") {
              throw nextError;
            }
            console.warn(
              "Unable to load good-prediction whitelist for 2021:",
              nextError,
            );
          }
        }

        if (year === 2021) {
          filteredRows = filteredRows.slice(0, catalogLimit);
        }

        if (!ignore) {
          setCatalog(filteredRows);
        }
      } catch (nextError) {
        if (!ignore && nextError?.name !== "AbortError") {
          setError(nextError.message ?? "Unable to load overview");
          setCatalog([]);
        }
      } finally {
        if (!ignore) {
          setLoading(false);
        }
      }
    }

    loadOverview();
    return () => {
      ignore = true;
      overviewController.abort();
    };
  }, [catalogLimit, enabled, year]);

  return {
    catalog,
    loading,
    error,
  };
}
