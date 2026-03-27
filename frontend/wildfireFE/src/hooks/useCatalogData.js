import { useEffect, useState } from "react";

import {
  fetchGoodPredictions,
  fetchOverview,
} from "../util/api.js";
import { annotateCatalogWithLocations } from "../util/geocode.js";

export default function useCatalogData({
  year,
  catalogLimit,
  mapboxToken,
}) {
  const [catalog, setCatalog] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let ignore = false;
    const geocodeController = new AbortController();
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

        try {
          const enriched = await annotateCatalogWithLocations(
            filteredRows,
            mapboxToken,
            geocodeController.signal,
          );

          if (!ignore) {
            setCatalog(enriched);
          }
        } catch (nextError) {
          if (nextError?.name !== "AbortError") {
            console.warn("Location lookup failed:", nextError);
          }
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
      geocodeController.abort();
    };
  }, [catalogLimit, mapboxToken, year]);

  return {
    catalog,
    loading,
    error,
  };
}
