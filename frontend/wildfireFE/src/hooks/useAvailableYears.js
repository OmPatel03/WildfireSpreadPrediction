import { useEffect, useState } from "react";

import { fetchYears } from "../util/api.js";

export default function useAvailableYears({
  defaultYear,
  fallbackYearOptions = [defaultYear],
}) {
  const [yearOptions, setYearOptions] = useState(fallbackYearOptions);
  const [resolvedInitialYear, setResolvedInitialYear] = useState(defaultYear);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let ignore = false;
    const controller = new AbortController();

    async function loadYears() {
      setLoading(true);
      setError(null);

      try {
        const years = await fetchYears({ signal: controller.signal });
        if (ignore || !Array.isArray(years) || years.length === 0) {
          return;
        }

        setYearOptions(years);
        setResolvedInitialYear(
          years.includes(defaultYear) ? defaultYear : years[years.length - 1],
        );
      } catch (nextError) {
        if (!ignore && nextError?.name !== "AbortError") {
          console.warn("Unable to load available years:", nextError);
          setError(nextError);
        }
      } finally {
        if (!ignore) {
          setLoading(false);
        }
      }
    }

    loadYears();
    return () => {
      ignore = true;
      controller.abort();
    };
  }, [defaultYear]);

  return {
    yearOptions,
    resolvedInitialYear,
    loading,
    error,
  };
}
