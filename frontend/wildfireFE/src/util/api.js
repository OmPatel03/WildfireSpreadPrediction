const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const GRAPHQL_URL =
  import.meta.env.VITE_GRAPHQL_URL ?? `${API_BASE_URL.replace(/\/$/, "")}/graphql`;

async function fetchJson(url, options = {}) {
  const headers = {
    Accept: "application/json",
    ...options.headers,
  };
  const response = await fetch(url, { ...options, headers });
  if (!response.ok) {
    throw new Error(`Request failed (${response.status})`);
  }
  return response.json();
}

function buildUrl(path, params = {}) {
  const url = new URL(`${API_BASE_URL}${path}`);
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null || value === "") return;
    url.searchParams.set(key, String(value));
  });
  return url;
}

async function fetchOverview({ year, limit, offset = 0, signal }) {
  return fetchJson(buildUrl("/overview", { year, limit, offset }), { signal });
}

async function fetchYears({ signal } = {}) {
  return fetchJson(buildUrl("/years"), { signal });
}

async function fetchTimeline({ fireId, year, signal }) {
  return fetchJson(buildUrl(`/fires/${fireId}/timeline`, { year }), { signal });
}

async function fetchLayers({ fireId, year, sampleIndex, threshold, signal }) {
  const environmentScales = arguments[0]?.environmentScales ?? {};
  return fetchJson(
    buildUrl(`/fires/${fireId}/layers`, {
      year,
      sampleIndex,
      threshold,
      viirsM11Scale: environmentScales.viirs_m11,
      viirsI2Scale: environmentScales.viirs_i2,
      ndviScale: environmentScales.ndvi,
      evi2Scale: environmentScales.evi2,
      precipScale: environmentScales.precip,
      windSpeedScale: environmentScales.wind_speed,
    }),
    { signal },
  );
}

export {
  API_BASE_URL,
  GRAPHQL_URL,
  fetchJson,
  fetchLayers,
  fetchOverview,
  fetchTimeline,
  fetchYears,
};
