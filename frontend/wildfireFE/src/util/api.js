const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const GRAPHQL_URL =
  import.meta.env.VITE_GRAPHQL_URL ?? `${API_BASE_URL.replace(/\/$/, "")}/graphql`;

function parseRetryAfterMs(retryAfterHeader) {
  if (!retryAfterHeader) return null;
  const parsedSeconds = Number.parseFloat(retryAfterHeader);
  if (Number.isFinite(parsedSeconds) && parsedSeconds >= 0) {
    return Math.round(parsedSeconds * 1000);
  }
  return null;
}

function toRequestError(response, detail) {
  const suffix = detail ? `: ${detail}` : "";
  const error = new Error(`Request failed (${response.status})${suffix}`);
  error.status = response.status;
  return error;
}

async function sleep(ms, signal) {
  await new Promise((resolve, reject) => {
    const timeoutId = window.setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);

    const onAbort = () => {
      window.clearTimeout(timeoutId);
      reject(new DOMException("Aborted", "AbortError"));
    };

    if (signal) {
      if (signal.aborted) {
        onAbort();
        return;
      }
      signal.addEventListener("abort", onAbort, { once: true });
    }
  });
}

async function fetchJson(url, options = {}) {
  const headers = {
    Accept: "application/json",
    ...options.headers,
  };
  const maxRetries = 2;

  for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
    const response = await fetch(url, { ...options, headers });
    if (response.ok) {
      return response.json();
    }

    let detail = "";
    try {
      const payload = await response.clone().json();
      detail = payload?.detail ? String(payload.detail) : "";
    } catch {
      detail = "";
    }

    if (response.status === 429 && attempt < maxRetries) {
      const retryAfterMs = parseRetryAfterMs(response.headers.get("Retry-After"));
      const backoffMs = Math.min(1200, 250 * 2 ** attempt);
      const waitMs = retryAfterMs ?? backoffMs;
      await sleep(waitMs, options.signal);
      continue;
    }

    throw toRequestError(response, detail);
  }

  throw new Error("Request failed after retries.");
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

async function fetchLayers({
  fireId,
  year,
  sampleIndex,
  threshold,
  environmentScales = {},
  signal,
}) {
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
