const { cos, PI } = Math;
const LAT_STEP = 1 / 296;
const COLUMN_WIDTH_METERS = 375;
const EXTRUSION_HEIGHT_SCALE = 140;

function findCoord(n, m, centerLat, centerLong, row, col) {
  const latStep = LAT_STEP;
  const longStep =
    COLUMN_WIDTH_METERS / (111000 * cos((centerLat * Math.PI) / 180));
  const zeroLat =
    centerLat - Math.floor(n / 2) * latStep - (latStep / 2) * (n % 2);
  const zeroLong =
    centerLong - Math.floor(m / 2) * longStep - (longStep / 2) * (m % 2);
  const lat = zeroLat + (n - row - 1) * latStep;
  const lon = zeroLong + col * longStep;
  return [lat, lon];
}

function buildCoordinatesArray(n, m, centerLat, centerLong, probabilities) {
  const coordinates = [];
  let positive = 0;
  for (let row = 0; row < n; row++) {
    const rowData = probabilities[row] || [];
    for (let col = 0; col < m; col++) {
      const magnitude = probabilityToMagnitude(rowData[col]);
      if (magnitude > 0) positive++;
      const [lat, lon] = findCoord(n, m, centerLat, centerLong, row, col);
      const geometry = {
        type: "Feature",
        geometry: { type: "Point", coordinates: [lon, lat] },
        properties: {
          mag: magnitude,
          lat: lat,
        },
      };
      coordinates.push(geometry);
    }
  }
  return [coordinates, positive];
}

function buildExtrusionArray(n, m, centerLat, centerLong, probabilities) {
  const features = [];
  let positive = 0;
  const latStep = LAT_STEP;
  const longStep =
    COLUMN_WIDTH_METERS / (111000 * cos((centerLat * Math.PI) / 180));
  const halfLat = latStep / 2;
  const halfLong = longStep / 2;

  for (let row = 0; row < n; row++) {
    const rowData = probabilities[row] || [];
    for (let col = 0; col < m; col++) {
      const magnitude = probabilityToMagnitude(rowData[col]);
      if (magnitude <= 0) continue;
      positive++;

      const [lat, lon] = findCoord(n, m, centerLat, centerLong, row, col);
      const polygon = [
        [
          [lon - halfLong, lat - halfLat],
          [lon + halfLong, lat - halfLat],
          [lon + halfLong, lat + halfLat],
          [lon - halfLong, lat + halfLat],
          [lon - halfLong, lat - halfLat],
        ],
      ];

      features.push({
        type: "Feature",
        geometry: { type: "Polygon", coordinates: polygon },
        properties: {
          mag: magnitude,
          height: magnitude * EXTRUSION_HEIGHT_SCALE,
          lat,
          lon,
        },
      });
    }
  }

  return [features, positive];
}

function probabilityToMagnitude(p, maxProbability = 1) {
  const noiseFloor = 2.5e-5; // based on mean/10

  if (!Number.isFinite(p) || p <= noiseFloor) return 0;
  const safeMax =
    Number.isFinite(maxProbability) && maxProbability > 0 ? maxProbability : 1;

  const normalized = Math.min(1, Math.max(0, p / safeMax));
  const magnitude = Math.sqrt(normalized) * 6;
  return Math.min(6, Math.max(0, magnitude));
}

function computeMetrics(mask, groundTruth) {
  let truePositive = 0;
  let falsePositive = 0;
  let falseNegative = 0;
  let trueNegative = 0;

  const radius = 2; // consider neighboring pixels within this radius

  for (let i = 0; i < mask.length; i++) {
    const maskRow = mask[i] || [];
    const gtRow = groundTruth[i] || [];
    for (let j = 0; j < maskRow.length; j++) {
      const prediction = Boolean(maskRow[j]);
      const actual = Boolean(gtRow[j]);

      if (prediction && actual) {
        truePositive += 1;
      } else if (prediction && !actual) {
        falsePositive += 1;
      } else if (!prediction && actual) {
        falseNegative += 1;
      } else if (!prediction && !actual) {
        trueNegative += 1;
      }
    }
  }

  const precision =
    truePositive + falsePositive > 0
      ? truePositive / (truePositive + falsePositive)
      : 0;
  const recall =
    truePositive + falseNegative > 0
      ? truePositive / (truePositive + falseNegative)
      : 0;
  const f1 =
    precision + recall > 0
      ? (2 * precision * recall) / (precision + recall)
      : 0;

  const accuracy =
    truePositive /
    (truePositive + trueNegative + falsePositive + falseNegative);
  return { precision, recall, f1, accuracy };
}

export {
  buildCoordinatesArray,
  buildExtrusionArray,
  probabilityToMagnitude,
  computeMetrics,
};
