const { cos, PI } = Math;

function findCoord(n, m, centerLat, centerLong, row, col) {
  const latStep = 1 / 296;
  const longStep = 375 / (111000 * cos((centerLat * Math.PI) / 180));
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
    for (let col = 0; col < m; col++) {
      const prob = probabilityToMagnitude(probabilities[row][col]);
      if (prob > 0) positive++;
      const [lat, lon] = findCoord(n, m, centerLat, centerLong, row, col);
      const geometry = {
        type: "Feature",
        geometry: { type: "Point", coordinates: [lon, lat] },
        properties: {
          mag: probabilityToMagnitude(probabilities[row][col]),
          lat: lat,
        },
      };
      coordinates.push(geometry);
    }
  }
  return [coordinates, positive];
}

function probabilityToMagnitude(p, maxProbability = 1) {
  const noiseFloor = 2.5e-5; // based on mean/10

  // Drop noise
  if (p < noiseFloor) return 0;

  // Normalize to 0-1
  const normalized = p / maxProbability;

  // Smooth with sqrt and scale to useful heatmap range 0-6
  return Math.sqrt(normalized) * 6;
}

function computeMetrics(mask, groundTruth) {
  let truePositive = 0;
  let falsePositive = 0;
  let falseNegative = 0;

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
      }
    }
  }

  const precision =
    truePositive + falsePositive > 0 ? truePositive / (truePositive + falsePositive) : 0;
  const recall =
    truePositive + falseNegative > 0 ? truePositive / (truePositive + falseNegative) : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

  return { precision, recall, f1 };
}

export { buildCoordinatesArray, probabilityToMagnitude, computeMetrics };
