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

export { buildCoordinatesArray, probabilityToMagnitude };
