export const overviewRows = [
  {
    fireId: "fire-1",
    year: 2020,
    latitude: 34.1,
    longitude: -118.2,
    latestTargetDate: "2020-08-17",
    samples: 2,
  },
  {
    fireId: "fire-2",
    year: 2021,
    latitude: 36,
    longitude: -120,
    latestTargetDate: "2021-09-05",
    samples: 4,
    bbox: {
      minLon: -121,
      minLat: 35,
      maxLon: -119,
      maxLat: 37,
    },
  },
  {
    fireId: "fire-3",
    year: 2021,
    latitude: 38.2,
    longitude: -121.4,
    latestTargetDate: "2021-10-02",
    samples: 3,
  },
];

export const goodPredictions = [{ fireId: "fire-2" }];

export const enrichedOverviewRows = overviewRows.map((fire) =>
  fire.fireId === "fire-2"
    ? { ...fire, locationName: "Sierra Vista" }
    : fire,
);

export const timelinePayload = {
  defaultSampleIndex: 2,
  frames: [
    {
      sampleIndex: 2,
      targetDate: "2021-09-05",
      areaKm2: 10.4,
    },
    {
      sampleIndex: 3,
      targetDate: "2021-09-06",
      areaKm2: 12.1,
    },
  ],
};

export const selectedFire = {
  fireId: "fire-2",
  year: 2021,
  latitude: 36,
  longitude: -120,
  bbox: {
    minLon: -121,
    minLat: 35,
    maxLon: -119,
    maxLat: 37,
  },
};

export function createLayersResponse({
  fireId = "fire-2",
  sampleIndex = 2,
  predictionCoords = [
    [-120, 35],
    [-118, 37],
  ],
  groundTruthCoords = [[-119.5, 36.4]],
} = {}) {
  return {
    fire: { fireId },
    sampleIndex,
    summary: {
      predictedAreaKm2: 11.4,
    },
    layers: {
      predictionHeatmap: {
        type: "FeatureCollection",
        features: predictionCoords.map(([lon, lat]) => ({
          type: "Feature",
          geometry: {
            type: "Point",
            coordinates: [lon, lat],
          },
          properties: {},
        })),
      },
      groundTruthHeatmap: {
        type: "FeatureCollection",
        features: groundTruthCoords.map(([lon, lat]) => ({
          type: "Feature",
          geometry: {
            type: "Point",
            coordinates: [lon, lat],
          },
          properties: {},
        })),
      },
      modelInputs: null,
    },
  };
}
