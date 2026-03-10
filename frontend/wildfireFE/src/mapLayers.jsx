function buildMeterBasedRadiusExpression(meters, minZoom = 5, maxZoom = 24) {
  return [
    "interpolate",
    ["linear"],
    ["zoom"],
    ...Array.from({ length: maxZoom - minZoom + 1 }, (_, index) => {
      const zoom = minZoom + index;
      return [
        zoom,
        [
          "/",
          ["*", meters, ["^", 2, zoom]],
          [
            "*",
            156543.03392,
            ["cos", ["*", ["pi"], ["/", ["get", "lat"], 180]]],
          ],
        ],
      ];
    }).flat(),
  ];
}

export const heatmapLayer = {
  id: "wildfire-groundtruth-heat",
  type: "heatmap",
  maxzoom: 24,
  filter: [">", ["coalesce", ["get", "mag"], 0], 0],
  paint: {
    "heatmap-weight": ["coalesce", ["get", "mag"], 0],
    "heatmap-radius": buildMeterBasedRadiusExpression(625),
    "heatmap-color": [
      "interpolate",
      ["linear"],
      ["heatmap-density"],
      0.0,
      "rgba(0,0,0,0)",
      0.2,
      "rgba(33,102,172,0.35)",
      0.4,
      "rgba(103,169,207,0.55)",
      0.6,
      "rgba(209,229,240,0.7)",
      0.8,
      "rgba(253,219,199,0.85)",
      1.0,
      "rgba(239,138,98,0.75)",
    ],
    "heatmap-intensity": [
      "interpolate",
      ["linear"],
      ["zoom"],
      0,
      0.6,
      10,
      1.0,
      16,
      1.4,
    ],
    "heatmap-opacity": 1,
  },
};

export const predictionHeatmapLayer = {
  id: "wildfire-prediction-heat",
  type: "heatmap",
  maxzoom: 24,
  filter: [">", ["coalesce", ["get", "mag"], 0], 0],
  paint: {
    // Normalize magnitude (0-6) to 0-1 so prediction heatmap is less likely to saturate.
    "heatmap-weight": ["/", ["coalesce", ["get", "mag"], 0], 6],
    // Keep kernel size consistent in meters so colors don't collapse to blue when zooming in.
    "heatmap-radius": buildMeterBasedRadiusExpression(1750),
    "heatmap-color": [
      "interpolate",
      ["linear"],
      ["heatmap-density"],
      0.0,
      "rgba(0,0,0,0)",
      0.08,
      "rgba(33,102,172,0.25)",
      0.2,
      "rgba(67,147,195,0.4)",
      0.38,
      "rgba(146,197,222,0.55)",
      0.56,
      "rgba(244,165,130,0.72)",
      0.74,
      "rgba(214,96,77,0.84)",
      1.0,
      "rgba(178,24,43,0.95)",
    ],
    "heatmap-intensity": [
      "interpolate",
      ["linear"],
      ["zoom"],
      0,
      0.5,
      8,
      0.75,
      10,
      0.9,
      14,
      1.1,
      16,
      1.25,
    ],
    "heatmap-opacity": 0.95,
  },
};

export const predictionExtrusionLayer = {
  id: "wildfire-prediction-extrusion",
  type: "fill-extrusion",
  minzoom: 7,
  filter: [">", ["coalesce", ["get", "mag"], 0], 0],
  paint: {
    "fill-extrusion-color": [
      "interpolate",
      ["linear"],
      ["/", ["coalesce", ["get", "mag"], 0], 6],
      0.0,
      "rgb(33,102,172)",
      0.2,
      "rgb(67,147,195)",
      0.38,
      "rgb(146,197,222)",
      0.56,
      "rgb(244,165,130)",
      0.74,
      "rgb(214,96,77)",
      1.0,
      "rgb(178,24,43)",
    ],
    "fill-extrusion-height": ["coalesce", ["get", "height"], 0],
    "fill-extrusion-base": 0,
    "fill-extrusion-opacity": 0.82,
  },
};
