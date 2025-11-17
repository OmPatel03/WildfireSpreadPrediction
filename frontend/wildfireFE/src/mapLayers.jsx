// ...existing code...
export const heatmapLayer = {
  id: "earthquakes-heat",
  type: "heatmap",
  // keep heatmap visible at all zooms
  maxzoom: 24,
  paint: {
    "heatmap-weight": ["interpolate", ["linear"], ["get", "mag"], 0, 0, 6, 1],
    // intensity can still increase with zoom but the layer will remain visible
    "heatmap-intensity": ["interpolate", ["linear"], ["zoom"], 0, 1, 12, 3],
    "heatmap-color": [
      "interpolate",
      ["linear"],
      ["heatmap-density"],
      0,
      "rgba(33,102,172,0)",
      0.2,
      "rgb(103,169,207)",
      0.4,
      "rgb(209,229,240)",
      0.6,
      "rgb(253,219,199)",
      0.8,
      "rgb(239,138,98)",
      1,
      "rgb(178,24,43)",
    ],
    // larger radius at higher zooms so heat remains visible
    "heatmap-radius": ["interpolate", ["linear"], ["zoom"], 0, 2, 18, 50],
    // fixed opacity (do not fade out)
    "heatmap-opacity": 1,
  },
};

export const probabilityLayer = {
  id: "wildfire-probability",
  type: "circle",
  paint: {
    "circle-radius": 8,
    "circle-color": [
      "interpolate",
      ["linear"],
      ["get", "probability"], // ← Maps YOUR probability value (0-1)
      0,    "rgb(0,255,0)",      // green (low risk)
      0.25, "rgb(255,255,0)",    // yellow
      0.5,  "rgb(255,165,0)",    // orange
      0.75, "rgb(255,69,0)",     // red-orange
      1,    "rgb(178,24,43)"     // dark red (high risk)
    ],
    "circle-opacity": 0.7
  }
};
// ...existing code...
