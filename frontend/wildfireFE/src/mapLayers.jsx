export const heatmapLayer = {
  id: "earthquakes-heat",
  type: "heatmap",
  maxzoom: 24,
  paint: {
    "heatmap-weight": ["coalesce", ["get", "mag"], 1],

    // === 625 meter radius converted to pixels ===
    "heatmap-radius": [
      "interpolate",
      ["linear"],
      ["zoom"],

      // zoom 5 radius
      5,
      [
        "/",
        ["*", 625, ["^", 2, 5]],
        ["*", 156543.03392, ["cos", ["*", ["pi"], ["/", ["get", "lat"], 180]]]],
      ],

      // zoom 10 radius
      10,
      [
        "/",
        ["*", 625, ["^", 2, 10]],
        ["*", 156543.03392, ["cos", ["*", ["pi"], ["/", ["get", "lat"], 180]]]],
      ],

      // zoom 12 radius
      12,
      [
        "/",
        ["*", 625, ["^", 2, 12]],
        ["*", 156543.03392, ["cos", ["*", ["pi"], ["/", ["get", "lat"], 180]]]],
      ],

      // zoom 13 radius
      13,
      [
        "/",
        ["*", 625, ["^", 2, 13]],
        ["*", 156543.03392, ["cos", ["*", ["pi"], ["/", ["get", "lat"], 180]]]],
      ],

      // zoom 14 radius
      14,
      [
        "/",
        ["*", 625, ["^", 2, 14]],
        ["*", 156543.03392, ["cos", ["*", ["pi"], ["/", ["get", "lat"], 180]]]],
      ],

      // zoom 15 radius
      15,
      [
        "/",
        ["*", 625, ["^", 2, 15]],
        ["*", 156543.03392, ["cos", ["*", ["pi"], ["/", ["get", "lat"], 180]]]],
      ],

      // zoom 16 radius
      16,
      [
        "/",
        ["*", 625, ["^", 2, 16]],
        ["*", 156543.03392, ["cos", ["*", ["pi"], ["/", ["get", "lat"], 180]]]],
      ],

      // zoom 17 radius
      17,
      [
        "/",
        ["*", 625, ["^", 2, 17]],
        ["*", 156543.03392, ["cos", ["*", ["pi"], ["/", ["get", "lat"], 180]]]],
      ],

      // zoom 18 radius
      18,
      [
        "/",
        ["*", 625, ["^", 2, 18]],
        ["*", 156543.03392, ["cos", ["*", ["pi"], ["/", ["get", "lat"], 180]]]],
      ],

      // zoom 19 radius
      19,
      [
        "/",
        ["*", 625, ["^", 2, 19]],
        ["*", 156543.03392, ["cos", ["*", ["pi"], ["/", ["get", "lat"], 180]]]],
      ],

      // zoom 20 radius
      20,
      [
        "/",
        ["*", 625, ["^", 2, 20]],
        ["*", 156543.03392, ["cos", ["*", ["pi"], ["/", ["get", "lat"], 180]]]],
      ],

      // zoom 21 radius
      21,
      [
        "/",
        ["*", 625, ["^", 2, 21]],
        ["*", 156543.03392, ["cos", ["*", ["pi"], ["/", ["get", "lat"], 180]]]],
      ],

      // zoom 22 radius
      22,
      [
        "/",
        ["*", 625, ["^", 2, 22]],
        ["*", 156543.03392, ["cos", ["*", ["pi"], ["/", ["get", "lat"], 180]]]],
      ],

      // zoom 23 radius
      23,
      [
        "/",
        ["*", 625, ["^", 2, 23]],
        ["*", 156543.03392, ["cos", ["*", ["pi"], ["/", ["get", "lat"], 180]]]],
      ],

      // zoom 24 radius
      24,
      [
        "/",
        ["*", 625, ["^", 2, 24]],
        ["*", 156543.03392, ["cos", ["*", ["pi"], ["/", ["get", "lat"], 180]]]],
      ],
    ],

    // Much smoother color ramp (no sharp peak)
    // "heatmap-color": [
    //   "interpolate",
    //   ["linear"],
    //   ["heatmap-density"],

    //   0.0,
    //   "rgba(0,0,0,0)", // fully transparent at zero
    //   0.05,
    //   "rgba(33,102,172,0.35)", // faint blue very early
    //   0.2,
    //   "rgba(103,169,207,0.55)", // more blue
    //   0.4,
    //   "rgba(209,229,240,0.70)",
    //   0.65,
    //   "rgba(253,219,199,0.85)", // orange later in the density range
    //   0.85,
    //   "rgba(239,138,98,0.95)",
    //   1.0,
    //   "rgba(178,24,43,1.0)", // small red core
    // ],

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
    // slightly softer intensity so big blobs don’t get too harsh
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

export const probabilityLayer = {
  id: "wildfire-probability",
  type: "circle",
  paint: {
    "circle-radius": 8,
    "circle-color": [
      "interpolate",
      ["linear"],
      ["get", "probability"], // ← Maps YOUR probability value (0-1)
      0,
      "rgb(0,255,0)", // green (low risk)
      0.25,
      "rgb(255,255,0)", // yellow
      0.5,
      "rgb(255,165,0)", // orange
      0.75,
      "rgb(255,69,0)", // red-orange
      1,
      "rgb(178,24,43)", // dark red (high risk)
    ],
    "circle-opacity": 0.7,
  },
};
