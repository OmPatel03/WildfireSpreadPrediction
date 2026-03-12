import { useEffect, useMemo, useRef } from "react";

const INPUT_ORDER = [
  "viirs_m11",
  "viirs_i2",
  "ndvi",
  "evi2",
  "precip",
  "wind_speed",
  "elevation",
  "slope",
  "aspect",
];

function interpolateColor(stops, value) {
  const clamped = Math.min(1, Math.max(0, value));
  for (let index = 1; index < stops.length; index += 1) {
    const [endStop, endColor] = stops[index];
    const [startStop, startColor] = stops[index - 1];
    if (clamped <= endStop) {
      const ratio = (clamped - startStop) / Math.max(endStop - startStop, 1e-6);
      return startColor.map((channel, channelIndex) =>
        Math.round(channel + (endColor[channelIndex] - channel) * ratio),
      );
    }
  }
  return stops[stops.length - 1][1];
}

function RasterPreview({ raster, label, min, max, mean }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !Array.isArray(raster) || raster.length === 0) return;

    const height = raster.length;
    const width = Array.isArray(raster[0]) ? raster[0].length : 0;
    if (!width) return;

    const context = canvas.getContext("2d");
    if (!context) return;

    canvas.width = width;
    canvas.height = height;

    const image = context.createImageData(width, height);
    const values = raster.flat().filter((value) => Number.isFinite(value));
    if (!values.length) {
      context.clearRect(0, 0, width, height);
      return;
    }

    const sortedValues = [...values].sort((left, right) => left - right);
    const percentile = (ratio) => {
      const index = Math.min(
        sortedValues.length - 1,
        Math.max(0, Math.floor(ratio * (sortedValues.length - 1))),
      );
      return sortedValues[index];
    };

    const localMin = Number.isFinite(min) ? min : percentile(0.05);
    const localMax = Number.isFinite(max) ? max : percentile(0.95);
    const stops = [
      [0, [15, 23, 42]],
      [0.2, [59, 130, 246]],
      [0.45, [14, 165, 233]],
      [0.7, [249, 115, 22]],
      [1, [239, 68, 68]],
    ];

    let cursor = 0;
    for (let row = 0; row < height; row += 1) {
      for (let col = 0; col < width; col += 1) {
        const value = Number(raster[row][col]);
        const normalized = Number.isFinite(value)
          ? localMax > localMin
            ? (value - localMin) / (localMax - localMin)
            : 1
          : 0;
        const [r, g, b] = interpolateColor(stops, normalized);
        image.data[cursor] = r;
        image.data[cursor + 1] = g;
        image.data[cursor + 2] = b;
        image.data[cursor + 3] = 255;
        cursor += 4;
      }
    }

    context.putImageData(image, 0, 0);
  }, [max, min, raster]);

  return (
    <div className="input-raster-card">
      <div className="input-raster-header">
        <strong>{label}</strong>
        <span>{Number.isFinite(mean) ? mean.toFixed(2) : "—"}</span>
      </div>
      <canvas ref={canvasRef} className="input-raster-canvas" />
      <div className="input-raster-meta">
        <span>min {Number.isFinite(min) ? min.toFixed(2) : "—"}</span>
        <span>max {Number.isFinite(max) ? max.toFixed(2) : "—"}</span>
      </div>
      {!Number.isFinite(mean) ? <span className="input-raster-empty">No finite data</span> : null}
    </div>
  );
}

export default function ModelInputsPanel({
  isOpen,
  modelInputs,
  collapsed,
  onToggleCollapse,
}) {
  const items = useMemo(
    () =>
      INPUT_ORDER.map((key) => ({ key, ...modelInputs?.[key] }))
        .filter((item) => Array.isArray(item.raster) && item.raster.length > 0),
    [modelInputs],
  );

  if (!isOpen) return null;

  return (
    <div className={`model-inputs-panel app-overlay${collapsed ? " is-collapsed" : ""}`}>
      <div className="panel-header compact-panel-header">
        <div>
          <p className="eyebrow">Model inputs</p>
          <h2>Raster previews</h2>
        </div>
        <button type="button" className="panel-collapse-button" onClick={onToggleCollapse}>
          {collapsed ? "+" : "×"}
        </button>
      </div>

      {!collapsed && (
        <>
          <p className="muted">
            Notebook-style raster previews for the current fire and frame.
          </p>

          <div className="input-raster-grid">
            {items.map((item) => (
              <RasterPreview
                key={item.key}
                raster={item.raster}
                label={item.label ?? item.key}
                min={item.min}
                max={item.max}
                mean={item.mean}
              />
            ))}
          </div>
        </>
      )}
    </div>
  );
}