const LEGEND_ITEMS = [
  { key: "predictionHeatmap", label: "Prediction", swatchClass: "prediction" },
  { key: "groundTruthHeatmap", label: "Ground truth", swatchClass: "ground-truth" },
  { key: "differenceHeatmap", label: "True positive", swatchClass: "true-positive" },
  { key: "differenceHeatmap", label: "False positive", swatchClass: "false-positive" },
  { key: "differenceHeatmap", label: "False negative", swatchClass: "false-negative" },
  { key: "extent", label: "Extent", swatchClass: "extent" },
  { key: "origin", label: "Origin", swatchClass: "origin" },
];

function describeFire(fire) {
  return fire?.locationName ?? fire?.fireId ?? "No fire selected";
}

export default function MapHud({
  selectedFire,
  currentFrame,
  layerVisibility,
  layersLoading,
  layersError,
  timelineLoading,
}) {
  const activeLayerCount = Object.values(layerVisibility ?? {}).filter(Boolean).length;
  const hasSelection = Boolean(selectedFire);
  const timelineStatus = timelineLoading
    ? "Loading frames"
    : currentFrame?.label ?? "Awaiting frame";
  const predictionStatus = layersError
    ? "Layer load failed"
    : layersLoading
      ? "Refreshing overlays"
      : selectedFire && currentFrame
        ? "Frame ready"
        : "Ready for selection";

  return (
    <div className={`map-hud app-overlay${hasSelection ? "" : " is-idle"}`}>
      <div className="map-hud-card map-legend-card">
        <div className="map-hud-header">
          <div>
            <p className="eyebrow">Map legend</p>
            <h3>Layer cues</h3>
          </div>
          <span className="panel-meta-pill">{activeLayerCount} active</span>
        </div>

        <div className="map-legend-grid">
          {LEGEND_ITEMS.map((item) => {
            const active = Boolean(layerVisibility?.[item.key]);

            return (
              <div key={item.label} className={`legend-item${active ? " is-active" : ""}`}>
                <span className={`legend-swatch ${item.swatchClass}`} />
                <span>{item.label}</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* {hasSelection ? (
        <div className="map-hud-card map-status-card">
          <div className="map-hud-header">
            <div>
              <p className="eyebrow">Focus status</p>
              <h3>{describeFire(selectedFire)}</h3>
            </div>
            <span className="panel-meta-pill">{selectedFire.year}</span>
          </div>

          <div className="hud-frame-pill">
            {currentFrame?.targetDate ?? "No target date available"}
          </div>
          <div className="hud-status-list">
            <div className="hud-status-row">
              <span>Visible layers</span>
              <strong>{activeLayerCount}</strong>
            </div>
            <div className="hud-status-row">
              <span>Timeline</span>
              <strong>{timelineStatus}</strong>
            </div>
            <div className="hud-status-row">
              <span>Prediction</span>
              <strong>{predictionStatus}</strong>
            </div>
          </div>
        </div>
      ) : null} */}
    </div>
  );
}
