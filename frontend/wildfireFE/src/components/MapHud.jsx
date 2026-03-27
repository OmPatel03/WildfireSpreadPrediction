import "./MapHud.css";

const LEGEND_ITEMS = [
  { key: "predictionHeatmap", label: "Prediction", swatchClass: "prediction" },
  { key: "groundTruthHeatmap", label: "Ground truth", swatchClass: "ground-truth" },
  { key: "differenceHeatmap", label: "True positive", swatchClass: "true-positive" },
  { key: "differenceHeatmap", label: "False positive", swatchClass: "false-positive" },
  { key: "differenceHeatmap", label: "False negative", swatchClass: "false-negative" },
  { key: "extent", label: "Extent", swatchClass: "extent" },
  { key: "origin", label: "Origin", swatchClass: "origin" },
];

export default function MapHud({
  className = "",
  cardClassName = "",
  selectedFire,
  layerVisibility,
}) {
  const hasSelection = Boolean(selectedFire);

  return (
    <div className={`map-hud app-overlay${hasSelection ? "" : " is-idle"}${className ? ` ${className}` : ""}`}>
      <div className={`map-hud-card map-legend-card${cardClassName ? ` ${cardClassName}` : ""}`}>
        <div className="map-hud-header">
          <div>
            <p className="eyebrow">Map legend</p>
            <h3>Layer cues</h3>
          </div>
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
    </div>
  );
}
