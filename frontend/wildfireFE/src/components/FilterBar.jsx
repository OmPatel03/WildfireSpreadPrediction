const LAYER_OPTIONS = [
  { key: "overview", label: "Overview" },
  { key: "predictionHeatmap", label: "Prediction" },
  { key: "predictionPolygons", label: "3D mask" },
  { key: "groundTruthHeatmap", label: "Ground truth" },
  { key: "differenceHeatmap", label: "Difference" },
  { key: "extent", label: "Extent" },
  { key: "origin", label: "Origin" },
];

export default function FilterBar({
  year,
  yearOptions,
  onYearChange,
  threshold,
  onThresholdChange,
  catalogLimit,
  onCatalogLimitChange,
  mapStyle,
  mapStyles,
  onMapStyleChange,
  modelInputsOpen,
  onToggleModelInputs,
  viewMode,
  onViewModeChange,
  layerVisibility,
  onToggleLayer,
  environmentOpen,
  onToggleEnvironment,
}) {
  return (
    <div className="top-bar app-overlay">
      <div className="top-bar-header">
        <div className="top-bar-title">
          <p className="eyebrow">Control deck</p>
          <h1>Wildfire spread explorer</h1>
        </div>

        <div className="top-bar-summary">
          <span className="summary-chip">Year {year}</span>
          <span className="summary-chip">Threshold {threshold.toFixed(2)}</span>
          <span className="summary-chip">{catalogLimit} incidents</span>
        </div>
      </div>

      <div className="top-bar-section">
        <div className="control-group compact">
          <label htmlFor="year-select">Year</label>
          <select
            id="year-select"
            value={year}
            onChange={(event) => onYearChange(Number(event.target.value))}
          >
            {yearOptions.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group compact">
          <label htmlFor="catalog-limit">Catalog size</label>
          <select
            id="catalog-limit"
            value={catalogLimit}
            onChange={(event) => onCatalogLimitChange(Number(event.target.value))}
          >
            {[25, 50, 100, 200].map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group compact">
          <label htmlFor="map-style">Basemap</label>
          <select
            id="map-style"
            value={mapStyle}
            onChange={(event) => onMapStyleChange(event.target.value)}
          >
            {mapStyles.map((style) => (
              <option key={style.value} value={style.value}>
                {style.label}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group compact range-group threshold-row-group">
          <label htmlFor="threshold-range">
            Threshold <span>{threshold.toFixed(2)}</span>
          </label>
          <input
            id="threshold-range"
            type="range"
            min={0.1}
            max={0.95}
            step={0.05}
            value={threshold}
            onChange={(event) => onThresholdChange(Number(event.target.value))}
          />
        </div>

        <div className="toolbar-stack">
          <div className="toolbar-actions">
            <button
              type="button"
              className={modelInputsOpen ? "control-button active" : "control-button"}
              onClick={onToggleModelInputs}
            >
              Model inputs
            </button>

            <button
              type="button"
              className={environmentOpen ? "control-button active" : "control-button"}
              onClick={onToggleEnvironment}
            >
              Environment
            </button>
          </div>

          <div className="toolbar-mode-row">
            <div className="segmented-control">
              <button
                type="button"
                className={viewMode === "2d" ? "active" : ""}
                onClick={() => onViewModeChange("2d")}
              >
                2D
              </button>
              <button
                type="button"
                className={viewMode === "3d" ? "active" : ""}
                onClick={() => onViewModeChange("3d")}
              >
                3D
              </button>
            </div>
          </div>
        </div>

      </div>

      <div className="top-bar-section top-bar-section-secondary">
        <div className="layer-chip-row">
          {LAYER_OPTIONS.map((layer) => {
            const checked = Boolean(layerVisibility[layer.key]);

            return (
              <label key={layer.key} className={"layer-chip" + (checked ? " is-active" : "")}>
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => onToggleLayer(layer.key)}
                />
                <span className="layer-chip-indicator" />
                <span>{layer.label}</span>
              </label>
            );
          })}
        </div>
      </div>
    </div>
  );
}
