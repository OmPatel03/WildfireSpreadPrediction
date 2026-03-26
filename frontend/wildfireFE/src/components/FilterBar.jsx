const LAYER_OPTIONS = [
  { key: "overview", label: "Overview" },
  { key: "predictionHeatmap", label: "Prediction" },
  { key: "predictionPolygons", label: "3D mask" },
  { key: "groundTruthHeatmap", label: "Ground truth" },
  { key: "differenceHeatmap", label: "Difference" },
  { key: "extent", label: "Extent" },
  { key: "origin", label: "Origin" },
];

const CONTROL_TOOLTIPS = {
  year: "Choose which wildfire season to browse and analyze.",
  catalogLimit: "Set how many incidents are loaded into the catalog for this year.",
  basemap: "Pick the OpenStreetMap basemap style behind the wildfire layers.",
  flatProjection: "Show the 3D map in the standard flat mercator projection.",
  globeProjection: "Wrap the 3D map onto a globe projection.",
  threshold: "Adjust the prediction cutoff used for the selected fire's spread layers.",
};

const LAYER_TOOLTIPS = {
  overview: "Show catalog fire locations across the map.",
  predictionHeatmap: "Show predicted spread intensity for the selected fire.",
  predictionPolygons: "Show predicted spread polygons and 3D extrusions.",
  groundTruthHeatmap: "Show observed fire spread for comparison.",
  differenceHeatmap: "Highlight true positives, false positives, and false negatives.",
  extent: "Outline the selected incident boundary.",
  origin: "Mark the selected incident ignition point.",
};

export default function FilterBar({
  year,
  yearOptions,
  onYearChange,
  threshold,
  onThresholdChange,
  catalogLimit,
  onCatalogLimitChange,
  osmMapStyle,
  osmMapStyles,
  onOsmMapStyleChange,
  viewMode,
  osmProjection,
  onOsmProjectionChange,
  layerVisibility,
  onToggleLayer,
}) {
  const showOsmProjectionControl = viewMode === "3d";

  return (
    <div className="top-bar">
      <div className="top-bar-header">
        <div className="top-bar-title">
          <p className="eyebrow">Control deck</p>
          {/* <h1>Wildfire spread explorer</h1> */}
        </div>

        {/* <div className="top-bar-summary">
          <span className="summary-chip">Year {year}</span>
          <span className="summary-chip">Threshold {threshold.toFixed(2)}</span>
          <span className="summary-chip">{catalogLimit} incidents</span>
        </div> */}
      </div>

      <div className="top-bar-main">
        <div className={"top-bar-controls" + (showOsmProjectionControl ? " has-globe-control" : "")}>
          <div className="control-group compact">
            <label htmlFor="year-select">Year</label>
            <div className="control-input-anchor tooltip-anchor" data-tooltip={CONTROL_TOOLTIPS.year}>
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
          </div>

          <div className="control-group compact">
            <label htmlFor="catalog-limit">Catalog size</label>
            <div className="control-input-anchor tooltip-anchor" data-tooltip={CONTROL_TOOLTIPS.catalogLimit}>
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
          </div>

          <div className="control-group compact">
            <label htmlFor="map-style">Basemap</label>
            <div className="control-input-anchor tooltip-anchor" data-tooltip={CONTROL_TOOLTIPS.basemap}>
              <select
                id="map-style"
                value={osmMapStyle}
                onChange={(event) => onOsmMapStyleChange(event.target.value)}
              >
                {osmMapStyles.map((style) => (
                  <option key={style.value} value={style.value}>
                    {style.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {showOsmProjectionControl ? (
            <div className="control-group compact projection-group">
              <label>Projection</label>
              <div className="segmented-control" role="group" aria-label="OSM 3D projection">
                <button
                  type="button"
                  className={"tooltip-anchor" + (osmProjection === "mercator" ? " active" : "")}
                  data-tooltip={CONTROL_TOOLTIPS.flatProjection}
                  onClick={() => onOsmProjectionChange("mercator")}
                >
                  Flat
                </button>
                <button
                  type="button"
                  className={"tooltip-anchor" + (osmProjection === "globe" ? " active" : "")}
                  data-tooltip={CONTROL_TOOLTIPS.globeProjection}
                  onClick={() => onOsmProjectionChange("globe")}
                >
                  Globe
                </button>
              </div>
            </div>
          ) : null}

          <div className="control-group compact range-group threshold-row-group">
            <label htmlFor="threshold-range">
              Threshold <span>{threshold.toFixed(2)}</span>
            </label>
            <div className="control-input-anchor tooltip-anchor" data-tooltip={CONTROL_TOOLTIPS.threshold}>
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
          </div>
        </div>
      </div>

      <div className="top-bar-layers">
        <div className="layer-chip-row">
          {LAYER_OPTIONS.map((layer) => {
            const isDisabled = layer.key === "predictionPolygons" && viewMode !== "3d";
            const checked = !isDisabled && Boolean(layerVisibility[layer.key]);
            const tooltip = isDisabled
              ? "Available only in 3D mode."
              : LAYER_TOOLTIPS[layer.key];

            return (
              <label
                key={layer.key}
                className={
                  "layer-chip tooltip-anchor" +
                  (checked ? " is-active" : "") +
                  (isDisabled ? " is-disabled" : "")
                }
                data-tooltip={tooltip}
                aria-disabled={isDisabled}
              >
                <input
                  type="checkbox"
                  checked={checked}
                  disabled={isDisabled}
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
