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
  mapProvider,
  onMapProviderChange,
  mapStyle,
  mapStyles,
  onMapStyleChange,
  viewMode,
  osmProjection,
  onOsmProjectionChange,
  layerVisibility,
  onToggleLayer,
}) {
  const visibleMapStyles = mapProvider === "osm"
    ? [{ value: "standard", label: "Standard" }]
    : mapStyles;
  const visibleMapStyleValue = mapProvider === "osm" ? "standard" : mapStyle;
  const showOsmProjectionControl = mapProvider === "osm" && viewMode === "3d";

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

          <div className="control-group compact provider-group">
            <label>Provider</label>
            <div className="segmented-control" role="group" aria-label="Map provider">
              <button
                type="button"
                className={mapProvider === "gee" ? "active" : ""}
                onClick={() => onMapProviderChange("gee")}
              >
                GEE
              </button>
              <button
                type="button"
                className={mapProvider === "osm" ? "active" : ""}
                onClick={() => onMapProviderChange("osm")}
              >
                OSM
              </button>
            </div>
          </div>

          <div className="control-group compact">
            <label htmlFor="map-style">Basemap</label>
            <select
              id="map-style"
              value={visibleMapStyleValue}
              disabled={mapProvider === "osm"}
              onChange={(event) => onMapStyleChange(event.target.value)}
            >
              {visibleMapStyles.map((style) => (
                <option key={style.value} value={style.value}>
                  {style.label}
                </option>
              ))}
            </select>
          </div>

          {showOsmProjectionControl ? (
            <div className="control-group compact projection-group">
              <label>Projection</label>
              <div className="segmented-control" role="group" aria-label="OSM 3D projection">
                <button
                  type="button"
                  className={osmProjection === "mercator" ? "active" : ""}
                  onClick={() => onOsmProjectionChange("mercator")}
                >
                  Flat
                </button>
                <button
                  type="button"
                  className={osmProjection === "globe" ? "active" : ""}
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

      <div className="top-bar-layers">
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
