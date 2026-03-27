import FilterBar from "./FilterBar";

export default function TopControls({
  className = "",
  deckClassName = "",
  stackClassName = "",
  onBackToIntro,
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
  onViewModeChange,
  osmProjection,
  onOsmProjectionChange,
  layerVisibility,
  onToggleLayer,
  selectedFire,
  onResetApp,
  modelInputsOpen,
  onToggleModelInputs,
  environmentOpen,
  onToggleEnvironment,
}) {
  return (
    <div className={`top-controls-layout app-overlay${className ? ` ${className}` : ""}`}>
      <FilterBar
        className={deckClassName}
        year={year}
        yearOptions={yearOptions}
        onYearChange={onYearChange}
        threshold={threshold}
        onThresholdChange={onThresholdChange}
        catalogLimit={catalogLimit}
        onCatalogLimitChange={onCatalogLimitChange}
        osmMapStyle={osmMapStyle}
        osmMapStyles={osmMapStyles}
        onOsmMapStyleChange={onOsmMapStyleChange}
        viewMode={viewMode}
        layerVisibility={layerVisibility}
        onToggleLayer={onToggleLayer}
      />

      <div className={`control-actions-stack${stackClassName ? ` ${stackClassName}` : ""}`}>
        <div className="toolbar-actions">
          <button
            type="button"
            className="control-button"
            onClick={onBackToIntro}
          >
            Back to intro
          </button>

          <button
            type="button"
            className="control-button control-button-reset"
            onClick={onResetApp}
            disabled={!selectedFire}
          >
            Reset view
          </button>

          <button
            type="button"
            className={(modelInputsOpen ? "control-button active" : "control-button") + " tooltip-anchor"}
            data-tooltip="Open the panel with the model features for the selected frame."
            onClick={onToggleModelInputs}
          >
            Model inputs
          </button>

          <button
            type="button"
            className={(environmentOpen ? "control-button active" : "control-button") + " tooltip-anchor"}
            data-tooltip="Open controls for adjusting environmental factors."
            onClick={onToggleEnvironment}
          >
            Environment
          </button>

          <div
            className="segmented-control"
            style={{ "--active-segment-index": viewMode === "3d" ? 1 : 0 }}
          >
            <button
              type="button"
              className={"tooltip-anchor" + (viewMode === "2d" ? " active" : "")}
              data-tooltip="Use the standard top-down map."
              onClick={() => onViewModeChange("2d")}
            >
              2D
            </button>
            <button
              type="button"
              className={"tooltip-anchor" + (viewMode === "3d" ? " active" : "")}
              data-tooltip="Use the pitched 3D map."
              onClick={() => onViewModeChange("3d")}
            >
              3D
            </button>
          </div>

          {viewMode === "3d" ? (
            <div className="action-stack-segment-group">
              <div
                className="segmented-control"
                role="group"
                aria-label="OSM 3D projection"
                style={{ "--active-segment-index": osmProjection === "globe" ? 1 : 0 }}
              >
                <button
                  type="button"
                  className={"tooltip-anchor" + (osmProjection === "mercator" ? " active" : "")}
                  data-tooltip="Show the 3D map in the standard flat mercator projection."
                  onClick={() => onOsmProjectionChange("mercator")}
                >
                  Flat
                </button>
                <button
                  type="button"
                  className={"tooltip-anchor tooltip-align-right" + (osmProjection === "globe" ? " active" : "")}
                  data-tooltip="Wrap the 3D map onto a globe projection."
                  onClick={() => onOsmProjectionChange("globe")}
                >
                  Globe
                </button>
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
