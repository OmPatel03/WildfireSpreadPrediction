const SCALE_KEYS = [
  { key: "viirs_m11", label: "VIIRS M11" },
  { key: "viirs_i2", label: "VIIRS I2" },
  { key: "ndvi", label: "NDVI" },
  { key: "evi2", label: "EVI2" },
  { key: "precip", label: "Precip" },
  { key: "wind_speed", label: "Wind" },
];

export default function EnvironmentPanel({
  isOpen,
  scales,
  onScaleChange,
  onReset,
  collapsed,
  onToggleCollapse,
}) {
  if (!isOpen) return null;

  return (
    <div className={`environment-panel app-overlay${collapsed ? " is-collapsed" : ""}`}>
      <div className="panel-header compact-panel-header">
        <div>
          <p className="eyebrow">Environment</p>
          <h2>Scale model inputs</h2>
        </div>
        <div className="panel-actions">
          {!collapsed && (
            <button type="button" onClick={onReset}>Reset</button>
          )}
          <button type="button" className="panel-collapse-button" onClick={onToggleCollapse}>
            {collapsed ? "+" : "×"}
          </button>
        </div>
      </div>

      {!collapsed && (
        <>
          <p className="muted">
            Inspired by the 2021 notebook. Scale dynamic inputs from 0.5× to 2.0×.
            Terrain inputs remain fixed.
          </p>

          <div className="environment-grid">
            {SCALE_KEYS.map((item) => (
              <div key={item.key} className="control-group">
                <label htmlFor={`env-${item.key}`}>
                  {item.label} <span>{Number(scales[item.key] ?? 1).toFixed(1)}×</span>
                </label>
                <input
                  id={`env-${item.key}`}
                  type="range"
                  min={0.5}
                  max={2.0}
                  step={0.1}
                  value={scales[item.key] ?? 1}
                  onChange={(event) => onScaleChange(item.key, Number(event.target.value))}
                />
              </div>
            ))}
          </div>

          <div className="environment-static">
            <span className="environment-static-label">Fixed terrain</span>
            <span>Elevation</span>
            <span>Slope</span>
            <span>Aspect</span>
          </div>
        </>
      )}
    </div>
  );
}