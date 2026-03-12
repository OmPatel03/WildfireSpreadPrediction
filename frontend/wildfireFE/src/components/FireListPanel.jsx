function describeFire(fire) {
  if (fire?.locationName) return fire.locationName;
  if (
    typeof fire?.latitude === "number" &&
    typeof fire?.longitude === "number"
  ) {
    return `${fire.latitude.toFixed(2)}, ${fire.longitude.toFixed(2)}`;
  }
  return fire?.fireId ?? "Unknown fire";
}

export default function FireListPanel({
  fires,
  selectedId,
  loading,
  error,
  searchTerm,
  onSearchChange,
  page,
  totalPages,
  onPrevPage,
  onNextPage,
  onSelectFire,
  collapsed,
  onToggleCollapse,
}) {
  return (
    <aside className={`side-panel left-panel app-overlay${collapsed ? " is-collapsed" : ""}`}>
      <div className="panel-header">
        <div>
          <p className="eyebrow">Incident browser</p>
          <h2>Wildfire catalog</h2>
        </div>
        <button type="button" className="panel-collapse-button" onClick={onToggleCollapse}>
          {collapsed ? "+" : "×"}
        </button>
      </div>

      {!collapsed && (
        <>
          <div className="control-group">
            <label htmlFor="fire-search">Search fires</label>
            <input
              id="fire-search"
              type="search"
              value={searchTerm}
              onChange={(event) => onSearchChange(event.target.value)}
              placeholder="Fire ID or location"
            />
          </div>

          <div className="panel-meta-row">
            <span>{fires.length} fires</span>
            <div className="pagination-controls">
              <button type="button" onClick={onPrevPage} disabled={page === 0}>
                ‹
              </button>
              <span>
                {Math.min(page + 1, totalPages)} / {Math.max(totalPages, 1)}
              </span>
              <button
                type="button"
                onClick={onNextPage}
                disabled={page >= totalPages - 1}
              >
                ›
              </button>
            </div>
          </div>

          {loading && <p className="status-text">Loading overview…</p>}
          {error && <p className="status-text error">{error}</p>}

          <div className="fire-list">
            {fires.map((fire) => (
              <button
                key={fire.fireId}
                type="button"
                className={`fire-card${selectedId === fire.fireId ? " selected" : ""}`}
                onClick={() => onSelectFire(fire.fireId)}
              >
                <div className="fire-card-header">
                  <strong>{describeFire(fire)}</strong>
                  <span>{fire.year}</span>
                </div>
                <div className="fire-card-grid">
                  <span>ID</span>
                  <span>{fire.fireId}</span>
                  <span>Samples</span>
                  <span>{fire.samples}</span>
                  <span>Grid</span>
                  <span>
                    {fire.width} × {fire.height}
                  </span>
                  <span>Target</span>
                  <span>{fire.latestTargetDate ?? "n/a"}</span>
                </div>
              </button>
            ))}
          </div>
        </>
      )}
    </aside>
  );
}
