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
  totalCount,
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
          <p className="panel-subtitle">Browse active records and jump into a fire footprint.</p>
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
            <span className="panel-meta-pill">{totalCount} visible fires</span>
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

          {loading && <p className="status-card">Loading overview…</p>}
          {error && (
            <div className="status-card state-card error-card">
              <strong>Catalog unavailable for this dataset</strong>
              <p>Try another year or verify the API route, then refresh the catalog.</p>
              <p className="status-card-detail">{error}</p>
            </div>
          )}

          {!loading && !error && fires.length === 0 ? (
            <div className="status-card state-card state-card-info">
              <strong>{searchTerm ? "No fires match this search" : "No fires available"}</strong>
              <p>
                {searchTerm
                  ? "Try a fire ID, location, or clear the search to restore the catalog."
                  : "This year returned no visible incidents yet."}
              </p>
            </div>
          ) : (
            <div className="fire-list">
              {fires.map((fire) => (
                <button
                  key={fire.fireId}
                  type="button"
                  className={`fire-card${selectedId === fire.fireId ? " selected" : ""}`}
                  onClick={() => onSelectFire(fire.fireId)}
                >
                  <div className="fire-card-header">
                    <div className="fire-card-title">
                      <strong>{describeFire(fire)}</strong>
                      <span className="fire-card-id">{fire.fireId}</span>
                    </div>
                    <span className="fire-card-year">{fire.year}</span>
                  </div>
                  <div className="fire-card-grid">
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
          )}
        </>
      )}
    </aside>
  );
}
