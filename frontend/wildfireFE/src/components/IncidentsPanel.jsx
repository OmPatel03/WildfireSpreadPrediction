import "./IncidentsPanel.css";

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

function formatPercent(value) {
  return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "—";
}

function formatDateRange(fire) {
  if (!fire) return "—";
  if (fire.firstObservationDate && fire.lastObservationDate) {
    return `${fire.firstObservationDate} → ${fire.lastObservationDate}`;
  }
  return fire.latestTargetDate ?? "—";
}

function StatRow({ label, value }) {
  return (
    <div className="stat-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function MetricTile({ label, value }) {
  return (
    <div className="metric-tile">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function StateCard({ title, copy, tone = "info" }) {
  return (
    <div className={`state-card state-card-${tone}`}>
      <strong>{title}</strong>
      <p>{copy}</p>
    </div>
  );
}

export default function IncidentsPanel({
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
  view,
  onBackToCatalog,
  fire,
  summary,
  frame,
  timelineLoading,
  timelineError,
  layersLoading,
  layerError,
}) {
  const activeView = view === "detail" && fire ? "detail" : "catalog";

  return (
    <aside className={`side-panel left-panel incidents-panel app-overlay${collapsed ? " is-collapsed" : ""}`}>
      <div className="panel-header incidents-panel-header">
        {activeView === "detail" ? (
          <div className="incidents-panel-header-main">
            <button
              type="button"
              className="panel-back-button"
              onClick={onBackToCatalog}
              aria-label="Back to incident catalog"
            >
              ←
            </button>
            <div className="incidents-panel-title-block">
              <p className="eyebrow">Insights</p>
              <h2>{fire.locationName ?? fire.fireId}</h2>
              <p className="panel-subtitle">{fire.locationName ? fire.fireId : "Frame-level performance and incident context."}</p>
            </div>
          </div>
        ) : (
          <div className="incidents-panel-title-block">
            <p className="eyebrow">Catalog</p>
            <p className="panel-subtitle">Browse active records and jump into a fire footprint.</p>
          </div>
        )}
        <button type="button" className="panel-collapse-button" onClick={onToggleCollapse}>
          {collapsed ? "+" : "×"}
        </button>
      </div>

      {!collapsed && (
        <div className="incidents-panel-viewport">
          <div className={`incidents-panel-track${activeView === "detail" ? " is-detail" : ""}`}>
            <div className="incidents-panel-pane incidents-panel-pane-catalog">
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
                  {fires.map((catalogFire) => (
                    <button
                      key={catalogFire.fireId}
                      type="button"
                      className={`fire-card${selectedId === catalogFire.fireId ? " selected" : ""}`}
                      onClick={() => onSelectFire(catalogFire.fireId)}
                    >
                      <div className="fire-card-header">
                        <div className="fire-card-title">
                          <strong>{describeFire(catalogFire)}</strong>
                          <span className="fire-card-id">{catalogFire.fireId}</span>
                        </div>
                        <span className="fire-card-year">{catalogFire.year}</span>
                      </div>
                      <div className="fire-card-grid">
                        <span>Samples</span>
                        <span>{catalogFire.samples}</span>
                        <span>Grid</span>
                        <span>
                          {catalogFire.width} × {catalogFire.height}
                        </span>
                        <span>Target</span>
                        <span>{catalogFire.latestTargetDate ?? "n/a"}</span>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>

            <div className="incidents-panel-pane incidents-panel-pane-detail">
              {fire ? (
                <>
                  <div className="insight-card">
                    <h3>Model summary</h3>
                    {timelineLoading && (
                      <StateCard
                        tone="loading"
                        title="Timeline loading"
                        copy="Fetching frame order and target dates for this incident."
                      />
                    )}
                    {timelineError && (
                      <StateCard
                        tone="error"
                        title="Timeline unavailable"
                        copy={timelineError}
                      />
                    )}
                    {layersLoading && (
                      <StateCard
                        tone="loading"
                        title="Prediction layers updating"
                        copy="Refreshing overlay rasters and performance metrics for the selected frame."
                      />
                    )}
                    {layerError && (
                      <StateCard
                        tone="error"
                        title="Layer update failed"
                        copy={layerError}
                      />
                    )}
                    {summary &&
                      summary.positivePixels === 0 &&
                      summary.groundTruthPixels === 0 && (
                        <StateCard
                          tone="warning"
                          title="No active fire pixels"
                          copy="Move the timeline to inspect another date with active fire activity."
                        />
                      )}
                    {summary ? (
                      <>
                        <div className="metric-grid">
                          <MetricTile
                            label="Mean probability"
                            value={formatPercent(summary.meanProbability)}
                          />
                          <MetricTile label="Precision" value={formatPercent(summary.precision)} />
                          <MetricTile label="Recall" value={formatPercent(summary.recall)} />
                          <MetricTile label="F1" value={formatPercent(summary.f1)} />
                        </div>
                        <StatRow
                          label="Predicted positive"
                          value={`${summary.positivePixels} / ${summary.totalPixels}`}
                        />
                        <StatRow
                          label="Ground truth"
                          value={`${summary.groundTruthPixels} / ${summary.totalPixels}`}
                        />
                        <StatRow label="Accuracy" value={formatPercent(summary.accuracy)} />
                      </>
                    ) : (
                      !layersLoading &&
                      !timelineError &&
                      !layerError && (
                        <StateCard
                          tone="info"
                          title="No prediction summary yet"
                          copy="Select a frame or wait for layer data to finish loading."
                        />
                      )
                    )}
                  </div>

                  <div className="insight-card">
                    <h3>Confusion counts</h3>
                    {summary ? (
                      <>
                        <StatRow label="True positive" value={summary.truePositive} />
                        <StatRow label="False positive" value={summary.falsePositive} />
                        <StatRow label="False negative" value={summary.falseNegative} />
                        <StatRow label="True negative" value={summary.trueNegative} />
                      </>
                    ) : (
                      <StateCard
                        tone="info"
                        title="No confusion counts yet"
                        copy="Confusion totals appear after a frame summary has been loaded."
                      />
                    )}
                  </div>

                  <div className="insight-card incident-metadata-card incidents-panel-last-card">
                    <h3>Incident metadata</h3>
                    <div className="incident-meta-grid">
                      <div className="incident-id-tile">
                        <span>Fire ID</span>
                        <strong>{fire.fireId}</strong>
                      </div>

                      <MetricTile label="Samples" value={fire.samples} />
                      <MetricTile label="Grid" value={`${fire.width} × ${fire.height}`} />
                    </div>

                    <div className="incident-date-band">
                      <div className="incident-date-row">
                        <span>Observation window</span>
                        <strong>{formatDateRange(fire)}</strong>
                      </div>
                      <div className="incident-date-row">
                        <span>Current target</span>
                        <strong>{frame?.targetDate ?? "—"}</strong>
                      </div>
                    </div>
                  </div>
                </>
              ) : (
                <div className="insight-card empty-state compact-empty-state">
                  <p className="empty-title">Select a fire from the catalog or click one on the map.</p>
                  <MetricTile label="Visible fires" value={totalCount} />
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </aside>
  );
}
