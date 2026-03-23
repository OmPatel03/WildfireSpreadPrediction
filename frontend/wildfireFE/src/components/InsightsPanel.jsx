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

export default function InsightsPanel({
  fire,
  summary,
  frame,
  timelineLoading,
  timelineError,
  layersLoading,
  layerError,
  overviewCount,
  collapsed,
  onToggleCollapse,
}) {
  return (
    <aside className={`side-panel right-panel app-overlay${collapsed ? " is-collapsed" : ""}`}>
      <div className="panel-header">
        <div>
          <p className="eyebrow">Insights</p>
          <h2>{fire ? fire.locationName ?? fire.fireId : "Regional overview"}</h2>
          {fire ? <p className="panel-subtitle">Frame-level performance and incident context.</p> : null}
        </div>
        <button type="button" className="panel-collapse-button" onClick={onToggleCollapse}>
          {collapsed ? "+" : "×"}
        </button>
      </div>

      {!collapsed && !fire && (
        <div className="insight-card empty-state compact-empty-state">
          <p className="empty-title">Select a fire from the list or click one on the map.</p>
          <MetricTile label="Visible fires" value={overviewCount} />
        </div>
      )}

      {!collapsed && fire && (
        <>
          <div className="insight-hero">
            <div>
              <p className="eyebrow">Current focus</p>
              <h3>{fire.locationName ?? fire.fireId}</h3>
            </div>
            <div className="insight-hero-meta">
              <span className="panel-meta-pill">Frame {frame?.label ?? "Pending"}</span>
              <span className="panel-meta-pill">{frame?.targetDate ?? "No target date"}</span>
            </div>
          </div>

          <div className="insight-card">
            <h3>Incident metadata</h3>
            <StatRow label="Fire ID" value={fire.fireId} />
            <StatRow label="Samples" value={fire.samples} />
            <StatRow
              label="Grid"
              value={`${fire.width} × ${fire.height}`}
            />
            <StatRow label="Observation window" value={formatDateRange(fire)} />
            <StatRow label="Current target" value={frame?.targetDate ?? "—"} />
          </div>

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
        </>
      )}
    </aside>
  );
}
