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

export default function InsightsPanel({
  fire,
  summary,
  frame,
  timelineLoading,
  layersLoading,
  layerError,
  overviewCount,
}) {
  return (
    <aside className="side-panel right-panel app-overlay">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Insights</p>
          <h2>{fire ? fire.locationName ?? fire.fireId : "Regional overview"}</h2>
        </div>
      </div>

      {!fire && (
        <div className="insight-card empty-state">
          <p>Select a fire from the list or click one on the map.</p>
          <StatRow label="Visible fires" value={overviewCount} />
        </div>
      )}

      {fire && (
        <>
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
            {timelineLoading && <p className="status-text">Loading timeline…</p>}
            {layersLoading && <p className="status-text">Loading layers…</p>}
            {layerError && <p className="status-text error">{layerError}</p>}
            {summary &&
              summary.positivePixels === 0 &&
              summary.groundTruthPixels === 0 && (
                <p className="muted">
                  No active fire pixels are present in this frame. Move the timeline to
                  inspect another date.
                </p>
              )}
            {summary ? (
              <>
                <StatRow
                  label="Mean probability"
                  value={formatPercent(summary.meanProbability)}
                />
                <StatRow
                  label="Predicted positive"
                  value={`${summary.positivePixels} / ${summary.totalPixels}`}
                />
                <StatRow
                  label="Ground truth"
                  value={`${summary.groundTruthPixels} / ${summary.totalPixels}`}
                />
                <StatRow label="Precision" value={formatPercent(summary.precision)} />
                <StatRow label="Recall" value={formatPercent(summary.recall)} />
                <StatRow label="F1" value={formatPercent(summary.f1)} />
                <StatRow label="Accuracy" value={formatPercent(summary.accuracy)} />
              </>
            ) : (
              !layersLoading && <p className="muted">No prediction summary loaded yet.</p>
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
              <p className="muted">Select a frame to inspect confusion counts.</p>
            )}
          </div>
        </>
      )}
    </aside>
  );
}
