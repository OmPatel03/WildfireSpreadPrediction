export default function TimelinePanel({
  timeline,
  currentFrame,
  framePosition,
  onChangePosition,
  onStep,
  loading,
  error,
}) {
  const frameCount = timeline?.frames?.length ?? 0;

  return (
    <div className="timeline-panel app-overlay">
      <div className="timeline-header">
        <div>
          <p className="eyebrow">Timeline</p>
          <h3>
            {currentFrame?.label ?? (frameCount ? "Select a frame" : "No timeline")}
          </h3>
        </div>
        <div className="timeline-controls">
          <button
            type="button"
            onClick={() => onStep(-1)}
            disabled={loading || !frameCount || framePosition <= 0}
          >
            Prev
          </button>
          <button
            type="button"
            onClick={() => onStep(1)}
            disabled={loading || !frameCount || framePosition >= frameCount - 1}
          >
            Next
          </button>
        </div>
      </div>

      {loading && <p className="status-text">Loading timeline…</p>}
      {error && <p className="status-text error">{error}</p>}

      <input
        className="timeline-range"
        type="range"
        min={0}
        max={Math.max(frameCount - 1, 0)}
        step={1}
        value={Math.min(framePosition ?? 0, Math.max(frameCount - 1, 0))}
        onChange={(event) => onChangePosition(Number(event.target.value))}
        disabled={loading || frameCount <= 1}
      />

      <div className="timeline-footer">
        <span>
          Frame {frameCount ? (framePosition ?? 0) + 1 : 0} / {frameCount}
        </span>
        <span>{currentFrame?.targetDate ?? "No target date"}</span>
      </div>
    </div>
  );
}
