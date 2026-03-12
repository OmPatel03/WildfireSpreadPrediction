import { useEffect, useState } from "react";

export default function TimelinePanel({
  timeline,
  currentFrame,
  framePosition,
  onChangePosition,
  onStep,
  loading,
  error,
  collapsed,
  onToggleCollapse,
}) {
  const frameCount = timeline?.frames?.length ?? 0;
  const [pendingPosition, setPendingPosition] = useState(framePosition ?? 0);

  useEffect(() => {
    setPendingPosition(framePosition ?? 0);
  }, [framePosition]);

  const maxFramePosition = Math.max(frameCount - 1, 0);
  const sliderPosition = Math.min(pendingPosition ?? 0, maxFramePosition);

  const commitPendingPosition = () => {
    if (loading || frameCount <= 1) return;
    const nextPosition = Math.min(Math.max(pendingPosition ?? 0, 0), maxFramePosition);
    if (nextPosition !== framePosition) {
      onChangePosition(nextPosition);
    }
  };

  return (
    <div className={`timeline-panel app-overlay${collapsed ? " is-collapsed" : ""}`}>
      <div className="timeline-header">
        <div>
          <p className="eyebrow">Timeline</p>
          <h3>
            {currentFrame?.label ?? (frameCount ? "Select a frame" : "No timeline")}
          </h3>
        </div>
        <div className="timeline-controls">
          {!collapsed && (
            <>
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
            </>
          )}
          <button type="button" className="panel-collapse-button" onClick={onToggleCollapse}>
            {collapsed ? "+" : "×"}
          </button>
        </div>
      </div>

      {!collapsed && loading && <p className="status-text">Loading timeline…</p>}
      {!collapsed && error && <p className="status-text error">{error}</p>}

      {!collapsed && (
        <>
          <input
            className="timeline-range"
            type="range"
            min={0}
            max={maxFramePosition}
            step={1}
            value={sliderPosition}
            onChange={(event) => setPendingPosition(Number(event.target.value))}
            onMouseUp={commitPendingPosition}
            onTouchEnd={commitPendingPosition}
            onBlur={commitPendingPosition}
            disabled={loading || frameCount <= 1}
          />

          <div className="timeline-footer">
            <span>
              Frame {frameCount ? (framePosition ?? 0) + 1 : 0} / {frameCount}
            </span>
            <span>{currentFrame?.targetDate ?? "No target date"}</span>
          </div>
        </>
      )}
    </div>
  );
}
