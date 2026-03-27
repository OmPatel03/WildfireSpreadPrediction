import { useEffect, useState } from "react";
import "./TimelineDock.css";

export default function TimelineDock({
  className = "",
  timeline,
  currentFrame,
  framePosition,
  onChangePosition,
  onStep,
  loading,
  error,
}) {
  const frameCount = timeline?.frames?.length ?? 0;
  const [pendingPosition, setPendingPosition] = useState(framePosition ?? 0);

  useEffect(() => {
    setPendingPosition(framePosition ?? 0);
  }, [framePosition]);

  const maxFramePosition = Math.max(frameCount - 1, 0);
  const sliderPosition = Math.min(pendingPosition ?? 0, maxFramePosition);
  const sliderProgress = maxFramePosition > 0
    ? (sliderPosition / maxFramePosition) * 100
    : 0;
  const showDock = loading || error || frameCount > 0;

  if (!showDock) return null;

  const commitPendingPosition = () => {
    if (loading || frameCount <= 1) return;
    const nextPosition = Math.min(Math.max(pendingPosition ?? 0, 0), maxFramePosition);
    if (nextPosition !== framePosition) {
      onChangePosition(nextPosition);
    }
  };

  return (
    <div className={`timeline-dock app-overlay${error ? " is-error" : ""}${loading ? " is-loading" : ""}${className ? ` ${className}` : ""}`}>
      <div className="timeline-dock-header">
        <div>
          <p className="eyebrow">Timeline</p>
          <h3>{currentFrame?.targetDate ?? (loading ? "Loading timeline" : "Timeline unavailable")}</h3>
        </div>
        {frameCount > 0 ? (
          <span className="panel-meta-pill">
            Frame {(framePosition ?? 0) + 1} / {frameCount}
          </span>
        ) : null}
      </div>

      {error ? (
        <p className="timeline-dock-status error">{error}</p>
      ) : (
        <>
          <div className="timeline-dock-controls">
            <button
              type="button"
              className="timeline-dock-arrow"
              onClick={() => onStep(-1)}
              disabled={loading || !frameCount || framePosition <= 0}
              aria-label="Previous frame"
            >
              ←
            </button>
            <div className="timeline-dock-date">
              {loading ? "Building frame sequence..." : currentFrame?.label ?? "Awaiting frame"}
            </div>
            <button
              type="button"
              className="timeline-dock-arrow"
              onClick={() => onStep(1)}
              disabled={loading || !frameCount || framePosition >= frameCount - 1}
              aria-label="Next frame"
            >
              →
            </button>
          </div>

          <input
            className="timeline-range timeline-dock-range"
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
            aria-label="Timeline scrubber"
            style={{ "--timeline-progress": `${sliderProgress}%` }}
          />
        </>
      )}
    </div>
  );
}
