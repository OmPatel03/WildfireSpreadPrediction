import "./LandingPage.css";

export default function LandingPage({ onEnter, className = "" }) {
  return (
    <section
      className={`landing-screen app-overlay${className ? ` ${className}` : ""}`}
      aria-label="WISPR landing page"
    >
      <div className="landing-screen-backdrop" />

      <div className="landing-shell">
        <div className="landing-card landing-hero-card">
          {/* <p className="eyebrow">Wildfire Intelligence</p> */}
          <h1>WISPR</h1>
          <p className="landing-tagline">
            Predicting wildfire spread with modern machine learning to generate
            insights that can surpass traditional mechanistic models.
          </p>

          <div className="landing-actions">
            <button
              type="button"
              className="landing-primary-action"
              onClick={onEnter}
            >
              Enter WISPR
            </button>
            <span className="landing-action-hint">
              Open the interactive wildfire map and analysis workspace.
            </span>
          </div>
        </div>
      </div>
    </section>
  );
}
