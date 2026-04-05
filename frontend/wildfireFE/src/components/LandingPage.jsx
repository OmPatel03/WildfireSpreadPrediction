import { useEffect, useEffectEvent, useRef, useState } from "react";
import "./LandingPage.css";

const ABOUT_TRANSITION_MS = 220;

const ABOUT_HIGHLIGHTS = [
  {
    title: "Predict spread",
    description:
      "Load historical wildfire events and compare model-predicted spread against observed fire growth.",
  },
  {
    title: "Explore timelines",
    description:
      "Scrub through valid event frames to see how spread risk evolves across each wildfire sequence.",
  },
  {
    title: "Inspect inputs",
    description:
      "Review model inputs and environmental context, including thermal, vegetation, weather, and terrain layers.",
  },
];

const ABOUT_FLOW_STEPS = [
  "Historical wildfire cubes provide active-fire context, environmental features, and geospatial metadata for each event.",
  "The backend preprocesses the selected frame, runs the trained model, and returns prediction layers with summary metrics.",
  "WISPR turns those outputs into interactive map overlays, timelines, and input views that are easier to inspect and compare.",
];

export default function LandingPage({ onEnter, className = "" }) {
  const [isAboutOpen, setIsAboutOpen] = useState(false);
  const [isAboutVisible, setIsAboutVisible] = useState(false);
  const aboutTriggerRef = useRef(null);
  const aboutCloseButtonRef = useRef(null);
  const closeTimeoutRef = useRef(null);

  const clearCloseTimeout = () => {
    if (closeTimeoutRef.current === null || typeof window === "undefined") {
      return;
    }

    window.clearTimeout(closeTimeoutRef.current);
    closeTimeoutRef.current = null;
  };

  const focusAboutTrigger = () => {
    if (typeof window === "undefined") {
      aboutTriggerRef.current?.focus();
      return;
    }

    window.requestAnimationFrame(() => {
      aboutTriggerRef.current?.focus();
    });
  };

  const handleOpenAbout = () => {
    clearCloseTimeout();
    setIsAboutVisible(true);

    if (typeof window === "undefined") {
      setIsAboutOpen(true);
      return;
    }

    window.requestAnimationFrame(() => {
      setIsAboutOpen(true);
    });
  };

  const handleCloseAbout = () => {
    clearCloseTimeout();
    setIsAboutOpen(false);
    focusAboutTrigger();

    if (typeof window === "undefined") {
      setIsAboutVisible(false);
      return;
    }

    closeTimeoutRef.current = window.setTimeout(() => {
      setIsAboutVisible(false);
      closeTimeoutRef.current = null;
    }, ABOUT_TRANSITION_MS);
  };

  const handleEscapeKeyDown = useEffectEvent((event) => {
    if (event.key !== "Escape") {
      return;
    }

    event.preventDefault();
    handleCloseAbout();
  });

  useEffect(() => () => clearCloseTimeout(), []);

  useEffect(() => {
    if (!isAboutOpen || typeof window === "undefined") {
      return undefined;
    }

    window.addEventListener("keydown", handleEscapeKeyDown);
    return () => {
      window.removeEventListener("keydown", handleEscapeKeyDown);
    };
  }, [handleEscapeKeyDown, isAboutOpen]);

  useEffect(() => {
    if (!isAboutOpen || typeof window === "undefined") {
      return;
    }

    window.requestAnimationFrame(() => {
      aboutCloseButtonRef.current?.focus();
    });
  }, [isAboutOpen]);

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
            <div className="landing-action-row">
              <button
                type="button"
                className="landing-primary-action"
                onClick={onEnter}
              >
                Enter WISPR
              </button>
              <button
                ref={aboutTriggerRef}
                type="button"
                className="landing-secondary-action"
                onClick={handleOpenAbout}
                aria-haspopup="dialog"
                aria-expanded={isAboutOpen}
                aria-controls="landing-about-dialog"
              >
                About WISPR
              </button>
            </div>
            <span className="landing-action-hint">
              Open the interactive wildfire map and analysis workspace.
            </span>
          </div>
        </div>
      </div>

      {isAboutVisible ? (
        <div
          className={`landing-about-layer${isAboutOpen ? " is-open" : ""}`}
          onClick={handleCloseAbout}
        >
          <div
            className="landing-about-backdrop"
            data-testid="landing-about-backdrop"
            aria-hidden="true"
          />

          <div
            id="landing-about-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="landing-about-title"
            aria-describedby="landing-about-description"
            className="landing-card landing-about-modal"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="panel-header compact-panel-header landing-about-header">
              <div>
                <p className="eyebrow">About</p>
                <h2 id="landing-about-title">About WISPR</h2>
              </div>
              <button
                ref={aboutCloseButtonRef}
                type="button"
                className="panel-collapse-button"
                aria-label="Close About WISPR"
                onClick={handleCloseAbout}
              >
                ×
              </button>
            </div>

            <p id="landing-about-description" className="landing-about-intro">
              WISPR helps teams study historical wildfire spread through
              machine-learning predictions, interactive map layers, and
              model-aware analysis tools that turn complex fire behavior into
              clearer geospatial insight.
            </p>

            <div className="landing-about-highlights" aria-label="WISPR highlights">
              {ABOUT_HIGHLIGHTS.map((item) => (
                <article key={item.title} className="landing-about-highlight-card">
                  <h3>{item.title}</h3>
                  <p>{item.description}</p>
                </article>
              ))}
            </div>

            <div className="landing-about-flow">
              <div className="landing-about-flow-header">
                <p className="eyebrow">How WISPR Works</p>
                <h3>From wildfire records to interactive prediction layers</h3>
              </div>

              <ol className="landing-about-steps">
                {ABOUT_FLOW_STEPS.map((step, index) => (
                  <li key={step} className="landing-about-step">
                    <span className="landing-about-step-index">
                      {String(index + 1).padStart(2, "0")}
                    </span>
                    <p>{step}</p>
                  </li>
                ))}
              </ol>
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}
