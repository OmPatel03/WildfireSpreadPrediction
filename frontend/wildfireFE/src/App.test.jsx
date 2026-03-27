import { act, fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App.jsx";

vi.mock("./components/MapView", () => ({
  default: () => <div>Map View</div>,
}));

vi.mock("./components/MapAtmosphere", () => ({
  default: () => <div>Map Atmosphere</div>,
}));

vi.mock("./components/MapHud", () => ({
  default: ({ className, cardClassName }) => (
    <div data-testid="map-hud" className={className}>
      <div data-testid="map-hud-card" className={cardClassName}>Map HUD</div>
    </div>
  ),
}));

vi.mock("./components/IncidentsPanel", () => ({
  default: ({ className }) => <div data-testid="incidents-panel" className={className}>Incidents Panel</div>,
}));

vi.mock("./components/ModelInputsPanel", () => ({
  default: ({ className }) => <div data-testid="model-inputs-panel" className={className}>Model Inputs Panel</div>,
}));

vi.mock("./components/EnvironmentPanel", () => ({
  default: ({ className }) => <div data-testid="environment-panel" className={className}>Environment Panel</div>,
}));

vi.mock("./components/TimelineDock", () => ({
  default: ({ className }) => <div data-testid="timeline-dock" className={className}>Timeline Dock</div>,
}));

vi.mock("./components/TopControls", () => ({
  default: ({ className, deckClassName, stackClassName, onBackToIntro }) => (
    <div data-testid="top-controls" className={className}>
      <div data-testid="top-controls-deck" className={deckClassName}>Deck</div>
      <div data-testid="top-controls-stack" className={stackClassName}>Stack</div>
      <button type="button" onClick={onBackToIntro}>
        Back to intro
      </button>
    </div>
  ),
}));

vi.mock("./hooks/useAvailableYears", () => ({
  default: () => ({
    yearOptions: [2021],
    resolvedInitialYear: 2021,
  }),
}));

vi.mock("./hooks/useCatalogData", () => ({
  default: () => ({
    catalog: [],
    loading: false,
    error: null,
  }),
}));

vi.mock("./hooks/useFireTimeline", () => ({
  default: () => ({
    timeline: null,
    loading: false,
    error: null,
  }),
}));

vi.mock("./hooks/useFireLayers", () => ({
  default: () => ({
    layersResponse: null,
    loading: false,
    error: null,
  }),
}));

vi.mock("./hooks/useMapCamera", () => ({
  default: () => {},
}));

describe("App landing flow", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    window.sessionStorage.clear();
  });

  it("shows the landing page on first load and hides app overlays", () => {
    render(<App />);

    expect(
      screen.getByRole("heading", { name: "WISPR" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Enter WISPR" })).toBeInTheDocument();
    expect(screen.queryByText("Top Controls")).not.toBeInTheDocument();
  });

  it("enters the app and persists the session flag", () => {
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "Enter WISPR" }));

    expect(window.sessionStorage.getItem("wispr:entered-app")).toBe("true");
    expect(screen.getByLabelText("WISPR landing page")).toHaveClass("landing-screen-exit-to-app");
    expect(screen.getByTestId("top-controls-deck")).toHaveClass("app-overlay-enter-from-landing");
    expect(screen.getByTestId("top-controls-stack")).toHaveClass("app-overlay-enter-from-landing");
    expect(screen.getByTestId("top-controls")).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(650);
    });

    expect(screen.queryByRole("heading", { name: "WISPR" })).not.toBeInTheDocument();
  });

  it("restores the landing page when returning to intro", () => {
    window.sessionStorage.setItem("wispr:entered-app", "true");
    render(<App />);

    expect(screen.getByTestId("top-controls")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Back to intro" }));

    expect(window.sessionStorage.getItem("wispr:entered-app")).toBeNull();
    expect(screen.getByLabelText("WISPR landing page")).toHaveClass("landing-screen-enter-from-app");
    expect(screen.getByTestId("top-controls-deck")).toHaveClass("app-overlay-exit-to-landing");
    expect(screen.getByTestId("top-controls-stack")).toHaveClass("app-overlay-exit-to-landing");
    expect(screen.getByTestId("top-controls")).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(650);
    });

    expect(screen.getByRole("heading", { name: "WISPR" })).toBeInTheDocument();
    expect(screen.queryByTestId("top-controls")).not.toBeInTheDocument();
  });
});
