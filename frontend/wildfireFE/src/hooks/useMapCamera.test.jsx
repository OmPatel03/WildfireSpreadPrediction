import { render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import useMapCamera from "./useMapCamera";
import {
  createLayersResponse,
  selectedFire,
} from "../test/fixtures/fireData.js";

function CameraHarness(props) {
  useMapCamera(props);
  return null;
}

function createMapRef() {
  return {
    current: {
      resize: vi.fn(),
      easeTo: vi.fn(),
      fitBounds: vi.fn(),
      flyTo: vi.fn(),
    },
  };
}

describe("useMapCamera", () => {
  beforeEach(() => {
    Object.defineProperty(window, "innerWidth", {
      configurable: true,
      writable: true,
      value: 1440,
    });
    Object.defineProperty(window, "innerHeight", {
      configurable: true,
      writable: true,
      value: 900,
    });
  });

  it("recenters to the initial map view when no fire is selected", () => {
    const mapRef = createMapRef();

    render(
      <CameraHarness
        layersResponse={null}
        mapRef={mapRef}
        selectedFire={null}
        viewMode="2d"
      />,
    );

    expect(mapRef.current.resize).toHaveBeenCalledTimes(1);
    expect(mapRef.current.easeTo).toHaveBeenCalledWith({
      center: [-100, 40],
      zoom: 3.4,
      duration: 900,
      bearing: 0,
      pitch: 0,
    });
    expect(mapRef.current.fitBounds).not.toHaveBeenCalled();
    expect(mapRef.current.flyTo).not.toHaveBeenCalled();
  });

  it("fits to the selected fire bbox when there is no matching layer payload", () => {
    const mapRef = createMapRef();

    render(
      <CameraHarness
        layersResponse={null}
        mapRef={mapRef}
        selectedFire={selectedFire}
        viewMode="2d"
      />,
    );

    expect(mapRef.current.fitBounds).toHaveBeenCalledWith(
      [
        [-121, 35],
        [-119, 37],
      ],
      expect.objectContaining({
        maxZoom: 8.3,
        duration: 850,
        bearing: 0,
        pitch: 0,
      }),
    );
  });

  it("fits to feature bounds when the layer payload matches the selected fire", () => {
    const mapRef = createMapRef();
    const layersResponse = createLayersResponse({
      predictionCoords: [
        [-120, 35],
        [-118, 37],
      ],
      groundTruthCoords: [[-119.5, 36.4]],
    });

    render(
      <CameraHarness
        layersResponse={layersResponse}
        mapRef={mapRef}
        selectedFire={selectedFire}
        viewMode="2d"
      />,
    );

    expect(mapRef.current.fitBounds).toHaveBeenCalledWith(
      [
        [-120.7, 34.3],
        [-117.3, 37.7],
      ],
      expect.objectContaining({
        maxZoom: 12,
        duration: 850,
        bearing: 0,
        pitch: 0,
      }),
    );
  });

  it("ignores stale layer data for a different fire", () => {
    const mapRef = createMapRef();
    const staleLayersResponse = createLayersResponse({
      fireId: "fire-3",
      predictionCoords: [
        [-90, 20],
        [-89, 21],
      ],
    });

    render(
      <CameraHarness
        layersResponse={staleLayersResponse}
        mapRef={mapRef}
        selectedFire={selectedFire}
        viewMode="2d"
      />,
    );

    expect(mapRef.current.fitBounds).toHaveBeenCalledWith(
      [
        [-121, 35],
        [-119, 37],
      ],
      expect.objectContaining({
        maxZoom: 8.3,
      }),
    );
  });

  it("preserves 3d camera options when fitting the fire extent", () => {
    const mapRef = createMapRef();

    render(
      <CameraHarness
        layersResponse={null}
        mapRef={mapRef}
        selectedFire={selectedFire}
        viewMode="3d"
      />,
    );

    expect(mapRef.current.fitBounds).toHaveBeenCalledWith(
      [
        [-121, 35],
        [-119, 37],
      ],
      expect.objectContaining({
        maxZoom: 8.3,
        duration: 1000,
        bearing: -20,
        pitch: 55,
      }),
    );
  });

  it("flies to the fire coordinates when there is no usable bbox or layer extent", () => {
    const mapRef = createMapRef();
    const fireWithoutBbox = {
      fireId: "fire-9",
      latitude: 41.2,
      longitude: -109.7,
      bbox: {
        minLon: null,
        minLat: null,
        maxLon: null,
        maxLat: null,
      },
    };

    render(
      <CameraHarness
        layersResponse={null}
        mapRef={mapRef}
        selectedFire={fireWithoutBbox}
        viewMode="2d"
      />,
    );

    expect(mapRef.current.fitBounds).not.toHaveBeenCalled();
    expect(mapRef.current.flyTo).toHaveBeenCalledWith({
      center: [-109.7, 41.2],
      zoom: 8.2,
      duration: 850,
      bearing: 0,
      pitch: 0,
    });
  });
});
