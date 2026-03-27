import { useEffect } from "react";

const THREE_D_BEARING = -20;
const THREE_D_PITCH = 55;
const FIRE_DETAIL_MAX_ZOOM = 12;
const FIRE_LOADING_EXTENT_MAX_ZOOM = 8.3;
const INITIAL_VIEW = {
  longitude: -100,
  latitude: 40,
  zoom: 3.4,
};

export default function useMapCamera({
  mapRef,
  selectedFire,
  layersResponse,
  viewMode,
}) {
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const horizontalPadding = Math.min(
      Math.max(Math.round(window.innerWidth * 0.16), 140),
      240,
    );
    const topPadding = Math.min(
      Math.max(Math.round(window.innerHeight * 0.1), 64),
      110,
    );
    const bottomOverlayPadding = window.innerWidth <= 1100 ? 86 : 108;
    const cameraOptions = viewMode === "3d"
      ? {
          bearing: THREE_D_BEARING,
          pitch: THREE_D_PITCH,
        }
      : {
          bearing: 0,
          pitch: 0,
        };
    const fitPadding = {
      top: topPadding,
      right: horizontalPadding,
      bottom: bottomOverlayPadding,
      left: horizontalPadding,
    };

    if (!selectedFire) {
      map.resize?.();
      map.easeTo?.({
        center: [INITIAL_VIEW.longitude, INITIAL_VIEW.latitude],
        zoom: INITIAL_VIEW.zoom,
        duration: 900,
        ...cameraOptions,
      });
      return;
    }

    const bbox = selectedFire.bbox;
    const activeLayers = layersResponse?.fire?.fireId === selectedFire.fireId
      ? layersResponse.layers
      : null;

    map.resize?.();

    const candidateFeatures = [
      ...(activeLayers?.predictionHeatmap?.features ?? []),
      ...(activeLayers?.groundTruthHeatmap?.features ?? []),
    ];

    if (candidateFeatures.length > 0) {
      let minLat = Infinity;
      let minLon = Infinity;
      let maxLat = -Infinity;
      let maxLon = -Infinity;

      candidateFeatures.forEach((feature) => {
        const [lon, lat] = feature?.geometry?.coordinates ?? [];
        if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;
        minLat = Math.min(minLat, lat);
        minLon = Math.min(minLon, lon);
        maxLat = Math.max(maxLat, lat);
        maxLon = Math.max(maxLon, lon);
      });

      if (
        Number.isFinite(minLat) &&
        Number.isFinite(minLon) &&
        Number.isFinite(maxLat) &&
        Number.isFinite(maxLon)
      ) {
        const latPad = Math.max((maxLat - minLat) * 0.35, 0.01);
        const lonPad = Math.max((maxLon - minLon) * 0.35, 0.01);

        map.fitBounds(
          [
            [minLon - lonPad, minLat - latPad],
            [maxLon + lonPad, maxLat + latPad],
          ],
          {
            padding: fitPadding,
            maxZoom: FIRE_DETAIL_MAX_ZOOM,
            duration: viewMode === "3d" ? 1000 : 850,
            ...cameraOptions,
          },
        );
        return;
      }
    }

    if (
      bbox &&
      Number.isFinite(bbox.minLon) &&
      Number.isFinite(bbox.minLat) &&
      Number.isFinite(bbox.maxLon) &&
      Number.isFinite(bbox.maxLat)
    ) {
      map.fitBounds(
        [
          [bbox.minLon, bbox.minLat],
          [bbox.maxLon, bbox.maxLat],
        ],
        {
          padding: fitPadding,
          maxZoom: FIRE_LOADING_EXTENT_MAX_ZOOM,
          duration: viewMode === "3d" ? 1000 : 850,
          ...cameraOptions,
        },
      );
      return;
    }

    map.flyTo({
      center: [selectedFire.longitude, selectedFire.latitude],
      zoom: Math.min(8.2, FIRE_DETAIL_MAX_ZOOM),
      duration: viewMode === "3d" ? 1000 : 850,
      ...cameraOptions,
    });
  }, [layersResponse, mapRef, selectedFire, viewMode]);
}
