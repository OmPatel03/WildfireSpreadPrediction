import L from "leaflet";
import "leaflet.heat";
import { useEffect, useRef } from "react";
import { useMap } from "react-leaflet";

export default function HeatmapLayer({
  points,
  radius = 18,
  blur = 22,
  maxZoom = 12,
  minOpacity = 0.35,
  gradient,
}) {
  const map = useMap();
  const layerRef = useRef(null);

  useEffect(() => {
    const sanitizedPoints = (points ?? []).filter((point) => {
      if (!Array.isArray(point) || point.length < 2) return false;
      const lat = Number(point[0]);
      const lon = Number(point[1]);
      const intensity = point.length >= 3 ? Number(point[2]) : 1;
      return (
        Number.isFinite(lat) &&
        Number.isFinite(lon) &&
        Number.isFinite(intensity) &&
        intensity > 0
      );
    });

    if (layerRef.current) {
      map.removeLayer(layerRef.current);
      layerRef.current = null;
    }

    layerRef.current = L.heatLayer(sanitizedPoints, {
      radius,
      blur,
      maxZoom,
      minOpacity,
      gradient,
    });
    layerRef.current.addTo(map);

    // The heatmap canvas should never block clicks on fire markers underneath.
    const heatmapCanvas = layerRef.current?._canvas;
    if (heatmapCanvas?.style) {
      heatmapCanvas.style.pointerEvents = "none";
    }

    return undefined;
  }, [blur, gradient, map, maxZoom, minOpacity, points, radius]);

  useEffect(() => {
    return () => {
      if (layerRef.current) {
        map.removeLayer(layerRef.current);
        layerRef.current = null;
      }
    };
  }, [map]);

  return null;
}
