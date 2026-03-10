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
    if (!layerRef.current) {
      layerRef.current = L.heatLayer(points, {
        radius,
        blur,
        maxZoom,
        minOpacity,
        gradient,
      });
      layerRef.current.addTo(map);
      return undefined;
    }

    layerRef.current.setLatLngs(points);
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