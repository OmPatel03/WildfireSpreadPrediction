from __future__ import annotations

from math import cos, pi
from typing import Any, Dict

import numpy as np

from .domain import SpreadPrediction


LAT_STEP = 1 / 296
COLUMN_WIDTH_METERS = 375
MODEL_INPUT_CHANNELS = {
    "viirs_m11": {"label": "VIIRS M11"},
    "viirs_i2": {"label": "VIIRS I2"},
    "ndvi": {"label": "NDVI"},
    "evi2": {"label": "EVI2"},
    "precip": {"label": "Precipitation"},
    "wind_speed": {"label": "Wind speed"},
    "elevation": {"label": "Elevation"},
    "slope": {"label": "Slope"},
    "aspect": {"label": "Aspect"},
}


def _safe_longitude_step(latitude: float) -> float:
    safe_cos = max(abs(cos((latitude * pi) / 180.0)), 1e-6)
    return COLUMN_WIDTH_METERS / (111000 * safe_cos)


def _cell_center(
    rows: int, cols: int, center_lat: float, center_long: float, row: int, col: int
) -> tuple[float, float]:
    long_step = _safe_longitude_step(center_lat)
    zero_lat = center_lat - (rows // 2) * LAT_STEP - (LAT_STEP / 2) * (rows % 2)
    zero_long = center_long - (cols // 2) * long_step - (long_step / 2) * (cols % 2)
    lat = zero_lat + (rows - row - 1) * LAT_STEP
    lon = zero_long + col * long_step
    return lat, lon


def _probability_to_magnitude(probability: float, max_probability: float = 1.0) -> float:
    noise_floor = 2.5e-5
    if not np.isfinite(probability) or probability <= noise_floor:
        return 0.0
    safe_max = max(max_probability, 1e-6)
    normalized = min(1.0, max(0.0, float(probability) / safe_max))
    magnitude = np.sqrt(normalized) * 6.0
    return float(min(6.0, max(0.0, magnitude)))


def build_prediction_summary(prediction: SpreadPrediction) -> Dict[str, Any]:
    probs = prediction.probabilities
    mask = prediction.mask
    gt = prediction.ground_truth

    mask_bool = mask.astype(bool)
    gt_bool = gt.astype(bool)
    true_positive = int(np.logical_and(mask_bool, gt_bool).sum())
    false_positive = int(np.logical_and(mask_bool, np.logical_not(gt_bool)).sum())
    false_negative = int(np.logical_and(np.logical_not(mask_bool), gt_bool).sum())
    true_negative = int(
        np.logical_and(np.logical_not(mask_bool), np.logical_not(gt_bool)).sum()
    )
    precision = (
        float(true_positive) / float(true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )
    recall = (
        float(true_positive) / float(true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    total_pixels = int(mask.size)
    accuracy = (
        float(true_positive + true_negative) / float(total_pixels)
        if total_pixels > 0
        else 0.0
    )

    return {
        "meanProbability": float(np.mean(probs)),
        "maxProbability": float(np.max(probs)),
        "minProbability": float(np.min(probs)),
        "positivePixels": int(np.sum(mask)),
        "groundTruthPixels": int(np.sum(gt)),
        "totalPixels": total_pixels,
        "truePositive": true_positive,
        "falsePositive": false_positive,
        "falseNegative": false_negative,
        "trueNegative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def build_layer_collection(prediction: SpreadPrediction) -> Dict[str, Any]:
    probs = prediction.probabilities
    mask = prediction.mask
    gt = prediction.ground_truth
    rows, cols = probs.shape
    center_lat = float(prediction.metadata.latitude)
    center_long = float(prediction.metadata.longitude)
    long_step = _safe_longitude_step(center_lat)
    half_lat = LAT_STEP / 2
    half_long = long_step / 2
    max_probability = float(np.max(probs)) if probs.size else 1.0

    prediction_points = []
    prediction_polygons = []
    ground_truth_points = []
    difference_points = []

    for row in range(rows):
        for col in range(cols):
            probability = float(probs[row, col])
            pred_positive = bool(mask[row, col])
            gt_positive = bool(gt[row, col])
            lat, lon = _cell_center(rows, cols, center_lat, center_long, row, col)
            magnitude = _probability_to_magnitude(probability, max_probability)

            if pred_positive:
                prediction_points.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [lon, lat]},
                        "properties": {
                            "lat": lat,
                            "lon": lon,
                            "probability": probability,
                            "mag": magnitude,
                            "isPositive": pred_positive,
                        },
                    }
                )

            if pred_positive:
                polygon = [
                    [
                        [lon - half_long, lat - half_lat],
                        [lon + half_long, lat - half_lat],
                        [lon + half_long, lat + half_lat],
                        [lon - half_long, lat + half_lat],
                        [lon - half_long, lat - half_lat],
                    ]
                ]
                prediction_polygons.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": polygon},
                        "properties": {
                            "lat": lat,
                            "lon": lon,
                            "probability": probability,
                            "mag": magnitude,
                            "height": max(probability, 0.05) * 2400,
                        },
                    }
                )

            if gt_positive:
                ground_truth_points.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [lon, lat]},
                        "properties": {
                            "lat": lat,
                            "lon": lon,
                            "mag": 1,
                            "probability": 1.0,
                            "isPositive": True,
                        },
                    }
                )

            if pred_positive or gt_positive:
                outcome = (
                    "true_positive"
                    if pred_positive and gt_positive
                    else "false_positive"
                    if pred_positive
                    else "false_negative"
                )
                difference_points.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [lon, lat]},
                        "properties": {
                            "lat": lat,
                            "lon": lon,
                            "mag": 1,
                            "outcome": outcome,
                            "probability": probability,
                        },
                    }
                )

    min_lon, min_lat, max_lon, max_lat = prediction.metadata.bbox
    extent = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [min_lon, min_lat],
                            [max_lon, min_lat],
                            [max_lon, max_lat],
                            [min_lon, max_lat],
                            [min_lon, min_lat],
                        ]
                    ],
                },
                "properties": {
                    "fireId": prediction.metadata.fire_id,
                },
            }
        ],
    }

    origin = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [center_long, center_lat],
                },
                "properties": {
                    "fireId": prediction.metadata.fire_id,
                    "lat": center_lat,
                    "lon": center_long,
                },
            }
        ],
    }

    return {
        "predictionHeatmap": {
            "type": "FeatureCollection",
            "features": prediction_points,
        },
        "predictionPolygons": {
            "type": "FeatureCollection",
            "features": prediction_polygons,
        },
        "groundTruthHeatmap": {
            "type": "FeatureCollection",
            "features": ground_truth_points,
        },
        "differenceHeatmap": {
            "type": "FeatureCollection",
            "features": difference_points,
        },
        "extent": extent,
        "origin": origin,
        "modelInputs": {},
    }


def build_model_input_layer_collection(
    prediction: SpreadPrediction,
    model_inputs: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    collections: Dict[str, Any] = {}

    for key, array in model_inputs.items():
        if key not in MODEL_INPUT_CHANNELS:
            continue

        arr = np.asarray(array, dtype=np.float32)
        if arr.ndim != 2:
            continue

        finite_mask = np.isfinite(arr)
        if not finite_mask.any():
            collections[key] = {
                "label": MODEL_INPUT_CHANNELS[key]["label"],
                "height": int(arr.shape[0]),
                "width": int(arr.shape[1]),
                "raster": arr.tolist(),
                "min": None,
                "max": None,
                "mean": None,
            }
            continue

        finite_values = arr[finite_mask]
        collections[key] = {
            "label": MODEL_INPUT_CHANNELS[key]["label"],
            "height": int(arr.shape[0]),
            "width": int(arr.shape[1]),
            "raster": arr.tolist(),
            "min": float(np.min(finite_values)),
            "max": float(np.max(finite_values)),
            "mean": float(np.mean(finite_values)),
        }

    return collections


def build_geojson(prediction: SpreadPrediction) -> Dict[str, Any]:
    probs = prediction.probabilities
    mask = prediction.mask
    gt = prediction.ground_truth
    summary = build_prediction_summary(prediction)

    feature = {
        "type": "Feature",
        "id": prediction.metadata.fire_id,
        "geometry": {
            "type": "Point",
            "coordinates": [
                float(prediction.metadata.longitude),
                float(prediction.metadata.latitude),
            ],
        },
        "properties": {
            "year": prediction.metadata.year,
            "sampleIndex": prediction.sample_index,
            "totalSamples": prediction.total_samples,
            "threshold": prediction.threshold,
            "shape": {"height": int(probs.shape[0]), "width": int(probs.shape[1])},
            "observationDates": list(prediction.observation_dates),
            "targetDate": prediction.target_date,
            "prediction": {
                "mask": mask.astype(np.uint8).tolist(),
                "probabilities": probs.astype(float).tolist(),
            },
            "groundTruthMask": gt.astype(int).tolist(),
            "summary": summary,
        },
    }
    return {"type": "FeatureCollection", "features": [feature]}
