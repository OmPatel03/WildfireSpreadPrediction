from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .domain import SpreadPrediction


def build_geojson(prediction: SpreadPrediction) -> Dict[str, Any]:
    probs = prediction.probabilities
    mask = prediction.mask
    gt = prediction.ground_truth

    mask_bool = mask.astype(bool)
    gt_bool = gt.astype(bool)
    true_positive = int(np.logical_and(mask_bool, gt_bool).sum())
    false_positive = int(np.logical_and(mask_bool, np.logical_not(gt_bool)).sum())
    false_negative = int(np.logical_and(np.logical_not(mask_bool), gt_bool).sum())
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
            "summary": {
                "meanProbability": float(np.mean(probs)),
                "maxProbability": float(np.max(probs)),
                "minProbability": float(np.min(probs)),
                "positivePixels": int(np.sum(mask)),
                "groundTruthPixels": int(np.sum(gt)),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
        },
    }
    return {"type": "FeatureCollection", "features": [feature]}
