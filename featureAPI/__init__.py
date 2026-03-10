"""
FeatureAPI: Google Earth Engine Feature Extractor for Wildfire Spread Prediction

Fast extraction of 7 significant vegetation/environmental features from GEE.

Main classes:
- GEEFeatureExtractor: Extract features from Earth Engine
- normalize_features: Standardize features for model input
"""

from .gee_feature_extractor import GEEFeatureExtractor, normalize_features

__version__ = "1.0.0"
__author__ = "Wildfire Spread Prediction Team"

__all__ = [
    'GEEFeatureExtractor',
    'normalize_features',
]
