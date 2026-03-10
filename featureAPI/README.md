# FeatureAPI: Google Earth Engine Feature Extractor

Fast extraction of 7 significant vegetation/environmental features from Google Earth Engine for wildfire spread prediction.

## Features Extracted

The extractor pulls the following **7 core features** optimized for fire prediction:

| # | Feature | Source | Temporal | Resolution |
|---|---------|--------|----------|------------|
| 0 | VIIRS band M11 (thermal) | NOAA VIIRS | Daily | 1km |
| 1 | VIIRS band I2 (near-IR) | NOAA VIIRS | Daily | 375m |
| 2 | NDVI (vegetation greenness) | MODIS | 16-day | 250m |
| 3 | EVI2 (enhanced vegetation) | MODIS | 16-day | 250m |
| 4 | Total precipitation | CHIRPS | Daily | 5km |
| 5 | Wind speed | ERA5 Land | Monthly | 11km |
| 6 | Elevation | SRTM 90m DEM | Static | 90m |

## Setup

### 1. Install Google Earth Engine Python SDK

```bash
pip install earthengine-api
```

### 2. Authenticate with Google Earth Engine

```bash
earthengine authenticate
```

This opens a browser window to authenticate and save credentials.

### 3. Install Dependencies

```bash
pip install rasterio numpy matplotlib
```

## Usage

### Basic Example

```python
from gee_feature_extractor import GEEFeatureExtractor, normalize_features
import numpy as np

# Initialize extractor
extractor = GEEFeatureExtractor()

# Extract features for a location and date
latitude, longitude = 39.5, -121.6  # Northern California
date = "2023-08-15"

features, feature_names = extractor.extract_features(
    latitude=latitude,
    longitude=longitude,
    date=date,
    region_size_meters=1000,    # 1 km x 1 km region
    resolution_meters=250        # 250m pixel size (4x4 pixels)
)

# Shape: (7, H, W) where H,W depends on region_size and resolution
print(f"Extracted shape: {features.shape}")

# Normalize for model input
normalized = normalize_features(features)

# Ready to feed into Domain-Adversarial UTAE model!
```

### For Multi-Timestep Prediction

```python
# Extract features for multiple days
dates = ["2023-08-14", "2023-08-15", "2023-08-16"]
temporal_features = []

for date in dates:
    features, _ = extractor.extract_features(latitude, longitude, date)
    temporal_features.append(features)

# Stack into time series: (T, C, H, W)
temporal_array = np.stack(temporal_features, axis=0)
print(f"Temporal array shape: {temporal_array.shape}")  # (3, 7, H, W)

# Normalize across all timesteps
normalized_temporal = normalize_features(temporal_array.reshape(-1, temporal_array.shape[-2], temporal_array.shape[-1]))
normalized_temporal = normalized_temporal.reshape(temporal_array.shape)
```

## Output Format

Features are returned as a **numpy array** of shape `(7, H, W)` where:
- **7** = number of features
- **H, W** = spatial dimensions (depends on `region_size_meters` / `resolution_meters`)

For a 1km × 1km region at 250m resolution: shape = `(7, 4, 4)`

## Customization

### Adjust Normalization

Default normalization uses pre-computed means/stds. To use custom values:

```python
custom_means = np.array([300.0, 0.3, 0.4, 0.3, 100.0, 5.0, 1000.0])
custom_stds = np.array([50.0, 0.1, 0.2, 0.2, 50.0, 3.0, 500.0])

normalized = normalize_features(features, means=custom_means, stds=custom_stds)
```

### Change Region Size

```python
# Larger region: 5 km x 5 km at 500m resolution
features, _ = extractor.extract_features(
    latitude=39.5,
    longitude=-121.6,
    date="2023-08-15",
    region_size_meters=5000,  # 5 km
    resolution_meters=500      # 500 m pixels
)
print(features.shape)  # (7, 10, 10)
```

## Testing

Run the test notebook:

```bash
jupyter notebook test_gee_extractor.ipynb
```

This demonstrates:
1. Feature extraction from a real wildfire region
2. Visualization of spatial patterns
3. Normalization for model input
4. Integration with the Domain-Adversarial UTAE model

## Performance Notes

- **First extraction**: ~30-60 seconds (includes GEE processing)
- **Subsequent extractions**: ~10-20 seconds (cached GEE results)
- Network connection required for GEE communication
- Latitude/longitude must be in range [-90, 90] and [-180, 180]

## Troubleshooting

### "Authentication failed"
```bash
earthengine authenticate
```

### "VIIRS data not found"
Some regions may not have recent VIIRS data. The extractor returns zeros and continues with other features. Check availability on [NOAA VIIRS website](https://www.nesdis.noaa.gov/noaa-data).

### Rate limiting
If extracting many regions, add delays:
```python
import time
for loc in locations:
    features, _ = extractor.extract_features(loc[0], loc[1], date)
    time.sleep(2)  # Wait 2 seconds between requests
```

## Integration with Wildfire Model

Example integration with the backend API:

```python
# featureAPI/wildfire_predictor.py
from gee_feature_extractor import GEEFeatureExtractor, normalize_features
import torch
import sys

class WildfirePredictor:
    def __init__(self, model_checkpoint_path):
        self.extractor = GEEFeatureExtractor()
        
        # Load model
        sys.path.insert(0, '../src/wsts')
        from models import DomainAdversarialUTAELightning
        
        self.model = DomainAdversarialUTAELightning.load_from_checkpoint(
            model_checkpoint_path
        )
        self.model.eval()
    
    def predict(self, latitude, longitude, date):
        """Predict fire spread for a location"""
        # Extract and normalize features
        features, _ = self.extractor.extract_features(latitude, longitude, date)
        normalized = normalize_features(features)
        
        # Convert to torch tensor (B, T, C, H, W)
        x = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(1).float()
        
        # Predict
        with torch.no_grad():
            fire_logits = self.model(x)
            fire_prob = torch.sigmoid(fire_logits).numpy()
        
        return fire_prob[0, 0]  # Return spatial map
```

## References

- [Google Earth Engine Documentation](https://developers.google.com/earth-engine)
- [NOAA VIIRS Products](https://www.nesdis.noaa.gov/noaa-data)
- [MODIS MOD13Q1 - EVI/NDVI](https://lpdaac.usgs.gov/products/mod13q1v061/)
- [CHIRPS Rainfall Data](https://www.chc.ucsb.edu/data/chirps)
- [ERA5 Land Reanalysis](https://www.ecmwf.int/products/access-era5-land)
- [SRTM DEM](https://lpdaac.usgs.gov/products/srtmgl1v003/)

## License

This feature extractor integrates multiple open-source data sources. See individual dataset licenses:
- NOAA/VIIRS: Public domain
- MODIS: Public domain (NASA)
- CHIRPS: CC-BY 4.0
- ERA5: Copernicus Climate Change Service
- SRTM: Public domain (USGS)
