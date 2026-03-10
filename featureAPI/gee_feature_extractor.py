"""
Google Earth Engine Feature Extractor for Wildfire Spread Prediction
Extracts 7 core vegetation/environmental features optimized for fire prediction
"""

import ee
import numpy as np
import rasterio
from typing import Tuple, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import warnings


class GEEFeatureExtractor:
    """
    Extract 7 vegetation/environmental features from Google Earth Engine for wildfire spread prediction.
    
    Features (in order):
    0. VIIRS band M11 (thermal)
    1. VIIRS band I2 (near-IR)
    2. NDVI (vegetation greenness)
    3. EVI2 (enhanced vegetation index 2)
    4. Total precipitation (weather)
    5. Wind speed (weather)
    6. Elevation (static/topographic)
    """
    
    # Feature collection names and band names
    FEATURE_CONFIG = {
        'viirs_m11': {'collection': 'NOAA/VIIRS/001/VNP21A2H', 'band': 'LST'},
        'viirs_i2': {'collection': 'NOAA/VIIRS/001/VNP02DNB', 'band': 'M_CorrectedReflectance'},
        'ndvi': {'collection': 'MODIS/061/MOD13Q1', 'band': 'NDVI'},
        'evi2': {'collection': 'MODIS/061/MOD13Q1', 'band': 'EVI'},
        'precipitation': {'collection': 'UCSB-CHG/CHIRPS/DAILY', 'band': 'precipitation'},
        'wind_speed': {'collection': 'ECMWF/ERA5_LAND/MONTHLY_AGGR', 'band': 'u_component_of_wind_10m'},
        'elevation': {'collection': 'USGS/SRTM90_V4', 'band': 'elevation'},
    }
    
    FEATURE_NAMES = [
        'VIIRS band M11 (thermal)',
        'VIIRS band I2 (near-IR)',
        'NDVI',
        'EVI2',
        'Total precipitation',
        'Wind speed',
        'Elevation'
    ]
    
    def __init__(self, credentials_path: Optional[str] = None, project: Optional[str] = None):
        """
        Initialize GEE extractor.
        
        Args:
            credentials_path: Path to GEE credentials JSON. If None, assumes already authenticated.
            project: Google Cloud project ID for Earth Engine initialization.
        """
        try:
            if credentials_path:
                credentials = ee.ServiceAccountCredentials(None, credentials_path)
                ee.Initialize(credentials=credentials, project=project)
            else:
                ee.Initialize(project=project)
        except Exception as e:
            print(f"GEE initialization warning: {e}")
            print("Ensure you've run: earthengine authenticate --auth_mode=localhost")
            if project is None:
                print("Set a project via EE_PROJECT or pass project='<your-gcp-project-id>'")
    
    def extract_features(
        self,
        latitude: float,
        longitude: float,
        date: Union[str, datetime],
        region_size_meters: int = 1000,
        resolution_meters: int = 250,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract 7 features for a location and date.
        
        Args:
            latitude: Center latitude
            longitude: Center longitude
            date: Date string (YYYY-MM-DD) or datetime object
            region_size_meters: Size of region to extract (default 1km)
            resolution_meters: Output resolution in meters (default 250m = 4x4 pixels)
        
        Returns:
            (features_array, metadata)
            features_array: numpy array of shape (7, H, W) with feature rasters
            metadata: list of feature names
        """
        
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d")
        
        # Create region bounds
        region = self._create_region(latitude, longitude, region_size_meters)
        
        # Extract each feature
        features_list = []
        
        # 1. VIIRS M11 (thermal) - most recent observation
        viirs_m11 = self._extract_viirs_m11(date, region, resolution_meters)
        features_list.append(viirs_m11)
        
        # 2. VIIRS I2 (near-IR) - most recent observation
        viirs_i2 = self._extract_viirs_i2(date, region, resolution_meters)
        features_list.append(viirs_i2)
        
        # 3. NDVI - vegetation greenness (16-day composite)
        ndvi = self._extract_ndvi(date, region, resolution_meters)
        features_list.append(ndvi)
        
        # 4. EVI2 - enhanced vegetation index (16-day composite)
        evi2 = self._extract_evi2(date, region, resolution_meters)
        features_list.append(evi2)
        
        # 5. Total precipitation - last 30 days
        precip = self._extract_precipitation(date, region, resolution_meters)
        features_list.append(precip)
        
        # 6. Wind speed - monthly average
        wind = self._extract_wind_speed(date, region, resolution_meters)
        features_list.append(wind)
        
        # 7. Elevation - static
        elev = self._extract_elevation(region, resolution_meters)
        features_list.append(elev)
        
        # Stack into single array (7, H, W)
        features_array = np.stack(features_list, axis=0)
        
        return features_array, self.FEATURE_NAMES
    
    def _create_region(
        self,
        latitude: float,
        longitude: float,
        size_meters: int
    ) -> ee.Geometry:
        """Create a square region centered at lat/lon."""
        # Convert meters to degrees (approximate at equator)
        size_degrees = size_meters / 111320
        
        coords = [
            [longitude - size_degrees / 2, latitude - size_degrees / 2],
            [longitude + size_degrees / 2, latitude - size_degrees / 2],
            [longitude + size_degrees / 2, latitude + size_degrees / 2],
            [longitude - size_degrees / 2, latitude + size_degrees / 2],
            [longitude - size_degrees / 2, latitude - size_degrees / 2],
        ]
        return ee.Geometry.Polygon([coords])
    
    def _extract_viirs_m11(
        self,
        date: datetime,
        region: ee.Geometry,
        resolution: int
    ) -> np.ndarray:
        """Extract VIIRS thermal band M11."""
        try:
            collection = ee.ImageCollection('NOAA/VIIRS/001/VNP21A2H')
            
            # Filter by date range (±3 days)
            start_date = date - timedelta(days=3)
            end_date = date + timedelta(days=3)
            
            image = collection.filterDate(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            ).filterBounds(region).first()
            
            if image is None:
                warnings.warn("VIIRS M11 data not found, returning zeros")
                return np.zeros((int(1000/resolution), int(1000/resolution)))
            
            return self._image_to_array(image.select('LST'), region, resolution)
        
        except Exception as e:
            warnings.warn(f"Error extracting VIIRS M11: {e}")
            return np.zeros((int(1000/resolution), int(1000/resolution)))
    
    def _extract_viirs_i2(
        self,
        date: datetime,
        region: ee.Geometry,
        resolution: int
    ) -> np.ndarray:
        """Extract VIIRS near-IR band I2."""
        try:
            collection = ee.ImageCollection('NOAA/VIIRS/001/VNP02MOD')
            
            start_date = date - timedelta(days=3)
            end_date = date + timedelta(days=3)
            
            image = collection.filterDate(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            ).filterBounds(region).first()
            
            if image is None:
                warnings.warn("VIIRS I2 data not found, returning zeros")
                return np.zeros((int(1000/resolution), int(1000/resolution)))
            
            return self._image_to_array(image.select('I2'), region, resolution)
        
        except Exception as e:
            warnings.warn(f"Error extracting VIIRS I2: {e}")
            return np.zeros((int(1000/resolution), int(1000/resolution)))
    
    def _extract_ndvi(
        self,
        date: datetime,
        region: ee.Geometry,
        resolution: int
    ) -> np.ndarray:
        """Extract NDVI (16-day composite)."""
        try:
            collection = ee.ImageCollection('MODIS/061/MOD13Q1')
            
            start_date = date - timedelta(days=8)
            end_date = date + timedelta(days=8)
            
            image = collection.filterDate(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            ).filterBounds(region).first()
            
            if image is None:
                warnings.warn("NDVI data not found, returning zeros")
                return np.zeros((int(1000/resolution), int(1000/resolution)))
            
            # NDVI scaled by 0.0001
            ndvi = image.select('NDVI').multiply(0.0001)
            return self._image_to_array(ndvi, region, resolution)
        
        except Exception as e:
            warnings.warn(f"Error extracting NDVI: {e}")
            return np.zeros((int(1000/resolution), int(1000/resolution)))
    
    def _extract_evi2(
        self,
        date: datetime,
        region: ee.Geometry,
        resolution: int
    ) -> np.ndarray:
        """Extract EVI2 (16-day composite)."""
        try:
            collection = ee.ImageCollection('MODIS/061/MOD13Q1')
            
            start_date = date - timedelta(days=8)
            end_date = date + timedelta(days=8)
            
            image = collection.filterDate(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            ).filterBounds(region).first()
            
            if image is None:
                warnings.warn("EVI2 data not found, returning zeros")
                return np.zeros((int(1000/resolution), int(1000/resolution)))
            
            # EVI scaled by 0.0001
            evi = image.select('EVI').multiply(0.0001)
            return self._image_to_array(evi, region, resolution)
        
        except Exception as e:
            warnings.warn(f"Error extracting EVI2: {e}")
            return np.zeros((int(1000/resolution), int(1000/resolution)))
    
    def _extract_precipitation(
        self,
        date: datetime,
        region: ee.Geometry,
        resolution: int
    ) -> np.ndarray:
        """Extract cumulative precipitation (last 30 days)."""
        try:
            collection = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
            
            start_date = date - timedelta(days=30)
            end_date = date
            
            # Sum precipitation over 30 days
            precip = collection.filterDate(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            ).filterBounds(region).select('precipitation').sum()
            
            return self._image_to_array(precip, region, resolution)
        
        except Exception as e:
            warnings.warn(f"Error extracting precipitation: {e}")
            return np.zeros((int(1000/resolution), int(1000/resolution)))
    
    def _extract_wind_speed(
        self,
        date: datetime,
        region: ee.Geometry,
        resolution: int
    ) -> np.ndarray:
        """Extract wind speed (monthly average)."""
        try:
            collection = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
            
            # Get month-start date
            month_start = date.replace(day=1)
            month_end = (month_start + timedelta(days=32)).replace(day=1)
            
            image = collection.filterDate(
                month_start.strftime('%Y-%m-%d'),
                month_end.strftime('%Y-%m-%d')
            ).filterBounds(region).first()
            
            if image is None:
                warnings.warn("Wind speed data not found, returning zeros")
                return np.zeros((int(1000/resolution), int(1000/resolution)))
            
            # Convert u component to wind speed magnitude (simplified)
            wind_speed = image.select('u_component_of_wind_10m').abs()
            return self._image_to_array(wind_speed, region, resolution)
        
        except Exception as e:
            warnings.warn(f"Error extracting wind speed: {e}")
            return np.zeros((int(1000/resolution), int(1000/resolution)))
    
    def _extract_elevation(
        self,
        region: ee.Geometry,
        resolution: int
    ) -> np.ndarray:
        """Extract elevation (static SRTM)."""
        try:
            dem = ee.Image('USGS/SRTM90_V4')
            elevation = dem.select('elevation')
            return self._image_to_array(elevation, region, resolution)
        
        except Exception as e:
            warnings.warn(f"Error extracting elevation: {e}")
            return np.zeros((int(1000/resolution), int(1000/resolution)))
    
    def _image_to_array(
        self,
        image: ee.Image,
        region: ee.Geometry,
        resolution: int
    ) -> np.ndarray:
        """
        Convert GEE Image to numpy array via getDownloadURL.
        
        Args:
            image: ee.Image object
            region: ee.Geometry region
            resolution: Resolution in meters
        
        Returns:
            numpy array
        """
        try:
            url = image.getDownloadUrl({
                'scale': resolution,
                'region': region,
                'fileFormat': 'GeoTIFF'
            })
            
            # Download the GeoTIFF
            import urllib.request
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                urllib.request.urlretrieve(url['url'], tmp.name)
                
                # Read with rasterio
                with rasterio.open(tmp.name) as src:
                    data = src.read(1)
            
            # Clean up
            Path(tmp.name).unlink()
            
            return data
        
        except Exception as e:
            warnings.warn(f"Error converting image to array: {e}")
            size = int(1000 / resolution)
            return np.zeros((size, size))


def normalize_features(
    features: np.ndarray,
    means: Optional[np.ndarray] = None,
    stds: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Normalize 7-feature array to match model input distribution.
    
    Args:
        features: (7, H, W) array
        means: (7,) array of means (computed from training data)
        stds: (7,) array of standard deviations
    
    Returns:
        Normalized features array
    """
    if means is None:
        # Default reasonable values (can be tuned)
        means = np.array([300.0, 0.3, 0.4, 0.3, 100.0, 5.0, 1000.0])
    
    if stds is None:
        stds = np.array([50.0, 0.1, 0.2, 0.2, 50.0, 3.0, 500.0])
    
    # Reshape for broadcasting
    means = means[:, np.newaxis, np.newaxis]
    stds = stds[:, np.newaxis, np.newaxis]
    
    return (features - means) / (stds + 1e-8)
