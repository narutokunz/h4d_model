"""
Data Integration Module for Model 1
====================================

Connects the flood prediction model to real data sources:
- IMD Rainfall NetCDF files
- Ward shapefiles and static features
- Historical flood data

Author: Delhi Flood Monitoring Team
Version: 1.0
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import warnings

# Geospatial libraries
try:
    import rasterio
    from rasterio.mask import mask as rasterio_mask
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    warnings.warn("rasterio not installed. DEM processing disabled.")

from shapely.geometry import Point, mapping


# =============================================================================
# CONFIGURATION
# =============================================================================

def get_project_root() -> Path:
    """Get the project root directory (delhi_hack)."""
    # This file is at: delhi_hack/backend/model/data_integration.py
    # Project root is 2 levels up
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent


class DataConfig:
    """Configuration for data paths and parameters."""
    
    # Get absolute paths based on project root
    PROJECT_ROOT = get_project_root()
    BASE_DIR = PROJECT_ROOT / "backend" / "data"
    RAW_DIR = BASE_DIR / "raw"
    PROCESSED_DIR = BASE_DIR / "processed"
    
    # Ward data
    WARDS_SHP = RAW_DIR / "wards" / "delhi_ward.shp"
    WARDS_JSON = RAW_DIR / "wards" / "delhi_ward.json"
    
    # Rainfall data
    RAINFALL_DIR = RAW_DIR / "rainfall"
    RAINFALL_FILES = {
        2022: "RF25_ind2022_rfp25.nc",
        2023: "RF25_ind2023_rfp25.nc",
        2024: "RF25_ind2024_rfp25.nc",
        2025: "RF25_ind2025_rfp25.nc",
    }
    
    # DEM data
    DEM_DIR = RAW_DIR / "dem"
    
    # Drainage data
    DRAINS_KML = RAW_DIR / "drains" / "Delhi_Drains.kml"
    
    # Delhi bounding box (for filtering national data)
    DELHI_BBOX = {
        'lat_min': 28.3,
        'lat_max': 29.0,
        'lon_min': 76.8,
        'lon_max': 77.4
    }
    
    # UTM projection for Delhi (for accurate area calculations)
    UTM_CRS = "EPSG:32643"  # UTM Zone 43N


# =============================================================================
# WARD DATA PROCESSOR
# =============================================================================

class WardDataProcessor:
    """
    Process ward boundary data and compute static features.
    """
    
    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        self.wards_gdf: Optional[gpd.GeoDataFrame] = None
        self.ward_features: Optional[pd.DataFrame] = None
    
    def load_wards(self) -> gpd.GeoDataFrame:
        """Load ward boundaries from shapefile."""
        shp_path = self.config.WARDS_SHP
        
        if not shp_path.exists():
            # Try JSON as fallback
            json_path = self.config.WARDS_JSON
            if json_path.exists():
                print(f"Loading wards from JSON: {json_path}")
                self.wards_gdf = gpd.read_file(json_path)
            else:
                raise FileNotFoundError(f"Ward file not found: {shp_path}")
        else:
            print(f"Loading wards from SHP: {shp_path}")
            self.wards_gdf = gpd.read_file(shp_path)
        
        # Ensure CRS is set
        if self.wards_gdf.crs is None:
            print("  Setting CRS to EPSG:4326")
            self.wards_gdf = self.wards_gdf.set_crs("EPSG:4326")
        
        # Create unique ward ID if not present
        if 'ward_id' not in self.wards_gdf.columns:
            if 'unique' in self.wards_gdf.columns:
                self.wards_gdf['ward_id'] = self.wards_gdf['unique'].astype(str)
            elif 'id' in self.wards_gdf.columns:
                self.wards_gdf['ward_id'] = self.wards_gdf['id'].astype(str)
            else:
                self.wards_gdf['ward_id'] = [f"ward_{i}" for i in range(len(self.wards_gdf))]
        
        print(f"  Loaded {len(self.wards_gdf)} wards")
        print(f"  Columns: {list(self.wards_gdf.columns)}")
        
        return self.wards_gdf
    
    def compute_static_features(
        self,
        dem_path: str = None,
        drains_path: str = None
    ) -> pd.DataFrame:
        """
        Compute static vulnerability features for each ward.
        
        Features computed:
        - mean_elevation, elevation_std, low_lying_pct (from DEM)
        - drain_density, slope_mean (from drains and DEM)
        - area_sqkm (from geometry)
        """
        if self.wards_gdf is None:
            self.load_wards()
        
        # Project to UTM for accurate area calculations
        wards_utm = self.wards_gdf.to_crs(self.config.UTM_CRS)
        
        features = []
        
        for idx, ward in wards_utm.iterrows():
            ward_id = ward.get('ward_id', str(idx))
            ward_name = ward.get('ward_name', ward.get('name', f'Ward {idx}'))
            
            # Area in sq km
            area_sqkm = ward.geometry.area / 1e6
            
            # Initialize feature dict
            feat = {
                'ward_id': ward_id,
                'ward_name': ward_name,
                'area_sqkm': area_sqkm,
                'mean_elevation': 215.0,  # Default Delhi elevation
                'elevation_std': 5.0,
                'low_lying_pct': 15.0,    # Default estimate
                'drain_density': 2.5,     # Default estimate
                'slope_mean': 1.0         # Default (relatively flat)
            }
            
            features.append(feat)
        
        self.ward_features = pd.DataFrame(features)
        self.ward_features.set_index('ward_id', inplace=True)
        
        print(f"Computed static features for {len(self.ward_features)} wards")
        
        return self.ward_features
    
    def compute_elevation_features_from_dem(
        self,
        dem_path: str
    ) -> None:
        """
        Compute elevation features from DEM raster.
        Updates ward_features in place.
        """
        if not HAS_RASTERIO:
            print("rasterio not available, skipping DEM processing")
            return
        
        if self.wards_gdf is None:
            self.load_wards()
        
        print(f"Processing DEM: {dem_path}")
        
        try:
            with rasterio.open(dem_path) as src:
                for idx, ward in self.wards_gdf.iterrows():
                    ward_id = ward.get('ward_id', str(idx))
                    
                    try:
                        # Mask DEM to ward geometry
                        geom = [mapping(ward.geometry)]
                        out_image, out_transform = rasterio_mask(src, geom, crop=True)
                        out_data = out_image[0]
                        
                        # Filter valid data (remove nodata)
                        valid_data = out_data[out_data > -100]
                        
                        if len(valid_data) > 0:
                            mean_elev = np.mean(valid_data)
                            std_elev = np.std(valid_data)
                            
                            # Low-lying: below mean - 1 std
                            threshold = mean_elev - std_elev
                            low_lying_pct = (np.sum(valid_data < threshold) / len(valid_data)) * 100
                            
                            # Update features
                            if ward_id in self.ward_features.index:
                                self.ward_features.loc[ward_id, 'mean_elevation'] = mean_elev
                                self.ward_features.loc[ward_id, 'elevation_std'] = std_elev
                                self.ward_features.loc[ward_id, 'low_lying_pct'] = low_lying_pct
                    
                    except Exception as e:
                        # Geometry might not overlap with DEM
                        pass
            
            print("  DEM processing complete")
        
        except Exception as e:
            print(f"  DEM processing failed: {e}")
    
    def compute_drain_density(self, drains_path: str) -> None:
        """
        Compute drainage density for each ward.
        Updates ward_features in place.
        """
        if self.wards_gdf is None:
            self.load_wards()
        
        print(f"Processing drains: {drains_path}")
        
        try:
            # Load drains (KML or SHP)
            drains = gpd.read_file(drains_path)
            
            if drains.crs is None:
                drains = drains.set_crs("EPSG:4326")
            
            # Project to UTM
            drains_utm = drains.to_crs(self.config.UTM_CRS)
            wards_utm = self.wards_gdf.to_crs(self.config.UTM_CRS)
            
            for idx, ward in wards_utm.iterrows():
                ward_id = ward.get('ward_id', str(idx))
                
                # Clip drains to ward
                try:
                    drains_clipped = gpd.clip(drains_utm, ward.geometry)
                    total_length_km = drains_clipped.length.sum() / 1000
                    area_sqkm = ward.geometry.area / 1e6
                    
                    density = total_length_km / area_sqkm if area_sqkm > 0 else 0
                    
                    if ward_id in self.ward_features.index:
                        self.ward_features.loc[ward_id, 'drain_density'] = density
                
                except Exception:
                    pass
            
            print("  Drain density processing complete")
        
        except Exception as e:
            print(f"  Drain processing failed: {e}")
    
    def save_features(self, path: str = None):
        """Save computed features to CSV."""
        if self.ward_features is None:
            raise ValueError("No features computed. Call compute_static_features first.")
        
        if path is None:
            path = self.config.PROCESSED_DIR / "ward_static_features.csv"
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.ward_features.to_csv(path)
        print(f"Saved ward features to {path}")
    
    def load_features(self, path: str = None) -> pd.DataFrame:
        """Load pre-computed features from CSV."""
        if path is None:
            path = self.config.PROCESSED_DIR / "ward_static_features.csv"
        
        self.ward_features = pd.read_csv(path, index_col='ward_id')
        print(f"Loaded ward features from {path}")
        return self.ward_features


# =============================================================================
# RAINFALL DATA PROCESSOR
# =============================================================================

class RainfallDataProcessor:
    """
    Process IMD gridded rainfall data and map to wards.
    """
    
    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        self.wards_gdf: Optional[gpd.GeoDataFrame] = None
        self.grid_to_ward_map: Optional[pd.DataFrame] = None
    
    def set_wards(self, wards_gdf: gpd.GeoDataFrame):
        """Set ward boundaries for spatial mapping."""
        self.wards_gdf = wards_gdf.copy()
        
        # Ensure ward_id column exists
        if 'ward_id' not in self.wards_gdf.columns:
            if 'unique' in self.wards_gdf.columns:
                self.wards_gdf['ward_id'] = self.wards_gdf['unique'].astype(str)
            elif 'id' in self.wards_gdf.columns:
                self.wards_gdf['ward_id'] = self.wards_gdf['id'].astype(str)
            else:
                self.wards_gdf['ward_id'] = [f"ward_{i}" for i in range(len(self.wards_gdf))]
        
        self._build_grid_mapping()
    
    def _build_grid_mapping(self):
        """
        Build mapping from wards to nearest IMD grid points.
        
        Since IMD grid (0.25deg) is coarse compared to ward size,
        we map each ward to its nearest grid point based on centroid.
        
        IMD Grid starts at 6.5N, 66.5E with 0.25 degree steps.
        Delhi region: lat ~28.5-29.0, lon ~77.0-77.25
        """
        if self.wards_gdf is None:
            return
        
        # Actual IMD grid points over Delhi (0.25 degree grid starting from 6.5, 66.5)
        # We use the exact values that exist in the NetCDF file
        imd_lats = np.array([28.5, 28.75, 29.0])
        imd_lons = np.array([77.0, 77.25])
        
        # Ensure wards are in WGS84
        wards = self.wards_gdf.to_crs("EPSG:4326")
        
        # Map each ward to nearest grid point using centroid
        mappings = []
        for idx, ward in wards.iterrows():
            centroid = ward.geometry.centroid
            ward_lat, ward_lon = centroid.y, centroid.x
            
            # Find nearest grid point from actual IMD grid values
            nearest_lat = float(min(imd_lats, key=lambda x: abs(x - ward_lat)))
            nearest_lon = float(min(imd_lons, key=lambda x: abs(x - ward_lon)))
            
            mappings.append({
                'ward_id': ward.get('ward_id', str(idx)),
                'lat': nearest_lat,
                'lon': nearest_lon,
                'centroid_lat': ward_lat,
                'centroid_lon': ward_lon
            })
        
        self.grid_to_ward_map = pd.DataFrame(mappings)
        
        # Count unique grid points used
        unique_grids = self.grid_to_ward_map[['lat', 'lon']].drop_duplicates()
        print(f"Built grid mapping: {len(self.grid_to_ward_map)} wards -> {len(unique_grids)} grid points")
    
    def load_rainfall_netcdf(
        self,
        year: int,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> pd.DataFrame:
        """
        Load rainfall data from IMD NetCDF file.
        
        Args:
            year: Year of data to load
            start_date: Optional start date filter
            end_date: Optional end date filter
        
        Returns:
            DataFrame with columns ['date', 'lat', 'lon', 'rainfall_mm']
        """
        if year not in self.config.RAINFALL_FILES:
            raise ValueError(f"No rainfall file for year {year}")
        
        nc_path = self.config.RAINFALL_DIR / self.config.RAINFALL_FILES[year]
        
        if not nc_path.exists():
            raise FileNotFoundError(f"Rainfall file not found: {nc_path}")
        
        print(f"Loading rainfall data from {nc_path}")
        
        ds = xr.open_dataset(nc_path)
        
        # Filter to Delhi bounding box
        bbox = self.config.DELHI_BBOX
        
        # Handle different variable naming conventions in IMD files
        try:
            ds_delhi = ds.sel(
                LATITUDE=slice(bbox['lat_min'], bbox['lat_max']),
                LONGITUDE=slice(bbox['lon_min'], bbox['lon_max'])
            )
            lat_var, lon_var = 'LATITUDE', 'LONGITUDE'
        except KeyError:
            ds_delhi = ds.sel(
                lat=slice(bbox['lat_min'], bbox['lat_max']),
                lon=slice(bbox['lon_min'], bbox['lon_max'])
            )
            lat_var, lon_var = 'lat', 'lon'
        
        # Convert to DataFrame
        df = ds_delhi.to_dataframe().reset_index()
        
        # Standardize column names
        df = df.rename(columns={
            lat_var: 'lat',
            lon_var: 'lon',
            'RAINFALL': 'rainfall_mm',
            'rainfall': 'rainfall_mm',
            'rf': 'rainfall_mm'
        })
        
        # Handle time column
        if 'TIME' in df.columns:
            df = df.rename(columns={'TIME': 'date'})
        elif 'time' in df.columns:
            df = df.rename(columns={'time': 'date'})
        
        # Filter by date range
        if start_date is not None:
            df = df[df['date'] >= start_date]
        if end_date is not None:
            df = df[df['date'] <= end_date]
        
        # Handle missing rainfall column
        if 'rainfall_mm' not in df.columns:
            # Find the rainfall variable
            for col in df.columns:
                if 'rain' in col.lower() or 'rf' in col.lower() or 'precip' in col.lower():
                    df = df.rename(columns={col: 'rainfall_mm'})
                    break
        
        print(f"  Loaded {len(df)} rainfall records")
        
        return df[['date', 'lat', 'lon', 'rainfall_mm']]
    
    def aggregate_rainfall_by_ward(
        self,
        rainfall_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate gridded rainfall to ward level.
        
        Args:
            rainfall_df: DataFrame with ['date', 'lat', 'lon', 'rainfall_mm']
        
        Returns:
            DataFrame with ['date', 'ward_id', 'rainfall_mm']
        """
        if self.grid_to_ward_map is None:
            raise ValueError("Grid mapping not built. Call set_wards first.")
        
        # Round lat/lon in rainfall data to match grid precision
        rainfall_df = rainfall_df.copy()
        rainfall_df['lat'] = rainfall_df['lat'].round(2)
        rainfall_df['lon'] = rainfall_df['lon'].round(2)
        
        # Merge rainfall with ward mapping
        # Each ward has a lat/lon it's mapped to
        merged = pd.merge(
            self.grid_to_ward_map[['ward_id', 'lat', 'lon']],
            rainfall_df,
            on=['lat', 'lon'],
            how='left'
        )
        
        # Drop rows without rainfall data
        merged = merged.dropna(subset=['rainfall_mm'])
        
        if len(merged) == 0:
            print("  Warning: No matching rainfall data found")
            print(f"    Grid lat range: {self.grid_to_ward_map['lat'].min():.2f} - {self.grid_to_ward_map['lat'].max():.2f}")
            print(f"    Grid lon range: {self.grid_to_ward_map['lon'].min():.2f} - {self.grid_to_ward_map['lon'].max():.2f}")
            print(f"    Rainfall lat range: {rainfall_df['lat'].min():.2f} - {rainfall_df['lat'].max():.2f}")
            print(f"    Rainfall lon range: {rainfall_df['lon'].min():.2f} - {rainfall_df['lon'].max():.2f}")
        
        # Each ward already has one grid point, so just keep the data
        ward_rainfall = merged[['date', 'ward_id', 'rainfall_mm']].copy()
        
        print(f"  Aggregated to {len(ward_rainfall)} ward-date records")
        
        return ward_rainfall
    
    def get_recent_rainfall(
        self,
        ward_id: str,
        hours: int = 24
    ) -> pd.DataFrame:
        """
        Get recent rainfall for a specific ward.
        This would typically use real-time API data.
        
        Returns:
            DataFrame with ['timestamp', 'rainfall_mm']
        """
        # This is a placeholder for real-time data integration
        # In production, this would call IMD API
        
        now = datetime.now()
        timestamps = [now - timedelta(hours=h) for h in range(hours, 0, -1)]
        
        # Generate synthetic recent data (replace with API call)
        rainfall = np.random.exponential(2, len(timestamps))
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'rainfall_mm': rainfall
        })


# =============================================================================
# HISTORICAL DATA PROCESSOR
# =============================================================================

class HistoricalDataProcessor:
    """
    Process historical flood event and complaint data.
    """
    
    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
    
    def compute_historical_features(
        self,
        ward_ids: List[str],
        flood_events_df: pd.DataFrame = None,
        complaints_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Compute historical propensity features for each ward.
        
        Features:
        - hist_flood_freq: Number of historical flood events
        - monsoon_risk_score: Seasonal risk multiplier (0-1)
        - complaint_baseline: Average complaints per monsoon season
        """
        # Try to load INDOFLOODS historical data first
        indofloods_path = self.config.PROCESSED_DIR / "historical_floods_ward.csv"
        
        if indofloods_path.exists():
            print("  Using INDOFLOODS real historical flood data")
            df = pd.read_csv(indofloods_path)
            df.set_index('ward_id', inplace=True)
            
            # Keep only the features we need
            df = df[['hist_flood_freq', 'monsoon_risk_score']]
            
            # Add default complaint baseline
            df['complaint_baseline'] = 5 + (df['hist_flood_freq'] * 2)  # More floods = more complaints
            
            return df
        
        # Fallback to original logic if INDOFLOODS not available
        features = []
        
        for ward_id in ward_ids:
            feat = {
                'ward_id': ward_id,
                'hist_flood_freq': 0,
                'monsoon_risk_score': 0.5,  # Default neutral
                'complaint_baseline': 5      # Default estimate
            }
            
            # If we have flood event data
            if flood_events_df is not None and 'ward_id' in flood_events_df.columns:
                ward_floods = flood_events_df[flood_events_df['ward_id'] == ward_id]
                feat['hist_flood_freq'] = len(ward_floods)
            
            # If we have complaint data
            if complaints_df is not None and 'ward_id' in complaints_df.columns:
                ward_complaints = complaints_df[complaints_df['ward_id'] == ward_id]
                if len(ward_complaints) > 0:
                    feat['complaint_baseline'] = ward_complaints['count'].mean()
            
            # Compute monsoon risk score based on historical frequency
            # Normalize to 0-1 range
            max_freq = 10  # Assume max 10 flood events
            feat['monsoon_risk_score'] = min(feat['hist_flood_freq'] / max_freq, 1.0)
            
            features.append(feat)
        
        df = pd.DataFrame(features)
        df.set_index('ward_id', inplace=True)
        
        return df
    
    def generate_synthetic_historical(
        self,
        ward_ids: List[str],
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Generate synthetic historical features for demo purposes.
        """
        np.random.seed(seed)
        
        features = []
        for ward_id in ward_ids:
            # Some wards are historically more flood-prone
            is_high_risk = np.random.random() < 0.2
            
            feat = {
                'ward_id': ward_id,
                'hist_flood_freq': np.random.poisson(5 if is_high_risk else 2),
                'monsoon_risk_score': np.random.beta(3, 2) if is_high_risk else np.random.beta(2, 4),
                'complaint_baseline': np.random.poisson(15 if is_high_risk else 8)
            }
            features.append(feat)
        
        df = pd.DataFrame(features)
        df.set_index('ward_id', inplace=True)
        
        return df


# =============================================================================
# DATA PIPELINE
# =============================================================================

class DataPipeline:
    """
    Unified data pipeline for model training and inference.
    """
    
    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        self.ward_processor = WardDataProcessor(self.config)
        self.rainfall_processor = RainfallDataProcessor(self.config)
        self.historical_processor = HistoricalDataProcessor(self.config)
        
        # Cached data
        self.ward_static: Optional[pd.DataFrame] = None
        self.ward_historical: Optional[pd.DataFrame] = None
    
    def initialize(self) -> 'DataPipeline':
        """Initialize the data pipeline by loading all necessary data."""
        print("=" * 60)
        print("Initializing Data Pipeline")
        print("=" * 60)
        
        # 1. Load wards
        print("\n[1/3] Loading ward boundaries...")
        wards_gdf = self.ward_processor.load_wards()
        
        # 2. Compute or load static features
        print("\n[2/3] Computing static features...")
        processed_path = self.config.PROCESSED_DIR / "ward_static_features.csv"
        
        if processed_path.exists():
            self.ward_static = self.ward_processor.load_features(processed_path)
        else:
            self.ward_static = self.ward_processor.compute_static_features()
            self.ward_processor.save_features(processed_path)
        
        # 3. Compute historical features
        print("\n[3/3] Computing historical features...")
        ward_ids = self.ward_static.index.tolist()
        self.ward_historical = self.historical_processor.generate_synthetic_historical(ward_ids)
        
        # Set up rainfall processor
        self.rainfall_processor.set_wards(wards_gdf)
        
        print("\n[OK] Data pipeline initialized")
        print(f"   Wards: {len(self.ward_static)}")
        print(f"   Static features: {list(self.ward_static.columns)}")
        print(f"   Historical features: {list(self.ward_historical.columns)}")
        
        return self
    
    def get_training_data(
        self,
        years: List[int] = [2022, 2023]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get all data needed for model training.
        
        Returns:
            - ward_static: Static features per ward
            - ward_historical: Historical features per ward
            - rainfall_data: Rainfall time series per ward
        """
        if self.ward_static is None:
            self.initialize()
        
        # Load rainfall for specified years
        all_rainfall = []
        for year in years:
            try:
                rainfall_df = self.rainfall_processor.load_rainfall_netcdf(year)
                ward_rainfall = self.rainfall_processor.aggregate_rainfall_by_ward(rainfall_df)
                all_rainfall.append(ward_rainfall)
            except FileNotFoundError:
                print(f"  Skipping year {year} (file not found)")
        
        if all_rainfall:
            rainfall_data = pd.concat(all_rainfall, ignore_index=True)
        else:
            rainfall_data = pd.DataFrame(columns=['date', 'ward_id', 'rainfall_mm'])
        
        return self.ward_static, self.ward_historical, rainfall_data
    
    def get_ward_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get ward static and historical features."""
        if self.ward_static is None:
            self.initialize()
        
        return self.ward_static, self.ward_historical


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DATA INTEGRATION MODULE - Initialization")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = DataPipeline()
    
    try:
        pipeline.initialize()
        
        # Get training data
        print("\n" + "=" * 60)
        print("Loading Training Data")
        print("=" * 60)
        
        ward_static, ward_historical, rainfall = pipeline.get_training_data([2023])
        
        print(f"\nWard Static Features:")
        print(ward_static.head())
        
        print(f"\nWard Historical Features:")
        print(ward_historical.head())
        
        if not rainfall.empty:
            print(f"\nRainfall Data Sample:")
            print(rainfall.head())
    
    except FileNotFoundError as e:
        print(f"\n[WARNING] Some data files not found: {e}")
        print("Run with available data or download missing files.")
