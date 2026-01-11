"""
Data Preparation Script for Rainfall Threshold Model

This script downloads and processes IMD rainfall data (2020-2024) and prepares
the training dataset format for the threshold model.

Note: This script processes rainfall data. You will need to combine it with
waterlogging labels (waterlog_level) from your data sources to create the final
training dataset.
"""

import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from pathlib import Path
import sys
from typing import Optional, List
from datetime import datetime
import warnings

# Add backend to path to use existing utilities
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))

try:
    from model.ingest_rainfall import process_imd_rainfall
except ImportError:
    print("Warning: Could not import ingest_rainfall. Will use local implementation.")


class ThresholdDataPreparer:
    """
    Prepares training data for Rainfall Threshold Model from IMD NetCDF files.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize data preparer.
        
        Args:
            data_dir: Base data directory (default: uses relative paths)
        """
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent / 'backend' / 'data'
        else:
            self.data_dir = Path(data_dir)
        
        self.raw_dir = self.data_dir / 'raw' / 'rainfall'
        self.processed_dir = self.data_dir / 'processed'
        self.output_dir = Path(__file__).parent / 'data'
        self.output_dir.mkdir(exist_ok=True)
        
        # Delhi bounding box
        self.delhi_bbox = {
            'lat_min': 28.3,
            'lat_max': 29.0,
            'lon_min': 76.8,
            'lon_max': 77.4
        }
        
        # IMD NetCDF base URLs
        self.imd_base_url_25 = "https://www.imdpune.gov.in/Clim_Pred_LRF_New/Grided_Data_Download/25_deg/Rainfall/"
        self.imd_base_url_1 = "https://www.imdpune.gov.in/Clim_Pred_LRF_New/Grided_Data_Download/1_deg/Rainfall/"
    
    def download_imd_rainfall(self, year: int, resolution: str = '0.25') -> Optional[Path]:
        """
        Download IMD rainfall NetCDF file for a year.
        
        Note: IMD files must be downloaded manually from:
        - 0.25 deg: https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html
        - 1.0 deg: https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_1_NetCDF.html
        
        This function checks if file exists and provides download instructions.
        
        Args:
            year: Year to download
            resolution: '0.25' or '1.0' degree resolution
            
        Returns:
            Path to downloaded file if exists, None otherwise
        """
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Expected filename format: RF25_indYYYY_rfp25.nc (for 0.25 deg)
        if resolution == '0.25':
            filename = f"RF25_ind{year}_rfp25.nc"
        else:
            filename = f"RF1_ind{year}_rfp1.nc"
        
        file_path = self.raw_dir / filename
        
        if file_path.exists():
            print(f"✓ File exists: {file_path}")
            return file_path
        else:
            print(f"✗ File not found: {file_path}")
            if resolution == '0.25':
                url = f"{self.imd_base_url_25}{filename}"
                page_url = "https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html"
            else:
                url = f"{self.imd_base_url_1}{filename}"
                page_url = "https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_1_NetCDF.html"
            
            print(f"  Please download manually from: {page_url}")
            print(f"  Expected filename: {filename}")
            print(f"  Save to: {self.raw_dir}")
            return None
    
    def process_imd_netcdf(
        self,
        nc_file_path: Path,
        wards_shapefile_path: Path,
        year: int
    ) -> pd.DataFrame:
        """
        Process IMD NetCDF file and extract ward-level rainfall.
        
        Args:
            nc_file_path: Path to NetCDF file
            wards_shapefile_path: Path to wards shapefile
            year: Year being processed
            
        Returns:
            DataFrame with columns: ward_id, date, rainfall_mm (daily)
        """
        print(f"\nProcessing {year} rainfall data...")
        print(f"  NetCDF: {nc_file_path}")
        print(f"  Wards: {wards_shapefile_path}")
        
        # Load NetCDF
        try:
            ds = xr.open_dataset(nc_file_path)
        except Exception as e:
            raise FileNotFoundError(f"Error opening NetCDF file: {e}")
        
        # Load wards
        try:
            wards = gpd.read_file(wards_shapefile_path)
            if wards.crs is None:
                wards = wards.set_crs("EPSG:4326")
        except Exception as e:
            raise FileNotFoundError(f"Error loading wards shapefile: {e}")
        
        # Ensure ward_id column exists
        if 'ward_id' not in wards.columns:
            if 'unique' in wards.columns:
                wards['ward_id'] = wards['unique'].astype(str)
            elif 'id' in wards.columns:
                wards['ward_id'] = wards['id'].astype(str)
            else:
                raise ValueError("Wards shapefile must have 'ward_id', 'unique', or 'id' column")
        
        # Filter to Delhi bounding box
        try:
            delhi_bbox = {
                'LATITUDE': slice(self.delhi_bbox['lat_min'], self.delhi_bbox['lat_max']),
                'LONGITUDE': slice(self.delhi_bbox['lon_min'], self.delhi_bbox['lon_max'])
            }
            ds_delhi = ds.sel(**delhi_bbox)
            lat_col, lon_col = 'LATITUDE', 'LONGITUDE'
            time_col = 'TIME'
        except KeyError:
            try:
                delhi_bbox = {
                    'lat': slice(self.delhi_bbox['lat_min'], self.delhi_bbox['lat_max']),
                    'lon': slice(self.delhi_bbox['lon_min'], self.delhi_bbox['lon_max'])
                }
                ds_delhi = ds.sel(**delhi_bbox)
                lat_col, lon_col = 'lat', 'lon'
                time_col = 'time'
            except KeyError:
                raise ValueError("Could not identify lat/lon/time columns in NetCDF")
        
        # Convert to DataFrame
        df = ds_delhi.to_dataframe().reset_index()
        
        # Standardize column names
        col_mapping = {
            lat_col: 'lat',
            lon_col: 'lon',
            'RAINFALL': 'rainfall',
            'rainfall': 'rainfall',
            'rf': 'rainfall'
        }
        df = df.rename(columns=col_mapping)
        
        if time_col not in df.columns:
            # Try to find time dimension
            time_dims = [col for col in df.columns if 'time' in col.lower()]
            if time_dims:
                time_col = time_dims[0]
            else:
                raise ValueError("Could not find time dimension")
        
        df['date'] = pd.to_datetime(df[time_col])
        
        # Create grid point GeoDataFrame
        unique_grid = df[['lat', 'lon']].drop_duplicates()
        from shapely.geometry import Point
        geometry = [Point(lon, lat) for lon, lat in zip(unique_grid['lon'], unique_grid['lat'])]
        gdf_grid = gpd.GeoDataFrame(unique_grid, geometry=geometry, crs="EPSG:4326")
        
        # Ensure wards are in EPSG:4326
        if wards.crs != "EPSG:4326":
            wards = wards.to_crs("EPSG:4326")
        
        # Spatial join: map grid points to wards
        joined = gpd.sjoin(gdf_grid, wards[['ward_id', 'geometry']], how="left", predicate="within")
        grid_to_ward = joined[['lat', 'lon', 'ward_id']].dropna(subset=['ward_id'])
        
        # Merge rainfall data with ward mapping
        df_ward = df.merge(grid_to_ward, on=['lat', 'lon'], how='inner')
        
        # Aggregate by ward and date (mean rainfall)
        ward_rainfall = df_ward.groupby(['ward_id', 'date'])['rainfall'].mean().reset_index()
        ward_rainfall = ward_rainfall.rename(columns={'rainfall': 'rainfall_mm'})
        
        print(f"  Processed {len(ward_rainfall)} ward-day records")
        print(f"  Wards: {ward_rainfall['ward_id'].nunique()}")
        print(f"  Date range: {ward_rainfall['date'].min()} to {ward_rainfall['date'].max()}")
        
        return ward_rainfall
    
    def compute_hourly_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Convert daily rainfall to hourly features for threshold model.
        
        Creates:
        - rainfall_mm_hr: Hourly rainfall (assumes uniform distribution)
        - rainfall_3hr_mm: 3-hour cumulative
        - rainfall_24hr_mm: 24-hour cumulative
        
        Args:
            df_daily: DataFrame with daily rainfall (ward_id, date, rainfall_mm)
            
        Returns:
            DataFrame with hourly features
        """
        print("\nComputing hourly features...")
        
        df = df_daily.copy()
        df = df.sort_values(['ward_id', 'date'])
        
        # Create hourly records (assuming uniform distribution over 24 hours)
        # This is a simplification - real hourly data would be better
        hourly_records = []
        
        for ward_id in df['ward_id'].unique():
            ward_data = df[df['ward_id'] == ward_id].copy()
            
            for _, row in ward_data.iterrows():
                date = row['date']
                daily_rain = row['rainfall_mm']
                hourly_rain = daily_rain / 24.0  # Assume uniform distribution
                
                # Create 24 hourly records for each day
                for hour in range(24):
                    timestamp = date + pd.Timedelta(hours=hour)
                    
                    # Compute 3hr and 24hr cumulative
                    # Use rolling window over previous hours
                    past_24h = ward_data[
                        (ward_data['date'] >= date - pd.Timedelta(days=1)) &
                        (ward_data['date'] <= date)
                    ]['rainfall_mm'].sum()
                    
                    past_3h = hourly_rain * 3  # Approximation
                    
                    hourly_records.append({
                        'ward_id': ward_id,
                        'datetime': timestamp,
                        'rainfall_mm_hr': hourly_rain,
                        'rainfall_3hr_mm': past_3h,
                        'rainfall_24hr_mm': past_24h,
                        'date': date
                    })
        
        df_hourly = pd.DataFrame(hourly_records)
        print(f"  Created {len(df_hourly)} hourly records")
        
        return df_hourly
    
    def add_static_features(
        self,
        df: pd.DataFrame,
        static_features_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Add static ward features (drainage_score, elevation_category, etc.)
        
        Args:
            df: DataFrame with ward_id
            static_features_path: Path to static features CSV
            
        Returns:
            DataFrame with static features added
        """
        if static_features_path is None:
            static_features_path = self.processed_dir / 'ward_static_features.csv'
        
        if not static_features_path.exists():
            warnings.warn(f"Static features file not found: {static_features_path}")
            print("  Using default values for static features")
            
            # Add default values
            df['drainage_score'] = 0.5
            df['elevation_category'] = 'medium'
            return df
        
        print(f"\nLoading static features from {static_features_path}")
        static_df = pd.read_csv(static_features_path)
        
        # Ensure ward_id column
        if 'ward_id' not in static_df.columns:
            if 'unique' in static_df.columns:
                static_df['ward_id'] = static_df['unique'].astype(str)
            else:
                raise ValueError("Static features must have 'ward_id' or 'unique' column")
        
        # Merge static features
        df = df.merge(static_df, on='ward_id', how='left')
        
        # Fill missing values
        if 'drainage_score' not in df.columns:
            df['drainage_score'] = 0.5
        else:
            df['drainage_score'] = pd.to_numeric(df['drainage_score'], errors='coerce').fillna(0.5)
        
        if 'elevation_category' not in df.columns:
            df['elevation_category'] = 'medium'
        else:
            # Map elevation to categories if needed
            if df['elevation_category'].dtype == 'object':
                df['elevation_category'] = df['elevation_category'].fillna('medium')
            else:
                # If numeric, categorize
                df['elevation_category'] = pd.cut(
                    df['elevation_category'],
                    bins=3,
                    labels=['low', 'medium', 'high']
                ).fillna('medium')
        
        print(f"  Added static features for {df['ward_id'].nunique()} wards")
        
        return df
    
    def add_season_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add season feature (monsoon / non-monsoon).
        
        Monsoon: June-September
        Non-monsoon: October-May
        
        Args:
            df: DataFrame with date/datetime column
            
        Returns:
            DataFrame with season column
        """
        df = df.copy()
        
        # Determine date column
        date_col = 'datetime' if 'datetime' in df.columns else 'date'
        
        df['month'] = pd.to_datetime(df[date_col]).dt.month
        df['season'] = df['month'].apply(
            lambda m: 'monsoon' if 6 <= m <= 9 else 'non-monsoon'
        )
        df = df.drop(columns=['month'])
        
        return df
    
    def prepare_training_data(
        self,
        years: List[int] = [2020, 2021, 2022, 2023, 2024],
        resolution: str = '0.25',
        include_waterlog_labels: bool = False,
        waterlog_labels_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Prepare complete training dataset from IMD rainfall data.
        
        Args:
            years: List of years to process
            resolution: '0.25' or '1.0' degree
            include_waterlog_labels: Whether to include waterlogging labels
            waterlog_labels_path: Path to waterlogging labels CSV
            
        Returns:
            DataFrame ready for threshold model training
        """
        print("=" * 60)
        print("Preparing Training Data for Rainfall Threshold Model")
        print("=" * 60)
        print(f"\nYears: {years}")
        print(f"Resolution: {resolution} degree")
        print()
        
        # Find wards shapefile
        wards_shp = self.data_dir / 'raw' / 'wards' / 'delhi_ward.shp'
        if not wards_shp.exists():
            # Try alternative locations
            alt_locations = [
                self.data_dir.parent / 'raw' / 'wards' / 'delhi_ward.shp',
                Path('backend/data/raw/wards/delhi_ward.shp')
            ]
            for alt in alt_locations:
                if alt.exists():
                    wards_shp = alt
                    break
            else:
                raise FileNotFoundError(
                    f"Wards shapefile not found. Expected at: {wards_shp}\n"
                    "Please ensure wards shapefile is available."
                )
        
        # Process each year
        all_rainfall = []
        
        for year in years:
            nc_file = self.download_imd_rainfall(year, resolution)
            if nc_file is None:
                print(f"Skipping {year}: File not found")
                continue
            
            try:
                year_rainfall = self.process_imd_netcdf(nc_file, wards_shp, year)
                all_rainfall.append(year_rainfall)
            except Exception as e:
                print(f"Error processing {year}: {e}")
                continue
        
        if not all_rainfall:
            raise ValueError("No rainfall data processed. Please download IMD NetCDF files.")
        
        # Combine all years
        df_rainfall = pd.concat(all_rainfall, ignore_index=True)
        print(f"\nTotal rainfall records: {len(df_rainfall)}")
        
        # Convert to hourly features
        df_hourly = self.compute_hourly_features(df_rainfall)
        
        # Add static features
        df_hourly = self.add_static_features(df_hourly)
        
        # Add season feature
        df_hourly = self.add_season_feature(df_hourly)
        
        # Add waterlogging labels if provided
        if include_waterlog_labels and waterlog_labels_path and waterlog_labels_path.exists():
            print(f"\nLoading waterlogging labels from {waterlog_labels_path}")
            labels_df = pd.read_csv(waterlog_labels_path)
            # Merge logic depends on label file format
            # This is a placeholder - adjust based on your label format
            df_hourly = df_hourly.merge(
                labels_df,
                on=['ward_id', 'date'],
                how='left'
            )
            df_hourly['waterlog_level'] = df_hourly['waterlog_level'].fillna(0).astype(int)
        else:
            print("\n⚠ Warning: No waterlogging labels provided.")
            print("  The dataset will have 'waterlog_level' = 0 for all records.")
            print("  You need to add waterlogging labels to train the threshold model.")
            df_hourly['waterlog_level'] = 0
        
        # Select final columns
        final_columns = [
            'ward_id',
            'rainfall_mm_hr',
            'rainfall_3hr_mm',
            'rainfall_24hr_mm',
            'drainage_score',
            'elevation_category',
            'season',
            'waterlog_level',
            'datetime'
        ]
        
        df_final = df_hourly[[col for col in final_columns if col in df_hourly.columns]].copy()
        
        # Save output
        output_path = self.output_dir / f'threshold_training_data_{years[0]}_{years[-1]}.csv'
        df_final.to_csv(output_path, index=False)
        print(f"\n✓ Saved training data to: {output_path}")
        print(f"  Total records: {len(df_final)}")
        print(f"  Wards: {df_final['ward_id'].nunique()}")
        print(f"  Records with waterlog_level > 0: {(df_final['waterlog_level'] > 0).sum()}")
        
        return df_final


def main():
    """Main function to prepare training data."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Prepare training data for Rainfall Threshold Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python prepare_data.py --years 2020 2021 2022 2023 2024 --resolution 0.25
        """
    )
    
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        default=[2020, 2021, 2022, 2023, 2024],
        help='Years to process (default: 2020-2024)'
    )
    
    parser.add_argument(
        '--resolution',
        type=str,
        default='0.25',
        choices=['0.25', '1.0'],
        help='IMD resolution (default: 0.25)'
    )
    
    parser.add_argument(
        '--waterlog_labels',
        type=str,
        default=None,
        help='Path to waterlogging labels CSV (optional)'
    )
    
    args = parser.parse_args()
    
    preparer = ThresholdDataPreparer()
    
    try:
        df = preparer.prepare_training_data(
            years=args.years,
            resolution=args.resolution,
            include_waterlog_labels=args.waterlog_labels is not None,
            waterlog_labels_path=Path(args.waterlog_labels) if args.waterlog_labels else None
        )
        
        print("\n" + "=" * 60)
        print("Data preparation completed!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. If waterlogging labels were not included, add them to the dataset")
        print("2. Train the threshold model:")
        print(f"   python train_thresholds.py --data_path threshold_model/data/threshold_training_data_{args.years[0]}_{args.years[-1]}.csv")
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
