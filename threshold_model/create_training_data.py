"""
Create Training Data from Existing Processed Rainfall Data

This script uses already-processed rainfall data and creates waterlogging labels
based on historical flood data and rainfall thresholds.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
from typing import Optional


def load_processed_rainfall(years: list = [2022, 2023, 2024, 2025]) -> pd.DataFrame:
    """Load processed rainfall data from multiple years."""
    data_dir = Path(__file__).parent.parent / 'backend' / 'data' / 'processed'
    
    all_rainfall = []
    for year in years:
        file_path = data_dir / f'rainfall_ward_daily_{year}.csv'
        if file_path.exists():
            print(f"Loading {year} rainfall data...")
            df = pd.read_csv(file_path)
            # Standardize column names
            if 'rainfall' in df.columns:
                df = df.rename(columns={'rainfall': 'rainfall_mm'})
            df['year'] = year
            all_rainfall.append(df)
        else:
            print(f"Warning: {file_path} not found, skipping {year}")
    
    if not all_rainfall:
        raise FileNotFoundError("No processed rainfall data found")
    
    df_all = pd.concat(all_rainfall, ignore_index=True)
    print(f"Loaded {len(df_all)} total rainfall records")
    return df_all


def load_static_features() -> pd.DataFrame:
    """Load static ward features."""
    data_dir = Path(__file__).parent.parent / 'backend' / 'data' / 'processed'
    static_path = data_dir / 'ward_static_features.csv'
    
    if static_path.exists():
        df = pd.read_csv(static_path)
        # Ensure ward_id column
        if 'ward_id' not in df.columns:
            if 'unique' in df.columns:
                df['ward_id'] = df['unique'].astype(str)
        return df
    else:
        print("Warning: Static features not found, using defaults")
        return None


def load_flood_prone_wards() -> set:
    """Load flood-prone ward IDs."""
    data_dir = Path(__file__).parent.parent / 'backend' / 'data' / 'processed'
    flood_path = data_dir / 'historical_floods_ward.csv'
    
    if flood_path.exists():
        df = pd.read_csv(flood_path)
        if 'ward_id' in df.columns:
            flood_prone = set(df[df['is_flood_prone'] == True]['ward_id'].astype(str))
            print(f"Loaded {len(flood_prone)} flood-prone wards")
            return flood_prone
    
    return set()


def create_hourly_features(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Convert daily rainfall to hourly features."""
    print("Converting daily to hourly features...")
    
    df = df_daily.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ward_id', 'date'])
    
    hourly_records = []
    
    for ward_id in df['ward_id'].unique():
        ward_data = df[df['ward_id'] == ward_id].copy()
        
        for _, row in ward_data.iterrows():
            date = row['date']
            daily_rain = row['rainfall_mm']
            hourly_rain = daily_rain / 24.0  # Assume uniform distribution
            
            # Calculate cumulative rainfalls
            # 24hr: same as daily (already cumulative)
            # 3hr: approximate as 3/24 of daily
            past_3h = hourly_rain * 3
            
            # Create 24 hourly records for each day
            for hour in range(24):
                timestamp = date + pd.Timedelta(hours=hour)
                hourly_records.append({
                    'ward_id': ward_id,
                    'datetime': timestamp,
                    'date': date,
                    'rainfall_mm_hr': hourly_rain,
                    'rainfall_3hr_mm': past_3h,
                    'rainfall_24hr_mm': daily_rain
                })
    
    df_hourly = pd.DataFrame(hourly_records)
    print(f"Created {len(df_hourly)} hourly records")
    return df_hourly


def add_static_features(df: pd.DataFrame, static_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Add static ward features."""
    if static_df is None:
        # Use defaults
        df['drainage_score'] = 0.5
        df['elevation_category'] = 'medium'
    else:
        # Check what columns are available
        merge_cols = ['ward_id']
        if 'drainage_score' in static_df.columns:
            merge_cols.append('drainage_score')
        if 'elevation_category' in static_df.columns:
            merge_cols.append('elevation_category')
        elif 'mean_elevation' in static_df.columns:
            # Derive elevation category from mean elevation
            static_df = static_df.copy()
            static_df['elevation_category'] = pd.cut(
                static_df['mean_elevation'],
                bins=3,
                labels=['low', 'medium', 'high']
            ).astype(str).replace('nan', 'medium')
            merge_cols.append('elevation_category')
        
        df = df.merge(static_df[merge_cols], on='ward_id', how='left')
        
        # Set defaults for missing values
        if 'drainage_score' not in df.columns:
            df['drainage_score'] = 0.5
        else:
            df['drainage_score'] = pd.to_numeric(df['drainage_score'], errors='coerce').fillna(0.5)
        
        if 'elevation_category' not in df.columns:
            df['elevation_category'] = 'medium'
        else:
            df['elevation_category'] = df['elevation_category'].fillna('medium')
    
    return df


def add_season_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add season feature."""
    df = df.copy()
    df['month'] = pd.to_datetime(df['datetime']).dt.month
    df['season'] = df['month'].apply(
        lambda m: 'monsoon' if 6 <= m <= 9 else 'non-monsoon'
    )
    df = df.drop(columns=['month'])
    return df


def create_waterlogging_labels(
    df: pd.DataFrame,
    flood_prone_wards: set,
    severe_threshold_mm: float = 100.0,
    moderate_threshold_mm: float = 50.0
) -> pd.DataFrame:
    """
    Create waterlogging labels based on rainfall and flood-prone status.
    
    Logic:
    - Severe (2): Flood-prone ward AND daily rainfall >= severe_threshold
    - Moderate (1): (Flood-prone ward AND daily rainfall >= moderate_threshold) OR 
                    (Non-flood-prone AND daily rainfall >= severe_threshold)
    - None (0): Otherwise
    """
    print("Creating waterlogging labels...")
    
    df = df.copy()
    
    # Get daily rainfall for labeling
    df['daily_rainfall'] = df['rainfall_24hr_mm']
    
    # Initialize all as 0
    df['waterlog_level'] = 0
    
    # Severe waterlogging: Flood-prone wards with very high rainfall
    severe_mask = (
        (df['ward_id'].isin(flood_prone_wards)) &
        (df['daily_rainfall'] >= severe_threshold_mm)
    )
    df.loc[severe_mask, 'waterlog_level'] = 2
    print(f"  Severe waterlogging (level 2): {severe_mask.sum()} records")
    
    # Moderate waterlogging: 
    # 1. Flood-prone wards with moderate-high rainfall (but < severe)
    # 2. Non-flood-prone wards with very high rainfall
    moderate_mask = (
        (
            (df['ward_id'].isin(flood_prone_wards)) &
            (df['daily_rainfall'] >= moderate_threshold_mm) &
            (df['daily_rainfall'] < severe_threshold_mm)
        ) |
        (
            (~df['ward_id'].isin(flood_prone_wards)) &
            (df['daily_rainfall'] >= severe_threshold_mm)
        )
    ) & (df['waterlog_level'] == 0)  # Don't override severe
    
    df.loc[moderate_mask, 'waterlog_level'] = 1
    print(f"  Moderate waterlogging (level 1): {moderate_mask.sum()} records")
    
    print(f"  No waterlogging (level 0): {(df['waterlog_level'] == 0).sum()} records")
    print(f"  Total with waterlogging (>=1): {(df['waterlog_level'] >= 1).sum()} records")
    
    return df


def main():
    """Main function to create training data."""
    print("=" * 60)
    print("Creating Training Data for Rainfall Threshold Model")
    print("=" * 60)
    print()
    
    # Load data
    print("Step 1: Loading data...")
    df_rainfall = load_processed_rainfall(years=[2022, 2023, 2024, 2025])
    static_df = load_static_features()
    flood_prone_wards = load_flood_prone_wards()
    
    # Convert to hourly
    print("\nStep 2: Converting to hourly features...")
    df_hourly = create_hourly_features(df_rainfall)
    
    # Add static features
    print("\nStep 3: Adding static features...")
    df_hourly = add_static_features(df_hourly, static_df)
    
    # Add season
    print("\nStep 4: Adding season feature...")
    df_hourly = add_season_feature(df_hourly)
    
    # Create labels
    print("\nStep 5: Creating waterlogging labels...")
    df_hourly = create_waterlogging_labels(
        df_hourly,
        flood_prone_wards,
        severe_threshold_mm=100.0,  # 100mm daily = severe
        moderate_threshold_mm=50.0  # 50mm daily = moderate
    )
    
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
    
    # Save
    output_dir = Path(__file__).parent / 'data'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'threshold_training_data_2022_2025.csv'
    df_final.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print("Training data created successfully!")
    print("=" * 60)
    print(f"\nOutput: {output_path}")
    print(f"  Total records: {len(df_final):,}")
    print(f"  Wards: {df_final['ward_id'].nunique()}")
    print(f"  Records with waterlog_level = 0: {(df_final['waterlog_level'] == 0).sum():,}")
    print(f"  Records with waterlog_level = 1: {(df_final['waterlog_level'] == 1).sum():,}")
    print(f"  Records with waterlog_level = 2: {(df_final['waterlog_level'] == 2).sum():,}")
    print(f"  Date range: {df_final['datetime'].min()} to {df_final['datetime'].max()}")
    print()
    
    return df_final


if __name__ == '__main__':
    main()
