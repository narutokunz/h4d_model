"""
Data Preprocessing Module for Rainfall Threshold Adaptation Model

This module handles data loading, cleaning, and preparation for threshold learning.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings
import warnings


class ThresholdDataPreprocessor:
    """
    Preprocesses historical rainfall and waterlogging data for threshold learning.
    """
    
    def __init__(self, min_samples_per_ward: int = 5):
        """
        Initialize preprocessor.
        
        Args:
            min_samples_per_ward: Minimum samples required per ward for processing
        """
        self.min_samples_per_ward = min_samples_per_ward
        self.required_columns = [
            'ward_id',
            'rainfall_mm_hr',
            'rainfall_3hr_mm',
            'rainfall_24hr_mm',
            'drainage_score',
            'elevation_category',
            'season',
            'waterlog_level'
        ]
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Loaded DataFrame
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows from {file_path}")
        return df
    
    def validate_columns(self, df: pd.DataFrame) -> bool:
        """
        Validate that all required columns exist.
        
        Args:
            df: Input DataFrame
            
        Returns:
            True if all columns exist
        """
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}\n"
                f"Found columns: {list(df.columns)}"
            )
        return True
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate data.
        
        Args:
            df: Raw input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Validate columns
        self.validate_columns(df_clean)
        
        initial_rows = len(df_clean)
        
        # Remove rows with missing critical values
        critical_cols = ['ward_id', 'rainfall_mm_hr', 'waterlog_level']
        df_clean = df_clean.dropna(subset=critical_cols)
        
        # Validate waterlog_level values (0, 1, 2)
        invalid_levels = ~df_clean['waterlog_level'].isin([0, 1, 2])
        if invalid_levels.sum() > 0:
            warnings.warn(
                f"Found {invalid_levels.sum()} rows with invalid waterlog_level. Removing them."
            )
            df_clean = df_clean[~invalid_levels]
        
        # Validate rainfall values (should be non-negative)
        df_clean = df_clean[df_clean['rainfall_mm_hr'] >= 0]
        
        # Fill missing optional columns with defaults if needed
        if 'drainage_score' in df_clean.columns:
            df_clean['drainage_score'] = pd.to_numeric(
                df_clean['drainage_score'], errors='coerce'
            ).fillna(0.5)  # Default to medium drainage
        
        # Ensure elevation_category is valid
        if 'elevation_category' in df_clean.columns:
            valid_elevations = ['low', 'medium', 'high']
            df_clean['elevation_category'] = df_clean['elevation_category'].fillna('medium')
            df_clean = df_clean[
                df_clean['elevation_category'].isin(valid_elevations)
            ]
        
        # Ensure season is valid
        if 'season' in df_clean.columns:
            valid_seasons = ['monsoon', 'non-monsoon']
            df_clean['season'] = df_clean['season'].fillna('monsoon')
            df_clean = df_clean[df_clean['season'].isin(valid_seasons)]
        
        rows_removed = initial_rows - len(df_clean)
        if rows_removed > 0:
            print(f"Removed {rows_removed} rows during cleaning. Remaining: {len(df_clean)} rows")
        
        return df_clean
    
    def get_ward_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get statistics per ward for analysis.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with ward-level statistics
        """
        stats = df.groupby('ward_id').agg({
            'rainfall_mm_hr': ['count', 'mean', 'std', 'min', 'max'],
            'waterlog_level': ['count', 'sum']
        }).reset_index()
        
        stats.columns = [
            'ward_id',
            'total_samples',
            'mean_rainfall',
            'std_rainfall',
            'min_rainfall',
            'max_rainfall',
            'waterlog_samples',
            'waterlog_events'
        ]
        
        # Add count of waterlog_level >= 1
        waterlog_any = df[df['waterlog_level'] >= 1].groupby('ward_id').size()
        stats['waterlog_any_count'] = stats['ward_id'].map(waterlog_any).fillna(0).astype(int)
        
        # Add count of waterlog_level == 2 (severe)
        waterlog_severe = df[df['waterlog_level'] == 2].groupby('ward_id').size()
        stats['waterlog_severe_count'] = stats['ward_id'].map(waterlog_severe).fillna(0).astype(int)
        
        return stats
    
    def filter_wards_by_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out wards with insufficient samples.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with only wards meeting minimum sample requirements
        """
        ward_counts = df['ward_id'].value_counts()
        valid_wards = ward_counts[ward_counts >= self.min_samples_per_ward].index
        
        df_filtered = df[df['ward_id'].isin(valid_wards)].copy()
        
        removed_wards = len(df['ward_id'].unique()) - len(valid_wards)
        if removed_wards > 0:
            print(
                f"Filtered out {removed_wards} wards with < {self.min_samples_per_ward} samples. "
                f"Remaining: {len(valid_wards)} wards"
            )
        
        return df_filtered
    
    def prepare_data(
        self, 
        file_path: Optional[str] = None, 
        df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete data preparation pipeline.
        
        Args:
            file_path: Path to CSV file (if loading from file)
            df: DataFrame (if data already loaded)
            
        Returns:
            Tuple of (cleaned DataFrame, ward statistics DataFrame)
        """
        if df is None and file_path is None:
            raise ValueError("Either file_path or df must be provided")
        
        if df is None:
            df = self.load_data(file_path)
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Get statistics before filtering (to see all wards)
        stats = self.get_ward_statistics(df_clean)
        
        # Filter wards with insufficient samples
        df_clean = self.filter_wards_by_samples(df_clean)
        
        # Update stats to only include valid wards
        stats = stats[stats['ward_id'].isin(df_clean['ward_id'].unique())]
        
        print(f"\nData preparation complete:")
        print(f"  - Total samples: {len(df_clean)}")
        print(f"  - Total wards: {len(df_clean['ward_id'].unique())}")
        print(f"  - Samples with waterlog_level >= 1: {len(df_clean[df_clean['waterlog_level'] >= 1])}")
        print(f"  - Samples with waterlog_level == 2: {len(df_clean[df_clean['waterlog_level'] == 2])}")
        
        return df_clean, stats
