"""
Inference Module for Rainfall Threshold Model

This module provides functions to use trained thresholds for early warnings.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal, Optional

# Handle both package import and script import
try:
    from .utils import load_thresholds_from_csv, load_thresholds_from_json
except ImportError:
    from utils import load_thresholds_from_csv, load_thresholds_from_json


class ThresholdInference:
    """
    Inference class for using trained thresholds to generate alerts.
    """
    
    def __init__(self, thresholds_df: Optional[pd.DataFrame] = None):
        """
        Initialize inference with thresholds.
        
        Args:
            thresholds_df: DataFrame with trained thresholds (can be loaded later)
        """
        self.thresholds_df = thresholds_df
        
    def load_thresholds(self, file_path: str, format: Literal['csv', 'json'] = 'csv'):
        """
        Load thresholds from file.
        
        Args:
            file_path: Path to thresholds file
            format: File format ('csv' or 'json')
        """
        if format == 'csv':
            self.thresholds_df = load_thresholds_from_csv(file_path)
        elif format == 'json':
            self.thresholds_df = load_thresholds_from_json(file_path)
        else:
            raise ValueError("Format must be 'csv' or 'json'")
    
    def get_alert_level(
        self,
        ward_id: str,
        forecast_rainfall_mm_hr: float
    ) -> Literal['LOW', 'MEDIUM', 'HIGH']:
        """
        Get alert level for a ward given forecast rainfall.
        
        Args:
            ward_id: Ward identifier (e.g., "W12")
            forecast_rainfall_mm_hr: Forecast rainfall in mm/hr
            
        Returns:
            Alert level: 'LOW', 'MEDIUM', or 'HIGH'
        """
        if self.thresholds_df is None:
            raise ValueError("Thresholds not loaded. Call load_thresholds() first.")
        
        # Find ward thresholds
        ward_data = self.thresholds_df[self.thresholds_df['ward_id'] == ward_id]
        
        if len(ward_data) == 0:
            raise ValueError(f"Ward {ward_id} not found in thresholds")
        
        row = ward_data.iloc[0]
        alert_threshold = row['alert_threshold_mm_hr']
        critical_threshold = row['critical_threshold_mm_hr']
        
        # Check if thresholds are valid
        if pd.isna(alert_threshold) and pd.isna(critical_threshold):
            # No thresholds available - return LOW as default
            return 'LOW'
        
        # Determine alert level
        if not pd.isna(critical_threshold) and forecast_rainfall_mm_hr >= critical_threshold:
            return 'HIGH'
        elif not pd.isna(alert_threshold) and forecast_rainfall_mm_hr >= alert_threshold:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def get_thresholds(self, ward_id: str) -> dict:
        """
        Get thresholds for a specific ward.
        
        Args:
            ward_id: Ward identifier
            
        Returns:
            Dictionary with threshold values
        """
        if self.thresholds_df is None:
            raise ValueError("Thresholds not loaded. Call load_thresholds() first.")
        
        ward_data = self.thresholds_df[self.thresholds_df['ward_id'] == ward_id]
        
        if len(ward_data) == 0:
            raise ValueError(f"Ward {ward_id} not found in thresholds")
        
        row = ward_data.iloc[0]
        return {
            'ward_id': ward_id,
            'alert_threshold_mm_hr': row['alert_threshold_mm_hr'],
            'critical_threshold_mm_hr': row['critical_threshold_mm_hr'],
            'sample_count': row.get('sample_count', None)
        }
    
    def batch_predict(
        self,
        ward_rainfall_dict: dict[str, float]
    ) -> pd.DataFrame:
        """
        Get alert levels for multiple wards.
        
        Args:
            ward_rainfall_dict: Dictionary mapping ward_id to forecast_rainfall_mm_hr
            
        Returns:
            DataFrame with ward_id, forecast_rainfall, and alert_level
        """
        results = []
        
        for ward_id, forecast_rainfall in ward_rainfall_dict.items():
            try:
                alert_level = self.get_alert_level(ward_id, forecast_rainfall)
                results.append({
                    'ward_id': ward_id,
                    'forecast_rainfall_mm_hr': forecast_rainfall,
                    'alert_level': alert_level
                })
            except ValueError as e:
                # Ward not found
                results.append({
                    'ward_id': ward_id,
                    'forecast_rainfall_mm_hr': forecast_rainfall,
                    'alert_level': 'UNKNOWN',
                    'error': str(e)
                })
        
        return pd.DataFrame(results)
