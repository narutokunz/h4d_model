"""
Rainfall Threshold Training Module

This module implements quantile-based learning of ward-specific rainfall thresholds
for alert and critical waterlogging levels.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime
import warnings


class RainfallThresholdTrainer:
    """
    Trains ward-specific rainfall thresholds using quantile-based learning.
    
    For each ward:
    - Alert threshold: 80th percentile of rainfall_mm_hr where waterlog_level >= 1
    - Critical threshold: 95th percentile of rainfall_mm_hr where waterlog_level == 2
    """
    
    def __init__(
        self,
        alert_quantile: float = 0.80,
        critical_quantile: float = 0.95,
        min_alert_samples: int = 3,
        min_critical_samples: int = 2
    ):
        """
        Initialize trainer.
        
        Args:
            alert_quantile: Quantile for alert threshold (default 0.80 = 80th percentile)
            critical_quantile: Quantile for critical threshold (default 0.95 = 95th percentile)
            min_alert_samples: Minimum samples with waterlog_level >= 1 required
            min_critical_samples: Minimum samples with waterlog_level == 2 required
        """
        self.alert_quantile = alert_quantile
        self.critical_quantile = critical_quantile
        self.min_alert_samples = min_alert_samples
        self.min_critical_samples = min_critical_samples
        
        self.thresholds_df: Optional[pd.DataFrame] = None
        
    def compute_alert_threshold(
        self, 
        ward_data: pd.DataFrame,
        fallback_strategy: str = 'global'
    ) -> Tuple[float, int, str]:
        """
        Compute alert threshold for a single ward.
        
        Alert threshold = rainfall_mm_hr value at specified quantile 
        where waterlog_level >= 1
        
        Args:
            ward_data: DataFrame with data for a single ward
            fallback_strategy: Strategy if insufficient data ('global', 'ward_mean', 'skip')
            
        Returns:
            Tuple of (threshold_value, sample_count, method_used)
        """
        # Filter to rows with waterlog_level >= 1 (any waterlogging)
        waterlog_data = ward_data[ward_data['waterlog_level'] >= 1]
        
        if len(waterlog_data) < self.min_alert_samples:
            # Insufficient data - use fallback
            if fallback_strategy == 'skip':
                return np.nan, len(waterlog_data), 'insufficient_data'
            elif fallback_strategy == 'ward_mean':
                # Use mean of all rainfall where waterlog_level >= 1 for this ward
                mean_val = waterlog_data['rainfall_mm_hr'].mean()
                if pd.isna(mean_val):
                    return np.nan, len(waterlog_data), 'insufficient_data'
                return mean_val, len(waterlog_data), 'fallback_mean'
            else:  # 'global'
                # Will be computed at global level
                return np.nan, len(waterlog_data), 'needs_global_fallback'
        
        # Compute quantile threshold
        threshold = waterlog_data['rainfall_mm_hr'].quantile(self.alert_quantile)
        
        return threshold, len(waterlog_data), 'quantile'
    
    def compute_critical_threshold(
        self,
        ward_data: pd.DataFrame,
        fallback_strategy: str = 'global'
    ) -> Tuple[float, int, str]:
        """
        Compute critical threshold for a single ward.
        
        Critical threshold = rainfall_mm_hr value at specified quantile 
        where waterlog_level == 2 (severe waterlogging)
        
        Args:
            ward_data: DataFrame with data for a single ward
            fallback_strategy: Strategy if insufficient data ('global', 'ward_mean', 'skip')
            
        Returns:
            Tuple of (threshold_value, sample_count, method_used)
        """
        # Filter to rows with waterlog_level == 2 (severe waterlogging)
        severe_data = ward_data[ward_data['waterlog_level'] == 2]
        
        if len(severe_data) < self.min_critical_samples:
            # Insufficient data - use fallback
            if fallback_strategy == 'skip':
                return np.nan, len(severe_data), 'insufficient_data'
            elif fallback_strategy == 'ward_mean':
                # Use mean of severe waterlogging rainfall for this ward
                mean_val = severe_data['rainfall_mm_hr'].mean()
                if pd.isna(mean_val):
                    # Try using alert threshold as proxy
                    alert_data = ward_data[ward_data['waterlog_level'] >= 1]
                    if len(alert_data) >= 2:
                        mean_val = alert_data['rainfall_mm_hr'].quantile(0.90)
                    if pd.isna(mean_val):
                        return np.nan, len(severe_data), 'insufficient_data'
                    return mean_val, len(severe_data), 'fallback_from_alert'
                return mean_val, len(severe_data), 'fallback_mean'
            else:  # 'global'
                return np.nan, len(severe_data), 'needs_global_fallback'
        
        # Compute quantile threshold
        threshold = severe_data['rainfall_mm_hr'].quantile(self.critical_quantile)
        
        return threshold, len(severe_data), 'quantile'
    
    def compute_global_fallback_thresholds(
        self,
        df: pd.DataFrame
    ) -> Tuple[float, float]:
        """
        Compute global fallback thresholds from all wards.
        
        Used when a ward has insufficient data.
        
        Args:
            df: Full training DataFrame
            
        Returns:
            Tuple of (global_alert_threshold, global_critical_threshold)
        """
        # Global alert threshold: quantile of all rainfall where waterlog_level >= 1
        alert_data = df[df['waterlog_level'] >= 1]
        if len(alert_data) < self.min_alert_samples:
            warnings.warn("Insufficient global data for alert threshold fallback")
            global_alert = np.nan
        else:
            global_alert = alert_data['rainfall_mm_hr'].quantile(self.alert_quantile)
        
        # Global critical threshold: quantile of all rainfall where waterlog_level == 2
        critical_data = df[df['waterlog_level'] == 2]
        if len(critical_data) < self.min_critical_samples:
            warnings.warn("Insufficient global data for critical threshold fallback")
            global_critical = np.nan
        else:
            global_critical = critical_data['rainfall_mm_hr'].quantile(self.critical_quantile)
        
        return global_alert, global_critical
    
    def train_thresholds(
        self,
        df: pd.DataFrame,
        fallback_strategy: str = 'global'
    ) -> pd.DataFrame:
        """
        Train thresholds for all wards.
        
        Args:
            df: Cleaned training DataFrame
            fallback_strategy: Strategy for wards with insufficient data
                              ('global', 'ward_mean', 'skip')
            
        Returns:
            DataFrame with columns:
                ward_id, alert_threshold_mm_hr, critical_threshold_mm_hr,
                alert_sample_count, critical_sample_count, last_updated
        """
        if 'ward_id' not in df.columns or 'rainfall_mm_hr' not in df.columns:
            raise ValueError("DataFrame must contain 'ward_id' and 'rainfall_mm_hr' columns")
        
        results = []
        
        # Compute global fallback thresholds if needed
        global_alert, global_critical = None, None
        if fallback_strategy == 'global':
            global_alert, global_critical = self.compute_global_fallback_thresholds(df)
            print(f"Global fallback thresholds: Alert={global_alert:.2f}, Critical={global_critical:.2f}")
        
        # Process each ward
        for ward_id in sorted(df['ward_id'].unique()):
            ward_data = df[df['ward_id'] == ward_id].copy()
            
            # Compute alert threshold
            alert_thresh, alert_count, alert_method = self.compute_alert_threshold(
                ward_data, fallback_strategy
            )
            
            # Apply global fallback if needed
            if pd.isna(alert_thresh) and fallback_strategy == 'global' and global_alert is not None:
                alert_thresh = global_alert
                alert_method = 'global_fallback'
            
            # Compute critical threshold
            critical_thresh, critical_count, critical_method = self.compute_critical_threshold(
                ward_data, fallback_strategy
            )
            
            # Apply global fallback if needed
            if pd.isna(critical_thresh) and fallback_strategy == 'global' and global_critical is not None:
                critical_thresh = global_critical
                critical_method = 'global_fallback'
            
            # Ensure critical threshold >= alert threshold (if both are valid)
            if not pd.isna(alert_thresh) and not pd.isna(critical_thresh):
                if critical_thresh < alert_thresh:
                    # Adjust critical to be at least alert threshold * 1.2
                    critical_thresh = max(alert_thresh * 1.2, critical_thresh)
                    critical_method = f'{critical_method}_adjusted'
            
            results.append({
                'ward_id': ward_id,
                'alert_threshold_mm_hr': alert_thresh,
                'critical_threshold_mm_hr': critical_thresh,
                'alert_sample_count': alert_count,
                'critical_sample_count': critical_count,
                'alert_method': alert_method,
                'critical_method': critical_method,
                'total_samples': len(ward_data)
            })
        
        # Create DataFrame
        thresholds_df = pd.DataFrame(results)
        
        # Add last_updated timestamp
        thresholds_df['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Compute summary statistics
        valid_alert = thresholds_df['alert_threshold_mm_hr'].notna().sum()
        valid_critical = thresholds_df['critical_threshold_mm_hr'].notna().sum()
        
        print(f"\nThreshold training complete:")
        print(f"  - Total wards processed: {len(thresholds_df)}")
        print(f"  - Wards with valid alert threshold: {valid_alert}")
        print(f"  - Wards with valid critical threshold: {valid_critical}")
        print(f"  - Mean alert threshold: {thresholds_df['alert_threshold_mm_hr'].mean():.2f} mm/hr")
        print(f"  - Mean critical threshold: {thresholds_df['critical_threshold_mm_hr'].mean():.2f} mm/hr")
        
        self.thresholds_df = thresholds_df
        return thresholds_df
    
    def get_final_output(self) -> pd.DataFrame:
        """
        Get final output DataFrame in the required format.
        
        Returns:
            DataFrame with columns:
                ward_id, alert_threshold_mm_hr, critical_threshold_mm_hr,
                sample_count, last_updated
        """
        if self.thresholds_df is None:
            raise ValueError("No thresholds trained. Call train_thresholds() first.")
        
        # Create simplified output
        output = pd.DataFrame({
            'ward_id': self.thresholds_df['ward_id'],
            'alert_threshold_mm_hr': self.thresholds_df['alert_threshold_mm_hr'],
            'critical_threshold_mm_hr': self.thresholds_df['critical_threshold_mm_hr'],
            'sample_count': self.thresholds_df['total_samples'],
            'last_updated': self.thresholds_df['last_updated']
        })
        
        return output
