"""
Utility functions for saving, loading, and managing thresholds.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime


def save_thresholds_to_csv(
    thresholds_df: pd.DataFrame,
    output_path: str,
    include_methods: bool = False
) -> None:
    """
    Save thresholds DataFrame to CSV file.
    
    Args:
        thresholds_df: DataFrame with threshold data
        output_path: Path to save CSV file
        include_methods: If True, include method columns (for debugging)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Select columns for output
    if include_methods and 'alert_method' in thresholds_df.columns:
        cols = [
            'ward_id',
            'alert_threshold_mm_hr',
            'critical_threshold_mm_hr',
            'sample_count',
            'alert_sample_count',
            'critical_sample_count',
            'alert_method',
            'critical_method',
            'last_updated'
        ]
    else:
        cols = [
            'ward_id',
            'alert_threshold_mm_hr',
            'critical_threshold_mm_hr',
            'sample_count',
            'last_updated'
        ]
    
    # Filter to available columns
    available_cols = [col for col in cols if col in thresholds_df.columns]
    output_df = thresholds_df[available_cols].copy()
    
    output_df.to_csv(output_path, index=False)
    print(f"Thresholds saved to {output_path}")


def load_thresholds_from_csv(file_path: str) -> pd.DataFrame:
    """
    Load thresholds from CSV file.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataFrame with threshold data
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Threshold file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Loaded thresholds for {len(df)} wards from {file_path}")
    return df


def save_thresholds_to_json(
    thresholds_df: pd.DataFrame,
    output_path: str
) -> None:
    """
    Save thresholds to JSON file (database-ready format).
    
    Args:
        thresholds_df: DataFrame with threshold data
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dictionary format
    records = thresholds_df.to_dict('records')
    
    # Format for JSON (handle NaN values)
    for record in records:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
    
    output = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_wards': len(records),
            'schema_version': '1.0'
        },
        'thresholds': records
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"Thresholds saved to JSON: {output_path}")


def load_thresholds_from_json(file_path: str) -> pd.DataFrame:
    """
    Load thresholds from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        DataFrame with threshold data
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Threshold file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data['thresholds'])
    print(f"Loaded thresholds for {len(df)} wards from {file_path}")
    return df


def validate_thresholds(thresholds_df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate threshold DataFrame.
    
    Args:
        thresholds_df: DataFrame with threshold data
        
    Returns:
        Dictionary with validation results
    """
    required_columns = [
        'ward_id',
        'alert_threshold_mm_hr',
        'critical_threshold_mm_hr'
    ]
    
    missing_cols = set(required_columns) - set(thresholds_df.columns)
    if missing_cols:
        return {
            'valid': False,
            'error': f"Missing required columns: {missing_cols}"
        }
    
    # Check for negative thresholds
    negative_alert = (thresholds_df['alert_threshold_mm_hr'] < 0).sum()
    negative_critical = (thresholds_df['critical_threshold_mm_hr'] < 0).sum()
    
    # Check for critical < alert (where both are valid)
    invalid_order = 0
    valid_mask = (
        thresholds_df['alert_threshold_mm_hr'].notna() &
        thresholds_df['critical_threshold_mm_hr'].notna()
    )
    if valid_mask.sum() > 0:
        invalid_order = (
            thresholds_df.loc[valid_mask, 'critical_threshold_mm_hr'] <
            thresholds_df.loc[valid_mask, 'alert_threshold_mm_hr']
        ).sum()
    
    return {
        'valid': True,
        'total_wards': len(thresholds_df),
        'wards_with_alert': thresholds_df['alert_threshold_mm_hr'].notna().sum(),
        'wards_with_critical': thresholds_df['critical_threshold_mm_hr'].notna().sum(),
        'negative_alert_count': negative_alert,
        'negative_critical_count': negative_critical,
        'invalid_order_count': invalid_order,
        'warnings': []
    }
