"""
Example Inference Script for Rainfall Threshold Model

This script demonstrates how to use trained thresholds for early warnings.

Usage:
    python example_inference.py --thresholds_path outputs/rainfall_thresholds.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from inference import ThresholdInference
from utils import load_thresholds_from_csv


def main():
    """Example inference function."""
    parser = argparse.ArgumentParser(
        description='Example Inference for Rainfall Threshold Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python example_inference.py --thresholds_path outputs/rainfall_thresholds.csv
        """
    )
    
    parser.add_argument(
        '--thresholds_path',
        type=str,
        default='outputs/rainfall_thresholds.csv',
        help='Path to trained thresholds CSV file'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Rainfall Threshold Model - Inference Example")
    print("=" * 60)
    print()
    
    # Load thresholds
    print(f"Loading thresholds from: {args.thresholds_path}")
    inference = ThresholdInference()
    inference.load_thresholds(args.thresholds_path, format='csv')
    
    print(f"Loaded thresholds for {len(inference.thresholds_df)} wards")
    print()
    
    # Example 1: Single ward prediction
    print("-" * 60)
    print("Example 1: Single Ward Prediction")
    print("-" * 60)
    
    # Get first ward ID from thresholds
    sample_ward_id = inference.thresholds_df['ward_id'].iloc[0]
    
    # Get thresholds for this ward
    thresholds = inference.get_thresholds(sample_ward_id)
    print(f"\nWard: {sample_ward_id}")
    print(f"  Alert threshold: {thresholds['alert_threshold_mm_hr']:.2f} mm/hr")
    print(f"  Critical threshold: {thresholds['critical_threshold_mm_hr']:.2f} mm/hr")
    print(f"  Sample count: {thresholds['sample_count']}")
    
    # Test different forecast scenarios
    test_scenarios = [
        ("Low rainfall", thresholds['alert_threshold_mm_hr'] * 0.5),
        ("Alert-level rainfall", thresholds['alert_threshold_mm_hr'] * 1.1),
        ("Critical-level rainfall", thresholds['critical_threshold_mm_hr'] * 1.1),
    ]
    
    print(f"\nForecast scenarios:")
    for scenario_name, forecast_rainfall in test_scenarios:
        alert_level = inference.get_alert_level(sample_ward_id, forecast_rainfall)
        print(f"  {scenario_name:30s} ({forecast_rainfall:6.2f} mm/hr) -> {alert_level}")
    
    # Example 2: Batch prediction
    print("\n" + "-" * 60)
    print("Example 2: Batch Prediction")
    print("-" * 60)
    
    # Select a few sample wards
    sample_wards = inference.thresholds_df['ward_id'].head(5).tolist()
    
    # Create forecast dictionary
    forecast_dict = {}
    for ward_id in sample_wards:
        thresholds_ward = inference.get_thresholds(ward_id)
        # Use alert threshold as forecast (simulating alert-level conditions)
        forecast_dict[ward_id] = thresholds_ward['alert_threshold_mm_hr'] * 1.0
    
    # Batch predict
    batch_results = inference.batch_predict(forecast_dict)
    
    print(f"\nBatch prediction results for {len(batch_results)} wards:")
    print()
    print(batch_results.to_string(index=False))
    
    # Example 3: Alert level distribution
    print("\n" + "-" * 60)
    print("Example 3: Alert Level Distribution")
    print("-" * 60)
    
    # Simulate forecasts for all wards using their alert thresholds
    all_forecasts = {}
    for ward_id in inference.thresholds_df['ward_id'].head(20):
        thresholds_ward = inference.get_thresholds(ward_id)
        # Use alert threshold as forecast
        if pd.notna(thresholds_ward['alert_threshold_mm_hr']):
            all_forecasts[ward_id] = thresholds_ward['alert_threshold_mm_hr']
    
    all_results = inference.batch_predict(all_forecasts)
    alert_distribution = all_results['alert_level'].value_counts()
    
    print(f"\nAlert level distribution (for {len(all_results)} wards):")
    for level, count in alert_distribution.items():
        print(f"  {level:10s}: {count:3d} wards")
    
    print("\n" + "=" * 60)
    print("Inference example completed!")
    print("=" * 60)
    print()


def interactive_example():
    """
    Interactive example for testing individual ward predictions.
    Run this function for interactive testing.
    """
    thresholds_path = 'outputs/rainfall_thresholds.csv'
    
    print("Loading thresholds...")
    inference = ThresholdInference()
    inference.load_thresholds(thresholds_path, format='csv')
    
    print(f"Loaded thresholds for {len(inference.thresholds_df)} wards")
    print("\nAvailable wards:")
    print(inference.thresholds_df[['ward_id', 'alert_threshold_mm_hr', 'critical_threshold_mm_hr']].head(10).to_string(index=False))
    print()
    
    while True:
        try:
            ward_id = input("Enter ward_id (or 'q' to quit): ").strip()
            if ward_id.lower() == 'q':
                break
            
            forecast_str = input("Enter forecast rainfall (mm/hr): ").strip()
            forecast_rainfall = float(forecast_str)
            
            alert_level = inference.get_alert_level(ward_id, forecast_rainfall)
            thresholds = inference.get_thresholds(ward_id)
            
            print(f"\nResult:")
            print(f"  Ward: {ward_id}")
            print(f"  Forecast: {forecast_rainfall:.2f} mm/hr")
            print(f"  Alert Level: {alert_level}")
            print(f"  Alert Threshold: {thresholds['alert_threshold_mm_hr']:.2f} mm/hr")
            print(f"  Critical Threshold: {thresholds['critical_threshold_mm_hr']:.2f} mm/hr")
            print()
            
        except ValueError as e:
            print(f"Error: {e}\n")
        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == '__main__':
    # Check if running in interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_example()
    else:
        main()
