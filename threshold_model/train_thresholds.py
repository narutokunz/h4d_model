"""
Main Training Script for Rainfall Threshold Adaptation Model

This script trains ward-specific rainfall thresholds for early warnings.

Usage:
    python train_thresholds.py --data_path <path_to_data.csv> --output_dir <output_directory>
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from preprocess import ThresholdDataPreprocessor
from threshold_trainer import RainfallThresholdTrainer
from utils import save_thresholds_to_csv, save_thresholds_to_json, validate_thresholds


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train Rainfall Threshold Adaptation Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python train_thresholds.py --data_path data/historical_data.csv --output_dir outputs/
        """
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to input CSV file with historical data'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='Output directory for trained thresholds (default: outputs/)'
    )
    
    parser.add_argument(
        '--alert_quantile',
        type=float,
        default=0.80,
        help='Quantile for alert threshold (default: 0.80)'
    )
    
    parser.add_argument(
        '--critical_quantile',
        type=float,
        default=0.95,
        help='Quantile for critical threshold (default: 0.95)'
    )
    
    parser.add_argument(
        '--min_samples',
        type=int,
        default=5,
        help='Minimum samples per ward (default: 5)'
    )
    
    parser.add_argument(
        '--fallback_strategy',
        type=str,
        default='global',
        choices=['global', 'ward_mean', 'skip'],
        help='Fallback strategy for small-data wards (default: global)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        default='both',
        choices=['csv', 'json', 'both'],
        help='Output format (default: both)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Rainfall Threshold Adaptation Model - Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Data path: {args.data_path}")
    print(f"  Output directory: {output_dir}")
    print(f"  Alert quantile: {args.alert_quantile}")
    print(f"  Critical quantile: {args.critical_quantile}")
    print(f"  Min samples per ward: {args.min_samples}")
    print(f"  Fallback strategy: {args.fallback_strategy}")
    print(f"  Output format: {args.format}")
    print()
    
    # Step 1: Preprocess data
    print("Step 1: Preprocessing data...")
    preprocessor = ThresholdDataPreprocessor(min_samples_per_ward=args.min_samples)
    df_clean, stats = preprocessor.prepare_data(file_path=args.data_path)
    
    # Step 2: Train thresholds
    print("\nStep 2: Training thresholds...")
    trainer = RainfallThresholdTrainer(
        alert_quantile=args.alert_quantile,
        critical_quantile=args.critical_quantile
    )
    thresholds_df = trainer.train_thresholds(df_clean, fallback_strategy=args.fallback_strategy)
    
    # Step 3: Get final output
    print("\nStep 3: Generating final output...")
    final_output = trainer.get_final_output()
    
    # Step 4: Validate thresholds
    print("\nStep 4: Validating thresholds...")
    validation = validate_thresholds(final_output)
    if not validation['valid']:
        print(f"ERROR: {validation['error']}")
        sys.exit(1)
    else:
        print(f"Validation passed:")
        print(f"  Total wards: {validation['total_wards']}")
        print(f"  Wards with alert threshold: {validation['wards_with_alert']}")
        print(f"  Wards with critical threshold: {validation['wards_with_critical']}")
        if validation['invalid_order_count'] > 0:
            print(f"  WARNING: {validation['invalid_order_count']} wards with critical < alert")
    
    # Step 5: Save outputs
    print("\nStep 5: Saving outputs...")
    
    # Save final output (clean format)
    if args.format in ['csv', 'both']:
        csv_path = output_dir / 'rainfall_thresholds.csv'
        save_thresholds_to_csv(final_output, str(csv_path))
        print(f"  Saved to: {csv_path}")
    
    if args.format in ['json', 'both']:
        json_path = output_dir / 'rainfall_thresholds.json'
        save_thresholds_to_json(final_output, str(json_path))
        print(f"  Saved to: {json_path}")
    
    # Save detailed output with methods (for debugging)
    detailed_path = output_dir / 'rainfall_thresholds_detailed.csv'
    save_thresholds_to_csv(thresholds_df, str(detailed_path), include_methods=True)
    print(f"  Detailed output saved to: {detailed_path}")
    
    # Save statistics
    stats_path = output_dir / 'ward_statistics.csv'
    stats.to_csv(stats_path, index=False)
    print(f"  Statistics saved to: {stats_path}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  Total wards processed: {len(final_output)}")
    print(f"  Valid alert thresholds: {final_output['alert_threshold_mm_hr'].notna().sum()}")
    print(f"  Valid critical thresholds: {final_output['critical_threshold_mm_hr'].notna().sum()}")
    print(f"\nOutput files saved to: {output_dir}")
    print()


if __name__ == '__main__':
    main()
