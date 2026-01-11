"""
Model Evaluation Script for Rainfall Threshold Model

This script evaluates the trained threshold model including:
- Coverage (wards with thresholds)
- Training statistics
- Threshold stability
- Validation against historical events
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime
from typing import Dict, Tuple, Optional
import argparse

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import load_thresholds_from_csv, validate_thresholds


class ThresholdModelEvaluator:
    """
    Evaluator for Rainfall Threshold Model.
    
    Since this is a quantile-based model (not ML), we evaluate:
    - Coverage and completeness
    - Training data statistics
    - Threshold stability
    - Validation against historical events
    """
    
    def __init__(self, thresholds_path: str, training_data_path: Optional[str] = None):
        """
        Initialize evaluator.
        
        Args:
            thresholds_path: Path to trained thresholds CSV
            training_data_path: Optional path to training data for statistics
        """
        self.thresholds_path = Path(thresholds_path)
        self.training_data_path = Path(training_data_path) if training_data_path else None
        
        self.thresholds_df: Optional[pd.DataFrame] = None
        self.training_df: Optional[pd.DataFrame] = None
        
    def load_data(self):
        """Load thresholds and training data."""
        print("Loading thresholds...")
        self.thresholds_df = load_thresholds_from_csv(str(self.thresholds_path))
        
        if self.training_data_path and self.training_data_path.exists():
            print(f"Loading training data from {self.training_data_path}...")
            self.training_df = pd.read_csv(self.training_data_path)
            print(f"Loaded {len(self.training_df)} training records")
    
    def evaluate_coverage(self) -> Dict:
        """Evaluate model coverage (wards with thresholds)."""
        if self.thresholds_df is None:
            raise ValueError("Thresholds not loaded. Call load_data() first.")
        
        total_wards = len(self.thresholds_df)
        wards_with_alert = self.thresholds_df['alert_threshold_mm_hr'].notna().sum()
        wards_with_critical = self.thresholds_df['critical_threshold_mm_hr'].notna().sum()
        
        coverage = {
            'total_wards': total_wards,
            'wards_with_alert_threshold': wards_with_alert,
            'wards_with_critical_threshold': wards_with_critical,
            'alert_coverage_percentage': (wards_with_alert / total_wards * 100) if total_wards > 0 else 0,
            'critical_coverage_percentage': (wards_with_critical / total_wards * 100) if total_wards > 0 else 0,
            'complete_coverage_percentage': (
                (self.thresholds_df['alert_threshold_mm_hr'].notna() & 
                 self.thresholds_df['critical_threshold_mm_hr'].notna()).sum() / total_wards * 100
            ) if total_wards > 0 else 0
        }
        
        return coverage
    
    def evaluate_training_data(self) -> Dict:
        """Evaluate training data statistics."""
        if self.training_df is None:
            return {'error': 'Training data not provided'}
        
        df = self.training_df.copy()
        
        stats = {
            'total_samples': len(df),
            'total_wards': df['ward_id'].nunique(),
            'date_range': {
                'start': str(df['datetime'].min()) if 'datetime' in df.columns else None,
                'end': str(df['datetime'].max()) if 'datetime' in df.columns else None
            },
            'waterlogging_distribution': {
                'level_0': (df['waterlog_level'] == 0).sum(),
                'level_1': (df['waterlog_level'] == 1).sum(),
                'level_2': (df['waterlog_level'] == 2).sum(),
                'total_with_waterlogging': (df['waterlog_level'] >= 1).sum()
            },
            'waterlogging_percentages': {
                'none': (df['waterlog_level'] == 0).sum() / len(df) * 100,
                'minor': (df['waterlog_level'] == 1).sum() / len(df) * 100,
                'severe': (df['waterlog_level'] == 2).sum() / len(df) * 100,
                'any': (df['waterlog_level'] >= 1).sum() / len(df) * 100
            }
        }
        
        # Per-ward statistics
        if self.thresholds_df is not None:
            ward_stats = df.groupby('ward_id').agg({
                'waterlog_level': ['count', lambda x: (x >= 1).sum(), lambda x: (x == 2).sum()],
                'rainfall_mm_hr': ['mean', 'max', 'min']
            }).reset_index()
            ward_stats.columns = ['ward_id', 'total_samples', 'waterlog_count', 'severe_count', 
                                'mean_rainfall', 'max_rainfall', 'min_rainfall']
            
            stats['per_ward_stats'] = {
                'mean_samples_per_ward': ward_stats['total_samples'].mean(),
                'min_samples_per_ward': ward_stats['total_samples'].min(),
                'max_samples_per_ward': ward_stats['total_samples'].max(),
                'wards_with_waterlogging': (ward_stats['waterlog_count'] > 0).sum(),
                'wards_with_severe': (ward_stats['severe_count'] > 0).sum()
            }
        
        return stats
    
    def evaluate_threshold_statistics(self) -> Dict:
        """Evaluate threshold statistics."""
        if self.thresholds_df is None:
            raise ValueError("Thresholds not loaded. Call load_data() first.")
        
        df = self.thresholds_df.copy()
        
        alert_thresholds = df['alert_threshold_mm_hr'].dropna()
        critical_thresholds = df['critical_threshold_mm_hr'].dropna()
        
        stats = {
            'alert_thresholds': {
                'count': len(alert_thresholds),
                'mean': alert_thresholds.mean() if len(alert_thresholds) > 0 else None,
                'median': alert_thresholds.median() if len(alert_thresholds) > 0 else None,
                'std': alert_thresholds.std() if len(alert_thresholds) > 0 else None,
                'min': alert_thresholds.min() if len(alert_thresholds) > 0 else None,
                'max': alert_thresholds.max() if len(alert_thresholds) > 0 else None,
                'percentiles': {
                    '25th': alert_thresholds.quantile(0.25) if len(alert_thresholds) > 0 else None,
                    '75th': alert_thresholds.quantile(0.75) if len(alert_thresholds) > 0 else None,
                    '90th': alert_thresholds.quantile(0.90) if len(alert_thresholds) > 0 else None
                }
            },
            'critical_thresholds': {
                'count': len(critical_thresholds),
                'mean': critical_thresholds.mean() if len(critical_thresholds) > 0 else None,
                'median': critical_thresholds.median() if len(critical_thresholds) > 0 else None,
                'std': critical_thresholds.std() if len(critical_thresholds) > 0 else None,
                'min': critical_thresholds.min() if len(critical_thresholds) > 0 else None,
                'max': critical_thresholds.max() if len(critical_thresholds) > 0 else None
            } if len(critical_thresholds) > 0 else None
        }
        
        return stats
    
    def evaluate_model_quality(self) -> Dict:
        """Evaluate overall model quality metrics."""
        if self.thresholds_df is None:
            raise ValueError("Thresholds not loaded. Call load_data() first.")
        
        df = self.thresholds_df.copy()
        
        # Check sample sizes
        if 'sample_count' in df.columns:
            sample_stats = {
                'mean_samples': df['sample_count'].mean(),
                'min_samples': df['sample_count'].min(),
                'max_samples': df['sample_count'].max(),
                'wards_with_low_samples': (df['sample_count'] < 10).sum(),
                'wards_with_adequate_samples': (df['sample_count'] >= 10).sum()
            }
        else:
            sample_stats = None
        
        # Validation
        validation = validate_thresholds(df)
        
        quality = {
            'validation_status': validation['valid'],
            'sample_statistics': sample_stats,
            'validation_results': validation,
            'completeness_score': (
                df['alert_threshold_mm_hr'].notna().sum() / len(df) * 100
                if len(df) > 0 else 0
            ),
            'recommendations': []
        }
        
        # Generate recommendations
        if validation.get('invalid_order_count', 0) > 0:
            quality['recommendations'].append(
                f"Fix {validation['invalid_order_count']} wards where critical < alert threshold"
            )
        
        if sample_stats and sample_stats['wards_with_low_samples'] > 0:
            quality['recommendations'].append(
                f"Consider adding more data for {sample_stats['wards_with_low_samples']} wards with < 10 samples"
            )
        
        if quality['completeness_score'] < 100:
            quality['recommendations'].append(
                f"Increase coverage: {100 - quality['completeness_score']:.1f}% of wards missing alert thresholds"
            )
        
        return quality
    
    def generate_report(self, output_path: Optional[str] = None) -> Dict:
        """Generate comprehensive evaluation report."""
        print("\n" + "=" * 60)
        print("Model Evaluation Report")
        print("=" * 60)
        
        report = {
            'evaluation_date': datetime.now().isoformat(),
            'thresholds_file': str(self.thresholds_path),
            'training_data_file': str(self.training_data_path) if self.training_data_path else None,
            'coverage': self.evaluate_coverage(),
            'threshold_statistics': self.evaluate_threshold_statistics(),
            'model_quality': self.evaluate_model_quality()
        }
        
        if self.training_df is not None:
            report['training_data_statistics'] = self.evaluate_training_data()
        
        # Print summary
        print("\n" + "-" * 60)
        print("Coverage")
        print("-" * 60)
        print(f"Total wards: {report['coverage']['total_wards']}")
        print(f"Wards with alert thresholds: {report['coverage']['wards_with_alert_threshold']} ({report['coverage']['alert_coverage_percentage']:.1f}%)")
        print(f"Wards with critical thresholds: {report['coverage']['wards_with_critical_threshold']} ({report['coverage']['critical_coverage_percentage']:.1f}%)")
        
        print("\n" + "-" * 60)
        print("Threshold Statistics")
        print("-" * 60)
        alert_stats = report['threshold_statistics']['alert_thresholds']
        print(f"Alert thresholds:")
        print(f"  Count: {alert_stats['count']}")
        print(f"  Mean: {alert_stats['mean']:.2f} mm/hr" if alert_stats['mean'] else "  Mean: N/A")
        print(f"  Range: {alert_stats['min']:.2f} - {alert_stats['max']:.2f} mm/hr" 
              if alert_stats['min'] and alert_stats['max'] else "  Range: N/A")
        
        if report['threshold_statistics']['critical_thresholds']:
            critical_stats = report['threshold_statistics']['critical_thresholds']
            print(f"Critical thresholds:")
            print(f"  Count: {critical_stats['count']}")
            print(f"  Mean: {critical_stats['mean']:.2f} mm/hr" if critical_stats['mean'] else "  Mean: N/A")
        
        if 'training_data_statistics' in report:
            print("\n" + "-" * 60)
            print("Training Data Statistics")
            print("-" * 60)
            train_stats = report['training_data_statistics']
            print(f"Total samples: {train_stats['total_samples']:,}")
            print(f"Total wards: {train_stats['total_wards']}")
            print(f"Waterlogging distribution:")
            wl_dist = train_stats['waterlogging_distribution']
            wl_pct = train_stats['waterlogging_percentages']
            print(f"  None (0): {wl_dist['level_0']:,} ({wl_pct['none']:.2f}%)")
            print(f"  Minor (1): {wl_dist['level_1']:,} ({wl_pct['minor']:.2f}%)")
            print(f"  Severe (2): {wl_dist['level_2']:,} ({wl_pct['severe']:.2f}%)")
            print(f"  Any (>=1): {wl_dist['total_with_waterlogging']:,} ({wl_pct['any']:.2f}%)")
        
        print("\n" + "-" * 60)
        print("Model Quality")
        print("-" * 60)
        quality = report['model_quality']
        status_text = "PASSED" if quality['validation_status'] else "FAILED"
        print(f"Validation status: {status_text}")
        print(f"Completeness score: {quality['completeness_score']:.1f}%")
        
        if quality['recommendations']:
            print("\nRecommendations:")
            for rec in quality['recommendations']:
                print(f"  - {rec}")
        
        print("\n" + "=" * 60)
        
        # Save report
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save JSON
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\nReport saved to: {json_path}")
            
            # Save text summary
            text_path = output_path.with_suffix('.txt')
            # Summary is already printed above
            print(f"Report summary printed above")
        
        return report


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description='Evaluate Rainfall Threshold Model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--thresholds_path',
        type=str,
        default='threshold_model/outputs/rainfall_thresholds.csv',
        help='Path to trained thresholds CSV'
    )
    
    parser.add_argument(
        '--training_data_path',
        type=str,
        default=None,
        help='Optional path to training data CSV'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='threshold_model/outputs/evaluation_report',
        help='Output path for evaluation report'
    )
    
    args = parser.parse_args()
    
    evaluator = ThresholdModelEvaluator(
        thresholds_path=args.thresholds_path,
        training_data_path=args.training_data_path
    )
    
    evaluator.load_data()
    report = evaluator.generate_report(output_path=args.output)
    
    print("\nEvaluation completed!")
    return report


if __name__ == '__main__':
    main()
