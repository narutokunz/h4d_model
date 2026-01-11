"""
Regression Model for Validation and Visualization

This script trains a regression model to validate the threshold model approach.
It provides traditional ML metrics (MSE, RMSE, R²) and visualizations.

NOTE: This is ONLY for validation/visualization purposes.
The production system uses the quantile-based threshold learning model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from datetime import datetime

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class RegressionValidator:
    """
    Regression model for validating threshold learning approach.
    """
    
    def __init__(self, target_col='waterlog_level'):
        """
        Initialize validator.
        
        Args:
            target_col: Target column to predict ('waterlog_level' or 'rainfall_mm_hr')
        """
        self.target_col = target_col
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load training data."""
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} records")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """
        Prepare features and target for regression.
        
        Returns:
            X (features), y (target)
        """
        df = df.copy()
        
        # Select features
        feature_cols = [
            'rainfall_mm_hr',
            'rainfall_3hr_mm',
            'rainfall_24hr_mm',
            'drainage_score'
        ]
        
        # Add categorical features (one-hot encode)
        if 'elevation_category' in df.columns:
            elevation_dummies = pd.get_dummies(df['elevation_category'], prefix='elevation')
            feature_cols.extend(elevation_dummies.columns)
            df = pd.concat([df, elevation_dummies], axis=1)
        
        if 'season' in df.columns:
            season_dummies = pd.get_dummies(df['season'], prefix='season')
            feature_cols.extend(season_dummies.columns)
            df = pd.concat([df, season_dummies], axis=1)
        
        # Select available features
        available_features = [col for col in feature_cols if col in df.columns]
        X = df[available_features].copy()
        
        # Handle missing values
        X = X.fillna(X.median() if X.select_dtypes(include=[np.number]).columns.any() else 0)
        
        # Prepare target
        if self.target_col == 'waterlog_level':
            y = df['waterlog_level'].astype(float)
        elif self.target_col == 'rainfall_mm_hr':
            y = df['rainfall_mm_hr'].astype(float)
        else:
            raise ValueError(f"Unknown target column: {self.target_col}")
        
        # Remove rows with missing target
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        self.feature_names = list(X.columns)
        
        print(f"Features: {len(self.feature_names)}")
        print(f"  {', '.join(self.feature_names)}")
        print(f"Target: {self.target_col}")
        print(f"Samples: {len(X)}")
        
        return X, y
    
    def train_model(self, X_train, y_train):
        """Train regression model."""
        print("\nTraining regression model...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Use GradientBoostingRegressor (good for interpretability)
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbose=0
        )
        
        self.model.fit(X_train_scaled, y_train)
        print("Model trained successfully!")
        
        return self.model
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        return y_pred
    
    def evaluate(self, y_true, y_pred):
        """Calculate evaluation metrics."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2
        }
        
        return metrics
    
    def plot_actual_vs_predicted(self, y_true, y_pred, output_path: Path):
        """Plot actual vs predicted scatter plot."""
        plt.figure(figsize=(10, 8))
        
        plt.scatter(y_true, y_pred, alpha=0.5, s=50)
        
        # Diagonal reference line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Values', fontsize=12, fontweight='bold')
        plt.ylabel('Predicted Values', fontsize=12, fontweight='bold')
        plt.title(f'Actual vs Predicted ({self.target_col})', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_residuals(self, y_true, y_pred, output_path: Path):
        """Plot residual error plot."""
        residuals = y_true - y_pred
        
        plt.figure(figsize=(10, 8))
        
        plt.scatter(y_pred, residuals, alpha=0.5, s=50)
        
        # Zero reference line
        plt.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Error')
        
        plt.xlabel('Predicted Values', fontsize=12, fontweight='bold')
        plt.ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
        plt.title(f'Residual Plot ({self.target_col})', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_rmse(self, rmse, output_path: Path):
        """Plot RMSE bar chart."""
        plt.figure(figsize=(8, 6))
        
        plt.bar([self.target_col], [rmse], width=0.5, color='steelblue', alpha=0.7)
        plt.ylabel('RMSE', fontsize=12, fontweight='bold')
        plt.title(f'Root Mean Squared Error ({self.target_col})', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value label on bar
        plt.text(self.target_col, rmse, f'{rmse:.4f}', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Save
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_feature_importance(self, output_path: Path):
        """Plot feature importance."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 8))
        
        # Plot top features
        top_n = min(15, len(self.feature_names))
        top_indices = indices[:top_n]
        
        plt.barh(range(top_n), importances[top_indices], color='steelblue', alpha=0.7)
        plt.yticks(range(top_n), [self.feature_names[i] for i in top_indices])
        plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
        plt.title(f'Feature Importance ({self.target_col})', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        
        # Save
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()


def main():
    """Main function to train, evaluate, and visualize regression model."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Regression Model for Validation and Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        default='threshold_model/data/threshold_training_data_2022_2025.csv',
        help='Path to training data CSV'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        default='waterlog_level',
        choices=['waterlog_level', 'rainfall_mm_hr'],
        help='Target column to predict'
    )
    
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Test set size (default: 0.2)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='threshold_model/outputs/regression_validation',
        help='Output directory for plots and results'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Regression Model Validation and Visualization")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Data path: {args.data_path}")
    print(f"  Target: {args.target}")
    print(f"  Test size: {args.test_size}")
    print(f"  Output directory: {args.output_dir}")
    print()
    
    # Initialize validator
    validator = RegressionValidator(target_col=args.target)
    
    # Step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    df = validator.load_data(args.data_path)
    X, y = validator.prepare_features(df)
    
    # Step 2: Split into train/test
    print(f"\nStep 2: Splitting data (train/test = {1-args.test_size:.1f}/{args.test_size:.1f})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, shuffle=True
    )
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Test samples: {len(X_test):,}")
    
    # Step 3: Train regression model
    print("\nStep 3: Training regression model...")
    validator.train_model(X_train, y_train)
    
    # Step 4: Generate predictions
    print("\nStep 4: Generating predictions...")
    y_train_pred = validator.predict(X_train)
    y_test_pred = validator.predict(X_test)
    
    # Step 5: Compute evaluation metrics
    print("\nStep 5: Computing evaluation metrics...")
    train_metrics = validator.evaluate(y_train, y_train_pred)
    test_metrics = validator.evaluate(y_test, y_test_pred)
    
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    print("\nTraining Set:")
    print(f"  MSE:  {train_metrics['MSE']:.6f}")
    print(f"  RMSE: {train_metrics['RMSE']:.6f}")
    print(f"  R²:   {train_metrics['R²']:.6f}")
    
    print("\nTest Set:")
    print(f"  MSE:  {test_metrics['MSE']:.6f}")
    print(f"  RMSE: {test_metrics['RMSE']:.6f}")
    print(f"  R²:   {test_metrics['R²']:.6f}")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualization: Actual vs Predicted
    print("\nGenerating visualizations...")
    print("\n1. Actual vs Predicted Scatter Plot")
    validator.plot_actual_vs_predicted(
        y_test, y_test_pred,
        output_dir / f'actual_vs_predicted_{args.target}.png'
    )
    
    # Visualization: Residual Plot
    print("\n2. Residual Error Plot")
    validator.plot_residuals(
        y_test, y_test_pred,
        output_dir / f'residuals_{args.target}.png'
    )
    
    # Visualization: RMSE Bar Chart
    print("\n3. RMSE Visualization")
    validator.plot_rmse(
        test_metrics['RMSE'],
        output_dir / f'rmse_{args.target}.png'
    )
    
    # Visualization: Feature Importance
    print("\n4. Feature Importance Plot")
    validator.plot_feature_importance(
        output_dir / f'feature_importance_{args.target}.png'
    )
    
    # Save metrics to file
    metrics_df = pd.DataFrame({
        'metric': ['MSE', 'RMSE', 'R²'],
        'train': [train_metrics['MSE'], train_metrics['RMSE'], train_metrics['R²']],
        'test': [test_metrics['MSE'], test_metrics['RMSE'], test_metrics['R²']]
    })
    metrics_path = output_dir / f'metrics_{args.target}.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to: {metrics_path}")
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print(f"\nAll plots saved to: {output_dir}")
    print(f"Target variable: {args.target}")
    print(f"Model: GradientBoostingRegressor")
    print(f"\nTest Set Performance:")
    print(f"  R² Score: {test_metrics['R²']:.4f}")
    print(f"  RMSE: {test_metrics['RMSE']:.4f}")
    print()
    
    return validator, test_metrics


if __name__ == '__main__':
    # Set matplotlib backend to non-interactive for script execution
    import matplotlib
    matplotlib.use('Agg')
    
    main()
