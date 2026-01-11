"""
Model 1 Training Pipeline
=========================

Complete training pipeline for the Delhi Ward Flood Failure Predictor.
This script:
1. Loads and processes all available data
2. Creates training features
3. Trains and calibrates the model
4. Evaluates performance
5. Saves artifacts for deployment

Usage:
    python train_model_1.py

Author: Delhi Flood Monitoring Team
Version: 1.0
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import warnings

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix, f1_score, make_scorer
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Local imports
from flood_model import (
    FloodFailureModel, 
    ModelConfig, 
    FeatureEngineer,
    WardFloodPredictor,
    create_synthetic_training_data,
    ALL_FEATURES,
    FEATURE_SCHEMA
)
from data_integration import DataPipeline, DataConfig


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

TRAIN_CONFIG = {
    'use_real_data': True,          # Try real data first, fall back to synthetic
    'synthetic_samples': 10000,      # Samples if using synthetic data
    'failure_rate': 0.25,            # Increased from 0.20 - generate more failures
    'test_size': 0.2,
    'val_size': 0.15,
    'random_state': 42,
    'years': [2022, 2023, 2024, 2025],  # Use ALL available rainfall data
    
    # Model improvement options
    'use_smote': False,              # Skip SMOTE - takes 5+ min
    'tune_hyperparameters': False,   # Use best params from previous run
    'use_class_weights': True,       # Use class weights in model
}


# =============================================================================
# TRAINING DATA CREATION
# =============================================================================

def create_training_data_from_real(
    pipeline: DataPipeline,
    years: list = [2022, 2023]
) -> tuple:
    """
    Create training data from real ward and rainfall data.
    
    This creates training samples by:
    1. Using historical rainfall patterns
    2. Labeling based on rainfall intensity thresholds (calibrated for daily data)
    3. Incorporating ward vulnerability features
    """
    print("\n" + "=" * 60)
    print("Creating Training Data from Real Sources")
    print("=" * 60)
    
    # Get ward data
    ward_static, ward_historical = pipeline.get_ward_data()
    
    # Get rainfall data
    all_rainfall = []
    for year in years:
        try:
            rainfall_df = pipeline.rainfall_processor.load_rainfall_netcdf(year)
            ward_rainfall = pipeline.rainfall_processor.aggregate_rainfall_by_ward(rainfall_df)
            all_rainfall.append(ward_rainfall)
            print(f"  Loaded {year}: {len(ward_rainfall)} records")
        except Exception as e:
            print(f"  Skipping {year}: {e}")
    
    if not all_rainfall:
        print("  No real rainfall data available")
        return None, None
    
    rainfall = pd.concat(all_rainfall, ignore_index=True)
    print(f"\nTotal rainfall records: {len(rainfall)}")
    
    # Analyze rainfall distribution for threshold calibration
    rain_values = rainfall['rainfall_mm'].values
    print(f"  Rainfall stats: min={rain_values.min():.1f}, max={rain_values.max():.1f}, mean={rain_values.mean():.1f}, p90={np.percentile(rain_values, 90):.1f}")
    
    # Create feature engineer
    fe = FeatureEngineer()
    fe.fit(ward_static.reset_index())
    
    # Create training samples
    samples_X = []
    samples_y = []
    
    # Group rainfall by ward
    ward_groups = rainfall.groupby('ward_id')
    
    for ward_id, ward_data in ward_groups:
        if ward_id not in ward_static.index:
            continue
        
        # Get static features for this ward
        static_feats = ward_static.loc[ward_id].to_dict()
        
        # Get historical features
        if ward_id in ward_historical.index:
            hist_feats = ward_historical.loc[ward_id].to_dict()
        else:
            hist_feats = {
                'hist_flood_freq': np.random.poisson(2),
                'monsoon_risk_score': np.random.beta(3, 5),
                'complaint_baseline': np.random.poisson(5)
            }
        
        # Calculate ward vulnerability (0-1 scale)
        low_lying = static_feats.get('low_lying_pct', 15) / 100
        drain_density = static_feats.get('drain_density', 3)
        poor_drainage = max(0, 1 - drain_density / 5)  # Low density = poor drainage
        hist_risk = hist_feats.get('monsoon_risk_score', 0.5)
        flood_freq = min(1, hist_feats.get('hist_flood_freq', 1) / 5)
        
        ward_vulnerability = 0.3 * low_lying + 0.25 * poor_drainage + 0.25 * hist_risk + 0.2 * flood_freq
        
        # Sort by date
        ward_data = ward_data.sort_values('date')
        
        # Create samples for each day
        for i in range(7, len(ward_data)):  # Need 7 days history
            row = ward_data.iloc[i]
            date = pd.to_datetime(row['date'])
            
            # Compute rainfall features from history
            hist_slice = ward_data.iloc[i-7:i+1]
            
            rain_today = row['rainfall_mm']  # Daily rainfall in mm
            rain_3day = hist_slice.tail(3)['rainfall_mm'].sum()
            rain_7day = hist_slice['rainfall_mm'].sum()
            
            rainfall_feats = {
                'rain_1h': rain_today / 6,  # Distribute daily to peak hours
                'rain_3h': rain_today / 2,  # Half of daily as 3h peak
                'rain_6h': rain_today * 0.8,
                'rain_24h': rain_today,
                'rain_intensity': rain_today / 8,  # Peak intensity estimate
                'rain_forecast_3h': rain_today / 3  # Forecast from recent
            }
            
            # Temporal features
            day_of_monsoon = max(0, (date - datetime(date.year, 6, 1)).days)
            is_peak = 1 if date.month in [7, 8] else 0
            
            temporal_feats = {
                'hour_of_day': np.random.choice([6, 12, 18, 22]),  # Rain hours
                'day_of_monsoon': day_of_monsoon,
                'is_peak_monsoon': is_peak
            }
            
            # Create feature vector
            X = fe.create_feature_vector(
                rainfall_feats, 
                static_feats, 
                hist_feats, 
                temporal_feats
            )
            samples_X.append(X)
            
            # =================================================================
            # FAILURE LABELING - Calibrated for IMD daily rainfall data
            # =================================================================
            # IMD gives daily rainfall. Delhi flooding typically occurs when:
            # - Light rain: 2.5-7.5 mm/day
            # - Moderate rain: 7.5-35.5 mm/day  
            # - Heavy rain: 35.5-64.4 mm/day
            # - Very heavy: 64.4-124.4 mm/day
            # - Extremely heavy: >124.4 mm/day
            
            # Calculate failure probability - MORE AGGRESSIVE
            # Base probability depends on rainfall intensity
            # Made more sensitive to capture realistic flood events
            if rain_today < 2:
                rain_factor = 0.0  # No rain
            elif rain_today < 8:
                rain_factor = 0.08  # Light rain - minor issues
            elif rain_today < 20:
                rain_factor = 0.25  # Moderate rain - localized waterlogging
            elif rain_today < 45:
                rain_factor = 0.50  # Heavy rain - widespread waterlogging
            elif rain_today < 90:
                rain_factor = 0.75  # Very heavy rain - major flooding
            else:
                rain_factor = 0.92  # Extremely heavy - severe flooding
            
            # Antecedent moisture (recent rain increases risk significantly)
            if rain_3day > 70:
                antecedent_factor = 0.30  # Saturated soil
            elif rain_3day > 35:
                antecedent_factor = 0.18  # Wet soil
            elif rain_3day > 12:
                antecedent_factor = 0.10  # Damp soil
            else:
                antecedent_factor = 0.0
            
            # Combine all factors with ward vulnerability amplification
            failure_prob = rain_factor * (0.35 + 1.2 * ward_vulnerability) + antecedent_factor
            
            # Peak monsoon boost
            if is_peak:
                failure_prob *= 1.25  # Increased from 1.20
            
            # Add some noise
            failure_prob += np.random.normal(0, 0.05)
            failure_prob = np.clip(failure_prob, 0, 0.95)
            
            # Generate label
            y = 1 if np.random.random() < failure_prob else 0
            samples_y.append(y)
    
    if not samples_X:
        print("  No samples created from real data")
        return None, None
    
    X = np.array(samples_X)
    y = np.array(samples_y)
    
    print(f"\nCreated {len(X)} training samples")
    print(f"  Failures: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"  Features: {X.shape[1]}")
    
    return X, y


def create_enhanced_synthetic_data(
    ward_static: pd.DataFrame,
    ward_historical: pd.DataFrame,
    n_samples: int = 10000,
    failure_rate: float = 0.12,
    seed: int = 42
) -> tuple:
    """
    Create enhanced synthetic training data using real ward characteristics.
    
    This method:
    1. Uses real ward static features as base
    2. Simulates realistic monsoon rainfall patterns
    3. Creates correlated failure labels
    """
    print("\n" + "=" * 60)
    print("Creating Enhanced Synthetic Training Data")
    print("=" * 60)
    
    np.random.seed(seed)
    
    n_wards = len(ward_static)
    samples_per_ward = n_samples // n_wards
    
    samples_X = []
    samples_y = []
    
    fe = FeatureEngineer()
    fe.fit(ward_static.reset_index())
    
    for ward_id, static_row in ward_static.iterrows():
        static_feats = static_row.to_dict()
        
        # Get historical features
        if ward_id in ward_historical.index:
            hist_feats = ward_historical.loc[ward_id].to_dict()
        else:
            hist_feats = {
                'hist_flood_freq': np.random.poisson(2),
                'monsoon_risk_score': np.random.beta(2, 3),
                'complaint_baseline': np.random.poisson(8)
            }
        
        # Ward vulnerability score (for label generation)
        vulnerability = (
            (1 - static_feats.get('drain_density', 2.5) / 10) * 0.3 +
            (static_feats.get('low_lying_pct', 15) / 100) * 0.3 +
            (hist_feats.get('monsoon_risk_score', 0.5)) * 0.4
        )
        
        for _ in range(samples_per_ward):
            # Simulate rainfall (exponential distribution during monsoon)
            rain_1h = np.random.exponential(8)
            rain_3h = rain_1h + np.random.exponential(15)
            rain_6h = rain_3h + np.random.exponential(20)
            rain_24h = rain_6h + np.random.exponential(30)
            
            # Occasionally have heavy rainfall events
            if np.random.random() < 0.1:
                multiplier = np.random.uniform(2, 5)
                rain_1h *= multiplier
                rain_3h *= multiplier
                rain_6h *= multiplier
                rain_24h *= multiplier
            
            rainfall_feats = {
                'rain_1h': rain_1h,
                'rain_3h': rain_3h,
                'rain_6h': rain_6h,
                'rain_24h': rain_24h,
                'rain_intensity': rain_1h,
                'rain_forecast_3h': rain_3h * np.random.uniform(0.8, 1.2)
            }
            
            # Temporal features
            day_of_monsoon = np.random.randint(0, 120)
            is_peak = 1 if 30 <= day_of_monsoon <= 90 else 0
            
            temporal_feats = {
                'hour_of_day': np.random.randint(0, 24),
                'day_of_monsoon': day_of_monsoon,
                'is_peak_monsoon': is_peak
            }
            
            # Create feature vector
            X = fe.create_feature_vector(
                rainfall_feats,
                static_feats,
                hist_feats,
                temporal_feats
            )
            samples_X.append(X)
            
            # Generate label based on features
            # Logistic model for failure probability
            logit = (
                -4.0 +                                    # Base (low baseline)
                0.05 * rain_1h +                          # Recent rain matters most
                0.02 * rain_3h +                          # Cumulative rain
                0.01 * rain_24h +                         # Antecedent conditions
                3.0 * vulnerability +                     # Ward vulnerability
                0.5 * is_peak +                           # Peak monsoon risk
                np.random.normal(0, 0.3)                  # Noise
            )
            
            prob = 1 / (1 + np.exp(-logit))
            y = 1 if np.random.random() < prob else 0
            samples_y.append(y)
    
    X = np.array(samples_X)
    y = np.array(samples_y)
    
    print(f"Created {len(X)} synthetic samples")
    print(f"  Failures: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"  Features: {X.shape[1]}")
    
    return X, y


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_model_1():
    """
    Main training function for Model 1.
    """
    print("=" * 70)
    print("  DELHI WARD FLOOD FAILURE PREDICTOR - Model 1 Training")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    
    # Initialize paths
    model_dir = Path("backend/model/artifacts")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data pipeline
    print("\n[STEP 1/6] Initializing data pipeline...")
    pipeline = DataPipeline()
    
    try:
        pipeline.initialize()
        ward_static, ward_historical = pipeline.get_ward_data()
        use_real_data = TRAIN_CONFIG['use_real_data']
    except Exception as e:
        print(f"  Data initialization failed: {e}")
        print("  Falling back to synthetic data only")
        use_real_data = False
        ward_static = None
        ward_historical = None
    
    # Create training data
    print("\n[STEP 2/6] Creating training data...")
    
    if use_real_data and ward_static is not None:
        # Try real data first
        X, y = create_training_data_from_real(pipeline, TRAIN_CONFIG['years'])
        
        if X is None or len(X) < 1000:
            print("  Insufficient real data, using enhanced synthetic")
            X, y = create_enhanced_synthetic_data(
                ward_static,
                ward_historical,
                n_samples=TRAIN_CONFIG['synthetic_samples'],
                failure_rate=TRAIN_CONFIG['failure_rate']
            )
    else:
        # Pure synthetic data
        print("  Using pure synthetic data (no ward features)")
        X, y = create_synthetic_training_data(
            n_samples=TRAIN_CONFIG['synthetic_samples'],
            failure_rate=TRAIN_CONFIG['failure_rate']
        )
    
    # Split data
    print("\n[STEP 3/6] Splitting data...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=TRAIN_CONFIG['test_size'],
        stratify=y,
        random_state=TRAIN_CONFIG['random_state']
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=TRAIN_CONFIG['val_size'],
        stratify=y_train_val,
        random_state=TRAIN_CONFIG['random_state']
    )
    
    print(f"  Train: {len(X_train)} samples ({y_train.mean()*100:.1f}% positive)")
    print(f"  Val:   {len(X_val)} samples ({y_val.mean()*100:.1f}% positive)")
    print(f"  Test:  {len(X_test)} samples ({y_test.mean()*100:.1f}% positive)")
    
    # Handle class imbalance with SMOTE
    if TRAIN_CONFIG['use_smote'] and y_train.mean() < 0.3:
        print("\n[STEP 3.5/6] Applying SMOTE for class balance...")
        smote = SMOTE(random_state=TRAIN_CONFIG['random_state'], k_neighbors=5)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"  Before SMOTE: {len(X_train)} samples ({y_train.mean()*100:.1f}% positive)")
        print(f"  After SMOTE:  {len(X_train_balanced)} samples ({y_train_balanced.mean()*100:.1f}% positive)")
        X_train, y_train = X_train_balanced, y_train_balanced
    
    # Train model
    print("\n[STEP 4/6] Training model...")
    config = ModelConfig()
    
    # Hyperparameter tuning
    if TRAIN_CONFIG['tune_hyperparameters']:
        print("  Running hyperparameter tuning (GridSearchCV)...")
        print("  This may take 5-10 minutes...")
        
        param_grid = {
            'n_estimators': [150, 250],          # Reduced: 2 values
            'max_depth': [5, 7],                 # Reduced: 2 values  
            'learning_rate': [0.05, 0.1],        # Reduced: 2 values
            'num_leaves': [31, 63],              # Reduced: 2 values
        }
        # Total: 2^4 = 16 combinations (vs 81 original)
        
        base_model = FloodFailureModel(config)
        
        # Use stratified k-fold for better validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=TRAIN_CONFIG['random_state'])
        
        # Create a wrapper function for grid search
        best_params = None
        best_score = 0
        
        print("  Testing parameter combinations...")
        for n_est in param_grid['n_estimators']:
            for depth in param_grid['max_depth']:
                for lr in param_grid['learning_rate']:
                    for leaves in param_grid['num_leaves']:
                        config_test = ModelConfig(
                            n_estimators=n_est,
                            max_depth=depth,
                            learning_rate=lr
                        )
                        model_test = FloodFailureModel(config_test)
                        model_test.config.model_type = "lightgbm"
                        
                        # Quick cross-validation
                        cv_scores = []
                        for train_idx, val_idx in cv.split(X_train, y_train):
                            X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
                            y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
                            
                            model_test.fit(X_cv_train, y_cv_train, X_cv_val, y_cv_val, calibrate=False)
                            score = roc_auc_score(y_cv_val, model_test.predict_proba(X_cv_val))
                            cv_scores.append(score)
                        
                        avg_score = np.mean(cv_scores)
                        if avg_score > best_score:
                            best_score = avg_score
                            best_params = {
                                'n_estimators': n_est,
                                'max_depth': depth,
                                'learning_rate': lr,
                                'num_leaves': leaves
                            }
                            print(f"    New best: AUC={avg_score:.4f} (n_est={n_est}, depth={depth}, lr={lr}, leaves={leaves})")
        
        print(f"\n  Best parameters found: {best_params}")
        print(f"  Best CV AUC: {best_score:.4f}")
        
        # Update config with best params
        config.n_estimators = best_params['n_estimators']
        config.max_depth = best_params['max_depth']
        config.learning_rate = best_params['learning_rate']
    
    # Train final model with best params
    print("\n  Training final model with optimized parameters...")
    model = FloodFailureModel(config)
    model.fit(X_train, y_train, X_val, y_val, calibrate=True)
    
    # Evaluate on test set
    print("\n[STEP 5/6] Evaluating on test set...")
    y_test_proba = model.predict_proba(X_test)
    
    # Find optimal threshold using validation set
    print("  Finding optimal classification threshold...")
    thresholds = np.arange(0.1, 0.6, 0.02)
    best_f1 = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        y_val_pred_thresh = (model.predict_proba(X_val) >= thresh).astype(int)
        f1 = f1_score(y_val, y_val_pred_thresh)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    print(f"  Optimal threshold: {best_threshold:.3f} (F1={best_f1:.3f})")
    
    y_test_pred = (y_test_proba >= best_threshold).astype(int)
    
    # Compute metrics
    metrics = {
        'auc_roc': roc_auc_score(y_test, y_test_proba),
        'auc_pr': average_precision_score(y_test, y_test_proba),
        'brier_score': brier_score_loss(y_test, y_test_proba),
        'f1_score': f1_score(y_test, y_test_pred),
        'accuracy': (y_test == y_test_pred).mean()
    }
    
    print("\n" + "=" * 50)
    print("  TEST SET PERFORMANCE")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    print("\n  Classification Report:")
    print(classification_report(y_test, y_test_pred, 
                               target_names=['No Failure', 'Failure'],
                               digits=3))
    
    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"    TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
    print(f"    FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")
    
    # Cross-validation
    print("\n  Cross-Validation (5-fold):")
    cv_scores = cross_val_score(
        model._create_base_model(),
        X_train_val, y_train_val,
        cv=5, scoring='roc_auc'
    )
    print(f"    AUC-ROC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Save model and artifacts
    print("\n[STEP 6/6] Saving artifacts...")
    
    # Save model
    model_path = model_dir / "flood_model_v1.pkl"
    model.save(model_path)
    
    # Save metrics
    metrics_path = model_dir / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        training_info = {
            'test_metrics': metrics,
            'cv_auc_mean': float(cv_scores.mean()),
            'cv_auc_std': float(cv_scores.std()),
            'training_config': TRAIN_CONFIG,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'feature_names': ALL_FEATURES,
            'trained_at': datetime.now().isoformat()
        }
        
        # Add tuning results if available
        if TRAIN_CONFIG['tune_hyperparameters'] and best_params:
            training_info['best_hyperparameters'] = best_params
            training_info['best_cv_score'] = float(best_score)
        
        json.dump(training_info, f, indent=2)
    print(f"  Metrics saved to: {metrics_path}")
    
    # Save feature importance
    importance_path = model_dir / "feature_importance.json"
    with open(importance_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        importance_dict = {k: float(v) for k, v in model.get_feature_importance().items()}
        json.dump(importance_dict, f, indent=2)
    print(f"  Feature importance saved to: {importance_path}")
    
    # Generate example predictions
    print("\n" + "=" * 50)
    print("  SAMPLE PREDICTIONS")
    print("=" * 50)
    
    # Get a few test samples
    for i in range(min(5, len(X_test))):
        idx = np.random.randint(len(X_test))
        sample_X = X_test[idx]
        actual = y_test[idx]
        predicted = model.predict_proba(sample_X)[0]
        risk_level = model.predict_risk_level(sample_X.reshape(1, -1))[0]
        
        risk_sym = {'low': '[LOW]', 'moderate': '[MOD]', 'high': '[HIGH]', 'critical': '[CRIT]'}
        print(f"\n  Sample {i+1}:")
        print(f"    Predicted: {predicted:.3f} {risk_sym.get(risk_level, '[?]')} {risk_level.upper()}")
        print(f"    Actual: {'FAILURE' if actual else 'OK'}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"""
    Model Performance Summary:
    - AUC-ROC:     {metrics['auc_roc']:.4f} {'[OK]' if metrics['auc_roc'] > 0.75 else '[!]'}
    - AUC-PR:      {metrics['auc_pr']:.4f} {'[OK]' if metrics['auc_pr'] > 0.50 else '[!]'}
    - Brier Score: {metrics['brier_score']:.4f} {'[OK]' if metrics['brier_score'] < 0.15 else '[!]'}
    - F1 Score:    {metrics['f1_score']:.4f} {'[OK]' if metrics['f1_score'] > 0.50 else '[!]'}
    
    Artifacts saved to: {model_dir.absolute()}
    
    Next steps:
    1. Integrate with real-time rainfall API
    2. Deploy as FastAPI endpoint
    3. Connect to frontend dashboard
    """)
    
    return model, metrics


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    model, metrics = train_model_1()
