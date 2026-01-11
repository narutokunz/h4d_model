"""
Model 1: Delhi Ward Flood Failure Predictor (DWFFP)
====================================================

Core ML model for real-time ward-level flood risk prediction.
This is the backbone of the entire flood early warning system.

Author: Delhi Flood Monitoring Team
Version: 1.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import warnings
import json
import pickle
from datetime import datetime, timedelta

# ML Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix, brier_score_loss,
    f1_score, precision_score, recall_score
)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression

# Try importing optional dependencies
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    warnings.warn("LightGBM not installed. Falling back to sklearn GradientBoosting.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    warnings.warn("SHAP not installed. Explainability features limited.")


# =============================================================================
# CONFIGURATION
# =============================================================================

def _get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


@dataclass
class ModelConfig:
    """Configuration for the flood prediction model."""
    
    # Model parameters
    model_type: str = "lightgbm"  # "lightgbm", "gradient_boosting", "random_forest"
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.05
    min_samples_leaf: int = 20
    
    # Feature parameters
    rainfall_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'light': 2.5,      # mm/hr
        'moderate': 7.5,   # mm/hr
        'heavy': 35.5,     # mm/hr
        'very_heavy': 64.5 # mm/hr
    })
    
    # Alert thresholds
    risk_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'low': 0.30,
        'moderate': 0.60,
        'high': 0.80
    })
    
    # Training parameters
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    
    # Paths - use absolute paths based on module location
    model_dir: Path = field(default_factory=lambda: _get_project_root() / "backend" / "model" / "artifacts")
    data_dir: Path = field(default_factory=lambda: _get_project_root() / "backend" / "data")


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

FEATURE_SCHEMA = {
    # Dynamic rainfall features (updated hourly)
    'dynamic_rainfall': [
        'rain_1h',           # Rainfall in last 1 hour (mm)
        'rain_3h',           # Cumulative rainfall in last 3 hours (mm)
        'rain_6h',           # Cumulative rainfall in last 6 hours (mm)
        'rain_24h',          # Antecedent precipitation index (mm)
        'rain_intensity',    # Current rainfall intensity (mm/hr)
        'rain_forecast_3h',  # Nowcast for next 3 hours (mm)
    ],
    
    # Static vulnerability features (pre-computed per ward)
    'static_vulnerability': [
        'mean_elevation',    # Mean DEM elevation (meters)
        'elevation_std',     # Elevation standard deviation (meters)
        'low_lying_pct',     # Percentage of low-lying area (%)
        'drain_density',     # Drainage density (km/km²)
        'slope_mean',        # Average terrain slope (degrees)
    ],
    
    # Historical propensity features (pre-computed)
    'historical': [
        'hist_flood_freq',      # Historical flood frequency (count)
        'monsoon_risk_score',   # Seasonal risk multiplier (0-1)
        'complaint_baseline',   # Average complaints per monsoon (count)
    ],
    
    # Temporal context features
    'temporal': [
        'hour_of_day',       # Hour (0-23)
        'day_of_monsoon',    # Days since June 1
        'is_peak_monsoon',   # Boolean: July-August
    ],
    
    # Engineered interaction features
    'interactions': [
        'rain_x_vulnerability',  # rain_3h * (1 - drain_density_norm)
        'rain_x_lowlying',       # rain_6h * low_lying_pct
        'antecedent_stress',     # rain_24h * (1 - elevation_norm)
    ]
}

# All features in order
ALL_FEATURES = (
    FEATURE_SCHEMA['dynamic_rainfall'] +
    FEATURE_SCHEMA['static_vulnerability'] +
    FEATURE_SCHEMA['historical'] +
    FEATURE_SCHEMA['temporal'] +
    FEATURE_SCHEMA['interactions']
)


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

class FeatureEngineer:
    """
    Feature engineering pipeline for flood prediction.
    Transforms raw data into model-ready features.
    """
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.scalers: Dict[str, Any] = {}
        self.feature_stats: Dict[str, Dict] = {}
        self._fitted = False
    
    def fit(self, ward_static_df: pd.DataFrame) -> 'FeatureEngineer':
        """
        Fit scalers and compute statistics from static ward data.
        
        Args:
            ward_static_df: DataFrame with static ward features
        """
        # Compute normalization statistics for static features
        for col in FEATURE_SCHEMA['static_vulnerability']:
            if col in ward_static_df.columns:
                self.feature_stats[col] = {
                    'min': ward_static_df[col].min(),
                    'max': ward_static_df[col].max(),
                    'mean': ward_static_df[col].mean(),
                    'std': ward_static_df[col].std()
                }
        
        # Fit scalers
        self.scalers['drain_density'] = MinMaxScaler()
        self.scalers['elevation'] = MinMaxScaler()
        
        if 'drain_density' in ward_static_df.columns:
            self.scalers['drain_density'].fit(ward_static_df[['drain_density']])
        if 'mean_elevation' in ward_static_df.columns:
            self.scalers['elevation'].fit(ward_static_df[['mean_elevation']])
        
        self._fitted = True
        return self
    
    def compute_rainfall_features(
        self,
        rainfall_history: pd.DataFrame,
        forecast_3h: float = 0.0
    ) -> Dict[str, float]:
        """
        Compute rainfall features from recent rainfall data.
        
        Args:
            rainfall_history: DataFrame with columns ['timestamp', 'rainfall_mm']
                             sorted by timestamp descending (most recent first)
            forecast_3h: Forecasted rainfall for next 3 hours (mm)
        
        Returns:
            Dictionary of rainfall features
        """
        if rainfall_history.empty:
            return {
                'rain_1h': 0, 'rain_3h': 0, 'rain_6h': 0, 'rain_24h': 0,
                'rain_intensity': 0, 'rain_forecast_3h': forecast_3h
            }
        
        now = rainfall_history['timestamp'].max()
        
        def sum_rainfall_hours(hours: int) -> float:
            cutoff = now - timedelta(hours=hours)
            mask = rainfall_history['timestamp'] > cutoff
            return rainfall_history.loc[mask, 'rainfall_mm'].sum()
        
        rain_1h = sum_rainfall_hours(1)
        rain_3h = sum_rainfall_hours(3)
        rain_6h = sum_rainfall_hours(6)
        rain_24h = sum_rainfall_hours(24)
        
        # Intensity = current rate (mm/hr)
        rain_intensity = rain_1h  # Simplified: last hour's rainfall
        
        return {
            'rain_1h': rain_1h,
            'rain_3h': rain_3h,
            'rain_6h': rain_6h,
            'rain_24h': rain_24h,
            'rain_intensity': rain_intensity,
            'rain_forecast_3h': forecast_3h
        }
    
    def compute_temporal_features(self, timestamp: datetime) -> Dict[str, float]:
        """
        Compute temporal context features.
        
        Args:
            timestamp: Current datetime
        
        Returns:
            Dictionary of temporal features
        """
        # Monsoon start reference (June 1)
        monsoon_start = datetime(timestamp.year, 6, 1)
        
        # Days since monsoon start
        if timestamp >= monsoon_start:
            day_of_monsoon = (timestamp - monsoon_start).days
        else:
            # If before June, use negative or 0
            day_of_monsoon = 0
        
        # Peak monsoon: July-August (months 7, 8)
        is_peak_monsoon = 1 if timestamp.month in [7, 8] else 0
        
        return {
            'hour_of_day': timestamp.hour,
            'day_of_monsoon': day_of_monsoon,
            'is_peak_monsoon': is_peak_monsoon
        }
    
    def compute_interaction_features(
        self,
        rainfall_features: Dict[str, float],
        static_features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute engineered interaction features.
        
        These capture non-linear relationships between rainfall and vulnerability.
        """
        # Normalize drain density and elevation to 0-1
        drain_density = static_features.get('drain_density', 0)
        mean_elevation = static_features.get('mean_elevation', 200)
        low_lying_pct = static_features.get('low_lying_pct', 0)
        
        # Normalize using fitted stats
        if 'drain_density' in self.feature_stats:
            dd_stats = self.feature_stats['drain_density']
            drain_density_norm = (drain_density - dd_stats['min']) / (dd_stats['max'] - dd_stats['min'] + 1e-6)
        else:
            drain_density_norm = drain_density / 10  # Assume max ~10 km/km²
        
        if 'mean_elevation' in self.feature_stats:
            elev_stats = self.feature_stats['mean_elevation']
            elevation_norm = (mean_elevation - elev_stats['min']) / (elev_stats['max'] - elev_stats['min'] + 1e-6)
        else:
            elevation_norm = (mean_elevation - 200) / 50  # Assume range 200-250m
        
        # Clip to 0-1
        drain_density_norm = np.clip(drain_density_norm, 0, 1)
        elevation_norm = np.clip(elevation_norm, 0, 1)
        
        # Interaction features
        rain_3h = rainfall_features.get('rain_3h', 0)
        rain_6h = rainfall_features.get('rain_6h', 0)
        rain_24h = rainfall_features.get('rain_24h', 0)
        
        return {
            'rain_x_vulnerability': rain_3h * (1 - drain_density_norm),
            'rain_x_lowlying': rain_6h * (low_lying_pct / 100),
            'antecedent_stress': rain_24h * (1 - elevation_norm)
        }
    
    def create_feature_vector(
        self,
        rainfall_features: Dict[str, float],
        static_features: Dict[str, float],
        historical_features: Dict[str, float],
        temporal_features: Dict[str, float]
    ) -> np.ndarray:
        """
        Combine all features into a single feature vector.
        
        Returns:
            numpy array of shape (n_features,)
        """
        # Compute interactions
        interaction_features = self.compute_interaction_features(
            rainfall_features, static_features
        )
        
        # Combine all features in schema order
        feature_vector = []
        
        for feat in FEATURE_SCHEMA['dynamic_rainfall']:
            feature_vector.append(rainfall_features.get(feat, 0))
        
        for feat in FEATURE_SCHEMA['static_vulnerability']:
            feature_vector.append(static_features.get(feat, 0))
        
        for feat in FEATURE_SCHEMA['historical']:
            feature_vector.append(historical_features.get(feat, 0))
        
        for feat in FEATURE_SCHEMA['temporal']:
            feature_vector.append(temporal_features.get(feat, 0))
        
        for feat in FEATURE_SCHEMA['interactions']:
            feature_vector.append(interaction_features.get(feat, 0))
        
        return np.array(feature_vector, dtype=np.float32)


# =============================================================================
# MAIN MODEL CLASS
# =============================================================================

class FloodFailureModel:
    """
    Main flood failure prediction model.
    
    Predicts P(failure | features) for each ward.
    Uses LightGBM with isotonic calibration for reliable probabilities.
    """
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.model = None
        self.calibrator = None
        self.feature_engineer = FeatureEngineer(self.config)
        self.feature_names = ALL_FEATURES
        self.is_fitted = False
        self._training_metrics: Dict[str, float] = {}
        self._feature_importance: Dict[str, float] = {}
    
    def _create_base_model(self):
        """Create the base classifier."""
        if self.config.model_type == "lightgbm" and HAS_LIGHTGBM:
            return lgb.LGBMClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                min_child_samples=self.config.min_samples_leaf,
                num_leaves=31,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                reg_alpha=0.1,
                reg_lambda=0.1,
                scale_pos_weight=8.0,  # Aggressively boost minority class
                random_state=self.config.random_state,
                verbose=-1
            )
        elif self.config.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_samples_leaf,
                class_weight='balanced',
                random_state=self.config.random_state,
                n_jobs=-1
            )
        else:
            # Fallback to sklearn GradientBoosting
            return GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                min_samples_leaf=self.config.min_samples_leaf,
                random_state=self.config.random_state
            )
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        calibrate: bool = True
    ) -> 'FloodFailureModel':
        """
        Train the flood prediction model.
        
        Args:
            X: Training features, shape (n_samples, n_features)
            y: Binary labels, shape (n_samples,)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            calibrate: Whether to apply probability calibration
        
        Returns:
            self
        """
        print(f"Training {self.config.model_type} model...")
        print(f"Training samples: {len(X)}, Features: {X.shape[1]}")
        print(f"Class distribution: {np.bincount(y.astype(int))}")
        
        # Split for calibration if needed
        if calibrate and X_val is None:
            X_train, X_cal, y_train, y_cal = train_test_split(
                X, y, test_size=0.2, stratify=y, 
                random_state=self.config.random_state
            )
        else:
            X_train, y_train = X, y
            X_cal, y_cal = X_val, y_val
        
        # Create and train base model
        self.model = self._create_base_model()
        self.model.fit(X_train, y_train)
        
        # Calibrate probabilities
        if calibrate and X_cal is not None:
            print("Calibrating probabilities...")
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            raw_probs = self.model.predict_proba(X_cal)[:, 1]
            self.calibrator.fit(raw_probs, y_cal)
        
        # Compute metrics
        self._compute_training_metrics(X_train, y_train, X_cal, y_cal)
        
        # Extract feature importance
        self._extract_feature_importance()
        
        self.is_fitted = True
        print("Model training complete!")
        
        return self
    
    def _compute_training_metrics(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray]
    ):
        """Compute and store training metrics."""
        # Training metrics
        y_train_proba = self.predict_proba(X_train)
        y_train_pred = (y_train_proba >= 0.5).astype(int)
        
        self._training_metrics['train_auc_roc'] = roc_auc_score(y_train, y_train_proba)
        self._training_metrics['train_auc_pr'] = average_precision_score(y_train, y_train_proba)
        self._training_metrics['train_brier'] = brier_score_loss(y_train, y_train_proba)
        self._training_metrics['train_f1'] = f1_score(y_train, y_train_pred)
        
        # Validation metrics
        if X_val is not None and y_val is not None:
            y_val_proba = self.predict_proba(X_val)
            y_val_pred = (y_val_proba >= 0.5).astype(int)
            
            self._training_metrics['val_auc_roc'] = roc_auc_score(y_val, y_val_proba)
            self._training_metrics['val_auc_pr'] = average_precision_score(y_val, y_val_proba)
            self._training_metrics['val_brier'] = brier_score_loss(y_val, y_val_proba)
            self._training_metrics['val_f1'] = f1_score(y_val, y_val_pred)
        
        print("\n=== Training Metrics ===")
        for key, value in self._training_metrics.items():
            print(f"  {key}: {value:.4f}")
    
    def _extract_feature_importance(self):
        """Extract and store feature importance."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            for i, feat in enumerate(self.feature_names):
                self._feature_importance[feat] = importances[i]
            
            # Sort and print top features
            sorted_importance = sorted(
                self._feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            print("\n=== Top 10 Feature Importances ===")
            for feat, imp in sorted_importance[:10]:
                print(f"  {feat}: {imp:.4f}")
    
    def _predict_proba_internal(self, X: np.ndarray) -> np.ndarray:
        """
        Internal prediction method that bypasses the fitted check.
        Used during training before is_fitted is set.
        """
        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Get raw probabilities
        raw_proba = self.model.predict_proba(X)[:, 1]
        
        # Apply calibration if available
        if self.calibrator is not None:
            calibrated_proba = self.calibrator.transform(raw_proba)
        else:
            calibrated_proba = raw_proba
        
        return calibrated_proba
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict calibrated probability of failure.
        
        Args:
            X: Features, shape (n_samples, n_features) or (n_features,)
        
        Returns:
            Probabilities, shape (n_samples,)
        """
        if not self.is_fitted and self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        return self._predict_proba_internal(X)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary failure outcome.
        
        Args:
            X: Features
            threshold: Decision threshold
        
        Returns:
            Binary predictions, shape (n_samples,)
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def predict_risk_level(self, X: np.ndarray) -> List[str]:
        """
        Predict risk level category for each sample.
        
        Returns:
            List of risk levels: 'low', 'moderate', 'high', 'critical'
        """
        proba = self.predict_proba(X)
        thresholds = self.config.risk_thresholds
        
        risk_levels = []
        for p in proba:
            if p < thresholds['low']:
                risk_levels.append('low')
            elif p < thresholds['moderate']:
                risk_levels.append('moderate')
            elif p < thresholds['high']:
                risk_levels.append('high')
            else:
                risk_levels.append('critical')
        
        return risk_levels
    
    def explain_prediction(
        self,
        X: np.ndarray,
        feature_values: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Provide interpretable explanation for a prediction.
        
        Args:
            X: Single feature vector
            feature_values: Optional dict of original feature values
        
        Returns:
            Dictionary with explanation components
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        probability = float(self.predict_proba(X)[0])
        risk_level = self.predict_risk_level(X)[0]
        
        # Feature contributions (simplified - using importance * value)
        contributions = {}
        for i, feat in enumerate(self.feature_names):
            importance = self._feature_importance.get(feat, 0)
            value = X[0, i]
            contribution = importance * value
            contributions[feat] = {
                'importance': importance,
                'value': float(value),
                'contribution': float(contribution)
            }
        
        # Sort by absolute contribution
        sorted_contrib = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]['contribution']),
            reverse=True
        )
        top_contributors = dict(sorted_contrib[:5])
        
        # Generate natural language explanation
        explanation_text = self._generate_explanation_text(
            probability, risk_level, top_contributors
        )
        
        return {
            'probability': probability,
            'risk_level': risk_level,
            'top_contributors': top_contributors,
            'explanation': explanation_text
        }
    
    def _generate_explanation_text(
        self,
        probability: float,
        risk_level: str,
        top_contributors: Dict[str, Dict]
    ) -> str:
        """Generate human-readable explanation."""
        # Use ASCII-compatible symbols for Windows console
        risk_symbol = {
            'low': '[LOW]',
            'moderate': '[MODERATE]',
            'high': '[HIGH]',
            'critical': '[CRITICAL]'
        }
        
        lines = [
            f"{risk_symbol.get(risk_level, '[?]')} Risk Level: {risk_level.upper()} ({probability*100:.1f}%)",
            "",
            "Key factors contributing to this risk:"
        ]
        
        for feat, data in list(top_contributors.items())[:3]:
            # Make feature names readable
            readable_name = feat.replace('_', ' ').title()
            lines.append(f"  - {readable_name}: {data['value']:.2f}")
        
        return "\n".join(lines)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance dictionary."""
        return self._feature_importance.copy()
    
    def get_training_metrics(self) -> Dict[str, float]:
        """Get training metrics dictionary."""
        return self._training_metrics.copy()
    
    def save(self, path: str = None):
        """Save model to disk."""
        if path is None:
            path = self.config.model_dir / "flood_model_v1.pkl"
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'calibrator': self.calibrator,
            'config': self.config,
            'feature_names': self.feature_names,
            'feature_importance': self._feature_importance,
            'training_metrics': self._training_metrics,
            'feature_engineer_stats': self.feature_engineer.feature_stats,
            'version': '1.0',
            'trained_at': datetime.now().isoformat()
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'FloodFailureModel':
        """Load model from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(config=model_data['config'])
        instance.model = model_data['model']
        instance.calibrator = model_data['calibrator']
        instance.feature_names = model_data['feature_names']
        instance._feature_importance = model_data['feature_importance']
        instance._training_metrics = model_data['training_metrics']
        instance.feature_engineer.feature_stats = model_data['feature_engineer_stats']
        instance.is_fitted = True
        
        print(f"Model loaded from {path}")
        print(f"  Version: {model_data.get('version', 'unknown')}")
        print(f"  Trained: {model_data.get('trained_at', 'unknown')}")
        
        return instance


# =============================================================================
# WARD PREDICTOR (HIGH-LEVEL API)
# =============================================================================

class WardFloodPredictor:
    """
    High-level API for ward-level flood predictions.
    
    This class provides the interface used by the rest of the application.
    It manages ward data, feature engineering, and batch predictions.
    """
    
    def __init__(
        self,
        model: FloodFailureModel = None,
        ward_static_data: pd.DataFrame = None,
        ward_historical_data: pd.DataFrame = None
    ):
        self.model = model or FloodFailureModel()
        self.feature_engineer = self.model.feature_engineer
        
        # Store ward data
        self.ward_static = ward_static_data
        self.ward_historical = ward_historical_data
        
        # Ward ID mapping
        self.ward_ids: List[str] = []
        if ward_static_data is not None:
            self.ward_ids = ward_static_data.index.tolist()
    
    def set_ward_data(
        self,
        static_df: pd.DataFrame,
        historical_df: pd.DataFrame = None
    ):
        """
        Set ward static and historical data.
        
        Args:
            static_df: DataFrame indexed by ward_id with static features
            historical_df: DataFrame indexed by ward_id with historical features
        """
        self.ward_static = static_df
        self.ward_historical = historical_df
        self.ward_ids = static_df.index.tolist()
        
        # Fit feature engineer on static data
        self.feature_engineer.fit(static_df)
    
    def predict_all_wards(
        self,
        rainfall_data: pd.DataFrame,
        forecast_3h: Dict[str, float] = None,
        timestamp: datetime = None
    ) -> pd.DataFrame:
        """
        Predict flood risk for all wards.
        
        Args:
            rainfall_data: DataFrame with columns ['ward_id', 'timestamp', 'rainfall_mm']
            forecast_3h: Dict mapping ward_id to 3h forecast (mm)
            timestamp: Current timestamp (defaults to now)
        
        Returns:
            DataFrame with columns:
                - ward_id
                - probability
                - risk_level
                - contributing_factors (JSON string)
        """
        timestamp = timestamp or datetime.now()
        forecast_3h = forecast_3h or {}
        
        results = []
        
        for ward_id in self.ward_ids:
            # Get ward-specific data
            static_features = self._get_static_features(ward_id)
            historical_features = self._get_historical_features(ward_id)
            
            # Filter rainfall for this ward
            ward_rainfall = rainfall_data[rainfall_data['ward_id'] == ward_id].copy()
            
            # Compute features
            rainfall_features = self.feature_engineer.compute_rainfall_features(
                ward_rainfall,
                forecast_3h=forecast_3h.get(ward_id, 0)
            )
            temporal_features = self.feature_engineer.compute_temporal_features(timestamp)
            
            # Create feature vector
            X = self.feature_engineer.create_feature_vector(
                rainfall_features,
                static_features,
                historical_features,
                temporal_features
            )
            
            # Predict
            probability = float(self.model.predict_proba(X)[0])
            risk_level = self.model.predict_risk_level(X.reshape(1, -1))[0]
            
            # Get explanation
            explanation = self.model.explain_prediction(X)
            
            results.append({
                'ward_id': ward_id,
                'probability': probability,
                'risk_level': risk_level,
                'rain_1h': rainfall_features['rain_1h'],
                'rain_3h': rainfall_features['rain_3h'],
                'explanation': explanation['explanation'],
                'timestamp': timestamp.isoformat()
            })
        
        return pd.DataFrame(results)
    
    def _get_static_features(self, ward_id: str) -> Dict[str, float]:
        """Get static features for a ward."""
        if self.ward_static is None or ward_id not in self.ward_static.index:
            return {feat: 0 for feat in FEATURE_SCHEMA['static_vulnerability']}
        
        row = self.ward_static.loc[ward_id]
        return {
            feat: row.get(feat, 0) 
            for feat in FEATURE_SCHEMA['static_vulnerability']
        }
    
    def _get_historical_features(self, ward_id: str) -> Dict[str, float]:
        """Get historical features for a ward."""
        if self.ward_historical is None or ward_id not in self.ward_historical.index:
            return {feat: 0 for feat in FEATURE_SCHEMA['historical']}
        
        row = self.ward_historical.loc[ward_id]
        return {
            feat: row.get(feat, 0)
            for feat in FEATURE_SCHEMA['historical']
        }
    
    def get_ward_risk_summary(self) -> Dict[str, int]:
        """Get count of wards in each risk category (from last prediction)."""
        # This would use cached predictions
        # For now, return placeholder
        return {
            'low': 0,
            'moderate': 0,
            'high': 0,
            'critical': 0
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_synthetic_training_data(
    n_samples: int = 5000,
    failure_rate: float = 0.15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic training data for model development.
    
    This generates realistic feature distributions based on 
    domain knowledge of Delhi flood patterns.
    
    Args:
        n_samples: Number of samples to generate
        failure_rate: Proportion of failure events
        random_state: Random seed
    
    Returns:
        X: Features array (n_samples, n_features)
        y: Labels array (n_samples,)
    """
    np.random.seed(random_state)
    
    # Number of features
    n_features = len(ALL_FEATURES)
    X = np.zeros((n_samples, n_features))
    
    # Generate base features with realistic distributions
    feature_idx = 0
    
    # === Dynamic Rainfall Features ===
    # During monsoon, rainfall follows exponential distribution
    X[:, feature_idx] = np.random.exponential(5, n_samples)  # rain_1h
    feature_idx += 1
    X[:, feature_idx] = np.random.exponential(15, n_samples)  # rain_3h
    feature_idx += 1
    X[:, feature_idx] = np.random.exponential(30, n_samples)  # rain_6h
    feature_idx += 1
    X[:, feature_idx] = np.random.exponential(50, n_samples)  # rain_24h
    feature_idx += 1
    X[:, feature_idx] = np.random.exponential(5, n_samples)  # rain_intensity
    feature_idx += 1
    X[:, feature_idx] = np.random.exponential(10, n_samples)  # rain_forecast_3h
    feature_idx += 1
    
    # === Static Vulnerability Features ===
    X[:, feature_idx] = np.random.normal(215, 15, n_samples)  # mean_elevation
    feature_idx += 1
    X[:, feature_idx] = np.random.exponential(5, n_samples)  # elevation_std
    feature_idx += 1
    X[:, feature_idx] = np.random.beta(2, 5, n_samples) * 100  # low_lying_pct (0-100)
    feature_idx += 1
    X[:, feature_idx] = np.random.exponential(3, n_samples)  # drain_density
    feature_idx += 1
    X[:, feature_idx] = np.random.exponential(2, n_samples)  # slope_mean
    feature_idx += 1
    
    # === Historical Features ===
    X[:, feature_idx] = np.random.poisson(3, n_samples)  # hist_flood_freq
    feature_idx += 1
    X[:, feature_idx] = np.random.beta(2, 3, n_samples)  # monsoon_risk_score
    feature_idx += 1
    X[:, feature_idx] = np.random.poisson(10, n_samples)  # complaint_baseline
    feature_idx += 1
    
    # === Temporal Features ===
    X[:, feature_idx] = np.random.randint(0, 24, n_samples)  # hour_of_day
    feature_idx += 1
    X[:, feature_idx] = np.random.randint(0, 120, n_samples)  # day_of_monsoon
    feature_idx += 1
    X[:, feature_idx] = np.random.binomial(1, 0.3, n_samples)  # is_peak_monsoon
    feature_idx += 1
    
    # === Interaction Features ===
    # These are computed from above features
    drain_density_norm = X[:, 9] / 10  # Normalized drain density
    elevation_norm = (X[:, 6] - 200) / 50  # Normalized elevation
    
    X[:, feature_idx] = X[:, 1] * (1 - drain_density_norm)  # rain_x_vulnerability
    feature_idx += 1
    X[:, feature_idx] = X[:, 2] * (X[:, 8] / 100)  # rain_x_lowlying
    feature_idx += 1
    X[:, feature_idx] = X[:, 3] * (1 - elevation_norm)  # antecedent_stress
    feature_idx += 1
    
    # === Generate Labels ===
    # Failure probability based on a realistic model
    logit = (
        0.1 * X[:, 0] +          # rain_1h
        0.05 * X[:, 1] +         # rain_3h
        0.03 * X[:, 3] +         # rain_24h
        -0.02 * X[:, 6] +        # mean_elevation (higher = safer)
        0.03 * X[:, 8] +         # low_lying_pct
        -0.1 * X[:, 9] +         # drain_density (higher = safer)
        0.2 * X[:, 11] +         # hist_flood_freq
        0.1 * X[:, 17] +         # rain_x_vulnerability
        0.15 * X[:, 18] +        # rain_x_lowlying
        np.random.normal(0, 0.5, n_samples)  # noise
        - 3.0  # baseline (to control failure rate)
    )
    
    prob = 1 / (1 + np.exp(-logit))
    y = (np.random.random(n_samples) < prob).astype(int)
    
    # Ensure minimum samples of each class
    actual_rate = y.mean()
    print(f"Generated {n_samples} samples with {y.sum()} failures ({actual_rate*100:.1f}%)")
    
    return X, y


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DELHI WARD FLOOD FAILURE PREDICTOR - Model Training")
    print("=" * 60)
    
    # 1. Create configuration
    config = ModelConfig()
    
    # 2. Generate synthetic training data
    print("\n[1/4] Generating synthetic training data...")
    X, y = create_synthetic_training_data(n_samples=5000, failure_rate=0.15)
    
    # 3. Split data
    print("\n[2/4] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 4. Train model
    print("\n[3/4] Training model...")
    model = FloodFailureModel(config)
    model.fit(X_train, y_train, X_val, y_val, calibrate=True)
    
    # 5. Evaluate on test set
    print("\n[4/4] Evaluating on test set...")
    y_test_proba = model.predict_proba(X_test)
    y_test_pred = model.predict(X_test)
    
    test_metrics = {
        'AUC-ROC': roc_auc_score(y_test, y_test_proba),
        'AUC-PR': average_precision_score(y_test, y_test_proba),
        'Brier Score': brier_score_loss(y_test, y_test_proba),
        'F1 Score': f1_score(y_test, y_test_pred),
        'Precision': precision_score(y_test, y_test_pred),
        'Recall': recall_score(y_test, y_test_pred)
    }
    
    print("\n=== Test Set Performance ===")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_test_pred, target_names=['No Failure', 'Failure']))
    
    # 6. Save model
    print("\n[SAVE] Saving model...")
    model.save()
    
    # 7. Demo: Single prediction with explanation
    print("\n" + "=" * 60)
    print("DEMO: Single Ward Prediction")
    print("=" * 60)
    
    sample_idx = np.where(y_test == 1)[0][0]  # Get a failure case
    sample_X = X_test[sample_idx]
    
    explanation = model.explain_prediction(sample_X)
    print(explanation['explanation'])
    
    print("\n✅ Model 1 training complete!")
    print(f"   Model saved to: {config.model_dir / 'flood_model_v1.pkl'}")
