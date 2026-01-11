"""
Rainfall Threshold Adaptation Model

A separate model for learning ward-specific rainfall thresholds for early warnings.
This model does NOT predict flood risk directly - it only learns rainfall thresholds
that indicate alert and critical waterlogging levels.
"""

from .preprocess import ThresholdDataPreprocessor
from .threshold_trainer import RainfallThresholdTrainer
from .inference import ThresholdInference
from .utils import (
    save_thresholds_to_csv,
    load_thresholds_from_csv,
    save_thresholds_to_json,
    load_thresholds_from_json,
    validate_thresholds
)

__version__ = '1.0.0'
__all__ = [
    'ThresholdDataPreprocessor',
    'RainfallThresholdTrainer',
    'ThresholdInference',
    'save_thresholds_to_csv',
    'load_thresholds_from_csv',
    'save_thresholds_to_json',
    'load_thresholds_from_json',
    'validate_thresholds'
]
