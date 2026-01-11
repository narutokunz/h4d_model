# Rainfall Threshold Adaptation Model

A separate model for learning ward-specific rainfall thresholds for early warnings in Delhi's water-logging early warning platform.

## Overview

This model **does NOT directly predict flood risk**. Instead, it learns ward-specific rainfall thresholds that indicate:
- **Alert-level waterlogging**: Threshold for minor waterlogging events
- **Critical-level waterlogging**: Threshold for severe waterlogging events

### Why Separate?
- Keeps the main flood risk model stable
- Allows continuous learning per ward
- Provides interpretable, actionable thresholds

## Model Architecture

The model uses **quantile-based learning**:
- For each ward, compute rainfall thresholds from historical data
- Alert threshold: 80th percentile of `rainfall_mm_hr` where `waterlog_level >= 1`
- Critical threshold: 95th percentile of `rainfall_mm_hr` where `waterlog_level == 2`
- Includes fallback strategies for wards with insufficient data

## Data Requirements

The input dataset should be a CSV file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `ward_id` | categorical | Ward identifier (e.g., "W12") |
| `rainfall_mm_hr` | float | Rainfall in mm/hour |
| `rainfall_3hr_mm` | float | 3-hour cumulative rainfall (mm) |
| `rainfall_24hr_mm` | float | 24-hour cumulative rainfall (mm) |
| `drainage_score` | float (0-1) | Drainage quality score (lower = poorer) |
| `elevation_category` | categorical | Low / Medium / High |
| `season` | categorical | Monsoon / Non-monsoon |
| `waterlog_level` | integer | Target label: 0 = none, 1 = minor, 2 = severe |

## Installation

Dependencies:
- pandas
- numpy
- xarray (for NetCDF processing)
- geopandas (for spatial operations)
- scikit-learn (optional, for future enhancements)

## Data Preparation

**Important:** The threshold model requires both rainfall data AND waterlogging labels.

### Step 1: Download IMD Rainfall Data

Download IMD NetCDF files for the last 5 years (2020-2024) from:
- 0.25° resolution: https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html
- 1.0° resolution: https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_1_NetCDF.html

Save files to: `backend/data/raw/rainfall/`

Expected filename format:
- 0.25°: `RF25_indYYYY_rfp25.nc` (e.g., `RF25_ind2020_rfp25.nc`)
- 1.0°: `RF1_indYYYY_rfp1.nc` (e.g., `RF1_ind2020_rfp1.nc`)

### Step 2: Prepare Training Data

Run the data preparation script:

```bash
python threshold_model/prepare_data.py --years 2020 2021 2022 2023 2024 --resolution 0.25
```

This will:
1. Process IMD NetCDF files and extract ward-level rainfall
2. Create hourly features (rainfall_mm_hr, rainfall_3hr_mm, rainfall_24hr_mm)
3. Add static features (drainage_score, elevation_category)
4. Add season feature
5. Create training dataset format

**Note:** The script will create a dataset without waterlogging labels. You need to add waterlogging labels (waterlog_level: 0=none, 1=minor, 2=severe) before training.

### Step 3: Add Waterlogging Labels

You need to add waterlogging labels to the prepared dataset. The labels should be based on:
- Historical flood/waterlogging reports
- Citizen complaints
- News reports
- Government records

The dataset must have a `waterlog_level` column with values:
- `0`: No waterlogging
- `1`: Minor waterlogging
- `2`: Severe waterlogging

## Usage

### Training

Train the model on historical data:

```bash
python train_thresholds.py --data_path data/historical_data.csv --output_dir outputs/
```

**Arguments:**
- `--data_path`: Path to input CSV file (required)
- `--output_dir`: Output directory (default: `outputs/`)
- `--alert_quantile`: Quantile for alert threshold (default: 0.80)
- `--critical_quantile`: Quantile for critical threshold (default: 0.95)
- `--min_samples`: Minimum samples per ward (default: 5)
- `--fallback_strategy`: Strategy for small-data wards: `global`, `ward_mean`, or `skip` (default: `global`)
- `--format`: Output format: `csv`, `json`, or `both` (default: `both`)

**Example:**
```bash
python train_thresholds.py \
    --data_path data/historical_waterlogging.csv \
    --output_dir outputs/ \
    --alert_quantile 0.80 \
    --critical_quantile 0.95 \
    --fallback_strategy global
```

### Inference

Use trained thresholds for early warnings:

```python
from threshold_model.inference import ThresholdInference

# Load thresholds
inference = ThresholdInference()
inference.load_thresholds('outputs/rainfall_thresholds.csv', format='csv')

# Get alert level for a ward
alert_level = inference.get_alert_level('W12', forecast_rainfall=15.5)
# Returns: 'LOW', 'MEDIUM', or 'HIGH'

# Batch prediction
forecast_dict = {
    'W12': 15.5,
    'W13': 25.0,
    'W14': 8.0
}
results = inference.batch_predict(forecast_dict)
```

### Example Script

Run the example inference script:

```bash
python example_inference.py --thresholds_path outputs/rainfall_thresholds.csv
```

For interactive testing:

```bash
python example_inference.py --interactive
```

## Output Format

The model outputs a CSV/JSON file with the following columns:

| Column | Description |
|--------|-------------|
| `ward_id` | Ward identifier |
| `alert_threshold_mm_hr` | Alert-level rainfall threshold (mm/hr) |
| `critical_threshold_mm_hr` | Critical-level rainfall threshold (mm/hr) |
| `sample_count` | Number of samples used for training |
| `last_updated` | Timestamp of last update |

## Alert Levels

Given a forecast rainfall value, the model returns:

- **LOW**: Rainfall < alert threshold
- **MEDIUM**: Alert threshold ≤ Rainfall < critical threshold
- **HIGH**: Rainfall ≥ critical threshold

## Module Structure

```
threshold_model/
├── __init__.py              # Package initialization
├── preprocess.py            # Data preprocessing
├── threshold_trainer.py     # Threshold training logic
├── inference.py             # Inference and prediction
├── utils.py                 # Utility functions (save/load)
├── train_thresholds.py      # Main training script
├── example_inference.py     # Example usage
└── README.md                # This file
```

## Key Features

- ✅ Quantile-based learning (interpretable)
- ✅ Ward-specific thresholds
- ✅ Fallback strategies for small-data wards
- ✅ Production-ready code with error handling
- ✅ Clean output format (CSV/JSON)
- ✅ Batch and single prediction support
- ✅ No deep learning (lightweight, fast)

## Notes

- This model is **separate** from the main flood risk model
- It does NOT predict flood probability - only learns thresholds
- Designed for government-grade urban flood management systems
- Handles data quality issues gracefully
- Suitable for continuous learning and updates

## License

Part of the Delhi Water-Logging Early Warning Platform project.
