# Model Information and Metrics

## Model Trained: Rainfall Threshold Adaptation Model

### Model Type
**Quantile-Based Threshold Learning** (NOT machine learning regression)

This model does NOT use traditional ML regression. Instead, it:
- Computes quantile thresholds per ward (80th percentile for alert, 95th percentile for critical)
- Uses historical data to learn ward-specific rainfall thresholds
- Does NOT predict values - it learns threshold values

### Why No Accuracy/R²?
Since this is a **quantile-based model** (not regression), it doesn't have:
- ❌ Accuracy percentage
- ❌ R² (R-squared)
- ❌ MSE/RMSE in traditional sense

Instead, it has:
- ✅ Coverage percentage (wards with thresholds)
- ✅ Completeness score (threshold quality)
- ✅ Sample statistics per ward

### Current Model Metrics (2022-2025 Data)

**Coverage:**
- Alert thresholds: 3/3 wards (100%)
- Critical thresholds: 0/3 wards (0% - no severe events)

**Training Data:**
- Total samples: 78,840 hourly records
- Wards: 3 (029N, 060E, 085S)
- Date range: 2022-01-01 to 2026-01-02

**Threshold Statistics:**
- Alert threshold range: 2.16 - 4.55 mm/hr
- Mean alert threshold: 3.75 mm/hr

---

## Regression Model for Validation

A **separate regression model** has been created for validation/visualization purposes.

This model:
- Uses GradientBoostingRegressor
- Predicts `waterlog_level` or `rainfall_mm_hr`
- Provides traditional ML metrics (MSE, RMSE, R²)
- Creates visualizations (scatter plots, residuals, feature importance)

### To Run Regression Model:

1. **Install dependencies** (if not already):
   ```bash
   pip install scikit-learn matplotlib
   ```

2. **Run validation script**:
   ```bash
   python threshold_model/regression_validation.py \
       --data_path threshold_model/data/threshold_training_data_2022_2025.csv \
       --target waterlog_level \
       --output_dir threshold_model/outputs/regression_validation
   ```

3. **Check results**:
   - Metrics: `threshold_model/outputs/regression_validation/metrics_waterlog_level.csv`
   - Plots: `threshold_model/outputs/regression_validation/*.png`

---

## Summary

**Threshold Model (Production):**
- Type: Quantile-based learning
- Metrics: Coverage, completeness, sample statistics
- Purpose: Learn rainfall thresholds for early warnings

**Regression Model (Validation):**
- Type: GradientBoostingRegressor (ML)
- Metrics: MSE, RMSE, R²
- Purpose: Validate approach and visualize performance
