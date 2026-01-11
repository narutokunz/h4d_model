# Model Training and Evaluation Summary

## Current Model Status (2022-2025 Data)

### Training Data
- **Years**: 2022, 2023, 2025 (2024 not available)
- **Total samples**: 78,840 hourly records
- **Wards**: 3 wards (029N, 060E, 085S)
- **Date range**: 2022-01-01 to 2026-01-02

### Model Coverage
- **Alert thresholds**: 3/3 wards (100% coverage)
- **Critical thresholds**: 0/3 wards (0% coverage - no severe waterlogging events)
- **Completeness score**: 100%

### Threshold Statistics
- **Alert threshold range**: 2.16 - 4.55 mm/hr
- **Mean alert threshold**: 3.75 mm/hr
- **Wards with adequate samples**: 3/3 (all have 26,280 samples each)

### Training Data Distribution
- **No waterlogging (level 0)**: 78,768 samples (99.91%)
- **Minor waterlogging (level 1)**: 72 samples (0.09%)
- **Severe waterlogging (level 2)**: 0 samples (0.00%)
- **Any waterlogging (>=1)**: 72 samples (0.09%)

### Model Quality Metrics
- **Validation status**: PASSED
- **Sample statistics**: All wards have adequate samples (>10 samples)
- **Coverage**: 100% of wards have alert thresholds

---

## Training on 1990-2025 Data

### Requirements

To train on 1990-2025 data (36 years), you need to:

1. **Download IMD NetCDF files** for years 1990-2025
   - Download from: https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html
   - Save to: `backend/data/raw/rainfall/`
   - Expected format: `RF25_indYYYY_rfp25.nc` (e.g., `RF25_ind1990_rfp25.nc`)

2. **Process NetCDF files** to extract ward-level rainfall
   - Use: `backend/model/ingest_rainfall.py`
   - Or: Process files manually for each year
   - Output: `backend/data/processed/rainfall_ward_daily_YYYY.csv`

3. **Create training data** with waterlogging labels
   - Use: `threshold_model/create_training_data.py`
   - Or modify to process multiple years

4. **Train the model**
   - Use: `threshold_model/train_thresholds.py`

### Estimated Data Size

For 36 years (1990-2025):
- **Days**: ~13,150 days (including leap years)
- **Wards**: ~272 Delhi wards (if all wards processed)
- **Expected records**: ~3.6 million ward-day records
- **Hourly records** (if converted): ~87 million records

---

## Model Evaluation Metrics

Since this is a **quantile-based model** (not machine learning), "accuracy" is measured differently:

### 1. Coverage Metrics
- **Alert threshold coverage**: Percentage of wards with alert thresholds
- **Critical threshold coverage**: Percentage of wards with critical thresholds
- **Completeness score**: Overall model completeness

### 2. Training Data Metrics
- **Total samples**: Number of training records
- **Samples per ward**: Average/min/max samples per ward
- **Waterlogging distribution**: Percentage of each waterlogging level
- **Wards with waterlogging events**: Number of wards with actual events

### 3. Threshold Quality Metrics
- **Threshold statistics**: Mean, median, std, range
- **Sample sizes**: Sufficient data per ward
- **Validation status**: Data quality checks

### 4. Model Performance (Future)

For production use, you can evaluate:
- **Threshold stability**: How stable are thresholds over time?
- **Event detection**: How well do thresholds detect actual waterlogging events?
- **False positive rate**: How often do thresholds trigger without events?
- **False negative rate**: How often do events occur below thresholds?

---

## Running Evaluation

To check your model's metrics:

```bash
python threshold_model/evaluate_model.py \
    --thresholds_path threshold_model/outputs/rainfall_thresholds.csv \
    --training_data_path threshold_model/data/threshold_training_data_2022_2025.csv \
    --output threshold_model/outputs/evaluation_report
```

This generates:
- **evaluation_report.json**: Detailed metrics in JSON format
- **Console output**: Summary report

---

## Current Model Limitations

1. **Limited wards**: Only 3 wards in current training data
   - **Solution**: Process NetCDF files for all Delhi wards

2. **No critical thresholds**: No severe waterlogging events (level 2) in data
   - **Solution**: Add more training data with severe events, or use lower thresholds

3. **Limited time range**: Only 3 years of data
   - **Solution**: Process data for 1990-2025 (or available years)

4. **Heuristic labels**: Waterlogging labels are rule-based
   - **Solution**: Use actual historical waterlogging reports/complaints

---

## Next Steps for 1990-2025 Training

1. **Check available NetCDF files**
   ```bash
   ls backend/data/raw/rainfall/RF25_ind*.nc
   ```

2. **Process missing years** (if files exist)
   ```bash
   cd backend/model
   python ingest_rainfall.py  # Modify to process 1990-2025
   ```

3. **Create training data for all years**
   ```python
   # Modify create_training_data.py to use years 1990-2025
   years = list(range(1990, 2026))
   ```

4. **Train model on extended data**
   ```bash
   python threshold_model/train_thresholds.py \
       --data_path threshold_model/data/threshold_training_data_1990_2025.csv \
       --output_dir threshold_model/outputs/
   ```

5. **Evaluate model**
   ```bash
   python threshold_model/evaluate_model.py \
       --thresholds_path threshold_model/outputs/rainfall_thresholds.csv \
       --training_data_path threshold_model/data/threshold_training_data_1990_2025.csv
   ```
