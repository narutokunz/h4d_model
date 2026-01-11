# Data Preparation Guide for Rainfall Threshold Model

This guide explains how to prepare training data for the Rainfall Threshold Adaptation Model using IMD rainfall data (2020-2024).

## Overview

The threshold model requires:
1. **Rainfall data** (from IMD NetCDF files, 2020-2024)
2. **Waterlogging labels** (from your data sources)

## Step 1: Download IMD Rainfall Data

Download IMD gridded rainfall NetCDF files from:
- **0.25° resolution**: https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html
- **1.0° resolution**: https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_1_NetCDF.html

### Download Years
Download data for **2020, 2021, 2022, 2023, 2024** (last 5 years).

### File Format
- **0.25° files**: `RF25_indYYYY_rfp25.nc` (e.g., `RF25_ind2020_rfp25.nc`)
- **1.0° files**: `RF1_indYYYY_rfp1.nc` (e.g., `RF1_ind2020_rfp1.nc`)

### Save Location
Save all `.nc` files to:
```
backend/data/raw/rainfall/
```

## Step 2: Run Data Preparation Script

Run the preparation script:

```bash
cd threshold_model
python prepare_data.py --years 2020 2021 2022 2023 2024 --resolution 0.25
```

### Arguments:
- `--years`: Years to process (default: 2020-2024)
- `--resolution`: IMD resolution - `0.25` or `1.0` (default: `0.25`)
- `--waterlog_labels`: Optional path to waterlogging labels CSV

### Output
The script creates:
- `threshold_model/data/threshold_training_data_2020_2024.csv`

**Important:** This dataset will have `waterlog_level = 0` for all records. You need to add actual waterlogging labels.

## Step 3: Add Waterlogging Labels

The training dataset requires a `waterlog_level` column with values:
- `0`: No waterlogging
- `1`: Minor waterlogging  
- `2`: Severe waterlogging

### Sources for Waterlogging Labels

1. **Historical flood reports** (if available in your system)
2. **Citizen complaints** (Praja, OpenCity, etc.)
3. **News reports** (flood/waterlogging events)
4. **Government records** (MCD, NDMC, etc.)
5. **Social media data** (Twitter, Facebook)

### Label Assignment Logic

For each ward-date combination:
- `waterlog_level = 0`: No reports of waterlogging
- `waterlog_level = 1`: Minor waterlogging reported (some complaints, minor flooding)
- `waterlog_level = 2`: Severe waterlogging reported (major flooding, significant disruption)

### Example Label File Format

Your label file should have columns:
```csv
ward_id,date,waterlog_level
012E,2020-07-15,1
012E,2020-08-20,2
011E,2020-07-15,0
...
```

Then merge with the prepared dataset:
```python
import pandas as pd

# Load prepared data
df = pd.read_csv('threshold_model/data/threshold_training_data_2020_2024.csv')

# Load labels
labels = pd.read_csv('path/to/waterlogging_labels.csv')

# Merge
df = df.merge(labels, on=['ward_id', 'date'], how='left')
df['waterlog_level'] = df['waterlog_level'].fillna(0).astype(int)

# Save
df.to_csv('threshold_model/data/training_data_with_labels.csv', index=False)
```

## Step 4: Train the Model

Once you have the complete dataset with waterlogging labels:

```bash
python train_thresholds.py \
    --data_path threshold_model/data/training_data_with_labels.csv \
    --output_dir outputs/
```

## Data Requirements Summary

The final training dataset must have these columns:

| Column | Type | Description |
|--------|------|-------------|
| `ward_id` | string | Ward identifier (e.g., "012E") |
| `rainfall_mm_hr` | float | Rainfall in mm/hour |
| `rainfall_3hr_mm` | float | 3-hour cumulative rainfall (mm) |
| `rainfall_24hr_mm` | float | 24-hour cumulative rainfall (mm) |
| `drainage_score` | float (0-1) | Drainage quality score |
| `elevation_category` | string | "low", "medium", or "high" |
| `season` | string | "monsoon" or "non-monsoon" |
| `waterlog_level` | int | Target label: 0, 1, or 2 |

## Troubleshooting

### Issue: NetCDF files not found
- Ensure files are downloaded from IMD website
- Check file paths and names match expected format
- Verify files are in `backend/data/raw/rainfall/`

### Issue: Wards shapefile not found
- Ensure Delhi wards shapefile exists
- Expected location: `backend/data/raw/wards/delhi_ward.shp`
- The script will look in common locations

### Issue: No waterlogging labels
- The model cannot be trained without waterlogging labels
- Collect labels from available sources (see Step 3)
- Consider using historical flood data if available

### Issue: Insufficient data per ward
- Minimum 5 samples per ward recommended
- Wards with < 5 samples will use fallback thresholds
- Consider combining years or using global fallback

## References

- IMD Rainfall Data: https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html
- Data Citation: Pai et al. (2014) for 0.25° data
