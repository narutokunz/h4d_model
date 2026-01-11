# Model 1: Ward Flood Failure Prediction System

## 🎯 Executive Summary

**Model Name**: Delhi Ward Flood Failure Predictor (DWFFP)  
**Version**: 1.0  
**Type**: Binary Classification with Calibrated Probability Output  
**Algorithm**: Gradient Boosted Decision Trees (LightGBM) with Isotonic Calibration  

---

## 1. Problem Formulation

### 1.1 Prediction Task
```
P(failure_{ward,t+Δt} = 1 | X_{ward,t}, S_{ward}, H_{ward})
```

Where:
- `failure` = Binary outcome (0/1)
- `ward` = One of 272 MCD wards
- `t` = Current timestamp
- `Δt` = Prediction horizon (1-3 hours)
- `X_{ward,t}` = Dynamic features (rainfall, antecedent conditions)
- `S_{ward}` = Static vulnerability features (elevation, drainage)
- `H_{ward}` = Historical propensity features

### 1.2 Definition of "Failure"
A ward experiences failure if ANY of:
1. Severe waterlogging reported
2. Drain overflow or blockage
3. Road inundation affecting mobility
4. Citizen complaint spike > 2σ above baseline
5. Historical flood-prone threshold exceeded

### 1.3 Why This Model Works

| Requirement | Solution |
|-------------|----------|
| Interpretable | Feature importance + SHAP values |
| Stable | Ensemble method reduces variance |
| Explainable | Natural language rule extraction |
| Low data dependency | Works with 8 core features |
| Real-time capable | <50ms inference per ward |

---

## 2. Feature Engineering

### 2.1 Feature Categories

#### A. Dynamic Rainfall Features (Updated Hourly)
| Feature | Description | Source | Unit |
|---------|-------------|--------|------|
| `rain_1h` | Rainfall in last 1 hour | IMD API/Gridded | mm |
| `rain_3h` | Cumulative rainfall (3h) | IMD API/Gridded | mm |
| `rain_6h` | Cumulative rainfall (6h) | IMD API/Gridded | mm |
| `rain_24h` | Antecedent precipitation (24h) | IMD API/Gridded | mm |
| `rain_intensity` | Current rainfall intensity | Derived | mm/hr |
| `rain_forecast_3h` | Nowcast for next 3h | IMD Nowcast API | mm |

#### B. Static Vulnerability Features (Pre-computed)
| Feature | Description | Source | Unit |
|---------|-------------|--------|------|
| `mean_elevation` | Mean DEM elevation | CartoDEM | meters |
| `elevation_std` | Elevation variability | CartoDEM | meters |
| `low_lying_pct` | % area below depression threshold | CartoDEM | % |
| `drain_density` | Drain length / ward area | OpenCity KML | km/km² |
| `slope_mean` | Average terrain slope | CartoDEM | degrees |

#### C. Historical Propensity Features (Pre-computed)
| Feature | Description | Source | Unit |
|---------|-------------|--------|------|
| `hist_flood_freq` | Historical flood count | INDOFLOODS/Complaints | count |
| `monsoon_risk_score` | Seasonal risk multiplier | Historical analysis | 0-1 |
| `complaint_baseline` | Avg complaints per monsoon | Praja Report | count |

#### D. Temporal Context Features
| Feature | Description | Source | Unit |
|---------|-------------|--------|------|
| `hour_of_day` | Hour (0-23) | System | int |
| `day_of_monsoon` | Days since June 1 | System | int |
| `is_peak_monsoon` | July-August flag | System | bool |

### 2.2 Feature Interactions (Engineered)
```python
# Critical interaction terms
rain_x_vulnerability = rain_3h * (1 - drain_density_normalized)
rain_x_lowlying = rain_6h * low_lying_pct
antecedent_stress = rain_24h * (1 - mean_elevation_normalized)
```

---

## 3. Model Architecture

### 3.1 Why LightGBM?

| Criterion | LightGBM Score | Alternatives |
|-----------|---------------|--------------|
| Interpretability | ⭐⭐⭐⭐ | RF: ⭐⭐⭐⭐, XGB: ⭐⭐⭐⭐, NN: ⭐ |
| Speed | ⭐⭐⭐⭐⭐ | RF: ⭐⭐⭐, XGB: ⭐⭐⭐⭐, NN: ⭐⭐ |
| Small data performance | ⭐⭐⭐⭐⭐ | RF: ⭐⭐⭐⭐, XGB: ⭐⭐⭐⭐, NN: ⭐⭐ |
| Handles missing values | ⭐⭐⭐⭐⭐ | RF: ⭐⭐, XGB: ⭐⭐⭐⭐, NN: ⭐ |
| Probability calibration | ⭐⭐⭐⭐ | All similar with post-hoc |
| Production stability | ⭐⭐⭐⭐⭐ | All similar |

### 3.2 Model Configuration
```python
model_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,          # Prevent overfitting
    'max_depth': 6,            # Interpretable depth
    'learning_rate': 0.05,     # Stable convergence
    'feature_fraction': 0.8,   # Feature sampling
    'bagging_fraction': 0.8,   # Row sampling
    'bagging_freq': 5,
    'min_child_samples': 20,   # Minimum leaf samples
    'reg_alpha': 0.1,          # L1 regularization
    'reg_lambda': 0.1,         # L2 regularization
    'class_weight': 'balanced', # Handle class imbalance
    'random_state': 42,
    'verbose': -1
}
```

### 3.3 Probability Calibration
Raw model outputs → Isotonic Regression → Calibrated P(failure)

This ensures:
- Probabilities are meaningful (0.3 means 30% chance)
- Probabilities align with observed frequencies
- Decision thresholds are reliable

---

## 4. Training Pipeline

### 4.1 Data Flow
```
Raw Data → Feature Engineering → Train/Val/Test Split → Model Training → Calibration → Evaluation
                                       ↓
                              (Time-based split)
                              Train: Historical
                              Val: Recent monsoon
                              Test: Holdout period
```

### 4.2 Cross-Validation Strategy
- **Method**: Temporal + Spatial Grouped K-Fold
- **Folds**: 5
- **Grouping**: By ward (prevent spatial leakage)
- **Time ordering**: Maintained (prevent temporal leakage)

### 4.3 Class Imbalance Handling
1. **SMOTE** for training oversampling (if needed)
2. **Class weights** in loss function
3. **Threshold tuning** for optimal F1/Recall trade-off

---

## 5. Evaluation Metrics

### 5.1 Primary Metrics
| Metric | Target | Reasoning |
|--------|--------|-----------|
| **AUC-ROC** | > 0.80 | Discrimination ability |
| **AUC-PR** | > 0.60 | Performance on minority class |
| **Brier Score** | < 0.15 | Probability calibration quality |

### 5.2 Operational Metrics
| Metric | Target | Reasoning |
|--------|--------|-----------|
| **Recall @ 0.5** | > 0.75 | Catch most failures |
| **Precision @ 0.5** | > 0.50 | Reduce false alarms |
| **F1 Score** | > 0.60 | Balance precision/recall |

### 5.3 Alert Thresholds
| Risk Level | Probability Range | Action |
|------------|------------------|--------|
| 🟢 Low | 0.00 - 0.30 | Normal monitoring |
| 🟡 Moderate | 0.30 - 0.60 | Increased vigilance |
| 🟠 High | 0.60 - 0.80 | Pre-emptive alerts |
| 🔴 Critical | 0.80 - 1.00 | Emergency response |

---

## 6. Explainability Framework

### 6.1 Global Explainability
- **Feature Importance**: Gain-based ranking
- **Permutation Importance**: Model-agnostic validation
- **Partial Dependence Plots**: Feature effect curves

### 6.2 Local Explainability (Per Prediction)
- **SHAP Values**: Additive feature contributions
- **Decision Path**: Tree traversal explanation
- **Counterfactual**: "What would change the prediction?"

### 6.3 Natural Language Explanations
```
"Ward X has 72% flood risk because:
 - Heavy rainfall (45mm in 3h) contributing +28%
 - Low-lying terrain (85th percentile) contributing +18%
 - Poor drainage density contributing +15%
 - Historical flood-prone zone contributing +11%"
```

---

## 7. Production Deployment

### 7.1 Inference Pipeline
```
IMD API → Feature Extraction → Model Inference → Calibration → Risk Score → API Response
   ↓
(Every hour)
```

### 7.2 Model Serving
- **Format**: ONNX for cross-platform compatibility
- **Latency**: <50ms per ward, <5s for all 272 wards
- **Fallback**: Rule-based scoring if model unavailable

### 7.3 Monitoring
- **Data drift**: Feature distribution monitoring
- **Prediction drift**: Score distribution shifts
- **Performance decay**: Weekly backtesting

---

## 8. Limitations & Mitigations

| Limitation | Mitigation |
|------------|------------|
| No real-time complaint data | Use historical patterns as proxy |
| Spatial mismatch (rainfall grid vs wards) | Interpolation + area-weighted averaging |
| Class imbalance (failures rare) | Synthetic data + class weights |
| Missing elevation for some areas | Imputation from neighbors |
| Model may not generalize to extreme events | Ensemble with physics-based thresholds |

---

## 9. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-11 | Initial architecture |

---

## 10. References

1. IMD Gridded Rainfall Documentation
2. INDOFLOODS Database (Zenodo)
3. CartoDEM Technical Specifications
4. LightGBM Documentation
5. SHAP: A Unified Approach to Interpreting Model Predictions
