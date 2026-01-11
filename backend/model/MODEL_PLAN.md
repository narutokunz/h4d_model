# Ward Failure Prediction Model - Implementation Plan

Based on the "Ward Failure Probability Model" specification.

## Core Objective
Predict $P(\text{failure}_{t+3h} | \text{rainfall}_{t}, \text{static\_vulnerability}, \text{dynamic\_stress})$ for each of the 250 MCD wards in Delhi.

## Methodology

### 1. Target Variable
Binary classification: `failure` (1 if waterlogging/flood reported, 0 otherwise).
Target Window: Next 3 hours.

### 2. Features
#### A. Rainfall & Temporal (Dynamic)
*   **Rain_1h, Rain_3h**: Observed accumlated rainfall.
*   **Rain_Forecast_3h**: Short-term nowcast.
*   **Antecedent_Rain_24h**: Soil saturation proxy.

#### B. Ward Vulnerability (Static)
*   **Mean_Elevation**: From CartoDEM.
*   **Drain_Density**: From Delhi Drains map.
*   **Low_Lying_Pct**: % of area in depressions.

#### C. Dynamic Stress (State-based)
*   **Pothole_Index**: Count of recent road complaints.
*   **History_Flood_Freq**: Historical failure count for this ward.

## Pipeline Architecture

### Step 1: Data Ingestion & Feature Engineering
*   **`ingest_rainfall.py`**: Parse IMD NetCDF/API.
*   **`ingest_geo.py`**: Process Ward Shapefiles and CartoDEM (zonal statistics).
*   **`ingest_complaints.py`**: Parse Praja/Civic reports into time-indexed labels.
*   **`merge_features.py`**: Create the master training table `(ward_id, timestamp) -> features...`.

### Step 2: Modeling (`train_model.py`)
*   **Algorithm**: Random Forest Classifier (v1) / Logistic Regression (baseline).
*   **Training**: Train/Test split by Time (e.g., 2021-2022 Train, 2023 Test).
*   **Evaluation**: ROC-AUC, Precision/Recall at decision thresholds.
*   **Explainability**: Feature Importance plots.

### Step 3: Inference / API (`predict.py`)
*   Serve predictions via FastAPI.

## Immediate Next Steps (Bootstrapping)
Since real datasets (NetCDF, shapefiles) are large and require manual download:
1.  Create **Synthetic Data Generator** to simulate the schema.
2.  Implement the **Training Pipeline** using synthetic data.
3.  Visualize **Feature Importance** to validate the logic.
