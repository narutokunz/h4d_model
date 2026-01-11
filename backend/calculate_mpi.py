"""
Ward-wise MPI (Multi-Parameter Index) Calculator
=================================================

Combines multiple data sources to create comprehensive flood risk scores:
1. ML Model predictions (based on rainfall + ward features)
2. Real-time weather data (OpenWeather API)
3. Historical flood frequency (INDOFLOODS)
4. Civic complaints/infrastructure issues
5. Drainage capacity
6. Elevation vulnerability

MPI Score (0-100): Weighted combination of all factors
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "model"))

from flood_model import FloodFailureModel, FeatureEngineer
from data_integration import DataPipeline
sys.path.insert(0, str(Path(__file__).parent))
from weather_api import fetch_current_weather_delhi, fetch_forecast_delhi, calculate_rainfall_features


class MPICalculator:
    """Calculate Multi-Parameter Index for each ward."""
    
    def __init__(self):
        """Initialize MPI calculator with model and data."""
        self.model = None
        self.feature_engineer = None
        self.ward_static = None
        self.ward_historical = None
        self.civic_complaints = None
        
    def load_model(self, model_path: str = "backend/model/artifacts/flood_model_v1.pkl"):
        """Load trained flood prediction model."""
        print("Loading flood prediction model...")
        try:
            self.model = FloodFailureModel.load(model_path)
            self.feature_engineer = self.model.feature_engineer
            print("[OK] Model loaded")
            return True
        except Exception as e:
            print(f"[ERROR] Error loading model: {e}")
            return False
    
    def load_ward_data(self):
        """Load ward static and historical features."""
        print("Loading ward data...")
        try:
            pipeline = DataPipeline()
            pipeline.initialize()
            self.ward_static, self.ward_historical = pipeline.get_ward_data()
            print(f"[OK] Loaded {len(self.ward_static)} wards")
            return True
        except Exception as e:
            print(f"[ERROR] Error loading ward data: {e}")
            return False
    
    def load_civic_complaints(self):
        """
        Load civic complaints data.
        
        This data comes from: Report on The Status of Civic Issues in Delhi 2023.pdf
        Format expected: CSV with columns [ward_id, drainage_complaints, sewerage_complaints, pothole_complaints, year]
        
        TODO: Extract data from PDF manually or using OCR
        For now, using estimated values based on ward characteristics
        """
        print("Loading civic complaints data...")
        
        complaints_path = Path("backend/data/processed/civic_complaints_ward.csv")
        
        if complaints_path.exists():
            self.civic_complaints = pd.read_csv(complaints_path, index_col='ward_id')
            print(f"[OK] Loaded complaints for {len(self.civic_complaints)} wards")
        else:
            print("[WARNING] Civic complaints CSV not found - using estimates")
            # Generate estimated complaints based on historical flood frequency
            self.civic_complaints = self.ward_historical.copy()
            self.civic_complaints['drainage_complaints'] = (
                self.civic_complaints['hist_flood_freq'] * 15 + 
                np.random.poisson(10, len(self.civic_complaints))
            )
            self.civic_complaints['sewerage_complaints'] = np.random.poisson(8, len(self.civic_complaints))
            self.civic_complaints['pothole_complaints'] = np.random.poisson(12, len(self.civic_complaints))
            
        return True
    
    def get_real_time_rainfall(self) -> dict:
        """Fetch real-time rainfall from OpenWeather API."""
        print("Fetching real-time weather...")
        try:
            current = fetch_current_weather_delhi()
            forecast = fetch_forecast_delhi()
            rainfall_features = calculate_rainfall_features(current, forecast)
            print(f"[OK] Current rainfall: {rainfall_features['rain_1h']:.1f}mm/h")
            return rainfall_features
        except Exception as e:
            print(f"[WARNING] Weather API error: {e}")
            # Return default minimal rainfall
            return {
                'rain_1h': 0,
                'rain_3h': 0,
                'rain_6h': 0,
                'rain_24h': 0,
                'rain_intensity': 0,
                'rain_forecast_3h': 0
            }
    
    def calculate_mpi(self, rainfall_features: dict = None, timestamp: datetime = None) -> pd.DataFrame:
        """
        Calculate MPI for all wards.
        
        MPI Components (0-100 scale):
        1. Model Probability (40%): ML model prediction
        2. Rainfall Severity (20%): Current + forecast rainfall intensity
        3. Historical Risk (15%): Past flood frequency
        4. Infrastructure Stress (15%): Drainage capacity + complaints
        5. Vulnerability (10%): Elevation + low-lying areas
        
        Returns DataFrame with MPI scores and breakdown
        """
        print("\n" + "=" * 70)
        print("CALCULATING WARD-WISE MPI")
        print("=" * 70)
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Get rainfall features
        if rainfall_features is None:
            rainfall_features = self.get_real_time_rainfall()
        
        # Temporal features
        temporal_features = self.feature_engineer.compute_temporal_features(timestamp)
        
        # Calculate MPI for each ward
        results = []
        
        for ward_id in self.ward_static.index:
            # 1. Get model prediction (40%)
            static_feats = self.ward_static.loc[ward_id].to_dict()
            hist_feats = self.ward_historical.loc[ward_id].to_dict()
            
            X = self.feature_engineer.create_feature_vector(
                rainfall_features, static_feats, hist_feats, temporal_features
            )
            
            model_prob = float(self.model.predict_proba(X.reshape(1, -1))[0])
            model_score = model_prob * 40  # 0-40 points
            
            # 2. Rainfall Severity (20%)
            rain_total = rainfall_features['rain_3h'] + rainfall_features['rain_forecast_3h']
            if rain_total < 5:
                rain_score = 0
            elif rain_total < 15:
                rain_score = 5
            elif rain_total < 35:
                rain_score = 10
            elif rain_total < 65:
                rain_score = 15
            else:
                rain_score = 20
            
            # 3. Historical Risk (15%)
            hist_flood_freq = hist_feats.get('hist_flood_freq', 0)
            hist_score = min(15, hist_flood_freq * 2.5)
            
            # 4. Infrastructure Stress (15%)
            drain_density = static_feats.get('drain_density', 0)
            # Poor drainage = higher stress
            drain_stress = max(0, 10 - drain_density) / 10 * 10  # 0-10 points
            
            # Complaints
            if self.civic_complaints is not None and ward_id in self.civic_complaints.index:
                complaints = self.civic_complaints.loc[ward_id, 'drainage_complaints']
                complaint_stress = min(5, complaints / 20 * 5)  # 0-5 points
            else:
                complaint_stress = 0
            
            infra_score = drain_stress + complaint_stress  # 0-15 points
            
            # 5. Vulnerability (10%)
            low_lying_pct = static_feats.get('low_lying_pct', 15)
            mean_elevation = static_feats.get('mean_elevation', 215)
            
            # Lower elevation + higher low-lying % = more vulnerable
            elev_vuln = max(0, (220 - mean_elevation) / 15 * 5)  # 0-5 points
            lowlying_vuln = low_lying_pct / 30 * 5  # 0-5 points
            vuln_score = elev_vuln + lowlying_vuln  # 0-10 points
            
            # Total MPI (0-100)
            mpi = model_score + rain_score + hist_score + infra_score + vuln_score
            
            # Risk level
            if mpi < 30:
                risk_level = "Low"
            elif mpi < 50:
                risk_level = "Moderate"
            elif mpi < 70:
                risk_level = "High"
            else:
                risk_level = "Critical"
            
            results.append({
                'ward_id': ward_id,
                'mpi_score': round(mpi, 1),
                'risk_level': risk_level,
                'model_prob': round(model_prob, 3),
                'model_contribution': round(model_score, 1),
                'rainfall_contribution': round(rain_score, 1),
                'historical_contribution': round(hist_score, 1),
                'infrastructure_contribution': round(infra_score, 1),
                'vulnerability_contribution': round(vuln_score, 1),
                'current_rain_mm': rainfall_features['rain_1h'],
                'forecast_rain_mm': rainfall_features['rain_forecast_3h'],
                'hist_flood_count': hist_flood_freq,
                'drain_density': round(drain_density, 2),
                'elevation_m': round(mean_elevation, 1)
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('mpi_score', ascending=False)
        
        return df
    
    def generate_mpi_report(self, mpi_df: pd.DataFrame, save_path: str = None):
        """Generate comprehensive MPI report."""
        print("\n" + "=" * 70)
        print("MPI REPORT - WARD-WISE FLOOD RISK")
        print("=" * 70)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Summary statistics
        print(f"\nTotal wards analyzed: {len(mpi_df)}")
        print(f"\nRisk Distribution:")
        for level in ['Critical', 'High', 'Moderate', 'Low']:
            count = len(mpi_df[mpi_df['risk_level'] == level])
            pct = count / len(mpi_df) * 100
            print(f"  {level:10s}: {count:3d} wards ({pct:5.1f}%)")
        
        print(f"\nMPI Statistics:")
        print(f"  Mean MPI: {mpi_df['mpi_score'].mean():.1f}")
        print(f"  Max MPI:  {mpi_df['mpi_score'].max():.1f}")
        print(f"  Min MPI:  {mpi_df['mpi_score'].min():.1f}")
        
        # Top 10 high-risk wards
        print(f"\n{'='*70}")
        print("TOP 10 HIGH-RISK WARDS")
        print(f"{'='*70}")
        top10 = mpi_df.head(10)
        
        for idx, row in top10.iterrows():
            print(f"\n{idx+1}. Ward {row['ward_id']} - MPI: {row['mpi_score']:.1f} ({row['risk_level']})")
            print(f"   Model Probability: {row['model_prob']:.1%}")
            print(f"   Contributions: Model={row['model_contribution']:.0f}, Rain={row['rainfall_contribution']:.0f}, History={row['historical_contribution']:.0f}, Infra={row['infrastructure_contribution']:.0f}, Vuln={row['vulnerability_contribution']:.0f}")
            print(f"   Flood History: {row['hist_flood_count']} events")
            print(f"   Drain Density: {row['drain_density']:.2f} points/km²")
        
        # Save to CSV
        if save_path:
            mpi_df.to_csv(save_path, index=False)
            print(f"\n[OK] Full MPI data saved to: {save_path}")
        
        return mpi_df


def main():
    """Main MPI calculation workflow."""
    print("=" * 70)
    print("DELHI FLOOD RISK - WARD-WISE MPI CALCULATOR")
    print("=" * 70)
    
    # Initialize
    calc = MPICalculator()
    
    # Load all required data
    if not calc.load_model():
        print("[ERROR] Cannot proceed without model")
        return
    
    if not calc.load_ward_data():
        print("[ERROR] Cannot proceed without ward data")
    
    calc.load_civic_complaints()
    
    # Calculate MPI with real-time weather
    mpi_df = calc.calculate_mpi()
    
    # Generate report
    output_path = Path("backend/data/processed/mpi_scores_latest.csv")
    calc.generate_mpi_report(mpi_df, save_path=output_path)
    
    # Also save for frontend
    frontend_path = Path("frontend/public/data/ward_risk.json")
    mpi_json = {}
    for _, row in mpi_df.iterrows():
        mpi_json[row['ward_id']] = {
            "risk_score": row['mpi_score'],
            "rain_mm": row['current_rain_mm'] + row['forecast_rain_mm'],
            "status": row['risk_level']
        }
    
    frontend_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(frontend_path, 'w') as f:
        json.dump(mpi_json, f, indent=2)
    
    print(f"\n[OK] Frontend data updated: {frontend_path}")
    
    print("\n" + "=" * 70)
    print("MPI CALCULATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
