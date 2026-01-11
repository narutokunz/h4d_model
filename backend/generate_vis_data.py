import geopandas as gpd
import pandas as pd
import numpy as np
import json

def generate_ward_risk_mock():
    # Load Wards to get IDs
    wards_path = r"backend\data\raw\wards\delhi_ward.shp"
    gdf = gpd.read_file(wards_path)
    
    # Create mock data
    # We will generate a 'Risk Score' (0-100) and 'Rainfall' (mm)
    
    data = {}
    
    for _, row in gdf.iterrows():
        # Use 'unique' as the key if available, else 'id'
        ward_id = str(row.get('unique', row.get('id', 'unknown')))
        
        # Simulate localized clusters of high risk (e.g. near Yamuna or low lying)
        # Random logic: IDs starting with '0' might be central/old delhi? check later.
        
        base_risk = np.random.randint(10, 60)
        
        # Add some hotspots
        if np.random.rand() < 0.15: # 15% high risk
            base_risk += 40
            
        data[ward_id] = {
            "risk_score": min(100, base_risk),
            "rain_mm": round(np.random.uniform(0, 50), 1),
            "status": "Critical" if base_risk > 80 else ("Watch" if base_risk > 50 else "Safe")
        }
        
    # Save to Frontend Public folder
    output_path = r"frontend\public\data\ward_risk.json"
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"Generated mock risk data for {len(data)} wards at {output_path}")

if __name__ == "__main__":
    generate_ward_risk_mock()
