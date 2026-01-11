"""
Extract and process INDOFLOODS data for Delhi flood event labels
"""
import json
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import datetime

def extract_indofloods_metadata():
    """Extract file information from INDOFLOODS metadata"""
    print("=" * 70)
    print("EXTRACTING INDOFLOODS DATA")
    print("=" * 70)
    
    json_path = Path('backend/data/raw/14584655.json')
    
    with open(json_path) as f:
        metadata = json.load(f)
    
    print(f"\nDataset: {metadata.get('metadata', {}).get('title')}")
    print(f"Record ID: {metadata.get('id')}")
    
    # Extract file download links
    files_data = metadata.get('files', {})
    
    if isinstance(files_data, dict):
        files = files_data.get('entries', [])
    else:
        files = []
    
    print(f"\nAvailable files ({len(files)}):")
    if files:
        for file_info in files:
            if isinstance(file_info, dict):
                key = file_info.get('key', '')
                size_mb = file_info.get('size', 0) / (1024 * 1024)
                print(f"  - {key} ({size_mb:.1f} MB)")
                
                # Get download link
                links = file_info.get('links', {})
                if 'self' in links:
                    print(f"    URL: {links['self']}")
    else:
        print("  (File structure not in expected format - check Zenodo directly)")
    
    # Check if files are already downloaded
    print("\n" + "=" * 70)
    print("CHECKING FOR LOCAL DATA FILES")
    print("=" * 70)
    
    raw_dir = Path('backend/data/raw')
    
    # Look for CSV/Excel files that might be flood data
    data_files = list(raw_dir.rglob('*.csv')) + list(raw_dir.rglob('*.xlsx'))
    
    print(f"\nFound {len(data_files)} data files in raw directory")
    
    # Try to find INDOFLOODS data
    indofloods_files = [f for f in data_files if 'flood' in f.name.lower() or 'indo' in f.name.lower()]
    
    if indofloods_files:
        print("\nPotential INDOFLOODS files found:")
        for f in indofloods_files:
            print(f"  - {f}")
    else:
        print("\n⚠️  No INDOFLOODS CSV/Excel files found locally")
        print("\nTo download:")
        print("1. Visit: https://zenodo.org/records/14584655")
        print("2. Download the CSV files containing flood events")
        print("3. Place in backend/data/raw/indofloods/")
    
    return files

def create_flood_labels_from_events(flood_events_file: str = None):
    """
    Create failure labels from actual flood event data
    
    This function will:
    1. Load flood events for Delhi
    2. Match events to wards and dates
    3. Create binary failure labels
    4. Save to processed data folder
    """
    print("\n" + "=" * 70)
    print("CREATING LABELS FROM FLOOD EVENTS")
    print("=" * 70)
    
    if flood_events_file is None:
        print("\n⚠️  No flood events file provided")
        print("This function needs the actual INDOFLOODS CSV/Excel file")
        return None
    
    # Load flood events
    try:
        if flood_events_file.endswith('.csv'):
            events = pd.read_csv(flood_events_file)
        else:
            events = pd.read_excel(flood_events_file)
        
        print(f"\nLoaded {len(events)} flood events")
        print(f"Columns: {events.columns.tolist()}")
        
        # Filter for Delhi
        delhi_events = events[
            (events['State'].str.contains('Delhi', case=False, na=False)) |
            (events['District'].str.contains('Delhi', case=False, na=False))
        ]
        
        print(f"Delhi events: {len(delhi_events)}")
        
        if len(delhi_events) > 0:
            print("\nSample events:")
            print(delhi_events[['Date', 'Location', 'Casualties']].head())
            
            # Save Delhi-specific events
            output_path = Path('backend/data/processed/delhi_flood_events.csv')
            delhi_events.to_csv(output_path, index=False)
            print(f"\n✓ Saved Delhi flood events to: {output_path}")
            
            return delhi_events
        else:
            print("⚠️  No Delhi-specific events found")
            return None
            
    except Exception as e:
        print(f"Error loading flood events: {e}")
        return None

def generate_improved_synthetic_labels():
    """
    Generate improved synthetic labels based on known Delhi flood patterns
    
    Uses:
    - 2023 heavy rainfall period (July-August)
    - Ward vulnerability (low elevation + poor drainage)
    - Rainfall intensity thresholds
    """
    print("\n" + "=" * 70)
    print("GENERATING IMPROVED SYNTHETIC LABELS")
    print("=" * 70)
    print("\nBased on:")
    print("  - Delhi 2023 floods (July 8-14) - 150+ mm rainfall")
    print("  - Yamuna flooding (flood plain wards)")
    print("  - Poor drainage areas (high density, low elevation)")
    
    # Load processed rainfall data
    rainfall_path = Path('backend/data/processed/rainfall_ward_daily.csv')
    
    if not rainfall_path.exists():
        print(f"\n⚠️  Rainfall data not found at {rainfall_path}")
        return None
    
    rainfall_df = pd.read_csv(rainfall_path)
    rainfall_df['date'] = pd.to_datetime(rainfall_df['date'])
    
    # Load ward features
    wards_path = Path('backend/data/processed/ward_static_features.csv')
    wards_df = pd.read_csv(wards_path)
    
    print(f"\nRainfall data: {len(rainfall_df)} records")
    print(f"Ward features: {len(wards_df)} wards")
    
    # Create failure labels based on REALISTIC criteria
    labels = []
    
    for _, row in rainfall_df.iterrows():
        ward_id = row['ward_id']
        date = row['date']
        rain = row['rainfall_mm']
        
        # Get ward characteristics
        ward_feat = wards_df[wards_df['ward_id'] == ward_id]
        
        if len(ward_feat) == 0:
            continue
        
        ward_feat = ward_feat.iloc[0]
        
        # Vulnerability factors
        low_elevation = ward_feat['mean_elevation'] < 210  # Below average
        poor_drainage = ward_feat['drain_density'] < 0.5   # Low drain density
        high_low_lying = ward_feat['low_lying_pct'] > 20   # Significant low areas
        
        vulnerability_score = (
            0.3 * low_elevation +
            0.4 * poor_drainage +
            0.3 * high_low_lying
        )
        
        # Failure probability based on rainfall + vulnerability
        # Known Delhi flood patterns:
        # - Heavy rain (>50mm/day) causes flooding in vulnerable areas
        # - Very heavy (>100mm/day) causes widespread flooding
        # - Sustained rain (>30mm for 3+ days) saturates drainage
        
        if rain > 100:  # Very heavy rainfall
            failure_prob = 0.6 + 0.3 * vulnerability_score
        elif rain > 50:  # Heavy rainfall
            failure_prob = 0.3 + 0.4 * vulnerability_score
        elif rain > 30:  # Moderate rainfall
            failure_prob = 0.1 + 0.3 * vulnerability_score
        else:
            failure_prob = 0.02 + 0.05 * vulnerability_score
        
        # July-August 2023 boost (known flood period)
        if pd.to_datetime(date).year == 2023 and pd.to_datetime(date).month in [7, 8]:
            failure_prob *= 1.3
        
        # Generate label with probability
        import numpy as np
        failure = 1 if np.random.random() < failure_prob else 0
        
        labels.append({
            'ward_id': ward_id,
            'date': date,
            'rainfall_mm': rain,
            'failure': failure,
            'failure_prob': failure_prob,
            'vulnerability_score': vulnerability_score
        })
    
    labels_df = pd.DataFrame(labels)
    
    print(f"\nGenerated {len(labels_df)} labeled samples")
    print(f"Failure rate: {labels_df['failure'].mean()*100:.1f}%")
    print(f"High-risk samples (prob>0.5): {(labels_df['failure_prob']>0.5).sum()}")
    
    # Save
    output_path = Path('backend/data/processed/flood_labels_improved.csv')
    labels_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved improved labels to: {output_path}")
    
    return labels_df

if __name__ == "__main__":
    # Step 1: Extract metadata
    files = extract_indofloods_metadata()
    
    # Step 2: Try to create labels from actual events (if available)
    # Check if INDOFLOODS data files exist
    indofloods_dir = Path('backend/data/raw/indofloods')
    
    if indofloods_dir.exists():
        csv_files = list(indofloods_dir.glob('*.csv'))
        if csv_files:
            print(f"\nFound INDOFLOODS CSV: {csv_files[0]}")
            flood_events = create_flood_labels_from_events(str(csv_files[0]))
    else:
        print("\n⚠️  INDOFLOODS data not downloaded yet")
    
    # Step 3: Generate improved synthetic labels (always do this as fallback)
    improved_labels = generate_improved_synthetic_labels()
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("\n1. If you want REAL flood data:")
    print("   - Download INDOFLOODS CSV from https://zenodo.org/records/14584655")
    print("   - Place in backend/data/raw/indofloods/")
    print("   - Run this script again")
    print("\n2. To use improved synthetic labels:")
    print("   - Labels saved to backend/data/processed/flood_labels_improved.csv")
    print("   - Retrain model with: python backend/model/train_model_1.py")
    print("\n3. Improved labels now include:")
    print("   - ✓ Ward vulnerability (elevation + drainage)")
    print("   - ✓ Realistic rainfall thresholds (based on 2023 floods)")
    print("   - ✓ Higher failure rate in vulnerable wards")
