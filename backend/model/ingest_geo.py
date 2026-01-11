import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask
import numpy as np
import os
from shapely.geometry import mapping

def process_geo_features(
    wards_path: str,
    dem_path: str,
    drains_path: str,
    output_path='../data/processed/ward_static_features.csv'
):
    """
    Calculates static ward features from Shapefiles and DEM.
    
    Features:
    - Mean Elevation
    - Low Lying Percentage (Area < threshold)
    - Drain Density (Length of drains / Ward Area)
    """
    print("Loading Geospatial Data...")
    try:
        wards = gpd.read_file(wards_path)
        # Check CRS
        if wards.crs is None:
            print("Wards SHP has no CRS. Assuming EPSG:4326 (Lat/Lon).")
            wards = wards.set_crs("EPSG:4326")
            
        drains = gpd.read_file(drains_path) # Can be KML or SHP
        if drains.crs is None:
             print("Drains map has no CRS. Assuming EPSG:4326.")
             drains = drains.set_crs("EPSG:4326")
             
    except Exception as e:
        print(f"File load error: {e}")
        return

    # Ensure CRS is projected for accurate area/length (e.g., UTM 43N for Delhi)
    # EPSG:32643 is UTM Zone 43N
    target_crs = "EPSG:32643"
    try:
        wards = wards.to_crs(target_crs)
        drains = drains.to_crs(target_crs)
    except Exception as e:
        print(f"Reprojection error: {e}")
        return
    
    print("Calculating Drain Density...")
    ward_stats = []
    
    # 1. Drain Density
    for idx, ward in wards.iterrows():
        # Clip drains to this ward
        drains_clipped = gpd.clip(drains, ward.geometry)
        total_drain_length_km = drains_clipped.length.sum() / 1000 # meters to km
        ward_area_sqkm = ward.geometry.area / 10**6 # sq meters to sq km
        
        density = 0
        if ward_area_sqkm > 0:
            density = total_drain_length_km / ward_area_sqkm
            
        ward_stats.append({
            'ward_no': ward.get('ward_no', idx), # Fallback if no ID
            'ward_name': ward.get('ward_name', f"Ward_{idx}"),
            'area_sqkm': ward_area_sqkm,
            'drain_density': density,
            'geometry': ward.geometry
        })
    
    stats_df = pd.DataFrame(ward_stats)
    
    # 2. Elevation Statistics (Zonal Stats)
    print("Calculating Elevation Statistics (this may take time)...")
    try:
        with rasterio.open(dem_path) as src:
            elevations = []
            low_lying_pcts = []
            
            for _, row in stats_df.iterrows():
                geom = [mapping(row['geometry'])]
                try:
                    out_image, out_transform = mask(src, geom, crop=True)
                    out_data = out_image[0] # Band 1
                    
                    # Remove nodata values (usually -9999 or similar, assuming < -100 here as safe bet)
                    valid_data = out_data[out_data > -100] 
                    
                    if len(valid_data) > 0:
                        mean_elev = np.mean(valid_data)
                        # Define low lying as < Mean - 1 StdDev (Relative depression)
                        # Or absolute: e.g. < 205m
                        threshold = np.mean(valid_data) - np.std(valid_data)
                        low_lying_count = np.sum(valid_data < threshold)
                        low_lying_pct = (low_lying_count / len(valid_data)) * 100
                    else:
                        mean_elev = np.nan
                        low_lying_pct = np.nan
                        
                    elevations.append(mean_elev)
                    low_lying_pcts.append(low_lying_pct)
                    
                except ValueError: 
                    # Geometry might not overlap with DEM
                    elevations.append(np.nan)
                    low_lying_pcts.append(np.nan)
            
            stats_df['mean_elevation'] = elevations
            stats_df['low_lying_pct'] = low_lying_pcts
            
    except Exception as e:
        print(f"DEM processing failed (skipping): {e}")
        stats_df['mean_elevation'] = 0
        stats_df['low_lying_pct'] = 0

    # Clean up geometry before saving to CSV
    final_df = stats_df.drop(columns=['geometry'])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"Saved ward features to {output_path}")

if __name__ == "__main__":
    process_geo_features(
        wards_path=r"backend\data\raw\wards\delhi_ward.shp",
        # Use the specific tile we found.
        dem_path=r"backend\data\raw\dem\C1_DEM_16B_2005-2014_v3_R-1_76E29N_h43q\cdnh43q_v3r1\cdnh43q.tif",
        drains_path=r"backend\data\raw\drains\Delhi_Drains.kml"
    )
