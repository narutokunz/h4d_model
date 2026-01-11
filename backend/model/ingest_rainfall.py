import pandas as pd
import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import os

def process_imd_rainfall(
    nc_file_path: str,
    wards_shapefile_path: str,
    output_path='../data/processed/rainfall_ward_daily.csv'
):
    """
    Reads IMD Gridded Rainfall (NetCDF) and maps it to Delhi Wards.
    
    Args:
        nc_file_path: Path to IMD .nc file (e.g. imd_rain_2022.nc)
        wards_shapefile_path: Path to Delhi Wards .shp
    """
    print(f"Loading Rainfall Data from {nc_file_path}...")
    try:
        # Open NetCDF (Standard IMD structure: lat, lon, time, rain)
        ds = xr.open_dataset(nc_file_path)
    except FileNotFoundError:
        print(f"Error: File {nc_file_path} not found. Please download IMD data.")
        return

    print(f"Loading Wards from {wards_shapefile_path}...")
    try:
        wards = gpd.read_file(wards_shapefile_path)
        if wards.crs is None:
             print("Wards SHP has no CRS. Assuming EPSG:4326.")
             wards = wards.set_crs("EPSG:4326")
    except FileNotFoundError:
        print(f"Error: File {wards_shapefile_path} not found.")
        return

    # Filter rainfall grid to Delhi Bounding Box (approx 28.4 to 28.9 N, 76.8 to 77.3 E)
    # IMD files use uppercase LATITUDE/LONGITUDE
    try:
        delhi_bbox = {'LATITUDE': slice(28.3, 29.0), 'LONGITUDE': slice(76.8, 77.4)}
        ds_delhi = ds.sel(**delhi_bbox)
        lat_col = 'LATITUDE'
        lon_col = 'LONGITUDE'
    except KeyError:
        # Fallback to lowercase if different version
        delhi_bbox = {'lat': slice(28.3, 29.0), 'lon': slice(76.8, 77.4)}
        ds_delhi = ds.sel(**delhi_bbox)
        lat_col = 'lat'
        lon_col = 'lon'
    
    print("Mapping grid points to wards...")
    
    # Create a DataFrame of Grid Points
    # Create meshgrid of lat/lons
    lons, lats = np.meshgrid(ds_delhi[lon_col], ds_delhi[lat_col])
    grid_points = pd.DataFrame({
        'lat': lats.flatten(),
        'lon': lons.flatten()
    })
    
    # Convert to GeoDataFrame
    geometry = [Point(xy) for xy in zip(grid_points.lon, grid_points.lat)]
    gdf_grid = gpd.GeoDataFrame(grid_points, geometry=geometry, crs="EPSG:4326")
    
    # Spatial Join: Match each grid point to a Ward
    # Uses the ward shapefile CRS
    if wards.crs != "EPSG:4326":
        wards = wards.to_crs("EPSG:4326")
        
    joined = gpd.sjoin(gdf_grid, wards, how="left", predicate="within")
    
    # Now we have a map of (Lat, Lon) -> Ward_ID
    # We can aggregate rainfall for each ward
    
    # Use actual column names from shapefile: 'unique' for ward ID
    mapping = joined.dropna(subset=['index_right'])[['lat', 'lon', 'unique']]
    mapping = mapping.rename(columns={'unique': 'ward_id'})
    
    print(f"Found {len(mapping)} grid points covering Delhi wards.")
    
    results = []
    
    # Iterate through time steps in the NetCDF
    # This can be optimized, but loop is clearer for logic
    times = ds_delhi.TIME.values
    
    for t in times:
        data_slice = ds_delhi.sel(TIME=t).to_dataframe().reset_index()
        
        # Rename columns to match expected format
        data_slice = data_slice.rename(columns={'LATITUDE': 'lat', 'LONGITUDE': 'lon', 'RAINFALL': 'rainfall'})
        
        # Merge with ward mapping
        merged = pd.merge(data_slice, mapping, on=['lat', 'lon'], how='inner')
        
        # Group by Ward and take Mean rainfall
        ward_rain = merged.groupby('ward_id')['rainfall'].mean().reset_index()
        ward_rain['date'] = t
        results.append(ward_rain)
        
    final_df = pd.concat(results)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"Saved processed rainfall data to {output_path}")

if __name__ == "__main__":
    # Process all available years of rainfall data
    # Paths relative to project root (where the script is executed)
    for year in [2022, 2023, 2024, 2025]:
        nc_file = f"backend/data/raw/rainfall/RF25_ind{year}_rfp25.nc"
        output_file = f"backend/data/processed/rainfall_ward_daily_{year}.csv"
        
        try:
            print(f"\n{'='*60}")
            print(f"Processing {year} Rainfall Data")
            print(f"{'='*60}")
            process_imd_rainfall(
                nc_file_path=nc_file, 
                wards_shapefile_path=r"backend\data\raw\wards\delhi_ward.shp",
                output_path=output_file
            )
        except FileNotFoundError as e:
            print(f"Skipping {year}: File not found")
        except Exception as e:
            print(f"Error processing {year}: {e}")
