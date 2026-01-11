import geopandas as gpd
import os

def analyze_shapefile(path, name):
    print(f"--- Analyzing {name} ---")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    try:
        gdf = gpd.read_file(path)
        print(f"Shape: {gdf.shape} (Rows, Columns)")
        print(f"CRS: {gdf.crs}")
        print("Columns:", gdf.columns.tolist())
        print("First 2 rows:")
        print(gdf.head(2).drop(columns='geometry')) # Drop geometry to keep output clean
        print("\n")
    except Exception as e:
        print(f"Error reading {name}: {e}\n")

if __name__ == "__main__":
    # Analyze Wards
    analyze_shapefile(r"backend\data\raw\wards\delhi_ward.shp", "Delhi Wards")
    
    # Analyze Districts
    analyze_shapefile(r"backend\data\raw\districts\Districts.shp", "Delhi Districts")
