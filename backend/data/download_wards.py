import os
import requests

def download_file(url, save_path):
    print(f"Downloading {url}...")
    try:
        r = requests.get(url)
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(r.content)
        print(f"Saved to {save_path}")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def main():
    base_url = "https://raw.githubusercontent.com/HindustanTimesLabs/shapefiles/master/city/delhi/ward/"
    save_dir = r"backend\data\raw\wards"
    os.makedirs(save_dir, exist_ok=True)
    
    # Common shapefile extensions
    extensions = ['shp', 'shx', 'dbf', 'prj']
    
    # Assuming the basename is 'ward' based on the directory name
    # If this fails, we might need to check 'Delhi_Wards' or similar
    basename = "ward" 
    
    success_count = 0
    for ext in extensions:
        filename = f"{basename}.{ext}"
        url = f"{base_url}{filename}"
        save_path = os.path.join(save_dir, f"Delhi_Wards.{ext}") # Renaming for clarity as per previous steps
        
        if download_file(url, save_path):
            success_count += 1
            
    if success_count == 0:
        print("No files downloaded. The basename might not be 'ward'.")
    else:
        print(f"Successfully downloaded {success_count} files.")

if __name__ == "__main__":
    main()
