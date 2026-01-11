import os
import requests
import zipfile
import io

def download_and_extract_districts():
    base_url = "https://raw.githubusercontent.com/HindustanTimesLabs/shapefiles/master/city/delhi/district/"
    save_dir = r"backend\data\raw\districts"
    os.makedirs(save_dir, exist_ok=True)
    
    zip_filename = "delhi_1997-2012_district.zip"
    url = f"{base_url}{zip_filename}"
    save_path = os.path.join(save_dir, zip_filename)
    
    print(f"Downloading {url}...")
    try:
        r = requests.get(url)
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(r.content)
        print(f"Saved to {save_path}")
        
        print("Extracting...")
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall(save_dir)
        print("Extraction complete.")
        
    except Exception as e:
        print(f"Failed to download or extract {url}: {e}")

if __name__ == "__main__":
    download_and_extract_districts()
