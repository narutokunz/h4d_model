# Data Setup Instructions

To run the model with real data, you must download the following files and place them in the correct directories.

## 1. Rainfall Data (IMD)
*   **Source**: [IMD Pune Gridded Data](https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html)
*   **File**: Download the NetCDF file for recent years (e.g., `Rainfall_ind_2022_rfp25.nc`).
*   **Action**: Rename/Move to:
    `backend/data/raw/rainfall/Rainfall_ind_2022_rfp25.nc`

## 2. Ward Boundaries
*   **Source**: [Hindustan Times Labs / GitHub](https://github.com/HindustanTimesLabs/shapefiles/tree/master/city/delhi/ward)
*   **Files**: Download all shapefile components (`.shp`, `.shx`, `.dbf`, `.prj`).
*   **Action**: Move to:
    `backend/data/raw/wards/Delhi_Wards.shp` (and related files)

## 3. Elevation (DEM)
*   **Source**: [Bhuvan NRSC](https://bhuvan-app3.nrsc.gov.in/data/download/index.php) (CartoDEM)
*   **File**: Download the GeoTIFF tile covering Delhi.
*   **Action**: Move to:
    `backend/data/raw/elevation/Delhi_DEM.tif`

## 4. Drainage Network
*   **Source**: [OpenCity Delhi Drains](https://staging.opencity.in/dataset/delhi-drains-maps)
*   **Files**: Download the KML or Shapefile.
*   **Action**: Move to:
    `backend/data/raw/drains/Delhi_Drains.shp` (If KML, convert or update script path).

---

## How to Run Processing
Once files are in place, run:

```bash
# 1. Process Rainfall
python backend/model/ingest_rainfall.py

# 2. Process Ward Features (elevation, drains)
python backend/model/ingest_geo.py
```
