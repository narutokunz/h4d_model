# Project Data Sources

This document tracks the data sources identified for the Delhi Flood and Drainage Monitoring project.

## 1. Rainfall data (India / Delhi)

### A. IMD gridded rainfall (historical)
*   **What**: Daily rainfall at 0.25° x 0.25° resolution across India (1901–present).
*   **Where**: IMD Pune gridded data portal.
*   **Link**: [IMD Pune](https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html)
*   **Use**: Historical rainfall for threshold learning.

### B. IMD real-time and district-level rainfall via API
*   **What**: Current and forecast rainfall, district-wise.
*   **Where**: IMD API documentation.
*   **Links**:
    *   [District rainfall](https://mausam.imd.gov.in/api/districtwise_rainfall_api.php)
    *   [Nowcast](https://mausam.imd.gov.in/api/nowcast_district_api.php)
*   **Note**: Requires IP whitelisting.

### C. Delhi rainfall CSV dataset
*   **What**: Historical rainfall data for Delhi.
*   **Where**: OpenCity India.
*   **Link**: [OpenCity](https://data.opencity.in/dataset/?organization=india-meteorological-department&groups=delhi&res_format=CSV)

### D. IMD daily district rainfall
*   **What**: Monthly files with daily rainfall measurements per district.
*   **Where**: OpenDataBay / data.gov.in.

## 2. DEM / Elevation data

### A. CartoDEM 30m (Cartosat-1)
*   **What**: 30m resolution DEM for India.
*   **Where**: Bhuvan NRSC Open Data Archive.
*   **Link**: [Bhuvan](https://bhuvan-app3.nrsc.gov.in/data/download/index.php)
*   **Use**: Low-lying area detection, slope calculation.

## 3. Drainage network and infrastructure

### A. Delhi drains KML maps
*   **What**: Stormwater drain maps (IFC, PWD, Najafgarh, Barapulla).
*   **Where**: OpenCity India.
*   **Link**: [OpenCity Drains](https://staging.opencity.in/dataset/delhi-drains-maps)
*   **Use**: Drainage stress layer.

### B. General drain network (OSM)
*   **What**: Drain lines extracted from OpenStreetMap.
*   **Links**:
    *   [MAPOG](https://www.igismap.com/download-drains-data-in-shapefile-kml-mid-15-gis-formats-using-gis-data-by-mapog/)
    *   [OpenStreetMap (MapTiler)](https://www.maptiler.com/on-prem-datasets/dataset/osm/asia/india/new-delhi/)

## 4. Administrative boundaries

### A. Delhi ward boundaries (MCD)
*   **What**: Shapefile for 250 MCD wards (2017).
*   **Where**: GitHub (Hindustan Times).
*   **Link**: [GitHub](https://github.com/HindustanTimesLabs/shapefiles/tree/master/city/delhi/ward)
*   **Use**: Ward-level aggregation.

### B. Official ward map
*   **Link**: [MCD Portal](https://mcdonline.nic.in/portal/downloadFile/mcd_map_full_zone_image_cd_23030712390030_230322122342342.pdf)

### C. GSDL Delhi (Geospatial Delhi Limited)
*   **Link**: [GSDL](https://gsdl.org.in)

## 5. Historical waterlogging / flood events

### A. INDOFLOODS database
*   **What**: Flood event database with catchment attributes.
*   **Where**: Zenodo.
*   **Link**: [Zenodo](https://zenodo.org/records/14584655)
*   **Use**: Validation.

### B. India Flood Atlas
*   **What**: Gridded simulations (1901–2020).
*   **Link**: [India Flood Atlas](https://indiafloodatlas.in)

### C. Delhi-specific reports
*   **NIDM 2023**: [PDF](https://nidm.gov.in/PDF/pubs/Proc_YUFloodsNIDM_24.pdf)
*   **CWC 2023 Case Study**: [PDF](https://cwc.gov.in/sites/default/files/delhi-floods-2023-case-study.pdf)
*   **NITI Aayog**: [PDF](https://www.niti.gov.in/sites/default/files/2021-03/Flood-Report.pdf)

## 6. Complaints and civic issue data

### A. Delhi civic complaints (MCD + DJB)
*   **What**: Report with zone-wise complaint data (2016–2022).
*   **Link**: [Praja Report](https://www.praja.org/praja_docs/praja_downloads/Report%20on%20The%20Status%20of%20Civic%20Issues%20in%20Delhi%202023.pdf)
*   **Use**: Historical complaint-based labels (proxy for waterlogging).

### B. National complaint portal
*   **Link**: [India.gov.in](https://services.india.gov.in/service/detail/sewerage-blocked--over-flow-of-manholes)

## 7. Open data portals
*   **data.gov.in**: [Link](https://data.gov.in)
*   **India-WRIS**: [Link](https://nwic.gov.in/data)
*   **Bhuvan**: [Link](https://bhuvan.nrsc.gov.in)
