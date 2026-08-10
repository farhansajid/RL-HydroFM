# Datasets & Benchmarks for Water Resources & Flood Monitoring

This repository provides standardized few-shot benchmarks for Earth observation and water resources monitoring, specifically tailored for evaluating **multi-source geospatial foundation models**.

---

## 1. Supported Benchmarks

### A. EuroSAT-Water (Sentinel-2 Multi-Spectral)
* **Description:** Extracted from the EuroSAT Sentinel-2 Earth observation dataset, focusing on freshwater, wetland, and surrounding ecological land-cover classes.
* **Classes (5):**
  1. `River` (Linear river channels, tributaries, estuaries)
  2. `SeaLake` (Open surface water, natural lakes, reservoirs)
  3. `PermanentCrop` (Irrigated agricultural land, orchards)
  4. `Pasture` (Grassland & meadow hydrology)
  5. `HerbaceousVegetation` (Riparian zones, wetlands)
* **Image Dimensions:** 64×64 pixels across 13 spectral bands (RGB & NIR/SWIR).
* **Splits:** Standardized 50% Train, 25% Validation, 25% Test.

### B. Kaggle Sentinel-2 Water Bodies Dataset
* **Description:** Open-access high-resolution Sentinel-2 satellite imagery curated for surface water identification, reservoir storage monitoring, and drought assessment.
* **Classes (4):** `Open Water`, `Turbid Water / Sediment`, `Wetland / Mangrove`, `Dry Land / Urban`.
* **Resolution:** 10m Ground Sample Distance (GSD).

### C. Sen12-Flood (Paired Sentinel-1 SAR + Sentinel-2 Optical)
* **Description:** Multi-modal Earth observation benchmark featuring paired Sentinel-1 SAR (VV/VH polarization backscatter) and Sentinel-2 optical imagery acquired during real flood events across Africa, Asia, and North America.
* **Modality Advantage:** SAR penetrates dense cloud cover during heavy precipitation events, while optical imagery provides fine-grained spectral discrimination under clear skies.
* **Classes (3):** `Flooded Inundation`, `Permanent Water Body`, `Non-Flooded Terrain`.

### D. RESISC45-Water (High-Resolution Aerial/Satellite)
* **Description:** High-resolution optical dataset (NWPU-RESISC45) subset featuring complex hydrological and coastal categories.
* **Classes (7):** `lake`, `river`, `wetland`, `sea_ice`, `harbor`, `beach`, `island`.

---

## 2. Directory Layout

The datasets are structured into standardized class-named subdirectories under the `./DATA` root:

```
DATA/
├── eurosat_water/
│   ├── 2750/
│   │   ├── HerbaceousVegetation/
│   │   ├── Pasture/
│   │   ├── PermanentCrop/
│   │   ├── River/
│   │   └── SeaLake/
│   └── split.json
├── sentinel2_water/
│   ├── 2750/
│   │   ├── Dry_Land/
│   │   ├── Open_Water/
│   │   ├── Turbid_Water/
│   │   └── Wetland/
│   └── split.json
└── sen12_flood/
    ├── optical/
    ├── sar/
    └── split.json
```

---

## 3. Automatic Dataset Preparation

Run the automated data preparation script:

```bash
python datasets_loader/water_benchmarks.py --generate_synthetic --target_root ./DATA
```
