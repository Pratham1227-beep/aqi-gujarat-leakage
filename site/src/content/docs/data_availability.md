---
title: "Data Availability Audit"
order: 2
---

# Data Availability & Qualification Report

Real empirical findings evaluated from `data/raw/` downloads (`India_AQI_Bulletins_Master.csv` and Kaggle air quality archive).

## 1. City Availability Matrix

| City | Source | Raw File | Years / Date Range | Records | Pollutants Present | Missingness | Qualification Status | Notes |
|------|--------|----------|-------------------|---------|-------------------|-------------|----------------------|-------|
| **Ahmedabad** | Kaggle + CPCB Bulletin | `city_day.csv`, `India_AQI_Bulletins_Master.csv` | 2015-01-01 to 2020-07-01 (Kaggle), 2015-05-01 to 2023-12-31 (Bulletin) | 2,009 (Kaggle), 2,538 (Bulletin) | PM2.5, PM10, NO2, SO2, CO, O3 | < 15% missing in core 3-year window; 100% valid in Bulletin | **QUALIFIED: Setups A, B, C** | Primary Gujarat study city. Station `GJ001` (Maninagar). |
| **Chennai** | Kaggle + CPCB Bulletin | `city_day.csv`, `India_AQI_Bulletins_Master.csv` | 2015-01-01 to 2020-07-01 (Kaggle), 2015-05-01 to 2023-12-31 (Bulletin) | 2,009 (Kaggle), 3,075 (Bulletin) | PM2.5, PM10, NO2, SO2, CO, O3, NH3 | 6.2% missing in Kaggle; 100% valid in Bulletin | **QUALIFIED: Setups A, B, C** | Replication anchor. Target persistence $R^2 \approx 0.475$, MAE $\approx 12.4$. |
| **Surat** | CPCB Bulletin | `India_AQI_Bulletins_Master.csv` | 2022-11-04 to 2023-12-31 | 327 | None in raw downloads (bulletin AQI only) | 0% missing over active span | **PARTIAL: Setup C (Fallback)** | Lacks station data in Kaggle. Bulletin span is ~1.1 years (under 3-year threshold); runs Setup C under Phase 1 fallback. |
| **Gandhinagar** | CPCB Bulletin | `India_AQI_Bulletins_Master.csv` | 2019-02-05 to 2023-12-31 | 1,704 | Bulletin AQI only (station `GJ003` has no Kaggle rows) | 0% missing in Bulletin | **QUALIFIED: Setup C only** | Secondary Gujarat city. 4.9 years of daily AQI. |
| **Ankleshwar** | CPCB Bulletin | `India_AQI_Bulletins_Master.csv` | 2019-02-05 to 2023-12-31 | 1,502 | Bulletin AQI only (station `GJ002` has no Kaggle rows) | 0% missing in Bulletin | **QUALIFIED: Setup C only** | Secondary Gujarat industrial city. 4.9 years of daily AQI. |
| **Vapi** | CPCB Bulletin | `India_AQI_Bulletins_Master.csv` | 2019-02-14 to 2023-12-31 | 1,349 | Bulletin AQI only (station `GJ005` has no Kaggle rows) | 0% missing in Bulletin | **QUALIFIED: Setup C only** | Secondary Gujarat industrial city. 4.9 years of daily AQI. |
| **Vadodara** | CPCB Bulletin | N/A | None | 0 | None | N/A | **DISQUALIFIED** | Not present in CPCB bulletins or Kaggle archive. |
| **Rajkot** | CPCB Bulletin | N/A | None | 0 | None | N/A | **DISQUALIFIED** | Not present in CPCB bulletins or Kaggle archive. |

## 2. Ingestion Manifest Summary (`data/raw/MANIFEST.csv`)

- **`city_day.csv`**: 29,531 rows, CC0, SHA256: `0d84b21c...`
- **`city_hour.csv`**: 707,875 rows, CC0, SHA256: `fae47406...`
- **`station_day.csv`**: 108,035 rows, CC0, SHA256: `f07787eb...`
- **`station_hour.csv`**: 2,589,083 rows, CC0, SHA256: `26a423c6...`
- **`stations.csv`**: 230 rows, CC0, SHA256: `ce2b7081...`
- **`India_AQI_Bulletins_Master.csv`**: 299,974 rows, CC-BY, SHA256: `c2590a04...`

## 3. CCR Station Data Note (Ground Rule 2)

Automated scraping of the live CPCB CCR portal is restricted by captchas and session tokens. All available station data for Gujarat and Chennai has been collected via the CPCB archive release. Any additional hourly CCR station files provided manually by the user will be placed under `data/raw/ccr/`.

## 4. Processed Daily Datasets (`data/processed/`)

Generated via `build_interim.py` and `build_processed.py`:

| City | Date Range | Total Calendar Days | Valid AQI Days | Missing AQI (%) | Valid PM2.5 Days | Invalid CO Removed | Target Setups |
|------|------------|---------------------|----------------|-----------------|------------------|--------------------|---------------|
| **Ahmedabad** | 2015-05-01 to 2023-12-31 | 3,167 | 2,538 | 19.9% | 1,149 (36.3%) | 10 (>100 mg/m³) | Setups A, B, C |
| **Chennai** | 2015-05-01 to 2023-12-31 | 3,167 | 3,075 | 2.9% | 1,792 (56.6%) | 0 | Setups A, B, C (Anchor) |
| **Gandhinagar** | 2019-02-05 to 2023-12-31 | 1,791 | 1,704 | 4.9% | N/A (Bulletin only) | 0 | Setup C |
| **Ankleshwar** | 2019-02-05 to 2023-12-31 | 1,791 | 1,502 | 16.1% | N/A (Bulletin only) | 0 | Setup C |
| **Vapi** | 2019-02-14 to 2023-12-31 | 1,782 | 1,349 | 24.3% | N/A (Bulletin only) | 0 | Setup C |
| **Surat** | 2022-11-04 to 2023-12-31 | 423 | 327 | 22.7% | N/A (Bulletin only) | 0 | Setup C (Fallback) |
