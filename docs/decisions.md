# Project Decisions Log

All architectural decisions, defaults selected, and deviations are logged here with dates.

| Date | Topic | Decision | Rationale |
|------|-------|----------|-----------|
| 2026-11-23 | Environment & Platform | Python 3.11+ / Windows sandbox setup with ruff, mypy, pytest | Standard reproducible dev environment |
| 2026-11-23 | Code Style & Ruff Config | Ignored N806 for feature matrix uppercase `X` | Standard mathematical / ML notation convention |
| 2026-11-23 | Seed Management | Centralized seeds in `src/aqi/seeds.py` (numpy, random, torch) | Ground Rule 6 compliance |
| 2026-11-23 | Qualification Rule | >= 3 years, < 15% missingness, pollutants required for A/B, C can run with AQI alone | Phase 0 Step 5 requirement |
| 2026-11-23 | Target Ground Truth Source | Use CPCB Bulletin `Index Value` (Zenodo 11146316) as `aqi_official` | Kaggle `city_day.csv` AQI column has unclipped averages exceeding 500 (up to 2049 for Ahmedabad); official CPCB scale is capped at 500 |
| 2026-11-23 | Pollutant Feature Merging | Merge Kaggle `city_day.csv` pollutants with CPCB bulletin `aqi_official` for Ahmedabad and Chennai | Allows evaluation of same-day pollutant leakage (Setups A and B) against official bulletin targets |
| 2026-11-23 | Bulletin-Only Cities for Setup C | Include Gandhinagar, Ankleshwar, Vapi, Surat for Setup C | Enables spatial generalization across Gujarat industrial/urban centers using autoregressive lags and calendar features |
| 2026-11-23 | Physical Range Filtering | Dropped 10 impossible CO values (>100 mg/m³) for Ahmedabad | Compliance with CPCB instrument operational envelopes |
| 2026-11-23 | City Name Harmonization | Harmonized `'GandhiNagar'` in bulletin to canonical lowercase key `'gandhinagar'` | Avoid zero-match queries due to camel-casing in raw bulletin CSV |
| 2026-11-23 | Explicit Calendar Reindexing | Reindex all series to a continuous daily DatetimeIndex | Explicitly exposes missing days as NaN for temporal leak-free windowing and imputation |
