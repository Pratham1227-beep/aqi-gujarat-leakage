# Data Availability Status

Summary of city datasets, sources, time ranges, and availability status.

| City | Source | Years | Frequency | Pollutants | Status | Notes |
|------|--------|-------|-----------|------------|--------|-------|
| Ahmedabad | CPCB Bulletin / CCR | 2015-2023 | Daily / Hourly | PM2.5, PM10, NO2, SO2, CO, O3, NH3 | Unchecked | Primary study city for Setups A, B, and C |
| Surat | CPCB Bulletin / CCR | 2015-2023 | Daily / Hourly | PM2.5, PM10, NO2, SO2, CO, O3, NH3 | Unchecked | Primary study city; fallback: Setup C only on bulletin AQI |
| Chennai | CPCB Bulletin / CCR | 2015-2023 | Daily / Hourly | PM2.5, PM10, NO2, SO2, CO, O3, NH3 | Unchecked | Replication anchor (Setup C persistence R2 ~0.475, MAE ~12.4) |
| Vadodara | CPCB Bulletin | 2018-2023 | Daily | AQI / partial pollutants | Unchecked | Secondary Gujarat city (subject to qualification) |
| Rajkot | CPCB Bulletin | 2018-2023 | Daily | AQI / partial pollutants | Unchecked | Secondary Gujarat city (subject to qualification) |

## Qualification Criteria (Phase 0, Step 5)

1. At least 3 years of daily AQI data.
2. Less than 15% missing observations over the qualification span.
3. Criteria pollutants (PM2.5/PM10, NO2, SO2, CO, O3, NH3) present for Setups A and B.
4. Cities with AQI only qualify for Setup C.
