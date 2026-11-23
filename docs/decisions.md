# Project Decisions Log

All architectural decisions, defaults selected, and deviations are logged here with dates.

| Date | Topic | Decision | Rationale |
|------|-------|----------|-----------|
| 2026-11-23 | Environment & Platform | Python 3.11+ / Windows sandbox setup with ruff, mypy, pytest | Standard reproducible dev environment |
| 2026-11-23 | Code Style & Ruff Config | Ignored N806 for feature matrix uppercase `X` | Standard mathematical / ML notation convention |
| 2026-11-23 | Seed Management | Centralized seeds in `src/aqi/seeds.py` (numpy, random, torch) | Ground Rule 6 compliance |
| 2026-11-23 | Qualification Rule | >= 3 years, < 15% missingness, pollutants required for A/B, C can run with AQI alone | Phase 0 Step 5 requirement |
