# Experimental Protocol (v1 Frozen)

## 1. Study Setups

| Setup | Split | Inputs | Target | Purpose |
|-------|-------|--------|--------|---------|
| **A** | Random 80/20 (10 seeds) | Same-day pollutants (day $t$) | That day's AQI ($t$) | Test extent of random split + concurrent input inflation |
| **B** | Chronological 80/20 | Same-day pollutants (day $t$) | That day's AQI ($t$) | Test effect of time-ordered split with concurrent inputs |
| **C** | Rolling-origin, expanding window, monthly refit (last 20%) | Data up to day $t$ only | Next-day AQI ($t+1$, also $t+3$, $t+7$) | Leak-free realistic operational forecasting |

## 2. Leakage Guard Rules

1. **Shift First, Split After**: Any feature derived from the target or pollutant history must be shifted by at least 1 day before any split or model fitting.
2. **Strict Temporal Separation**: In setups B and C, $\max(\text{train\_dates}) < \min(\text{test\_dates})$.
3. **No Target Contamination**: Preprocessing, imputation, and feature scalers must only be fit on training windows.
