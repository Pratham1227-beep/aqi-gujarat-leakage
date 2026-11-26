---
title: "Experimental Protocol"
order: 1
---

# Experimental Protocol (v1 Frozen)

## 1. Study Setups

| Setup | Split Scheme | Input Features ($X$) | Target ($y$) | Evaluation Purpose |
|:---|:---|:---|:---|:---|
| **A** | Random 80/20 train/test (10 seeds) | Same-day pollutant concentrations ($t$) + calendar ($t$) | That day's AQI ($t$) | Quantifies peak inflation from concurrent input leakage + random split autocorrelation leakage |
| **B** | Chronological 80/20 train/test | Same-day pollutant concentrations ($t$) + calendar ($t$) | That day's AQI ($t$) | Isolates input reconstruction leakage under strict temporal ordering |
| **C** | Rolling-origin, expanding window, 30-day refit | Strictly past information ($t \le \text{origin}$) only | Next-day AQI ($t+1$) and 3-day ahead ($t+3$) | Leak-free realistic operational forecasting benchmarked against persistence |

---

## 2. Forecast Horizons & Feature Definitions

### Task N (Nowcast — Setups A & B)
- **Target**: $y_t = \text{AQI}_t$ (official bulletin index on day $t$).
- **Features**: Day-$t$ concentrations ($PM_{2.5}, PM_{10}, NO_2, SO_2, CO, O_3, NH_3$), calendar features (day of week, month, day of year, cyclic sin/cos transforms, season).
- **Note**: Reconstructs current-day air quality; mirrors the leaked Indian literature where concurrent inputs are treated as predictors.

### Task F (Forecast — Setup C)
- **Target Horizons**:
  - $h = 1$: Next-day AQI ($y_{t+1}$)
  - $h = 3$: 3-day ahead AQI ($y_{t+3}$)
- **Features ($X$)**:
  - Target lags: $\text{AQI}_{t}, \text{AQI}_{t-1}, \text{AQI}_{t-2}, \text{AQI}_{t-6}, \text{AQI}_{t-13}$ (shifted $\ge 1$ relative to target)
  - Pollutant lags: $PM_{2.5}, PM_{10}, NO_2, SO_2, CO, O_3, NH_3$ lagged by 1, 2, 3, 7 days (when available)
  - Rolling historical statistics: 3-day, 7-day, 14-day, 30-day rolling means and standard deviations of AQI
  - Calendar features: Target day calendar features (day of week, month, day of year, cyclic encodings, season)
  - **Zero Lookahead Rule**: Absolutely no unlagged pollutants, same-day target values, or future weather are included in $X$.

---

## 3. Rolling-Origin Evaluation Scheme (Setup C)

1. **Test Window**: The final 20% of continuous daily observations for each qualified city.
2. **Initial Training Window**: The first 80% of calendar dates.
3. **Window Expansion**: Expanding window (all historical observations up to forecast origin $t$ are retained in the training set).
4. **Refit Frequency**: Monthly cadence (every 30 calendar days). Predictions between refits use the most recently trained model weights.
5. **No Lookahead Imputation**: Missing value imputation (median / forward-fill) is fitted **strictly on the training window** and applied to the test observation. Test targets are never imputed.

---

## 4. Evaluation Metrics & Statistical Testing

1. **Primary Loss Metric**: Mean Absolute Error (MAE), reported with 95% confidence intervals via stationary block bootstrap ($B = 1,000$ iterations, block size $b = 7$ days).
2. **Secondary Metrics**:
   - Root Mean Squared Error (RMSE)
   - Coefficient of Determination ($R^2$)
   - Directional Accuracy (DA): percentage of days where sign of $\Delta y_{t+h}$ is correctly predicted.
3. **Statistical Significance vs Baseline**:
   - Diebold-Mariano test comparing each model against the **Persistence Forecaster** ($\hat{y}_{t+h} = y_t$).
   - Uses Newey-West heteroskedasticity and autocorrelation consistent (HAC) variance estimator with lag bandwidth $k = h - 1$.
   - Significance threshold: $p < 0.05$ (two-tailed).

---

## 5. Automated Leakage Guards

The pipeline enforces three mandatory automated verification tests:

1. **Temporal Separation Guard (`tests/test_leak_timestamp.py`)**:
   - Asserts $\max(\text{train\_dates}) < \min(\text{test\_dates})$ for every split in Setups B and C.
2. **Shuffled-Target Falsification Guard (`tests/test_leak_shuffled_target.py`)**:
   - Verifies that randomly permuting the training target completely eliminates predictive skill ($R^2 \le 0.05$, skill score $\le 0.05$ vs persistence).
3. **Planted-Leak Guard (`tests/test_leak_planted.py`)**:
   - Automatically computes pairwise feature-target correlation and raises a `LeakageError` if any feature achieves $|r| \ge 0.9999$ with the out-of-sample target.
