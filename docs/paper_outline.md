# Paper Outline

## Title: The Leakage Illusion: Evaluating Next-Day AQI Predictability in Gujarat Under Rolling-Origin Validation

### Abstract
- Brief statement of the problem: inflation in reported ML accuracy for air quality index (AQI) prediction due to random train-test splitting and concurrent same-day pollutant inputs.
- Empirical study across Gujarat cities (Ahmedabad, Surat) and replication anchor (Chennai).
- Key findings: Setup A vs Setup B vs Setup C performance, failure/success of ML vs persistence.

### 1. Introduction
- Motivation: Severe air pollution in Western India and need for reliable early warnings.
- The methodological flaw in prior literature: using random splits and same-day pollutant concentrations to "predict" AQI.
- Contributions of this work.

### 2. Related Work
- Review of Indian and international AQI prediction studies.
- Synthesis of evaluation methodologies (random vs chronological vs rolling-origin).

### 3. Data & Preprocessing
- CPCB bulletin and CCR monitoring networks.
- Breakpoints and AQI reconstruction check.
- Missingness analysis and imputation rules (inside training folds only).

### 4. Experimental Protocol
- Formal definitions of Setups A, B, and C.
- Forecasting horizons ($h \in \{1, 3, 7\}$ days).
- Baselines and candidate ML/DL models.
- Statistical testing: Block bootstrap confidence intervals and Diebold-Mariano tests.

### 5. Results
- Claim 1: Accuracy inflation in Setups A and B versus Setup C.
- Claim 2: ML model performance versus persistence baseline in Setup C.
- Claim 3: Model ranking changes across validation setups.
- Horizon decay analysis ($t+1, t+3, t+7$).

### 6. Discussion & Limitations
- Why same-day pollutants reconstruct rather than forecast AQI.
- Data scarcity and station sensor degradation.
- Operational recommendations for regulatory forecasting.

### 7. Conclusion & Reproducibility Statement
- Final summary and link to public code repository and reproduction instructions.
