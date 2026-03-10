# Dissertation Improvement Ideas

## High Impact, Feasible

1. **Prediction Intervals / Uncertainty Quantification**
   - Add bootstrap or quantile regression for confidence bands
   - Common examiner question: "how confident is the model?"

2. **Temporal Validation**
   - Clear train-on-2004-2018 / test-on-2019-2024 evaluation
   - Demonstrates generalisability beyond LOOCV

3. **7-Country Pooling Results**
   - IE, NL, BE, DK already added (`src/france/add_new_countries.py`) but not evaluated
   - Even "more countries doesn't always help" is a valid finding

4. **Fix the Risk Page**
   - Currently uses heuristics, disconnected from ML models
   - Either connect to ML or remove — examiner could question consistency

5. **Error / Residual Analysis**
   - Which years/regions does the model fail on?
   - e.g. "model underestimates in drought years" shows analytical depth

## Medium Impact

6. **Naive Baseline Comparison**
   - Beat "predict last year's yield" or "5-year rolling mean"
   - Standard in forecasting papers, strengthens claims

7. **Feature Ablation Study**
   - Performance with 5, 15, 30, 45 features
   - Does more always help? Strengthens feature engineering narrative

8. **Interactive SHAP in Flask**
   - Show SHAP waterfall for each prediction in the web app

## Nice-to-Have

9. **Methodology Diagram**
   - One figure: raw data → detrending → anomalisation → pooling → Ridge → calibration → prediction

10. **Climate Change Scenario**
    - Use model with UKCP18 projections to show yield impact under +2°C
