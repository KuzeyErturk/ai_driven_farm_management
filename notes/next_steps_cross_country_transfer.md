# Next Steps: Solving Cross-Country Transferability

## The Transferability Problem

Our cross-country transfer tests (Experiment 2 in `src/france/cross_country_comparison.py`) showed that training a model on one country and predicting another gives **negative R²** — worse than simply predicting the mean yield. This means weather-yield relationships learned in one country do not directly apply to another.

### Why Transfer Fails

The failure is **structural**, not a data quantity issue:

1. **Different cultivars** — German wheat averages 9.8 t/ha vs UK's 8.0 t/ha. The same weather produces fundamentally different yields because the plants are different varieties bred for local conditions.

2. **Different soils and management** — Planting dates, fertilizer regimes, tillage practices, and harvest timing vary systematically by country. These are confounded with weather in the training data.

3. **Different climate baselines** — France's "normal" summer (25°C) is the UK's "hot" summer. A temperature value of 22°C means "cool" in France but "warm" in the UK. The same numerical weather input carries different agronomic meaning in each country.

4. **Different feature importance** — Our feature importance analysis (`src/france/feature_selection.py`) confirmed that the most important weather features differ substantially by country. For example, Wheat in the UK relies on `Grain_Filling_Temp` and `Autumn_Sun`, while France relies on `Spring_Rain_Squared` and `Flowering_Sun`. Only a few features (e.g., `Flowering_Temp`) are shared across all three countries.

5. **Space-for-time substitution limitation** — Evans et al. (2025, Nature Climate Change) warns that using spatial variation across locations to predict temporal change can be "misleading in both magnitude AND direction." Our negative transfer results are consistent with this warning.

### What Currently Works (Partially)

Our pooled Ridge model (all 47 features + binary Is_France/Is_Germany indicators) gives pooled LOOCV R² = 0.540 vs UK-only R² = 0.381. However:
- The high pooled R² is inflated by France (R² = 0.810)
- Binary country indicators are crude — they only adjust the **intercept** (base yield level), not the **slopes** (how each weather feature affects yield)
- We cannot confirm the pooled model actually predicts UK yields better than the UK-only model

---

## Potential Approaches to Solve This

### 1. Mixed-Effects Models (HIGH feasibility)
- Learn shared weather response slopes **plus** random country-specific intercepts **and** slopes
- Each country gets its own temperature-yield curve, rainfall-yield curve, etc.
- More principled than binary indicators — captures "temperature affects French wheat differently than UK wheat"
- Easy to implement with Python `statsmodels` (MixedLM) or R's `lme4`
- Standard approach in agricultural statistics
- **Why it could work:** Separates the transferable part (general weather response) from the non-transferable part (country-specific calibration)

### 2. Add Climatically Similar Countries (HIGH feasibility)
- Ireland, Netherlands, Belgium, Denmark — maritime climates closer to UK
- Smaller structural gap between these countries and the UK = easier transfer
- Northern France only (rather than all of France) would also reduce the gap
- Data sources: Eurostat yields + E-OBS weather (same pipeline we already built for France/Germany)
- **Why it could work:** Transfer is easier when countries share similar cultivars, soils, and climate baselines

### 3. Quantile Mapping / Bias Correction (HIGH feasibility)
- Pre-process yields to remove country baseline differences before pooling
- Train model on the "de-baselined" yields, then add country baselines back after prediction
- Common technique in climate science for bias-correcting model outputs
- **Why it could work:** Removes the most obvious source of transfer failure (different base yield levels) without needing complex algorithms

### 4. Domain Adaptation — CORAL/TCA (MEDIUM feasibility)
- Align feature distributions across countries so French weather "looks like" UK weather to the model
- Removes distributional shift before training
- Needs more data than currently available to work well
- **Why it could work:** Addresses the "same temperature means different things" problem by mapping features to a common space

### 5. Multi-Task Learning (MEDIUM feasibility)
- Neural network with shared hidden layers (learn common weather patterns) + country-specific output heads
- Each country's head learns its own yield calibration
- Needs more data, but could work with an augmented country set
- **Why it could work:** Explicitly separates shared knowledge from country-specific knowledge in the model architecture

### 6. Meta-Learning / MAML (LOW feasibility)
- "Learn to learn" — train on many countries so the model can quickly adapt to a new country from few samples
- Needs many source countries (10+) to be effective
- Interesting but not practical with current 3-country setup
- **Why it could work in theory:** Designed exactly for few-shot adaptation, but needs a larger set of source tasks (countries)

### 7. Long-History Detrended Anomaly Transfer (MEDIUM-HIGH feasibility)
- **Core idea:** Use the full historical depth of each country's data (especially France: 1900–2018) rather than trimming everyone to the UK's 2004–2018 window. Train robust weather-anomaly relationships on 60–100+ years of detrended data, then apply to UK weather anomalies.
- **Available historical depth:**
  - France (Schauberger/GDHY): **1900–2018** — 118 years of département-level yield + weather
  - Germany (OpenAgrar): ~1999–2021
  - IE/NL/BE/DK (Eurostat): ~2000–2023
  - UK: 2004–2018 (unchanged)
- **Why it differs from current transfer:** Current approach trains on 13 overlapping years per country. This approach uses France's full 60+ year detrended series to learn the *shape* of weather-yield responses (non-linear thresholds, interaction effects) with far more statistical power — more extreme years observed, better-estimated coefficients.
- **Implementation steps:**
  1. Load full historical yield + weather for each country (no year trimming)
  2. Detrend yields per country (loess or first-difference) to remove technology/variety confound
  3. Train weather → yield-anomaly models on long histories (France especially)
  4. Transfer the anomaly response curves to UK weather, then add UK's own trend back
- **Why it could work:**
  - Avoids direct level transfer — you're transferring the *response shape*, not the yield level
  - 60+ years captures more weather extremes (droughts, heatwaves) than 13 years, giving better-estimated non-linear responses
  - Technology/variety trends are removed by detrending, isolating pure weather signal
  - Continental-scale weather events (2003 heatwave, 1976 drought) hit multiple countries — long histories capture more shared extremes
  - EU countries adopted similar varieties/practices over time with lags — long French series captures yield plateau trends the UK also experienced
- **Challenges:**
  - Pre-1980 weather data quality degrades (ERA5 reanalysis back to 1940, E-OBS to ~1950)
  - Non-stationarity: weather-yield relationships change as varieties improve heat tolerance, irrigation expands — a 1970 relationship may not hold in 2010
  - Detrending must be done carefully to not remove climate-change signal that we want to capture
  - France is the only country with truly deep history; others add only 3–7 extra years
- **Best variant:** Use France's long detrended anomaly series as the primary training source. Our anomaly-based transfer already showed the first positive R² for OSR (+0.139), so this is a natural extension with much more training data.

---

## Key Insight

Simply adding more years of data from the same 3 countries will not solve the transfer problem. The issue is structural (cultivars, soils, management) not statistical (sample size). Solutions must either:
- **(a)** Explicitly model country-specific effects separately from shared weather responses (mixed-effects models)
- **(b)** Reduce the structural gap by using more climatically/agriculturally similar countries
- **(c)** Align distributions before training (domain adaptation, bias correction)

Or ideally, a combination of all three.

---

## References
- Evans et al. (2025) Nature Climate Change — warns about space-for-time substitution
- Ceglar et al. (2019) — agro-climate zone migration supports climate-analogue rationale
- Paudel et al. (2022) Field Crops Research — within-country pooling precedent
- Riedesel et al. (2024) — German vs UK yield differences due to site conditions and varieties
