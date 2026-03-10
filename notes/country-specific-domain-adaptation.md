# Scenario-Adaptive Source Country Selection

## Motivation

Current best approach (pooled detrended anomaly, France+UK) treats all source data equally regardless of the weather scenario being predicted. But different countries have experienced different weather extremes — France has more heatwave/drought years, Ireland has more cold/wet years. The idea: **dynamically select which source country's data to emphasise based on how similar the prediction scenario is to each country's historical weather**.

Example: predicting a hot, dry UK summer → weight France data more heavily (they've seen many hot, dry summers). Predicting a cold, wet spring → weight Ireland/Denmark.

## Available Data (7 countries)

| Country | Regions | Rows/crop | Climate character |
|---------|---------|-----------|-------------------|
| UK | 4 | 52 | Target — maritime temperate |
| France | 4 | 51-52 | Continental-Mediterranean mix, more heat extremes |
| Germany | 4 | 52 | Continental, cold winters |
| Ireland | 1 | 13 | Oceanic, wet, mild |
| Netherlands | 1 | 7 | Maritime, flat, moderate |
| Belgium | 1 | 6 | Maritime transition |
| Denmark | 1 | 7 | Maritime-continental, cold |

**Note:** NL/BE/DK have <10 samples each — too small for standalone models. Spring_Barley has only 1-2 rows for NL/BE/DK. UK Barley in pooled file is unsplit; separate winter/spring barley CSVs exist.

## Current Baselines to Beat

| Crop | UK-only R² | Pooled Anomaly R² (FR+UK) |
|------|-----------|---------------------------|
| Wheat | 0.284 | 0.696 |
| Winter_Barley | 0.501 | 0.675 |
| Spring_Barley | 0.408 | 0.764 |
| Oats | -0.455 | 0.478 |
| OSR | 0.285 | 0.585 |

## Proposed Experiments

### Experiment 1: Sample-Level KNN

**Simplest approach.** For each held-out UK test point, find the k nearest historical samples across ALL 7 countries (in z-scored weather space), train a local Ridge on those k neighbours.

- Distance: Euclidean in StandardScaler-transformed z-score space
- Try k = 20, 30, 50, 100, all
- Track which countries contribute neighbours (diagnostic)
- If France always dominates the neighbours regardless of scenario → adaptive selection adds nothing

**Why it could work:** A drought UK year will naturally pull in French drought years as neighbours, even though France is climatically different. The anomalisation makes weather comparable.

### Experiment 2: Country-Level Weighted Blending

For each test point, compute weather similarity to each source country's distribution, then train Ridge with country-level sample weights.

- **2a:** Distance on 4 climate summary vars (Summer_Tmax, Annual_Rain, Winter_Tmin, Spring_Rain)
- **2b:** Distance on full 45-feature z-score vector
- Weight = exp(-d² / 2σ²), σ = median distance
- UK training samples always get weight 1.0

**More stable than KNN** because it doesn't discard any data — just reweights.

### Experiment 3: Adaptive Country Subset Selection

For each test point, rank countries by similarity, select top-N most similar, pool only those with UK data.

- Try N = 1, 2, 3, all
- Minimum 10 samples per selected country (filters out NL/BE/DK alone)
- Track selection frequency: how often is each country selected?
- If selection is stable across folds → no benefit over static subset

### Experiment 4: Per-Country Model Ensemble

Train separate Ridge model per country group, blend predictions using test-point similarity weights.

- Groups: France, Germany, Ireland, Benelux+DK (merged for sample size)
- Each model: country_group + UK_train → Ridge → predict
- Final prediction = weighted average of 4 group predictions
- Weights based on test-point-to-group-centroid distance

**Most expressive** but risks overfitting with small groups.

### Experiment 5: Static Ablation (Control)

Non-adaptive. Test all fixed country subsets:
- UK-only, UK+FR, UK+DE, UK+IE, UK+FR+DE, UK+FR+IE, UK+FR+DE+IE, UK+all7, UK+Maritime(IE+NL+BE+DK)

**Purpose:** If a fixed subset beats all adaptive methods, the adaptive machinery isn't worth it. Also reveals which countries help vs hurt.

## Key Design Decisions

1. **All experiments use the detrended anomaly pipeline** — detrend yields per (Region,Crop), z-score weather per region. This is proven essential.

2. **LOOCV on UK rows** — same evaluation as all previous experiments, fair comparison.

3. **NL/BE/DK grouped as "Benelux+DK"** for country-level methods (too small alone). For KNN, they participate naturally as individual samples.

4. **Country-level matching, not region-level** — sample sizes don't support region-level source selection.

5. **Ridge regression throughout** — consistent with all other experiments. Alpha tuning via nested LOOCV where beneficial.

## Expected Outcome

**Honest prediction:** The adaptive methods will likely perform **similarly to the static pooled baseline** (R² ~0.65-0.76) because:
- After anomalisation, weather is already comparable across countries
- Ridge naturally learns which features/samples are informative
- Small data limits the benefit of sophisticated selection

**But the experiment is valuable because:**
- If adaptive selection helps → novel contribution (scenario-dependent transfer)
- If it doesn't help → validates that anomalisation already solves the problem (also a finding)
- The ablation reveals optimal country subsets (may find UK+FR+IE > UK+FR)
- Diagnostics (neighbour composition, selection frequency) tell a compelling story

## Implementation

Single file: `src/france/scenario_adaptive_selection.py`

Reuse from existing code:
- `detrend_yields()` from `long_history_transfer.py`
- `standardise_weather_anomalies()` from `long_history_transfer.py`
- `select_features()` from `long_history_transfer.py`
- `_compute_climate_weights()` pattern from `domain_adaptation.py`
- Data loading pattern from `cross_country_comparison.py`
