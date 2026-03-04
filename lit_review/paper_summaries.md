# Literature Review — Paper Summaries

## 1. Pugh et al. (2016) — Climate Analogues & Crop Yield Ceilings

**Citation:** Pugh, T.A.M., Muller, C., Elliott, J., et al. (2016). Climate analogues suggest limited potential for intensification of production on current croplands under climate change. *Nature Communications*, 7, 12608.
**DOI:** 10.1038/ncomms12608
**URL:** https://www.nature.com/articles/ncomms12608

**Keywords:** Climate analogues, attainable yield, crop intensification, food security, climate change

**Summary:** Uses the climate analogue method — matching present-day locations to projected future climates — to assess how crop productivity will change by 2050. Finds strong reductions in attainable yields of major cereals across a large fraction of current cropland. The total land area suitable for high yields is similar by 2050 to today, but in different locations, requiring large shifts in land-use patterns and crop choice.

**Methodology:** Gridded climate and agricultural data combined with 5 GCMs. Spatial analysis of cropland suitability for 2041-2059 and 2081-2099.

**Key Findings:**
- Major cereals face substantial productivity declines across current cropland by mid-century
- Suitable areas shift geographically rather than disappearing
- "Greatly reduced opportunity for agricultural intensification" on current croplands

**Relevance to Our Work:** Conceptual foundation for our cross-country approach. Pugh et al. use analogues descriptively (yield ceilings); we extend this to ML data augmentation — pooling yield-weather data from analogue countries to train better predictive models. This is a key novelty differentiator.

---

## 2. Ceglar et al. (2019) — Agro-Climate Zones Migrating Northward

**Citation:** Ceglar, A., Zampieri, M., Toreti, A. and Dentener, F. (2019). Observed northward migration of agro-climate zones in Europe will further accelerate under climate change. *Earth's Future*, 7(9), 1088-1101.
**DOI:** 10.1029/2019EF001178
**URL:** https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019EF001178

**Keywords:** Agro-climate zones, northward migration, growing season, temperature extremes, Europe

**Summary:** Derives 8 main agro-climatic zones across Europe using K-means cluster analysis on two agro-meteorological indicators: active temperature sum (ATS) and thermal growing season length (GSL). Identifies 8 zones: Boreal North, Boreal South, Nemoral, Continental, Pannonian, Northern Maritime, Southern Maritime, and Mediterranean. The UK falls primarily in the Northern Maritime zone. Comparing 1975-1995 to 1996-2016, Eastern European zones migrated northward up to 100 km/decade (nemoral zone most pronounced). Western Europe saw strong northward shift of maritime zones — northern maritime climate appeared in central-northern Germany, replacing continental climate; southern maritime expanded through much of France. Under 2°C warming (RCP8.5), migration velocities in Eastern Europe may double. Southern maritime climate is projected to prevail in central/northern France (replacing northern maritime), with growing season lengthening ~20 days and ATS increasing up to 700 degree-days.

**Methodology:** K-means clustering on gridded interannual averages of GSL and ATS from MARS-MCYFS database (1975-2016). Five EURO-CORDEX high-resolution regional climate models (RCP8.5) for projections to 2°C global warming. Winter wheat phenology simulated via WOFOST crop model. Four temperature-based extreme indicators: frost days (<0°C), tropical nights (>20°C consecutive), mild heat stress (>31°C for 2+ consecutive days), strong heat stress (>35°C for 3+ consecutive days).

**Key Findings:**
- 8 agro-climate zones identified; nemoral zone border shifted northward ~100 km/decade in Eastern Europe
- Continental and Pannonian zones progressed northward at ~70 km/decade
- Western Europe: northern maritime zone expanded into central-northern Germany (previously continental); southern maritime expanded across France
- Under 2°C warming: eastern European migration velocities may reach 130-200 km/decade
- Southern maritime projected to replace northern maritime in central/northern France — THIS MEANS France's current climate is migrating toward the UK
- Frost-affected area decreased across Europe, but late spring frost risk unchanged (earlier growing season onset = more frost exposure)
- Mild heat stress area doubled in Mediterranean and southern maritime zones during 1996-2016
- Even southern boreal zone experienced heat stress events that were absent in 1975-1995
- Shortening of grain filling period contributes to wheat yield stagnation (alongside direct heat stress)

**Relevance to Our Work:** PROVIDES THE SCIENTIFIC BASIS for why cross-country pooling is justified. Specifically: (1) The northern maritime zone (UK's current zone) is being replaced by southern maritime conditions — France's current climate IS the UK's near-future climate. (2) Southern maritime climate projected to prevail in central/northern France under 2°C warming, meaning today's French weather-yield patterns preview tomorrow's UK conditions. (3) Their finding that late spring frost risk persists despite warming supports our inclusion of Spring_Frost and Winter_Frost features. (4) Their threshold-based extreme indicators (>31°C, >35°C) support our use of extreme temperature features. This is arguably the most important citation for justifying our approach.

---

## 3. Moore & Lobell (2015) — Climate Fingerprint on European Crop Yields

**Citation:** Moore, F.C. & Lobell, D.B. (2015). The fingerprint of climate trends on European crop yields. *PNAS*, 112(9), 2670-2675.
**DOI:** 10.1073/pnas.1409606112
**URL:** https://www.pnas.org/doi/10.1073/pnas.1409606112

**Keywords:** Climate attribution, European agriculture, wheat, barley, yield stagnation, policy effects

**Summary:** First formal attribution of long-term yield trends to climate change in European agriculture. Uses a distinctive "fingerprint" approach: compares the spatial pattern of observed yield trends across 349 subnational regions (11 countries including UK) with the predicted pattern from climate-yield response functions. Key innovation: separates long-run (cross-sectional, including adaptation) and short-run (interannual, less adaptation) response functions to test for both climate impact detection and adaptation.

**Methodology:** Panel data 1989-2009 across subnational regions in Belgium, Germany, Greece, Spain, France, Ireland, Italy, Luxembourg, Netherlands, Portugal, UK. Response functions estimated using detrended yield and weather data with controls for soil quality, altitude, irrigation, subsidies, and country fixed effects. Quadratic functions in growing-season temperature and rainfall. Bootstrap-t procedure (500 resamples, block-bootstrapped by country) for significance testing that accounts for uncertainty in response function parameters.

**Key Findings:**
- Climate fingerprint statistically detectable for ALL four crops (wheat, barley, maize, sugar beet) at 5% level
- Wheat: β₁ = 0.41 (p < 0.002); Barley: β₁ = 0.50 (p = 0.012); Maize: β₁ = 1.39 (p = 0.002); Sugar beet: β₁ = 0.82 (p = 0.02)
- Production-weighted impacts: wheat −2.5%, barley −3.8%, maize +0.3%, sugar beet +0.2%
- In absence of warming since 1989, wheat yields would be 3.5% higher, barley 3.8% higher
- Italy hardest hit (10%+ losses from warming + drying)
- Climate trends explain ~10% of wheat/barley yield stagnation; CAP subsidy decoupling (1993 and 2004 reforms) explains more
- If yields had continued growing at pre-1995 rates, wheat and barley yields would be 30% and 37% higher today
- Test for adaptation (test 2) has very low statistical power — "null results showing no evidence for adaptation should be interpreted with caution"

**Relevance to Our Work:** (1) Establishes that climate signals ARE detectable in European crop yields using the same type of subnational data we use. (2) The 10% figure provides context — our R² values of 0.25-0.54 capture more than just long-term climate trends (also inter-annual weather variability, which is our focus). (3) Their inclusion of UK data (alongside France, Germany) shows cross-country analysis of yield-climate relationships is methodologically standard. (4) The low power for detecting adaptation warns us that our negative cross-country transfer results don't necessarily mean adaptation doesn't matter. (5) Their finding that policy (CAP reforms) dominates climate in explaining yield stagnation supports our use of Area_hectares as a feature (proxy for policy/economic effects).

---

## 4. Gammans et al. (2017) — Climate Impacts on French Cereal Yields

**Citation:** Gammans, M., Merel, P. and Ortiz-Bobea, A. (2017). Negative impacts of climate change on cereal yields: statistical evidence from France. *Environmental Research Letters*, 12(5), 054007.
**DOI:** 10.1088/1748-9326/aa6b0c
**URL:** https://iopscience.iop.org/article/10.1088/1748-9326/aa6b0c

**Keywords:** Climate change, cereal yields, France, panel data, temperature response, wheat, barley

**Summary:** Uses 65 years of French department-level yield data with panel regression. Finds all crop yields respond negatively to spring-summer warming and excess precipitation. Projects 21% winter wheat decline, 17.3% winter barley decline, 33.6% spring barley decline by end-of-century under RCP8.5.

**Methodology:** Department-level yields (1950-2015) from French Ministry of Agriculture. E-OBS weather at 0.25° resolution. Fixed-effects regression with flexible temperature specifications. Growing seasons: winter wheat/spring barley Mar 1-Jul 31, winter barley Mar 1-Jul 15.

**Key Findings:**
- Temperatures above 32°C reduce wheat yields by -2.9% per day
- Spring barley is most heat-sensitive (-4.6% at 32°C+)
- Yields peak at 200-236mm spring-summer precipitation
- Historical tech gains (1.6-1.7%/yr) exceed projected climate losses

**Relevance to Our Work:** Uses the SAME French department-level data structure as our Schauberger dataset. Their fixed-effects panel approach is an alternative to our ML methods for the same problem. Their crop-specific sensitivity findings inform which weather features matter most. They also used E-OBS — same weather source as us.

---

## 5. Paudel et al. (2022) — ML Crop Yield Forecasting Across Europe

**Citation:** Paudel, D., Boogaard, H., de Wit, A., et al. (2022). Machine learning for regional crop yield forecasting in Europe. *Field Crops Research*, 276, 108377.
**DOI:** 10.1016/j.fcr.2021.108377
**URL:** https://www.sciencedirect.com/science/article/pii/S0378429021003233

**Keywords:** Machine learning, crop yield forecasting, Europe, regional prediction, WOFOST, subnational

**Summary:** Proposes and evaluates a generic ML workflow for regional crop yield forecasting across 35 case studies (9 countries: BG, DE, ES, FR, HU, IT, NL, PL, RO; 6 crops: soft wheat, spring barley, sunflower, grain maize, sugar beets, potatoes). Models built per crop-country combination by pooling all regions within a country. Data from MCYFS (1999-2018). ML outperforms linear trend baselines significantly (Wilcoxon p = 3e-7) and performs comparably to the EU's operational MCYFS forecasting system at 60 days before harvest (Wilcoxon p = 0.95, no significant difference). However, ML predictions are conservative — staying close to trend means and struggling with extreme years.

**Methodology:** Features from WOFOST crop model outputs, weather observations, remote sensing (NDVI, fAPAR), soil water holding capacity, elevation, slope, crop area, irrigated crop area, average field sizes. Dynamic crop calendars (per-region, per-year). Four ML algorithms compared: Ridge Regression, KNN, SVR, Gradient Boosted Decision Trees (GBDT). 5-fold sliding temporal validation for feature selection and hyperparameter tuning; per-test-year model refitting using all prior data. Bayesian optimization for hyperparameters (replacing grid search in baseline). Forecasts at 120, 90, 60, 30 days before harvest and end-of-season.

**Key Findings:**
- Optimized ML: median regional NRMSE = 16.57% (vs 17.27% baseline, 20.35% trend) at 60 days before harvest
- ML significantly better than trend for ALL 35 cases at 120 days before harvest
- At national level: ML median NRMSE 8.41% vs MCYFS 8.81% (60 days early) — comparable performance
- At end-of-season: MCYFS significantly better than ML (6.74% vs 7.49%) because analysts update forecasts with expert knowledge and news reports
- ML captured regional differences for average harvests (~71% regions matched for potatoes 2013) but not extremes (grain maize 2015: only 52% matched; French wheat 2016: 53% matched)
- ML prediction residuals had lower variance and fewer outliers than trend residuals
- Z-score features for weather extremes "not always effective"
- Spring barley (FR) and sunflower (FR) had suspicious near-identical reported yields — data quality concerns
- Overfitting detected in cases where baseline outperformed optimized model (validation vs test distribution mismatch)
- "We pooled data from possibly very different regions to have a large enough data size for machine learning" — explicit acknowledgment of data size constraint

**Relevance to Our Work:** (1) KEY COMPARISON: They build SEPARATE models per country by pooling within-country regions; we build a SINGLE POOLED model across countries. Their within-country pooling rationale (data size for ML) is exactly our cross-country rationale. (2) Their use of Ridge, SVR, GBDT mirrors our algorithm choices (RF, Ridge, SVR). (3) Their finding that ML struggles with extremes matches our experience. (4) Their use of WOFOST crop model features + remote sensing achieves only marginally better NRMSE than trend — our simpler weather-only approach is reasonable. (5) Their 16-17% regional NRMSE benchmark contextualizes our performance. (6) Their French data quality concerns are relevant if we use French data for cross-country pooling. (7) Their explicit acknowledgment that "different regions" can be pooled when data is scarce directly supports our cross-country extension of the same principle.

---

## 6. Corcoran et al. (2023) — UK Crop Yield Data Bottlenecks

**Citation:** Corcoran, E., Afshar, M., Curceac, S., et al. (2023). Current data and modeling bottlenecks for predicting crop yields in the United Kingdom. *Frontiers in Sustainable Food Systems*, 7, 1023169.
**DOI:** 10.3389/fsufs.2023.1023169
**URL:** https://www.frontiersin.org/journals/sustainable-food-systems/articles/10.3389/fsufs.2023.1023169/full

**Keywords:** UK crop yields, data bottlenecks, wheat, crop management, climate data, modelling

**Summary:** Identifies substantial disparities between predicted and actual UK crop yields. Openly available crop management and plant physiology data is scarce, especially for non-wheat crops. Climate and soil datasets at broad scales could enable expansion beyond field-level models.

**Methodology:** Systematic review using Scopus, Web of Science, UK government open data. Variables categorized into 6 groups.

**Key Findings:**
- Major data gaps in crop management and physiology data (especially non-wheat)
- Contemporary yield predictions diverge substantially from observations
- Existing models focus on field-level with minimal landscape extrapolation
- Large-scale climate and soil datasets underutilized

**Relevance to Our Work:** THE MOST DIRECTLY RELEVANT PAPER. Explicitly identifies the data scarcity problem our cross-country pooling addresses. Our 336-observation dataset across 4 UK regions exemplifies their documented limitations. Their finding that non-wheat crops are data-poor explains our lower R² for Oats and OSR. CITE THIS as the motivation for the cross-country approach.

---

## 7. Ronchetti et al. (2024) — Harmonized EU Subnational Crop Statistics

**Citation:** Ronchetti, G., Nisini Scacchiafichi, L., et al. (2024). Harmonized European Union subnational crop statistics can reveal climate impacts and crop cultivation shifts. *Earth System Science Data*, 16, 1623-1649.
**DOI:** 10.5194/essd-16-1623-2024
**URL:** https://essd.copernicus.org/articles/16/1623/2024/

**Keywords:** Subnational crop statistics, EU harmonization, NUTS regions, cultivation shifts, yield extremes

**Summary:** Presents harmonized subnational crop statistics: 344,282 records, 961 regions, 27 EU countries, 1975-2020, for wheat, barley, grain maize, sunflower, sugar beet. Documents crop cultivation shifts: grain maize and durum wheat displaced ~10 km/year northeastward; sunflower shifted 25 km/year eastward.

**Methodology:** Three-step harmonization from National Statistical Institutes + EUROSTAT. NUTS 2016 boundary reconciliation. Production zone centroid displacement analysis.

**Key Findings:**
- Median time series span 21 years across EU regions
- Crop production zones shifting northeastward
- 2016 severe French wheat loss and 2018 drought impacts documented
- Spring/winter barley shifts minimal (~5 km/year)

**Relevance to Our Work:** Shows that the kind of cross-country harmonized data we built manually (UK+France+Germany) could be extended to 27 EU countries. Their documented northward shifts support the climate analogue rationale.

---

## 8. Zhu et al. (2021) — ML Attribution of European Wheat Yield Shocks

**Citation:** Zhu, P., Abramoff, R., et al. (2021). Uncovering the Past and Future Climate Drivers of Wheat Yield Shocks in Europe With Machine Learning. *Earth's Future*, 9, e2020EF001815.
**DOI:** 10.1029/2020EF001815
**URL:** https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020EF001815

**Keywords:** Wheat yield shocks, machine learning, SHAP, climate drivers, heat stress, Europe

**Summary:** Builds a data-driven attribution framework using Random Forest + SHAP (Shapley additive explanations) to identify primary climate drivers of European wheat yield shocks across 1,435 subnational administrative units in 17 countries (1980-2018). Yield shocks defined as lowest 10th percentile of relative yield anomalies (detrended via LOESS). Europe divided into 4 regions (North, South, East, West) with separate RF models per region. Five primary climate driver categories: extreme warming, high water demand, excessive water supply, low water supply, and cold stress. 20 climate predictors (5 variable types × 4 growth stages: autumn, winter, vegetative period, reproductive period). Also evaluates 8 process-based crop models against the data-driven attribution.

**Methodology:** Random Forest classification (yield shock = binary variable). ntree = 1000, mtry tuned via 5-fold CV (11-14 depending on region). AUC > 0.85 for all regions (North highest at 0.95). SHAP values used to decompose predicted shock probability into individual predictor contributions. Climate data from JRC MARS database at 25km resolution. Daily variables: Tmin, Tmax, precipitation, wind speed, global radiation, vapour pressure deficit, potential evapotranspiration. Five climate variable types per growth stage: fraction of warm days (>30°C or >90th percentile), mean precipitation, fraction of rainy days, PET, fraction of cold days (<0°C).

**Key Findings:**
- 2003 had highest yield shock area across Europe (compound heat + drought)
- Yield shock frequency higher in southern, warmer areas (Romania, Spain, southern France)
- North and East Europe show increasing trend in area fractions of yield shock
- Overall 1980-2018: low water supply was most pervasive primary driver (32% of shocks), followed by extreme warming, high water demand, excessive water supply
- Regional variation: South/East Europe dominated by low water supply; North/West Europe by excessive water supply
- PARADIGM SHIFT: from 1980-1999 to 2000-2018, dominant driver shifted from low water supply to extreme warming for whole Europe
- Reproductive period is the critical stage for extreme warming-driven shocks; vegetative period is critical for water-driven shocks
- Partial dependence plots: same predictor can have quite different effects across regions — "spatially varying sensitivities justify our partitioning of Europe into several regions"
- Crop models captured low water supply impacts well (>60% agreement with 3+ of 6 models) but FAILED for excessive water supply and autumn/winter stresses
- Future projections (2070-2099): extreme warming becomes dominant driver — 46% (RCP4.5) to 54% (RCP8.5) of areas heat-driven
- Crop model agreement DECREASES for future climate — existing models may be inadequate for forecasting future yield shocks
- Sensitivity analysis: results robust to alternative detrending (Savitzky-Golay) and ±14 day phenology shifts

**Relevance to Our Work:** (1) Strongly supports including temperature variables (especially extreme heat) in our models — warming has already surpassed water limitation as the primary yield shock driver in 2000-2018. (2) Their regional RF approach (separate models for North/West/East/South) parallels our UK-only vs cross-country comparison. (3) The finding that the same predictor has different effects across regions directly supports building region-specific models rather than one-size-fits-all. (4) Their SHAP-based attribution methodology could be applied to our models for interpretability. (5) Crop models failing on excess water and winter stresses suggests our empirical weather-based approach may capture relationships that process-based models miss. (6) The stage-specific analysis (autumn, winter, vegetative, reproductive) suggests our aggregated seasonal features may miss within-season timing effects — a potential improvement direction. (7) Their use of the JRC MARS database and LOESS detrending aligns with standard practices.

---

## 9. Dinh & Aires (2022) — Leave-Two-Out Cross-Validation for Small Datasets

**Citation:** Dinh, T.L.A. and Aires, F. (2022). Nested leave-two-out cross-validation for the optimal crop yield model selection. *Geoscientific Model Development*, 15, 3519-3535.
**DOI:** 10.5194/gmd-15-3519-2022
**URL:** https://gmd.copernicus.org/articles/15/3519/2022/

**Keywords:** Cross-validation, model selection, small samples, overfitting, LOOCV, crop yield

**Summary:** Proposes leave-two-out (LTO) cross-validation as alternative to LOO for small crop yield datasets. LOO favours overly complex models; LTO consistently selects simpler, more generalizable ones. Tested on Vietnamese coffee (n=19) and French maize (n=22).

**Methodology:** Comparison of LOO vs LTO. Linear regression and neural networks tested with varying complexity.

**Key Findings:**
- LOO recommends increasingly complex models
- LTO selects simpler, more robust alternatives
- Simple linear models more reliable for small datasets (n<25)
- Neural networks overfit excessively with limited samples

**Relevance to Our Work:** Directly addresses our core challenge (n=84 per crop). Our finding that RF overfits (train R²~0.88 vs LOOCV ~0.19) and Ridge outperforms aligns perfectly. Our use of LOOCV is validated but their warning about LOO favouring complex models applies — Ridge's regularization naturally counters this tendency.

---

## 10. Riedesel et al. (2024) — Site-Specific Heat/Drought Impacts on German Wheat

**Citation:** Riedesel, L., Moller, M., Piepho, H.-P., et al. (2024). Site conditions determine heat and drought induced yield losses in wheat and rye in Germany. *Environmental Research Letters*, 19(3), 034024.
**DOI:** 10.1088/1748-9326/ad24d0
**URL:** https://iopscience.iop.org/article/10.1088/1748-9326/ad24d0

**Keywords:** Heat stress, drought, wheat, rye, Germany, site conditions, phenological phases, G×E×M

**Summary:** First comprehensive analysis of site-specific combined heat and drought stress effects on wheat and rye across all German cereal growing regions using 28,187 wheat trials (403 varieties, 89 sites) and 10,290 rye trials (93 varieties, 85 sites) from pre-registration variety trials (1993-2021). Uses mixed linear models with G×E×M (genotype × environment × management) specific covariates. Combined heat-drought weather indices (WIs) defined as days exceeding temperature thresholds (27°C, 29°C, 31°C) AND soil moisture below thresholds (50%, 30%, 10% of plant available water capacity). Phenological stages derived from PHASE model + variety trial data + growing-degree-day calculations for missing stages.

**Methodology:** Linear mixed models (equation 7 in paper) with genotype, site, year, trial series as factors. Fixed regression coefficients for: (1) environmental covariates (soil quality, precipitation, soil type, SQR), (2) management covariates (N-fertilization, crop protection), (3) combined heat-drought WIs per phenological period. Variance reduction method for covariate selection (forward selection, threshold ≥0.5%). Six stage-to-stage periods analyzed: stem elongation→booting, booting→heading, heading→anthesis, anthesis→milk ripening, milk ripening→dough ripening, dough ripening→yellow ripening. Plus three cross-stage periods. Site clusters: precipitation (>/<650mm), soil quality (>/<50 points), soil type (loam/sand), SQR (>/<60 points).

**Key Findings:**
- Combined heat-drought stress increasing significantly from 1993-2021, especially post-heading periods
- Phenological acceleration: stem elongation and yellow ripening shifted 8-10 days earlier; heading changed little → grain-filling period SHORTENING over time
- Strongest explanatory power for combined WIs during heading→dough ripening (HDR) period for both crops
- Heading→anthesis is the single most significant stage-to-stage period for wheat yield effects
- Site-specific effects are DRAMATIC: poor sites (low soil quality, sandy soil, <650mm precipitation) suffer 2-3× more yield loss than good sites across all stress intensities
- Drought is the driving force: yield losses on poor sites increase more with drought intensity than with heat intensity
- Rye reaches anthesis 20 days earlier than wheat, experiencing less pre-anthesis stress
- Modern varieties: significantly higher absolute AND relative stress losses in wheat; higher absolute but NOT relative losses in rye
- Average wheat yield: 9.8 t/ha; rye: 8.8 t/ha (German variety trial conditions — much higher than UK national averages)
- Breeding progress outweighs negative annual weather trend, but breeding has NOT improved stress tolerance
- Site conditions outweigh genetic advantages: rye's theoretical stress tolerance is negated by being grown on marginal sites

**Relevance to Our Work:** (1) Explains why German yield-weather relationships may not transfer directly to UK — German wheat trials average 9.8 t/ha vs UK ~7-8 t/ha due to different site conditions, varieties, and management. (2) Their finding that site/soil interactions with weather cause 2-3× variation in stress losses means our weather-only approach (without soil data) will have limited explanatory power for German data. (3) The phenological timing finding STRONGLY supports our Grain_Filling_Rain and Grain_Filling_Sun features — the heading→dough ripening period is where weather effects concentrate. (4) The grain-filling period is shortening over time — a confounding trend in our data. (5) Their threshold-based WIs (>27°C, >29°C, >31°C) suggest our Summer_Tmax feature may be too aggregated — threshold exceedance counts could be more informative. (6) The finding that combined heat+drought is worse than either alone suggests interaction features could improve our models.

---

## 11. Evans et al. (2025) — Reconsidering Space-for-Time Substitution

**Citation:** Evans, M.E.K., Adler, P.B., Angert, A.L., et al. (2025). Reconsidering space-for-time substitution in climate change ecology. *Nature Climate Change*, 15(8), 809-812.
**DOI:** 10.1038/s41558-025-02392-0
**URL:** https://www.nature.com/articles/s41558-025-02392-0

**Keywords:** Space-for-time substitution, climate analogues, ecoclimate sensitivity, acclimation, forecasting

**Summary:** A Comment piece in Nature Climate Change warning that space-for-time substitution (SFTS) — using patterns observed across spatial climate gradients to predict impacts of climate change — can be "misleading not just in the magnitude but in the direction of effects." Central example: *Pinus ponderosa* trees grow faster at warmer locations (spatial pattern) but slower in warmer-than-average years (temporal pattern). So SFTS forecasting predicts trees should BENEFIT from warming, whereas temporal data shows they SUFFER. This sign reversal has been found across multiple systems: other tree species, grassland productivity, pathogen-driven forest mortality, herbaceous plant demography, bird abundance.

**Two key reasons SFTS fails:**
1. **Causality assumption violation:** SFTS assumes climate directly causes the observed spatial pattern. But spatial correlations may reflect confounding factors (soil, management, genetics) that co-vary with climate. Machine learning and AI "excel at matching patterns" but this makes the causality problem worse — an era-specific warning.
2. **Lagging ecological processes:** Organisms, communities, and ecosystems adjust to climate via processes operating on different timescales — fast (physiological acclimation, plasticity) to slow (evolution, dispersal, species colonization). Transient dynamics during the lag period can produce responses opposite to equilibrium predictions.

**Recommendations:**
- Qualitatively assess whether SFTS assumptions are likely met for the study system
- SFTS more reliable when: strong causal climate→response link exists, evolution/dispersal are rapid, forecast horizon is long relative to lagging processes
- Compare spatial vs temporal climate sensitivities — if consistent, uncertainty is lower
- Use multiple validation approaches beyond cross-validation (hindcasting, near-term forecasting, out-of-sample validation)
- Develop forecasting approaches using causal inference or process-based models

**Key Findings:**
- Spatial patterns may fundamentally differ from temporal responses — sign can be REVERSED
- Acclimation at different timescales causes sensitivity shifts
- Direction of effects can be wrong, not just magnitude
- Abundant spatial data + flexible ML "excel at matching patterns" but heighten the causality risk
- "Transient dynamics can lead to permanent or quasi-permanent undesirable outcomes"

**Relevance to Our Work:** CRITICAL LIMITATION TO ACKNOWLEDGE. (1) Our cross-country pooling is fundamentally an SFTS approach — we assume spatial weather-yield relationships (France/Germany) transfer to temporal changes (UK's future climate). Evans et al. warn this can fail due to different cultivars, soils, farming practices, and adaptation timescales between countries. (2) Our negative transfer test results (all cross-country R² negative) are CONSISTENT WITH their warning — spatial relationships don't transfer in our case. (3) Their warning about ML "excelling at matching patterns" without causality is directly relevant to our RF models. (4) However, our system differs from their ecological examples in important ways: agricultural crops are actively managed, breeding/adaptation is faster than for trees, and we use inter-annual variability (not just spatial gradients). (5) We should cite this explicitly as a limitation and frame our negative transfer results as partially validating Evans et al.'s concerns about SFTS in agricultural contexts.

---

## 12. Challinor et al. (2014) — Meta-Analysis of Crop Yield Under Climate Change

**Citation:** Challinor, A.J., Watson, J., Lobell, D.B., et al. (2014). A meta-analysis of crop yield under climate change and adaptation. *Nature Climate Change*, 4(4), 287-291.
**DOI:** 10.1038/nclimate2153
**URL:** https://www.nature.com/articles/nclimate2153

**Keywords:** Meta-analysis, crop yield, climate change, adaptation, food security, temperature response

**Summary:** Synthesizes 1,700+ published crop yield simulations (from 66 studies) into a unified dataset to evaluate impacts of climate change and adaptation on wheat, rice, and maize. Uses OLS with robust covariance matrix estimates (clustered by study). Separates analysis by temperate/tropical regions, C3/C4 metabolism, and presence/absence of adaptation. Adaptations studied are incremental, crop-level adjustments (planting date, fertilizer, irrigation, cultivar change) — not systemic or transformational changes.

**Methodology:** General linear model with 3 continuous variables (ΔT, ΔCO₂, ΔP) and 3 categorical variables (adaptation yes/no, temperate/tropical, C3/C4). n = 882 for the main model. 33 paired studies (with and without adaptation) for quantifying adaptation benefits. 500 bootstrap samples for confidence intervals. Temporal analysis using 20-year and 10-year bins.

**Key Findings (from full text):**
- Temperature coefficient: −4.90% per °C (p < 0.001) — the strongest predictor
- Precipitation: +0.53% per % change (p = 0.003)
- CO₂: +0.06% per ppm (p = 0.002) — partial offset
- Adaptation: +7.16% average benefit (p = 0.022)
- Region (temperate vs tropical): not significant (p = 0.47)
- C3 vs C4 metabolism: not significant (p = 0.99)
- Adaptation benefits relatively consistent across temperature increases and rainfall changes (7-15% range)
- Cultivar adjustment was the most effective adaptation, followed by irrigation
- Maize: little evidence for adaptation potential, especially in tropics (counterintuitive — due to different modelling methods in adaptation vs non-adaptation studies)
- Temporal consensus: yield decreases in 2nd half of century are stronger and more certain than 1st half
- ALL positive yield changes in 2070s/2090s come from temperate regions — "strong consensus that yields of tropical crops will decrease"
- Yield variability increases projected (CV increases of 50-300% in some studies)
- Key limitations acknowledged: most studies don't simulate pests/weeds/diseases, many assume continued water availability, CO₂ fertilization effects uncertain

**Relevance to Our Work:** (1) The −4.9% per °C coefficient provides a benchmark for interpreting cross-country yield differences. France is ~3-5°C warmer in summer than Scotland — the Challinor coefficient would predict 15-25% lower yields, but French yields are actually HIGHER, showing adaptation + CO₂ + management effects dominate. (2) Their finding that cultivar adjustment is the most effective adaptation highlights why cross-country transfer (different cultivars) is problematic. (3) The increasing yield variability projection supports our interest in predicting extreme years. (4) Their meta-analysis approach (pooling diverse studies) is analogous to our data pooling approach. (5) The −4.9%/°C figure for temperature is consistent with Gammans et al.'s French findings and supports temperature being our most important predictor category.

---

## 13. Battisti & Naylor (2009) — Historical Warnings of Future Food Insecurity

**Citation:** Battisti, D.S. and Naylor, R.L. (2009). Historical warnings of future food insecurity with unprecedented seasonal heat. *Science*, 323(5911), 240-244.
**DOI:** 10.1126/science.1164363
**URL:** https://www.science.org/doi/10.1126/science.1164363

**Keywords:** Food insecurity, seasonal heat, growing season temperature, extreme events, adaptation

**Summary:** Combines 1900-2006 observational climate data with 23 GCM projections to demonstrate that future growing season temperatures will be unprecedented. Summer (JJA for Northern Hemisphere) defined as proxy for main growing season. Over 90% probability that tropical/subtropical seasonal temperatures will exceed ALL historical extremes by century's end. In temperate regions, the hottest seasons on record will become the median — not the extreme — by 2090.

**Historical case studies (from full text):**
- **2003 European heat wave:** ~52,000 deaths. France: mean summer temp 3.6°C (3.5σ) above long-term mean. Italy: maize yields dropped 36% from prior year. France: maize/fodder −30%, fruit −25%, wheat −21%. By end of century, 2003-like summers become the NORM for France.
- **1972 USSR:** Peak temperatures >30°C during key wheat/coarse grain development stages caused 13% grain production decline. Contrary to popular narrative, drought was NOT the main factor — only 0.5σ below mean precipitation. The heat wave triggered Soviet intervention in international markets (unprecedented policy shift), wheat prices tripled ($60→$208/metric ton in 2 years). Demonstrates how regional crop failures cascade into global food security impacts.
- **Crop yield losses:** Direct yield losses of 2.5-16% per 1°C increase in seasonal temperature for major grains (experimental and model-based estimates).

**Key Findings:**
- >90% probability that by 2090, tropical/subtropical growing season temps exceed ALL 1900-2006 records
- In temperate regions (including Europe), the hottest season on record becomes the future median
- "The projected seasonal average temperature represents the median, not the tail, of the climate distribution"
- Food deficits will be GLOBAL — "extremely difficult to balance food deficits in one part of the world with surpluses in another"
- Major adaptation investments needed NOW: heat-tolerant varieties, irrigation systems, breeding programs
- National and international agricultural investments have been "waning in recent decades"

**Relevance to Our Work:** (1) Foundational motivation for climate-crop research — crop yields will face unprecedented heat stress. (2) Their key point that unprecedented temperatures render historical relationships unreliable is a fundamental challenge for our statistical models — if future UK temps exceed anything in our training data (2004-2024), our models cannot extrapolate reliably. (3) This DIRECTLY supports the argument for cross-country pooling: by including French/German data where current temperatures are higher, we expand the range of conditions in training data, making the model more robust to future UK warming. France's current summers approximate the UK's projected future. (4) The 2003 case study (France −21% wheat) and 1972 case study (USSR −13% grain) demonstrate the magnitude of yield shocks that our models should aim to predict. (5) Their 2.5-16% per °C loss range contextualizes Challinor et al.'s −4.9%/°C finding.
