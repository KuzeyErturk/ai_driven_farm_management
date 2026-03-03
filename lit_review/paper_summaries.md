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

**Summary:** Derives 8 main agro-climatic zones across Europe using statistical cluster analysis. Finds that Eastern European zones have migrated northward ~100 km per decade, and future migration may double that rate. Mediterranean regions risk losing crop suitability while northern Europe gains capacity. Benefits from longer growing seasons are "often outbalanced by the risk of late frost and summer heat waves."

**Methodology:** Cluster analysis on two indicators: active temperature sum and thermal growing season length. Analysis of 1975-2016 observations and 5 high-resolution climate projections.

**Key Findings:**
- Agro-climate zones migrating northward ~100 km/decade
- Future rate may be double the historical
- Northern Europe gains capacity but with increased frost/heat risk

**Relevance to Our Work:** PROVIDES THE SCIENTIFIC BASIS for why cross-country pooling is justified. If zones are shifting northward, today's French/German weather-yield relationships ARE relevant to the UK's future. This is arguably the most important citation for justifying our approach.

---

## 3. Moore & Lobell (2015) — Climate Fingerprint on European Crop Yields

**Citation:** Moore, F.C. & Lobell, D.B. (2015). The fingerprint of climate trends on European crop yields. *PNAS*, 112(9), 2670-2675.
**DOI:** 10.1073/pnas.1409606112
**URL:** https://www.pnas.org/doi/10.1073/pnas.1409606112

**Keywords:** Climate attribution, European agriculture, wheat, barley, yield stagnation, policy effects

**Summary:** Investigates whether long-term climate changes have detectably impacted European crop yields. Finds that temperature and precipitation trends reduced continent-wide wheat yields by 2.5% and barley by 3.8%. Climate trends explain ~10% of yield stagnation; policy changes (CAP decoupling) explain more.

**Methodology:** Two statistical tests based on spatial pattern analysis. Panel data 1989-2009 across European regions with controls for soil, altitude, irrigation, subsidies.

**Key Findings:**
- Climate impacts detectable but modest relative to policy effects
- Mediterranean regions suffered most (5%+ declines)
- Northern regions modestly benefited from increased rainfall
- ~10% of wheat/barley yield stagnation attributed to climate

**Relevance to Our Work:** Establishes that climate signals ARE detectable in European crop yields, supporting our weather-based models. The 10% figure provides context — our R² values of 0.25-0.54 capture more than just climate trends (also inter-annual weather variability).

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

**Summary:** Evaluates ML for crop yield prediction at subnational level across 35 case studies in 9 European countries and 6 crops. Uses WOFOST crop model outputs + weather + remote sensing + soil. ML outperforms linear trend baselines (p < 3e-7). Captures spatial patterns well for average years but not extremes.

**Methodology:** NUTS-2/3 level data. Features from crop simulation, weather, remote sensing, soil. 5-fold sliding temporal validation.

**Key Findings:**
- Regional ML models had lower error than linear trends
- Extreme years poorly predicted
- Combining process-based model features with ML improves forecasting
- Separate models built per country-crop combination

**Relevance to Our Work:** Key comparison paper. They build SEPARATE models per country; we build a SINGLE POOLED model across countries. They use crop model outputs as features; we use raw weather. Showing that our simpler approach (weather-only, pooled) achieves competitive results would be a strong finding.

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

**Summary:** Applies ML + SHAP game theory to attribute European wheat yield shocks at subnational level. Finds a paradigm shift: water limitation was the dominant driver historically (32% of shocks), but extreme warming surpassed it in 2000-2018. By 2070-2099, 46-54% of areas will be heat-driven.

**Methodology:** ML attribution with SHAP. Daily weather at 25km. Four growing stages analyzed separately. Six crop models for future projections.

**Key Findings:**
- Historical: water limitation dominant (32% of yield shocks)
- Recent (2000-2018): extreme warming became dominant
- Paradigm shift projected to accelerate under RCP4.5/8.5
- 2003 was the year with highest yield shock area

**Relevance to Our Work:** Supports including temperature variables (especially extreme heat) in our models. Their stage-specific analysis suggests our aggregated seasonal features may miss within-season timing effects — a potential improvement direction.

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

**Keywords:** Heat stress, drought, wheat, rye, Germany, site conditions, phenological phases

**Summary:** Assesses genotype-environment-management interactions with weather in German wheat and rye variety trials (1993-2021). Combined heat-drought stress most damaging during reproductive phase. Poor soils amplify losses 2-3x. Modern varieties show higher absolute stress losses despite higher baselines.

**Methodology:** 28,187 wheat trials across 89 German sites. Mixed linear models with weather indices. Phenological staging via PHASE model.

**Key Findings:**
- Reproductive phase (heading to milk ripening) is most vulnerable
- Sandy, low-precipitation sites suffer 2-3x more yield loss
- Combined heat+drought more damaging than either alone
- Modern varieties more stress-sensitive in absolute terms

**Relevance to Our Work:** Explains why German models need different features than UK. Site/soil interactions with weather suggest our weather-only approach may underperform where soil variation is important. Their phenological timing finding supports our Grain_Filling_Rain and Grain_Filling_Sun features.

---

## 11. Evans et al. (2025) — Reconsidering Space-for-Time Substitution

**Citation:** Evans, M.E.K., Adler, P.B., Angert, A.L., et al. (2025). Reconsidering space-for-time substitution in climate change ecology. *Nature Climate Change*, 15(8), 809-812.
**DOI:** 10.1038/s41558-025-02392-0
**URL:** https://www.nature.com/articles/s41558-025-02392-0

**Keywords:** Space-for-time substitution, climate analogues, ecoclimate sensitivity, acclimation

**Summary:** Warns that space-for-time substitution can be "misleading not just in the magnitude but in the direction of effects." Case studies from tree growth, coral reefs, and soil microbiomes show that spatial climate gradients can predict responses incorrectly vs actual temporal responses. Acclimation processes cause sensitivities to differ across timescales.

**Key Findings:**
- Spatial patterns may fundamentally differ from temporal responses
- Acclimation at different timescales causes sensitivity shifts
- Direction of effects can be wrong, not just magnitude

**Relevance to Our Work:** CRITICAL LIMITATION TO ACKNOWLEDGE. Our cross-country pooling assumes French weather-yield relationships transfer to UK conditions — this paper warns that may not hold due to adaptation, different cultivars, soil, and farming practices. We should discuss this explicitly as a limitation and note that our negative transfer test results (all R² negative) partially validate this concern.

---

## 12. Challinor et al. (2014) — Meta-Analysis of Crop Yield Under Climate Change

**Citation:** Challinor, A.J., Watson, J., Lobell, D.B., et al. (2014). A meta-analysis of crop yield under climate change and adaptation. *Nature Climate Change*, 4(4), 287-291.
**DOI:** 10.1038/nclimate2153
**URL:** https://www.nature.com/articles/nclimate2153

**Keywords:** Meta-analysis, crop yield, climate change, adaptation, food security, temperature response

**Summary:** Synthesizes 1,700+ published crop yield simulations. Without adaptation, wheat, rice, and maize face aggregate losses at 2°C local warming. Crop-level adaptations increase simulated yields by 7-15%. Yield losses greater in the second half of the century. Tropical regions show strongest consensus on declines.

**Methodology:** Meta-analysis with OLS models. Three continuous variables (temperature, CO2, precipitation change) and three categorical (adaptation, temperate/tropical, C3/C4).

**Key Findings:**
- ~5% yield decline per 1°C warming (without adaptation)
- Adaptations add 7-15% yield benefit
- Wheat and rice more adaptable than maize
- Yield variability likely to increase

**Relevance to Our Work:** Provides the global context. The 5% per °C benchmark helps interpret our cross-country yield differences (e.g., France is 5°C warmer in summer than UK but yields are only modestly different — suggesting current adaptation is effective).

---

## 13. Battisti & Naylor (2009) — Historical Warnings of Future Food Insecurity

**Citation:** Battisti, D.S. and Naylor, R.L. (2009). Historical warnings of future food insecurity with unprecedented seasonal heat. *Science*, 323(5911), 240-244.
**DOI:** 10.1126/science.1164363
**URL:** https://www.science.org/doi/10.1126/science.1164363

**Keywords:** Food insecurity, seasonal heat, growing season temperature, extreme events, adaptation

**Summary:** Combines 1900-2006 observational climate data with 23 GCM projections. Over 90% probability that tropical/subtropical growing season temperatures will exceed all historical extremes by century's end. In temperate regions, the hottest seasons on record will become the norm. Uses historical heat events (e.g., 1972 USSR, 2003 Europe) as warnings.

**Key Findings:**
- >90% chance tropical growing season temps exceed all historical extremes
- Temperate record temperatures become future norm
- 2003 European heat wave as warning case
- Historical relationships may not hold under unprecedented conditions

**Relevance to Our Work:** Foundational motivation for climate-crop research. Their point that unprecedented temperatures render historical relationships unreliable is a challenge for our statistical models — if future UK temps exceed training data, extrapolation becomes unreliable. This supports the argument for cross-country pooling (expands the range of conditions in training data).
