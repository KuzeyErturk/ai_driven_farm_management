# AI-Driven Crop Yield Prediction System

Kuzey Erturk - Loughborough University - Final Year Project

## Overview

A machine learning system for predicting UK crop yields (Wheat, Winter Barley, Spring Barley, Oats, Oilseed Rape) from seasonal weather features. The project develops a **detrended anomaly transfer** approach that pools weather–yield relationships across European countries (UK, France, Ireland, Netherlands, Belgium, Denmark, Germany) to overcome the UK's limited training data (13 years, 9 regions).

Key contributions:
- **Cross-country pooled anomaly models** that improve UK prediction R² from 0.20–0.35 (UK-only) to 0.44–0.76 (pooled)
- **Domain adaptation comparison** of methods (Ridge + indicators, MixedLM, bias correction, CORAL)
- **SHAP explainability analysis** comparing UK-only vs pooled vs cross-country feature importance
- **Flask web application** for interactive yield prediction and scenario analysis

## Project Structure

```
agricultural-ai-project/
├── src/
│   ├── features/                       # Feature engineering
│   │   ├── seasonal_features.py        # Seasonal weather features from Met Office data
│   │   └── barley_features.py          # Winter/Spring barley separation
│   │
│   ├── models/                         # UK-only modelling
│   │   ├── baseline_model_config.py    # Crop-specific model selection (RF/Ridge/SVR)
│   │   ├── crop_models.py              # Per-crop model training and evaluation
│   │   ├── barley_models.py            # Winter/Spring barley analysis
│   │   ├── regional_models.py          # Regional variation analysis
│   │   ├── model_improvements.py       # Feature selection and tuning
│   │   ├── shap_analysis.py            # SHAP explainability (3-part analysis)
│   │   └── dissertation_final_results.py  # Final results for dissertation tables
│   │
│   └── france/                         # Cross-country transfer learning
│       ├── config.py                   # Paths, NUTS mappings, crop codes
│       ├── download_yields.py          # Eurostat yield data download
│       ├── build_france_datasets.py    # France dataset construction
│       ├── add_new_countries.py        # IE/NL/BE/DK data pipeline (Eurostat + E-OBS)
│       ├── process_eobs_weather.py     # E-OBS weather data processing
│       ├── feature_selection.py        # Decorrelated feature selection for MixedLM
│       ├── cross_country_comparison.py # Pooling experiments (FR + UK)
│       ├── domain_adaptation.py        # Domain adaptation methods comparison
│       ├── long_history_transfer.py    # Detrended anomaly transfer (best method)
│       └── scenario_adaptive_selection.py  # Adaptive source selection (negative result)
│
├── flask_app/                          # Web application
│   ├── app.py                          # Flask routes and prediction logic
│   ├── models/                         # Trained models: feature_info.json + model_config.json
│   │                                   #   (the .pkl model/scaler files are generated locally
│   │                                   #    by utils/train_pooled_models.py — not committed)
│   ├── templates/                      # index.html (prediction), scenario.html (scenario analysis)
│   ├── static/css/                     # Stylesheet
│   └── utils/
│       └── train_pooled_models.py      # Trains FR+UK pooled anomaly models for the app
│
├── data/                               # Processed datasets only (raw data not committed)
│   ├── processed/                      # UK processed datasets (2004–2024)
│   ├── france/processed/               # France processed (Schauberger yields + E-OBS weather)
│   ├── belgium/, denmark/, germany/,   # Additional European country data (processed)
│   │   ireland/, netherlands/
│   ├── pooled/                         # Pooled multi-country anomaly dataset
│   └── outputs/                        # Model prediction outputs
│
├── eda/                                # Exploratory data analysis scripts
├── plots/                              # Generated figures
└── requirements.txt
```

> **Raw data is not included in the repository.** Only processed datasets are committed. To reconstruct the raw inputs, download them from the sources listed below and place them under `data/raw/` and `data/france/raw/`, then re-run the `src/` pipeline scripts.

## Model Performance

### Pooled Anomaly Models (LOOCV, UK test regions)

| Crop | UK-Only R² | Pooled R² | Best Model |
|------|-----------|-----------|------------|
| Wheat | 0.284 | **0.696** | Ridge |
| Winter Barley | 0.238 | **0.675** | Ridge |
| Spring Barley | 0.221 | **0.764** | Ridge |
| Oats | -0.455 | **0.478** | Ridge |
| OSR | 0.285 | **0.585** | Ridge |

## Setup & Running the Flask Application

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the pooled models (generates the .pkl files the app loads)
cd flask_app
python utils/train_pooled_models.py

# 3. Run the app
python app.py
# Open http://localhost:5000
```

The processed datasets required for training (`data/processed/` and `data/france/processed/`) are included in the repository, so step 2 works out of the box.

## Data Sources

- **UK Yields**: DEFRA crop statistics (2004–2024)
- **UK Weather**: Met Office regional monthly data
- **European Yields**: Eurostat (2004–2018)
- **European Weather**: E-OBS gridded dataset (0.1° resolution)
- **France Historical Yields**: Schauberger et al. (2021) dataset (1900–2018)
