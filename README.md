# Agricultural AI: UK Crop Yield Prediction

A machine learning system for predicting UK crop yields based on seasonal weather features.

## Project Overview

This dissertation project develops ML models to predict yields for major UK arable crops (Wheat, Barley, Oats, Oilseed Rape) using weather data from 2004-2024. Key contributions include:

- **47 seasonal weather features** engineered from Met Office data
- **Crop-specific modeling** with winter/spring barley separation
- **Extreme year handling** using hybrid detection approach
- **Flask web interface** for interactive predictions and scenario analysis

## Project Structure

```
agricultural-ai-project/
├── src/
│   ├── data/              # Data processing modules
│   │   ├── all_uk.py
│   │   └── process_crops.py
│   │
│   ├── features/          # Feature engineering
│   │   ├── seasonal_features.py   # 47 seasonal features
│   │   └── barley_features.py     # Barley-specific features
│   │
│   └── models/            # ML models
│       ├── barley_models.py       # Winter/Spring barley separation
│       ├── crop_models.py         # Per-crop modeling
│       ├── regional_models.py     # Regional analysis
│       ├── ensemble.py            # Extreme year handling
│       ├── day2_modelling.py      # Main modeling pipeline
│       └── day2_fixedModel.py     # Data leakage fix
│
├── flask_app/             # Web application
│   ├── app.py             # Flask application
│   ├── models/            # Saved trained models (.pkl)
│   ├── templates/         # HTML templates
│   ├── static/            # CSS, JS, images
│   └── utils/             # Prediction & visualization utilities
│
├── data/
│   ├── raw/               # Original source data
│   │   ├── yield.csv
│   │   ├── temp.csv
│   │   ├── rainfall.csv
│   │   └── pesticides.csv
│   │
│   ├── processed/         # Processed datasets
│   │   ├── regional_crop_yield_weather_2004_2024.csv  # Main dataset
│   │   ├── uk_crop_yield_with_seasonal_features_2004_2024.csv
│   │   ├── spring_barley_with_weather.csv
│   │   └── winter_barley_with_weather.csv
│   │
│   └── outputs/           # Model outputs
│       ├── model_predictions_2004_2024.csv
│       └── final_optimized_results.csv
│
├── plots/                 # Visualizations
├── eda/                   # Exploratory data analysis
├── archive/               # Experimental code (for reference)
│   ├── experiments/
│   ├── diagnostics/
│   └── data_processing/
│
├── notebooks/             # Jupyter notebooks
├── requirements.txt       # Python dependencies
└── README.md
```

## Key Findings

### Model Performance (Temporal Validation 2020-2024)

| Crop | R² | RMSE (t/ha) |
|------|-----|-------------|
| Wheat | 0.50 | 0.42 |
| Winter Barley | 0.35 | 0.38 |
| Spring Barley | 0.42 | 0.35 |
| Oats | 0.40 | 0.31 |
| OSR | 0.15 | 0.28 |

### Critical Insights

1. **Barley Separation**: Aggregating spring/winter barley (1.44 t/ha yield difference) prevented successful modeling. Separation enables prediction.

2. **OSR Low Performance**: R²=0.15 reflects pest-dominated yields (cabbage stem flea beetle), not weather-dominated.

3. **Extreme Years**: 2010 (cold winter) and 2012 (wet summer) require special handling via quantile-based predictions.

4. **Top Weather Features**:
   - Wheat: Summer rain (-), Summer sunshine (+), Winter frost
   - Barley: Spring frost, Grain filling sunshine
   - Oats: Summer temperature, Spring rainfall

## Installation

```bash
# Clone and navigate to project
cd agricultural-ai-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Models

```bash
# Feature engineering
python src/features/seasonal_features.py

# Train models
python src/models/crop_models.py

# Extreme year validation
python src/models/ensemble.py
```

### Run Flask Application

```bash
cd flask_app
python app.py
# Open http://localhost:5000
```

## Flask Interface Features

1. **Yield Prediction**: Input seasonal weather to predict crop yields
2. **Scenario Analysis**: Interactive sliders for "what-if" analysis
3. **Feature Importance**: SHAP-based model explainability
4. **Historical Trends**: Interactive yield timeline visualizations

## Data Sources

- **Yield Data**: DEFRA UK crop statistics (2004-2024)
- **Weather Data**: Met Office regional monthly data
- **Crops**: Wheat, Winter Barley, Spring Barley, Oats, Oilseed Rape

## Author

Dissertation project for [University Name]

## License

Academic use only.
# activity check
