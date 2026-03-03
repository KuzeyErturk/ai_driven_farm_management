# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
# Activate venv first (from project root)
source ../venv/bin/activate

# Run Flask dev server (from flask_app/)
python app.py
# Serves on http://localhost:5000
```

## Retraining Models

```bash
python utils/train_models.py
```

This trains RandomForest models for each crop and saves `.pkl` files to `models/`. Models use 8 weather features defined in `WEATHER_RANGES` in `app.py`. The scaler and model are saved as separate `.pkl` files per crop.

## Architecture

This is a demo Flask web app for UK crop yield prediction, separate from the dissertation's research models in `../src/models/`.

**Request flow:** Browser → `app.py` routes → loads `.pkl` model from `models/` → returns JSON prediction. If no model file exists, `calculate_fallback_prediction()` provides rule-based estimates.

**Key modules in `utils/`:**
- `predictor.py` — `CropPredictor` class with model loading, scaling, and prediction (not currently wired into `app.py`; `app.py` has its own inline prediction logic)
- `risk_analyzer.py` — `RiskAnalyzer` class used by `/api/risk` endpoint; calculates drought/flood/frost/heat/disease risk scores, climate projections, and financial impact
- `visualizer.py` — matplotlib chart generators that return base64-encoded PNGs or save to file
- `train_models.py` — standalone training script (not imported, run directly)

**Pages:** `index.html` (prediction form), `scenario.html` (interactive sliders), `risk.html` (risk dashboard)

## Data

- Primary dataset: `../data/processed/regional_crop_yield_weather_2004_2024.csv`
- 5 crops: Wheat, Winter_Barley, Spring_Barley, Oats, OSR
- 4 regions: England, Wales, Scotland, Northern Ireland
- 336 total observations (84 per crop, 2004–2024)

## Important Caveats

- `app.py` duplicates prediction logic that also exists in `utils/predictor.py` — the `CropPredictor` class is not used by the routes
- `app.py` loads models without using the saved scaler (loads raw `.pkl` model and passes unscaled features), while `predictor.py` correctly applies the scaler — predictions from `app.py` may be inaccurate
- Flask app models are simplified demos; the real dissertation models with per-crop feature selection live in `../src/models/`
- The `models_backup_20260226_153324/` directory is a backup of original models before retraining
