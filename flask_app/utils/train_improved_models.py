"""
Train Improved Crop-Specific Models for Flask Application
==========================================================
Uses the best algorithm per crop identified in the dissertation research:
- Wheat: RandomForest(max_depth=8)
- Winter_Barley: RandomForest(max_depth=5)
- Spring_Barley: Ridge(alpha=10)
- Oats: RandomForest(max_depth=3)
- OSR: SVR(kernel=linear)

Each crop uses its own feature set from the dissertation's Table 9.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneOut, cross_val_score
import joblib
import os
import json

# --- Configuration (mirrors baseline_model_config.py) ---

CROP_FEATURES = {
    'Wheat': ['Area_hectares', 'Winter_Sun', 'Summer_Sun', 'Extreme_Summer_Rain', 'Grain_Filling_Rain'],
    'Winter_Barley': ['Area_hectares', 'Spring_Tmax', 'Winter_Sun', 'Planting_Rain', 'Winter_Frost'],
    'Spring_Barley': ['Area_hectares', 'Spring_Rain', 'Spring_Tmax', 'Summer_Sun'],
    'Oats': ['Area_hectares', 'Spring_Rain', 'Rain_Deviation_from_Optimal', 'Flowering_Sun'],
    'OSR': ['Area_hectares', 'Summer_Tmin', 'Planting_Rain', 'Winter_Frost'],
}

CROP_MODELS = {
    'Wheat': ('RandomForest', {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 3, 'random_state': 42}),
    'Winter_Barley': ('RandomForest', {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 3, 'random_state': 42}),
    'Spring_Barley': ('Ridge', {'alpha': 10}),
    'Oats': ('RandomForest', {'n_estimators': 100, 'max_depth': 3, 'min_samples_split': 3, 'random_state': 42}),
    'OSR': ('SVR', {'kernel': 'linear'}),
}

DATA_SOURCES = {
    'Wheat': 'data/processed/regional_crop_yield_weather_2004_2024.csv',
    'Winter_Barley': 'data/processed/winter_barley_with_weather.csv',
    'Spring_Barley': 'data/processed/spring_barley_with_weather.csv',
    'Oats': 'data/processed/regional_crop_yield_weather_2004_2024.csv',
    'OSR': 'data/processed/regional_crop_yield_weather_2004_2024.csv',
}

# Crop name as it appears in the regional dataset
CROP_NAME_IN_DATA = {
    'Wheat': 'Wheat',
    'Oats': 'Oats',
    'OSR': 'Oilseed_Rape',
}

# Feature metadata for the frontend
FEATURE_METADATA = {
    'Area_hectares': {'unit': 'ha', 'label': 'Crop Area', 'input_type': 'number'},
    'Winter_Sun': {'unit': 'hours', 'label': 'Winter Sunshine', 'input_type': 'range'},
    'Summer_Sun': {'unit': 'hours', 'label': 'Summer Sunshine', 'input_type': 'range'},
    'Extreme_Summer_Rain': {'unit': '', 'label': 'Extreme Summer Rain Event', 'input_type': 'toggle'},
    'Grain_Filling_Rain': {'unit': 'mm', 'label': 'Grain Filling Rain', 'input_type': 'range'},
    'Spring_Tmax': {'unit': '°C', 'label': 'Spring Max Temperature', 'input_type': 'range'},
    'Planting_Rain': {'unit': 'mm', 'label': 'Planting Rain', 'input_type': 'range'},
    'Winter_Frost': {'unit': 'days', 'label': 'Winter Frost Days', 'input_type': 'range'},
    'Spring_Rain': {'unit': 'mm', 'label': 'Spring Rainfall', 'input_type': 'range'},
    'Rain_Deviation_from_Optimal': {'unit': 'mm', 'label': 'Rain Deviation from Optimal', 'input_type': 'range'},
    'Flowering_Sun': {'unit': 'hours', 'label': 'Flowering Sunshine', 'input_type': 'range'},
    'Summer_Tmin': {'unit': '°C', 'label': 'Summer Min Temperature', 'input_type': 'range'},
}


def build_model(model_type, params):
    """Create a model instance from type and params."""
    if model_type == 'RandomForest':
        return RandomForestRegressor(**params)
    elif model_type == 'Ridge':
        return Ridge(**params)
    elif model_type == 'SVR':
        return SVR(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_crop_data(crop, project_root):
    """Load the correct dataset for a crop and filter to the right rows."""
    data_path = os.path.join(project_root, DATA_SOURCES[crop])
    df = pd.read_csv(data_path)

    # For barley-specific CSVs, all rows are already the correct crop
    if crop in ('Winter_Barley', 'Spring_Barley'):
        return df

    # For the regional dataset, filter by crop name
    crop_name = CROP_NAME_IN_DATA.get(crop, crop)
    return df[df['Crop'] == crop_name].copy()


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 70)
    print("TRAINING IMPROVED CROP-SPECIFIC MODELS")
    print("=" * 70)

    model_config = {}
    results = []

    for crop, features in CROP_FEATURES.items():
        print(f"\n{'='*50}")
        print(f"Training: {crop.upper()}")
        model_type, model_params = CROP_MODELS[crop]
        print(f"  Algorithm: {model_type}({model_params})")
        print(f"  Features: {features}")
        print(f"{'='*50}")

        # Load data
        crop_data = load_crop_data(crop, project_root)
        print(f"  Data source: {DATA_SOURCES[crop]}")
        print(f"  Observations: {len(crop_data)}")

        # Verify all features exist
        missing = [f for f in features if f not in crop_data.columns]
        if missing:
            print(f"  ERROR: Missing features: {missing}")
            continue

        X = crop_data[features].copy()
        y = crop_data['Yield_t_per_ha'].copy()

        # Drop any rows with NaN
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]
        print(f"  Valid observations: {len(y)}")
        print(f"  Yield range: {y.min():.2f} - {y.max():.2f} t/ha")

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train model on all data
        model = build_model(model_type, model_params)
        model.fit(X_scaled, y)

        # Evaluate: training R² and LOOCV R²
        train_pred = model.predict(X_scaled)
        train_r2 = r2_score(y, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y, train_pred))

        # LOOCV - collect all out-of-fold predictions, then compute R²
        loo = LeaveOneOut()
        oof_predictions = np.zeros(len(y))
        y_arr = y.values
        for train_idx, test_idx in loo.split(X_scaled):
            cv_model = build_model(model_type, model_params)
            cv_model.fit(X_scaled[train_idx], y_arr[train_idx])
            oof_predictions[test_idx] = cv_model.predict(X_scaled[test_idx])
        loocv_r2 = r2_score(y_arr, oof_predictions)

        print(f"\n  Training R²:  {train_r2:.3f}")
        print(f"  LOOCV R²:     {loocv_r2:.3f}")
        print(f"  Training RMSE: {train_rmse:.3f} t/ha")

        # Save model and scaler
        model_name = crop.lower()
        joblib.dump(model, os.path.join(models_dir, f'{model_name}_model.pkl'))
        joblib.dump(scaler, os.path.join(models_dir, f'{model_name}_scaler.pkl'))
        print(f"  Saved: {model_name}_model.pkl, {model_name}_scaler.pkl")

        # Build feature config with ranges from data
        feature_config = {}
        for feat in features:
            col = X[feat] if feat in X.columns else crop_data[feat]
            meta = FEATURE_METADATA.get(feat, {'unit': '', 'label': feat.replace('_', ' '), 'input_type': 'range'})

            if meta['input_type'] == 'toggle':
                feat_min, feat_max, feat_default = 0, 1, 0
            else:
                feat_min = float(np.floor(col.min()))
                feat_max = float(np.ceil(col.max()))
                feat_default = float(round(col.median()))

            feature_config[feat] = {
                'min': feat_min,
                'max': feat_max,
                'default': feat_default,
                'unit': meta['unit'],
                'label': meta['label'],
                'input_type': meta['input_type'],
            }

        model_config[crop] = {
            'model_type': model_type,
            'features': feature_config,
            'loocv_r2': round(loocv_r2, 3),
            'train_r2': round(train_r2, 3),
        }

        results.append({
            'Crop': crop,
            'Model': model_type,
            'N': len(y),
            'Train_R2': train_r2,
            'LOOCV_R2': loocv_r2,
            'RMSE': train_rmse,
        })

    # Save model_config.json (replace NaN with null for valid JSON)
    config_path = os.path.join(models_dir, 'model_config.json')
    config_json = json.dumps(model_config, indent=2)
    config_json = config_json.replace('NaN', 'null')
    with open(config_path, 'w') as f:
        f.write(config_json)
    print(f"\nSaved model config: {config_path}")

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"\n{'Crop':<18} {'Model':<16} {'N':>5} {'Train R²':>10} {'LOOCV R²':>10} {'RMSE':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['Crop']:<18} {r['Model']:<16} {r['N']:>5} {r['Train_R2']:>10.3f} {r['LOOCV_R2']:>10.3f} {r['RMSE']:>8.3f}")

    print(f"\nAll models saved to: {os.path.abspath(models_dir)}")
    print("=" * 70)


if __name__ == '__main__':
    main()
