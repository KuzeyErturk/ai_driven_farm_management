"""
Train and Save Models for Flask Application
============================================
Trains Random Forest models for each crop and saves them as pickle files.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 70)
print("TRAINING MODELS FOR FLASK APPLICATION")
print("=" * 70)

# Load data
data_path = os.path.join(os.path.dirname(__file__), '../../data/processed/regional_crop_yield_weather_2004_2024.csv')
df = pd.read_csv(data_path)

# Features to use
FEATURES = [
    'Summer_Rain', 'Summer_Sun', 'Winter_Frost', 'Spring_Rain',
    'Spring_Frost', 'Summer_Tmax', 'Grain_Filling_Rain', 'Grain_Filling_Sun'
]

# Model output directory
models_dir = os.path.join(os.path.dirname(__file__), '../models')
os.makedirs(models_dir, exist_ok=True)

# Train models for each crop
crops = df['Crop'].unique()
results = []

for crop in crops:
    print(f"\n{'='*50}")
    print(f"Training: {crop.upper()}")
    print(f"{'='*50}")

    crop_data = df[df['Crop'] == crop].copy()

    # Check available features
    available_features = [f for f in FEATURES if f in crop_data.columns]

    if len(available_features) < 4:
        print(f"  Warning: Only {len(available_features)} features available, skipping...")
        continue

    # Use all data for training (Flask app will use for predictions)
    X = crop_data[available_features]
    y = crop_data['Yield_t_per_ha']

    print(f"  Data: {len(y)} observations")
    print(f"  Features: {len(available_features)}")
    print(f"  Yield range: {y.min():.2f} - {y.max():.2f} t/ha")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Random Forest
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_split=5,
        random_state=42
    )
    rf.fit(X_scaled, y)

    # Cross-validation score estimate
    train_pred = rf.predict(X_scaled)
    train_r2 = r2_score(y, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y, train_pred))

    print(f"\n  Training Performance:")
    print(f"  R² (train): {train_r2:.3f}")
    print(f"  RMSE (train): {train_rmse:.3f} t/ha")

    # Feature importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:5]
    print(f"\n  Top Features:")
    for i, idx in enumerate(indices, 1):
        print(f"    {i}. {available_features[idx]}: {importances[idx]:.3f}")

    # Save model and scaler
    model_name = crop.lower().replace(' ', '_')
    model_path = os.path.join(models_dir, f'{model_name}_model.pkl')
    scaler_path = os.path.join(models_dir, f'{model_name}_scaler.pkl')

    joblib.dump(rf, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\n  Saved: {model_path}")

    results.append({
        'Crop': crop,
        'N_Observations': len(y),
        'N_Features': len(available_features),
        'Train_R2': train_r2,
        'Train_RMSE': train_rmse,
        'Model_Path': model_path
    })

# Save feature list for reference
feature_info = {
    'features': FEATURES,
    'feature_ranges': {
        'Summer_Rain': {'min': 100, 'max': 400, 'unit': 'mm'},
        'Summer_Sun': {'min': 400, 'max': 700, 'unit': 'hours'},
        'Winter_Frost': {'min': 10, 'max': 50, 'unit': 'days'},
        'Spring_Rain': {'min': 100, 'max': 300, 'unit': 'mm'},
        'Spring_Frost': {'min': 0, 'max': 20, 'unit': 'days'},
        'Summer_Tmax': {'min': 18, 'max': 26, 'unit': 'C'},
        'Grain_Filling_Rain': {'min': 50, 'max': 250, 'unit': 'mm'},
        'Grain_Filling_Sun': {'min': 200, 'max': 500, 'unit': 'hours'},
    }
}

import json
with open(os.path.join(models_dir, 'feature_info.json'), 'w') as f:
    json.dump(feature_info, f, indent=2)

print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(results)
print(f"\n{'Crop':<20} {'N_Obs':>8} {'R² (train)':>12} {'RMSE':>8}")
print("-" * 50)
for _, row in results_df.iterrows():
    print(f"{row['Crop']:<20} {row['N_Observations']:>8} {row['Train_R2']:>12.3f} {row['Train_RMSE']:>8.3f}")

print(f"\nModels saved to: {models_dir}")
print("=" * 70)
