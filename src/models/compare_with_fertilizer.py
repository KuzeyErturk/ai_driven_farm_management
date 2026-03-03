"""
Compare Models: With vs Without Fertilizer Features
====================================================
Tests if adding fertilizer data improves model performance.

IMPORTANT: This script does NOT modify any existing files or models.
           It creates a comparison using the NEW fertilizer-enhanced datasets.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("MODEL COMPARISON: Weather vs Weather + Fertilizer")
print("=" * 70)
print("\nUsing YOUR exact model configuration:")
print("  - RandomForest(n_estimators=100, max_depth=8, min_samples_split=3)")
print("  - Train: 2004-2019, Test: 2020+")
print("  - Same features as your original models")
print()

# ============================================================================
# MODEL CONFIGURATION (matching your existing models)
# ============================================================================

RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 8,
    'min_samples_split': 3,
    'random_state': 42
}

# Feature sets for each crop (from your barley_models.py)
WINTER_BARLEY_FEATURES = ['Area_hectares', 'Winter_Frost', 'Autumn_Rain',
                          'Spring_Tmax', 'Winter_Sun', 'Planting_Rain']

SPRING_BARLEY_FEATURES = ['Area_hectares', 'Spring_Rain', 'Spring_Frost',
                          'Late_Spring_Frost', 'Summer_Sun', 'Grain_Filling_Sun']

FERTILIZER_FEATURES = ['Nitrogen_kg_ha', 'Phosphorus_kg_ha']


def run_model(data, features, name):
    """Run model with given features and return metrics."""
    train = data[data['Year'] <= 2019]
    test = data[data['Year'] >= 2020]

    if len(test) == 0 or len(train) == 0:
        return None, None, 0, 0

    # Check all features exist
    available_features = [f for f in features if f in data.columns]
    if len(available_features) != len(features):
        missing = set(features) - set(available_features)
        print(f"    Warning: Missing features: {missing}")
        features = available_features

    X_train = train[features].fillna(train[features].mean())
    y_train = train['Yield_t_per_ha']
    X_test = test[features].fillna(test[features].mean())
    y_test = test['Yield_t_per_ha']

    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return r2, rmse, len(train), len(test)


# ============================================================================
# WINTER BARLEY COMPARISON
# ============================================================================

print("=" * 70)
print("WINTER BARLEY")
print("=" * 70)

# Original (without fertilizer)
winter_orig = pd.read_csv('data/processed/winter_barley_with_weather.csv')
r2_orig, rmse_orig, n_train, n_test = run_model(
    winter_orig, WINTER_BARLEY_FEATURES, "Original"
)
print(f"\n  Original (weather only):")
print(f"    Features: {len(WINTER_BARLEY_FEATURES)}")
print(f"    Train/Test: {n_train}/{n_test}")
print(f"    R² = {r2_orig:.3f}, RMSE = {rmse_orig:.2f} t/ha")

# With fertilizer
winter_fert = pd.read_csv('data/processed/winter_barley_weather_fertilizer.csv')
features_with_fert = WINTER_BARLEY_FEATURES + FERTILIZER_FEATURES
r2_fert, rmse_fert, _, _ = run_model(
    winter_fert, features_with_fert, "With Fertilizer"
)
print(f"\n  With Fertilizer:")
print(f"    Features: {len(features_with_fert)} (+2)")
print(f"    R² = {r2_fert:.3f}, RMSE = {rmse_fert:.2f} t/ha")

improvement = r2_fert - r2_orig
print(f"\n  IMPACT: R² change = {improvement:+.3f} {'✓ IMPROVED' if improvement > 0 else '✗ worse'}")


# ============================================================================
# SPRING BARLEY COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("SPRING BARLEY")
print("=" * 70)

# Original
spring_orig = pd.read_csv('data/processed/spring_barley_with_weather.csv')
r2_orig, rmse_orig, n_train, n_test = run_model(
    spring_orig, SPRING_BARLEY_FEATURES, "Original"
)
print(f"\n  Original (weather only):")
print(f"    Features: {len(SPRING_BARLEY_FEATURES)}")
print(f"    Train/Test: {n_train}/{n_test}")
print(f"    R² = {r2_orig:.3f}, RMSE = {rmse_orig:.2f} t/ha")

# With fertilizer
spring_fert = pd.read_csv('data/processed/spring_barley_weather_fertilizer.csv')
features_with_fert = SPRING_BARLEY_FEATURES + FERTILIZER_FEATURES
r2_fert, rmse_fert, _, _ = run_model(
    spring_fert, features_with_fert, "With Fertilizer"
)
print(f"\n  With Fertilizer:")
print(f"    Features: {len(features_with_fert)} (+2)")
print(f"    R² = {r2_fert:.3f}, RMSE = {rmse_fert:.2f} t/ha")

improvement = r2_fert - r2_orig
print(f"\n  IMPACT: R² change = {improvement:+.3f} {'✓ IMPROVED' if improvement > 0 else '✗ worse'}")


# ============================================================================
# OTHER CROPS (using seasonal features)
# ============================================================================

# General seasonal features for wheat, oats, OSR
SEASONAL_FEATURES = [
    'Area_hectares',
    'Winter_Tmax', 'Spring_Tmax', 'Summer_Tmax',
    'Winter_Tmin', 'Spring_Tmin', 'Summer_Tmin',
    'Flowering_Temp', 'Grain_Filling_Temp',
    'Spring_GDD', 'Summer_GDD',
    'Spring_Rain', 'Summer_Rain',
    'Grain_Filling_Rain', 'Planting_Rain',
    'Spring_Rain_Squared', 'Summer_Rain_Squared',
    'Winter_Sun', 'Spring_Sun', 'Summer_Sun',
    'Grain_Filling_Sun', 'Flowering_Sun',
    'Spring_Frost', 'Late_Spring_Frost',
    'Spring_Temp_x_Rain', 'Summer_Temp_x_Sun',
    'Heat_Stress', 'Cold_Spring', 'Extreme_Summer_Rain', 'Drought_Spring'
]

for crop_name in ['Wheat', 'Oats', 'Oilseed_Rape']:
    print("\n" + "=" * 70)
    print(crop_name.upper())
    print("=" * 70)

    # Original
    df_orig = pd.read_csv('data/processed/uk_crop_yield_with_seasonal_features_2004_2024.csv')
    crop_orig = df_orig[df_orig['Crop'] == crop_name].copy()

    available = [f for f in SEASONAL_FEATURES if f in crop_orig.columns]
    r2_orig, rmse_orig, n_train, n_test = run_model(crop_orig, available, "Original")

    if r2_orig is not None:
        print(f"\n  Original (weather only):")
        print(f"    Features: {len(available)}")
        print(f"    Train/Test: {n_train}/{n_test}")
        print(f"    R² = {r2_orig:.3f}, RMSE = {rmse_orig:.2f} t/ha")

        # With fertilizer
        df_fert = pd.read_csv('data/processed/uk_crop_yield_seasonal_fertilizer_2004_2024.csv')
        crop_fert = df_fert[df_fert['Crop'] == crop_name].copy()

        features_with_fert = available + FERTILIZER_FEATURES
        r2_fert, rmse_fert, _, _ = run_model(crop_fert, features_with_fert, "With Fertilizer")

        print(f"\n  With Fertilizer:")
        print(f"    Features: {len(features_with_fert)} (+2)")
        print(f"    R² = {r2_fert:.3f}, RMSE = {rmse_fert:.2f} t/ha")

        improvement = r2_fert - r2_orig
        print(f"\n  IMPACT: R² change = {improvement:+.3f} {'✓ IMPROVED' if improvement > 0 else '✗ worse' if improvement < -0.01 else '≈ similar'}")
    else:
        print(f"  Insufficient data for {crop_name}")


# ============================================================================
# SUMMARY TABLE
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY: FERTILIZER IMPACT ON MODEL PERFORMANCE")
print("=" * 70)

print("""
NOTES:
  1. The baseline R² values may differ slightly from your Table 11
     due to random seed variations. This is normal.
  2. Focus on the RELATIVE change (improvement) from adding fertilizer.
  3. Original datasets and models are UNCHANGED.

INTERPRETATION:
  - Positive change: Fertilizer data helps the model
  - Near-zero change: Fertilizer data adds no value
  - Negative change: Fertilizer data hurts (likely overfitting)

RECOMMENDATION:
  If fertilizer improves R² by >0.05, consider integrating it into
  your final model for your dissertation.
""")

print("=" * 70)
print("✅ Comparison complete. Original models unchanged.")
print("=" * 70)
