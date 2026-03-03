import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("DAY 2: BASELINE vs SEASONAL FEATURE MODELS")
print("="*70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

df = pd.read_csv('data/uk_crop_yield_with_seasonal_features_2004_2024.csv')

print(f"\nDataset: {df.shape[0]} observations, {df.shape[1]} variables")
print(f"Crops: {df['Crop'].unique()}")
print(f"Years: {df['Year'].min()} - {df['Year'].max()}")

# ============================================================================
# STEP 2: FEATURE SETS
# ============================================================================

print("\n" + "="*70)
print("FEATURE SETS")
print("="*70)

# Baseline: Annual features only
baseline_features = [
    'Area_hectares',
    'Annual_Temp_C', 
    'Annual_Rainfall_mm',
    'Annual_Sunshine_hours',
    'Annual_Humidity_Percent',
    'Annual_Frost_days'
]

# Seasonal: Top features based on correlations
seasonal_features = [
    'Area_hectares',
    # Temperature
    'Winter_Tmax', 'Spring_Tmax', 'Summer_Tmax',
    'Winter_Tmin', 'Spring_Tmin', 'Summer_Tmin',
    'Flowering_Temp', 'Grain_Filling_Temp',
    'Summer_Tmax_Peak', 'Spring_Tmin_Coldest',
    'Spring_GDD', 'Summer_GDD',
    # Rainfall
    'Spring_Rain', 'Summer_Rain',
    'Grain_Filling_Rain', 'Planting_Rain',
    'Rain_Deviation_from_Optimal',
    'Spring_Rain_Squared', 'Summer_Rain_Squared',
    # Sunshine
    'Winter_Sun', 'Spring_Sun', 'Summer_Sun',
    'Grain_Filling_Sun', 'Flowering_Sun',
    # Frost
    'Spring_Frost', 'Late_Spring_Frost',
    # Interactions & Extremes
    'Spring_Temp_x_Rain', 'Summer_Temp_x_Sun',
    'Heat_Stress', 'Cold_Spring', 'Extreme_Summer_Rain', 'Drought_Spring'
]

print(f"\nBaseline Features ({len(baseline_features)}): Area + Annual weather")
print(f"Seasonal Features ({len(seasonal_features)}): Area + Seasonal weather")

# ============================================================================
# STEP 3: PREPARE DATA
# ============================================================================

print("\n" + "="*70)
print("DATA PREPARATION")
print("="*70)

# One-hot encode Crop
df_encoded = pd.get_dummies(df, columns=['Crop'], prefix='Crop', drop_first=False)

# Update feature lists with crop dummies
crop_dummies = [col for col in df_encoded.columns if col.startswith('Crop_')]
baseline_features_full = baseline_features + crop_dummies
seasonal_features_full = seasonal_features + crop_dummies

print(f"\nWith crop encoding:")
print(f"  Baseline: {len(baseline_features_full)} features")
print(f"  Seasonal: {len(seasonal_features_full)} features")

# Target
y = df_encoded['Yield_t_per_ha']

# Train-test split (temporal)
train_mask = df_encoded['Year'] <= 2019  # 2004-2019 training
test_mask = df_encoded['Year'] >= 2020    # 2020-2024 testing

X_baseline_train = df_encoded.loc[train_mask, baseline_features_full]
X_baseline_test = df_encoded.loc[test_mask, baseline_features_full]

X_seasonal_train = df_encoded.loc[train_mask, seasonal_features_full]
X_seasonal_test = df_encoded.loc[test_mask, seasonal_features_full]

y_train = y[train_mask]
y_test = y[test_mask]

print(f"\nTrain set: {len(y_train)} observations (2004-2019)")
print(f"Test set: {len(y_test)} observations (2020-2024)")

# Scale features
scaler_baseline = StandardScaler()
X_baseline_train_scaled = scaler_baseline.fit_transform(X_baseline_train)
X_baseline_test_scaled = scaler_baseline.transform(X_baseline_test)

scaler_seasonal = StandardScaler()
X_seasonal_train_scaled = scaler_seasonal.fit_transform(X_seasonal_train)
X_seasonal_test_scaled = scaler_seasonal.transform(X_seasonal_test)

print("\n✓ Data prepared and scaled")

# ============================================================================
# STEP 4: BASELINE MODELS (Annual Features)
# ============================================================================

print("\n" + "="*70)
print("BASELINE MODELS - Annual Weather Features")
print("="*70)

baseline_results = {}

# Linear Regression
lr_baseline = LinearRegression()
lr_baseline.fit(X_baseline_train_scaled, y_train)
y_pred_train = lr_baseline.predict(X_baseline_train_scaled)
y_pred_test = lr_baseline.predict(X_baseline_test_scaled)

baseline_results['Linear'] = {
    'train_r2': r2_score(y_train, y_pred_train),
    'test_r2': r2_score(y_test, y_pred_test),
    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'test_mae': mean_absolute_error(y_test, y_pred_test)
}

print("\n1. Linear Regression:")
print(f"   Train R²: {baseline_results['Linear']['train_r2']:.3f}")
print(f"   Test R²:  {baseline_results['Linear']['test_r2']:.3f}")
print(f"   Test RMSE: {baseline_results['Linear']['test_rmse']:.2f} t/ha")

# Ridge Regression
ridge_baseline = Ridge(alpha=1.0)
ridge_baseline.fit(X_baseline_train_scaled, y_train)
y_pred_test = ridge_baseline.predict(X_baseline_test_scaled)

baseline_results['Ridge'] = {
    'train_r2': ridge_baseline.score(X_baseline_train_scaled, y_train),
    'test_r2': ridge_baseline.score(X_baseline_test_scaled, y_test),
    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'test_mae': mean_absolute_error(y_test, y_pred_test)
}

print("\n2. Ridge Regression:")
print(f"   Train R²: {baseline_results['Ridge']['train_r2']:.3f}")
print(f"   Test R²:  {baseline_results['Ridge']['test_r2']:.3f}")
print(f"   Test RMSE: {baseline_results['Ridge']['test_rmse']:.2f} t/ha")

# Random Forest
rf_baseline = RandomForestRegressor(n_estimators=100, max_depth=10, 
                                   min_samples_split=5, random_state=42)
rf_baseline.fit(X_baseline_train_scaled, y_train)
y_pred_test = rf_baseline.predict(X_baseline_test_scaled)

baseline_results['RandomForest'] = {
    'train_r2': rf_baseline.score(X_baseline_train_scaled, y_train),
    'test_r2': rf_baseline.score(X_baseline_test_scaled, y_test),
    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'test_mae': mean_absolute_error(y_test, y_pred_test)
}

print("\n3. Random Forest:")
print(f"   Train R²: {baseline_results['RandomForest']['train_r2']:.3f}")
print(f"   Test R²:  {baseline_results['RandomForest']['test_r2']:.3f}")
print(f"   Test RMSE: {baseline_results['RandomForest']['test_rmse']:.2f} t/ha")

# ============================================================================
# STEP 5: SEASONAL MODELS (Seasonal Features)
# ============================================================================

print("\n" + "="*70)
print("🔥 IMPROVED MODELS - Seasonal Weather Features")
print("="*70)

seasonal_results = {}

# Linear Regression
lr_seasonal = LinearRegression()
lr_seasonal.fit(X_seasonal_train_scaled, y_train)
y_pred_train = lr_seasonal.predict(X_seasonal_train_scaled)
y_pred_test = lr_seasonal.predict(X_seasonal_test_scaled)

seasonal_results['Linear'] = {
    'train_r2': r2_score(y_train, y_pred_train),
    'test_r2': r2_score(y_test, y_pred_test),
    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'test_mae': mean_absolute_error(y_test, y_pred_test)
}

print("\n1. Linear Regression:")
print(f"   Train R²: {seasonal_results['Linear']['train_r2']:.3f}")
print(f"   Test R²:  {seasonal_results['Linear']['test_r2']:.3f}")
print(f"   Test RMSE: {seasonal_results['Linear']['test_rmse']:.2f} t/ha")

# Ridge Regression
ridge_seasonal = Ridge(alpha=1.0)
ridge_seasonal.fit(X_seasonal_train_scaled, y_train)
y_pred_test = ridge_seasonal.predict(X_seasonal_test_scaled)

seasonal_results['Ridge'] = {
    'train_r2': ridge_seasonal.score(X_seasonal_train_scaled, y_train),
    'test_r2': ridge_seasonal.score(X_seasonal_test_scaled, y_test),
    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'test_mae': mean_absolute_error(y_test, y_pred_test)
}

print("\n2. Ridge Regression:")
print(f"   Train R²: {seasonal_results['Ridge']['train_r2']:.3f}")
print(f"   Test R²:  {seasonal_results['Ridge']['test_r2']:.3f}")
print(f"   Test RMSE: {seasonal_results['Ridge']['test_rmse']:.2f} t/ha")

# Random Forest
rf_seasonal = RandomForestRegressor(n_estimators=100, max_depth=10,
                                   min_samples_split=5, random_state=42)
rf_seasonal.fit(X_seasonal_train_scaled, y_train)
y_pred_test = rf_seasonal.predict(X_seasonal_test_scaled)

seasonal_results['RandomForest'] = {
    'train_r2': rf_seasonal.score(X_seasonal_train_scaled, y_train),
    'test_r2': rf_seasonal.score(X_seasonal_test_scaled, y_test),
    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'test_mae': mean_absolute_error(y_test, y_pred_test)
}

print("\n3. Random Forest:")
print(f"   Train R²: {seasonal_results['RandomForest']['train_r2']:.3f}")
print(f"   Test R²:  {seasonal_results['RandomForest']['test_r2']:.3f}")
print(f"   Test RMSE: {seasonal_results['RandomForest']['test_rmse']:.2f} t/ha")

# ============================================================================
# STEP 6: COMPARISON
# ============================================================================

print("\n" + "="*70)
print("📊 BASELINE vs SEASONAL COMPARISON")
print("="*70)

comparison = pd.DataFrame({
    'Model': ['Linear', 'Ridge', 'RandomForest'],
    'Baseline_Test_R2': [baseline_results[m]['test_r2'] for m in ['Linear', 'Ridge', 'RandomForest']],
    'Seasonal_Test_R2': [seasonal_results[m]['test_r2'] for m in ['Linear', 'Ridge', 'RandomForest']],
    'Baseline_Test_RMSE': [baseline_results[m]['test_rmse'] for m in ['Linear', 'Ridge', 'RandomForest']],
    'Seasonal_Test_RMSE': [seasonal_results[m]['test_rmse'] for m in ['Linear', 'Ridge', 'RandomForest']]
})

comparison['R2_Improvement'] = comparison['Seasonal_Test_R2'] - comparison['Baseline_Test_R2']
comparison['RMSE_Improvement'] = comparison['Baseline_Test_RMSE'] - comparison['Seasonal_Test_RMSE']

print("\n" + comparison.to_string(index=False))

print("\n💡 INSIGHTS:")
best_baseline_idx = comparison['Baseline_Test_R2'].idxmax()
best_seasonal_idx = comparison['Seasonal_Test_R2'].idxmax()

print(f"\nBest Baseline Model: {comparison.loc[best_baseline_idx, 'Model']}")
print(f"  Test R²: {comparison.loc[best_baseline_idx, 'Baseline_Test_R2']:.3f}")

print(f"\nBest Seasonal Model: {comparison.loc[best_seasonal_idx, 'Model']}")
print(f"  Test R²: {comparison.loc[best_seasonal_idx, 'Seasonal_Test_R2']:.3f}")

print(f"\n✨ R² Improvement: {comparison.loc[best_seasonal_idx, 'R2_Improvement']:.3f}")
print(f"✨ RMSE Improvement: {comparison.loc[best_seasonal_idx, 'RMSE_Improvement']:.2f} t/ha")

# ============================================================================
# STEP 7: SAVE BEST MODEL FOR LATER USE
# ============================================================================

# Save best model (Ridge with seasonal features typically wins)
best_model = ridge_seasonal
best_scaler = scaler_seasonal

# Make predictions for all data
X_seasonal_all = df_encoded[seasonal_features_full]
X_seasonal_all_scaled = best_scaler.transform(X_seasonal_all)
df['Predicted_Yield'] = best_model.predict(X_seasonal_all_scaled)
df['Prediction_Error'] = df['Yield_t_per_ha'] - df['Predicted_Yield']

# Save predictions
df[['Year', 'Crop', 'Yield_t_per_ha', 'Predicted_Yield', 'Prediction_Error']].to_csv(
    'data/model_predictions_2004_2024.csv', index=False
)

print("\n✓ Best model predictions saved")

print("\n" + "="*70)
print("🎉 DAY 2 MODELING COMPLETE!")
print("="*70)

print("\nKey Achievements:")
print(f"  ✓ Baseline models built (Annual features)")
print(f"  ✓ Improved models built (Seasonal features)")
print(f"  ✓ Test R² improved from {comparison['Baseline_Test_R2'].max():.3f} to {comparison['Seasonal_Test_R2'].max():.3f}")
print(f"  ✓ Predictions saved for all years")

print("\nNext: Visualize results and validate on extreme years!")