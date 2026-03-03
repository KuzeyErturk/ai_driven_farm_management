import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("CROP-SPECIFIC MODELS: Weather Becomes Important!")
print("="*70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

df = pd.read_csv('data/uk_crop_yield_with_seasonal_features_2004_2024.csv')
df = df.drop('Production_tonnes', axis=1)

print(f"\nDataset: {df.shape[0]} observations")
print(f"Crops: {df['Crop'].unique()}")

# ============================================================================
# STEP 2: FEATURE SETS
# ============================================================================

# Baseline: Annual features
baseline_features = [
    'Area_hectares',
    'Annual_Temp_C', 
    'Annual_Rainfall_mm',
    'Annual_Sunshine_hours',
    'Annual_Humidity_Percent',
    'Annual_Frost_days'
]

# Seasonal: Full set
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

print(f"\nFeatures: {len(baseline_features)} baseline, {len(seasonal_features)} seasonal")

# ============================================================================
# STEP 3: BUILD CROP-SPECIFIC MODELS
# ============================================================================

crops = df['Crop'].unique()
results = []

print("\n" + "="*70)
print("BUILDING CROP-SPECIFIC MODELS")
print("="*70)

for crop in crops:
    print(f"\n{'='*70}")
    print(f"CROP: {crop.upper()}")
    print(f"{'='*70}")
    
    # Filter data for this crop
    crop_data = df[df['Crop'] == crop].copy()
    
    print(f"Observations: {len(crop_data)}")
    print(f"Years: {crop_data['Year'].min()} - {crop_data['Year'].max()}")
    print(f"Yield range: {crop_data['Yield_t_per_ha'].min():.1f} - {crop_data['Yield_t_per_ha'].max():.1f} t/ha")
    print(f"Yield mean: {crop_data['Yield_t_per_ha'].mean():.1f} ± {crop_data['Yield_t_per_ha'].std():.1f} t/ha")
    
    # Train-test split
    train_mask = crop_data['Year'] <= 2019
    test_mask = crop_data['Year'] >= 2020
    
    n_train = train_mask.sum()
    n_test = test_mask.sum()
    
    print(f"Train/Test split: {n_train}/{n_test} observations")
    
    # Prepare target
    y_train = crop_data.loc[train_mask, 'Yield_t_per_ha']
    y_test = crop_data.loc[test_mask, 'Yield_t_per_ha']
    
    # --------------------------------------------------------------------
    # BASELINE MODEL (Annual Features)
    # --------------------------------------------------------------------
    
    X_baseline_train = crop_data.loc[train_mask, baseline_features]
    X_baseline_test = crop_data.loc[test_mask, baseline_features]
    
    # Scale
    scaler_baseline = StandardScaler()
    X_baseline_train_scaled = scaler_baseline.fit_transform(X_baseline_train)
    X_baseline_test_scaled = scaler_baseline.transform(X_baseline_test)
    
    # Ridge model
    ridge_baseline = Ridge(alpha=1.0)
    ridge_baseline.fit(X_baseline_train_scaled, y_train)
    
    y_pred_test_baseline = ridge_baseline.predict(X_baseline_test_scaled)
    
    baseline_r2 = r2_score(y_test, y_pred_test_baseline)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test_baseline))
    
    print(f"\n📊 BASELINE (Annual Features):")
    print(f"   Test R²: {baseline_r2:.3f}")
    print(f"   Test RMSE: {baseline_rmse:.2f} t/ha")
    
    # Cross-validation score (for small sample)
    if n_train >= 5:
        cv_scores = cross_val_score(ridge_baseline, X_baseline_train_scaled, y_train, 
                                    cv=min(5, n_train), scoring='r2')
        print(f"   CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # --------------------------------------------------------------------
    # SEASONAL MODEL (Seasonal Features)
    # --------------------------------------------------------------------
    
    X_seasonal_train = crop_data.loc[train_mask, seasonal_features]
    X_seasonal_test = crop_data.loc[test_mask, seasonal_features]
    
    # Scale
    scaler_seasonal = StandardScaler()
    X_seasonal_train_scaled = scaler_seasonal.fit_transform(X_seasonal_train)
    X_seasonal_test_scaled = scaler_seasonal.transform(X_seasonal_test)
    
    # Ridge model
    ridge_seasonal = Ridge(alpha=1.0)
    ridge_seasonal.fit(X_seasonal_train_scaled, y_train)
    
    y_pred_test_seasonal = ridge_seasonal.predict(X_seasonal_test_scaled)
    
    seasonal_r2 = r2_score(y_test, y_pred_test_seasonal)
    seasonal_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test_seasonal))
    
    print(f"\n🔥 SEASONAL (Seasonal Features):")
    print(f"   Test R²: {seasonal_r2:.3f}")
    print(f"   Test RMSE: {seasonal_rmse:.2f} t/ha")
    
    # Cross-validation score
    if n_train >= 5:
        cv_scores_seasonal = cross_val_score(ridge_seasonal, X_seasonal_train_scaled, y_train,
                                             cv=min(5, n_train), scoring='r2')
        print(f"   CV R²: {cv_scores_seasonal.mean():.3f} ± {cv_scores_seasonal.std():.3f}")
    
    # --------------------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------------------
    
    improvement_r2 = seasonal_r2 - baseline_r2
    improvement_rmse = baseline_rmse - seasonal_rmse
    improvement_pct = (improvement_r2 / max(baseline_r2, 0.01)) * 100
    
    print(f"\n💡 IMPROVEMENT:")
    print(f"   ΔR²: {improvement_r2:+.3f} ({improvement_pct:+.1f}%)")
    print(f"   ΔRMSE: {improvement_rmse:+.2f} t/ha")
    
    if seasonal_r2 > baseline_r2 + 0.05:
        print(f"   ✅ SIGNIFICANT IMPROVEMENT - Seasonal features help!")
    elif seasonal_r2 > baseline_r2:
        print(f"   ✓ Modest improvement")
    else:
        print(f"   ⚠️ No improvement (small sample or stable yields)")
    
    # Top features for this crop (from seasonal model)
    feature_importance = np.abs(ridge_seasonal.coef_)
    feature_names = seasonal_features
    top_indices = np.argsort(feature_importance)[-5:][::-1]
    
    print(f"\n🌟 Top 5 Features for {crop}:")
    for i, idx in enumerate(top_indices, 1):
        print(f"   {i}. {feature_names[idx]}: {ridge_seasonal.coef_[idx]:+.3f}")
    
    # Store results
    results.append({
        'Crop': crop,
        'N_obs': len(crop_data),
        'N_train': n_train,
        'N_test': n_test,
        'Yield_Mean': crop_data['Yield_t_per_ha'].mean(),
        'Yield_Std': crop_data['Yield_t_per_ha'].std(),
        'Baseline_R2': baseline_r2,
        'Baseline_RMSE': baseline_rmse,
        'Seasonal_R2': seasonal_r2,
        'Seasonal_RMSE': seasonal_rmse,
        'Improvement_R2': improvement_r2,
        'Improvement_RMSE': improvement_rmse
    })

# ============================================================================
# STEP 4: SUMMARY TABLE
# ============================================================================

print("\n" + "="*70)
print("📊 CROP-SPECIFIC MODEL SUMMARY")
print("="*70)

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

# ============================================================================
# STEP 5: VISUALIZATION
# ============================================================================

print("\n" + "="*70)
print("📈 CREATING VISUALIZATIONS")
print("="*70)

# Plot 1: R² Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# R² comparison
ax1 = axes[0]
x = np.arange(len(results_df))
width = 0.35

ax1.bar(x - width/2, results_df['Baseline_R2'], width, label='Baseline (Annual)', 
        color='lightcoral', edgecolor='black')
ax1.bar(x + width/2, results_df['Seasonal_R2'], width, label='Seasonal', 
        color='lightgreen', edgecolor='black')

ax1.set_xlabel('Crop', fontsize=12)
ax1.set_ylabel('Test R²', fontsize=12)
ax1.set_title('Crop-Specific Model Performance', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(results_df['Crop'], rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax1.axhline(y=0, color='black', linewidth=0.8)

# RMSE comparison
ax2 = axes[1]
ax2.bar(x - width/2, results_df['Baseline_RMSE'], width, label='Baseline (Annual)', 
        color='lightcoral', edgecolor='black')
ax2.bar(x + width/2, results_df['Seasonal_RMSE'], width, label='Seasonal', 
        color='lightgreen', edgecolor='black')

ax2.set_xlabel('Crop', fontsize=12)
ax2.set_ylabel('Test RMSE (t/ha)', fontsize=12)
ax2.set_title('Prediction Error by Crop', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(results_df['Crop'], rotation=45, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('plots/crop_specific_model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: plots/crop_specific_model_comparison.png")
plt.show()

# Plot 2: Improvement by Crop
fig, ax = plt.subplots(figsize=(10, 6))

colors = ['green' if x > 0 else 'red' for x in results_df['Improvement_R2']]
ax.barh(results_df['Crop'], results_df['Improvement_R2'], color=colors, 
        edgecolor='black', alpha=0.7)

ax.set_xlabel('R² Improvement (Seasonal - Baseline)', fontsize=12)
ax.set_ylabel('Crop', fontsize=12)
ax.set_title('Seasonal Features Improvement by Crop', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=1)
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (crop, val) in enumerate(zip(results_df['Crop'], results_df['Improvement_R2'])):
    ax.text(val, i, f' {val:+.3f}', va='center', 
            ha='left' if val > 0 else 'right', fontsize=10)

plt.tight_layout()
plt.savefig('plots/seasonal_improvement_by_crop.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots/seasonal_improvement_by_crop.png")
plt.show()

# ============================================================================
# STEP 6: KEY INSIGHTS
# ============================================================================

print("\n" + "="*70)
print("🔍 KEY INSIGHTS")
print("="*70)

avg_baseline = results_df['Baseline_R2'].mean()
avg_seasonal = results_df['Seasonal_R2'].mean()
avg_improvement = results_df['Improvement_R2'].mean()

print(f"\nAverage Performance Across Crops:")
print(f"  Baseline R²: {avg_baseline:.3f}")
print(f"  Seasonal R²: {avg_seasonal:.3f}")
print(f"  Average Improvement: {avg_improvement:+.3f}")

best_crop = results_df.loc[results_df['Seasonal_R2'].idxmax(), 'Crop']
best_r2 = results_df['Seasonal_R2'].max()

print(f"\nBest Predicted Crop: {best_crop} (R² = {best_r2:.3f})")

most_improved = results_df.loc[results_df['Improvement_R2'].idxmax(), 'Crop']
most_improved_val = results_df['Improvement_R2'].max()

print(f"Most Improved by Seasonal Features: {most_improved} (+{most_improved_val:.3f})")

print("\n💡 INTERPRETATION:")
if avg_seasonal > avg_baseline:
    print(f"  ✅ Seasonal features improve predictions across crops!")
    print(f"  → Without crop type dummy, weather becomes MORE important")
    print(f"  → Average R² = {avg_seasonal:.3f} (weather-driven prediction)")
else:
    print(f"  ⚠️ Mixed results - some crops benefit, others don't")
    print(f"  → UK yields may be too stable for strong weather prediction")

print("\n" + "="*70)
print("🎉 CROP-SPECIFIC ANALYSIS COMPLETE!")
print("="*70)

print("\nFiles saved:")
print("  • plots/crop_specific_model_comparison.png")
print("  • plots/seasonal_improvement_by_crop.png")

print("\nNext: Interpret results and write up findings!")