import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

print("="*70)
print("REGIONAL CROP-SPECIFIC MODELS")
print("="*70)

# ============================================================================
# LOAD REGIONAL DATA
# ============================================================================

df = pd.read_csv('data/regional_crop_yield_weather_2004_2024.csv')

print(f"\nDataset: {len(df)} observations")
print(f"Crops: {df['Crop'].unique()}")
print(f"Regions: {df['Region'].unique()}")
print(f"Years: {df['Year'].min()} - {df['Year'].max()}")

# ============================================================================
# FEATURE SETS
# ============================================================================

# Top features per crop (based on earlier correlation analysis)
crop_features = {
    'Wheat': ['Area_hectares', 'Winter_Sun', 'Summer_Sun', 'Extreme_Summer_Rain', 'Grain_Filling_Rain'],
    'Barley': ['Area_hectares', 'Winter_Sun', 'Winter_Tmax', 'Extreme_Summer_Rain', 'Winter_Frost'],
    'Oats': ['Area_hectares', 'Spring_Rain', 'Rain_Deviation_from_Optimal', 'Flowering_Sun'],
    'Oilseed_Rape': ['Area_hectares', 'Summer_Tmin', 'Planting_Rain', 'Winter_Frost']
}

# Baseline features
baseline_features = ['Area_hectares']  # Just area as simplest baseline

# ============================================================================
# BUILD MODELS FOR EACH CROP
# ============================================================================

results = []

for crop in df['Crop'].unique():
    print(f"\n{'='*70}")
    print(f"CROP: {crop.upper()}")
    print(f"{'='*70}")
    
    # Filter data
    crop_data = df[df['Crop'] == crop].copy()
    
    print(f"Total observations: {len(crop_data)}")
    print(f"Regions: {crop_data['Region'].nunique()}")
    print(f"Years: {crop_data['Year'].min()} - {crop_data['Year'].max()}")
    print(f"Yield: {crop_data['Yield_t_per_ha'].mean():.1f} ± {crop_data['Yield_t_per_ha'].std():.1f} t/ha")
    
    # Train-test split (temporal)
    train_mask = crop_data['Year'] <= 2019
    test_mask = crop_data['Year'] >= 2020
    
    n_train = train_mask.sum()
    n_test = test_mask.sum()
    
    print(f"Train/Test: {n_train}/{n_test} observations")
    
    y_train = crop_data.loc[train_mask, 'Yield_t_per_ha']
    y_test = crop_data.loc[test_mask, 'Yield_t_per_ha']
    
    # ----------------------------------------------------------------
    # BASELINE: Area only
    # ----------------------------------------------------------------
    
    X_base_train = crop_data.loc[train_mask, baseline_features]
    X_base_test = crop_data.loc[test_mask, baseline_features]
    
    scaler_base = StandardScaler()
    X_base_train_sc = scaler_base.fit_transform(X_base_train)
    X_base_test_sc = scaler_base.transform(X_base_test)
    
    ridge_base = Ridge(alpha=1.0)
    ridge_base.fit(X_base_train_sc, y_train)
    
    y_pred_base = ridge_base.predict(X_base_test_sc)
    base_r2 = r2_score(y_test, y_pred_base)
    base_rmse = np.sqrt(mean_squared_error(y_test, y_pred_base))
    
    print(f"\n📊 BASELINE (Area only):")
    print(f"   Test R²: {base_r2:.3f}")
    print(f"   Test RMSE: {base_rmse:.2f} t/ha")
    
    # ----------------------------------------------------------------
    # SEASONAL: Top weather features
    # ----------------------------------------------------------------
    
    seasonal_feats = crop_features[crop]
    
    X_seas_train = crop_data.loc[train_mask, seasonal_feats]
    X_seas_test = crop_data.loc[test_mask, seasonal_feats]
    
    scaler_seas = StandardScaler()
    X_seas_train_sc = scaler_seas.fit_transform(X_seas_train)
    X_seas_test_sc = scaler_seas.transform(X_seas_test)
    
    ridge_seas = Ridge(alpha=1.0)
    ridge_seas.fit(X_seas_train_sc, y_train)
    
    y_pred_seas = ridge_seas.predict(X_seas_test_sc)
    seas_r2 = r2_score(y_test, y_pred_seas)
    seas_rmse = np.sqrt(mean_squared_error(y_test, y_pred_seas))
    
    print(f"\n🔥 SEASONAL ({len(seasonal_feats)} features):")
    print(f"   Features: {', '.join(seasonal_feats)}")
    print(f"   Test R²: {seas_r2:.3f}")
    print(f"   Test RMSE: {seas_rmse:.2f} t/ha")
    
    # ----------------------------------------------------------------
    # COMPARISON
    # ----------------------------------------------------------------
    
    improvement_r2 = seas_r2 - base_r2
    improvement_rmse = base_rmse - seas_rmse
    
    print(f"\n💡 IMPROVEMENT:")
    print(f"   ΔR²: {improvement_r2:+.3f}")
    print(f"   ΔRMSE: {improvement_rmse:+.2f} t/ha")
    
    if seas_r2 > 0.3:
        status = "✅ GOOD prediction"
    elif seas_r2 > 0:
        status = "✓ Positive prediction"
    else:
        status = "⚠️ Still negative (needs more work)"
    
    print(f"   {status}")
    
    # Feature importance
    print(f"\n🌟 Feature Coefficients:")
    for feat, coef in zip(seasonal_feats, ridge_seas.coef_):
        print(f"   • {feat}: {coef:+.3f}")
    
    # Store results
    results.append({
        'Crop': crop,
        'N_obs': len(crop_data),
        'N_train': n_train,
        'N_test': n_test,
        'Baseline_R2': base_r2,
        'Seasonal_R2': seas_r2,
        'Improvement_R2': improvement_r2,
        'Baseline_RMSE': base_rmse,
        'Seasonal_RMSE': seas_rmse
    })

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("📊 SUMMARY: REGIONAL CROP-SPECIFIC MODELS")
print("="*70)

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

print("\n💡 KEY FINDINGS:")
print(f"\nAverage Baseline R²: {results_df['Baseline_R2'].mean():.3f}")
print(f"Average Seasonal R²: {results_df['Seasonal_R2'].mean():.3f}")
print(f"Average Improvement: {results_df['Improvement_R2'].mean():+.3f}")

positive_r2 = (results_df['Seasonal_R2'] > 0).sum()
print(f"\nCrops with positive R²: {positive_r2}/4")

if positive_r2 > 0:
    print("\nSuccessfully predicted crops:")
    for _, row in results_df[results_df['Seasonal_R2'] > 0].iterrows():
        print(f"  • {row['Crop']}: R² = {row['Seasonal_R2']:.3f}")

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(results_df))
width = 0.35

ax.bar(x - width/2, results_df['Baseline_R2'], width, label='Baseline (Area only)',
       color='lightcoral', edgecolor='black')
ax.bar(x + width/2, results_df['Seasonal_R2'], width, label='Seasonal Weather',
       color='lightgreen', edgecolor='black')

ax.set_xlabel('Crop', fontsize=12)
ax.set_ylabel('Test R²', fontsize=12)
ax.set_title('Regional Crop-Specific Models\n(84 obs per crop)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(results_df['Crop'], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='black', linewidth=1)

plt.tight_layout()
plt.savefig('plots/regional_crop_models.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: plots/regional_crop_models.png")
plt.show()

print("\n" + "="*70)
print("🎉 REGIONAL MODELING COMPLETE!")
print("="*70)

print("\n✨ WITH 84 OBSERVATIONS PER CROP:")
print("  • 4x more data than national level")
print("  • Better statistical power")
print("  • Regional weather variation captured")
print("\nExpected: Positive R² values now!")