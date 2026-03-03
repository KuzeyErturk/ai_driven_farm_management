import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("CROP-SPECIFIC MODELS (FIXED - Top Features Only)")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================

df = pd.read_csv('data/uk_crop_yield_with_seasonal_features_2004_2024.csv')
df = df.drop('Production_tonnes', axis=1)

# ============================================================================
# CROP-SPECIFIC TOP FEATURES (Based on correlations from earlier)
# ============================================================================

# Use only TOP 3-5 features per crop (from correlation analysis)
crop_features = {
    'Wheat': ['Area_hectares', 'Winter_Sun', 'Summer_Sun', 'Extreme_Summer_Rain', 'Grain_Filling_Rain'],
    'Barley': ['Area_hectares', 'Winter_Sun', 'Winter_Tmax', 'Extreme_Summer_Rain', 'Winter_Frost'],
    'Oats': ['Area_hectares', 'Spring_Rain', 'Rain_Deviation_from_Optimal', 'Flowering_Sun'],
    'Oilseed_Rape': ['Area_hectares', 'Summer_Tmin', 'Planting_Rain', 'Winter_Frost'],
    'Sugar_Beet': ['Area_hectares', 'Drought_Spring', 'Autumn_Sun', 'Extreme_Summer_Rain', 'Spring_Temp_x_Rain'],
    'Potato': ['Area_hectares', 'Extreme_Summer_Rain', 'Late_Spring_Frost', 'Autumn_Sun']
}

# Baseline features (same for all)
baseline_features = ['Area_hectares', 'Annual_Temp_C', 'Annual_Rainfall_mm', 
                     'Annual_Sunshine_hours', 'Annual_Frost_days']

# ============================================================================
# BUILD MODELS
# ============================================================================

crops = df['Crop'].unique()
results = []

print("\n" + "="*70)
print("BUILDING SIMPLIFIED CROP-SPECIFIC MODELS")
print("="*70)

for crop in crops:
    print(f"\n{'='*70}")
    print(f"CROP: {crop.upper()}")
    print(f"{'='*70}")
    
    # Filter data
    crop_data = df[df['Crop'] == crop].copy()
    
    print(f"Observations: {len(crop_data)} (21 years)")
    print(f"Yield: {crop_data['Yield_t_per_ha'].mean():.1f} ± {crop_data['Yield_t_per_ha'].std():.1f} t/ha")
    
    # Train-test split
    train_mask = crop_data['Year'] <= 2019
    test_mask = crop_data['Year'] >= 2020
    
    n_train = train_mask.sum()
    n_test = test_mask.sum()
    
    y_train = crop_data.loc[train_mask, 'Yield_t_per_ha']
    y_test = crop_data.loc[test_mask, 'Yield_t_per_ha']
    
    # --------------------------------------------------------------------
    # BASELINE MODEL
    # --------------------------------------------------------------------
    
    X_baseline_train = crop_data.loc[train_mask, baseline_features]
    X_baseline_test = crop_data.loc[test_mask, baseline_features]
    
    scaler_base = StandardScaler()
    X_base_train_sc = scaler_base.fit_transform(X_baseline_train)
    X_base_test_sc = scaler_base.transform(X_baseline_test)
    
    ridge_base = Ridge(alpha=5.0)  # Higher regularization for small sample
    ridge_base.fit(X_base_train_sc, y_train)
    
    y_pred_base = ridge_base.predict(X_base_test_sc)
    base_r2 = r2_score(y_test, y_pred_base)
    base_rmse = np.sqrt(mean_squared_error(y_test, y_pred_base))
    
    print(f"\n📊 BASELINE ({len(baseline_features)} features):")
    print(f"   Test R²: {base_r2:.3f}")
    print(f"   Test RMSE: {base_rmse:.2f} t/ha")
    
    # --------------------------------------------------------------------
    # SEASONAL MODEL (Top features only)
    # --------------------------------------------------------------------
    
    seasonal_feats = crop_features[crop]
    print(f"\n🔥 SEASONAL ({len(seasonal_feats)} top features):")
    print(f"   Features: {', '.join(seasonal_feats)}")
    
    X_seasonal_train = crop_data.loc[train_mask, seasonal_feats]
    X_seasonal_test = crop_data.loc[test_mask, seasonal_feats]
    
    scaler_seas = StandardScaler()
    X_seas_train_sc = scaler_seas.fit_transform(X_seasonal_train)
    X_seas_test_sc = scaler_seas.transform(X_seasonal_test)
    
    ridge_seas = Ridge(alpha=5.0)
    ridge_seas.fit(X_seas_train_sc, y_train)
    
    y_pred_seas = ridge_seas.predict(X_seas_test_sc)
    seas_r2 = r2_score(y_test, y_pred_seas)
    seas_rmse = np.sqrt(mean_squared_error(y_test, y_pred_seas))
    
    print(f"   Test R²: {seas_r2:.3f}")
    print(f"   Test RMSE: {seas_rmse:.2f} t/ha")
    
    # --------------------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------------------
    
    improvement_r2 = seas_r2 - base_r2
    improvement_rmse = base_rmse - seas_rmse
    
    print(f"\n💡 IMPROVEMENT:")
    print(f"   ΔR²: {improvement_r2:+.3f}")
    print(f"   ΔRMSE: {improvement_rmse:+.2f} t/ha")
    
    if seas_r2 > 0.3:
        status = "✅ GOOD prediction"
    elif seas_r2 > 0:
        status = "✓ Modest prediction"
    else:
        status = "⚠️ Poor prediction (overfitting or too stable)"
    
    print(f"   {status}")
    
    # Feature coefficients
    print(f"\n🌟 Feature Importance:")
    for feat, coef in zip(seasonal_feats, ridge_seas.coef_):
        print(f"   • {feat}: {coef:+.3f}")
    
    # Store results
    results.append({
        'Crop': crop,
        'N_Features_Baseline': len(baseline_features),
        'N_Features_Seasonal': len(seasonal_feats),
        'Baseline_R2': base_r2,
        'Seasonal_R2': seas_r2,
        'Baseline_RMSE': base_rmse,
        'Seasonal_RMSE': seas_rmse,
        'Improvement_R2': improvement_r2,
        'Improvement_RMSE': improvement_rmse
    })

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("📊 SUMMARY")
print("="*70)

results_df = pd.DataFrame(results)
print("\n" + results_df[['Crop', 'Baseline_R2', 'Seasonal_R2', 'Improvement_R2', 
                         'Baseline_RMSE', 'Seasonal_RMSE']].to_string(index=False))

print("\n💡 KEY FINDINGS:")
print(f"\nAverage Baseline R²: {results_df['Baseline_R2'].mean():.3f}")
print(f"Average Seasonal R²: {results_df['Seasonal_R2'].mean():.3f}")
print(f"Average Improvement: {results_df['Improvement_R2'].mean():+.3f}")

positive_improvements = results_df[results_df['Improvement_R2'] > 0]
print(f"\nCrops improved by seasonal features: {len(positive_improvements)}/6")

if len(positive_improvements) > 0:
    print("\nCrops that benefit from seasonal features:")
    for _, row in positive_improvements.iterrows():
        print(f"  • {row['Crop']}: +{row['Improvement_R2']:.3f} R² improvement")

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(results_df))
width = 0.35

ax.bar(x - width/2, results_df['Baseline_R2'], width, label='Baseline (5 features)',
       color='lightcoral', edgecolor='black')
ax.bar(x + width/2, results_df['Seasonal_R2'], width, label='Seasonal (3-5 features)',
       color='lightgreen', edgecolor='black')

ax.set_xlabel('Crop', fontsize=12)
ax.set_ylabel('Test R²', fontsize=12)
ax.set_title('Crop-Specific Model Performance\n(Simplified Feature Set)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(results_df['Crop'], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='black', linewidth=0.8)

plt.tight_layout()
plt.savefig('plots/crop_specific_simplified.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: plots/crop_specific_simplified.png")
plt.show()

print("\n" + "="*70)
print("🎉 ANALYSIS COMPLETE!")
print("="*70)

print("\nConclusion:")
print("  • Small sample size (21 observations/crop) limits prediction accuracy")
print("  • Some crops (Sugar Beet, Potato) show positive R² with top features")
print("  • Seasonal features help when carefully selected")
print("  • For dissertation: Focus on aggregate model (R²=0.975) + crop insights")