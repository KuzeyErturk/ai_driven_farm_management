import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

print("="*70)
print("RANDOM FOREST WITH HUMIDITY + FERTILIZERS")
print("="*70)

# ============================================================================
# LOAD ENHANCED DATA
# ============================================================================

df = pd.read_csv('data/regional_crop_yield_weather_enhanced.csv')

print(f"\nDataset: {len(df)} observations")
print(f"Total variables: {len(df.columns)}")

# ============================================================================
# UPDATED FEATURE SETS (ADDING HUMIDITY + FERTILIZERS)
# ============================================================================

# Original features + new ones
crop_features_original = {
    'Wheat': ['Area_hectares', 'Winter_Sun', 'Summer_Sun', 'Extreme_Summer_Rain', 'Grain_Filling_Rain'],
    'Barley': ['Area_hectares', 'Winter_Sun', 'Winter_Tmax', 'Extreme_Summer_Rain', 'Winter_Frost'],
    'Oats': ['Area_hectares', 'Spring_Rain', 'Rain_Deviation_from_Optimal', 'Flowering_Sun'],
    'Oilseed_Rape': ['Area_hectares', 'Summer_Tmin', 'Planting_Rain', 'Winter_Frost']
}

# Enhanced features (add humidity + fertilizers where they correlate)
crop_features_enhanced = {
    'Wheat': ['Area_hectares', 'Winter_Sun', 'Summer_Sun', 'Extreme_Summer_Rain', 
              'Grain_Filling_Rain', 'Annual_Humidity_Percent'],
    
    'Barley': ['Area_hectares', 'Winter_Sun', 'Winter_Tmax', 'Extreme_Summer_Rain', 
               'Winter_Frost', 'Annual_Humidity_Percent', 'Total_Nitrogen_kg_ha', 
               'Total_Phosphate_kg_ha', 'Total_Potash_kg_ha'],
    
    'Oats': ['Area_hectares', 'Spring_Rain', 'Rain_Deviation_from_Optimal', 
             'Flowering_Sun', 'Total_Nitrogen_kg_ha'],
    
    'Oilseed_Rape': ['Area_hectares', 'Summer_Tmin', 'Planting_Rain', 'Winter_Frost',
                     'Annual_Humidity_Percent', 'Total_Nitrogen_kg_ha']
}

# ============================================================================
# TEST BOTH FEATURE SETS
# ============================================================================

results = []

for crop in df['Crop'].unique():
    print(f"\n{'='*70}")
    print(f"CROP: {crop.upper()}")
    print(f"{'='*70}")
    
    crop_data = df[df['Crop'] == crop].copy()
    
    print(f"Observations: {len(crop_data)}")
    print(f"Yield: {crop_data['Yield_t_per_ha'].mean():.1f} ± {crop_data['Yield_t_per_ha'].std():.1f} t/ha")
    
    # Train-test split
    train_mask = crop_data['Year'] <= 2019
    test_mask = crop_data['Year'] >= 2020
    
    n_train = train_mask.sum()
    n_test = test_mask.sum()
    
    y_train = crop_data.loc[train_mask, 'Yield_t_per_ha']
    y_test = crop_data.loc[test_mask, 'Yield_t_per_ha']
    
    # ----------------------------------------------------------------
    # MODEL 1: ORIGINAL FEATURES (baseline)
    # ----------------------------------------------------------------
    
    features_orig = crop_features_original[crop]
    
    X_train_orig = crop_data.loc[train_mask, features_orig]
    X_test_orig = crop_data.loc[test_mask, features_orig]
    
    rf_orig = RandomForestRegressor(n_estimators=100, max_depth=8, 
                                    min_samples_split=3, random_state=42)
    rf_orig.fit(X_train_orig, y_train)
    
    y_pred_orig = rf_orig.predict(X_test_orig)
    r2_orig = r2_score(y_test, y_pred_orig)
    rmse_orig = np.sqrt(mean_squared_error(y_test, y_pred_orig))
    
    print(f"\n📊 ORIGINAL FEATURES ({len(features_orig)}):")
    print(f"   {', '.join(features_orig)}")
    print(f"   Test R²: {r2_orig:.3f}")
    print(f"   Test RMSE: {rmse_orig:.2f} t/ha")
    
    # ----------------------------------------------------------------
    # MODEL 2: ENHANCED FEATURES (+ humidity + fertilizers)
    # ----------------------------------------------------------------
    
    features_enh = crop_features_enhanced[crop]
    
    X_train_enh = crop_data.loc[train_mask, features_enh]
    X_test_enh = crop_data.loc[test_mask, features_enh]
    
    rf_enh = RandomForestRegressor(n_estimators=100, max_depth=8,
                                   min_samples_split=3, random_state=42)
    rf_enh.fit(X_train_enh, y_train)
    
    y_pred_enh = rf_enh.predict(X_test_enh)
    r2_enh = r2_score(y_test, y_pred_enh)
    rmse_enh = np.sqrt(mean_squared_error(y_test, y_pred_enh))
    
    print(f"\n🔥 ENHANCED FEATURES ({len(features_enh)}):")
    print(f"   Added: {', '.join([f for f in features_enh if f not in features_orig])}")
    print(f"   Test R²: {r2_enh:.3f}")
    print(f"   Test RMSE: {rmse_enh:.2f} t/ha")
    
    # ----------------------------------------------------------------
    # COMPARISON
    # ----------------------------------------------------------------
    
    improvement = r2_enh - r2_orig
    
    print(f"\n💡 IMPROVEMENT:")
    print(f"   ΔR²: {improvement:+.3f}")
    print(f"   ΔRMSE: {rmse_orig - rmse_enh:+.2f} t/ha")
    
    if improvement > 0.05:
        print(f"   ✅ SIGNIFICANT IMPROVEMENT!")
    elif improvement > 0.01:
        print(f"   ✓ Modest improvement")
    elif improvement > -0.01:
        print(f"   ≈ Similar performance")
    else:
        print(f"   ⚠️ Worse (overfitting)")
    
    # Feature importance
    print(f"\n🌟 Top 5 Feature Importance (Enhanced Model):")
    importances = rf_enh.feature_importances_
    indices = np.argsort(importances)[::-1][:5]
    
    for i, idx in enumerate(indices, 1):
        feat_name = features_enh[idx]
        importance = importances[idx]
        print(f"   {i}. {feat_name}: {importance:.3f}")
    
    # Store results
    results.append({
        'Crop': crop,
        'N_Features_Orig': len(features_orig),
        'N_Features_Enh': len(features_enh),
        'Original_R2': r2_orig,
        'Enhanced_R2': r2_enh,
        'Improvement': improvement,
        'Original_RMSE': rmse_orig,
        'Enhanced_RMSE': rmse_enh
    })

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("📊 SUMMARY: ORIGINAL vs ENHANCED FEATURES")
print("="*70)

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

print(f"\n💡 OVERALL RESULTS:")
print(f"   Average Original R²: {results_df['Original_R2'].mean():.3f}")
print(f"   Average Enhanced R²: {results_df['Enhanced_R2'].mean():.3f}")
print(f"   Average Improvement: {results_df['Improvement'].mean():+.3f}")

improved_count = (results_df['Improvement'] > 0.01).sum()
print(f"\n   Crops improved: {improved_count}/4")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: R² comparison
ax1 = axes[0]
x = np.arange(len(results_df))
width = 0.35

ax1.bar(x - width/2, results_df['Original_R2'], width, label='Original Features',
        color='lightcoral', edgecolor='black')
ax1.bar(x + width/2, results_df['Enhanced_R2'], width, label='+ Humidity + Fertilizers',
        color='lightgreen', edgecolor='black')

ax1.set_xlabel('Crop', fontsize=12)
ax1.set_ylabel('Test R²', fontsize=12)
ax1.set_title('Impact of Adding Humidity + Fertilizers', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(results_df['Crop'], rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax1.axhline(y=0, color='black', linewidth=1)

# Plot 2: Improvement
ax2 = axes[1]
colors = ['green' if val > 0.01 else ('yellow' if val > -0.01 else 'red') 
          for val in results_df['Improvement']]
ax2.barh(results_df['Crop'], results_df['Improvement'], color=colors,
         edgecolor='black', alpha=0.7)

ax2.set_xlabel('R² Improvement', fontsize=12)
ax2.set_ylabel('Crop', fontsize=12)
ax2.set_title('Improvement from New Features', fontsize=14, fontweight='bold')
ax2.axvline(x=0, color='black', linewidth=1)
ax2.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (crop, val) in enumerate(zip(results_df['Crop'], results_df['Improvement'])):
    ax2.text(val, i, f' {val:+.3f}', va='center',
             ha='left' if val > 0 else 'right', fontsize=10)

plt.tight_layout()
plt.savefig('plots/humidity_fertilizer_impact.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: plots/humidity_fertilizer_impact.png")
plt.show()

# ============================================================================
# CONCLUSION
# ============================================================================

print("\n" + "="*70)
print("🎯 CONCLUSION")
print("="*70)

if results_df['Improvement'].mean() > 0.05:
    print("\n✅ HUMIDITY + FERTILIZERS SIGNIFICANTLY HELP!")
    print("   Adding these variables improved predictions.")
elif results_df['Improvement'].mean() > 0.01:
    print("\n✓ Modest improvement from humidity + fertilizers")
    print("   Some crops benefit, worth including.")
elif results_df['Improvement'].mean() > -0.01:
    print("\n≈ MINIMAL IMPACT")
    print("   Humidity + fertilizers don't add much value.")
else:
    print("\n⚠️ NO IMPROVEMENT (possible overfitting)")
    print("   Original features were better.")

# Best model per crop
print("\n📋 BEST MODELS PER CROP:")
for _, row in results_df.iterrows():
    if row['Enhanced_R2'] > row['Original_R2']:
        best = f"Enhanced (R²={row['Enhanced_R2']:.3f})"
    else:
        best = f"Original (R²={row['Original_R2']:.3f})"
    print(f"   {row['Crop']}: {best}")

print("\n" + "="*70)
print("✅ HUMIDITY + FERTILIZER ANALYSIS COMPLETE!")
print("="*70)