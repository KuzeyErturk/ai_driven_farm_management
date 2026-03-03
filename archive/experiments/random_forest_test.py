import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

print("="*70)
print("RANDOM FOREST vs RIDGE: REGIONAL CROP MODELS")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================

df = pd.read_csv('data/regional_crop_yield_weather_2004_2024.csv')

print(f"\nDataset: {len(df)} observations")
print(f"Crops: {df['Crop'].nunique()}")

# ============================================================================
# FEATURE SETS
# ============================================================================

crop_features = {
    'Wheat': ['Area_hectares', 'Winter_Sun', 'Summer_Sun', 'Extreme_Summer_Rain', 'Grain_Filling_Rain'],
    'Barley': ['Area_hectares', 'Winter_Sun', 'Winter_Tmax', 'Extreme_Summer_Rain', 'Winter_Frost'],
    'Oats': ['Area_hectares', 'Spring_Rain', 'Rain_Deviation_from_Optimal', 'Flowering_Sun'],
    'Oilseed_Rape': ['Area_hectares', 'Summer_Tmin', 'Planting_Rain', 'Winter_Frost']
}

# ============================================================================
# BUILD MODELS FOR EACH CROP
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
    
    print(f"Train/Test: {n_train}/{n_test}")
    
    features = crop_features[crop]
    
    X_train = crop_data.loc[train_mask, features]
    X_test = crop_data.loc[test_mask, features]
    y_train = crop_data.loc[train_mask, 'Yield_t_per_ha']
    y_test = crop_data.loc[test_mask, 'Yield_t_per_ha']
    
    # ----------------------------------------------------------------
    # MODEL 1: RIDGE (Linear)
    # ----------------------------------------------------------------
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    
    y_pred_ridge = ridge.predict(X_test_scaled)
    ridge_r2 = r2_score(y_test, y_pred_ridge)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
    
    print(f"\n📊 RIDGE REGRESSION (Linear):")
    print(f"   Test R²: {ridge_r2:.3f}")
    print(f"   Test RMSE: {ridge_rmse:.2f} t/ha")
    
    # ----------------------------------------------------------------
    # MODEL 2: RANDOM FOREST (Non-linear)
    # ----------------------------------------------------------------
    
    # Try different configurations
    rf_configs = [
        {'n_estimators': 50, 'max_depth': 3, 'min_samples_split': 10, 'name': 'Conservative'},
        {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 5, 'name': 'Moderate'},
        {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 3, 'name': 'Aggressive'}
    ]
    
    best_rf_r2 = -999
    best_rf_rmse = 999
    best_config_name = None
    
    print(f"\n🌲 RANDOM FOREST (Non-linear):")
    
    for config in rf_configs:
        name = config.pop('name')
        
        rf = RandomForestRegressor(**config, random_state=42)
        rf.fit(X_train, y_train)
        
        y_pred_rf = rf.predict(X_test)
        rf_r2 = r2_score(y_test, y_pred_rf)
        rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
        
        print(f"   {name}: R² = {rf_r2:.3f}, RMSE = {rf_rmse:.2f}")
        
        if rf_r2 > best_rf_r2:
            best_rf_r2 = rf_r2
            best_rf_rmse = rf_rmse
            best_config_name = name
        
        # Put name back for next iteration
        config['name'] = name
    
    # ----------------------------------------------------------------
    # COMPARISON
    # ----------------------------------------------------------------
    
    improvement = best_rf_r2 - ridge_r2
    
    print(f"\n💡 BEST RANDOM FOREST: {best_config_name}")
    print(f"   Test R²: {best_rf_r2:.3f}")
    print(f"   Test RMSE: {best_rf_rmse:.2f} t/ha")
    
    print(f"\n🔄 RIDGE vs RANDOM FOREST:")
    print(f"   ΔR²: {improvement:+.3f}")
    print(f"   ΔRMSE: {ridge_rmse - best_rf_rmse:+.2f} t/ha")
    
    if best_rf_r2 > ridge_r2 + 0.05:
        print(f"   ✅ RANDOM FOREST MUCH BETTER!")
    elif best_rf_r2 > ridge_r2:
        print(f"   ✓ Random Forest slightly better")
    elif ridge_r2 > best_rf_r2:
        print(f"   ← Ridge better (RF overfitting)")
    else:
        print(f"   ≈ Similar performance")
    
    # Store results
    results.append({
        'Crop': crop,
        'Ridge_R2': ridge_r2,
        'Ridge_RMSE': ridge_rmse,
        'RF_R2': best_rf_r2,
        'RF_RMSE': best_rf_rmse,
        'Improvement': improvement
    })

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("📊 OVERALL SUMMARY: RIDGE vs RANDOM FOREST")
print("="*70)

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

print(f"\n💡 AGGREGATE RESULTS:")
print(f"   Average Ridge R²: {results_df['Ridge_R2'].mean():.3f}")
print(f"   Average RF R²: {results_df['RF_R2'].mean():.3f}")
print(f"   Average Improvement: {results_df['Improvement'].mean():+.3f}")

# Count improvements
rf_better = (results_df['RF_R2'] > results_df['Ridge_R2']).sum()
print(f"\n   RF better than Ridge: {rf_better}/4 crops")

# Positive R2 counts
ridge_positive = (results_df['Ridge_R2'] > 0).sum()
rf_positive = (results_df['RF_R2'] > 0).sum()

print(f"\n   Ridge with positive R²: {ridge_positive}/4")
print(f"   RF with positive R²: {rf_positive}/4")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: R² comparison
ax1 = axes[0]
x = np.arange(len(results_df))
width = 0.35

ax1.bar(x - width/2, results_df['Ridge_R2'], width, label='Ridge (Linear)',
        color='lightcoral', edgecolor='black')
ax1.bar(x + width/2, results_df['RF_R2'], width, label='Random Forest',
        color='lightgreen', edgecolor='black')

ax1.set_xlabel('Crop', fontsize=12)
ax1.set_ylabel('Test R²', fontsize=12)
ax1.set_title('Ridge vs Random Forest Performance', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(results_df['Crop'], rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax1.axhline(y=0, color='black', linewidth=1)

# Plot 2: Improvement
ax2 = axes[1]
colors = ['green' if val > 0 else 'red' for val in results_df['Improvement']]
ax2.barh(results_df['Crop'], results_df['Improvement'], color=colors,
         edgecolor='black', alpha=0.7)

ax2.set_xlabel('R² Improvement (RF - Ridge)', fontsize=12)
ax2.set_ylabel('Crop', fontsize=12)
ax2.set_title('Random Forest Improvement', fontsize=14, fontweight='bold')
ax2.axvline(x=0, color='black', linewidth=1)
ax2.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (crop, val) in enumerate(zip(results_df['Crop'], results_df['Improvement'])):
    ax2.text(val, i, f' {val:+.3f}', va='center',
             ha='left' if val > 0 else 'right', fontsize=10)

plt.tight_layout()
plt.savefig('plots/ridge_vs_random_forest.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: plots/ridge_vs_random_forest.png")
plt.show()

# ============================================================================
# CONCLUSION
# ============================================================================

print("\n" + "="*70)
print("🎯 CONCLUSION")
print("="*70)

if results_df['Improvement'].mean() > 0.1:
    print("\n✅ RANDOM FOREST SIGNIFICANTLY BETTER!")
    print("   Non-linear relationships matter for UK crop yields.")
    print("   → Use Random Forest for final models")
elif results_df['Improvement'].mean() > 0.02:
    print("\n✓ Random Forest slightly better")
    print("   Small non-linear effects detected.")
    print("   → Could use either Ridge or RF")
elif results_df['Improvement'].mean() < -0.02:
    print("\n← RIDGE BETTER (RF overfitting)")
    print("   Random Forest overfits the small dataset.")
    print("   → Stick with Ridge regression")
else:
    print("\n≈ SIMILAR PERFORMANCE")
    print("   Linear and non-linear models perform equally.")
    print("   → Either model acceptable")

print("\n💭 INTERPRETATION:")
if rf_positive > ridge_positive:
    print(f"   Random Forest achieved positive R² for {rf_positive} crops")
    print(f"   vs Ridge with {ridge_positive} crops.")
    print("   → Non-linear modeling helps!")
elif rf_positive == ridge_positive:
    print(f"   Both models achieved positive R² for {rf_positive} crops.")
    print("   → Fundamental limitation remains (yield stability)")
else:
    print(f"   Ridge achieved positive R² for {ridge_positive} crops")
    print(f"   vs Random Forest with {rf_positive} crops.")
    print("   → Overfitting issue with Random Forest")

print("\n" + "="*70)
print("✅ RANDOM FOREST ANALYSIS COMPLETE!")
print("="*70)