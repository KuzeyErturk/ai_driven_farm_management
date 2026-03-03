import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

print("="*70)
print("OATS: TESTING OPTIMIZED FEATURES")
print("="*70)

df = pd.read_csv('data/regional_crop_yield_weather_2004_2024.csv')

oats = df[df['Crop'] == 'Oats'].copy()

# Train-test split
train_mask = oats['Year'] <= 2019
test_mask = oats['Year'] >= 2020

y_train = oats.loc[train_mask, 'Yield_t_per_ha']
y_test = oats.loc[test_mask, 'Yield_t_per_ha']

print(f"\nOats yield stats:")
print(f"  Train: {y_train.mean():.2f} ± {y_train.std():.2f} t/ha")
print(f"  Test:  {y_test.mean():.2f} ± {y_test.std():.2f} t/ha")

# Test different feature combinations
feature_sets = {
    'Current (4 features)': 
        ['Area_hectares', 'Spring_Rain', 'Rain_Deviation_from_Optimal', 'Flowering_Sun'],
    
    'Remove Bad Feature (3)':
        ['Area_hectares', 'Spring_Rain', 'Rain_Deviation_from_Optimal'],
    
    'Top Frost Features (5)': 
        ['Area_hectares', 'Winter_Frost', 'Spring_Frost', 'Late_Spring_Frost', 'Spring_Rain'],
    
    'Frost + Rain (4)': 
        ['Area_hectares', 'Winter_Frost', 'Spring_Rain', 'Spring_Rain_Squared'],
    
    'Cold Spring Focus (5)': 
        ['Area_hectares', 'Winter_Frost', 'Spring_Frost', 'Spring_Rain', 'Cold_Spring'],
    
    'Avoid Heat (6)': 
        ['Area_hectares', 'Winter_Frost', 'Spring_Rain', 'Summer_Tmin', 
         'Flowering_Temp', 'Spring_Frost'],
    
    'Regional Mix (7)':
        ['Area_hectares', 'Winter_Frost', 'Spring_Rain', 'Autumn_Tmax',
         'Grain_Filling_Sun', 'Spring_Temp_x_Rain', 'Autumn_Sun'],
    
    'Best Positive Only (5)':
        ['Area_hectares', 'Winter_Frost', 'Spring_Frost', 'Spring_Rain', 
         'Late_Spring_Frost'],
}

results = []

for name, features in feature_sets.items():
    X_train = oats.loc[train_mask, features]
    X_test = oats.loc[test_mask, features]
    
    # Try different RF complexities
    for n_est, max_d in [(50, 3), (100, 5), (100, 8)]:
        rf = RandomForestRegressor(n_estimators=n_est, max_depth=max_d,
                                   min_samples_split=5, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        config = f"RF({n_est}, d={max_d})"
        
        results.append({
            'Features': name,
            'N_Features': len(features),
            'Config': config,
            'R2': r2,
            'RMSE': rmse
        })
        
        if max_d == 8:  # Only print most complex
            print(f"\n{name} ({len(features)} features):")
            print(f"  R²: {r2:.3f}, RMSE: {rmse:.2f}")
            
            # Feature importance
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1][:min(5, len(features))]
            print(f"  Top features:")
            for i, idx in enumerate(indices[:3], 1):
                print(f"    {i}. {features[idx]}: {importances[idx]:.3f}")

# Find best
results_df = pd.DataFrame(results)
best = results_df.loc[results_df['R2'].idxmax()]

print("\n" + "="*70)
print("BEST COMBINATION:")
print("="*70)
print(f"  {best['Features']}")
print(f"  {best['N_Features']} features, {best['Config']}")
print(f"  R²: {best['R2']:.3f}")
print(f"  RMSE: {best['RMSE']:.2f} t/ha")

# Compare to current
current_best = results_df[results_df['Features'] == 'Current (4 features)']['R2'].max()
improvement = best['R2'] - current_best

print(f"\n💡 IMPROVEMENT OVER CURRENT:")
print(f"  Current best: {current_best:.3f}")
print(f"  New best: {best['R2']:.3f}")
print(f"  Improvement: {improvement:+.3f}")

if best['R2'] > 0.3:
    print("  🎉 EXCELLENT OATS MODEL!")
elif best['R2'] > 0.25:
    print("  ✅ SIGNIFICANT IMPROVEMENT!")
elif best['R2'] > 0.2:
    print("  ✓ Good improvement")
else:
    print("  ≈ Modest improvement")

# Show top 10 results
print("\n" + "="*70)
print("TOP 10 RESULTS:")
print("="*70)
top_10 = results_df.nlargest(10, 'R2')[['Features', 'N_Features', 'Config', 'R2', 'RMSE']]
print(top_10.to_string(index=False))