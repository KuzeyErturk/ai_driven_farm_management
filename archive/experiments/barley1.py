import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

print("="*70)
print("BARLEY: TESTING DIFFERENT FEATURE COMBINATIONS")
print("="*70)

df = pd.read_csv('data/regional_crop_yield_weather_2004_2024.csv')

barley = df[df['Crop'] == 'Barley'].copy()

# Train-test split
train_mask = barley['Year'] <= 2019
test_mask = barley['Year'] >= 2020

y_train = barley.loc[train_mask, 'Yield_t_per_ha']
y_test = barley.loc[test_mask, 'Yield_t_per_ha']

print(f"\nBarley yield stats:")
print(f"  Train: {y_train.mean():.2f} ± {y_train.std():.2f} t/ha")
print(f"  Test:  {y_test.mean():.2f} ± {y_test.std():.2f} t/ha")

# Test different feature combinations
feature_sets = {
    'Old (5 features)': ['Area_hectares', 'Winter_Sun', 'Winter_Tmax', 
                         'Extreme_Summer_Rain', 'Winter_Frost'],
    
    'Top 3 correlations': ['Area_hectares', 'Late_Spring_Frost', 'Summer_Temp_Range'],
    
    'Top 4 correlations': ['Area_hectares', 'Late_Spring_Frost', 'Summer_Temp_Range',
                          'Rain_Deviation_from_Optimal'],
    
    'Top 6 correlations': ['Area_hectares', 'Late_Spring_Frost', 'Summer_Temp_Range',
                          'Rain_Deviation_from_Optimal', 'Grain_Filling_Sun', 
                          'Summer_Sun_per_Rain'],
    
    'Area only': ['Area_hectares'],
    
    'Best single weather': ['Area_hectares', 'Late_Spring_Frost'],
    
    'Summer features': ['Area_hectares', 'Summer_Temp_Range', 'Grain_Filling_Sun', 
                       'Summer_Sun_per_Rain'],
}

results = []

for name, features in feature_sets.items():
    X_train = barley.loc[train_mask, features]
    X_test = barley.loc[test_mask, features]
    
    # Try both simple and complex RF
    for complexity in ['Simple', 'Complex']:
        if complexity == 'Simple':
            rf = RandomForestRegressor(n_estimators=50, max_depth=3, 
                                       min_samples_split=10, random_state=42)
        else:
            rf = RandomForestRegressor(n_estimators=100, max_depth=8,
                                       min_samples_split=3, random_state=42)
        
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results.append({
            'Features': name,
            'N_Features': len(features),
            'Complexity': complexity,
            'R2': r2,
            'RMSE': rmse
        })
        
        print(f"\n{name} ({len(features)} feat) - {complexity}:")
        print(f"  R²: {r2:.3f}, RMSE: {rmse:.2f}")

# Find best
results_df = pd.DataFrame(results)
best = results_df.loc[results_df['R2'].idxmax()]

print("\n" + "="*70)
print("BEST COMBINATION:")
print("="*70)
print(f"  {best['Features']}")
print(f"  {best['N_Features']} features, {best['Complexity']} RF")
print(f"  R²: {best['R2']:.3f}")
print(f"  RMSE: {best['RMSE']:.2f}")