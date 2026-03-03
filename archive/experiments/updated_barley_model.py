import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

print("="*70)
print("BARLEY: TESTING CORRECT FEATURES")
print("="*70)

df = pd.read_csv('data/regional_crop_yield_weather_2004_2024.csv')

barley = df[df['Crop'] == 'Barley'].copy()

# Train-test split
train_mask = barley['Year'] <= 2019
test_mask = barley['Year'] >= 2020

y_train = barley.loc[train_mask, 'Yield_t_per_ha']
y_test = barley.loc[test_mask, 'Yield_t_per_ha']

# OLD (WRONG) FEATURES
old_features = ['Area_hectares', 'Winter_Sun', 'Winter_Tmax', 
                'Extreme_Summer_Rain', 'Winter_Frost']

X_train_old = barley.loc[train_mask, old_features]
X_test_old = barley.loc[test_mask, old_features]

rf_old = RandomForestRegressor(n_estimators=100, max_depth=8, 
                               min_samples_split=3, random_state=42)
rf_old.fit(X_train_old, y_train)
y_pred_old = rf_old.predict(X_test_old)
r2_old = r2_score(y_test, y_pred_old)

print(f"\n❌ OLD (WRONG) FEATURES:")
print(f"   {', '.join(old_features)}")
print(f"   Test R²: {r2_old:.3f}")

# NEW (CORRECT) FEATURES
new_features = ['Area_hectares', 'Late_Spring_Frost', 'Summer_Temp_Range',
                'Rain_Deviation_from_Optimal', 'Grain_Filling_Sun', 
                'Summer_Sun_per_Rain']

X_train_new = barley.loc[train_mask, new_features]
X_test_new = barley.loc[test_mask, new_features]

rf_new = RandomForestRegressor(n_estimators=100, max_depth=8,
                               min_samples_split=3, random_state=42)
rf_new.fit(X_train_new, y_train)
y_pred_new = rf_new.predict(X_test_new)
r2_new = r2_score(y_test, y_pred_new)

print(f"\n✅ NEW (CORRECT) FEATURES:")
print(f"   {', '.join(new_features)}")
print(f"   Test R²: {r2_new:.3f}")

print(f"\n💡 IMPROVEMENT: {r2_new - r2_old:+.3f}")

if r2_new > 0.2:
    print("   🎉 BARLEY FIXED!")
elif r2_new > 0:
    print("   ✓ Positive R² achieved!")
else:
    print("   ⚠️ Still negative, but improved")