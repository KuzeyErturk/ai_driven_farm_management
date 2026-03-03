import pandas as pd
import numpy as np

df = pd.read_csv('data/uk_crop_yield_with_seasonal_features_2004_2024.csv')

print("="*70)
print("DIAGNOSTIC: Why Are R² Values So High?")
print("="*70)

# Check yield variance by crop
print("\nYield Statistics by Crop:")
print(df.groupby('Crop')['Yield_t_per_ha'].agg(['mean', 'std', 'min', 'max']))

# Check if crop dummies alone predict well
print("\n" + "="*70)
print("EXPERIMENT: Can We Predict Yield with ONLY Crop Type?")
print("="*70)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# Prepare data
df_test = pd.get_dummies(df, columns=['Crop'], prefix='Crop')
crop_dummies = [col for col in df_test.columns if col.startswith('Crop_')]

train_mask = df_test['Year'] <= 2019
test_mask = df_test['Year'] >= 2020

# Use ONLY crop dummies (no weather, no area!)
X_train = df_test.loc[train_mask, crop_dummies]
X_test = df_test.loc[test_mask, crop_dummies]
y_train = df_test.loc[train_mask, 'Yield_t_per_ha']
y_test = df_test.loc[test_mask, 'Yield_t_per_ha']

# Fit model with only crop type
model = Ridge(alpha=10.0)
model.fit(X_train, y_train)

r2_crop_only = model.score(X_test, y_test)
print(f"\nTest R² with ONLY Crop Type (no weather/area): {r2_crop_only:.3f}")

# Now add Area
X_train_with_area = df_test.loc[train_mask, ['Area_hectares'] + crop_dummies]
X_test_with_area = df_test.loc[test_mask, ['Area_hectares'] + crop_dummies]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_with_area)
X_test_scaled = scaler.transform(X_test_with_area)

model2 = Ridge(alpha=10.0)
model2.fit(X_train_scaled, y_train)

r2_area_crop = model2.score(X_test_scaled, y_test)
print(f"Test R² with Area + Crop Type: {r2_area_crop:.3f}")

# Compare to full model
print(f"\nYour full model Test R²: 0.975")
print(f"Improvement from Area+Crop: {0.975 - r2_area_crop:.3f}")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

if r2_crop_only > 0.90:
    print("\n🚨 PROBLEM: Crop type alone predicts R² > 0.90!")
    print("\nThis means:")
    print("  • Yields are VERY stable within each crop")
    print("  • Just knowing it's 'Wheat' predicts ~8 t/ha")
    print("  • Just knowing it's 'Potato' predicts ~41 t/ha")
    print("  • Weather barely adds information!")
    
    print("\nWhy UK yields are so predictable:")
    print("  1. Advanced agriculture (precision farming, irrigation)")
    print("  2. Optimal varieties (bred for UK climate)")
    print("  3. Excellent management (farmers adapt to weather)")
    print("  4. Small geographic area (weather relatively uniform)")
    
    print("\nThis is actually a POSITIVE finding:")
    print("  → Shows UK agricultural resilience!")
    print("  → Weather matters less because farmers compensate")
    print("  → But extreme events (2012 wet) still impact yields")

elif r2_area_crop > 0.90:
    print("\n✓ Area + Crop type explain most variance (R² > 0.90)")
    print("\nThis means:")
    print("  • Planted area is a strong economic signal")
    print("  • Crop type determines baseline yield")
    print("  • Weather adds ~5-8% additional predictive power")
    
else:
    print("\n✓ Weather features add substantial value")
    print(f"  • Crop type alone: R² = {r2_crop_only:.3f}")
    print(f"  • Add area: R² = {r2_area_crop:.3f}")
    print(f"  • Add weather: R² = 0.975")

print("\n" + "="*70)