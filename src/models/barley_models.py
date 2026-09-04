import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Seperate models for winter and spring barley
print("SPRING vs WINTER BARLEY: SEPARATE MODELS")

spring = pd.read_csv('data/spring_barley_with_weather.csv')
winter = pd.read_csv('data/winter_barley_with_weather.csv')

print(f"\nSpring Barley: {len(spring)} observations")
print(f"Winter Barley: {len(winter)} observations")


weather_cols = [col for col in spring.columns if col not in 
                ['Year', 'Region', 'Crop', 'Area_hectares', 'Yield_t_per_ha']]

print("TOP CORRELATIONS WITH YIELD")

spring_corr = spring[weather_cols].corrwith(spring['Yield_t_per_ha']).sort_values(ascending=False)
winter_corr = winter[weather_cols].corrwith(winter['Yield_t_per_ha']).sort_values(ascending=False)

print("\n SPRING BARLEY (Top 10):")
print(spring_corr.head(10))

print("\n WINTER BARLEY (Top 10):")
print(winter_corr.head(10))


def build_barley_model(data, barley_type, features):
    
    # Train-test split
    train_mask = data['Year'] <= 2019
    test_mask = data['Year'] >= 2020
    
    X_train = data.loc[train_mask, features]
    X_test = data.loc[test_mask, features]
    y_train = data.loc[train_mask, 'Yield_t_per_ha']
    y_test = data.loc[test_mask, 'Yield_t_per_ha']
    
    print(f"{barley_type}:")
    print(f"  Train: {len(y_train)} obs, {y_train.mean():.2f} ± {y_train.std():.2f} t/ha")
    print(f"  Test: {len(y_test)} obs, {y_test.mean():.2f} ± {y_test.std():.2f} t/ha")
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, 
                               min_samples_split=3, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n RESULTS:")
    print(f"  R²: {r2:.3f}")
    print(f"  RMSE: {rmse:.2f} t/ha")
    
    # Feature importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:5]
    
    print(f"\n  Top 5 Features:")
    for i, idx in enumerate(indices, 1):
        print(f"    {i}. {features[idx]}: {importances[idx]:.3f}")
    
    return r2, rmse, y_test, y_pred

print("MODELING RESULTS")

# Spring Barley features
spring_features = ['Area_hectares', 'Spring_Rain', 'Spring_Frost', 
                   'Late_Spring_Frost', 'Summer_Sun', 'Grain_Filling_Sun']

# Winter Barley features
winter_features = ['Area_hectares', 'Winter_Frost', 'Autumn_Rain',
                   'Spring_Tmax', 'Winter_Sun', 'Planting_Rain']

spring_r2, spring_rmse, spring_y_test, spring_y_pred = build_barley_model(
    spring, " SPRING BARLEY", spring_features)

winter_r2, winter_rmse, winter_y_test, winter_y_pred = build_barley_model(
    winter, " WINTER BARLEY", winter_features)

print(f"  Spring Barley R²: {spring_r2:.3f}")
print(f"  Winter Barley R²: {winter_r2:.3f}")
print(f"  Average: {(spring_r2 + winter_r2)/2:.3f}")

if spring_r2 > 0 and winter_r2 > 0:
    print(f"\n  BOTH MODELS WORK!")
    print(f"  Separating spring/winter solved the problem!")
elif spring_r2 > 0 or winter_r2 > 0:
    print(f"\n  One model works (partial success)")
else:
    print(f"\n  Still challenging crops")


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Spring Barley
ax1 = axes[0]
ax1.scatter(spring_y_test, spring_y_pred, alpha=0.6, s=100, edgecolors='black')
ax1.plot([spring_y_test.min(), spring_y_test.max()], 
         [spring_y_test.min(), spring_y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Yield (t/ha)', fontsize=12)
ax1.set_ylabel('Predicted Yield (t/ha)', fontsize=12)
ax1.set_title(f'Spring Barley\nR² = {spring_r2:.3f}', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Winter Barley
ax2 = axes[1]
ax2.scatter(winter_y_test, winter_y_pred, alpha=0.6, s=100, 
            edgecolors='black', color='orange')
ax2.plot([winter_y_test.min(), winter_y_test.max()], 
         [winter_y_test.min(), winter_y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax2.set_xlabel('Actual Yield (t/ha)', fontsize=12)
ax2.set_ylabel('Predicted Yield (t/ha)', fontsize=12)
ax2.set_title(f'Winter Barley\nR² = {winter_r2:.3f}', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/spring_vs_winter_barley_models.png', dpi=300, bbox_inches='tight')
print("\nSaved: plots/spring_vs_winter_barley_models.png")
plt.show()