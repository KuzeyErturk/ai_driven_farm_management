import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Import baseline configuration
from baseline_model_config import (
    RF_PARAMS, CROP_FEATURES, DATA_SOURCES,
    BASELINE_RESULTS, CROP_NAME_IN_DATA
)

print("MODEL IMPROVEMENT EXPERIMENTS")
print("\nGoal: Improve upon baseline R² scores for dissertation")
print()
regional = pd.read_csv('data/processed/regional_crop_yield_weather_2004_2024.csv')
spring_barley = pd.read_csv('data/processed/spring_barley_with_weather.csv')
winter_barley = pd.read_csv('data/processed/winter_barley_with_weather.csv')

# Load fertilizer-enhanced data
regional_fert = pd.read_csv('data/processed/regional_crop_yield_weather_fertilizer_2004_2024.csv')
spring_barley_fert = pd.read_csv('data/processed/spring_barley_weather_fertilizer.csv')
winter_barley_fert = pd.read_csv('data/processed/winter_barley_weather_fertilizer.csv')

FERTILIZER_FEATURES = ['Nitrogen_kg_ha', 'Phosphorus_kg_ha']

def get_data(crop, with_fertilizer=False):
    if crop == 'Winter_Barley':
        return winter_barley_fert if with_fertilizer else winter_barley
    elif crop == 'Spring_Barley':
        return spring_barley_fert if with_fertilizer else spring_barley
    else:
        data = regional_fert if with_fertilizer else regional
        crop_name = CROP_NAME_IN_DATA.get(crop, crop)
        return data[data['Crop'] == crop_name]

# 2004-2019 train
# 2019 and onwards test
def train_test_split_temporal(data):
    train = data[data['Year'] <= 2019]
    test = data[data['Year'] >= 2020]
    return train, test


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Train model and return R² and RMSE."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return r2, rmse

print("EXPERIMENT 1: Baseline Verification")

baseline_scores = {}
for crop in ['Wheat', 'Winter_Barley', 'Spring_Barley', 'Oats', 'OSR']:
    data = get_data(crop, with_fertilizer=False)
    features = CROP_FEATURES[crop]

    train, test = train_test_split_temporal(data)
    X_train, y_train = train[features], train['Yield_t_per_ha']
    X_test, y_test = test[features], test['Yield_t_per_ha']

    rf = RandomForestRegressor(**RF_PARAMS)
    r2, rmse = evaluate_model(rf, X_train, y_train, X_test, y_test)
    baseline_scores[crop] = r2

    print(f"{crop:<15} Baseline R² = {r2:.3f} (Expected: {BASELINE_RESULTS[crop]['expected']:.2f})")


print("EXPERIMENT 2: Adding Fertilizer Features")
fert_scores = {}
for crop in ['Wheat', 'Winter_Barley', 'Spring_Barley', 'Oats', 'OSR']:
    data = get_data(crop, with_fertilizer=True)
    features = CROP_FEATURES[crop] + FERTILIZER_FEATURES

    train, test = train_test_split_temporal(data)
    X_train, y_train = train[features], train['Yield_t_per_ha']
    X_test, y_test = test[features], test['Yield_t_per_ha']

    rf = RandomForestRegressor(**RF_PARAMS)
    r2, rmse = evaluate_model(rf, X_train, y_train, X_test, y_test)
    fert_scores[crop] = r2

    change = r2 - baseline_scores[crop]
    symbol = "↑" if change > 0.01 else "↓" if change < -0.01 else "→"
    print(f"{crop:<15} R² = {r2:.3f} ({symbol} {change:+.3f} from baseline)")

print("EXPERIMENT 3: Hyperparameter Tuning (Random Forest)")


tuned_scores = {}
best_params = {}

for crop in ['Wheat', 'Winter_Barley', 'Spring_Barley', 'Oats', 'OSR']:
    data = get_data(crop, with_fertilizer=False)
    features = CROP_FEATURES[crop]

    train, test = train_test_split_temporal(data)
    X_train, y_train = train[features], train['Yield_t_per_ha']
    X_test, y_test = test[features], test['Yield_t_per_ha']

    # Grid search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 8, 10],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [1, 2, 3]
    }

    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    tuned_scores[crop] = r2
    best_params[crop] = grid_search.best_params_

    change = r2 - baseline_scores[crop]
    symbol = "↑" if change > 0.01 else "↓" if change < -0.01 else "→"
    print(f"{crop:<15} R² = {r2:.3f} ({symbol} {change:+.3f}) Best: max_depth={grid_search.best_params_['max_depth']}, n_est={grid_search.best_params_['n_estimators']}")
    
print("EXPERIMENT 4: Alternative Algorithms")
algorithms = {
    'Ridge': Ridge(alpha=1.0),
    'GradientBoost': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
}

alt_scores = {alg: {} for alg in algorithms}

for crop in ['Wheat', 'Winter_Barley', 'Spring_Barley', 'Oats', 'OSR']:
    data = get_data(crop, with_fertilizer=False)
    features = CROP_FEATURES[crop]

    train, test = train_test_split_temporal(data)
    X_train, y_train = train[features], train['Yield_t_per_ha']
    X_test, y_test = test[features], test['Yield_t_per_ha']

    # Scale features for linear models
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print(f"\n{crop}:")
    for alg_name, model in algorithms.items():
        if alg_name in ['Ridge', 'ElasticNet']:
            r2, rmse = evaluate_model(model, X_train_sc, y_train, X_test_sc, y_test)
        else:
            r2, rmse = evaluate_model(model, X_train, y_train, X_test, y_test)

        alt_scores[alg_name][crop] = r2
        change = r2 - baseline_scores[crop]
        symbol = "↑" if change > 0.01 else "↓" if change < -0.01 else "→"
        print(f"  {alg_name:<15} R² = {r2:.3f} ({symbol} {change:+.3f})")

print("EXPERIMENT 5: Combined Best (Tuned RF + Fertilizer)")

combined_scores = {}
for crop in ['Wheat', 'Winter_Barley', 'Spring_Barley', 'Oats', 'OSR']:
    data = get_data(crop, with_fertilizer=True)
    features = CROP_FEATURES[crop] + FERTILIZER_FEATURES

    train, test = train_test_split_temporal(data)
    X_train, y_train = train[features], train['Yield_t_per_ha']
    X_test, y_test = test[features], test['Yield_t_per_ha']

    # Use tuned params if available
    params = best_params.get(crop, RF_PARAMS)
    params['random_state'] = 42

    rf = RandomForestRegressor(**params)
    r2, rmse = evaluate_model(rf, X_train, y_train, X_test, y_test)
    combined_scores[crop] = r2

    change = r2 - baseline_scores[crop]
    symbol = "↑" if change > 0.01 else "↓" if change < -0.01 else "→"
    print(f"{crop:<15} R² = {r2:.3f} ({symbol} {change:+.3f} from baseline)")

print("FINAL SUMMARY: Best R² per Crop")

print(f"\n{'Crop':<15} {'Baseline':<10} {'Fertilizer':<12} {'Tuned':<10} {'Combined':<10} {'Best':<10} {'Improvement'}")

for crop in ['Wheat', 'Winter_Barley', 'Spring_Barley', 'Oats', 'OSR']:
    base = baseline_scores[crop]
    fert = fert_scores[crop]
    tuned = tuned_scores[crop]
    combined = combined_scores[crop]

    best = max(base, fert, tuned, combined)
    improvement = best - base

    best_method = 'Baseline'
    if best == fert:
        best_method = 'Fertilizer'
    elif best == tuned:
        best_method = 'Tuned'
    elif best == combined:
        best_method = 'Combined'

    print(f"{crop:<15} {base:<10.3f} {fert:<12.3f} {tuned:<10.3f} {combined:<10.3f} {best:<10.3f} {improvement:+.3f} ({best_method})")

