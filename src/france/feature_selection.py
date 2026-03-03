"""
Per-Country Feature Selection for Cross-Country Comparison
============================================================
Tests all 47 weather features per country × crop using multiple methods:
1. Lasso (L1 regularization — automatic sparsity)
2. ElasticNet (L1+L2 — handles correlated features)
3. RF feature importance (non-linear relationships)

Then compares:
- UK-original features (from dissertation)
- Per-country optimized features
- Shared features (features important in ALL countries)

Usage:
    python src/france/feature_selection.py
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, ElasticNet, Ridge, LassoCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneOut
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PATHS, YEAR_START, YEAR_END

sys.path.insert(0, PATHS['models'])
from baseline_model_config import CROP_FEATURES, IMPROVED_RESULTS


# ============================================================================
# CONFIG
# ============================================================================

CROPS = list(CROP_FEATURES.keys())

CROP_NAME_IN_DATA = {
    'Wheat': 'Wheat', 'Winter_Barley': 'Winter_Barley',
    'Spring_Barley': 'Spring_Barley', 'Oats': 'Oats', 'OSR': 'Oilseed_Rape',
}

# All 47 weather features (excluding Area_hectares which is always included)
WEATHER_FEATURES = [
    'Winter_Tmax', 'Spring_Tmax', 'Summer_Tmax', 'Autumn_Tmax',
    'Winter_Tmin', 'Spring_Tmin', 'Summer_Tmin', 'Autumn_Tmin',
    'Spring_Temp_Mean', 'Summer_Temp_Mean',
    'Flowering_Temp', 'Grain_Filling_Temp',
    'Summer_Tmax_Peak', 'Spring_Tmin_Coldest',
    'Spring_Temp_Range', 'Summer_Temp_Range',
    'Spring_GDD', 'Summer_GDD',
    'Winter_Rain', 'Spring_Rain', 'Summer_Rain', 'Autumn_Rain',
    'Planting_Rain', 'Grain_Filling_Rain', 'Harvest_Rain',
    'Annual_Rain', 'Rain_Deviation_from_Optimal',
    'Spring_Rain_Squared', 'Summer_Rain_Squared',
    'Winter_Sun', 'Spring_Sun', 'Summer_Sun', 'Autumn_Sun',
    'Grain_Filling_Sun', 'Flowering_Sun', 'Summer_Sun_per_Rain',
    'Winter_Frost', 'Spring_Frost', 'Autumn_Frost', 'Late_Spring_Frost',
    'Spring_Temp_x_Rain', 'Summer_Temp_x_Rain', 'Summer_Temp_x_Sun',
    'Heat_Stress', 'Cold_Spring', 'Extreme_Summer_Rain', 'Drought_Spring',
]


def loocv_evaluate(model_fn, X, y):
    """LOOCV with per-fold scaling."""
    if len(y) < 10:
        return np.nan, np.nan
    loo = LeaveOneOut()
    y_pred = np.zeros(len(y))
    for tr, te in loo.split(X):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])
        model = model_fn()
        model.fit(X_tr, y[tr])
        y_pred[te] = model.predict(X_te)
    return r2_score(y, y_pred), np.sqrt(mean_squared_error(y, y_pred))


def load_data():
    """Load all country datasets."""
    uk_reg = pd.read_csv(os.path.join(PATHS['uk_processed'],
                                      'regional_crop_yield_weather_2004_2024.csv'))
    uk_spring = pd.read_csv(os.path.join(PATHS['uk_processed'],
                                         'spring_barley_with_weather.csv'))
    uk_winter = pd.read_csv(os.path.join(PATHS['uk_processed'],
                                         'winter_barley_with_weather.csv'))
    fr_reg = pd.read_csv(os.path.join(PATHS['france_processed'],
                                      'france_regional_crop_yield_weather_2004_2018.csv'))
    de_reg = pd.read_csv(os.path.join(PATHS['germany_processed'],
                                      'germany_regional_crop_yield_weather_2004_2018.csv'))
    pooled_reg = pd.read_csv(os.path.join(PATHS['pooled'],
                                          'pooled_regional_crop_yield_weather_2004_2018.csv'))
    pooled_spring = pd.read_csv(os.path.join(PATHS['pooled'],
                                             'pooled_spring_barley_with_weather.csv'))
    pooled_winter = pd.read_csv(os.path.join(PATHS['pooled'],
                                             'pooled_winter_barley_with_weather.csv'))

    return {
        'uk_regional': uk_reg, 'uk_spring': uk_spring, 'uk_winter': uk_winter,
        'france_regional': fr_reg,
        'france_spring': fr_reg[fr_reg['Crop'] == 'Spring_Barley'],
        'france_winter': fr_reg[fr_reg['Crop'] == 'Winter_Barley'],
        'germany_regional': de_reg,
        'germany_spring': de_reg[de_reg['Crop'] == 'Spring_Barley'],
        'germany_winter': de_reg[de_reg['Crop'] == 'Winter_Barley'],
        'pooled_regional': pooled_reg,
        'pooled_spring': pooled_spring,
        'pooled_winter': pooled_winter,
    }


def get_crop_df(datasets, crop, country):
    """Get data for a crop × country."""
    if crop == 'Spring_Barley':
        key = 'spring'
    elif crop == 'Winter_Barley':
        key = 'winter'
    else:
        key = 'regional'

    df = datasets[f'{country}_{key}'].copy()

    if key == 'regional':
        crop_in_data = CROP_NAME_IN_DATA.get(crop, crop)
        df = df[df['Crop'] == crop_in_data]

    df = df[(df['Year'] >= YEAR_START) & (df['Year'] <= YEAR_END)]
    return df


def get_available_features(df):
    """Get weather features that exist and have variance in the data."""
    available = []
    for f in WEATHER_FEATURES:
        if f in df.columns and df[f].notna().any() and df[f].std() > 0.001:
            available.append(f)
    return available


# ============================================================================
# FEATURE SELECTION METHODS
# ============================================================================

def lasso_feature_selection(X, y, feature_names, n_features=6):
    """Use LassoCV to find optimal features."""
    sc = StandardScaler()
    X_sc = sc.fit_transform(X)

    # Use cross-validated Lasso to find best alpha
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(X_sc, y)

    # Get non-zero coefficients
    importance = np.abs(lasso.coef_)
    top_idx = np.argsort(importance)[::-1]

    # Return top n_features
    selected = [feature_names[i] for i in top_idx[:n_features] if importance[i] > 0]

    # If Lasso killed too many, use top by absolute coefficient
    if len(selected) < 3:
        selected = [feature_names[i] for i in top_idx[:n_features]]

    return selected, {feature_names[i]: importance[i] for i in top_idx[:10]}


def rf_feature_selection(X, y, feature_names, n_features=6):
    """Use RF feature importance."""
    sc = StandardScaler()
    X_sc = sc.fit_transform(X)

    rf = RandomForestRegressor(n_estimators=200, max_depth=5,
                               min_samples_split=3, random_state=42)
    rf.fit(X_sc, y)

    importance = rf.feature_importances_
    top_idx = np.argsort(importance)[::-1]

    selected = [feature_names[i] for i in top_idx[:n_features]]
    return selected, {feature_names[i]: importance[i] for i in top_idx[:10]}


def combined_feature_selection(X, y, feature_names, n_features=6):
    """Combine Lasso + RF rankings for robust selection."""
    lasso_feats, lasso_imp = lasso_feature_selection(X, y, feature_names, n_features=15)
    rf_feats, rf_imp = rf_feature_selection(X, y, feature_names, n_features=15)

    # Score each feature: rank in Lasso + rank in RF
    all_feats = set(lasso_feats[:10]) | set(rf_feats[:10])
    scores = {}
    for f in all_feats:
        lasso_rank = lasso_feats.index(f) if f in lasso_feats else 15
        rf_rank = rf_feats.index(f) if f in rf_feats else 15
        scores[f] = lasso_rank + rf_rank  # Lower = better

    # Sort by combined rank, take top n
    sorted_feats = sorted(scores.keys(), key=lambda f: scores[f])

    # Always include Area_hectares if available
    selected = sorted_feats[:n_features]

    return selected


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    print("=" * 75)
    print("PER-COUNTRY FEATURE SELECTION (ALL 47 FEATURES)")
    print("=" * 75)

    datasets = load_data()

    # Store results
    country_features = {}  # {crop: {country: [features]}}
    all_results = {}       # {crop: {config_name: {country: r2}}}

    # Best model per crop (from dissertation)
    BEST_MODELS = {
        'Wheat':         lambda: RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_split=3, random_state=42),
        'Winter_Barley': lambda: RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_split=3, random_state=42),
        'Spring_Barley': lambda: Ridge(alpha=10.0),
        'Oats':          lambda: RandomForestRegressor(n_estimators=200, max_depth=3, min_samples_split=5, random_state=42),
        'OSR':           lambda: SVR(kernel='linear', C=1.0),
    }

    # Also test algorithms that handle many features well
    REGULARIZED_MODELS = {
        'Ridge_a10':     lambda: Ridge(alpha=10.0),
        'Lasso_a1':      lambda: Lasso(alpha=1.0, max_iter=10000),
        'ElasticNet':    lambda: ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=10000),
        'RF_d3':         lambda: RandomForestRegressor(n_estimators=200, max_depth=3, min_samples_split=5, random_state=42),
        'RF_d5':         lambda: RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_split=3, random_state=42),
        'SVR_linear':    lambda: SVR(kernel='linear', C=1.0),
    }

    for crop in CROPS:
        print(f"\n{'='*75}")
        print(f"  {crop}")
        print(f"{'='*75}")

        country_features[crop] = {}
        all_results[crop] = {}

        for country in ['uk', 'france', 'germany']:
            df = get_crop_df(datasets, crop, country)
            available = get_available_features(df)
            y = df['Yield_t_per_ha'].values

            # Add Area_hectares
            all_feats = ['Area_hectares'] + available
            all_feats = [f for f in all_feats if f in df.columns]
            X_all = df[all_feats].values

            print(f"\n  {country.upper()} (n={len(y)}, {len(all_feats)} features available)")

            # --- Feature selection ---
            selected = combined_feature_selection(X_all, y, all_feats, n_features=6)

            # Ensure Area_hectares is included
            if 'Area_hectares' in df.columns and 'Area_hectares' not in selected:
                selected = ['Area_hectares'] + selected[:5]

            country_features[crop][country] = selected
            print(f"    Selected features: {selected}")

            # --- Compare configurations ---
            configs = {
                'UK_original': [f for f in CROP_FEATURES[crop] if f in df.columns and df[f].notna().any() and df[f].std() > 0.001],
                'Country_optimized': selected,
                'All_47_Ridge': all_feats,
                'All_47_Lasso': all_feats,
            }

            for config_name, feats in configs.items():
                if len(feats) == 0:
                    continue

                X = df[feats].dropna().values
                y_clean = df.loc[df[feats].dropna().index, 'Yield_t_per_ha'].values

                if len(y_clean) < 10:
                    continue

                if config_name == 'All_47_Ridge':
                    model_fn = lambda: Ridge(alpha=10.0)
                elif config_name == 'All_47_Lasso':
                    model_fn = lambda: Lasso(alpha=1.0, max_iter=10000)
                else:
                    model_fn = BEST_MODELS[crop]

                r2, rmse = loocv_evaluate(model_fn, X, y_clean)

                key = f"{config_name}"
                if key not in all_results[crop]:
                    all_results[crop][key] = {}
                all_results[crop][key][country] = r2

                print(f"    {config_name:<25} LOOCV R²={r2:.3f}  (n_feats={len(feats)})")

        # --- Find shared features (important in at least 2 countries) ---
        feat_counts = {}
        for country in ['uk', 'france', 'germany']:
            for f in country_features[crop][country]:
                feat_counts[f] = feat_counts.get(f, 0) + 1

        shared = [f for f, c in sorted(feat_counts.items(), key=lambda x: -x[1]) if c >= 2][:6]
        print(f"\n  SHARED features (important in 2+ countries): {shared}")

        # Evaluate shared features
        for country in ['uk', 'france', 'germany']:
            df = get_crop_df(datasets, crop, country)
            avail_shared = [f for f in shared if f in df.columns and df[f].notna().any()]
            if len(avail_shared) < 2:
                continue
            X = df[avail_shared].dropna().values
            y_clean = df.loc[df[avail_shared].dropna().index, 'Yield_t_per_ha'].values
            if len(y_clean) < 10:
                continue
            r2, _ = loocv_evaluate(BEST_MODELS[crop], X, y_clean)
            if 'Shared_features' not in all_results[crop]:
                all_results[crop]['Shared_features'] = {}
            all_results[crop]['Shared_features'][country] = r2

    # ========================================================================
    # SUMMARY TABLE
    # ========================================================================
    print("\n" + "=" * 75)
    print("SUMMARY: LOOCV R² BY FEATURE CONFIGURATION")
    print("=" * 75)

    for crop in CROPS:
        print(f"\n  {crop}:")
        print(f"    {'Configuration':<25} {'UK':>8} {'France':>8} {'Germany':>8} {'Average':>8}")
        print(f"    {'-'*60}")

        for config in ['UK_original', 'Country_optimized', 'Shared_features', 'All_47_Ridge', 'All_47_Lasso']:
            if config not in all_results[crop]:
                continue
            vals = all_results[crop][config]
            uk_r2 = vals.get('uk', np.nan)
            fr_r2 = vals.get('france', np.nan)
            de_r2 = vals.get('germany', np.nan)
            avg = np.nanmean([uk_r2, fr_r2, de_r2])

            uk_s = f"{uk_r2:>8.3f}" if not np.isnan(uk_r2) else f"{'N/A':>8}"
            fr_s = f"{fr_r2:>8.3f}" if not np.isnan(fr_r2) else f"{'N/A':>8}"
            de_s = f"{de_r2:>8.3f}" if not np.isnan(de_r2) else f"{'N/A':>8}"

            print(f"    {config:<25} {uk_s} {fr_s} {de_s} {avg:>8.3f}")

    # ========================================================================
    # SELECTED FEATURES PER COUNTRY
    # ========================================================================
    print("\n" + "=" * 75)
    print("OPTIMAL FEATURES PER COUNTRY × CROP")
    print("=" * 75)

    for crop in CROPS:
        print(f"\n  {crop}:")
        print(f"    UK original:  {CROP_FEATURES[crop]}")
        for country in ['uk', 'france', 'germany']:
            feats = country_features[crop].get(country, [])
            print(f"    {country:<10} optimized: {feats}")

    # ========================================================================
    # BEST OVERALL CONFIGURATION
    # ========================================================================
    print("\n" + "=" * 75)
    print("BEST CONFIGURATION PER CROP (HIGHEST AVERAGE R²)")
    print("=" * 75)

    for crop in CROPS:
        best_config = None
        best_avg = -999
        for config, vals in all_results[crop].items():
            avg = np.nanmean(list(vals.values()))
            if avg > best_avg:
                best_avg = avg
                best_config = config
        print(f"  {crop:<16} → {best_config} (avg R²={best_avg:.3f})")

    # Save country_features for use in cross_country_comparison
    import json
    output = {}
    for crop in CROPS:
        output[crop] = {}
        for country in ['uk', 'france', 'germany']:
            output[crop][country] = country_features[crop].get(country, [])

    out_path = os.path.join(PATHS['france_processed'], '..', '..', 'pooled',
                            'optimal_features_per_country.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved optimal features: {out_path}")

    print("\n" + "=" * 75)
    print("DONE")
    print("=" * 75)


if __name__ == '__main__':
    main()
