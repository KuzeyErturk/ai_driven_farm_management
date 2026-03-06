"""
Domain Adaptation for Cross-Country Crop Yield Transfer
=========================================================
Implements domain adaptation techniques to improve cross-country transfer:

1. Mixed-Effects Models — random country intercepts + slopes via statsmodels MixedLM
2. Quantile Mapping / Bias Correction — remove country baseline yield differences
3. CORAL Domain Adaptation — align feature distributions across countries

Compares all approaches against the pooled Ridge baseline (Is_France/Is_Germany).

Usage:
    python src/france/domain_adaptation.py
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneOut
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PATHS, YEAR_START, YEAR_END

sys.path.insert(0, PATHS['models'])
from baseline_model_config import CROP_FEATURES

# ============================================================================
# CONFIG
# ============================================================================

CROPS = list(CROP_FEATURES.keys())

CROP_NAME_IN_DATA = {
    'Wheat': 'Wheat', 'Winter_Barley': 'Winter_Barley',
    'Spring_Barley': 'Spring_Barley', 'Oats': 'Oats', 'OSR': 'Oilseed_Rape',
}

ALL_WEATHER_FEATURES = [
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

# Key weather variables for random slopes (most important for yield)
KEY_WEATHER_FOR_SLOPES = [
    'Summer_Tmax', 'Spring_Rain', 'Summer_Rain',
    'Grain_Filling_Temp', 'Summer_GDD',
]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_datasets():
    """Load all required datasets."""
    data = {}

    # UK
    data['uk_regional'] = pd.read_csv(os.path.join(
        PATHS['uk_processed'], 'regional_crop_yield_weather_2004_2024.csv'))
    data['uk_spring'] = pd.read_csv(os.path.join(
        PATHS['uk_processed'], 'spring_barley_with_weather.csv'))
    data['uk_winter'] = pd.read_csv(os.path.join(
        PATHS['uk_processed'], 'winter_barley_with_weather.csv'))

    # France
    data['france_regional'] = pd.read_csv(os.path.join(
        PATHS['france_processed'], 'france_regional_crop_yield_weather_2004_2018.csv'))
    data['france_spring'] = pd.read_csv(os.path.join(
        PATHS['france_processed'], 'france_spring_barley_with_weather.csv'))
    data['france_winter'] = pd.read_csv(os.path.join(
        PATHS['france_processed'], 'france_winter_barley_with_weather.csv'))

    # Germany
    data['germany_regional'] = pd.read_csv(os.path.join(
        PATHS['germany_processed'], 'germany_regional_crop_yield_weather_2004_2018.csv'))
    data['germany_spring'] = pd.read_csv(os.path.join(
        PATHS['germany_processed'], 'germany_spring_barley_with_weather.csv'))
    data['germany_winter'] = pd.read_csv(os.path.join(
        PATHS['germany_processed'], 'germany_winter_barley_with_weather.csv'))

    # Pooled
    data['pooled_regional'] = pd.read_csv(os.path.join(
        PATHS['pooled'], 'pooled_regional_crop_yield_weather_2004_2018.csv'))
    data['pooled_spring'] = pd.read_csv(os.path.join(
        PATHS['pooled'], 'pooled_spring_barley_with_weather.csv'))
    data['pooled_winter'] = pd.read_csv(os.path.join(
        PATHS['pooled'], 'pooled_winter_barley_with_weather.csv'))

    return data


def get_crop_data(datasets, crop, country):
    """Get data for a specific crop and country."""
    if crop == 'Spring_Barley':
        key = 'spring'
    elif crop == 'Winter_Barley':
        key = 'winter'
    else:
        key = 'regional'

    df = datasets[f'{country}_{key}']

    if key == 'regional':
        crop_in_data = CROP_NAME_IN_DATA.get(crop, crop)
        df = df[df['Crop'] == crop_in_data].copy()

    df = df[(df['Year'] >= YEAR_START) & (df['Year'] <= YEAR_END)]
    return df


def get_pooled_df(datasets, crop):
    """Get pooled DataFrame for a crop."""
    if crop == 'Spring_Barley':
        return datasets['pooled_spring'].copy()
    elif crop == 'Winter_Barley':
        return datasets['pooled_winter'].copy()
    else:
        pooled = datasets['pooled_regional'].copy()
        crop_in_data = CROP_NAME_IN_DATA.get(crop, crop)
        return pooled[pooled['Crop'] == crop_in_data].copy()


def prepare_pooled_features(pooled_df):
    """Prepare feature matrix from pooled data. Returns df with clean features + metadata."""
    features = ['Area_hectares'] + [f for f in ALL_WEATHER_FEATURES if f in pooled_df.columns]
    available = [f for f in features if pooled_df[f].notna().any() and pooled_df[f].std() > 0.001]

    keep_cols = ['Year', 'Region', 'Country', 'Yield_t_per_ha'] + available
    df = pooled_df[keep_cols].dropna().copy()
    return df, available


# ============================================================================
# BASELINE: POOLED RIDGE WITH BINARY INDICATORS
# ============================================================================

def loocv_ridge_baseline(df, features):
    """Pooled Ridge with Is_France/Is_Germany indicators (current approach)."""
    df = df.copy()
    df['Is_France'] = (df['Country'] == 'France').astype(int)
    df['Is_Germany'] = (df['Country'] == 'Germany').astype(int)
    feat_cols = features + ['Is_France', 'Is_Germany']

    X = df[feat_cols].values
    y = df['Yield_t_per_ha'].values

    loo = LeaveOneOut()
    y_pred = np.zeros(len(y))
    for tr, te in loo.split(X):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])
        model = Ridge(alpha=10.0)
        model.fit(X_tr, y[tr])
        y_pred[te] = model.predict(X_te)

    return r2_score(y, y_pred), np.sqrt(mean_squared_error(y, y_pred)), y_pred, df['Country'].values


# ============================================================================
# APPROACH 1: MIXED-EFFECTS MODELS
# ============================================================================

def loocv_mixed_effects_intercept(df, features):
    """
    Mixed-effects model: fixed weather slopes + random country intercepts.
    Uses statsmodels MixedLM with Country as group variable.
    """
    X_cols = features
    df = df.copy().reset_index(drop=True)

    # Scale features for numerical stability
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(df[X_cols]),
        columns=X_cols
    )
    X_scaled['Country'] = df['Country'].values
    X_scaled['Yield_t_per_ha'] = df['Yield_t_per_ha'].values

    y = X_scaled['Yield_t_per_ha'].values
    n = len(y)
    y_pred = np.zeros(n)

    loo = LeaveOneOut()
    failed = 0

    for tr, te in loo.split(np.arange(n)):
        train = X_scaled.iloc[tr]
        test = X_scaled.iloc[te]

        try:
            formula_parts = ' + '.join(X_cols)
            # MixedLM: Yield ~ weather features (fixed) + (1 | Country)
            model = MixedLM(
                endog=train['Yield_t_per_ha'],
                exog=sm.add_constant(train[X_cols]),
                groups=train['Country'],
            )
            result = model.fit(reml=True, maxiter=200)

            # Predict: fixed effects only (country RE not available for LOO test point
            # from the same dataset, but the RE is estimated from training data)
            X_test = sm.add_constant(test[X_cols])
            y_pred[te] = result.predict(X_test)

            # Add random effect for the test country if available
            test_country = test['Country'].values[0]
            if test_country in result.random_effects:
                y_pred[te] += result.random_effects[test_country].values[0]

        except Exception:
            # Fallback to Ridge if MixedLM fails
            sc = StandardScaler()
            X_tr = sc.fit_transform(train[X_cols].values)
            X_te = sc.transform(test[X_cols].values)
            ridge = Ridge(alpha=10.0)
            ridge.fit(X_tr, train['Yield_t_per_ha'].values)
            y_pred[te] = ridge.predict(X_te)
            failed += 1

    if failed > 0:
        print(f"    (MixedLM fallback on {failed}/{n} folds)")

    return r2_score(y, y_pred), np.sqrt(mean_squared_error(y, y_pred)), y_pred, df['Country'].values


def loocv_mixed_effects_slopes(df, features):
    """
    Mixed-effects model: fixed weather slopes + random country intercepts AND slopes.
    Random slopes on key weather variables (Summer_Tmax, Spring_Rain, etc.)
    Each country gets its own temperature-yield curve, rainfall-yield curve, etc.
    """
    X_cols = features
    df = df.copy().reset_index(drop=True)

    # Identify which key slope variables are in our features
    slope_vars = [v for v in KEY_WEATHER_FOR_SLOPES if v in X_cols]
    if not slope_vars:
        # Fallback: use first 3 features as slope vars
        slope_vars = X_cols[:3]

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(df[X_cols]),
        columns=X_cols
    )
    X_scaled['Country'] = df['Country'].values
    X_scaled['Yield_t_per_ha'] = df['Yield_t_per_ha'].values

    y = X_scaled['Yield_t_per_ha'].values
    n = len(y)
    y_pred = np.zeros(n)

    loo = LeaveOneOut()
    failed = 0

    for tr, te in loo.split(np.arange(n)):
        train = X_scaled.iloc[tr]
        test = X_scaled.iloc[te]

        try:
            # Random effects design matrix: intercept + key weather slopes per country
            re_formula = np.column_stack([
                np.ones(len(train)),
                train[slope_vars].values
            ])

            model = MixedLM(
                endog=train['Yield_t_per_ha'],
                exog=sm.add_constant(train[X_cols]),
                groups=train['Country'],
                exog_re=re_formula,
            )
            result = model.fit(reml=True, maxiter=200)

            # Fixed effects prediction
            X_test = sm.add_constant(test[X_cols])
            y_pred[te] = result.predict(X_test)

            # Add random effects for test country
            test_country = test['Country'].values[0]
            if test_country in result.random_effects:
                re_vals = result.random_effects[test_country].values
                # Random intercept
                y_pred[te] += re_vals[0]
                # Random slopes
                for j, sv in enumerate(slope_vars):
                    if j + 1 < len(re_vals):
                        y_pred[te] += re_vals[j + 1] * test[sv].values[0]

        except Exception:
            # Fallback to random intercept only
            try:
                model = MixedLM(
                    endog=train['Yield_t_per_ha'],
                    exog=sm.add_constant(train[X_cols]),
                    groups=train['Country'],
                )
                result = model.fit(reml=True, maxiter=200)
                X_test = sm.add_constant(test[X_cols])
                y_pred[te] = result.predict(X_test)
                test_country = test['Country'].values[0]
                if test_country in result.random_effects:
                    y_pred[te] += result.random_effects[test_country].values[0]
            except Exception:
                sc = StandardScaler()
                X_tr = sc.fit_transform(train[X_cols].values)
                X_te = sc.transform(test[X_cols].values)
                ridge = Ridge(alpha=10.0)
                ridge.fit(X_tr, train['Yield_t_per_ha'].values)
                y_pred[te] = ridge.predict(X_te)
                failed += 1

    if failed > 0:
        print(f"    (MixedLM slopes fallback on {failed}/{n} folds)")

    return r2_score(y, y_pred), np.sqrt(mean_squared_error(y, y_pred)), y_pred, df['Country'].values


# ============================================================================
# APPROACH 2: QUANTILE MAPPING / BIAS CORRECTION
# ============================================================================

def loocv_bias_corrected(df, features):
    """
    Bias correction: remove per-country mean yield before training,
    add it back after prediction. Uses Ridge on residuals.
    """
    df = df.copy().reset_index(drop=True)
    X = df[features].values
    y = df['Yield_t_per_ha'].values
    countries = df['Country'].values

    n = len(y)
    y_pred = np.zeros(n)
    loo = LeaveOneOut()

    for tr, te in loo.split(X):
        # Compute per-country mean yield from training data
        country_means = {}
        for c in np.unique(countries[tr]):
            mask = countries[tr] == c
            country_means[c] = y[tr][mask].mean()

        # Remove country bias from training yields
        y_train_corrected = np.array([
            y[tr][i] - country_means[countries[tr][i]]
            for i in range(len(tr))
        ])

        # Scale and train Ridge on residuals
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])

        model = Ridge(alpha=10.0)
        model.fit(X_tr, y_train_corrected)

        # Predict residual, add back country mean
        test_country = countries[te[0]]
        residual_pred = model.predict(X_te)
        y_pred[te] = residual_pred + country_means.get(test_country, np.mean(list(country_means.values())))

    return r2_score(y, y_pred), np.sqrt(mean_squared_error(y, y_pred)), y_pred, countries


# ============================================================================
# APPROACH 3: CORAL DOMAIN ADAPTATION
# ============================================================================

def coral_transform(X_source, X_target):
    """
    CORAL: CORrelation ALignment.
    Align source feature covariance to target feature covariance.
    """
    # Source and target covariance
    Cs = np.cov(X_source, rowvar=False) + np.eye(X_source.shape[1]) * 1e-6
    Ct = np.cov(X_target, rowvar=False) + np.eye(X_target.shape[1]) * 1e-6

    # Whitening source then coloring with target
    # X_aligned = X_source @ Cs^{-1/2} @ Ct^{1/2}
    Ds, Vs = np.linalg.eigh(Cs)
    Ds = np.maximum(Ds, 1e-6)
    Cs_neg_half = Vs @ np.diag(1.0 / np.sqrt(Ds)) @ Vs.T

    Dt, Vt = np.linalg.eigh(Ct)
    Dt = np.maximum(Dt, 1e-6)
    Ct_half = Vt @ np.diag(np.sqrt(Dt)) @ Vt.T

    X_aligned = (X_source - X_source.mean(axis=0)) @ Cs_neg_half @ Ct_half + X_target.mean(axis=0)
    return X_aligned


def loocv_coral(df, features):
    """
    CORAL domain adaptation: align non-UK features to UK distribution,
    then train pooled Ridge.
    """
    df = df.copy().reset_index(drop=True)
    X = df[features].values
    y = df['Yield_t_per_ha'].values
    countries = df['Country'].values

    n = len(y)
    y_pred = np.zeros(n)
    loo = LeaveOneOut()

    for tr, te in loo.split(X):
        # Get UK data indices in training set
        uk_mask = countries[tr] == 'UK'

        if uk_mask.sum() < 5:
            # Not enough UK data, fall back to regular Ridge
            sc = StandardScaler()
            X_tr = sc.fit_transform(X[tr])
            X_te = sc.transform(X[te])
            model = Ridge(alpha=10.0)
            model.fit(X_tr, y[tr])
            y_pred[te] = model.predict(X_te)
            continue

        X_uk = X[tr][uk_mask]

        # CORAL-align non-UK training data to UK distribution
        X_train_aligned = X[tr].copy()
        for country in np.unique(countries[tr]):
            if country == 'UK':
                continue
            c_mask = countries[tr] == country
            if c_mask.sum() < 5:
                continue
            X_train_aligned[c_mask] = coral_transform(X[tr][c_mask], X_uk)

        # Align test point based on its country
        test_country = countries[te[0]]
        X_test_point = X[te].copy()
        if test_country != 'UK':
            c_mask_all = countries[tr] == test_country
            if c_mask_all.sum() >= 5:
                # Use training data from same country to estimate transform
                X_country_train = X[tr][c_mask_all]
                Cs = np.cov(X_country_train, rowvar=False) + np.eye(X[te].shape[1]) * 1e-6
                Ct = np.cov(X_uk, rowvar=False) + np.eye(X[te].shape[1]) * 1e-6
                Ds, Vs = np.linalg.eigh(Cs)
                Ds = np.maximum(Ds, 1e-6)
                Cs_neg_half = Vs @ np.diag(1.0 / np.sqrt(Ds)) @ Vs.T
                Dt, Vt = np.linalg.eigh(Ct)
                Dt = np.maximum(Dt, 1e-6)
                Ct_half = Vt @ np.diag(np.sqrt(Dt)) @ Vt.T
                X_test_point = (X_test_point - X_country_train.mean(axis=0)) @ Cs_neg_half @ Ct_half + X_uk.mean(axis=0)

        # Scale and train
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_train_aligned)
        X_te_sc = sc.transform(X_test_point)

        model = Ridge(alpha=10.0)
        model.fit(X_tr_sc, y[tr])
        y_pred[te] = model.predict(X_te_sc)

    return r2_score(y, y_pred), np.sqrt(mean_squared_error(y, y_pred)), y_pred, countries


# ============================================================================
# TRANSFER TEST: TRAIN ON POOLED, PREDICT UK
# ============================================================================

def transfer_to_uk(df, features, method='ridge'):
    """
    Train on non-UK data, predict UK. Tests true cross-country transfer.
    Returns R² on UK test data.
    """
    df = df.copy().reset_index(drop=True)
    uk_mask = df['Country'] == 'UK'
    non_uk = df[~uk_mask]
    uk = df[uk_mask]

    if len(uk) < 5 or len(non_uk) < 10:
        return np.nan, np.nan

    X_train = non_uk[features].values
    y_train = non_uk['Yield_t_per_ha'].values
    X_test = uk[features].values
    y_test = uk['Yield_t_per_ha'].values

    if method == 'ridge':
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_train)
        X_te = sc.transform(X_test)
        model = Ridge(alpha=10.0)
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)

    elif method == 'mixed_effects':
        # Train mixed-effects on non-UK, predict UK using fixed effects only
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=features
        )
        X_scaled['Country'] = non_uk['Country'].values
        X_scaled['Yield_t_per_ha'] = y_train

        try:
            model = MixedLM(
                endog=X_scaled['Yield_t_per_ha'],
                exog=sm.add_constant(X_scaled[features]),
                groups=X_scaled['Country'],
            )
            result = model.fit(reml=True, maxiter=200)

            X_test_sc = pd.DataFrame(
                scaler.transform(X_test),
                columns=features
            )
            y_pred = result.predict(sm.add_constant(X_test_sc))
            # No country RE for UK since it's not in training
        except Exception:
            return np.nan, np.nan

    elif method == 'bias_corrected':
        countries_train = non_uk['Country'].values
        country_means = {}
        for c in np.unique(countries_train):
            country_means[c] = y_train[countries_train == c].mean()

        y_train_corrected = np.array([
            y_train[i] - country_means[countries_train[i]]
            for i in range(len(y_train))
        ])

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_train)
        X_te = sc.transform(X_test)
        model = Ridge(alpha=10.0)
        model.fit(X_tr, y_train_corrected)

        # For UK, we don't know the country mean — use grand mean of training
        grand_mean = np.mean(list(country_means.values()))
        y_pred = model.predict(X_te) + grand_mean

    elif method == 'coral':
        # Pick France as reference (most data), align Germany to France, predict UK
        fr_mask = non_uk['Country'] == 'France'
        de_mask = non_uk['Country'] == 'Germany'

        X_train_aligned = X_train.copy()
        if fr_mask.sum() >= 5 and de_mask.sum() >= 5:
            X_train_aligned[de_mask.values] = coral_transform(
                X_train[de_mask.values], X_train[fr_mask.values]
            )

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_train_aligned)
        X_te = sc.transform(X_test)
        model = Ridge(alpha=10.0)
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)

    else:
        return np.nan, np.nan

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return r2, rmse


# ============================================================================
# PER-COUNTRY R² FROM LOOCV
# ============================================================================

def per_country_r2(y_true, y_pred, countries):
    """Compute R² for each country from LOOCV predictions."""
    results = {}
    for c in np.unique(countries):
        mask = countries == c
        if mask.sum() < 3:
            results[c] = np.nan
        else:
            results[c] = r2_score(y_true[mask], y_pred[mask])
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    import matplotlib.pyplot as plt

    print("=" * 75)
    print("DOMAIN ADAPTATION FOR CROSS-COUNTRY TRANSFER")
    print("=" * 75)

    os.makedirs(PATHS['plots'], exist_ok=True)

    print("\n  Loading datasets...")
    datasets = load_datasets()
    print("  Done.")

    # Store all results
    all_results = {}

    methods = [
        ('Ridge+Indicators', loocv_ridge_baseline),
        ('MixedLM (intercept)', loocv_mixed_effects_intercept),
        ('MixedLM (slopes)', loocv_mixed_effects_slopes),
        ('Bias Correction', loocv_bias_corrected),
        ('CORAL', loocv_coral),
    ]

    # =====================================================================
    # EXPERIMENT 1: POOLED LOOCV — all methods
    # =====================================================================
    print("\n" + "=" * 75)
    print("EXPERIMENT 1: POOLED LOOCV — DOMAIN ADAPTATION METHODS")
    print("=" * 75)

    for crop in CROPS:
        print(f"\n  --- {crop} ---")
        pooled_df = get_pooled_df(datasets, crop)
        df, features = prepare_pooled_features(pooled_df)
        n = len(df)
        print(f"  N={n}, features={len(features)}")

        all_results[crop] = {}

        for method_name, method_fn in methods:
            print(f"    {method_name}...", end=" ", flush=True)
            r2, rmse, y_pred, countries = method_fn(df, features)

            # Per-country breakdown
            y_true = df['Yield_t_per_ha'].values
            pc = per_country_r2(y_true, y_pred, countries)

            all_results[crop][method_name] = {
                'r2': r2, 'rmse': rmse,
                'per_country': pc,
                'y_pred': y_pred, 'y_true': y_true,
                'countries': countries,
            }
            uk_r2 = pc.get('UK', np.nan)
            fr_r2 = pc.get('France', np.nan)
            de_r2 = pc.get('Germany', np.nan)
            print(f"R²={r2:.3f} (UK={uk_r2:.3f}, FR={fr_r2:.3f}, DE={de_r2:.3f})")

    # =====================================================================
    # SUMMARY TABLE
    # =====================================================================
    print("\n" + "=" * 75)
    print("SUMMARY: POOLED LOOCV R² BY METHOD")
    print("=" * 75)

    method_names = [m[0] for m in methods]

    # Overall R²
    print(f"\n  {'Method':<25} ", end="")
    for crop in CROPS:
        print(f" {crop:>12}", end="")
    print(f" {'AVERAGE':>10}")
    print("  " + "-" * (25 + 13 * len(CROPS) + 10))

    for mn in method_names:
        print(f"  {mn:<25} ", end="")
        vals = []
        for crop in CROPS:
            r2 = all_results[crop][mn]['r2']
            vals.append(r2)
            print(f" {r2:>12.3f}", end="")
        print(f" {np.nanmean(vals):>10.3f}")

    # Per-country breakdown
    for country_label in ['UK', 'France', 'Germany']:
        print(f"\n  {country_label} R²:")
        print(f"  {'Method':<25} ", end="")
        for crop in CROPS:
            print(f" {crop:>12}", end="")
        print(f" {'AVERAGE':>10}")
        print("  " + "-" * (25 + 13 * len(CROPS) + 10))

        for mn in method_names:
            print(f"  {mn:<25} ", end="")
            vals = []
            for crop in CROPS:
                r2 = all_results[crop][mn]['per_country'].get(country_label, np.nan)
                vals.append(r2)
                print(f" {r2:>12.3f}", end="")
            print(f" {np.nanmean(vals):>10.3f}")

    # =====================================================================
    # EXPERIMENT 2: TRANSFER TEST (train non-UK, predict UK)
    # =====================================================================
    print("\n" + "=" * 75)
    print("EXPERIMENT 2: TRANSFER TEST — Train on FR+DE, Predict UK")
    print("=" * 75)

    transfer_methods = ['ridge', 'mixed_effects', 'bias_corrected', 'coral']
    transfer_labels = ['Ridge (plain)', 'MixedLM (FE only)', 'Bias Corrected', 'CORAL']

    print(f"\n  {'Method':<25} ", end="")
    for crop in CROPS:
        print(f" {crop:>12}", end="")
    print(f" {'AVERAGE':>10}")
    print("  " + "-" * (25 + 13 * len(CROPS) + 10))

    for tm, tl in zip(transfer_methods, transfer_labels):
        print(f"  {tl:<25} ", end="")
        vals = []
        for crop in CROPS:
            pooled_df = get_pooled_df(datasets, crop)
            df, features = prepare_pooled_features(pooled_df)
            r2, rmse = transfer_to_uk(df, features, method=tm)
            vals.append(r2)
            if np.isnan(r2):
                print(f" {'N/A':>12}", end="")
            else:
                print(f" {r2:>12.3f}", end="")
        print(f" {np.nanmean(vals):>10.3f}")

    # =====================================================================
    # PLOT: Method comparison
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Overall R² by method
    ax = axes[0]
    x = np.arange(len(CROPS))
    w = 0.15
    colors = ['#95a5a6', '#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

    for i, mn in enumerate(method_names):
        vals = [all_results[crop][mn]['r2'] for crop in CROPS]
        ax.bar(x + (i - 2) * w, vals, w, label=mn, color=colors[i])

    ax.set_xlabel('Crop')
    ax.set_ylabel('LOOCV R²')
    ax.set_title('Pooled LOOCV R² by Domain Adaptation Method')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in CROPS])
    ax.legend(fontsize=8, loc='lower right')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Plot 2: UK-specific R² by method
    ax = axes[1]
    for i, mn in enumerate(method_names):
        vals = [all_results[crop][mn]['per_country'].get('UK', np.nan) for crop in CROPS]
        ax.bar(x + (i - 2) * w, vals, w, label=mn, color=colors[i])

    ax.set_xlabel('Crop')
    ax.set_ylabel('UK LOOCV R²')
    ax.set_title('UK-Specific R² from Pooled Models')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in CROPS])
    ax.legend(fontsize=8, loc='lower right')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(PATHS['plots'], 'domain_adaptation_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: plots/domain_adaptation_comparison.png")

    # =====================================================================
    # KEY FINDINGS
    # =====================================================================
    print("\n" + "=" * 75)
    print("KEY FINDINGS")
    print("=" * 75)

    # Find best method per metric
    avg_overall = {mn: np.nanmean([all_results[c][mn]['r2'] for c in CROPS]) for mn in method_names}
    avg_uk = {mn: np.nanmean([all_results[c][mn]['per_country'].get('UK', np.nan) for c in CROPS]) for mn in method_names}

    best_overall = max(avg_overall, key=avg_overall.get)
    best_uk = max(avg_uk, key=avg_uk.get)

    print(f"""
  Best overall pooled R²:  {best_overall} (avg R²={avg_overall[best_overall]:.3f})
  Best UK R²:              {best_uk} (avg UK R²={avg_uk[best_uk]:.3f})

  Baseline (Ridge+Indicators):
    Overall: {avg_overall['Ridge+Indicators']:.3f}
    UK:      {avg_uk['Ridge+Indicators']:.3f}

  Improvement over baseline:
    Overall: {avg_overall[best_overall] - avg_overall['Ridge+Indicators']:+.3f} ({best_overall})
    UK:      {avg_uk[best_uk] - avg_uk['Ridge+Indicators']:+.3f} ({best_uk})
""")

    print("=" * 75)
    print("DONE")
    print("=" * 75)


if __name__ == '__main__':
    main()
