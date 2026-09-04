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

# Loading the data
def load_datasets():
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

    # Pooled ( both france + uk + germany)
    data['pooled_regional'] = pd.read_csv(os.path.join(
        PATHS['pooled'], 'pooled_regional_crop_yield_weather_2004_2018.csv'))
    data['pooled_spring'] = pd.read_csv(os.path.join(
        PATHS['pooled'], 'pooled_spring_barley_with_weather.csv'))
    data['pooled_winter'] = pd.read_csv(os.path.join(
        PATHS['pooled'], 'pooled_winter_barley_with_weather.csv'))

    return data


def get_crop_data(datasets, crop, country):
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
    if crop == 'Spring_Barley':
        return datasets['pooled_spring'].copy()
    elif crop == 'Winter_Barley':
        return datasets['pooled_winter'].copy()
    else:
        pooled = datasets['pooled_regional'].copy()
        crop_in_data = CROP_NAME_IN_DATA.get(crop, crop)
        return pooled[pooled['Crop'] == crop_in_data].copy()


def prepare_pooled_features(pooled_df):
    features = ['Area_hectares'] + [f for f in ALL_WEATHER_FEATURES if f in pooled_df.columns]
    available = [f for f in features if pooled_df[f].notna().any() and pooled_df[f].std() > 0.001]

    keep_cols = ['Year', 'Region', 'Country', 'Yield_t_per_ha'] + available
    df = pooled_df[keep_cols].dropna().copy()
    return df, available

def loocv_ridge_baseline(df, features):
    df = df.copy()
    # Create binary indicator for every non-UK country
    non_uk_countries = sorted([c for c in df['Country'].unique() if c != 'UK'])
    for c in non_uk_countries:
        df[f'Is_{c}'] = (df['Country'] == c).astype(int)
    feat_cols = features + [f'Is_{c}' for c in non_uk_countries]

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


def _select_top_features(df, features, n_top=10):
    correlations = []
    for f in features:
        if df[f].std() > 0.001:
            corr = abs(df[f].corr(df['Yield_t_per_ha']))
            if not np.isnan(corr):
                correlations.append((f, corr))
    correlations.sort(key=lambda x: x[1], reverse=True)

    selected = []
    for f, _ in correlations:
        if len(selected) >= n_top:
            break
        too_correlated = False
        for s in selected:
            if abs(df[f].corr(df[s])) > 0.85:
                too_correlated = True
                break
        if not too_correlated:
            selected.append(f)
    return selected


def loocv_mixed_effects_intercept(df, features):
    df = df.copy().reset_index(drop=True)

    # Reduce feature set for MixedLM stability
    X_cols = _select_top_features(df, features, n_top=10)

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
            X_train_const = sm.add_constant(train[X_cols])
            model = MixedLM(
                endog=train['Yield_t_per_ha'],
                exog=X_train_const,
                groups=train['Country'],
            )
            result = model.fit(reml=True, maxiter=500)

            # Predict: ensure test has same constant column
            X_test = sm.add_constant(test[X_cols], has_constant='add')
            y_pred[te] = result.predict(X_test)

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
    df = df.copy().reset_index(drop=True)

    # Reduce feature set
    X_cols = _select_top_features(df, features, n_top=10)

    # Identify which key slope variables are in our reduced features
    slope_vars = [v for v in KEY_WEATHER_FOR_SLOPES if v in X_cols]
    if not slope_vars:
        slope_vars = X_cols[:3]
    # Limit to 3 slope vars max for stability
    slope_vars = slope_vars[:3]

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
            result = model.fit(reml=True, maxiter=500)

            # Fixed effects prediction
            X_test = sm.add_constant(test[X_cols], has_constant='add')
            y_pred[te] = result.predict(X_test)

            # Add random effects for test country
            test_country = test['Country'].values[0]
            if test_country in result.random_effects:
                re_vals = result.random_effects[test_country].values
                y_pred[te] += re_vals[0]  # random intercept
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
                result = model.fit(reml=True, maxiter=500)
                X_test = sm.add_constant(test[X_cols], has_constant='add')
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

def loocv_bias_corrected(df, features):
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

def coral_transform(X_source, X_target):
    # CORAL domain adaptation (Sun et al., 2016) — whitens source, re-colours to target covariance
    Cs = np.cov(X_source, rowvar=False) + np.eye(X_source.shape[1]) * 1e-6
    Ct = np.cov(X_target, rowvar=False) + np.eye(X_target.shape[1]) * 1e-6

    Ds, Vs = np.linalg.eigh(Cs)
    Ds = np.maximum(Ds, 1e-6)
    Cs_neg_half = Vs @ np.diag(1.0 / np.sqrt(Ds)) @ Vs.T

    Dt, Vt = np.linalg.eigh(Ct)
    Dt = np.maximum(Dt, 1e-6)
    Ct_half = Vt @ np.diag(np.sqrt(Dt)) @ Vt.T

    X_aligned = (X_source - X_source.mean(axis=0)) @ Cs_neg_half @ Ct_half + X_target.mean(axis=0)
    return X_aligned


def loocv_coral(df, features):
    df = df.copy().reset_index(drop=True)
    X = df[features].values
    y = df['Yield_t_per_ha'].values
    countries = df['Country'].values
    n_features = X.shape[1]

    n = len(y)
    y_pred = np.zeros(n)
    loo = LeaveOneOut()

    # Minimum samples needed for stable covariance estimation
    min_samples = max(20, n_features + 5)

    for tr, te in loo.split(X):
        uk_mask = countries[tr] == 'UK'

        if uk_mask.sum() < min_samples:
            # Not enough UK data for stable CORAL, fall back to Ridge
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
            if c_mask.sum() < min_samples:
                continue  # Skip alignment for small countries
            try:
                X_train_aligned[c_mask] = coral_transform(X[tr][c_mask], X_uk)
            except Exception:
                pass  # Keep original if transform fails

        # Align test point
        test_country = countries[te[0]]
        X_test_point = X[te].copy()
        if test_country != 'UK':
            c_mask_all = countries[tr] == test_country
            if c_mask_all.sum() >= min_samples:
                try:
                    X_country_train = X[tr][c_mask_all]
                    Cs = np.cov(X_country_train, rowvar=False) + np.eye(n_features) * 1e-4
                    Ct = np.cov(X_uk, rowvar=False) + np.eye(n_features) * 1e-4
                    Ds, Vs = np.linalg.eigh(Cs)
                    Ds = np.maximum(Ds, 1e-4)
                    Cs_neg_half = Vs @ np.diag(1.0 / np.sqrt(Ds)) @ Vs.T
                    Dt, Vt = np.linalg.eigh(Ct)
                    Dt = np.maximum(Dt, 1e-4)
                    Ct_half = Vt @ np.diag(np.sqrt(Dt)) @ Vt.T
                    X_test_point = (X_test_point - X_country_train.mean(axis=0)) @ Cs_neg_half @ Ct_half + X_uk.mean(axis=0)
                except Exception:
                    pass

        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_train_aligned)
        X_te_sc = sc.transform(X_test_point)

        model = Ridge(alpha=10.0)
        model.fit(X_tr_sc, y[tr])
        y_pred[te] = model.predict(X_te_sc)

    return r2_score(y, y_pred), np.sqrt(mean_squared_error(y, y_pred)), y_pred, countries


def _compute_climate_weights(df, features, target='UK'):
    # Use key climate variables for distance
    climate_vars = [v for v in ['Summer_Tmax', 'Annual_Rain', 'Winter_Tmin', 'Spring_Rain']
                    if v in features]
    if not climate_vars:
        climate_vars = features[:4]

    target_data = df[df['Country'] == target]
    target_mean = target_data[climate_vars].mean().values

    # Per-country climate distance
    countries = df['Country'].unique()
    country_distances = {}
    for c in countries:
        if c == target:
            continue
        c_mean = df[df['Country'] == c][climate_vars].mean().values
        # Euclidean distance on standardised climate vars
        std = df[climate_vars].std().values
        std[std < 0.001] = 1
        country_distances[c] = np.sqrt(np.sum(((c_mean - target_mean) / std) ** 2))

    # Convert to weights: w = exp(-d^2/sigma^2), sigma = median distance
    if not country_distances:
        return np.ones(len(df))
    sigma = np.median(list(country_distances.values()))
    if sigma < 0.01:
        sigma = 1.0

    weights = np.ones(len(df))
    for i, row in df.iterrows():
        c = row['Country']
        if c == target:
            weights[i] = 1.0
        elif c in country_distances:
            weights[i] = np.exp(-(country_distances[c] ** 2) / (sigma ** 2))

    return weights


def _select_transferable_features(df, features, min_countries=5):
    countries = df['Country'].unique()
    selected = []

    for f in features:
        if df[f].std() < 0.001:
            continue
        signs = []
        magnitudes = []
        for c in countries:
            sub = df[df['Country'] == c]
            if len(sub) < 5:
                continue
            corr = sub[f].corr(sub['Yield_t_per_ha'])
            if not np.isnan(corr):
                signs.append(np.sign(corr))
                magnitudes.append(abs(corr))

        if len(signs) < min_countries:
            continue

        n_pos = sum(1 for s in signs if s > 0)
        n_neg = sum(1 for s in signs if s < 0)
        n_meaningful = sum(1 for m in magnitudes if m > 0.1)
        if (n_pos >= min_countries or n_neg >= min_countries) and n_meaningful >= 3:
            avg_mag = np.mean(magnitudes)
            selected.append((f, avg_mag))

    selected.sort(key=lambda x: x[1], reverse=True)
    result = [f for f, _ in selected]

    # Ensure at least 5 features
    if len(result) < 5:
        fallback = _select_top_features(df, features, n_top=10)
        for f in fallback:
            if f not in result:
                result.append(f)
            if len(result) >= 10:
                break

    return result


# Tranfer the test: train on non-uk data to predict the UK


def transfer_to_uk(df, features, method='ridge'):
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
        top_feats = _select_top_features(df, features, n_top=10)
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(non_uk[top_feats].values),
            columns=top_feats
        )
        X_scaled['Country'] = non_uk['Country'].values
        X_scaled['Yield_t_per_ha'] = y_train

        try:
            model = MixedLM(
                endog=X_scaled['Yield_t_per_ha'],
                exog=sm.add_constant(X_scaled[top_feats]),
                groups=X_scaled['Country'],
            )
            result = model.fit(reml=True, maxiter=500)
            X_test_sc = pd.DataFrame(
                scaler.transform(uk[top_feats].values),
                columns=top_feats
            )
            y_pred = result.predict(sm.add_constant(X_test_sc, has_constant='add'))
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

        grand_mean = np.mean(list(country_means.values()))
        y_pred = model.predict(X_te) + grand_mean

    elif method == 'coral':
        country_counts = non_uk['Country'].value_counts()
        ref_country = country_counts.index[0]
        ref_mask = non_uk['Country'] == ref_country

        X_train_aligned = X_train.copy()
        for other_country in non_uk['Country'].unique():
            if other_country == ref_country:
                continue
            c_mask = (non_uk['Country'] == other_country).values
            if c_mask.sum() >= 5 and ref_mask.sum() >= 5:
                try:
                    X_train_aligned[c_mask] = coral_transform(
                        X_train[c_mask], X_train[ref_mask.values]
                    )
                except Exception:
                    pass

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_train_aligned)
        X_te = sc.transform(X_test)
        model = Ridge(alpha=10.0)
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)


    elif method == 'anomaly':
        # Train on yield anomalies (z-scored within country), predict UK anomaly,
        # estimate UK baseline from Ireland
        countries_train = non_uk['Country'].values
        country_stats = {}
        for c in np.unique(countries_train):
            mask = countries_train == c
            country_stats[c] = {'mean': y_train[mask].mean(), 'std': max(y_train[mask].std(), 0.01)}

        # Normalise to anomalies: (yield - country_mean) / country_std
        y_anomaly = np.array([
            (y_train[i] - country_stats[countries_train[i]]['mean']) / country_stats[countries_train[i]]['std']
            for i in range(len(y_train))
        ])

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_train)
        X_te = sc.transform(X_test)
        model = Ridge(alpha=10.0)
        model.fit(X_tr, y_anomaly)

        # Predict UK anomaly
        anomaly_pred = model.predict(X_te)

        if 'Ireland' in country_stats:
            uk_mean_est = country_stats['Ireland']['mean']
            uk_std_est = country_stats['Ireland']['std']
        else:
            uk_mean_est = np.mean([s['mean'] for s in country_stats.values()])
            uk_std_est = np.mean([s['std'] for s in country_stats.values()])

        y_pred = anomaly_pred * uk_std_est + uk_mean_est

    elif method == 'anomaly_climate_weighted':
        countries_train = non_uk['Country'].values
        country_stats = {}
        for c in np.unique(countries_train):
            mask = countries_train == c
            country_stats[c] = {'mean': y_train[mask].mean(), 'std': max(y_train[mask].std(), 0.01)}

        y_anomaly = np.array([
            (y_train[i] - country_stats[countries_train[i]]['mean']) / country_stats[countries_train[i]]['std']
            for i in range(len(y_train))
        ])

        # Climate-similarity weights
        weights = _compute_climate_weights(df, features, target='UK')
        train_weights = weights[~uk_mask.values]

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_train)
        X_te = sc.transform(X_test)
        model = Ridge(alpha=10.0)
        model.fit(X_tr, y_anomaly, sample_weight=train_weights)

        anomaly_pred = model.predict(X_te)

        if 'Ireland' in country_stats:
            uk_mean_est = country_stats['Ireland']['mean']
            uk_std_est = country_stats['Ireland']['std']
        else:
            uk_mean_est = np.mean([s['mean'] for s in country_stats.values()])
            uk_std_est = np.mean([s['std'] for s in country_stats.values()])

        y_pred = anomaly_pred * uk_std_est + uk_mean_est

    elif method == 'ireland_only':
        # Transfer from Ireland only
        ie_mask = non_uk['Country'] == 'Ireland'
        ie_data = non_uk[ie_mask]
        if len(ie_data) < 5:
            return np.nan, np.nan

        X_ie = ie_data[features].values
        y_ie = ie_data['Yield_t_per_ha'].values

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_ie)
        X_te = sc.transform(X_test)
        model = Ridge(alpha=10.0)
        model.fit(X_tr, y_ie)
        y_pred = model.predict(X_te)

    elif method == 'climate_weighted':
        weights = _compute_climate_weights(df, features, target='UK')
        train_weights = weights[~uk_mask.values]

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_train)
        X_te = sc.transform(X_test)
        model = Ridge(alpha=10.0)
        model.fit(X_tr, y_train, sample_weight=train_weights)
        y_pred = model.predict(X_te)

    elif method == 'transferable_features':
        # Only use features with consistent cross-country effects
        tf = _select_transferable_features(df, features, min_countries=4)
        X_tr_tf = non_uk[tf].values
        X_te_tf = uk[tf].values

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr_tf)
        X_te = sc.transform(X_te_tf)
        model = Ridge(alpha=10.0)
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)

    elif method == 'per_country_zscore':
        # Z-score features within each country, use Ireland stats for UK
        # Use only features with enough variance in all countries
        good_feats = []
        for f_idx, f in enumerate(features):
            all_ok = True
            for c in non_uk['Country'].unique():
                c_data = non_uk[non_uk['Country'] == c][f]
                if c_data.std() < 0.01 or c_data.isna().any():
                    all_ok = False
                    break
            if all_ok:
                good_feats.append(f_idx)

        if len(good_feats) < 3:
            return np.nan, np.nan

        feat_subset = [features[i] for i in good_feats]
        X_train_sub = non_uk[feat_subset].values
        X_test_sub = uk[feat_subset].values

        country_feature_stats = {}
        for c in non_uk['Country'].unique():
            c_data = non_uk[non_uk['Country'] == c][feat_subset]
            country_feature_stats[c] = {
                'mean': c_data.mean().values,
                'std': np.maximum(c_data.std().values, 0.01)
            }

        X_tr_z = np.zeros_like(X_train_sub, dtype=float)
        for i, (_, row) in enumerate(non_uk.iterrows()):
            c = row['Country']
            X_tr_z[i] = (X_train_sub[i] - country_feature_stats[c]['mean']) / country_feature_stats[c]['std']

        if 'Ireland' in country_feature_stats:
            proxy = country_feature_stats['Ireland']
        else:
            proxy = {
                'mean': np.mean([s['mean'] for s in country_feature_stats.values()], axis=0),
                'std': np.mean([s['std'] for s in country_feature_stats.values()], axis=0)
            }
        X_te_z = (X_test_sub - proxy['mean']) / proxy['std']

        # Replace any remaining NaN/inf
        X_tr_z = np.nan_to_num(X_tr_z, nan=0.0, posinf=0.0, neginf=0.0)
        X_te_z = np.nan_to_num(X_te_z, nan=0.0, posinf=0.0, neginf=0.0)

        model = Ridge(alpha=10.0)
        model.fit(X_tr_z, y_train)
        y_pred = model.predict(X_te_z)

    elif method == 'combined':
        # combined: anomaly + climate weighting + transferable features + per-country z-score
        tf = _select_transferable_features(df, features, min_countries=4)

        countries_train = non_uk['Country'].values
        country_stats = {}
        country_feature_stats = {}
        for c in np.unique(countries_train):
            mask = countries_train == c
            country_stats[c] = {'mean': y_train[mask].mean(), 'std': max(y_train[mask].std(), 0.01)}
            c_data = non_uk[non_uk['Country'] == c][tf]
            country_feature_stats[c] = {
                'mean': c_data.mean().values,
                'std': np.maximum(c_data.std().values, 0.01)
            }

        # Yield anomalies
        y_anomaly = np.array([
            (y_train[i] - country_stats[countries_train[i]]['mean']) / country_stats[countries_train[i]]['std']
            for i in range(len(y_train))
        ])

        # Per-country z-scored features (transferable only)
        X_train_tf = non_uk[tf].values
        X_test_tf = uk[tf].values
        X_tr_z = np.zeros_like(X_train_tf, dtype=float)
        for i, (_, row) in enumerate(non_uk.iterrows()):
            c = row['Country']
            X_tr_z[i] = (X_train_tf[i] - country_feature_stats[c]['mean']) / country_feature_stats[c]['std']

        if 'Ireland' in country_feature_stats:
            proxy = country_feature_stats['Ireland']
        else:
            proxy = {
                'mean': np.mean([s['mean'] for s in country_feature_stats.values()], axis=0),
                'std': np.mean([s['std'] for s in country_feature_stats.values()], axis=0)
            }
        X_te_z = (X_test_tf - proxy['mean']) / proxy['std']

        # Clean NaN/inf
        X_tr_z = np.nan_to_num(X_tr_z, nan=0.0, posinf=0.0, neginf=0.0)
        X_te_z = np.nan_to_num(X_te_z, nan=0.0, posinf=0.0, neginf=0.0)

        # Climate weights
        weights = _compute_climate_weights(df, tf, target='UK')
        train_weights = weights[~uk_mask.values]

        model = Ridge(alpha=10.0)
        model.fit(X_tr_z, y_anomaly, sample_weight=train_weights)
        anomaly_pred = model.predict(X_te_z)

        if 'Ireland' in country_stats:
            uk_mean_est = country_stats['Ireland']['mean']
            uk_std_est = country_stats['Ireland']['std']
        else:
            uk_mean_est = np.mean([s['mean'] for s in country_stats.values()])
            uk_std_est = np.mean([s['std'] for s in country_stats.values()])

        y_pred = anomaly_pred * uk_std_est + uk_mean_est

    # If we are finetuning :
    elif method == 'finetune':
        # Use first 5 UK years as calibration, rest as test
        uk_sorted = uk.sort_values('Year')
        n_cal = min(5, len(uk_sorted) // 2)
        uk_cal = uk_sorted.iloc[:n_cal]
        uk_test = uk_sorted.iloc[n_cal:]

        if len(uk_test) < 3:
            return np.nan, np.nan

        X_test_final = uk_test[features].values
        y_test = uk_test['Yield_t_per_ha'].values

        # Pre-train on non-UK
        sc = StandardScaler()
        X_all_train = np.vstack([X_train, uk_cal[features].values])
        y_all_train = np.concatenate([y_train, uk_cal['Yield_t_per_ha'].values])

        X_tr = sc.fit_transform(X_all_train)
        X_te = sc.transform(X_test_final)
        model = Ridge(alpha=10.0)
        model.fit(X_tr, y_all_train)
        y_pred = model.predict(X_te)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return r2, rmse

    else:
        return np.nan, np.nan

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return r2, rmse


# R^2 scores for each country using LOOCV
def per_country_r2(y_true, y_pred, countries):
    results = {}
    for c in np.unique(countries):
        mask = countries == c
        if mask.sum() < 3:
            results[c] = np.nan
        else:
            results[c] = r2_score(y_true[mask], y_pred[mask])
    return results

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
    # First, check the Pooled LOOCV using all the methods
    for crop in CROPS:
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

    # Summary table

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

    all_countries = sorted(set(
        c for crop in CROPS
        for mn in method_names
        for c in all_results[crop][mn]['per_country'].keys()
    ))
    for country_label in all_countries:
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

    # Tranfer test method - train ion non-uk to predict uk

    transfer_methods = [
        'ridge', 'bias_corrected', 'mixed_effects',
        'anomaly', 'anomaly_climate_weighted',
        'ireland_only', 'climate_weighted',
        'transferable_features', 'per_country_zscore',
        'combined', 'finetune',
    ]
    transfer_labels = [
        'Ridge (plain)', 'Bias Corrected', 'MixedLM (FE only)',
        'Anomaly Model', 'Anomaly+ClimWeight',
        'Ireland Only', 'Climate Weighted',
        'Transferable Feats', 'Per-Country Z-score',
        'COMBINED', 'Fine-tune (5yr UK)',
    ]

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

    # Find best method per metric
    avg_overall = {mn: np.nanmean([all_results[c][mn]['r2'] for c in CROPS]) for mn in method_names}
    avg_uk = {mn: np.nanmean([all_results[c][mn]['per_country'].get('UK', np.nan) for c in CROPS]) for mn in method_names}

    best_overall = max(avg_overall, key=avg_overall.get)
    best_uk = max(avg_uk, key=avg_uk.get)

    # Various print statements was used to check the results and compare, also plots can be done/drawn to see and visualise some results
if __name__ == '__main__':
    main()
