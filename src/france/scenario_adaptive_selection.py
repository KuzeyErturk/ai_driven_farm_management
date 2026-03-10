"""
Scenario-Adaptive Source Country Selection
============================================
Tests whether dynamically selecting/weighting source countries based on
weather scenario similarity improves UK crop yield predictions over the
static pooled anomaly baseline (France+UK).

Experiments:
  1. Sample-Level KNN: select k nearest training samples per UK test point
  2. Country-Level Weighted Blending: RBF-weighted Ridge by country similarity
  3. Adaptive Country Subset Selection: rank & select top-N countries per fold
  4. Per-Country Model Ensemble: blend per-group Ridge models by similarity
  5. Static Ablation: fixed country subsets to establish best static baseline

Usage:
    python src/france/scenario_adaptive_selection.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneOut
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PATHS, YEAR_START, YEAR_END

sys.path.insert(0, PATHS['models'])
from baseline_model_config import CROP_FEATURES

from long_history_transfer import (
    detrend_yields, standardise_weather_anomalies, select_features,
    ALL_WEATHER_FEATURES, CROP_NAME_IN_DATA,
)

# ============================================================================
# CONFIG
# ============================================================================

CROPS = list(CROP_FEATURES.keys())

# Region → Country mapping
REGION_TO_COUNTRY = {
    'England': 'UK', 'Scotland': 'UK', 'Wales': 'UK', 'Northern Ireland': 'UK',
    'Nord': 'France', 'Ouest': 'France', 'Est': 'France', 'Sud': 'France',
    'Nord_DE': 'Germany', 'West_DE': 'Germany', 'Ost_DE': 'Germany', 'Sued_DE': 'Germany',
    'Ireland': 'Ireland',
    'Netherlands': 'Netherlands', 'Belgium': 'Belgium', 'Denmark': 'Denmark',
}

# Maritime climate countries (similar to UK)
MARITIME_COUNTRIES = ['UK', 'France', 'Ireland', 'Netherlands', 'Belgium']

# Country groups for ensemble (Exp 4)
COUNTRY_GROUPS = {
    'France': ['France'],
    'Germany': ['Germany'],
    'Ireland': ['Ireland'],
    'Benelux_DK': ['Netherlands', 'Belgium', 'Denmark'],
}


# ============================================================================
# STEP 1: DATA LOADING & PREPARATION
# ============================================================================

def load_pooled_crop_data(crop):
    """
    Load pooled multi-country data for a crop, detrend yields,
    and standardise weather to anomalies.
    Returns ready-to-use DataFrame with Country column.
    """
    pooled_path = os.path.join(PATHS['pooled'],
                               'pooled_regional_crop_yield_weather_2004_2018.csv')

    crop_in_data = CROP_NAME_IN_DATA.get(crop, crop)

    if crop in ('Spring_Barley', 'Winter_Barley'):
        # Barley has separate pooled files (no Country column, use Region)
        barley_type = 'spring' if crop == 'Spring_Barley' else 'winter'
        path = os.path.join(PATHS['pooled'],
                            f'pooled_{barley_type}_barley_with_weather.csv')
        df = pd.read_csv(path)
        # Infer Country from Region
        df['Country'] = df['Region'].map(REGION_TO_COUNTRY)
    else:
        df = pd.read_csv(pooled_path)
        df = df[df['Crop'] == crop_in_data].copy()

    # Filter year range
    df = df[(df['Year'] >= YEAR_START) & (df['Year'] <= YEAR_END)].copy()
    df = df.dropna(subset=['Yield_t_per_ha'])

    # Ensure Country column exists
    if 'Country' not in df.columns:
        df['Country'] = df['Region'].map(REGION_TO_COUNTRY)

    df = df.dropna(subset=['Country'])

    # Detrend yields per region
    df['Crop'] = crop  # Ensure consistent crop name for detrending
    df = detrend_yields(df, method='linear')

    # Standardise weather to z-score anomalies per region
    features = select_features(df, ALL_WEATHER_FEATURES)
    df = standardise_weather_anomalies(df, features)

    return df


def group_small_countries(df):
    """Merge NL/BE/DK into 'Benelux_DK' group for ensemble experiments."""
    df = df.copy()
    df['CountryGroup'] = df['Country'].replace({
        'Netherlands': 'Benelux_DK',
        'Belgium': 'Benelux_DK',
        'Denmark': 'Benelux_DK',
    })
    return df


# ============================================================================
# STEP 2: UTILITY FUNCTIONS
# ============================================================================

def compute_sample_distances(test_point, training_data, features):
    """Compute Euclidean distances from test_point to each training row in scaled space."""
    scaler = StandardScaler()
    X_train = training_data[features].values
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(test_point[features].values.reshape(1, -1))
    dists = np.sqrt(np.sum((X_train_sc - X_test_sc) ** 2, axis=1))
    return dists


def compute_country_centroids(df, features):
    """Compute per-country mean weather vector."""
    centroids = {}
    for country in df['Country'].unique():
        c_data = df[df['Country'] == country][features]
        centroids[country] = c_data.mean().values
    return centroids


def compute_country_weights(test_point, centroids, features, sigma=None):
    """
    Compute RBF weights per country based on distance to test point.
    w_c = exp(-d_c² / σ²), where σ defaults to median distance.
    """
    scaler = StandardScaler()
    # Fit scaler on all centroids + test point for consistent scaling
    all_vecs = np.vstack([list(centroids.values()), test_point])
    scaler.fit(all_vecs)

    test_sc = scaler.transform(test_point.reshape(1, -1)).ravel()
    distances = {}
    for country, centroid in centroids.items():
        c_sc = scaler.transform(centroid.reshape(1, -1)).ravel()
        distances[country] = np.sqrt(np.sum((c_sc - test_sc) ** 2))

    if sigma is None:
        sigma = np.median(list(distances.values()))
        if sigma < 0.01:
            sigma = 1.0

    weights = {}
    for country, d in distances.items():
        weights[country] = np.exp(-(d ** 2) / (sigma ** 2))

    return weights


# ============================================================================
# EXPERIMENT 5: STATIC ABLATION (baseline — run first)
# ============================================================================

def experiment_static_ablation(crop, df, features):
    """
    Run pooled anomaly LOOCV for each fixed country subset.
    Establishes which static subsets work best.
    """
    uk_mask = df['Country'] == 'UK'
    uk_df = df[uk_mask].copy()
    non_uk_df = df[~uk_mask].copy()

    if len(uk_df) < 5:
        return None

    X_uk = uk_df[features].values
    y_uk_anom = uk_df['Yield_anomaly'].values
    uk_trends = uk_df['Yield_trend'].values
    y_uk_actual = uk_df['Yield_t_per_ha'].values

    all_countries = sorted(non_uk_df['Country'].unique())

    # Define subsets to test
    subsets = {
        'UK-only': [],
        '+FR': ['France'],
        '+DE': ['Germany'],
        '+IE': ['Ireland'],
        '+FR+DE': ['France', 'Germany'],
        '+FR+IE': ['France', 'Ireland'],
        '+FR+DE+IE': ['France', 'Germany', 'Ireland'],
        '+all7': all_countries,
        '+Maritime': [c for c in MARITIME_COUNTRIES if c != 'UK' and c in all_countries],
    }

    results = {}
    for name, source_countries in subsets.items():
        # Get source data
        if source_countries:
            source_mask = non_uk_df['Country'].isin(source_countries)
            source_df = non_uk_df[source_mask]
            X_source = source_df[features].values
            y_source_anom = source_df['Yield_anomaly'].values
        else:
            X_source = np.empty((0, len(features)))
            y_source_anom = np.empty(0)

        # LOOCV over UK
        loo = LeaveOneOut()
        uk_pred = np.zeros(len(y_uk_anom))

        for tr, te in loo.split(X_uk):
            X_train = np.vstack([X_source, X_uk[tr]]) if len(X_source) > 0 else X_uk[tr]
            y_train = np.concatenate([y_source_anom, y_uk_anom[tr]]) if len(y_source_anom) > 0 else y_uk_anom[tr]

            sc = StandardScaler()
            X_train_sc = sc.fit_transform(X_train)
            X_test_sc = sc.transform(X_uk[te])

            model = Ridge(alpha=1.0)
            model.fit(X_train_sc, y_train)
            uk_pred[te] = model.predict(X_test_sc) + uk_trends[te]

        r2 = r2_score(y_uk_actual, uk_pred)
        rmse = np.sqrt(mean_squared_error(y_uk_actual, uk_pred))
        n_source = len(X_source)
        results[name] = {'r2': r2, 'rmse': rmse, 'n_source': n_source}

    return results


# ============================================================================
# EXPERIMENT 1: SAMPLE-LEVEL KNN
# ============================================================================

def experiment_knn_selection(crop, df, features, k_values=None):
    """
    Per UK test point, select k nearest training samples (from all countries),
    fit Ridge, predict. Tests whether local neighbourhood beats global pooling.
    """
    if k_values is None:
        k_values = [20, 30, 50, 100]

    uk_mask = df['Country'] == 'UK'
    uk_df = df[uk_mask].copy()
    non_uk_df = df[~uk_mask].copy()

    if len(uk_df) < 5:
        return None

    X_uk = uk_df[features].values
    y_uk_anom = uk_df['Yield_anomaly'].values
    uk_trends = uk_df['Yield_trend'].values
    y_uk_actual = uk_df['Yield_t_per_ha'].values

    X_source = non_uk_df[features].values
    y_source_anom = non_uk_df['Yield_anomaly'].values
    source_countries = non_uk_df['Country'].values

    # Add "all" as a k value (= standard pooled baseline)
    k_values_ext = k_values + [len(X_source) + len(X_uk) - 1]

    results = {}
    for k in k_values_ext:
        k_label = f'k={k}' if k < len(X_source) + len(X_uk) else 'k=all'
        loo = LeaveOneOut()
        uk_pred = np.zeros(len(y_uk_anom))
        neighbour_countries = []  # Track which countries neighbours come from

        for tr, te in loo.split(X_uk):
            # Pool: all source + UK-train
            X_pool = np.vstack([X_source, X_uk[tr]])
            y_pool = np.concatenate([y_source_anom, y_uk_anom[tr]])
            pool_countries = np.concatenate([source_countries,
                                             np.array(['UK'] * len(tr))])

            # Scale and compute distances
            sc = StandardScaler()
            X_pool_sc = sc.fit_transform(X_pool)
            X_test_sc = sc.transform(X_uk[te])

            dists = np.sqrt(np.sum((X_pool_sc - X_test_sc) ** 2, axis=1))

            # Select k nearest
            actual_k = min(k, len(X_pool))
            knn_idx = np.argsort(dists)[:actual_k]

            X_train_k = X_pool_sc[knn_idx]
            y_train_k = y_pool[knn_idx]

            model = Ridge(alpha=1.0)
            model.fit(X_train_k, y_train_k)
            uk_pred[te] = model.predict(X_test_sc) + uk_trends[te]

            # Track neighbour composition
            fold_countries = Counter(pool_countries[knn_idx])
            neighbour_countries.append(fold_countries)

        r2 = r2_score(y_uk_actual, uk_pred)
        rmse = np.sqrt(mean_squared_error(y_uk_actual, uk_pred))

        # Aggregate neighbour composition across folds
        total_counts = Counter()
        for nc in neighbour_countries:
            total_counts.update(nc)
        total_n = sum(total_counts.values())
        composition = {c: round(n / total_n * 100, 1) for c, n in total_counts.most_common()}

        results[k_label] = {'r2': r2, 'rmse': rmse, 'composition': composition}

    return results


# ============================================================================
# EXPERIMENT 2: COUNTRY-LEVEL WEIGHTED BLENDING
# ============================================================================

def experiment_weighted_blending(crop, df, features):
    """
    Per UK test point, compute country-level similarity weights,
    then fit weighted Ridge. Two variants: 4-var climate summary vs full features.
    """
    climate_vars = [v for v in ['Summer_Tmax', 'Annual_Rain', 'Winter_Tmin', 'Spring_Rain']
                    if v in features]

    uk_mask = df['Country'] == 'UK'
    uk_df = df[uk_mask].copy()
    non_uk_df = df[~uk_mask].copy()

    if len(uk_df) < 5:
        return None

    X_uk = uk_df[features].values
    y_uk_anom = uk_df['Yield_anomaly'].values
    uk_trends = uk_df['Yield_trend'].values
    y_uk_actual = uk_df['Yield_t_per_ha'].values

    results = {}
    for variant_name, dist_features in [('4-var climate', climate_vars),
                                         ('full features', features)]:
        if not dist_features:
            continue

        loo = LeaveOneOut()
        uk_pred = np.zeros(len(y_uk_anom))

        for tr, te in loo.split(X_uk):
            # Compute country centroids from non-UK + UK-train
            train_df = pd.concat([non_uk_df, uk_df.iloc[tr]])
            centroids = compute_country_centroids(train_df, dist_features)

            # Test point weather vector
            test_weather = uk_df.iloc[te[0]][dist_features].values.astype(float)

            # Get country weights
            weights = compute_country_weights(test_weather, centroids, dist_features)

            # Build sample weights: each sample gets its country's weight
            X_train_all = np.vstack([non_uk_df[features].values, X_uk[tr]])
            y_train_all = np.concatenate([non_uk_df['Yield_anomaly'].values, y_uk_anom[tr]])
            countries_all = np.concatenate([non_uk_df['Country'].values,
                                           np.array(['UK'] * len(tr))])
            sample_weights = np.array([weights.get(c, 0.5) for c in countries_all])

            sc = StandardScaler()
            X_train_sc = sc.fit_transform(X_train_all)
            X_test_sc = sc.transform(X_uk[te])

            model = Ridge(alpha=1.0)
            model.fit(X_train_sc, y_train_all, sample_weight=sample_weights)
            uk_pred[te] = model.predict(X_test_sc) + uk_trends[te]

        r2 = r2_score(y_uk_actual, uk_pred)
        rmse = np.sqrt(mean_squared_error(y_uk_actual, uk_pred))
        results[variant_name] = {'r2': r2, 'rmse': rmse}

    return results


# ============================================================================
# EXPERIMENT 3: ADAPTIVE COUNTRY SUBSET SELECTION
# ============================================================================

def experiment_adaptive_subset(crop, df, features):
    """
    Per UK test point, rank countries by similarity, select top-N, pool + fit Ridge.
    """
    climate_vars = [v for v in ['Summer_Tmax', 'Annual_Rain', 'Winter_Tmin', 'Spring_Rain']
                    if v in features]
    if not climate_vars:
        climate_vars = features[:4]

    uk_mask = df['Country'] == 'UK'
    uk_df = df[uk_mask].copy()
    non_uk_df = df[~uk_mask].copy()

    if len(uk_df) < 5:
        return None

    X_uk = uk_df[features].values
    y_uk_anom = uk_df['Yield_anomaly'].values
    uk_trends = uk_df['Yield_trend'].values
    y_uk_actual = uk_df['Yield_t_per_ha'].values

    non_uk_countries = sorted(non_uk_df['Country'].unique())
    n_values = [1, 2, 3, min(6, len(non_uk_countries))]

    results = {}
    for n in n_values:
        loo = LeaveOneOut()
        uk_pred = np.zeros(len(y_uk_anom))
        selection_counts = Counter()  # Track which countries get selected

        for tr, te in loo.split(X_uk):
            # Compute centroids from non-UK data
            centroids = compute_country_centroids(non_uk_df, climate_vars)

            # Test point weather
            test_weather = uk_df.iloc[te[0]][climate_vars].values.astype(float)

            # Rank countries by distance
            weights = compute_country_weights(test_weather, centroids, climate_vars)
            ranked = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            selected = [c for c, _ in ranked[:n]]
            selection_counts.update(selected)

            # Pool selected countries + UK-train
            source_mask = non_uk_df['Country'].isin(selected)
            X_source = non_uk_df[source_mask][features].values
            y_source = non_uk_df[source_mask]['Yield_anomaly'].values

            X_train = np.vstack([X_source, X_uk[tr]])
            y_train = np.concatenate([y_source, y_uk_anom[tr]])

            sc = StandardScaler()
            X_train_sc = sc.fit_transform(X_train)
            X_test_sc = sc.transform(X_uk[te])

            model = Ridge(alpha=1.0)
            model.fit(X_train_sc, y_train)
            uk_pred[te] = model.predict(X_test_sc) + uk_trends[te]

        r2 = r2_score(y_uk_actual, uk_pred)
        rmse = np.sqrt(mean_squared_error(y_uk_actual, uk_pred))

        # Normalise selection frequency
        total = sum(selection_counts.values())
        freq = {c: round(count / total * 100, 1)
                for c, count in selection_counts.most_common()}

        results[f'top-{n}'] = {'r2': r2, 'rmse': rmse, 'selection_freq': freq}

    return results


# ============================================================================
# EXPERIMENT 4: PER-COUNTRY MODEL ENSEMBLE
# ============================================================================

def experiment_ensemble(crop, df, features):
    """
    Train separate Ridge models per country group (France, Germany, Ireland,
    Benelux_DK), blend predictions by similarity weight.
    """
    df = group_small_countries(df)

    uk_mask = df['Country'] == 'UK'
    uk_df = df[uk_mask].copy()
    non_uk_df = df[~uk_mask].copy()

    if len(uk_df) < 5:
        return None

    X_uk = uk_df[features].values
    y_uk_anom = uk_df['Yield_anomaly'].values
    uk_trends = uk_df['Yield_trend'].values
    y_uk_actual = uk_df['Yield_t_per_ha'].values

    # Climate vars for weighting
    climate_vars = [v for v in ['Summer_Tmax', 'Annual_Rain', 'Winter_Tmin', 'Spring_Rain']
                    if v in features]
    if not climate_vars:
        climate_vars = features[:4]

    # Check which groups have enough data
    groups = {}
    for gname, countries in COUNTRY_GROUPS.items():
        group_data = non_uk_df[non_uk_df['Country'].isin(countries)]
        if len(group_data) >= 5:
            groups[gname] = group_data

    if len(groups) < 2:
        return None

    loo = LeaveOneOut()
    uk_pred = np.zeros(len(y_uk_anom))

    for tr, te in loo.split(X_uk):
        test_weather = uk_df.iloc[te[0]][climate_vars].values.astype(float)

        # Compute group centroids for weighting
        group_centroids = {}
        for gname, gdata in groups.items():
            group_centroids[gname] = gdata[climate_vars].mean().values

        # Add UK-train as a "group" too
        uk_train_centroid = uk_df.iloc[tr][climate_vars].mean().values
        group_centroids['UK_train'] = uk_train_centroid

        # Get weights
        weights = compute_country_weights(test_weather, group_centroids, climate_vars)

        # Train per-group models + UK-train model
        predictions = {}
        for gname, gdata in groups.items():
            X_g = np.vstack([gdata[features].values, X_uk[tr]])
            y_g = np.concatenate([gdata['Yield_anomaly'].values, y_uk_anom[tr]])

            sc = StandardScaler()
            X_g_sc = sc.fit_transform(X_g)
            X_te_sc = sc.transform(X_uk[te])

            model = Ridge(alpha=1.0)
            model.fit(X_g_sc, y_g)
            predictions[gname] = model.predict(X_te_sc)[0]

        # UK-only model
        if len(tr) >= 3:
            sc = StandardScaler()
            X_uk_tr_sc = sc.fit_transform(X_uk[tr])
            X_te_sc = sc.transform(X_uk[te])
            model = Ridge(alpha=1.0)
            model.fit(X_uk_tr_sc, y_uk_anom[tr])
            predictions['UK_train'] = model.predict(X_te_sc)[0]

        # Weighted average
        total_weight = sum(weights.get(g, 0.1) for g in predictions)
        blended = sum(weights.get(g, 0.1) * p for g, p in predictions.items()) / total_weight

        uk_pred[te] = blended + uk_trends[te]

    r2 = r2_score(y_uk_actual, uk_pred)
    rmse = np.sqrt(mean_squared_error(y_uk_actual, uk_pred))

    return {'r2': r2, 'rmse': rmse, 'n_groups': len(groups)}


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("SCENARIO-ADAPTIVE SOURCE COUNTRY SELECTION")
    print("=" * 70)

    all_results = {}

    for crop in CROPS:
        print(f"\n{'='*70}")
        print(f"CROP: {crop}")
        print(f"{'='*70}")

        # Load and prepare data
        df = load_pooled_crop_data(crop)
        features = select_features(df, ALL_WEATHER_FEATURES)

        # Drop rows with NaN in features or yield anomaly
        df = df.dropna(subset=features + ['Yield_anomaly', 'Yield_trend', 'Yield_t_per_ha'])

        uk_n = (df['Country'] == 'UK').sum()
        countries = df['Country'].value_counts()
        print(f"\n  Data: {len(df)} rows, {len(features)} features")
        print(f"  Countries: {dict(countries)}")

        if uk_n < 5:
            print(f"  Skipping: only {uk_n} UK rows")
            continue

        crop_results = {'crop': crop, 'n_uk': uk_n, 'n_total': len(df)}

        # --- Experiment 5: Static Ablation (run first as baseline) ---
        print(f"\n  [Exp 5] Static Ablation...")
        ablation = experiment_static_ablation(crop, df, features)
        if ablation:
            crop_results['ablation'] = ablation
            print(f"    {'Subset':<20} {'R²':>8} {'RMSE':>8} {'N source':>10}")
            print(f"    {'-'*48}")
            for name, res in sorted(ablation.items(), key=lambda x: x[1]['r2'], reverse=True):
                print(f"    {name:<20} {res['r2']:>8.3f} {res['rmse']:>8.3f} {res['n_source']:>10}")

        # --- Experiment 1: Sample-Level KNN ---
        print(f"\n  [Exp 1] Sample-Level KNN...")
        knn = experiment_knn_selection(crop, df, features)
        if knn:
            crop_results['knn'] = knn
            print(f"    {'k':<10} {'R²':>8} {'RMSE':>8}  Top-3 neighbours")
            print(f"    {'-'*60}")
            for k_label, res in knn.items():
                top3 = ', '.join(f"{c}:{p}%" for c, p in list(res['composition'].items())[:3])
                print(f"    {k_label:<10} {res['r2']:>8.3f} {res['rmse']:>8.3f}  {top3}")

        # --- Experiment 2: Country-Level Weighted Blending ---
        print(f"\n  [Exp 2] Country-Level Weighted Blending...")
        weighted = experiment_weighted_blending(crop, df, features)
        if weighted:
            crop_results['weighted'] = weighted
            for vname, res in weighted.items():
                print(f"    {vname:<20} R²={res['r2']:.3f}, RMSE={res['rmse']:.3f}")

        # --- Experiment 3: Adaptive Country Subset Selection ---
        print(f"\n  [Exp 3] Adaptive Country Subset Selection...")
        adaptive = experiment_adaptive_subset(crop, df, features)
        if adaptive:
            crop_results['adaptive'] = adaptive
            print(f"    {'N':<10} {'R²':>8} {'RMSE':>8}  Selection frequency")
            print(f"    {'-'*65}")
            for n_label, res in adaptive.items():
                freq = ', '.join(f"{c}:{p}%" for c, p in list(res['selection_freq'].items())[:4])
                print(f"    {n_label:<10} {res['r2']:>8.3f} {res['rmse']:>8.3f}  {freq}")

        # --- Experiment 4: Per-Country Model Ensemble ---
        print(f"\n  [Exp 4] Per-Country Model Ensemble...")
        ensemble = experiment_ensemble(crop, df, features)
        if ensemble:
            crop_results['ensemble'] = ensemble
            print(f"    R²={ensemble['r2']:.3f}, RMSE={ensemble['rmse']:.3f} "
                  f"({ensemble['n_groups']} groups)")

        all_results[crop] = crop_results

    # ========================================================================
    # SUMMARY TABLES
    # ========================================================================
    print("\n\n" + "=" * 70)
    print("SUMMARY TABLES")
    print("=" * 70)

    # --- Table 1: All adaptive methods vs baselines ---
    print(f"\n  Table 1: All Methods vs Baselines (UK LOOCV R²)")
    print(f"  {'Crop':<16} {'UK-only':>8} {'Pool(FR)':>8} {'Pool7':>8} "
          f"{'KNN-30':>8} {'Wt-4var':>8} {'Adpt-2':>8} {'Ensemb':>8}")
    print(f"  {'-'*72}")

    for crop in CROPS:
        if crop not in all_results:
            continue
        r = all_results[crop]
        abl = r.get('ablation', {})
        uk_only = abl.get('UK-only', {}).get('r2', float('nan'))
        pool_fr = abl.get('+FR', {}).get('r2', float('nan'))
        pool_all = abl.get('+all7', {}).get('r2', float('nan'))
        knn_30 = r.get('knn', {}).get('k=30', {}).get('r2', float('nan'))
        wt_4var = r.get('weighted', {}).get('4-var climate', {}).get('r2', float('nan'))
        adpt_2 = r.get('adaptive', {}).get('top-2', {}).get('r2', float('nan'))
        ens = r.get('ensemble', {}).get('r2', float('nan'))

        print(f"  {crop:<16} {uk_only:>8.3f} {pool_fr:>8.3f} {pool_all:>8.3f} "
              f"{knn_30:>8.3f} {wt_4var:>8.3f} {adpt_2:>8.3f} {ens:>8.3f}")

    # --- Table 2: Static ablation ---
    print(f"\n  Table 2: Static Ablation — Best Subsets (UK LOOCV R²)")
    subsets_order = ['UK-only', '+FR', '+DE', '+IE', '+FR+DE', '+FR+IE',
                     '+FR+DE+IE', '+all7', '+Maritime']
    header = f"  {'Subset':<14}" + ''.join(f'{c:>14}' for c in CROPS)
    print(header)
    print(f"  {'-'*len(header)}")
    for subset in subsets_order:
        vals = []
        for crop in CROPS:
            if crop in all_results and 'ablation' in all_results[crop]:
                v = all_results[crop]['ablation'].get(subset, {}).get('r2', float('nan'))
            else:
                v = float('nan')
            vals.append(v)
        row = f"  {subset:<14}" + ''.join(f'{v:>14.3f}' for v in vals)
        print(row)

    # --- Table 3: KNN neighbour composition ---
    print(f"\n  Table 3: KNN Neighbour Composition (k=30)")
    for crop in CROPS:
        if crop not in all_results:
            continue
        knn = all_results[crop].get('knn', {}).get('k=30', {})
        comp = knn.get('composition', {})
        if comp:
            print(f"  {crop}: {comp}")

    # --- Table 4: Country selection frequency (Exp 3, top-2) ---
    print(f"\n  Table 4: Country Selection Frequency (Adaptive, top-2)")
    for crop in CROPS:
        if crop not in all_results:
            continue
        adpt = all_results[crop].get('adaptive', {}).get('top-2', {})
        freq = adpt.get('selection_freq', {})
        if freq:
            print(f"  {crop}: {freq}")

    # --- Known baselines for comparison ---
    print(f"\n  Known Baselines (from previous experiments):")
    print(f"    UK-only Ridge LOOCV:     Wheat=0.284, W.Barley=0.238, "
          f"S.Barley=0.221, Oats=0.248, OSR=0.196")
    print(f"    Pooled Anomaly (FR+UK):  Wheat=0.696, W.Barley=0.675, "
          f"S.Barley=0.764, Oats=0.478, OSR=0.585")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
