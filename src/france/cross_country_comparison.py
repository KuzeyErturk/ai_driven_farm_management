"""
Cross-Country Comparison: UK vs France vs Germany Crop Yield Prediction
=========================================================================
Runs 5 experiments comparing crop yield models across three countries.

Experiments:
    1. Descriptive comparison — yield/weather distributions across countries
    2. Transfer test — Train on one country, predict another
    3. Country-only models — LOOCV on each country's data separately
    4. Pooled models — LOOCV on combined data (with/without Country feature)
    5. Feature importance comparison — Which weather variables matter where?

Usage:
    python src/france/cross_country_comparison.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneOut
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PATHS, YEAR_START, YEAR_END

sys.path.insert(0, PATHS['models'])
from baseline_model_config import CROP_FEATURES, BASELINE_RESULTS, IMPROVED_RESULTS


# ============================================================================
# CONFIG
# ============================================================================

CROPS = list(CROP_FEATURES.keys())  # Wheat, Winter_Barley, Spring_Barley, Oats, OSR

BEST_MODELS = {
    'Wheat':         lambda: RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_split=3, random_state=42),
    'Winter_Barley': lambda: RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_split=3, random_state=42),
    'Spring_Barley': lambda: Ridge(alpha=10.0),
    'Oats':          lambda: RandomForestRegressor(n_estimators=200, max_depth=3, min_samples_split=5, random_state=42),
    'OSR':           lambda: SVR(kernel='linear', C=1.0),
}

# Ridge with all 47 features — best overall from feature_selection.py
RIDGE_MODEL = lambda: Ridge(alpha=10.0)

BASELINE_RF = lambda: RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_split=3, random_state=42)

# All 47 weather features
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

CROP_NAME_IN_DATA = {
    'Wheat': 'Wheat', 'Winter_Barley': 'Winter_Barley',
    'Spring_Barley': 'Spring_Barley', 'Oats': 'Oats', 'OSR': 'Oilseed_Rape',
}

COUNTRIES = ['UK', 'France', 'Germany']
COUNTRY_COLORS = {'UK': '#3498db', 'France': '#e74c3c', 'Germany': '#f39c12'}


# ============================================================================
# HELPERS
# ============================================================================

def loocv_evaluate(model_fn, X, y):
    """LOOCV with per-fold scaling."""
    if len(y) < 5:
        return np.nan, np.nan, np.zeros(len(y))
    loo = LeaveOneOut()
    y_pred = np.zeros(len(y))
    for tr, te in loo.split(X):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])
        model = model_fn()
        model.fit(X_tr, y[tr])
        y_pred[te] = model.predict(X_te)
    return r2_score(y, y_pred), np.sqrt(mean_squared_error(y, y_pred)), y_pred


def train_predict(model_fn, X_train, y_train, X_test):
    """Train on one set, predict another."""
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_train)
    X_te = sc.transform(X_test)
    model = model_fn()
    model.fit(X_tr, y_train)
    return model.predict(X_te)


def load_datasets():
    """Load all required datasets."""
    data = {}

    # UK
    uk_reg = pd.read_csv(os.path.join(PATHS['uk_processed'],
                                      'regional_crop_yield_weather_2004_2024.csv'))
    data['uk_regional'] = uk_reg
    data['uk_spring'] = pd.read_csv(os.path.join(PATHS['uk_processed'],
                                                  'spring_barley_with_weather.csv'))
    data['uk_winter'] = pd.read_csv(os.path.join(PATHS['uk_processed'],
                                                  'winter_barley_with_weather.csv'))

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

    # Truncate to overlap period
    df = df[(df['Year'] >= YEAR_START) & (df['Year'] <= YEAR_END)]
    return df


def _get_pooled_df(datasets, crop):
    """Get pooled DataFrame for a crop."""
    if crop == 'Spring_Barley':
        return datasets['pooled_spring'].copy()
    elif crop == 'Winter_Barley':
        return datasets['pooled_winter'].copy()
    else:
        pooled = datasets['pooled_regional'].copy()
        crop_in_data = CROP_NAME_IN_DATA.get(crop, crop)
        return pooled[pooled['Crop'] == crop_in_data].copy()


def get_features_target(df, crop, use_all=False):
    """Extract X, y for a crop. Drops features that are all NaN."""
    if use_all:
        features = ['Area_hectares'] + [f for f in ALL_WEATHER_FEATURES if f in df.columns]
    else:
        features = [f for f in CROP_FEATURES[crop] if f in df.columns]
    # Drop features that are all NaN or have no variance
    available = [f for f in features if df[f].notna().any() and df[f].std() > 0.001]
    X = df[available].values
    y = df['Yield_t_per_ha'].values
    # Drop rows with any NaN in features
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    return X[mask], y[mask], available


# ============================================================================
# EXPERIMENT 1: DESCRIPTIVE COMPARISON
# ============================================================================

def experiment_1(datasets):
    print("\n" + "=" * 75)
    print("EXPERIMENT 1: DESCRIPTIVE COMPARISON (UK vs France vs Germany)")
    print("=" * 75)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, crop in enumerate(CROPS):
        ax = axes[idx // 3, idx % 3]
        box_data, labels = [], []
        for country in ['uk', 'france', 'germany']:
            df = get_crop_data(datasets, crop, country)
            yields = df['Yield_t_per_ha'].dropna()
            box_data.append(yields)
            labels.append({'uk': 'UK', 'france': 'FR', 'germany': 'DE'}[country])

            country_label = labels[-1]
            print(f"  {crop:<16} {country_label}: mean={yields.mean():.2f}, std={yields.std():.2f}, n={len(yields)}")

        bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
        for j, color in enumerate(['#3498db', '#e74c3c', '#f39c12']):
            bp['boxes'][j].set_facecolor(color)
        ax.set_title(crop.replace('_', ' '))
        ax.set_ylabel('Yield (t/ha)')

    if len(CROPS) < 6:
        axes[1, 2].set_visible(False)

    fig.suptitle(f'Crop Yield Distributions by Country ({YEAR_START}-{YEAR_END})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PATHS['plots'], 'france_uk_yield_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Weather comparison
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    weather_vars = [
        ('Summer_Tmax', 'Summer Max Temp (C)'),
        ('Annual_Rain', 'Annual Rainfall (mm)'),
        ('Summer_Sun', 'Summer Sunshine (hours)'),
        ('Winter_Frost', 'Winter Frost Days'),
    ]

    for idx, (var, label) in enumerate(weather_vars):
        ax = axes2[idx // 2, idx % 2]
        box_data, labels = [], []
        for country in ['uk', 'france', 'germany']:
            df = get_crop_data(datasets, 'Wheat', country)
            if var in df.columns:
                vals = df[var].dropna()
                box_data.append(vals)
                labels.append({'uk': 'UK', 'france': 'FR', 'germany': 'DE'}[country])
                print(f"  {label:<25} {labels[-1]}: {vals.mean():.1f} +/- {vals.std():.1f}")
        if box_data:
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
            for j, color in enumerate(['#3498db', '#e74c3c', '#f39c12'][:len(box_data)]):
                bp['boxes'][j].set_facecolor(color)
        ax.set_title(label)

    fig2.suptitle(f'Weather Comparison by Country ({YEAR_START}-{YEAR_END})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PATHS['plots'], 'france_uk_weather_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plots saved.")


# ============================================================================
# EXPERIMENT 2: TRANSFER TEST
# ============================================================================

def experiment_2(datasets):
    print("\n" + "=" * 75)
    print("EXPERIMENT 2: TRANSFER TEST")
    print("=" * 75)

    pairs = [('uk', 'france'), ('uk', 'germany'), ('france', 'germany')]
    transfer_results = {}

    for crop in CROPS:
        transfer_results[crop] = {}
        for src, tgt in pairs:
            src_data = get_crop_data(datasets, crop, src)
            tgt_data = get_crop_data(datasets, crop, tgt)

            # Use intersection of available features across both countries
            _, _, src_feats = get_features_target(src_data, crop)
            _, _, tgt_feats = get_features_target(tgt_data, crop)
            common_feats = [f for f in src_feats if f in tgt_feats]

            if len(common_feats) == 0:
                transfer_results[crop][f'{src}_to_{tgt}'] = np.nan
                continue

            X_src = src_data[common_feats].dropna().values
            y_src = src_data.loc[src_data[common_feats].dropna().index, 'Yield_t_per_ha'].values
            X_tgt = tgt_data[common_feats].dropna().values
            y_tgt = tgt_data.loc[tgt_data[common_feats].dropna().index, 'Yield_t_per_ha'].values

            if len(y_src) < 5 or len(y_tgt) < 5:
                transfer_results[crop][f'{src}_to_{tgt}'] = np.nan
                continue

            y_pred = train_predict(BEST_MODELS[crop], X_src, y_src, X_tgt)
            r2 = r2_score(y_tgt, y_pred)
            transfer_results[crop][f'{src}_to_{tgt}'] = r2

    # Print table
    print(f"\n  {'Crop':<16}", end="")
    for src, tgt in pairs:
        label = f"{src[:2].upper()}>{tgt[:2].upper()}"
        print(f" {label:>10}", end="")
    print()
    print("  " + "-" * 50)

    for crop in CROPS:
        print(f"  {crop:<16}", end="")
        for src, tgt in pairs:
            r2 = transfer_results[crop][f'{src}_to_{tgt}']
            if np.isnan(r2):
                print(f" {'N/A':>10}", end="")
            else:
                print(f" {r2:>10.3f}", end="")
        print()

    return transfer_results


# ============================================================================
# EXPERIMENT 3: COUNTRY-ONLY MODELS
# ============================================================================

def experiment_3(datasets):
    print("\n" + "=" * 75)
    print("EXPERIMENT 3: COUNTRY-ONLY MODELS (LOOCV)")
    print("=" * 75)

    print(f"\n  {'Crop':<16}", end="")
    for country in ['uk', 'france', 'germany']:
        label = {'uk': 'UK', 'france': 'France', 'germany': 'Germany'}[country]
        print(f" {label:>10}", end="")
    print(f" {'Model':<15}")
    print("  " + "-" * 60)

    country_results = {}

    for crop in CROPS:
        country_results[crop] = {}
        row = f"  {crop:<16}"

        for country in ['uk', 'france', 'germany']:
            df = get_crop_data(datasets, crop, country)
            X, y, _ = get_features_target(df, crop)
            r2, rmse, pred = loocv_evaluate(BEST_MODELS[crop], X, y)
            country_results[crop][country] = {'r2': r2, 'n': len(y)}
            if np.isnan(r2):
                row += f" {'N/A':>10}"
            else:
                row += f" {r2:>10.3f}"

        model_name = type(BEST_MODELS[crop]()).__name__[:12]
        row += f" {model_name:<15}"
        print(row)

    # Averages
    print("  " + "-" * 60)
    avg_row = f"  {'AVERAGE':<16}"
    for country in ['uk', 'france', 'germany']:
        vals = [country_results[c][country]['r2'] for c in CROPS
                if not np.isnan(country_results[c][country]['r2'])]
        avg_row += f" {np.mean(vals):>10.3f}" if vals else f" {'N/A':>10}"
    print(avg_row)

    return country_results


# ============================================================================
# EXPERIMENT 4: POOLED MODELS
# ============================================================================

def experiment_4(datasets):
    print("\n" + "=" * 75)
    print("EXPERIMENT 4: POOLED MODELS — ORIGINAL vs RIDGE-ALL-47")
    print("=" * 75)

    pooled_results = {}

    # --- Part A: Original features + original models ---
    print(f"\n  A) ORIGINAL FEATURES (hand-picked for UK)")
    print(f"  {'Crop':<16} {'N':>4} {'UK':>8} {'FR':>8} {'DE':>8} {'Pool+C':>8}")
    print("  " + "-" * 55)

    for crop in CROPS:
        per_country_orig = {}
        for country in ['uk', 'france', 'germany']:
            df = get_crop_data(datasets, crop, country)
            X, y, _ = get_features_target(df, crop, use_all=False)
            r2, _, _ = loocv_evaluate(BEST_MODELS[crop], X, y)
            per_country_orig[country] = r2

        # Pooled with Country
        pooled_df = _get_pooled_df(datasets, crop)
        X_p, y_p, feats = get_features_target(pooled_df, crop, use_all=False)
        pooled_c = pooled_df.dropna(subset=feats + ['Yield_t_per_ha'])
        pooled_c = pooled_c.copy()
        pooled_c['Is_France'] = (pooled_c['Country'] == 'France').astype(int)
        pooled_c['Is_Germany'] = (pooled_c['Country'] == 'Germany').astype(int)
        feats_c = feats + ['Is_France', 'Is_Germany']
        X_pc = pooled_c[feats_c].values
        y_pc = pooled_c['Yield_t_per_ha'].values
        mask = ~np.isnan(X_pc).any(axis=1) & ~np.isnan(y_pc)
        orig_pooled_r2, _, _ = loocv_evaluate(BEST_MODELS[crop], X_pc[mask], y_pc[mask])

        print(f"  {crop:<16} {len(y_p):>4} {per_country_orig['uk']:>8.3f} "
              f"{per_country_orig['france']:>8.3f} {per_country_orig['germany']:>8.3f} "
              f"{orig_pooled_r2:>8.3f}")

        pooled_results[crop] = {
            'n': len(y_p),
            'orig_uk': per_country_orig['uk'],
            'orig_france': per_country_orig['france'],
            'orig_germany': per_country_orig['germany'],
            'orig_pooled_country': orig_pooled_r2,
        }

    # --- Part B: All 47 features + Ridge ---
    print(f"\n  B) ALL 47 FEATURES + RIDGE (α=10)")
    print(f"  {'Crop':<16} {'N':>4} {'UK':>8} {'FR':>8} {'DE':>8} {'Pool+C':>8}")
    print("  " + "-" * 55)

    for crop in CROPS:
        per_country_ridge = {}
        for country in ['uk', 'france', 'germany']:
            df = get_crop_data(datasets, crop, country)
            X, y, _ = get_features_target(df, crop, use_all=True)
            r2, _, _ = loocv_evaluate(RIDGE_MODEL, X, y)
            per_country_ridge[country] = r2

        # Pooled Ridge with Country
        pooled_df = _get_pooled_df(datasets, crop)
        X_p, y_p, feats = get_features_target(pooled_df, crop, use_all=True)
        pooled_c = pooled_df.dropna(subset=feats + ['Yield_t_per_ha'])
        pooled_c = pooled_c.copy()
        pooled_c['Is_France'] = (pooled_c['Country'] == 'France').astype(int)
        pooled_c['Is_Germany'] = (pooled_c['Country'] == 'Germany').astype(int)
        feats_c = feats + ['Is_France', 'Is_Germany']
        X_pc = pooled_c[feats_c].values
        y_pc = pooled_c['Yield_t_per_ha'].values
        mask = ~np.isnan(X_pc).any(axis=1) & ~np.isnan(y_pc)
        ridge_pooled_r2, _, _ = loocv_evaluate(RIDGE_MODEL, X_pc[mask], y_pc[mask])

        print(f"  {crop:<16} {len(y_p):>4} {per_country_ridge['uk']:>8.3f} "
              f"{per_country_ridge['france']:>8.3f} {per_country_ridge['germany']:>8.3f} "
              f"{ridge_pooled_r2:>8.3f}")

        pooled_results[crop].update({
            'ridge_uk': per_country_ridge['uk'],
            'ridge_france': per_country_ridge['france'],
            'ridge_germany': per_country_ridge['germany'],
            'ridge_pooled_country': ridge_pooled_r2,
        })

    # Averages
    print(f"\n  AVERAGES:")
    for label, keys in [
        ('Original features', ['orig_uk', 'orig_france', 'orig_germany', 'orig_pooled_country']),
        ('Ridge all-47',      ['ridge_uk', 'ridge_france', 'ridge_germany', 'ridge_pooled_country']),
    ]:
        avgs = {k: np.nanmean([pooled_results[c][k] for c in CROPS]) for k in keys}
        print(f"    {label:<22} UK={avgs[keys[0]]:.3f}  FR={avgs[keys[1]]:.3f}  "
              f"DE={avgs[keys[2]]:.3f}  Pooled+C={avgs[keys[3]]:.3f}")

    # Plot — compare original vs Ridge
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(CROPS))
    w = 0.12

    for i, (key, label, color) in enumerate([
        ('orig_uk', 'UK (orig)', '#3498db'),
        ('orig_france', 'FR (orig)', '#e74c3c'),
        ('orig_germany', 'DE (orig)', '#f39c12'),
        ('ridge_uk', 'UK (Ridge47)', '#1a5276'),
        ('ridge_france', 'FR (Ridge47)', '#922b21'),
        ('ridge_germany', 'DE (Ridge47)', '#b7950b'),
        ('ridge_pooled_country', 'Pooled Ridge+C', '#9b59b6'),
    ]):
        vals = [pooled_results[c][key] for c in CROPS]
        ax.bar(x + (i - 3) * w, vals, w, label=label, color=color)

    ax.set_xlabel('Crop')
    ax.set_ylabel('LOOCV R²')
    ax.set_title('Cross-Country Pooled Model Comparison (LOOCV R²)')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in CROPS])
    ax.legend(fontsize=9)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PATHS['plots'], 'france_pooled_model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: plots/france_pooled_model_comparison.png")

    return pooled_results


# ============================================================================
# EXPERIMENT 5: FEATURE IMPORTANCE
# ============================================================================

def experiment_5(datasets):
    print("\n" + "=" * 75)
    print("EXPERIMENT 5: FEATURE IMPORTANCE COMPARISON")
    print("=" * 75)

    for crop in CROPS:
        features = [f for f in CROP_FEATURES[crop]]
        print(f"\n  {crop}:")
        print(f"    {'Feature':<30}", end="")
        for country in ['uk', 'france', 'germany']:
            label = {'uk': 'UK', 'france': 'FR', 'germany': 'DE'}[country]
            print(f" {label:>8}", end="")
        print()
        print("    " + "-" * 55)

        importances = {}
        for country in ['uk', 'france', 'germany']:
            df = get_crop_data(datasets, crop, country)
            avail = [f for f in features if f in df.columns]
            X = df[avail].values
            y = df['Yield_t_per_ha'].values

            if len(y) < 5:
                importances[country] = [np.nan] * len(avail)
                continue

            rf = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
            sc = StandardScaler()
            rf.fit(sc.fit_transform(X), y)
            importances[country] = rf.feature_importances_

        avail = [f for f in features if f in get_crop_data(datasets, crop, 'uk').columns]
        for i, feat in enumerate(avail):
            print(f"    {feat:<30}", end="")
            for country in ['uk', 'france', 'germany']:
                imp = importances[country]
                if i < len(imp) and not np.isnan(imp[i]):
                    print(f" {imp[i]:>8.3f}", end="")
                else:
                    print(f" {'N/A':>8}", end="")
            print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 75)
    print("CROSS-COUNTRY COMPARISON: UK vs FRANCE vs GERMANY")
    print("Crop Yield Prediction with Weather Features")
    print("=" * 75)

    os.makedirs(PATHS['plots'], exist_ok=True)

    print("\n  Loading datasets...")
    datasets = load_datasets()
    print("  All datasets loaded.")

    experiment_1(datasets)
    transfer_results = experiment_2(datasets)
    country_results = experiment_3(datasets)
    pooled_results = experiment_4(datasets)
    experiment_5(datasets)

    # Final summary
    print("\n" + "=" * 75)
    print("FINAL SUMMARY")
    print("=" * 75)

    print(f"\n  A) ORIGINAL APPROACH (UK hand-picked features, per-crop models)")
    print(f"  {'Crop':<16} | {'UK':>6} | {'FR':>6} | {'DE':>6} | {'Pool+C':>8}")
    print("  " + "-" * 50)
    for crop in CROPS:
        uk = pooled_results[crop]['orig_uk']
        fr = pooled_results[crop]['orig_france']
        de = pooled_results[crop]['orig_germany']
        p = pooled_results[crop]['orig_pooled_country']
        print(f"  {crop:<16} | {uk:>6.3f} | {fr:>6.3f} | {de:>6.3f} | {p:>8.3f}")
    avg_o = {k: np.nanmean([pooled_results[c][k] for c in CROPS])
             for k in ['orig_uk', 'orig_france', 'orig_germany', 'orig_pooled_country']}
    print("  " + "-" * 50)
    print(f"  {'AVERAGE':<16} | {avg_o['orig_uk']:>6.3f} | {avg_o['orig_france']:>6.3f} | "
          f"{avg_o['orig_germany']:>6.3f} | {avg_o['orig_pooled_country']:>8.3f}")

    print(f"\n  B) IMPROVED APPROACH (All 47 features, Ridge α=10)")
    print(f"  {'Crop':<16} | {'UK':>6} | {'FR':>6} | {'DE':>6} | {'Pool+C':>8}")
    print("  " + "-" * 50)
    for crop in CROPS:
        uk = pooled_results[crop]['ridge_uk']
        fr = pooled_results[crop]['ridge_france']
        de = pooled_results[crop]['ridge_germany']
        p = pooled_results[crop]['ridge_pooled_country']
        print(f"  {crop:<16} | {uk:>6.3f} | {fr:>6.3f} | {de:>6.3f} | {p:>8.3f}")
    avg_r = {k: np.nanmean([pooled_results[c][k] for c in CROPS])
             for k in ['ridge_uk', 'ridge_france', 'ridge_germany', 'ridge_pooled_country']}
    print("  " + "-" * 50)
    print(f"  {'AVERAGE':<16} | {avg_r['ridge_uk']:>6.3f} | {avg_r['ridge_france']:>6.3f} | "
          f"{avg_r['ridge_germany']:>6.3f} | {avg_r['ridge_pooled_country']:>8.3f}")

    print(f"""
  KEY FINDINGS:
    Original approach (UK features):
      UK={avg_o['orig_uk']:.3f}, FR={avg_o['orig_france']:.3f}, DE={avg_o['orig_germany']:.3f}, Pooled={avg_o['orig_pooled_country']:.3f}

    Improved approach (Ridge, all 47 features):
      UK={avg_r['ridge_uk']:.3f}, FR={avg_r['ridge_france']:.3f}, DE={avg_r['ridge_germany']:.3f}, Pooled={avg_r['ridge_pooled_country']:.3f}

    Improvement:
      UK:  {avg_o['orig_uk']:.3f} → {avg_r['ridge_uk']:.3f} ({avg_r['ridge_uk']-avg_o['orig_uk']:+.3f})
      FR:  {avg_o['orig_france']:.3f} → {avg_r['ridge_france']:.3f} ({avg_r['ridge_france']-avg_o['orig_france']:+.3f})
      DE:  {avg_o['orig_germany']:.3f} → {avg_r['ridge_germany']:.3f} ({avg_r['ridge_germany']-avg_o['orig_germany']:+.3f})
      Pooled: {avg_o['orig_pooled_country']:.3f} → {avg_r['ridge_pooled_country']:.3f} ({avg_r['ridge_pooled_country']-avg_o['orig_pooled_country']:+.3f})

  DATA: All REAL — yields (Schauberger 2022, OpenAgrar 2024, DEFRA) +
  weather (E-OBS v31/v32 gridded observations, 0.1deg resolution).

  Plots: {PATHS['plots']}/
""")

    print("=" * 75)
    print("DONE")
    print("=" * 75)


if __name__ == '__main__':
    main()
