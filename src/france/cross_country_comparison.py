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

BASELINE_RF = lambda: RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_split=3, random_state=42)

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


def get_features_target(df, crop):
    """Extract X, y for a crop. Drops features that are all NaN (e.g. sunshine)."""
    features = [f for f in CROP_FEATURES[crop] if f in df.columns]
    # Drop features that are all NaN or all zero (sunshine not available for France/Germany)
    available = [f for f in features if df[f].notna().any() and not (df[f] == 0).all()]
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
    print("EXPERIMENT 4: POOLED MODELS (UK + France + Germany)")
    print("=" * 75)

    print(f"\n  {'Crop':<16} {'N':>4} {'UK':>8} {'FR':>8} {'DE':>8} "
          f"{'Pooled':>8} {'Pool+C':>8} {'Model':<12}")
    print("  " + "-" * 75)

    pooled_results = {}

    for crop in CROPS:
        # Per-country LOOCV
        per_country = {}
        for country in ['uk', 'france', 'germany']:
            df = get_crop_data(datasets, crop, country)
            X, y, _ = get_features_target(df, crop)
            r2, _, _ = loocv_evaluate(BEST_MODELS[crop], X, y)
            per_country[country] = r2

        # Pooled data
        if crop == 'Spring_Barley':
            pooled_df = datasets['pooled_spring']
        elif crop == 'Winter_Barley':
            pooled_df = datasets['pooled_winter']
        else:
            pooled_df = datasets['pooled_regional']
            crop_in_data = CROP_NAME_IN_DATA.get(crop, crop)
            pooled_df = pooled_df[pooled_df['Crop'] == crop_in_data]

        X_pooled, y_pooled, features = get_features_target(pooled_df, crop)

        # Pooled without Country
        pooled_r2, _, _ = loocv_evaluate(BEST_MODELS[crop], X_pooled, y_pooled)

        # Pooled with Country as one-hot (France, Germany binary features)
        pooled_clean = pooled_df.dropna(subset=features + ['Yield_t_per_ha'])
        pooled_clean = pooled_clean[~(pooled_clean[features] == 0).all(axis=1)]
        pooled_c = pooled_clean.copy()
        pooled_c['Is_France'] = (pooled_c['Country'] == 'France').astype(int)
        pooled_c['Is_Germany'] = (pooled_c['Country'] == 'Germany').astype(int)
        features_c = features + ['Is_France', 'Is_Germany']
        X_pooled_c = pooled_c[features_c].values
        y_pooled_c = pooled_c['Yield_t_per_ha'].values
        pooled_c_r2, _, _ = loocv_evaluate(BEST_MODELS[crop], X_pooled_c, y_pooled_c)

        n = len(y_pooled)
        model_name = type(BEST_MODELS[crop]()).__name__[:12]

        uk_str = f"{per_country['uk']:>8.3f}" if not np.isnan(per_country['uk']) else f"{'N/A':>8}"
        fr_str = f"{per_country['france']:>8.3f}" if not np.isnan(per_country['france']) else f"{'N/A':>8}"
        de_str = f"{per_country['germany']:>8.3f}" if not np.isnan(per_country['germany']) else f"{'N/A':>8}"

        print(f"  {crop:<16} {n:>4} {uk_str} {fr_str} {de_str} "
              f"{pooled_r2:>8.3f} {pooled_c_r2:>8.3f} {model_name:<12}")

        pooled_results[crop] = {
            'n': n,
            'uk_r2': per_country['uk'],
            'france_r2': per_country['france'],
            'germany_r2': per_country['germany'],
            'pooled_r2': pooled_r2,
            'pooled_country_r2': pooled_c_r2,
        }

    # Averages
    print("  " + "-" * 75)
    avg = {k: np.nanmean([pooled_results[c][k] for c in CROPS])
           for k in ['uk_r2', 'france_r2', 'germany_r2', 'pooled_r2', 'pooled_country_r2']}
    print(f"  {'AVERAGE':<16} {'':>4} {avg['uk_r2']:>8.3f} {avg['france_r2']:>8.3f} "
          f"{avg['germany_r2']:>8.3f} {avg['pooled_r2']:>8.3f} {avg['pooled_country_r2']:>8.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(CROPS))
    w = 0.15

    for i, (key, label, color) in enumerate([
        ('uk_r2', 'UK Only', '#3498db'),
        ('france_r2', 'France Only', '#e74c3c'),
        ('germany_r2', 'Germany Only', '#f39c12'),
        ('pooled_r2', 'Pooled', '#2ecc71'),
        ('pooled_country_r2', 'Pooled + Country', '#9b59b6'),
    ]):
        vals = [pooled_results[c][key] for c in CROPS]
        ax.bar(x + (i - 2) * w, vals, w, label=label, color=color)

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

    print(f"\n  {'Crop':<16} | {'UK':>6} | {'FR':>6} | {'DE':>6} | {'Pooled+C':>8}")
    print("  " + "-" * 50)
    for crop in CROPS:
        uk = IMPROVED_RESULTS[crop]['loocv_r2']
        fr = country_results[crop]['france']['r2']
        de = country_results[crop]['germany']['r2']
        p = pooled_results[crop]['pooled_country_r2']
        fr_s = f"{fr:>6.3f}" if not np.isnan(fr) else f"{'N/A':>6}"
        de_s = f"{de:>6.3f}" if not np.isnan(de) else f"{'N/A':>6}"
        print(f"  {crop:<16} | {uk:>6.3f} | {fr_s} | {de_s} | {p:>8.3f}")

    avg_uk = np.mean([IMPROVED_RESULTS[c]['loocv_r2'] for c in CROPS])
    avg_fr = np.nanmean([country_results[c]['france']['r2'] for c in CROPS])
    avg_de = np.nanmean([country_results[c]['germany']['r2'] for c in CROPS])
    avg_p = np.mean([pooled_results[c]['pooled_country_r2'] for c in CROPS])
    print("  " + "-" * 50)
    print(f"  {'AVERAGE':<16} | {avg_uk:>6.3f} | {avg_fr:>6.3f} | {avg_de:>6.3f} | {avg_p:>8.3f}")

    print(f"""
  KEY FINDINGS:
    1. UK models (n=84, 2004-2024):     avg LOOCV R² = {avg_uk:.3f}
    2. France models (n~52, real data):  avg LOOCV R² = {avg_fr:.3f}
    3. Germany models (n~52, real data): avg LOOCV R² = {avg_de:.3f}
    4. Pooled 3-country + Country feat:  avg LOOCV R² = {avg_p:.3f}

  DATA: All REAL — yields (Schauberger 2022, OpenAgrar 2024, DEFRA) +
  weather (E-OBS v32 gridded observations, 0.1deg resolution).
  Sunshine features unavailable (E-OBS radiation requires CDS registration).

  Plots: {PATHS['plots']}/
""")

    print("=" * 75)
    print("DONE")
    print("=" * 75)


if __name__ == '__main__':
    main()
