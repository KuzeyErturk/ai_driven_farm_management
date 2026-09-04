import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'france'))

from baseline_model_config import CROP_FEATURES, DATA_SOURCES, IMPROVED_RESULTS
from config import PATHS, YEAR_START, YEAR_END

PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# All 47 weather features (from cross_country_comparison.py)
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

CROPS = list(CROP_FEATURES.keys())

BEST_MODELS = {
    'Wheat':         lambda: RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_split=3, random_state=42),
    'Winter_Barley': lambda: RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_split=3, random_state=42),
    'Spring_Barley': lambda: Ridge(alpha=10.0),
    'Oats':          lambda: RandomForestRegressor(n_estimators=200, max_depth=3, min_samples_split=5, random_state=42),
    'OSR':           lambda: SVR(kernel='linear', C=1.0),
}

CROP_COLORS = {
    'Wheat': '#2ecc71', 'Winter_Barley': '#3498db', 'Spring_Barley': '#e67e22',
    'Oats': '#9b59b6', 'OSR': '#e74c3c',
}

def load_uk_datasets():
    regional = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed',
                                         'regional_crop_yield_weather_2004_2024.csv'))
    spring_barley = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed',
                                              'spring_barley_with_weather.csv'))
    winter_barley = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed',
                                              'winter_barley_with_weather.csv'))
    return {
        'Wheat':         regional[regional['Crop'] == 'Wheat'].copy(),
        'Winter_Barley': winter_barley.copy(),
        'Spring_Barley': spring_barley.copy(),
        'Oats':          regional[regional['Crop'] == 'Oats'].copy(),
        'OSR':           regional[regional['Crop'] == 'Oilseed_Rape'].copy(),
    }


def load_pooled_datasets():
    pooled_reg = pd.read_csv(os.path.join(PATHS['pooled'],
                                           'pooled_regional_crop_yield_weather_2004_2018.csv'))
    pooled_spring = pd.read_csv(os.path.join(PATHS['pooled'],
                                              'pooled_spring_barley_with_weather.csv'))
    pooled_winter = pd.read_csv(os.path.join(PATHS['pooled'],
                                              'pooled_winter_barley_with_weather.csv'))
    return {
        'pooled_regional': pooled_reg,
        'pooled_spring': pooled_spring,
        'pooled_winter': pooled_winter,
    }


def get_pooled_crop_data(pooled_data, crop):
    if crop == 'Spring_Barley':
        return pooled_data['pooled_spring'].copy()
    elif crop == 'Winter_Barley':
        return pooled_data['pooled_winter'].copy()
    else:
        df = pooled_data['pooled_regional'].copy()
        crop_in_data = CROP_NAME_IN_DATA.get(crop, crop)
        return df[df['Crop'] == crop_in_data].copy()


def get_features_target(df, features_list):
    available = [f for f in features_list if f in df.columns
                 and df[f].notna().any() and df[f].std() > 0.001]
    X = df[available].values
    y = df['Yield_t_per_ha'].values
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    return X[mask], y[mask], available, mask

def run_uk_shap_analysis(uk_data):

    uk_shap_results = {}

    for crop in CROPS:
        print(f"\n  {crop}...")
        data = uk_data[crop]
        features = CROP_FEATURES[crop]
        X, y, avail, mask = get_features_target(data, features)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train model on full data
        model = BEST_MODELS[crop]()
        model.fit(X_scaled, y)

        # Compute SHAP values
        model_type = type(model).__name__
        if model_type == 'RandomForestRegressor':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
        elif model_type == 'Ridge':
            explainer = shap.LinearExplainer(model, X_scaled)
            shap_values = explainer.shap_values(X_scaled)
        elif model_type == 'SVR':
            explainer = shap.KernelExplainer(model.predict, X_scaled)
            shap_values = explainer.shap_values(X_scaled, nsamples=200)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Verify SHAP additivity
        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0]
        preds = model.predict(X_scaled)
        shap_sum = shap_values.sum(axis=1) + base_value
        max_error = np.max(np.abs(preds - shap_sum))
        print(f"    Model: {model_type}, Features: {len(avail)}, N={len(y)}")
        print(f"    SHAP additivity check: max |pred - (base + sum(SHAP))| = {max_error:.6f}")

        # Mean |SHAP| per feature
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        for i, feat in enumerate(avail):
            print(f"      {feat:<30} mean|SHAP|={mean_abs_shap[i]:.4f}")

        uk_shap_results[crop] = {
            'shap_values': shap_values,
            'X_scaled': X_scaled,
            'X_raw': X,
            'y': y,
            'features': avail,
            'model': model,
            'explainer': explainer,
            'base_value': base_value,
        }

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_values, X_scaled, feature_names=avail, show=False)
        plt.title(f'SHAP Summary — {crop.replace("_", " ")} (UK, {model_type})')
        plt.tight_layout()
        fname = os.path.join(PLOTS_DIR, f'shap_uk_summary_{crop.lower()}.png')
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close('all')
        print(f"    Saved: {fname}")

    _plot_uk_panel(uk_shap_results)

    return uk_shap_results


def _plot_uk_panel(uk_shap_results):
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=False)

    for idx, crop in enumerate(CROPS):
        ax = axes[idx]
        res = uk_shap_results[crop]
        mean_abs = np.mean(np.abs(res['shap_values']), axis=0)
        features = res['features']

        # Sort by importance
        order = np.argsort(mean_abs)
        ax.barh(range(len(features)), mean_abs[order], color=CROP_COLORS[crop], alpha=0.8)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels([features[i] for i in order], fontsize=8)
        ax.set_xlabel('Mean |SHAP|', fontsize=9)
        model_name = type(res['model']).__name__
        ax.set_title(f'{crop.replace("_", " ")}\n({model_name})', fontsize=10)

    fig.suptitle('UK Crop Yield Models — SHAP Feature Importance', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, 'shap_uk_panel.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"\n  Saved combined panel: {fname}")

# This is PART B of pooling the SHAP Analysis
def run_pooled_shap_analysis(pooled_data):
    """Train Ridge on pooled data with all 47 features + country indicators."""
    print("\n" + "=" * 75)
    print("PART B: POOLED CROSS-COUNTRY SHAP ANALYSIS (Ridge alpha=10)")
    print("=" * 75)

    pooled_shap_results = {}

    for crop in CROPS:
        print(f"\n  {crop}...")
        df = get_pooled_crop_data(pooled_data, crop)

        # Build feature list: Area_hectares + weather + country indicators
        weather_feats = ['Area_hectares'] + [f for f in ALL_WEATHER_FEATURES if f in df.columns]
        available = [f for f in weather_feats if df[f].notna().any() and df[f].std() > 0.001]

        # Add country indicators
        df = df.copy()
        df['Is_France'] = (df['Country'] == 'France').astype(int)
        df['Is_Germany'] = (df['Country'] == 'Germany').astype(int)
        all_feats = available + ['Is_France', 'Is_Germany']

        # Clean
        subset = df.dropna(subset=all_feats + ['Yield_t_per_ha']).copy()
        X = subset[all_feats].values
        y = subset['Yield_t_per_ha'].values
        countries = subset['Country'].values

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train the ridge
        model = Ridge(alpha=10.0)
        model.fit(X_scaled, y)

        # SHAP
        explainer = shap.LinearExplainer(model, X_scaled)
        shap_values = explainer.shap_values(X_scaled)

        # Verify
        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0]
        preds = model.predict(X_scaled)
        shap_sum = shap_values.sum(axis=1) + base_value
        max_error = np.max(np.abs(preds - shap_sum))
        print(f"    N={len(y)}, Features={len(all_feats)}")
        print(f"    SHAP additivity: max error = {max_error:.6f}")

        # Top-15 features
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:15]
        print(f"    Top 15 features:")
        for i in top_idx:
            print(f"      {all_feats[i]:<30} mean|SHAP|={mean_abs[i]:.4f}")

        # Country indicator SHAP
        is_fr_idx = all_feats.index('Is_France')
        is_de_idx = all_feats.index('Is_Germany')
        print(f"    Country SHAP — Is_France: mean={np.mean(shap_values[:, is_fr_idx]):.4f}, "
              f"Is_Germany: mean={np.mean(shap_values[:, is_de_idx]):.4f}")

        pooled_shap_results[crop] = {
            'shap_values': shap_values,
            'X_scaled': X_scaled,
            'X_raw': X,
            'y': y,
            'features': all_feats,
            'countries': countries,
            'model': model,
            'explainer': explainer,
            'base_value': base_value,
            'scaler': scaler,
            'df': subset,
        }

    # Plot: top-15 features per crop
    _plot_pooled_top_features(pooled_shap_results)

    return pooled_shap_results


def _plot_pooled_top_features(pooled_shap_results):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes_flat = axes.flatten()

    for idx, crop in enumerate(CROPS):
        ax = axes_flat[idx]
        res = pooled_shap_results[crop]
        mean_abs = np.mean(np.abs(res['shap_values']), axis=0)
        features = res['features']

        top_idx = np.argsort(mean_abs)[::-1][:15]
        top_feats = [features[i] for i in top_idx]
        top_vals = mean_abs[top_idx]

        # Color country indicators differently
        colors = []
        for f in top_feats:
            if f in ('Is_France', 'Is_Germany'):
                colors.append('#e74c3c')
            else:
                colors.append(CROP_COLORS[crop])

        ax.barh(range(len(top_feats)), top_vals[::-1], color=colors[::-1], alpha=0.8)
        ax.set_yticks(range(len(top_feats)))
        ax.set_yticklabels(top_feats[::-1], fontsize=8)
        ax.set_xlabel('Mean |SHAP|', fontsize=9)
        ax.set_title(f'{crop.replace("_", " ")}', fontsize=11, fontweight='bold')

    # Hide last subplot
    axes_flat[5].set_visible(False)

    fig.suptitle('Pooled Cross-Country Models (Ridge α=10) — Top 15 SHAP Features',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, 'shap_pooled_top_features.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"\n  Saved: {fname}")

# THis is Part C : cross-country comparison

def run_country_comparison(pooled_shap_results):

    # Pick a representative crop (Wheat) for detailed comparison
    focus_crop = 'Wheat'
    res = pooled_shap_results[focus_crop]
    shap_vals = res['shap_values']
    features = res['features']
    countries = res['countries']

    # Get top-10 features by overall mean |SHAP|
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:10]
    top_feats = [features[i] for i in top_idx]

    print(f"\n  {focus_crop} — Mean |SHAP| by country for top 10 features:")
    print(f"  {'Feature':<30} {'UK':>8} {'France':>8} {'Germany':>8}")
    print("  " + "-" * 56)

    country_shap = {}
    for country_name in ['UK', 'France', 'Germany']:
        mask = countries == country_name
        country_shap[country_name] = np.mean(np.abs(shap_vals[mask]), axis=0)

    for i in top_idx:
        print(f"  {features[i]:<30} {country_shap['UK'][i]:>8.4f} "
              f"{country_shap['France'][i]:>8.4f} {country_shap['Germany'][i]:>8.4f}")

    # Plot: grouped bar chart for all 5 crops
    _plot_country_comparison(pooled_shap_results)

    # Waterfall for a notable observation
    _plot_waterfall(pooled_shap_results)


def _plot_country_comparison(pooled_shap_results):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()
    country_colors = {'UK': '#3498db', 'France': '#e74c3c', 'Germany': '#f39c12'}

    for idx, crop in enumerate(CROPS):
        ax = axes_flat[idx]
        res = pooled_shap_results[crop]
        shap_vals = res['shap_values']
        features = res['features']
        countries = res['countries']

        # Top-8 features overall
        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:8]
        top_feats = [features[i] for i in top_idx]

        x = np.arange(len(top_feats))
        w = 0.25

        for j, country_name in enumerate(['UK', 'France', 'Germany']):
            mask = countries == country_name
            if mask.sum() == 0:
                continue
            c_mean_abs = np.mean(np.abs(shap_vals[mask]), axis=0)
            vals = c_mean_abs[top_idx]
            ax.bar(x + (j - 1) * w, vals, w, label=country_name,
                   color=country_colors[country_name], alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(top_feats, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Mean |SHAP|', fontsize=8)
        ax.set_title(crop.replace('_', ' '), fontsize=11, fontweight='bold')
        if idx == 0:
            ax.legend(fontsize=8)

    axes_flat[5].set_visible(False)

    fig.suptitle('SHAP Feature Importance by Country (Pooled Ridge Models)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, 'shap_country_comparison.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"\n  Saved: {fname}")


def _plot_waterfall(pooled_shap_results):
    crop = 'Wheat'
    res = pooled_shap_results[crop]
    df = res['df']
    features = res['features']

    # Try to find a 2018 UK observation (known drought year)
    candidates = df[(df['Year'] == 2018) & (df['Country'] == 'UK')]
    if len(candidates) == 0:
        # Fall back to the observation with lowest yield
        candidates = df.nsmallest(1, 'Yield_t_per_ha')

    obs_idx_in_df = candidates.index[0]
    # Find position in the filtered arrays
    obs_position = list(df.index).index(obs_idx_in_df)

    obs_year = df.loc[obs_idx_in_df, 'Year']
    obs_country = df.loc[obs_idx_in_df, 'Country']
    obs_region = df.loc[obs_idx_in_df, 'Region'] if 'Region' in df.columns else 'N/A'
    obs_yield = df.loc[obs_idx_in_df, 'Yield_t_per_ha']

    print(f"\n  Waterfall observation: {crop}, {obs_country} {obs_region}, {obs_year}")
    print(f"    Actual yield: {obs_yield:.2f} t/ha")
    print(f"    Base value: {res['base_value']:.2f}")
    pred = res['model'].predict(res['X_scaled'][obs_position:obs_position+1])[0]
    print(f"    Predicted yield: {pred:.2f} t/ha")

    # Create SHAP Explanation object for waterfall
    shap_vals_obs = res['shap_values'][obs_position]
    explanation = shap.Explanation(
        values=shap_vals_obs,
        base_values=res['base_value'],
        data=res['X_scaled'][obs_position],
        feature_names=features,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.title(f'SHAP Waterfall — {crop}, {obs_country} {obs_region} ({obs_year})\n'
              f'Actual: {obs_yield:.2f} t/ha | Predicted: {pred:.2f} t/ha',
              fontsize=11)
    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, 'shap_waterfall_example.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  Saved: {fname}")


def main():
    print("=" * 75)
    print("SHAP ANALYSIS FOR CROP YIELD PREDICTION MODELS")
    print("=" * 75)

    # Load data
    print("\nLoading UK datasets...")
    uk_data = load_uk_datasets()

    print("Loading pooled datasets...")
    pooled_data = load_pooled_datasets()

    # Part A: UK-only
    uk_shap_results = run_uk_shap_analysis(uk_data)

    # Part B: Pooled cross-country
    pooled_shap_results = run_pooled_shap_analysis(pooled_data)

    # Part C: Country comparison
    run_country_comparison(pooled_shap_results)

    print("\n" + "=" * 75)
    print("DONE — All SHAP plots saved to plots/")
    print("=" * 75)


if __name__ == '__main__':
    main()
