import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneOut
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Data paths
FRANCE_DATA = os.path.join(PROJECT_ROOT, 'data', 'france', 'processed',
                           'france_regional_crop_yield_weather_2004_2018.csv')
FRANCE_SPRING_BARLEY = os.path.join(PROJECT_ROOT, 'data', 'france', 'processed',
                                     'france_spring_barley_with_weather.csv')
FRANCE_WINTER_BARLEY = os.path.join(PROJECT_ROOT, 'data', 'france', 'processed',
                                     'france_winter_barley_with_weather.csv')
UK_DATA = os.path.join(PROJECT_ROOT, 'data', 'processed',
                       'regional_crop_yield_weather_2004_2024.csv')
UK_SPRING_BARLEY = os.path.join(PROJECT_ROOT, 'data', 'processed',
                                 'spring_barley_with_weather.csv')
UK_WINTER_BARLEY = os.path.join(PROJECT_ROOT, 'data', 'processed',
                                 'winter_barley_with_weather.csv')

YEAR_START = 2004
YEAR_END = 2016  # Common overlap period

CROPS = ['Wheat', 'Winter_Barley', 'Spring_Barley', 'Oats', 'OSR']

CROP_NAME_IN_DATA = {
    'Wheat': 'Wheat',
    'Oats': 'Oats',
    'OSR': 'Oilseed_Rape',
}

# Best alpha per crop (from tuned pooled anomaly experiments)
BEST_ALPHA = {
    'Wheat': 1.0,
    'Winter_Barley': 1.0,
    'Spring_Barley': 100.0,
    'Oats': 10.0,
    'OSR': 1.0,
}

# All 47 weather features used by the anomaly model
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

# Key features to show in UI per crop
UI_FEATURES = {
    'Wheat': ['Winter_Sun', 'Summer_Sun', 'Grain_Filling_Rain',
              'Grain_Filling_Temp', 'Summer_Tmax'],
    'Winter_Barley': ['Spring_Tmax', 'Winter_Sun', 'Planting_Rain',
                      'Winter_Frost', 'Grain_Filling_Rain'],
    'Spring_Barley': ['Spring_Rain', 'Spring_Tmax', 'Summer_Sun',
                      'Summer_Rain', 'Flowering_Temp'],
    'Oats': ['Spring_Rain', 'Flowering_Sun', 'Summer_Rain',
             'Spring_Tmax', 'Rain_Deviation_from_Optimal'],
    'OSR': ['Summer_Tmin', 'Planting_Rain', 'Winter_Frost',
            'Autumn_Rain', 'Spring_Tmax'],
}

# Feature metadata for the frontend
FEATURE_METADATA = {
    'Winter_Tmax': {'unit': '\u00b0C', 'label': 'Winter Max Temp'},
    'Spring_Tmax': {'unit': '\u00b0C', 'label': 'Spring Max Temp'},
    'Summer_Tmax': {'unit': '\u00b0C', 'label': 'Summer Max Temp'},
    'Autumn_Tmax': {'unit': '\u00b0C', 'label': 'Autumn Max Temp'},
    'Winter_Tmin': {'unit': '\u00b0C', 'label': 'Winter Min Temp'},
    'Spring_Tmin': {'unit': '\u00b0C', 'label': 'Spring Min Temp'},
    'Summer_Tmin': {'unit': '\u00b0C', 'label': 'Summer Min Temp'},
    'Autumn_Tmin': {'unit': '\u00b0C', 'label': 'Autumn Min Temp'},
    'Spring_Temp_Mean': {'unit': '\u00b0C', 'label': 'Spring Mean Temp'},
    'Summer_Temp_Mean': {'unit': '\u00b0C', 'label': 'Summer Mean Temp'},
    'Flowering_Temp': {'unit': '\u00b0C', 'label': 'Flowering Temperature'},
    'Grain_Filling_Temp': {'unit': '\u00b0C', 'label': 'Grain Filling Temp'},
    'Summer_Tmax_Peak': {'unit': '\u00b0C', 'label': 'Summer Peak Temp'},
    'Spring_Tmin_Coldest': {'unit': '\u00b0C', 'label': 'Spring Coldest Min'},
    'Spring_Temp_Range': {'unit': '\u00b0C', 'label': 'Spring Temp Range'},
    'Summer_Temp_Range': {'unit': '\u00b0C', 'label': 'Summer Temp Range'},
    'Spring_GDD': {'unit': '\u00b0C\u00b7d', 'label': 'Spring Growing Degree Days'},
    'Summer_GDD': {'unit': '\u00b0C\u00b7d', 'label': 'Summer Growing Degree Days'},
    'Winter_Rain': {'unit': 'mm', 'label': 'Winter Rainfall'},
    'Spring_Rain': {'unit': 'mm', 'label': 'Spring Rainfall'},
    'Summer_Rain': {'unit': 'mm', 'label': 'Summer Rainfall'},
    'Autumn_Rain': {'unit': 'mm', 'label': 'Autumn Rainfall'},
    'Planting_Rain': {'unit': 'mm', 'label': 'Planting Rain'},
    'Grain_Filling_Rain': {'unit': 'mm', 'label': 'Grain Filling Rain'},
    'Harvest_Rain': {'unit': 'mm', 'label': 'Harvest Rain'},
    'Annual_Rain': {'unit': 'mm', 'label': 'Annual Rainfall'},
    'Rain_Deviation_from_Optimal': {'unit': 'mm', 'label': 'Rain Deviation from Optimal'},
    'Spring_Rain_Squared': {'unit': 'mm\u00b2', 'label': 'Spring Rain (Squared)'},
    'Summer_Rain_Squared': {'unit': 'mm\u00b2', 'label': 'Summer Rain (Squared)'},
    'Winter_Sun': {'unit': 'hours', 'label': 'Winter Sunshine'},
    'Spring_Sun': {'unit': 'hours', 'label': 'Spring Sunshine'},
    'Summer_Sun': {'unit': 'hours', 'label': 'Summer Sunshine'},
    'Autumn_Sun': {'unit': 'hours', 'label': 'Autumn Sunshine'},
    'Grain_Filling_Sun': {'unit': 'hours', 'label': 'Grain Filling Sunshine'},
    'Flowering_Sun': {'unit': 'hours', 'label': 'Flowering Sunshine'},
    'Summer_Sun_per_Rain': {'unit': 'h/mm', 'label': 'Summer Sun per Rain'},
    'Winter_Frost': {'unit': 'days', 'label': 'Winter Frost Days'},
    'Spring_Frost': {'unit': 'days', 'label': 'Spring Frost Days'},
    'Autumn_Frost': {'unit': 'days', 'label': 'Autumn Frost Days'},
    'Late_Spring_Frost': {'unit': 'days', 'label': 'Late Spring Frost Days'},
    'Spring_Temp_x_Rain': {'unit': '', 'label': 'Spring Temp \u00d7 Rain'},
    'Summer_Temp_x_Rain': {'unit': '', 'label': 'Summer Temp \u00d7 Rain'},
    'Summer_Temp_x_Sun': {'unit': '', 'label': 'Summer Temp \u00d7 Sun'},
    'Heat_Stress': {'unit': '', 'label': 'Heat Stress Events'},
    'Cold_Spring': {'unit': '', 'label': 'Cold Spring Events'},
    'Extreme_Summer_Rain': {'unit': '', 'label': 'Extreme Summer Rain'},
    'Drought_Spring': {'unit': '', 'label': 'Spring Drought Events'},
}

# Loading of data
def load_france_data(crop):
    if crop == 'Spring_Barley':
        df = pd.read_csv(FRANCE_SPRING_BARLEY)
    elif crop == 'Winter_Barley':
        df = pd.read_csv(FRANCE_WINTER_BARLEY)
    else:
        main_df = pd.read_csv(FRANCE_DATA)
        crop_name = CROP_NAME_IN_DATA.get(crop, crop)
        df = main_df[main_df['Crop'] == crop_name].copy()

    df = df[(df['Year'] >= YEAR_START) & (df['Year'] <= YEAR_END)]
    return df

# Same for UK
def load_uk_data(crop):
    if crop == 'Spring_Barley':
        df = pd.read_csv(UK_SPRING_BARLEY)
    elif crop == 'Winter_Barley':
        df = pd.read_csv(UK_WINTER_BARLEY)
    else:
        main_df = pd.read_csv(UK_DATA)
        crop_name = CROP_NAME_IN_DATA.get(crop, crop)
        df = main_df[main_df['Crop'] == crop_name].copy()

    df = df[(df['Year'] >= YEAR_START) & (df['Year'] <= YEAR_END)]
    return df

# Detrending and anomalisation
def detrend_yields(df):
    df = df.copy()
    df['Yield_trend'] = np.nan
    df['Yield_anomaly'] = np.nan

    for _, group in df.groupby(['Region', 'Crop']):
        idx = group.index
        years = group['Year'].values.astype(float)
        yields = group['Yield_t_per_ha'].values

        if len(years) < 3:
            continue

        year_centered = years - years.mean()
        coeffs = np.polyfit(year_centered, yields, 1)
        trend = np.polyval(coeffs, year_centered)

        df.loc[idx, 'Yield_trend'] = trend
        df.loc[idx, 'Yield_anomaly'] = yields - trend

    return df


def compute_weather_stats(df, features):
    stats = {}
    for region in df['Region'].unique():
        region_data = df[df['Region'] == region]
        region_stats = {}
        for feat in features:
            if feat in region_data.columns and region_data[feat].notna().any():
                region_stats[feat] = {
                    'mean': float(region_data[feat].mean()),
                    'std': float(region_data[feat].std()),
                }
        stats[region] = region_stats

    # Also compute overall UK stats
    overall = {}
    for feat in features:
        if feat in df.columns and df[feat].notna().any():
            overall[feat] = {
                'mean': float(df[feat].mean()),
                'std': float(df[feat].std()),
            }
    stats['_overall'] = overall

    return stats

# This function converts weather features to z-scores per region which lets us calculate make the detrending more precise
def standardise_weather(df, features):
    """Convert weather features to per-region z-scores."""
    df = df.copy()
    for feat in features:
        if feat not in df.columns:
            continue
        region_mean = df.groupby('Region')[feat].transform('mean')
        region_std = df.groupby('Region')[feat].transform('std')
        region_std = region_std.replace(0, 1)
        df[feat] = (df[feat] - region_mean) / region_std
    return df


# Training of the pooled anomaly model
# This trains the Ridge model and evaluates via LOOCV on the UK data. 
def train_pooled_anomaly_model(france_df, uk_df, features, crop, alpha):
    # Select features with non-zero variance in both datasets
    available = []
    for f in features:
        if (f in france_df.columns and f in uk_df.columns
                and france_df[f].notna().any() and uk_df[f].notna().any()
                and france_df[f].std() > 0.001 and uk_df[f].std() > 0.001):
            available.append(f)

    if len(available) < 3:
        return None

    # Detrend yields
    france_dt = detrend_yields(france_df)
    uk_dt = detrend_yields(uk_df)

    # Standardise weather to z-scores
    france_std = standardise_weather(france_dt, available)
    uk_std = standardise_weather(uk_dt, available)

    france_clean = france_std.dropna(subset=available + ['Yield_anomaly'])
    uk_clean = uk_std.dropna(subset=available + ['Yield_anomaly', 'Yield_trend'])

    if len(france_clean) < 10 or len(uk_clean) < 5:
        return None

    X_france = france_clean[available].values
    y_france = france_clean['Yield_anomaly'].values

    X_uk = uk_clean[available].values
    y_uk_anom = uk_clean['Yield_anomaly'].values
    uk_trends = uk_clean['Yield_trend'].values
    y_uk_actual = uk_clean['Yield_t_per_ha'].values

    # loocv on uk
    loo = LeaveOneOut()
    uk_pred_yield = np.zeros(len(y_uk_anom))

    for tr, te in loo.split(X_uk):
        X_train = np.vstack([X_france, X_uk[tr]])
        y_train = np.concatenate([y_france, y_uk_anom[tr]])

        sc = StandardScaler()
        X_train_sc = sc.fit_transform(X_train)
        X_test_sc = sc.transform(X_uk[te])

        model = Ridge(alpha=alpha)
        model.fit(X_train_sc, y_train)

        pred_anom = model.predict(X_test_sc)
        uk_pred_yield[te] = pred_anom + uk_trends[te]

    loocv_r2 = r2_score(y_uk_actual, uk_pred_yield)
    loocv_rmse = np.sqrt(mean_squared_error(y_uk_actual, uk_pred_yield))

    # Train the final model on all data
    X_all = np.vstack([X_france, X_uk])
    y_all = np.concatenate([y_france, y_uk_anom])

    final_scaler = StandardScaler()
    X_all_sc = final_scaler.fit_transform(X_all)

    final_model = Ridge(alpha=alpha)
    final_model.fit(X_all_sc, y_all)

    train_r2 = r2_score(y_all, final_model.predict(X_all_sc))

    return {
        'model': final_model,
        'scaler': final_scaler,
        'features': available,
        'loocv_r2': loocv_r2,
        'loocv_rmse': loocv_rmse,
        'train_r2': train_r2,
        'n_france': len(france_clean),
        'n_uk': len(uk_clean),
        'alpha': alpha,
        'uk_base_yield': float(np.mean(y_uk_actual)),
    }

def main():
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)

    model_config = {}
    results = []

    for crop in CROPS:
        print(f"  CROP: {crop}")
        alpha = BEST_ALPHA[crop]
        print(f"  Alpha: {alpha}")

        # Load data
        france_df = load_france_data(crop)
        uk_df = load_uk_data(crop)

        print(f"  France: {len(france_df)} rows "
              f"({france_df['Year'].min()}-{france_df['Year'].max()})")
        print(f"  UK: {len(uk_df)} rows "
              f"({uk_df['Year'].min()}-{uk_df['Year'].max()})")

        # Compute UK weather stats
        uk_weather_stats = compute_weather_stats(uk_df, ALL_WEATHER_FEATURES)

        # Train pooled anomaly model
        result = train_pooled_anomaly_model(
            france_df, uk_df, ALL_WEATHER_FEATURES, crop, alpha)

        if result is None:
            print(f"  FAILED: insufficient data")
            continue
        # Save model, scaler, and metadata
        model_name = crop.lower()
        joblib.dump(result['model'], os.path.join(models_dir, f'{model_name}_model.pkl'))
        joblib.dump(result['scaler'], os.path.join(models_dir, f'{model_name}_scaler.pkl'))

        # Save metadata: UK weather stats, features, base yield
        meta = {
            'uk_weather_stats': uk_weather_stats,
            'model_features': result['features'],
            'uk_base_yield': result['uk_base_yield'],
            'alpha': alpha,
            'model_type': 'pooled_anomaly',
        }
        joblib.dump(meta, os.path.join(models_dir, f'{model_name}_meta.pkl'))
        print(f"  Saved: {model_name}_model.pkl, {model_name}_scaler.pkl, "
              f"{model_name}_meta.pkl")

        # Build UI feature config
        ui_feats = UI_FEATURES.get(crop, result['features'][:5])
        feature_config = {}
        for feat in ui_feats:
            if feat not in uk_df.columns:
                continue

            col = uk_df[feat].dropna()
            if len(col) == 0:
                continue

            meta_info = FEATURE_METADATA.get(feat, {
                'unit': '', 'label': feat.replace('_', ' ')
            })

            # Determine input type based on feature characteristics
            if feat in ('Extreme_Summer_Rain', 'Heat_Stress', 'Cold_Spring',
                        'Drought_Spring'):
                input_type = 'toggle'
                feat_min, feat_max, feat_default = 0, 1, 0
            else:
                input_type = 'range'
                feat_min = float(np.floor(col.min()))
                feat_max = float(np.ceil(col.max()))
                feat_default = float(round(col.median()))

            feature_config[feat] = {
                'min': feat_min,
                'max': feat_max,
                'default': feat_default,
                'unit': meta_info.get('unit', ''),
                'label': meta_info.get('label', feat.replace('_', ' ')),
                'input_type': input_type,
            }

        model_config[crop] = {
            'model_type': f'Pooled Anomaly Ridge (\u03b1={alpha})',
            'features': feature_config,
            'loocv_r2': round(result['loocv_r2'], 3),
            'train_r2': round(result['train_r2'], 3),
            'uk_base_yield': round(result['uk_base_yield'], 2),
        }

        results.append({
            'Crop': crop,
            'Alpha': alpha,
            'N_France': result['n_france'],
            'N_UK': result['n_uk'],
            'N_Features': len(result['features']),
            'LOOCV_R2': result['loocv_r2'],
            'RMSE': result['loocv_rmse'],
            'Train_R2': result['train_r2'],
        })

    # Save model_config.json
    config_path = os.path.join(models_dir, 'model_config.json')
    config_json = json.dumps(model_config, indent=2, ensure_ascii=False)
    config_json = config_json.replace('NaN', 'null')
    with open(config_path, 'w') as f:
        f.write(config_json)
    print(f"\nSaved model config: {config_path}")
    header = ("Crop             Alpha  N_FR  N_UK Feats  LOOCV R²  "
              "    RMSE  Train R²")
    print(f"\n{header}")
    print("-" * 70)
    for r in results:
        print(f"{r['Crop']:<16} {r['Alpha']:>6.1f} {r['N_France']:>5} "
              f"{r['N_UK']:>5} {r['N_Features']:>5} {r['LOOCV_R2']:>10.3f} "
              f"{r['RMSE']:>8.3f} {r['Train_R2']:>10.3f}")
    print(f"\nAll models saved to: {os.path.abspath(models_dir)}")


if __name__ == '__main__':
    main()
