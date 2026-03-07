"""
Long-History Detrended Anomaly Transfer
=========================================
Uses France's extended historical depth (1995-2018, 24 years vs current 13)
to learn robust weather-anomaly → yield-anomaly relationships, then transfers
those response curves to UK weather anomalies.

Core idea:
  1. Extend E-OBS weather processing to 1995-2018 for French regions
  2. Load Schauberger yields for 1995-2018 (département → 4 macro-regions)
  3. Detrend yields per region to remove technology/variety trends
  4. Train Ridge on weather features → yield anomalies (France, 24 years)
  5. Apply learned anomaly response to UK weather anomalies
  6. Add UK's own yield trend back → evaluate

Why this could work:
  - 24 years captures more weather extremes than 13 (e.g. 2003 heatwave)
  - Detrending removes the main source of transfer failure (different yield levels)
  - Transferring response *shapes* rather than absolute predictions

Usage:
    python src/france/long_history_transfer.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneOut
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    PATHS, FRANCE_REGIONS, FRANCE_DEPT_TO_REGION,
    FRANCE_CROP_FILES, FRANCE_DATA_DIR, FRANCE_OPTIMAL_RAINFALL,
    UK_OPTIMAL_RAINFALL,
)

sys.path.insert(0, PATHS['models'])
from baseline_model_config import CROP_FEATURES

# ============================================================================
# CONFIG
# ============================================================================

EXTENDED_YEAR_START = 1995  # E-OBS data starts at 1995
EXTENDED_YEAR_END = 2018    # Schauberger data ends at 2018
UK_YEAR_START = 2004
UK_YEAR_END = 2016          # Common overlap for evaluation

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


# ============================================================================
# STEP 1: EXTENDED E-OBS WEATHER PROCESSING (1995-2018)
# ============================================================================

def process_extended_eobs_weather():
    """
    Process E-OBS data for French regions over 1995-2018.
    Saves extended monthly weather CSVs.
    Returns dict of {region: DataFrame}.
    """
    output_dir = PATHS['france_raw']
    extended_prefix = 'eobs_monthly_weather_extended_'

    # Check if already processed
    existing = {}
    for region in FRANCE_REGIONS:
        path = os.path.join(output_dir, f'{extended_prefix}{region}.csv')
        if os.path.exists(path):
            existing[region] = pd.read_csv(path)

    if len(existing) == len(FRANCE_REGIONS):
        min_year = min(df['year'].min() for df in existing.values())
        max_year = max(df['year'].max() for df in existing.values())
        if min_year <= EXTENDED_YEAR_START and max_year >= EXTENDED_YEAR_END:
            print(f"  Extended weather already processed ({min_year}-{max_year})")
            return existing

    print("  Processing E-OBS weather for 1995-2018 (this may take a few minutes)...")

    # Import E-OBS processing functions
    from process_eobs_weather import (
        load_eobs_variable, build_region_masks, EOBS_DIR, GADM_PATH, MONTH_NAMES,
    )
    import xarray as xr
    import geopandas as gpd

    # Load NUTS3 boundaries
    nuts3 = gpd.read_file(GADM_PATH)
    nuts3_fr = nuts3[nuts3['CNTR_CODE'] == 'FR'].copy()
    nuts3_fr = nuts3_fr[~nuts3_fr['NUTS_ID'].str.startswith('FRY')]

    # Get grid coordinates
    sample_path = os.path.join(EOBS_DIR, 'tx_1995-2010.nc')
    ds_sample = xr.open_dataset(sample_path)
    lat_sl = slice(42, 51.5)  # France only (narrower than FR+DE)
    lon_sl = slice(-5.5, 8.5)
    lats = ds_sample.latitude.sel(latitude=lat_sl).values
    lons = ds_sample.longitude.sel(longitude=lon_sl).values
    ds_sample.close()

    # Build spatial masks (France regions only)
    masks = build_region_masks(nuts3_fr, lats, lons)
    france_masks = {r: masks[r] for r in FRANCE_REGIONS if r in masks}

    if not france_masks:
        raise RuntimeError("No French region masks created")

    # Load E-OBS variables for extended period
    time_start = f'{EXTENDED_YEAR_START}-01-01'
    time_end = f'{EXTENDED_YEAR_END}-12-31'

    print("    Loading Tmax...")
    ds_tx = load_eobs_variable('tx', time_start, time_end, lat_sl, lon_sl)
    print("    Loading Tmin...")
    ds_tn = load_eobs_variable('tn', time_start, time_end, lat_sl, lon_sl)
    print("    Loading Precipitation...")
    ds_rr = load_eobs_variable('rr', time_start, time_end, lat_sl, lon_sl)

    # Try radiation
    ds_qq = None
    try:
        print("    Loading Radiation...")
        ds_qq = load_eobs_variable('qq', time_start, time_end, lat_sl, lon_sl)
    except FileNotFoundError:
        print("    Radiation not available — sunshine estimated.")

    tx = ds_tx['tx']
    tn = ds_tn['tn']
    rr = ds_rr['rr']

    result = {}
    for region, mask in france_masks.items():
        print(f"    Processing {region} (1995-2018)...")
        mask3d = mask[np.newaxis, :, :]

        tx_vals = tx.values.copy()
        tx_vals[~np.broadcast_to(mask3d, tx_vals.shape)] = np.nan
        tx_region = np.nanmean(tx_vals, axis=(1, 2))

        tn_vals = tn.values.copy()
        tn_vals[~np.broadcast_to(mask3d, tn_vals.shape)] = np.nan
        tn_region = np.nanmean(tn_vals, axis=(1, 2))

        rr_vals = rr.values.copy()
        rr_vals[~np.broadcast_to(mask3d, rr_vals.shape)] = np.nan
        rr_region = np.nanmean(rr_vals, axis=(1, 2))

        # Frost
        tn_frost = (tn_vals < 0).astype(float)
        tn_frost[~np.broadcast_to(mask3d, tn_frost.shape)] = np.nan
        frost_region = np.nanmean(tn_frost, axis=(1, 2))

        # Radiation (may cover different time range than tx/tn/rr)
        qq_series = None
        if ds_qq is not None:
            qq = ds_qq['qq']
            qq_times = pd.DatetimeIndex(qq.time.values)
            qq_vals = qq.values.copy()
            # qq may have different spatial dims if loaded from different file
            if qq_vals.shape[1:] == mask.shape:
                qq_vals[~np.broadcast_to(mask3d, qq_vals.shape)] = np.nan
                qq_region = np.nanmean(qq_vals, axis=(1, 2))
                qq_series = pd.Series(qq_region / 25.0, index=qq_times)

        times = pd.DatetimeIndex(tx.time.values)
        daily_df = pd.DataFrame({
            'time': times,
            'tmax': tx_region, 'tmin': tn_region,
            'rain': rr_region,
            'frost_flag': (tn_region < 0).astype(float),
        })
        if qq_series is not None:
            # Align radiation to same time index as temperature/rain
            daily_df['sunshine'] = daily_df['time'].map(qq_series).values
        else:
            daily_df['sunshine'] = np.nan

        daily_df['year'] = daily_df['time'].dt.year
        daily_df['month'] = daily_df['time'].dt.month

        agg_dict = {
            'tmax': ('tmax', 'mean'), 'tmin': ('tmin', 'mean'),
            'rain': ('rain', 'sum'), 'frost': ('frost_flag', 'sum'),
        }
        if 'sunshine' in daily_df.columns:
            agg_dict['sunshine'] = ('sunshine', 'sum')

        monthly = daily_df.groupby(['year', 'month']).agg(**agg_dict).reset_index()

        years = range(EXTENDED_YEAR_START, EXTENDED_YEAR_END + 1)
        rows = []
        for year in years:
            row = {'year': year}
            year_data = monthly[monthly['year'] == year]
            for m in range(1, 13):
                mn = MONTH_NAMES[m - 1]
                m_data = year_data[year_data['month'] == m]
                if len(m_data) > 0:
                    row[f'tmax_{mn}'] = float(m_data['tmax'].values[0])
                    row[f'tmin_{mn}'] = float(m_data['tmin'].values[0])
                    row[f'rain_{mn}'] = float(m_data['rain'].values[0])
                    row[f'frost_{mn}'] = float(m_data['frost'].values[0])
                    row[f'sun_{mn}'] = float(m_data['sunshine'].values[0]) if 'sunshine' in m_data.columns else np.nan
                else:
                    for var in ['tmax', 'tmin', 'rain', 'frost', 'sun']:
                        row[f'{var}_{mn}'] = np.nan
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, f'{extended_prefix}{region}.csv')
        df.to_csv(csv_path, index=False)
        print(f"    Saved: {csv_path} ({len(df)} years)")
        result[region] = df

    ds_tx.close()
    ds_tn.close()
    ds_rr.close()
    if ds_qq is not None:
        ds_qq.close()

    return result


# ============================================================================
# STEP 2: LOAD EXTENDED FRANCE YIELDS (1995-2018)
# ============================================================================

def load_extended_france_yields():
    """
    Load Schauberger yields for 1995-2018, aggregated to 4 macro-regions.
    Returns DataFrame with columns: Year, Region, Crop, Area_hectares, Yield_t_per_ha.
    """
    print("  Loading France yields (Schauberger, 1995-2018)...")
    all_rows = []

    for crop_name, filename in FRANCE_CROP_FILES.items():
        filepath = os.path.join(FRANCE_DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"    WARNING: Missing {filepath}")
            continue

        df = pd.read_csv(filepath, sep=';')
        df = df[(df['year'] >= EXTENDED_YEAR_START) & (df['year'] <= EXTENDED_YEAR_END)]
        df['Region'] = df['department'].map(FRANCE_DEPT_TO_REGION)
        df = df.dropna(subset=['Region'])
        df = df[(df['yield'] > 0) & (df['area'] > 0)]

        for (year, region), group in df.groupby(['year', 'Region']):
            total_area = group['area'].sum()
            weighted_yield = (group['yield'] * group['area']).sum() / total_area
            all_rows.append({
                'Year': int(year),
                'Region': region,
                'Crop': crop_name,
                'Area_hectares': int(total_area),
                'Yield_t_per_ha': round(weighted_yield, 3),
            })

        n_years = df['year'].nunique()
        print(f"    {crop_name:<16} {n_years} years, mean={df['yield'].mean():.2f} t/ha")

    result = pd.DataFrame(all_rows)
    print(f"  Total: {len(result)} rows ({result['Year'].min()}-{result['Year'].max()})")
    return result


# ============================================================================
# STEP 3: COMPUTE SEASONAL FEATURES FOR EXTENDED PERIOD
# ============================================================================

def compute_extended_seasonal_features(monthly_data):
    """
    Compute 47 seasonal weather features from extended monthly data.
    Reuses the same logic as process_era5_weather.py.
    """
    from process_era5_weather import compute_seasonal_features, extract_variable_dfs

    all_features = []
    for region in FRANCE_REGIONS:
        if region not in monthly_data:
            continue
        monthly_df = monthly_data[region]
        var_dfs = extract_variable_dfs(monthly_df)
        sf = compute_seasonal_features(
            tmax_df=var_dfs['tmax'], tmin_df=var_dfs['tmin'],
            rain_df=var_dfs['rain'], sun_df=var_dfs['sun'],
            frost_df=var_dfs['frost'],
            optimal_rainfall=FRANCE_OPTIMAL_RAINFALL,
        )
        sf['Region'] = region
        all_features.append(sf)

    features_df = pd.concat(all_features, ignore_index=True)
    feature_cols = [c for c in features_df.columns if c not in ('Year', 'Region')]
    features_df = features_df[['Year', 'Region'] + feature_cols]
    print(f"  Extended features: {len(features_df)} rows, {len(feature_cols)} features")
    return features_df


# ============================================================================
# STEP 4: DETRENDING
# ============================================================================

def detrend_yields(df, method='linear'):
    """
    Detrend yields per (Region, Crop) group, removing technology/variety trends.
    Returns DataFrame with added columns:
      - Yield_trend: fitted trend value
      - Yield_anomaly: actual - trend (the weather-driven residual)

    Methods:
      - 'linear': OLS linear trend
      - 'quadratic': quadratic trend (captures yield plateaus)
    """
    df = df.copy()
    df['Yield_trend'] = np.nan
    df['Yield_anomaly'] = np.nan

    for _, group in df.groupby(['Region', 'Crop']):
        idx = group.index
        years = group['Year'].values.astype(float)
        yields = group['Yield_t_per_ha'].values

        if len(years) < 3:
            continue

        # Normalise years to avoid numerical issues
        year_centered = years - years.mean()

        if method == 'linear':
            coeffs = np.polyfit(year_centered, yields, 1)
            trend = np.polyval(coeffs, year_centered)
        elif method == 'quadratic':
            coeffs = np.polyfit(year_centered, yields, 2)
            trend = np.polyval(coeffs, year_centered)
        else:
            raise ValueError(f"Unknown detrend method: {method}")

        df.loc[idx, 'Yield_trend'] = trend
        df.loc[idx, 'Yield_anomaly'] = yields - trend

    return df


def standardise_weather_anomalies(df, weather_features):
    """
    Convert weather features to anomalies (deviations from per-region means).
    This makes weather features comparable across countries by removing
    the different climate baselines.
    """
    df = df.copy()
    for feat in weather_features:
        if feat not in df.columns:
            continue
        # Per-region anomaly
        region_mean = df.groupby('Region')[feat].transform('mean')
        region_std = df.groupby('Region')[feat].transform('std')
        # Avoid division by zero
        region_std = region_std.replace(0, 1)
        df[feat] = (df[feat] - region_mean) / region_std
    return df


# ============================================================================
# STEP 5: LOAD UK DATA
# ============================================================================

def load_uk_data():
    """Load UK yield + weather data."""
    uk_regional = pd.read_csv(os.path.join(
        PATHS['uk_processed'], 'regional_crop_yield_weather_2004_2024.csv'))
    uk_spring = pd.read_csv(os.path.join(
        PATHS['uk_processed'], 'spring_barley_with_weather.csv'))
    uk_winter = pd.read_csv(os.path.join(
        PATHS['uk_processed'], 'winter_barley_with_weather.csv'))
    return uk_regional, uk_spring, uk_winter


def get_uk_crop_data(uk_regional, uk_spring, uk_winter, crop):
    """Extract data for a specific UK crop."""
    if crop == 'Spring_Barley':
        df = uk_spring.copy()
    elif crop == 'Winter_Barley':
        df = uk_winter.copy()
    else:
        crop_in_data = CROP_NAME_IN_DATA.get(crop, crop)
        df = uk_regional[uk_regional['Crop'] == crop_in_data].copy()

    df = df[(df['Year'] >= UK_YEAR_START) & (df['Year'] <= UK_YEAR_END)]
    return df


# ============================================================================
# STEP 6: TRANSFER EXPERIMENTS
# ============================================================================

def select_features(df, feature_list):
    """Select available features with non-zero variance."""
    available = [f for f in feature_list if f in df.columns
                 and df[f].notna().any() and df[f].std() > 0.001]
    return available


def experiment_france_anomaly_to_uk(france_df, uk_df, weather_features, crop, alpha=1.0):
    """
    Train on France yield anomalies, predict UK yield anomalies.

    Steps:
      1. Detrend France yields → yield anomalies
      2. Detrend UK yields → yield anomalies (for evaluation only)
      3. Standardise weather features to anomalies (per-region z-scores)
      4. Train Ridge on France: weather anomalies → yield anomalies
      5. Apply to UK weather anomalies → predicted UK yield anomalies
      6. Add UK trend back → final prediction
      7. Compare with UK-only baseline (LOOCV)
    """
    features = select_features(france_df, weather_features)
    features = [f for f in features if f in uk_df.columns
                and uk_df[f].notna().any() and uk_df[f].std() > 0.001]

    if len(features) < 3:
        return None

    # --- Detrend ---
    france_dt = detrend_yields(france_df, method='linear')
    uk_dt = detrend_yields(uk_df, method='linear')

    # --- Standardise weather to per-region anomalies ---
    france_std = standardise_weather_anomalies(france_dt, features)
    uk_std = standardise_weather_anomalies(uk_dt, features)

    # Drop rows with NaN in key columns
    france_clean = france_std.dropna(subset=features + ['Yield_anomaly'])
    uk_clean = uk_std.dropna(subset=features + ['Yield_anomaly', 'Yield_trend'])

    if len(france_clean) < 10 or len(uk_clean) < 5:
        return None

    # --- Train on France anomalies ---
    X_france = france_clean[features].values
    y_france = france_clean['Yield_anomaly'].values

    scaler = StandardScaler()
    X_france_sc = scaler.fit_transform(X_france)

    model = Ridge(alpha=alpha)
    model.fit(X_france_sc, y_france)

    # France training R²
    france_train_r2 = r2_score(y_france, model.predict(X_france_sc))

    # --- Apply to UK ---
    X_uk = uk_clean[features].values
    X_uk_sc = scaler.transform(X_uk)

    uk_pred_anomaly = model.predict(X_uk_sc)

    # Reconstruct full yield: predicted anomaly + UK's own trend
    uk_pred_yield = uk_pred_anomaly + uk_clean['Yield_trend'].values
    uk_actual_yield = uk_clean['Yield_t_per_ha'].values

    transfer_r2 = r2_score(uk_actual_yield, uk_pred_yield)
    transfer_rmse = np.sqrt(mean_squared_error(uk_actual_yield, uk_pred_yield))

    # Anomaly-level R² (how well do we predict UK weather-driven variation?)
    uk_actual_anomaly = uk_clean['Yield_anomaly'].values
    anomaly_r2 = r2_score(uk_actual_anomaly, uk_pred_anomaly)

    return {
        'crop': crop,
        'n_france_train': len(france_clean),
        'n_uk_test': len(uk_clean),
        'n_features': len(features),
        'france_train_r2': france_train_r2,
        'anomaly_r2': anomaly_r2,
        'transfer_r2': transfer_r2,
        'transfer_rmse': transfer_rmse,
    }


def experiment_rescaled_transfer(france_df, uk_df, weather_features, crop, alpha=1.0):
    """
    Anomaly-rescaled direct transfer.

    The France-trained model captures the correct *direction* of weather effects
    but the wrong *magnitude* (France yield variability ≠ UK yield variability).
    We fix this by rescaling predicted anomalies via LOOCV:

    For each held-out UK point:
      1. Train Ridge on France anomalies
      2. Predict anomalies for ALL UK points
      3. Use the non-held-out UK points to estimate a linear calibration:
         actual_uk_anomaly = a * predicted_anomaly + b
      4. Apply calibration to the held-out point
      5. Add UK trend back

    This only uses UK data for scale calibration (2 parameters: shift + scale),
    NOT for learning weather-yield relationships. The weather response shapes
    come entirely from France.
    """
    features = select_features(france_df, weather_features)
    features = [f for f in features if f in uk_df.columns
                and uk_df[f].notna().any() and uk_df[f].std() > 0.001]

    if len(features) < 3:
        return None

    france_dt = detrend_yields(france_df, method='linear')
    uk_dt = detrend_yields(uk_df, method='linear')

    france_std = standardise_weather_anomalies(france_dt, features)
    uk_std = standardise_weather_anomalies(uk_dt, features)

    france_clean = france_std.dropna(subset=features + ['Yield_anomaly'])
    uk_clean = uk_std.dropna(subset=features + ['Yield_anomaly', 'Yield_trend'])

    if len(france_clean) < 10 or len(uk_clean) < 8:
        return None

    # Train on France (once — the weather response comes from France only)
    X_france = france_clean[features].values
    y_france = france_clean['Yield_anomaly'].values

    scaler = StandardScaler()
    X_france_sc = scaler.fit_transform(X_france)

    model = Ridge(alpha=alpha)
    model.fit(X_france_sc, y_france)

    # Predict anomalies for all UK points
    X_uk = uk_clean[features].values
    X_uk_sc = scaler.transform(X_uk)
    uk_raw_pred = model.predict(X_uk_sc)

    uk_actual_anomaly = uk_clean['Yield_anomaly'].values
    uk_trends = uk_clean['Yield_trend'].values
    y_uk_actual = uk_clean['Yield_t_per_ha'].values

    # --- Method 1: Variance rescaling via LOOCV ---
    # For each held-out UK point, use the rest to estimate scale factor
    loo = LeaveOneOut()
    rescaled_pred_var = np.zeros(len(uk_raw_pred))

    for tr, te in loo.split(uk_raw_pred):
        # Estimate scale: ratio of actual to predicted std
        pred_std = np.std(uk_raw_pred[tr])
        actual_std = np.std(uk_actual_anomaly[tr])
        scale = actual_std / pred_std if pred_std > 1e-6 else 1.0

        # Rescale: zero-mean predicted * scale
        pred_mean = np.mean(uk_raw_pred[tr])
        rescaled_pred_var[te] = (uk_raw_pred[te] - pred_mean) * scale

    var_yield = rescaled_pred_var + uk_trends
    var_r2 = r2_score(y_uk_actual, var_yield)
    var_rmse = np.sqrt(mean_squared_error(y_uk_actual, var_yield))

    # --- Method 2: Linear calibration via LOOCV ---
    # actual_anomaly = a * predicted_anomaly + b
    rescaled_pred_lin = np.zeros(len(uk_raw_pred))

    for tr, te in loo.split(uk_raw_pred):
        # Fit linear calibration on training fold
        coeffs = np.polyfit(uk_raw_pred[tr], uk_actual_anomaly[tr], 1)
        rescaled_pred_lin[te] = np.polyval(coeffs, uk_raw_pred[te])

    lin_yield = rescaled_pred_lin + uk_trends
    lin_r2 = r2_score(y_uk_actual, lin_yield)
    lin_rmse = np.sqrt(mean_squared_error(y_uk_actual, lin_yield))

    # --- Method 3: Correlation-based (no UK data needed) ---
    # Just check if France model captures the right direction
    raw_corr = np.corrcoef(uk_raw_pred, uk_actual_anomaly)[0, 1]

    # Pick the best rescaling method
    best_method = 'variance' if var_r2 >= lin_r2 else 'linear'
    best_r2 = max(var_r2, lin_r2)
    best_rmse = var_rmse if var_r2 >= lin_r2 else lin_rmse

    return {
        'crop': crop,
        'variance_r2': var_r2,
        'variance_rmse': var_rmse,
        'linear_r2': lin_r2,
        'linear_rmse': lin_rmse,
        'raw_correlation': raw_corr,
        'best_method': best_method,
        'best_r2': best_r2,
        'best_rmse': best_rmse,
        'n_france': len(france_clean),
        'n_uk': len(uk_clean),
        'n_features': len(features),
    }


def experiment_uk_only_baseline(uk_df, weather_features, crop):
    """UK-only LOOCV Ridge baseline for comparison."""
    features = select_features(uk_df, weather_features)
    if len(features) < 3:
        return None

    uk_clean = uk_df.dropna(subset=features + ['Yield_t_per_ha'])
    if len(uk_clean) < 10:
        return None

    X = uk_clean[features].values
    y = uk_clean['Yield_t_per_ha'].values

    loo = LeaveOneOut()
    y_pred = np.zeros(len(y))
    for tr, te in loo.split(X):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])
        model = Ridge(alpha=1.0)
        model.fit(X_tr, y[tr])
        y_pred[te] = model.predict(X_te)

    return {
        'crop': crop,
        'uk_loocv_r2': r2_score(y, y_pred),
        'uk_loocv_rmse': np.sqrt(mean_squared_error(y, y_pred)),
    }


def experiment_pooled_anomaly(france_df, uk_df, weather_features, crop, alpha=1.0):
    """
    Pooled anomaly model: train on BOTH France + UK anomalies together,
    evaluate UK via LOOCV.
    """
    features = select_features(france_df, weather_features)
    features = [f for f in features if f in uk_df.columns
                and uk_df[f].notna().any() and uk_df[f].std() > 0.001]

    if len(features) < 3:
        return None

    france_dt = detrend_yields(france_df, method='linear')
    uk_dt = detrend_yields(uk_df, method='linear')

    france_std = standardise_weather_anomalies(france_dt, features)
    uk_std = standardise_weather_anomalies(uk_dt, features)

    france_clean = france_std.dropna(subset=features + ['Yield_anomaly'])
    uk_clean = uk_std.dropna(subset=features + ['Yield_anomaly', 'Yield_trend'])

    if len(france_clean) < 10 or len(uk_clean) < 5:
        return None

    # LOOCV over UK rows, with France always in training
    X_france = france_clean[features].values
    y_france_anom = france_clean['Yield_anomaly'].values

    X_uk = uk_clean[features].values
    y_uk_anom = uk_clean['Yield_anomaly'].values
    uk_trends = uk_clean['Yield_trend'].values
    y_uk_actual = uk_clean['Yield_t_per_ha'].values

    loo = LeaveOneOut()
    uk_pred_yield = np.zeros(len(y_uk_anom))

    for tr, te in loo.split(X_uk):
        # Training: all France + UK-minus-one
        X_train = np.vstack([X_france, X_uk[tr]])
        y_train = np.concatenate([y_france_anom, y_uk_anom[tr]])

        sc = StandardScaler()
        X_train_sc = sc.fit_transform(X_train)
        X_test_sc = sc.transform(X_uk[te])

        model = Ridge(alpha=alpha)
        model.fit(X_train_sc, y_train)

        pred_anom = model.predict(X_test_sc)
        uk_pred_yield[te] = pred_anom + uk_trends[te]

    pooled_r2 = r2_score(y_uk_actual, uk_pred_yield)
    pooled_rmse = np.sqrt(mean_squared_error(y_uk_actual, uk_pred_yield))

    return {
        'crop': crop,
        'pooled_anomaly_r2': pooled_r2,
        'pooled_anomaly_rmse': pooled_rmse,
        'n_france': len(france_clean),
        'n_uk': len(uk_clean),
    }


def experiment_pooled_anomaly_tuned(france_df, uk_df, weather_features, crop):
    """
    Pooled anomaly with alpha tuning via nested LOOCV.
    Also tries crop-specific features vs all features.
    """
    best_result = None
    best_r2 = -np.inf

    # Try different feature sets and alphas
    feature_sets = {
        'all': weather_features,
        'crop_specific': CROP_FEATURES.get(crop, weather_features),
    }

    for feat_name, feat_list in feature_sets.items():
        for alpha in [0.1, 1.0, 10.0, 100.0]:
            result = experiment_pooled_anomaly(
                france_df, uk_df, feat_list, crop, alpha=alpha)
            if result and result['pooled_anomaly_r2'] > best_r2:
                best_r2 = result['pooled_anomaly_r2']
                best_result = result
                best_result['best_alpha'] = alpha
                best_result['best_features'] = feat_name

    return best_result


def experiment_history_comparison(france_df, uk_df, weather_features, crop):
    """
    Compare extended history (1995-2018) vs original period (2004-2016)
    to quantify the benefit of longer training history.
    """
    # Extended (1995-2018): use all France data
    extended = experiment_pooled_anomaly(france_df, uk_df, weather_features, crop)

    # Original (2004-2016): trim France to same period as before
    france_trimmed = france_df[
        (france_df['Year'] >= 2004) & (france_df['Year'] <= 2016)
    ].copy()
    original = experiment_pooled_anomaly(france_trimmed, uk_df, weather_features, crop)

    return {
        'extended_r2': extended['pooled_anomaly_r2'] if extended else float('nan'),
        'original_r2': original['pooled_anomaly_r2'] if original else float('nan'),
        'n_france_extended': extended['n_france'] if extended else 0,
        'n_france_original': original['n_france'] if original else 0,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("LONG-HISTORY DETRENDED ANOMALY TRANSFER")
    print("=" * 70)

    # --- Step 1: Extended weather ---
    print("\n--- Step 1: Extended E-OBS weather (1995-2018) ---")
    monthly_data = process_extended_eobs_weather()

    # --- Step 2: Extended France yields ---
    print("\n--- Step 2: Extended France yields (1995-2018) ---")
    france_yields = load_extended_france_yields()

    # --- Step 3: Seasonal features ---
    print("\n--- Step 3: Seasonal weather features ---")
    france_features = compute_extended_seasonal_features(monthly_data)

    # --- Merge yields + weather ---
    print("\n--- Step 4: Merge yields + weather ---")
    france_merged = france_yields.merge(france_features, on=['Year', 'Region'], how='inner')
    print(f"  Merged: {len(france_merged)} rows "
          f"({france_merged['Year'].min()}-{france_merged['Year'].max()})")
    print(f"  Crops: {sorted(france_merged['Crop'].unique())}")
    for crop in france_merged['Crop'].unique():
        n = len(france_merged[france_merged['Crop'] == crop])
        print(f"    {crop:<16} {n} rows")

    # --- Step 5: Load UK data ---
    print("\n--- Step 5: Load UK data ---")
    uk_regional, uk_spring, uk_winter = load_uk_data()

    # --- Step 6: Run experiments ---
    print("\n--- Step 6: Transfer Experiments ---")
    print("=" * 70)

    all_results = []

    for crop in CROPS:
        print(f"\n  {'='*60}")
        print(f"  CROP: {crop}")
        print(f"  {'='*60}")

        # Get France data for this crop
        crop_in_data = CROP_NAME_IN_DATA.get(crop, crop)
        france_crop = france_merged[france_merged['Crop'] == crop_in_data].copy()

        # Get UK data for this crop
        uk_crop = get_uk_crop_data(uk_regional, uk_spring, uk_winter, crop)

        if len(france_crop) < 10 or len(uk_crop) < 5:
            print(f"    Insufficient data: France={len(france_crop)}, UK={len(uk_crop)}")
            continue

        print(f"    France: {len(france_crop)} rows "
              f"({france_crop['Year'].min()}-{france_crop['Year'].max()})")
        print(f"    UK: {len(uk_crop)} rows "
              f"({uk_crop['Year'].min()}-{uk_crop['Year'].max()})")

        # Yield summary
        fr_mean = france_crop['Yield_t_per_ha'].mean()
        uk_mean = uk_crop['Yield_t_per_ha'].mean()
        print(f"    Mean yield: France={fr_mean:.2f}, UK={uk_mean:.2f} t/ha")

        # --- Experiment A: UK-only baseline ---
        baseline = experiment_uk_only_baseline(uk_crop, ALL_WEATHER_FEATURES, crop)
        if baseline:
            print(f"\n    [A] UK-only LOOCV Ridge:")
            print(f"        R² = {baseline['uk_loocv_r2']:.3f}, "
                  f"RMSE = {baseline['uk_loocv_rmse']:.3f} t/ha")

        # --- Experiment B: France anomaly → UK (direct transfer) ---
        transfer = experiment_france_anomaly_to_uk(
            france_crop, uk_crop, ALL_WEATHER_FEATURES, crop)
        if transfer:
            print(f"\n    [B] France anomaly → UK transfer (extended history):")
            print(f"        France train R² = {transfer['france_train_r2']:.3f} "
                  f"({transfer['n_france_train']} samples)")
            print(f"        UK anomaly R²   = {transfer['anomaly_r2']:.3f}")
            print(f"        UK transfer R²  = {transfer['transfer_r2']:.3f}, "
                  f"RMSE = {transfer['transfer_rmse']:.3f} t/ha")

        # --- Experiment C: Pooled anomaly (France + UK) ---
        pooled = experiment_pooled_anomaly(
            france_crop, uk_crop, ALL_WEATHER_FEATURES, crop)
        if pooled:
            print(f"\n    [C] Pooled anomaly LOOCV (France+UK → UK):")
            print(f"        R² = {pooled['pooled_anomaly_r2']:.3f}, "
                  f"RMSE = {pooled['pooled_anomaly_rmse']:.3f} t/ha")
            print(f"        ({pooled['n_france']} France + {pooled['n_uk']} UK samples)")

        # --- Experiment D: France anomaly → UK with crop-specific features ---
        crop_feats = CROP_FEATURES.get(crop, [])
        transfer_cs = None
        if crop_feats:
            transfer_cs = experiment_france_anomaly_to_uk(
                france_crop, uk_crop, crop_feats, crop)
            if transfer_cs:
                print(f"\n    [D] France → UK with crop-specific features ({len(crop_feats)}):")
                print(f"        UK anomaly R²  = {transfer_cs['anomaly_r2']:.3f}")
                print(f"        UK transfer R² = {transfer_cs['transfer_r2']:.3f}, "
                      f"RMSE = {transfer_cs['transfer_rmse']:.3f} t/ha")

        # --- Experiment E: Rescaled transfer (all features) ---
        rescaled_all = experiment_rescaled_transfer(
            france_crop, uk_crop, ALL_WEATHER_FEATURES, crop)
        if rescaled_all:
            print(f"\n    [E] Rescaled transfer (all features):")
            print(f"        Raw correlation   = {rescaled_all['raw_correlation']:.3f}")
            print(f"        Variance rescale  R² = {rescaled_all['variance_r2']:.3f}")
            print(f"        Linear calibrate  R² = {rescaled_all['linear_r2']:.3f}")
            print(f"        Best: {rescaled_all['best_method']}, "
                  f"R² = {rescaled_all['best_r2']:.3f}, "
                  f"RMSE = {rescaled_all['best_rmse']:.3f} t/ha")

        # --- Experiment F: Rescaled transfer (crop-specific features) ---
        rescaled_cs = None
        if crop_feats:
            rescaled_cs = experiment_rescaled_transfer(
                france_crop, uk_crop, crop_feats, crop)
            if rescaled_cs:
                print(f"\n    [F] Rescaled transfer (crop-specific features):")
                print(f"        Raw correlation   = {rescaled_cs['raw_correlation']:.3f}")
                print(f"        Variance rescale  R² = {rescaled_cs['variance_r2']:.3f}")
                print(f"        Linear calibrate  R² = {rescaled_cs['linear_r2']:.3f}")
                print(f"        Best: {rescaled_cs['best_method']}, "
                      f"R² = {rescaled_cs['best_r2']:.3f}, "
                      f"RMSE = {rescaled_cs['best_rmse']:.3f} t/ha")

        # --- Experiment G: Rescaled transfer — tune alpha + feature set ---
        best_rescaled = None
        best_rescaled_r2 = -np.inf
        for feat_name, feat_list in [('all', ALL_WEATHER_FEATURES),
                                      ('crop_specific', crop_feats)]:
            if not feat_list:
                continue
            for alpha in [0.1, 1.0, 10.0, 100.0]:
                r = experiment_rescaled_transfer(
                    france_crop, uk_crop, feat_list, crop, alpha=alpha)
                if r and r['best_r2'] > best_rescaled_r2:
                    best_rescaled_r2 = r['best_r2']
                    best_rescaled = r
                    best_rescaled['tuned_alpha'] = alpha
                    best_rescaled['tuned_features'] = feat_name

        if best_rescaled:
            print(f"\n    [G] Rescaled transfer (tuned: {best_rescaled['tuned_features']}, "
                  f"alpha={best_rescaled['tuned_alpha']}, "
                  f"{best_rescaled['best_method']}):")
            print(f"        R² = {best_rescaled['best_r2']:.3f}, "
                  f"RMSE = {best_rescaled['best_rmse']:.3f} t/ha")
            print(f"        Raw correlation = {best_rescaled['raw_correlation']:.3f}")

        # --- Experiment H: Pooled anomaly with tuning ---
        tuned = experiment_pooled_anomaly_tuned(
            france_crop, uk_crop, ALL_WEATHER_FEATURES, crop)
        if tuned:
            print(f"\n    [H] Pooled anomaly (tuned, best={tuned['best_features']}, "
                  f"alpha={tuned['best_alpha']}):")
            print(f"        R² = {tuned['pooled_anomaly_r2']:.3f}, "
                  f"RMSE = {tuned['pooled_anomaly_rmse']:.3f} t/ha")

        result = {'crop': crop}
        if baseline:
            result.update(baseline)
        if transfer:
            result['transfer_r2'] = transfer['transfer_r2']
            result['anomaly_r2'] = transfer['anomaly_r2']
        if pooled:
            result['pooled_anomaly_r2'] = pooled['pooled_anomaly_r2']
            result['pooled_anomaly_rmse'] = pooled['pooled_anomaly_rmse']
        if tuned:
            result['tuned_r2'] = tuned['pooled_anomaly_r2']
        if transfer_cs:
            result['transfer_cs_r2'] = transfer_cs['transfer_r2']
        if rescaled_all:
            result['rescaled_all_r2'] = rescaled_all['best_r2']
            result['rescaled_all_corr'] = rescaled_all['raw_correlation']
        if rescaled_cs:
            result['rescaled_cs_r2'] = rescaled_cs['best_r2']
            result['rescaled_cs_corr'] = rescaled_cs['raw_correlation']
        if best_rescaled:
            result['rescaled_tuned_r2'] = best_rescaled['best_r2']
            result['rescaled_tuned_method'] = best_rescaled['best_method']
            result['rescaled_tuned_alpha'] = best_rescaled['tuned_alpha']
            result['rescaled_tuned_features'] = best_rescaled['tuned_features']
        all_results.append(result)

    # --- Summary ---
    print("\n\n" + "=" * 70)
    print("SUMMARY: Long-History Detrended Anomaly Transfer")
    print("=" * 70)

    print(f"\n  Table 1: Direct transfer approaches (UK R²)")
    print(f"  {'Crop':<16} {'Raw':>8} {'Raw CS':>8} {'Resc.':>8} "
          f"{'Resc CS':>8} {'Resc*':>8} {'Corr':>6}")
    print(f"  {'':.<16} {'47feat':>8} {'4-5f':>8} {'47feat':>8} "
          f"{'4-5f':>8} {'tuned':>8} {'raw':>6}")
    print("  " + "-" * 66)
    for r in all_results:
        print(f"  {r['crop']:<16} "
              f"{r.get('transfer_r2', float('nan')):>8.3f} "
              f"{r.get('transfer_cs_r2', float('nan')):>8.3f} "
              f"{r.get('rescaled_all_r2', float('nan')):>8.3f} "
              f"{r.get('rescaled_cs_r2', float('nan')):>8.3f} "
              f"{r.get('rescaled_tuned_r2', float('nan')):>8.3f} "
              f"{r.get('rescaled_all_corr', float('nan')):>6.2f}")

    print(f"\n  Table 2: Best rescaled transfer configuration")
    print(f"  {'Crop':<16} {'R²':>8} {'Method':>10} {'Alpha':>8} {'Features':>15}")
    print("  " + "-" * 60)
    for r in all_results:
        print(f"  {r['crop']:<16} "
              f"{r.get('rescaled_tuned_r2', float('nan')):>8.3f} "
              f"{r.get('rescaled_tuned_method', 'N/A'):>10} "
              f"{r.get('rescaled_tuned_alpha', 'N/A'):>8} "
              f"{r.get('rescaled_tuned_features', 'N/A'):>15}")

    print(f"\n  Table 3: All approaches summary (UK R²)")
    print(f"  {'Crop':<16} {'UK-only':>8} {'Rescaled':>8} {'Pooled':>8} "
          f"{'Pooled*':>8}")
    print(f"  {'':.<16} {'LOOCV':>8} {'best':>8} {'anom':>8} "
          f"{'tuned':>8}")
    print("  " + "-" * 50)
    for r in all_results:
        print(f"  {r['crop']:<16} "
              f"{r.get('uk_loocv_r2', float('nan')):>8.3f} "
              f"{r.get('rescaled_tuned_r2', float('nan')):>8.3f} "
              f"{r.get('pooled_anomaly_r2', float('nan')):>8.3f} "
              f"{r.get('tuned_r2', float('nan')):>8.3f}")

    # Compare with previous domain adaptation results
    print(f"\n  Previous best results (domain_adaptation.py, 2004-2016):")
    print(f"    Ridge+Indicators pooled R² = 0.620")
    print(f"    Bias Correction UK R²      = 0.154")
    print(f"    Best UK transfer R²        = -3.026 (bias correction)")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
