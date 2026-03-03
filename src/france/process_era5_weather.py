"""
Process ERA5 Weather Data → 47 Seasonal Features
==================================================
Converts raw ERA5 NetCDF files (or synthetic monthly data) into the same
47 seasonal weather features used in the UK analysis.

Replicates the exact feature engineering logic from:
    src/features/seasonal_features.py (lines 164-282)

Features:
    - Temperature (18): seasonal Tmax/Tmin, means, extremes, GDD, ranges
    - Rainfall (11): seasonal totals, critical periods, deviation, squared
    - Sunshine (7): seasonal totals, critical periods, efficiency ratio
    - Frost (4): seasonal counts, late spring frost
    - Interactions (7): temp*rain, temp*sun, extreme indicators

Usage:
    python src/france/process_era5_weather.py

Input:  data/france/raw/era5_{region}_{year}.nc  (or synthetic .pkl/.csv)
Output: data/france/processed/france_seasonal_weather_features.csv
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    FRANCE_REGIONS, GERMANY_REGIONS, YEAR_START, YEAR_END,
    FRANCE_OPTIMAL_RAINFALL, GERMANY_OPTIMAL_RAINFALL, PATHS
)


MONTH_NAMES = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
               'jul', 'aug', 'sep', 'oct', 'nov', 'dec']


def process_netcdf_to_monthly(raw_dir, region, year):
    """
    Process ERA5 NetCDF file into monthly statistics for one region-year.

    Computes:
    - Monthly mean Tmax (daily max of 6-hourly 2m temperature)
    - Monthly mean Tmin (daily min of 6-hourly 2m temperature)
    - Monthly total precipitation (sum of hourly, converted m→mm)
    - Monthly total sunshine hours (from surface solar radiation)
    - Monthly frost days (days where Tmin < 0°C)
    """
    import xarray as xr

    filepath = os.path.join(raw_dir, f'era5_{region}_{year}.nc')
    if not os.path.exists(filepath):
        return None

    ds = xr.open_dataset(filepath)

    # Spatial average across the region's grid points
    # ERA5 variable names: t2m, tp, ssrd
    t2m = ds['t2m'].mean(dim=['latitude', 'longitude'])  # Kelvin
    tp = ds['tp'].mean(dim=['latitude', 'longitude'])      # meters
    ssrd = ds['ssrd'].mean(dim=['latitude', 'longitude'])  # J/m²

    # Convert to daily statistics
    t2m_celsius = t2m - 273.15  # K → °C

    # Group by day to get daily max/min
    daily_tmax = t2m_celsius.resample(time='1D').max()
    daily_tmin = t2m_celsius.resample(time='1D').min()
    daily_precip = tp.resample(time='1D').sum() * 1000  # m → mm
    daily_ssrd = ssrd.resample(time='1D').sum()

    # Monthly aggregates
    result = {'year': year}
    for m in range(1, 13):
        mn = MONTH_NAMES[m - 1]
        mask = daily_tmax.time.dt.month == m

        if mask.sum() == 0:
            continue

        result[f'tmax_{mn}'] = float(daily_tmax.sel(time=mask).mean())
        result[f'tmin_{mn}'] = float(daily_tmin.sel(time=mask).mean())
        result[f'rain_{mn}'] = float(daily_precip.sel(time=mask).sum())

        # Convert solar radiation to sunshine hours
        # Ratio method: sunshine ≈ SSRD / (max possible SSRD) * day_length
        # Simplified: 1 MJ/m² ≈ 0.45 sunshine hours (empirical for France)
        ssrd_monthly = float(daily_ssrd.sel(time=mask).sum())
        sunshine_hours = ssrd_monthly / 1e6 * 0.45  # J → MJ → hours
        result[f'sun_{mn}'] = sunshine_hours

        # Frost days: count of days where Tmin < 0°C
        frost_days = float((daily_tmin.sel(time=mask) < 0).sum())
        result[f'frost_{mn}'] = frost_days

    ds.close()
    return result


def load_monthly_data(raw_dir, regions=None, pkl_name=None):
    """
    Load monthly weather data. Prefers real E-OBS CSVs over synthetic data.
    Returns dict of {region: DataFrame} with columns like tmax_jan, tmin_jan, etc.
    """
    if regions is None:
        regions = []

    # Prefer real E-OBS data
    eobs_found = {}
    for region in regions:
        eobs_path = os.path.join(raw_dir, f'eobs_monthly_weather_{region}.csv')
        if os.path.exists(eobs_path):
            eobs_found[region] = pd.read_csv(eobs_path)

    if eobs_found:
        print(f"    Loaded REAL E-OBS weather for {len(eobs_found)} regions")
        return eobs_found

    # Fallback: synthetic pickle
    if pkl_name:
        synthetic_pkl = os.path.join(raw_dir, pkl_name)
        if os.path.exists(synthetic_pkl):
            print(f"    Loading SYNTHETIC weather from {pkl_name}...")
            import pickle
            with open(synthetic_pkl, 'rb') as f:
                return pickle.load(f)

    raise FileNotFoundError(
        "No weather data found. Run process_eobs_weather.py or download_era5_weather.py first."
    )


def compute_seasonal_features(tmax_df, tmin_df, rain_df, sun_df, frost_df,
                              optimal_rainfall=None):
    """
    Compute 47 seasonal weather features from monthly data.

    Replicates the exact logic from src/features/seasonal_features.py (lines 164-282).

    Parameters
    ----------
    tmax_df : DataFrame with columns [year, jan, feb, ..., dec]
    tmin_df : same
    rain_df : same (monthly totals in mm)
    sun_df  : same (monthly totals in hours)
    frost_df: same (monthly frost day counts)
    optimal_rainfall : float, annual optimal rainfall for deviation feature
    """
    if optimal_rainfall is None:
        optimal_rainfall = FRANCE_OPTIMAL_RAINFALL

    sf = pd.DataFrame()
    sf['Year'] = tmax_df['year']

    # === TEMPERATURE (18 features) ===

    # Seasonal means
    sf['Winter_Tmax'] = tmax_df[['dec', 'jan', 'feb']].mean(axis=1)
    sf['Spring_Tmax'] = tmax_df[['mar', 'apr', 'may']].mean(axis=1)
    sf['Summer_Tmax'] = tmax_df[['jun', 'jul', 'aug']].mean(axis=1)
    sf['Autumn_Tmax'] = tmax_df[['sep', 'oct', 'nov']].mean(axis=1)

    sf['Winter_Tmin'] = tmin_df[['dec', 'jan', 'feb']].mean(axis=1)
    sf['Spring_Tmin'] = tmin_df[['mar', 'apr', 'may']].mean(axis=1)
    sf['Summer_Tmin'] = tmin_df[['jun', 'jul', 'aug']].mean(axis=1)
    sf['Autumn_Tmin'] = tmin_df[['sep', 'oct', 'nov']].mean(axis=1)

    # Mean temperature
    sf['Spring_Temp_Mean'] = (sf['Spring_Tmax'] + sf['Spring_Tmin']) / 2
    sf['Summer_Temp_Mean'] = (sf['Summer_Tmax'] + sf['Summer_Tmin']) / 2

    # Critical period temperatures
    sf['Flowering_Temp'] = tmax_df[['may', 'jun']].mean(axis=1)
    sf['Grain_Filling_Temp'] = tmax_df[['jun', 'jul']].mean(axis=1)

    # Extreme temperatures
    sf['Summer_Tmax_Peak'] = tmax_df[['jun', 'jul', 'aug']].max(axis=1)
    sf['Spring_Tmin_Coldest'] = tmin_df[['mar', 'apr', 'may']].min(axis=1)

    # Diurnal range
    sf['Spring_Temp_Range'] = sf['Spring_Tmax'] - sf['Spring_Tmin']
    sf['Summer_Temp_Range'] = sf['Summer_Tmax'] - sf['Summer_Tmin']

    # Growing Degree Days (base 0°C, simplified)
    for season, cols in [('Spring', ['mar', 'apr', 'may']), ('Summer', ['jun', 'jul', 'aug'])]:
        gdd = 0
        for month in cols:
            days = 30  # Simplified
            monthly_gdd = days * np.maximum(0, (tmax_df[month] + tmin_df[month]) / 2 - 0)
            gdd = gdd + monthly_gdd
        sf[f'{season}_GDD'] = gdd

    # === RAINFALL (11 features) ===

    # Seasonal totals
    sf['Winter_Rain'] = rain_df[['dec', 'jan', 'feb']].sum(axis=1)
    sf['Spring_Rain'] = rain_df[['mar', 'apr', 'may']].sum(axis=1)
    sf['Summer_Rain'] = rain_df[['jun', 'jul', 'aug']].sum(axis=1)
    sf['Autumn_Rain'] = rain_df[['sep', 'oct', 'nov']].sum(axis=1)

    # Critical periods
    sf['Planting_Rain'] = rain_df[['mar', 'apr']].sum(axis=1)
    sf['Grain_Filling_Rain'] = rain_df[['jun', 'jul']].sum(axis=1)
    sf['Harvest_Rain'] = rain_df[['jul', 'aug']].sum(axis=1)

    # Deviation from optimal
    months_all = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                  'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    sf['Annual_Rain'] = rain_df[months_all].sum(axis=1)
    sf['Rain_Deviation_from_Optimal'] = np.abs(sf['Annual_Rain'] - optimal_rainfall)

    # Squared terms
    sf['Spring_Rain_Squared'] = sf['Spring_Rain'] ** 2
    sf['Summer_Rain_Squared'] = sf['Summer_Rain'] ** 2

    # === SUNSHINE (7 features) ===

    sf['Winter_Sun'] = sun_df[['dec', 'jan', 'feb']].sum(axis=1)
    sf['Spring_Sun'] = sun_df[['mar', 'apr', 'may']].sum(axis=1)
    sf['Summer_Sun'] = sun_df[['jun', 'jul', 'aug']].sum(axis=1)
    sf['Autumn_Sun'] = sun_df[['sep', 'oct', 'nov']].sum(axis=1)

    sf['Grain_Filling_Sun'] = sun_df[['jun', 'jul']].sum(axis=1)
    sf['Flowering_Sun'] = sun_df[['may', 'jun']].sum(axis=1)

    sf['Summer_Sun_per_Rain'] = sf['Summer_Sun'] / (sf['Summer_Rain'] + 1)

    # === FROST (4 features) ===

    sf['Winter_Frost'] = frost_df[['dec', 'jan', 'feb']].sum(axis=1)
    sf['Spring_Frost'] = frost_df[['mar', 'apr', 'may']].sum(axis=1)
    sf['Autumn_Frost'] = frost_df[['sep', 'oct', 'nov']].sum(axis=1)
    sf['Late_Spring_Frost'] = frost_df[['apr', 'may']].sum(axis=1)

    # === INTERACTION / EXTREME FEATURES (7 features) ===

    sf['Spring_Temp_x_Rain'] = sf['Spring_Temp_Mean'] * sf['Spring_Rain']
    sf['Summer_Temp_x_Rain'] = sf['Summer_Temp_Mean'] * sf['Summer_Rain']
    sf['Summer_Temp_x_Sun'] = sf['Summer_Tmax'] * sf['Summer_Sun']

    # Binary extreme indicators (same thresholds as UK for comparability)
    sf['Heat_Stress'] = (sf['Summer_Tmax_Peak'] > 21).astype(int)
    sf['Cold_Spring'] = (sf['Spring_Tmin_Coldest'] < 2).astype(int)
    sf['Extreme_Summer_Rain'] = (sf['Summer_Rain'] > 350).astype(int)
    sf['Drought_Spring'] = (sf['Spring_Rain'] < 180).astype(int)

    return sf


def extract_variable_dfs(monthly_df):
    """
    Convert a monthly DataFrame with columns like tmax_jan, tmin_jan, ...
    into separate DataFrames per variable with columns [year, jan, feb, ..., dec].
    """
    result = {}
    for var_prefix in ['tmax', 'tmin', 'rain', 'sun', 'frost']:
        cols = {'year': monthly_df['year']}
        for mn in MONTH_NAMES:
            col_name = f'{var_prefix}_{mn}'
            if col_name in monthly_df.columns:
                cols[mn] = monthly_df[col_name]
            else:
                cols[mn] = 0.0
        result[var_prefix] = pd.DataFrame(cols)
    return result


def process_country(country_name, regions, optimal_rainfall, raw_dir, out_dir, pkl_name=None):
    """Process weather data for one country, outputting 47 seasonal features."""
    print(f"\n  Processing {country_name}...")
    os.makedirs(out_dir, exist_ok=True)

    # Load monthly data (prefers real E-OBS over synthetic)
    monthly_data = load_monthly_data(raw_dir, regions=regions, pkl_name=pkl_name)

    all_features = []
    for region in regions:
        if region not in monthly_data:
            print(f"    WARNING: No data for {region}, skipping.")
            continue

        monthly_df = monthly_data[region]
        var_dfs = extract_variable_dfs(monthly_df)

        sf = compute_seasonal_features(
            tmax_df=var_dfs['tmax'],
            tmin_df=var_dfs['tmin'],
            rain_df=var_dfs['rain'],
            sun_df=var_dfs['sun'],
            frost_df=var_dfs['frost'],
            optimal_rainfall=optimal_rainfall,
        )
        sf['Region'] = region
        print(f"    {region}: {len(sf)} year-rows, {len(sf.columns) - 2} features")
        all_features.append(sf)

    features_df = pd.concat(all_features, ignore_index=True)
    feature_cols = [c for c in features_df.columns if c not in ('Year', 'Region')]
    features_df = features_df[['Year', 'Region'] + feature_cols]

    n_features = len(feature_cols)
    print(f"    Total features: {n_features} (expected 47)")

    output_name = f'{country_name.lower()}_seasonal_weather_features.csv'
    output_path = os.path.join(out_dir, output_name)
    features_df.to_csv(output_path, index=False)
    print(f"    Saved: {output_path}")

    return features_df


def main():
    print("=" * 70)
    print("PROCESS WEATHER → 47 SEASONAL FEATURES (FRANCE + GERMANY)")
    print("=" * 70)

    raw_dir = PATHS['france_raw']

    # France
    france_features = process_country(
        'France', FRANCE_REGIONS, FRANCE_OPTIMAL_RAINFALL,
        raw_dir, PATHS['france_processed'],
        pkl_name='synthetic_monthly_weather.pkl'
    )

    # Germany
    germany_features = process_country(
        'Germany', GERMANY_REGIONS, GERMANY_OPTIMAL_RAINFALL,
        raw_dir, PATHS['germany_processed'],
        pkl_name='synthetic_monthly_weather_germany.pkl'
    )

    # Summary
    for name, df in [('France', france_features), ('Germany', germany_features)]:
        print(f"\n  {name} summary:")
        print(f"    Shape: {df.shape}")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        summary = df[numeric_cols].mean().round(1)
        for col in ['Summer_Tmax', 'Winter_Tmin', 'Annual_Rain', 'Summer_Sun']:
            if col in summary:
                print(f"    {col}: {summary[col]}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


def get_expected_feature_names():
    """Return the 47 expected feature column names (matching UK schema)."""
    return [
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


if __name__ == '__main__':
    main()
