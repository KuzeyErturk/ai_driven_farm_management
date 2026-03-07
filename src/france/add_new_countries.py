"""
Add Climatically Similar Countries: Ireland, Netherlands, Belgium, Denmark
===========================================================================
Downloads yield data from Eurostat, processes E-OBS weather for each country,
computes seasonal weather features, and rebuilds the pooled dataset.

These maritime-climate countries are structurally closer to the UK than
France/Germany, which should improve cross-country transfer.

Data sources:
- Yields: Eurostat APRO_CPNH1 (national-level, 2004-2016)
- Weather: E-OBS v31/v32 gridded observations (0.1° resolution)
- Boundaries: NUTS 2021 LEVL_3 GeoJSON

Usage:
    python src/france/add_new_countries.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CROPS, YEAR_START, YEAR_END, PATHS

# ============================================================================
# CONFIG
# ============================================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

NEW_COUNTRIES = {
    'Ireland':     {'code': 'IE', 'nuts_prefix': 'IE', 'optimal_rain': 1100},
    'Netherlands': {'code': 'NL', 'nuts_prefix': 'NL', 'optimal_rain': 800},
    'Belgium':     {'code': 'BE', 'nuts_prefix': 'BE', 'optimal_rain': 850},
    'Denmark':     {'code': 'DK', 'nuts_prefix': 'DK', 'optimal_rain': 700},
}

# Eurostat crop codes → project crop names
# C1110 = Soft wheat, C1310 = Winter barley, C1320 = Spring barley,
# C1410 = Oats, I1110 = Rape and turnip rape seeds
EUROSTAT_CROP_MAP = {
    'Wheat':         'C1110',
    'Winter_Barley': 'C1310',
    'Spring_Barley': 'C1320',
    'Oats':          'C1410',
    'Oilseed_Rape':  'I1110',
}

EOBS_DIR = os.path.join(PROJECT_ROOT, 'data', 'weather', 'eobs')
GADM_PATH = os.path.join(PROJECT_ROOT, 'data', 'weather', 'gadm',
                          'NUTS_RG_01M_2021_4326_LEVL_3.geojson')

MONTH_NAMES = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
               'jul', 'aug', 'sep', 'oct', 'nov', 'dec']


# ============================================================================
# STEP 1: DOWNLOAD YIELDS FROM EUROSTAT
# ============================================================================

def download_eurostat_yields():
    """Download crop yields and areas from Eurostat APRO_CPNH1."""
    import eurostat

    print("\n  Downloading Eurostat APRO_CPNH1...")
    raw = eurostat.get_data('APRO_CPNH1')
    header = raw[0]
    rows = raw[1:]
    df = pd.DataFrame(rows, columns=header)

    country_codes = [c['code'] for c in NEW_COUNTRIES.values()]
    geo_col = 'geo\\TIME_PERIOD'
    df = df[df[geo_col].isin(country_codes)]

    year_cols = [str(y) for y in range(YEAR_START, YEAR_END + 1)]
    all_rows = []

    for country_name, country_info in NEW_COUNTRIES.items():
        code = country_info['code']
        print(f"\n    {country_name} ({code}):")

        for crop_name, eurostat_code in EUROSTAT_CROP_MAP.items():
            # Get yield
            yield_row = df[(df[geo_col] == code) &
                          (df['crops'] == eurostat_code) &
                          (df['strucpro'] == 'YI')]

            # Get area (1000 ha)
            area_row = df[(df[geo_col] == code) &
                         (df['crops'] == eurostat_code) &
                         (df['strucpro'] == 'AR')]

            if len(yield_row) == 0:
                print(f"      {crop_name}: no yield data")
                continue

            yr = yield_row.iloc[0]
            ar = area_row.iloc[0] if len(area_row) > 0 else None

            n_valid = 0
            for y in range(YEAR_START, YEAR_END + 1):
                yval = yr[str(y)]
                if pd.notna(yval) and yval > 0:
                    # Area in 1000 ha → convert to ha
                    area_ha = np.nan
                    if ar is not None:
                        aval = ar[str(y)]
                        if pd.notna(aval) and aval > 0:
                            area_ha = aval * 1000

                    all_rows.append({
                        'Year': y,
                        'Region': country_name,  # Single region per country
                        'Country': country_name,
                        'Crop': crop_name,
                        'Yield_t_per_ha': round(float(yval), 3),
                        'Area_hectares': int(area_ha) if pd.notna(area_ha) else np.nan,
                    })
                    n_valid += 1

            print(f"      {crop_name}: {n_valid}/{YEAR_END - YEAR_START + 1} years, "
                  f"mean={yr[year_cols].astype(float).mean():.2f} t/ha")

    result = pd.DataFrame(all_rows)
    return result


# ============================================================================
# STEP 2: PROCESS E-OBS WEATHER
# ============================================================================

def build_country_masks():
    """Build spatial masks for new countries using NUTS3 boundaries."""
    import geopandas as gpd
    import xarray as xr

    print("\n  Building spatial masks for new countries...")

    nuts3 = gpd.read_file(GADM_PATH)
    sample_path = os.path.join(EOBS_DIR, 'tx_1995-2010.nc')
    ds = xr.open_dataset(sample_path)
    # Crop to relevant area (covering IE, NL, BE, DK)
    lat_sl = slice(50, 58)  # Extended north for Denmark/Ireland
    lon_sl = slice(-11, 16)  # Extended west for Ireland

    lats = ds.latitude.sel(latitude=lat_sl).values
    lons = ds.longitude.sel(longitude=lon_sl).values
    ds.close()

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    masks = {}
    for country_name, info in NEW_COUNTRIES.items():
        prefix = info['nuts_prefix']
        country_nuts = nuts3[nuts3['CNTR_CODE'] == prefix].copy()

        if len(country_nuts) == 0:
            print(f"    WARNING: No NUTS3 regions found for {country_name} ({prefix})")
            continue

        # Dissolve all NUTS3 into single country polygon
        country_geom = country_nuts.dissolve().geometry.iloc[0]

        from shapely import contains_xy
        mask = contains_xy(country_geom, lon_grid, lat_grid)
        n_cells = mask.sum()
        print(f"    {country_name}: {n_cells} grid cells")
        masks[country_name] = mask

    return masks, lats, lons


def compute_monthly_weather(masks, lats, lons):
    """Extract monthly weather from E-OBS for new countries."""
    import xarray as xr

    lat_sl = slice(50, 58)
    lon_sl = slice(-11, 16)

    print("\n  Loading E-OBS data...")

    # Load variables
    datasets = {}
    for var, label in [('tx', 'Tmax'), ('tn', 'Tmin'), ('rr', 'Precipitation')]:
        print(f"    Loading {label}...")
        parts = []
        for period in ['1995-2010', '2011-2025']:
            path = os.path.join(EOBS_DIR, f'{var}_{period}.nc')
            if os.path.exists(path):
                ds = xr.open_dataset(path)
                ds = ds.sel(
                    time=slice(f'{YEAR_START}-01-01', f'{YEAR_END}-12-31'),
                    latitude=lat_sl, longitude=lon_sl
                )
                if len(ds.time) > 0:
                    parts.append(ds)
        datasets[var] = xr.concat(parts, dim='time') if len(parts) > 1 else parts[0]

    # Try radiation
    ds_qq = None
    try:
        path = os.path.join(EOBS_DIR, 'qq_ens_mean_0.1deg_reg_v31.0e.nc')
        if os.path.exists(path):
            print("    Loading Radiation...")
            ds_qq = xr.open_dataset(path)
            ds_qq = ds_qq.sel(
                time=slice(f'{YEAR_START}-01-01', f'{YEAR_END}-12-31'),
                latitude=lat_sl, longitude=lon_sl
            )
    except Exception:
        pass

    tx = datasets['tx']['tx']
    tn = datasets['tn']['tn']
    rr = datasets['rr']['rr']

    all_monthly = {}

    for country_name, mask in masks.items():
        print(f"    Processing {country_name}...")
        mask3d = mask[np.newaxis, :, :]

        # Spatial average within country
        tx_vals = tx.values.copy()
        tx_vals[~np.broadcast_to(mask3d, tx_vals.shape)] = np.nan
        tx_region = np.nanmean(tx_vals, axis=(1, 2))

        tn_vals = tn.values.copy()
        tn_vals[~np.broadcast_to(mask3d, tn_vals.shape)] = np.nan
        tn_region = np.nanmean(tn_vals, axis=(1, 2))

        rr_vals = rr.values.copy()
        rr_vals[~np.broadcast_to(mask3d, rr_vals.shape)] = np.nan
        rr_region = np.nanmean(rr_vals, axis=(1, 2))

        qq_region = None
        if ds_qq is not None:
            try:
                qq = ds_qq['qq']
                qq_vals = qq.values.copy()
                qq_vals[~np.broadcast_to(mask3d, qq_vals.shape)] = np.nan
                qq_region = np.nanmean(qq_vals, axis=(1, 2))
            except Exception:
                pass

        times = pd.DatetimeIndex(tx.time.values)
        daily = pd.DataFrame({
            'time': times, 'tmax': tx_region, 'tmin': tn_region, 'rain': rr_region,
            'frost_flag': (tn_region < 0).astype(float),
        })
        if qq_region is not None:
            daily['sunshine'] = qq_region / 25.0

        daily['year'] = daily['time'].dt.year
        daily['month'] = daily['time'].dt.month

        agg_dict = {
            'tmax': ('tmax', 'mean'), 'tmin': ('tmin', 'mean'),
            'rain': ('rain', 'sum'), 'frost': ('frost_flag', 'sum'),
        }
        if 'sunshine' in daily.columns:
            agg_dict['sunshine'] = ('sunshine', 'sum')

        monthly = daily.groupby(['year', 'month']).agg(**agg_dict).reset_index()

        rows = []
        for year in range(YEAR_START, YEAR_END + 1):
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
                    for prefix in ['tmax', 'tmin', 'rain', 'frost', 'sun']:
                        row[f'{prefix}_{mn}'] = np.nan
            rows.append(row)

        all_monthly[country_name] = pd.DataFrame(rows)

    for ds in datasets.values():
        ds.close()
    if ds_qq is not None:
        ds_qq.close()

    return all_monthly


# ============================================================================
# STEP 3: COMPUTE SEASONAL WEATHER FEATURES
# ============================================================================

def compute_seasonal_features(monthly_df, country_name, optimal_rain):
    """Convert monthly weather to seasonal features matching the UK schema."""
    rows = []
    for _, mrow in monthly_df.iterrows():
        year = int(mrow['year'])
        row = {'Year': year, 'Region': country_name}

        # --- Temperature features ---
        row['Winter_Tmax'] = np.nanmean([mrow['tmax_dec'], mrow['tmax_jan'], mrow['tmax_feb']]) if year > YEAR_START else np.nanmean([mrow['tmax_jan'], mrow['tmax_feb']])
        row['Spring_Tmax'] = np.nanmean([mrow['tmax_mar'], mrow['tmax_apr'], mrow['tmax_may']])
        row['Summer_Tmax'] = np.nanmean([mrow['tmax_jun'], mrow['tmax_jul'], mrow['tmax_aug']])
        row['Autumn_Tmax'] = np.nanmean([mrow['tmax_sep'], mrow['tmax_oct'], mrow['tmax_nov']])

        row['Winter_Tmin'] = np.nanmean([mrow['tmin_dec'], mrow['tmin_jan'], mrow['tmin_feb']]) if year > YEAR_START else np.nanmean([mrow['tmin_jan'], mrow['tmin_feb']])
        row['Spring_Tmin'] = np.nanmean([mrow['tmin_mar'], mrow['tmin_apr'], mrow['tmin_may']])
        row['Summer_Tmin'] = np.nanmean([mrow['tmin_jun'], mrow['tmin_jul'], mrow['tmin_aug']])
        row['Autumn_Tmin'] = np.nanmean([mrow['tmin_sep'], mrow['tmin_oct'], mrow['tmin_nov']])

        row['Spring_Temp_Mean'] = (row['Spring_Tmax'] + row['Spring_Tmin']) / 2
        row['Summer_Temp_Mean'] = (row['Summer_Tmax'] + row['Summer_Tmin']) / 2

        row['Flowering_Temp'] = np.nanmean([mrow['tmax_may'], mrow['tmax_jun'],
                                             mrow['tmin_may'], mrow['tmin_jun']]) / 1  # avg of may-jun
        row['Grain_Filling_Temp'] = np.nanmean([mrow['tmax_jun'], mrow['tmax_jul'],
                                                  mrow['tmin_jun'], mrow['tmin_jul']]) / 1

        row['Summer_Tmax_Peak'] = max(mrow['tmax_jun'], mrow['tmax_jul'], mrow['tmax_aug'])
        row['Spring_Tmin_Coldest'] = min(mrow['tmin_mar'], mrow['tmin_apr'], mrow['tmin_may'])

        row['Spring_Temp_Range'] = row['Spring_Tmax'] - row['Spring_Tmin']
        row['Summer_Temp_Range'] = row['Summer_Tmax'] - row['Summer_Tmin']

        # GDD (Growing Degree Days) — simplified: sum of (Tmean - 5) for months where Tmean > 5
        for season, months in [('Spring', ['mar', 'apr', 'may']), ('Summer', ['jun', 'jul', 'aug'])]:
            gdd = 0
            for m in months:
                tmean = (mrow[f'tmax_{m}'] + mrow[f'tmin_{m}']) / 2
                if tmean > 5:
                    gdd += (tmean - 5) * 30  # ~30 days per month
            row[f'{season}_GDD'] = gdd

        # --- Rainfall features ---
        row['Winter_Rain'] = sum(mrow[f'rain_{m}'] for m in ['dec', 'jan', 'feb']) if year > YEAR_START else sum(mrow[f'rain_{m}'] for m in ['jan', 'feb'])
        row['Spring_Rain'] = sum(mrow[f'rain_{m}'] for m in ['mar', 'apr', 'may'])
        row['Summer_Rain'] = sum(mrow[f'rain_{m}'] for m in ['jun', 'jul', 'aug'])
        row['Autumn_Rain'] = sum(mrow[f'rain_{m}'] for m in ['sep', 'oct', 'nov'])
        row['Planting_Rain'] = sum(mrow[f'rain_{m}'] for m in ['sep', 'oct', 'nov'])
        row['Grain_Filling_Rain'] = sum(mrow[f'rain_{m}'] for m in ['jun', 'jul'])
        row['Harvest_Rain'] = sum(mrow[f'rain_{m}'] for m in ['jul', 'aug'])
        row['Annual_Rain'] = sum(mrow[f'rain_{m}'] for m in MONTH_NAMES)
        row['Rain_Deviation_from_Optimal'] = abs(row['Annual_Rain'] - optimal_rain)
        row['Spring_Rain_Squared'] = row['Spring_Rain'] ** 2
        row['Summer_Rain_Squared'] = row['Summer_Rain'] ** 2

        # --- Sunshine features ---
        sun_available = pd.notna(mrow.get('sun_jan', np.nan))
        if sun_available:
            row['Winter_Sun'] = sum(mrow[f'sun_{m}'] for m in ['dec', 'jan', 'feb']) if year > YEAR_START else sum(mrow[f'sun_{m}'] for m in ['jan', 'feb'])
            row['Spring_Sun'] = sum(mrow[f'sun_{m}'] for m in ['mar', 'apr', 'may'])
            row['Summer_Sun'] = sum(mrow[f'sun_{m}'] for m in ['jun', 'jul', 'aug'])
            row['Autumn_Sun'] = sum(mrow[f'sun_{m}'] for m in ['sep', 'oct', 'nov'])
            row['Grain_Filling_Sun'] = sum(mrow[f'sun_{m}'] for m in ['jun', 'jul'])
            row['Flowering_Sun'] = sum(mrow[f'sun_{m}'] for m in ['may', 'jun'])
            row['Summer_Sun_per_Rain'] = row['Summer_Sun'] / max(row['Summer_Rain'], 1)
        else:
            for k in ['Winter_Sun', 'Spring_Sun', 'Summer_Sun', 'Autumn_Sun',
                       'Grain_Filling_Sun', 'Flowering_Sun', 'Summer_Sun_per_Rain']:
                row[k] = np.nan

        # --- Frost features ---
        row['Winter_Frost'] = sum(mrow[f'frost_{m}'] for m in ['dec', 'jan', 'feb']) if year > YEAR_START else sum(mrow[f'frost_{m}'] for m in ['jan', 'feb'])
        row['Spring_Frost'] = sum(mrow[f'frost_{m}'] for m in ['mar', 'apr', 'may'])
        row['Autumn_Frost'] = sum(mrow[f'frost_{m}'] for m in ['sep', 'oct', 'nov'])
        row['Late_Spring_Frost'] = mrow['frost_may']

        # --- Interaction features ---
        row['Spring_Temp_x_Rain'] = row['Spring_Temp_Mean'] * row['Spring_Rain']
        row['Summer_Temp_x_Rain'] = row['Summer_Temp_Mean'] * row['Summer_Rain']
        row['Summer_Temp_x_Sun'] = row['Summer_Temp_Mean'] * row.get('Summer_Sun', np.nan)

        # --- Stress indicators ---
        row['Heat_Stress'] = max(0, row['Summer_Tmax'] - 30)
        row['Cold_Spring'] = max(0, 5 - row['Spring_Temp_Mean'])
        row['Extreme_Summer_Rain'] = max(0, row['Summer_Rain'] - 300)
        row['Drought_Spring'] = max(0, 100 - row['Spring_Rain'])

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# STEP 4: MERGE AND BUILD POOLED DATASET
# ============================================================================

def merge_yields_weather(yields_df, weather_dfs):
    """Merge yields with seasonal weather features for each country."""
    merged_parts = []

    for country_name in NEW_COUNTRIES:
        if country_name not in weather_dfs:
            continue

        country_yields = yields_df[yields_df['Country'] == country_name].copy()
        weather = weather_dfs[country_name]

        merged = country_yields.merge(weather, on=['Year', 'Region'], how='inner')
        merged_parts.append(merged)

    if not merged_parts:
        return pd.DataFrame()

    return pd.concat(merged_parts, ignore_index=True)


def align_to_uk_schema(df, target_columns):
    """Ensure DataFrame has same columns as UK dataset."""
    for col in target_columns:
        if col not in df.columns:
            df[col] = np.nan
    return df[[c for c in target_columns if c in df.columns]].copy()


def rebuild_pooled_datasets(new_country_data):
    """Rebuild pooled datasets with new countries added."""
    print("\n  Rebuilding pooled datasets...")

    # Load existing datasets
    uk_regional = pd.read_csv(os.path.join(PATHS['uk_processed'],
                                            'regional_crop_yield_weather_2004_2024.csv'))
    uk_spring = pd.read_csv(os.path.join(PATHS['uk_processed'],
                                          'spring_barley_with_weather.csv'))
    uk_winter = pd.read_csv(os.path.join(PATHS['uk_processed'],
                                          'winter_barley_with_weather.csv'))

    france = pd.read_csv(os.path.join(PATHS['france_processed'],
                                       'france_regional_crop_yield_weather_2004_2018.csv'))
    germany = pd.read_csv(os.path.join(PATHS['germany_processed'],
                                        'germany_regional_crop_yield_weather_2004_2018.csv'))

    target_cols = list(uk_regional.columns)

    # Truncate UK to overlap period
    uk_reg = uk_regional[(uk_regional['Year'] >= YEAR_START) & (uk_regional['Year'] <= YEAR_END)].copy()
    uk_spr = uk_spring[(uk_spring['Year'] >= YEAR_START) & (uk_spring['Year'] <= YEAR_END)].copy()
    uk_win = uk_winter[(uk_winter['Year'] >= YEAR_START) & (uk_winter['Year'] <= YEAR_END)].copy()

    # Align new country data
    new_aligned = align_to_uk_schema(new_country_data, target_cols)

    # Add Country column
    uk_reg['Country'] = 'UK'
    france['Country'] = 'France'
    germany['Country'] = 'Germany'

    # Build pooled regional (all crops except Spring/Winter Barley which have separate files)
    parts = [uk_reg, france, germany]
    for country_name in NEW_COUNTRIES:
        country_data = new_aligned[new_country_data['Country'] == country_name].copy()
        country_data['Country'] = country_name
        if len(country_data) > 0:
            parts.append(country_data)

    pooled = pd.concat(parts, ignore_index=True)

    # Recompute Rain_Deviation_from_Optimal
    if 'Annual_Rain' in pooled.columns:
        pooled_optimal = pooled['Annual_Rain'].mean()
        pooled['Rain_Deviation_from_Optimal'] = np.abs(pooled['Annual_Rain'] - pooled_optimal)

    # Ensure Country column is after Region
    cols = list(pooled.columns)
    if 'Country' in cols:
        cols.remove('Country')
        region_idx = cols.index('Region') + 1
        cols.insert(region_idx, 'Country')
        pooled = pooled[cols]

    # Save pooled regional
    pooled_path = os.path.join(PATHS['pooled'],
                                'pooled_regional_crop_yield_weather_2004_2018.csv')
    pooled.to_csv(pooled_path, index=False)
    print(f"    Saved: {pooled_path} ({len(pooled)} rows)")

    # Build pooled barley files
    fr_spring = france[france['Crop'] == 'Spring_Barley']
    fr_winter = france[france['Crop'] == 'Winter_Barley']
    de_spring = germany[germany['Crop'] == 'Spring_Barley']
    de_winter = germany[germany['Crop'] == 'Winter_Barley']

    spring_parts = [uk_spr.assign(Country='UK'), fr_spring, de_spring]
    winter_parts = [uk_win.assign(Country='UK'), fr_winter, de_winter]

    for country_name in NEW_COUNTRIES:
        country_data = new_aligned[new_country_data['Country'] == country_name].copy()
        country_data['Country'] = country_name
        sb = country_data[country_data['Crop'] == 'Spring_Barley']
        wb = country_data[country_data['Crop'] == 'Winter_Barley']
        if len(sb) > 0:
            spring_parts.append(sb)
        if len(wb) > 0:
            winter_parts.append(wb)

    pooled_spring = pd.concat(spring_parts, ignore_index=True)
    pooled_winter = pd.concat(winter_parts, ignore_index=True)

    # Recompute Rain_Deviation
    for pdf in [pooled_spring, pooled_winter]:
        if 'Annual_Rain' in pdf.columns:
            pdf['Rain_Deviation_from_Optimal'] = np.abs(pdf['Annual_Rain'] - pdf['Annual_Rain'].mean())

    pooled_spring.to_csv(os.path.join(PATHS['pooled'],
                                       'pooled_spring_barley_with_weather.csv'), index=False)
    pooled_winter.to_csv(os.path.join(PATHS['pooled'],
                                       'pooled_winter_barley_with_weather.csv'), index=False)
    print(f"    Pooled spring barley: {len(pooled_spring)} rows")
    print(f"    Pooled winter barley: {len(pooled_winter)} rows")

    # Also save per-country processed files
    for country_name in NEW_COUNTRIES:
        country_dir = os.path.join(PROJECT_ROOT, 'data', country_name.lower(), 'processed')
        os.makedirs(country_dir, exist_ok=True)

        country_data = new_aligned[new_country_data['Country'] == country_name].copy()
        country_data['Country'] = country_name
        if len(country_data) > 0:
            path = os.path.join(country_dir,
                                f'{country_name.lower()}_regional_crop_yield_weather_2004_2018.csv')
            country_data.to_csv(path, index=False)
            print(f"    Saved: {path} ({len(country_data)} rows)")

            # Barley splits
            sb = country_data[country_data['Crop'] == 'Spring_Barley']
            wb = country_data[country_data['Crop'] == 'Winter_Barley']
            if len(sb) > 0:
                sb.to_csv(os.path.join(country_dir,
                                       f'{country_name.lower()}_spring_barley_with_weather.csv'), index=False)
            if len(wb) > 0:
                wb.to_csv(os.path.join(country_dir,
                                       f'{country_name.lower()}_winter_barley_with_weather.csv'), index=False)

    return pooled


# ============================================================================
# STEP 5: VALIDATION
# ============================================================================

def validate_pooled(pooled):
    """Print validation summary."""
    print("\n" + "=" * 70)
    print("  VALIDATION")
    print("=" * 70)

    print(f"\n  Countries: {sorted(pooled['Country'].unique())}")
    print(f"  Total rows: {len(pooled)}")

    for country in sorted(pooled['Country'].unique()):
        n = len(pooled[pooled['Country'] == country])
        print(f"    {country}: {n} rows")

    print(f"\n  Yield comparison (mean t/ha, {YEAR_START}-{YEAR_END}):")
    for crop in CROPS:
        crop_in_data = 'Oilseed_Rape' if crop == 'Oilseed_Rape' else crop
        vals = {}
        for country in sorted(pooled['Country'].unique()):
            sub = pooled[(pooled['Country'] == country) & (pooled['Crop'] == crop_in_data)]
            vals[country] = f"{sub['Yield_t_per_ha'].mean():.2f}" if len(sub) > 0 else "N/A"
        row = f"    {crop:<16}"
        for c, v in vals.items():
            row += f" {c}={v}"
        print(row)

    # Weather sanity check
    print(f"\n  Weather sanity check (Summer_Tmax mean):")
    for country in sorted(pooled['Country'].unique()):
        sub = pooled[pooled['Country'] == country]
        if 'Summer_Tmax' in sub.columns and sub['Summer_Tmax'].notna().any():
            print(f"    {country}: {sub['Summer_Tmax'].mean():.1f}°C")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("ADD NEW COUNTRIES: Ireland, Netherlands, Belgium, Denmark")
    print("=" * 70)

    # Step 1: Download yields
    print("\n--- STEP 1: Download yield data from Eurostat ---")
    yields_df = download_eurostat_yields()
    print(f"\n  Total yield records: {len(yields_df)}")

    # Step 2: Process weather
    print("\n--- STEP 2: Process E-OBS weather data ---")
    masks, lats, lons = build_country_masks()
    monthly_data = compute_monthly_weather(masks, lats, lons)

    # Save monthly weather CSVs
    raw_dir = PATHS['france_raw']
    for country_name, monthly_df in monthly_data.items():
        csv_path = os.path.join(raw_dir, f'eobs_monthly_weather_{country_name}.csv')
        monthly_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    # Step 3: Compute seasonal features
    print("\n--- STEP 3: Compute seasonal weather features ---")
    weather_features = {}
    for country_name in NEW_COUNTRIES:
        if country_name not in monthly_data:
            continue
        optimal_rain = NEW_COUNTRIES[country_name]['optimal_rain']
        weather_features[country_name] = compute_seasonal_features(
            monthly_data[country_name], country_name, optimal_rain
        )
        print(f"  {country_name}: {len(weather_features[country_name])} year-records")

    # Step 4: Merge yields + weather
    print("\n--- STEP 4: Merge yields with weather ---")
    merged = merge_yields_weather(yields_df, weather_features)
    print(f"  Merged records: {len(merged)}")

    if len(merged) == 0:
        print("  ERROR: No merged data. Check yield years vs weather years.")
        return

    # Step 5: Rebuild pooled datasets
    print("\n--- STEP 5: Rebuild pooled datasets ---")
    pooled = rebuild_pooled_datasets(merged)

    # Step 6: Validate
    validate_pooled(pooled)

    print("\n" + "=" * 70)
    print("DONE — New countries added to pooled datasets")
    print("=" * 70)


if __name__ == '__main__':
    main()
