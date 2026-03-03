"""
Build France, Germany, and Pooled Datasets
============================================
Merges crop yields with seasonal weather features for each country,
creates final CSVs matching the UK schema, and builds pooled multi-country datasets.

Usage:
    python src/france/build_france_datasets.py

Output:
    data/france/processed/france_regional_crop_yield_weather_2004_2018.csv
    data/germany/processed/germany_regional_crop_yield_weather_2004_2018.csv
    data/pooled/pooled_regional_crop_yield_weather_2004_2018.csv  (UK + France + Germany)
    data/pooled/pooled_spring_barley_with_weather.csv
    data/pooled/pooled_winter_barley_with_weather.csv
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CROPS, YEAR_START, YEAR_END, UK_OPTIMAL_RAINFALL, PATHS


def load_uk_data():
    """Load UK datasets."""
    regional = pd.read_csv(os.path.join(PATHS['uk_processed'],
                                        'regional_crop_yield_weather_2004_2024.csv'))
    spring_barley = pd.read_csv(os.path.join(PATHS['uk_processed'],
                                             'spring_barley_with_weather.csv'))
    winter_barley = pd.read_csv(os.path.join(PATHS['uk_processed'],
                                             'winter_barley_with_weather.csv'))
    return regional, spring_barley, winter_barley


def merge_yields_weather(yields_path, weather_path, label):
    """Merge yields and weather on (Year, Region)."""
    yields = pd.read_csv(yields_path)
    weather = pd.read_csv(weather_path)

    merged = yields.merge(weather, on=['Year', 'Region'], how='inner')
    print(f"  {label}: {len(yields)} yield rows + {len(weather)} weather rows → {len(merged)} merged")
    return merged


def align_to_uk_schema(df, target_columns):
    """Ensure DataFrame has same columns in same order as UK dataset."""
    for col in target_columns:
        if col not in df.columns:
            df[col] = np.nan
    return df[target_columns].copy()


def build_pooled(uk_df, *other_dfs, country_names=None):
    """
    Build pooled dataset from UK + other country DataFrames.
    Adds Country column and recomputes Rain_Deviation_from_Optimal using pooled mean.
    """
    uk = uk_df.copy()
    uk['Country'] = 'UK'

    parts = [uk]
    for i, df in enumerate(other_dfs):
        d = df.copy()
        d['Country'] = country_names[i] if country_names else f'Country_{i}'
        parts.append(d)

    pooled = pd.concat(parts, ignore_index=True)

    # Recompute Rain_Deviation_from_Optimal using pooled mean
    if 'Annual_Rain' in pooled.columns:
        pooled_optimal = pooled['Annual_Rain'].mean()
        pooled['Rain_Deviation_from_Optimal'] = np.abs(pooled['Annual_Rain'] - pooled_optimal)

    # Move Country column after Region
    cols = list(pooled.columns)
    cols.remove('Country')
    region_idx = cols.index('Region') + 1
    cols.insert(region_idx, 'Country')
    pooled = pooled[cols]

    return pooled


def main():
    print("=" * 70)
    print("BUILD FRANCE, GERMANY & POOLED DATASETS")
    print("=" * 70)

    for d in [PATHS['france_processed'], PATHS['germany_processed'], PATHS['pooled']]:
        os.makedirs(d, exist_ok=True)

    # --- Load UK data ---
    print("\n  Loading UK data...")
    uk_regional, uk_spring, uk_winter = load_uk_data()
    target_cols = list(uk_regional.columns)
    print(f"  UK regional: {uk_regional.shape} ({len(target_cols)} columns)")

    # --- Build France dataset ---
    print("\n  Building France dataset...")
    france_merged = merge_yields_weather(
        os.path.join(PATHS['france_processed'], 'france_crop_yields_2004_2018.csv'),
        os.path.join(PATHS['france_processed'], 'france_seasonal_weather_features.csv'),
        'France'
    )
    france_aligned = align_to_uk_schema(france_merged, target_cols)

    france_path = os.path.join(PATHS['france_processed'],
                               'france_regional_crop_yield_weather_2004_2018.csv')
    france_aligned.to_csv(france_path, index=False)
    print(f"  Saved: {france_path} ({len(france_aligned)} rows)")

    # Split barley
    france_spring = france_aligned[france_aligned['Crop'] == 'Spring_Barley'].copy()
    france_winter = france_aligned[france_aligned['Crop'] == 'Winter_Barley'].copy()
    france_spring.to_csv(os.path.join(PATHS['france_processed'],
                                      'france_spring_barley_with_weather.csv'), index=False)
    france_winter.to_csv(os.path.join(PATHS['france_processed'],
                                      'france_winter_barley_with_weather.csv'), index=False)

    # --- Build Germany dataset ---
    print("\n  Building Germany dataset...")
    germany_merged = merge_yields_weather(
        os.path.join(PATHS['germany_processed'], 'germany_crop_yields_2004_2018.csv'),
        os.path.join(PATHS['germany_processed'], 'germany_seasonal_weather_features.csv'),
        'Germany'
    )
    germany_aligned = align_to_uk_schema(germany_merged, target_cols)

    germany_path = os.path.join(PATHS['germany_processed'],
                                'germany_regional_crop_yield_weather_2004_2018.csv')
    germany_aligned.to_csv(germany_path, index=False)
    print(f"  Saved: {germany_path} ({len(germany_aligned)} rows)")

    # Split barley
    germany_spring = germany_aligned[germany_aligned['Crop'] == 'Spring_Barley'].copy()
    germany_winter = germany_aligned[germany_aligned['Crop'] == 'Winter_Barley'].copy()
    germany_spring.to_csv(os.path.join(PATHS['germany_processed'],
                                       'germany_spring_barley_with_weather.csv'), index=False)
    germany_winter.to_csv(os.path.join(PATHS['germany_processed'],
                                       'germany_winter_barley_with_weather.csv'), index=False)

    # --- Build pooled datasets (UK + France + Germany) ---
    print("\n  Building pooled UK+France+Germany datasets...")

    # Truncate UK to overlap period
    uk_reg_trunc = uk_regional[
        (uk_regional['Year'] >= YEAR_START) & (uk_regional['Year'] <= YEAR_END)
    ].copy()
    uk_spring_trunc = uk_spring[
        (uk_spring['Year'] >= YEAR_START) & (uk_spring['Year'] <= YEAR_END)
    ].copy()
    uk_winter_trunc = uk_winter[
        (uk_winter['Year'] >= YEAR_START) & (uk_winter['Year'] <= YEAR_END)
    ].copy()

    print(f"  UK truncated to {YEAR_START}-{YEAR_END}: {len(uk_reg_trunc)} rows")

    # Pooled regional
    pooled_regional = build_pooled(
        uk_reg_trunc, france_aligned, germany_aligned,
        country_names=['France', 'Germany']
    )
    pooled_path = os.path.join(PATHS['pooled'],
                               'pooled_regional_crop_yield_weather_2004_2018.csv')
    pooled_regional.to_csv(pooled_path, index=False)
    print(f"  Saved: {pooled_path} ({len(pooled_regional)} rows)")

    # Pooled barley
    pooled_spring = build_pooled(
        uk_spring_trunc, france_spring, germany_spring,
        country_names=['France', 'Germany']
    )
    pooled_winter = build_pooled(
        uk_winter_trunc, france_winter, germany_winter,
        country_names=['France', 'Germany']
    )
    pooled_spring.to_csv(os.path.join(PATHS['pooled'],
                                      'pooled_spring_barley_with_weather.csv'), index=False)
    pooled_winter.to_csv(os.path.join(PATHS['pooled'],
                                      'pooled_winter_barley_with_weather.csv'), index=False)

    # --- Validation ---
    print("\n" + "-" * 70)
    print("  VALIDATION")
    print("-" * 70)

    print(f"\n  Pooled dataset:")
    print(f"    Countries: {sorted(pooled_regional['Country'].unique())}")
    print(f"    Total rows: {len(pooled_regional)}")
    for country in ['UK', 'France', 'Germany']:
        n = len(pooled_regional[pooled_regional['Country'] == country])
        print(f"    {country}: {n} rows")
    print(f"    Columns: {len(pooled_regional.columns)}")

    print(f"\n  Yield comparison (mean t/ha, {YEAR_START}-{YEAR_END}):")
    for crop in CROPS:
        crop_in_data = 'Oilseed_Rape' if crop == 'Oilseed_Rape' else crop
        vals = {}
        for country, label in [(uk_reg_trunc, 'UK'), (france_aligned, 'FR'), (germany_aligned, 'DE')]:
            sub = country[country['Crop'] == crop_in_data]
            if crop in ['Spring_Barley', 'Winter_Barley']:
                if label == 'UK':
                    sub = uk_spring_trunc if crop == 'Spring_Barley' else uk_winter_trunc
                elif label == 'FR':
                    sub = france_spring if crop == 'Spring_Barley' else france_winter
                else:
                    sub = germany_spring if crop == 'Spring_Barley' else germany_winter
            vals[label] = sub['Yield_t_per_ha'].mean() if len(sub) > 0 else np.nan

        print(f"    {crop:<16} UK: {vals['UK']:5.2f}  FR: {vals['FR']:5.2f}  DE: {vals['DE']:5.2f}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
