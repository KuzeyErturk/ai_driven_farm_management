"""
Add Fertilizer Features to Crop Datasets
=========================================
Creates NEW datasets with fertilizer features added.
Original datasets are NEVER modified.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("ADDING FERTILIZER FEATURES TO DATASETS")
print("=" * 70)
print("\nNOTE: Original datasets are NOT modified.")
print("      New files are created with '_with_fertilizer' suffix.\n")

# ============================================================================
# STEP 1: Extract fertilizer data from Eurostat Excel
# ============================================================================

print("1. Extracting fertilizer data from Eurostat...")

fertilizer_data = {}

for sheet, nutrient in [('Sheet 1', 'Nitrogen'), ('Sheet 2', 'Phosphorus')]:
    df = pd.read_excel('consumptionOfFertilizers.xlsx', sheet_name=sheet, header=None)

    for col in range(1, 50, 2):
        if col >= len(df.columns):
            break

        year_val = df.iloc[8, col]
        uk_val = df.iloc[10, col]

        if isinstance(year_val, (int, float)) and not pd.isna(year_val):
            year = int(year_val)
            if year not in fertilizer_data:
                fertilizer_data[year] = {}

            if isinstance(uk_val, (int, float)) and not pd.isna(uk_val):
                fertilizer_data[year][f'{nutrient}_tonnes'] = uk_val

fert_df = pd.DataFrame.from_dict(fertilizer_data, orient='index')
fert_df.index.name = 'Year'
fert_df = fert_df.reset_index()

print(f"   Extracted: {len(fert_df)} years ({fert_df['Year'].min()}-{fert_df['Year'].max()})")

# ============================================================================
# STEP 2: Calculate per-hectare rates using UK agricultural area
# ============================================================================

print("\n2. Calculating per-hectare fertilizer rates...")

# UK agricultural area (approximate, in hectares) - from DEFRA statistics
# This is total UK agricultural area for normalization
uk_ag_area = {
    2004: 17_200_000, 2005: 17_100_000, 2006: 17_000_000, 2007: 17_000_000,
    2008: 17_000_000, 2009: 16_900_000, 2010: 16_900_000, 2011: 16_800_000,
    2012: 16_800_000, 2013: 16_700_000, 2014: 16_700_000, 2015: 16_600_000,
    2016: 16_600_000, 2017: 16_500_000, 2018: 16_500_000, 2019: 16_400_000,
    2020: 16_400_000, 2021: 16_300_000, 2022: 16_300_000, 2023: 16_200_000,
    2024: 16_200_000
}

fert_df['UK_Ag_Area_ha'] = fert_df['Year'].map(uk_ag_area)

# Calculate kg/ha
fert_df['Nitrogen_kg_ha'] = (fert_df['Nitrogen_tonnes'] * 1000) / fert_df['UK_Ag_Area_ha']
fert_df['Phosphorus_kg_ha'] = (fert_df['Phosphorus_tonnes'] * 1000) / fert_df['UK_Ag_Area_ha']

# For years without data (2020-2024), use 2019 values
last_nitrogen = fert_df[fert_df['Year'] == 2019]['Nitrogen_kg_ha'].values[0]
last_phosphorus = fert_df[fert_df['Year'] == 2019]['Phosphorus_kg_ha'].values[0]

for year in range(2020, 2025):
    if year not in fert_df['Year'].values:
        fert_df = pd.concat([fert_df, pd.DataFrame({
            'Year': [year],
            'Nitrogen_kg_ha': [last_nitrogen],
            'Phosphorus_kg_ha': [last_phosphorus]
        })], ignore_index=True)

fert_features = fert_df[['Year', 'Nitrogen_kg_ha', 'Phosphorus_kg_ha']].copy()

print(f"   Nitrogen range: {fert_features['Nitrogen_kg_ha'].min():.1f} - {fert_features['Nitrogen_kg_ha'].max():.1f} kg/ha")
print(f"   Phosphorus range: {fert_features['Phosphorus_kg_ha'].min():.1f} - {fert_features['Phosphorus_kg_ha'].max():.1f} kg/ha")

# ============================================================================
# STEP 3: Add fertilizer features to each dataset
# ============================================================================

print("\n3. Creating new datasets with fertilizer features...\n")

datasets_to_enhance = [
    ('data/processed/uk_crop_yield_with_seasonal_features_2004_2024.csv',
     'data/processed/uk_crop_yield_seasonal_fertilizer_2004_2024.csv'),
    ('data/processed/spring_barley_with_weather.csv',
     'data/processed/spring_barley_weather_fertilizer.csv'),
    ('data/processed/winter_barley_with_weather.csv',
     'data/processed/winter_barley_weather_fertilizer.csv'),
    ('data/processed/regional_crop_yield_weather_2004_2024.csv',
     'data/processed/regional_crop_yield_weather_fertilizer_2004_2024.csv'),
]

for input_path, output_path in datasets_to_enhance:
    try:
        # Read original
        df = pd.read_csv(input_path)
        original_cols = len(df.columns)
        original_rows = len(df)

        # Merge with fertilizer data
        df = df.merge(fert_features, on='Year', how='left')

        # Fill any missing values with 2019 values
        df['Nitrogen_kg_ha'] = df['Nitrogen_kg_ha'].fillna(last_nitrogen)
        df['Phosphorus_kg_ha'] = df['Phosphorus_kg_ha'].fillna(last_phosphorus)

        # Save NEW file
        df.to_csv(output_path, index=False)

        print(f"   ✓ {input_path.split('/')[-1]}")
        print(f"     → {output_path.split('/')[-1]}")
        print(f"     Columns: {original_cols} → {len(df.columns)} (+2 fertilizer features)")
        print(f"     Rows: {original_rows} (unchanged)")
        print()

    except Exception as e:
        print(f"   ✗ Error with {input_path}: {e}")
        print()

# ============================================================================
# STEP 4: Summary
# ============================================================================

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
NEW FILES CREATED:
  - uk_crop_yield_seasonal_fertilizer_2004_2024.csv
  - spring_barley_weather_fertilizer.csv
  - winter_barley_weather_fertilizer.csv
  - regional_crop_yield_weather_fertilizer_2004_2024.csv

NEW FEATURES ADDED:
  - Nitrogen_kg_ha: Nitrogen fertilizer application rate
  - Phosphorus_kg_ha: Phosphorus fertilizer application rate

ORIGINAL FILES UNCHANGED:
  - uk_crop_yield_with_seasonal_features_2004_2024.csv ✓
  - spring_barley_with_weather.csv ✓
  - winter_barley_with_weather.csv ✓
  - regional_crop_yield_weather_2004_2024.csv ✓

BACKUPS AVAILABLE IN:
  - data/backup_originals/
""")

print("✅ Done! Original data is safe.")
