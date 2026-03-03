"""
Integrate Eurostat Fertilizer Data into Crop Yield Dataset
===========================================================
Adds nitrogen and phosphorus consumption as features.
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def extract_fertilizer_data(excel_path):
    """Extract UK fertilizer consumption from Eurostat Excel file."""

    fertilizer_data = {}

    # Sheet 1: Nitrogen, Sheet 2: Phosphorus
    for sheet, nutrient in [('Sheet 1', 'Nitrogen'), ('Sheet 2', 'Phosphorus')]:
        df = pd.read_excel(excel_path, sheet_name=sheet, header=None)

        # Get years from row 8 (columns 1, 3, 5, ... contain years)
        # Get UK data from row 10
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

    # Convert to DataFrame
    fert_df = pd.DataFrame.from_dict(fertilizer_data, orient='index')
    fert_df.index.name = 'Year'
    fert_df = fert_df.reset_index()

    return fert_df


def calculate_per_hectare(fert_df, crop_df):
    """Calculate fertilizer application rate per hectare."""

    # Get total UK agricultural area per year from crop data
    # Sum area across all crops and regions for each year
    area_by_year = crop_df.groupby('Year')['Area_hectares'].sum().reset_index()
    area_by_year.columns = ['Year', 'Total_UK_Area_ha']

    # Merge with fertilizer data
    fert_df = fert_df.merge(area_by_year, on='Year', how='left')

    # Calculate kg per hectare (convert tonnes to kg)
    if 'Nitrogen_tonnes' in fert_df.columns:
        fert_df['Nitrogen_kg_ha'] = (fert_df['Nitrogen_tonnes'] * 1000) / fert_df['Total_UK_Area_ha']

    if 'Phosphorus_tonnes' in fert_df.columns:
        fert_df['Phosphorus_kg_ha'] = (fert_df['Phosphorus_tonnes'] * 1000) / fert_df['Total_UK_Area_ha']

    return fert_df


def integrate_fertilizer_features(crop_df, fert_df):
    """Add fertilizer features to crop dataset."""

    # Select only the per-hectare columns
    fert_features = fert_df[['Year', 'Nitrogen_kg_ha', 'Phosphorus_kg_ha']].copy()

    # Merge with crop data
    enhanced_df = crop_df.merge(fert_features, on='Year', how='left')

    # Handle missing years (2020-2024 don't have fertilizer data)
    # Forward fill from last known value
    for col in ['Nitrogen_kg_ha', 'Phosphorus_kg_ha']:
        if col in enhanced_df.columns:
            # Get the last known value (2019)
            last_known = fert_df[fert_df['Year'] == 2019][col].values
            if len(last_known) > 0:
                enhanced_df[col] = enhanced_df[col].fillna(last_known[0])

    return enhanced_df


def main():
    print("=" * 60)
    print("FERTILIZER DATA INTEGRATION")
    print("=" * 60)

    # Paths
    fertilizer_path = 'consumptionOfFertilizers.xlsx'
    crop_data_path = 'data/processed/regional_crop_yield_weather_2004_2024.csv'
    output_path = 'data/processed/regional_crop_yield_weather_fertilizer_2004_2024.csv'

    # Load existing crop data
    print("\n1. Loading existing crop yield dataset...")
    crop_df = pd.read_csv(crop_data_path)
    print(f"   Loaded {len(crop_df)} rows, {len(crop_df.columns)} columns")
    print(f"   Years: {crop_df['Year'].min()} - {crop_df['Year'].max()}")

    # Extract fertilizer data
    print("\n2. Extracting fertilizer data from Eurostat...")
    fert_df = extract_fertilizer_data(fertilizer_path)
    print(f"   Extracted {len(fert_df)} years of fertilizer data")
    print(f"   Years: {fert_df['Year'].min()} - {fert_df['Year'].max()}")
    print(f"\n   Nitrogen (tonnes): {fert_df['Nitrogen_tonnes'].min():,.0f} - {fert_df['Nitrogen_tonnes'].max():,.0f}")
    print(f"   Phosphorus (tonnes): {fert_df['Phosphorus_tonnes'].min():,.0f} - {fert_df['Phosphorus_tonnes'].max():,.0f}")

    # Calculate per hectare rates
    print("\n3. Calculating per-hectare application rates...")
    fert_df = calculate_per_hectare(fert_df, crop_df)

    # Show rates for overlapping years
    overlap = fert_df[fert_df['Year'].between(2004, 2019)]
    print(f"\n   Nitrogen (kg/ha): {overlap['Nitrogen_kg_ha'].min():.1f} - {overlap['Nitrogen_kg_ha'].max():.1f}")
    print(f"   Phosphorus (kg/ha): {overlap['Phosphorus_kg_ha'].min():.1f} - {overlap['Phosphorus_kg_ha'].max():.1f}")

    # Integrate features
    print("\n4. Integrating fertilizer features into dataset...")
    enhanced_df = integrate_fertilizer_features(crop_df, fert_df)
    print(f"   New dataset: {len(enhanced_df)} rows, {len(enhanced_df.columns)} columns")
    print(f"   New features: Nitrogen_kg_ha, Phosphorus_kg_ha")

    # Save enhanced dataset
    print(f"\n5. Saving enhanced dataset to {output_path}...")
    enhanced_df.to_csv(output_path, index=False)
    print("   Done!")

    # Show sample of new features
    print("\n" + "=" * 60)
    print("SAMPLE OF NEW FEATURES")
    print("=" * 60)
    sample = enhanced_df[['Year', 'Region', 'Crop', 'Yield_t_per_ha', 'Nitrogen_kg_ha', 'Phosphorus_kg_ha']].drop_duplicates(subset=['Year']).head(10)
    print(sample.to_string(index=False))

    # Quick correlation check
    print("\n" + "=" * 60)
    print("CORRELATION WITH YIELD")
    print("=" * 60)
    corr_n = enhanced_df['Yield_t_per_ha'].corr(enhanced_df['Nitrogen_kg_ha'])
    corr_p = enhanced_df['Yield_t_per_ha'].corr(enhanced_df['Phosphorus_kg_ha'])
    print(f"   Nitrogen vs Yield: {corr_n:.3f}")
    print(f"   Phosphorus vs Yield: {corr_p:.3f}")

    return enhanced_df


if __name__ == '__main__':
    main()
