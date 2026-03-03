import pandas as pd
import numpy as np

print("="*70)
print("ADDING HUMIDITY + FERTILIZERS TO REGIONAL DATA")
print("="*70)

# ============================================================================
# LOAD EXISTING REGIONAL DATA
# ============================================================================

regional_df = pd.read_csv('data/regional_crop_yield_weather_2004_2024.csv')

print(f"\nCurrent regional data: {len(regional_df)} observations")
print(f"Current variables: {len(regional_df.columns)}")

# ============================================================================
# LOAD NATIONAL DATA (HUMIDITY + FERTILIZERS)
# ============================================================================

# Load the original UK data that has humidity and fertilizers
national_df = pd.read_csv('data/uk_crop_yield_with_seasonal_features_2004_2024.csv')

# Extract unique year-level data (humidity and fertilizers are same for all crops in a year)
national_yearly = national_df[['Year', 'Annual_Humidity_Percent', 
                                 'Total_Nitrogen_kg_ha', 'Total_Phosphate_kg_ha', 
                                 'Total_Potash_kg_ha']].drop_duplicates()

print(f"\nNational yearly data: {len(national_yearly)} years")
print(f"Variables to add: Humidity, N, P, K")

# Check the data
print("\nSample of national data:")
print(national_yearly.head())

# ============================================================================
# MERGE WITH REGIONAL DATA
# ============================================================================

print("\n" + "="*70)
print("MERGING NATIONAL VARIABLES WITH REGIONAL DATA")
print("="*70)

# Merge on Year (these variables are national, same for all regions)
regional_enhanced = regional_df.merge(
    national_yearly,
    on='Year',
    how='left'
)

print(f"\n✓ Enhanced regional data: {len(regional_enhanced)} observations")
print(f"✓ Total variables: {len(regional_enhanced.columns)}")

# Check for missing values
missing = regional_enhanced[['Annual_Humidity_Percent', 'Total_Nitrogen_kg_ha', 
                              'Total_Phosphate_kg_ha', 'Total_Potash_kg_ha']].isnull().sum()

if missing.sum() > 0:
    print("\n⚠️  Missing values:")
    print(missing)
else:
    print("\n✓ No missing values in new variables!")

# Show correlations with yield
print("\n" + "="*70)
print("CORRELATIONS WITH YIELD")
print("="*70)

new_vars = ['Annual_Humidity_Percent', 'Total_Nitrogen_kg_ha', 
            'Total_Phosphate_kg_ha', 'Total_Potash_kg_ha']

print("\nOverall correlations:")
for var in new_vars:
    corr = regional_enhanced[var].corr(regional_enhanced['Yield_t_per_ha'])
    print(f"  {var}: {corr:+.3f}")

# By crop
print("\nCorrelations by crop:")
for crop in regional_enhanced['Crop'].unique():
    print(f"\n{crop}:")
    crop_data = regional_enhanced[regional_enhanced['Crop'] == crop]
    for var in new_vars:
        corr = crop_data[var].corr(crop_data['Yield_t_per_ha'])
        print(f"  {var}: {corr:+.3f}")

# ============================================================================
# SAVE ENHANCED DATASET
# ============================================================================

regional_enhanced.to_csv('data/regional_crop_yield_weather_enhanced.csv', index=False)

print("\n" + "="*70)
print("✓ SAVED: data/regional_crop_yield_weather_enhanced.csv")
print("="*70)

print(f"\nNew dataset:")
print(f"  Observations: {len(regional_enhanced)}")
print(f"  Variables: {len(regional_enhanced.columns)}")
print(f"  New additions: Humidity, N, P, K")

print("\nColumns added:")
print(new_vars)

print("\n" + "="*70)
print("✅ READY TO TEST WITH HUMIDITY + FERTILIZERS!")
print("="*70)