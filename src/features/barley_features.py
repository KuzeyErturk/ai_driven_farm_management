import pandas as pd
import numpy as np

print("="*70)
print("PROCESSING SPRING & WINTER BARLEY DATA")
print("="*70)

# ============================================================================
# SPRING BARLEY DATA
# ============================================================================

years = list(range(2004, 2025))

spring_barley_data = {
    'England': {
        'Area': [291439,274215,242772,258092,318953,413208,265551,316248,293991,570808,345476,372652,415604,481587,471081,443933,791710,470929,410358,408306,523614],
        'Yield': [5.1,5.4,5.1,5.1,5.4,5.6,5.0,5.2,5.0,5.6,5.8,6.2,5.7,5.5,5.0,6.4,5.4,5.6,5.9,5.1,5.4]
    },
    'Wales': {
        'Area': [15623,14196,12144,12765,14801,17923,13158,14683,14094,15428,14103,15136,13757,14438,15016,11542,13163,12105,12353,11062,14974],
        'Yield': [4.6,4.6,4.7,4.9,4.8,5.3,4.9,5.4,4.4,5.4,5.2,5.4,5.3,4.6,4.6,5.7,5.1,5.2,5.5,4.4,5.0]
    },
    'Scotland': {
        'Area': [257462,243298,220639,226019,262322,287011,242364,262948,289222,296444,274377,255878,238899,243838,250476,242090,258702,248921,244717,249463,257507],
        'Yield': [5.6,5.6,5.7,5.6,5.5,5.5,5.5,5.6,5.0,5.8,6.1,5.9,5.4,5.9,5.5,6.4,6.8,5.8,6.6,6.3,6.5]
    },
    'Northern Ireland': {
        'Area': [22492,21728,18226,18087,19546,21625,17558,17200,20211,20491,16846,15687,14708,14031,14894,11948,12370,12898,13353,13113,13597],
        'Yield': [4.8,4.3,4.6,5.0,4.5,4.9,5.1,5.2,4.7,5.0,5.2,5.5,5.0,4.7,5.2,5.4,5.3,6.0,5.8,4.4,6.0]
    }
}

# ============================================================================
# WINTER BARLEY DATA
# ============================================================================

winter_barley_data = {
    'England': {
        'Area': [350923,321312,322258,318829,345903,346598,320475,298254,328796,256899,363227,375812,375650,360873,335654,387918,253279,345220,371723,390983,325474],
        'Yield': [6.3,6.4,6.6,5.9,6.6,6.3,6.3,6.0,6.4,6.4,7.1,7.6,6.4,6.9,6.8,7.8,6.0,6.6,7.3,7.0,6.2]
    },
    'Wales': {
        'Area': [8643,7570,6985,6750,6748,6352,7279,8097,7731,4966,6580,6926,8132,7351,7670,8253,7473,8482,9069,8898,9039],
        'Yield': [6.3,6.0,5.5,5.5,6.1,6.0,6.2,6.3,5.7,6.3,6.5,7.2,6.2,6.6,6.1,7.6,6.0,6.9,7.0,6.6,5.8]
    },
    'Scotland': {
        'Area': [56348,51341,53762,52625,57612,45149,48010,45477,42816,42694,52507,51808,48031,47509,37542,48802,43091,43246,45659,46358,42874],
        'Yield': [7.2,7.4,7.7,7.3,7.4,7.1,7.0,7.1,6.5,6.6,7.8,7.8,6.8,7.4,7.1,8.2,7.3,7.7,7.9,7.5,7.6]
    },
    'Northern Ireland': {
        'Area': [4519,4012,4599,4730,6149,5120,6767,6848,5323,5266,6709,7021,7628,7114,5809,7739,7734,7944,8604,8340,6841],
        'Yield': [6.9,6.0,6.7,6.2,7.0,6.7,7.3,7.1,6.1,7.0,7.1,7.7,6.8,6.8,6.9,8.1,5.8,7.5,7.0,6.5,6.5]
    }
}

# ============================================================================
# CREATE DATAFRAMES
# ============================================================================

spring_rows = []
winter_rows = []

for region in ['England', 'Wales', 'Scotland', 'Northern Ireland']:
    for i, year in enumerate(years):
        spring_rows.append({
            'Year': year,
            'Region': region,
            'Crop': 'Spring_Barley',
            'Area_hectares': spring_barley_data[region]['Area'][i],
            'Yield_t_per_ha': spring_barley_data[region]['Yield'][i]
        })
        
        winter_rows.append({
            'Year': year,
            'Region': region,
            'Crop': 'Winter_Barley',
            'Area_hectares': winter_barley_data[region]['Area'][i],
            'Yield_t_per_ha': winter_barley_data[region]['Yield'][i]
        })

spring_df = pd.DataFrame(spring_rows)
winter_df = pd.DataFrame(winter_rows)

print(f"\n✓ Spring Barley: {len(spring_df)} observations")
print(f"✓ Winter Barley: {len(winter_df)} observations")

# ============================================================================
# COMPARE YIELDS
# ============================================================================

print("\n" + "="*70)
print("YIELD COMPARISON: SPRING vs WINTER BARLEY")
print("="*70)

print(f"\nSPRING BARLEY:")
print(f"  Overall: {spring_df['Yield_t_per_ha'].mean():.2f} ± {spring_df['Yield_t_per_ha'].std():.2f} t/ha")
print(f"  Train (2004-2019): {spring_df[spring_df['Year'] <= 2019]['Yield_t_per_ha'].mean():.2f} ± {spring_df[spring_df['Year'] <= 2019]['Yield_t_per_ha'].std():.2f}")
print(f"  Test (2020-2024): {spring_df[spring_df['Year'] >= 2020]['Yield_t_per_ha'].mean():.2f} ± {spring_df[spring_df['Year'] >= 2020]['Yield_t_per_ha'].std():.2f}")

print(f"\nWINTER BARLEY:")
print(f"  Overall: {winter_df['Yield_t_per_ha'].mean():.2f} ± {winter_df['Yield_t_per_ha'].std():.2f} t/ha")
print(f"  Train (2004-2019): {winter_df[winter_df['Year'] <= 2019]['Yield_t_per_ha'].mean():.2f} ± {winter_df[winter_df['Year'] <= 2019]['Yield_t_per_ha'].std():.2f}")
print(f"  Test (2020-2024): {winter_df[winter_df['Year'] >= 2020]['Yield_t_per_ha'].mean():.2f} ± {winter_df[winter_df['Year'] >= 2020]['Yield_t_per_ha'].std():.2f}")

print(f"\n💡 KEY DIFFERENCE:")
print(f"  Winter barley yields {winter_df['Yield_t_per_ha'].mean() - spring_df['Yield_t_per_ha'].mean():.2f} t/ha MORE than spring!")

# Check structural shift
spring_change = spring_df[spring_df['Year'] >= 2020]['Yield_t_per_ha'].mean() - spring_df[spring_df['Year'] <= 2019]['Yield_t_per_ha'].mean()
winter_change = winter_df[winter_df['Year'] >= 2020]['Yield_t_per_ha'].mean() - winter_df[winter_df['Year'] <= 2019]['Yield_t_per_ha'].mean()

print(f"\nPOST-2019 CHANGES:")
print(f"  Spring barley: {spring_change:+.2f} t/ha")
print(f"  Winter barley: {winter_change:+.2f} t/ha")

if abs(spring_change) > 0.2 or abs(winter_change) > 0.2:
    print(f"  ⚠️  Structural shift detected!")

# ============================================================================
# SAVE SEPARATE FILES
# ============================================================================

spring_df.to_csv('data/spring_barley_regional.csv', index=False)
winter_df.to_csv('data/winter_barley_regional.csv', index=False)

print("\n✓ Saved: data/spring_barley_regional.csv")
print("\n✓ Saved: data/winter_barley_regional.csv")

# ============================================================================
# COMBINE WITH WEATHER
# ============================================================================

print("\n" + "="*70)
print("MERGING WITH WEATHER DATA")
print("="*70)

# Load weather
weather = pd.read_csv('data/regional_seasonal_weather_features.csv')

# Merge both
spring_full = spring_df.merge(weather, on=['Year', 'Region'], how='inner')
winter_full = winter_df.merge(weather, on=['Year', 'Region'], how='inner')

print(f"\n✓ Spring Barley + Weather: {len(spring_full)} observations, {len(spring_full.columns)} variables")
print(f"✓ Winter Barley + Weather: {len(winter_full)} observations, {len(winter_full.columns)} variables")

spring_full.to_csv('data/spring_barley_with_weather.csv', index=False)
winter_full.to_csv('data/winter_barley_with_weather.csv', index=False)

print("\n✓ Saved: data/spring_barley_with_weather.csv")
print("✓ Saved: data/winter_barley_with_weather.csv")

print("\n" + "="*70)
print("✅ SPRING & WINTER BARLEY DATA READY!")
print("="*70)
print("\nNext: Build separate models for spring and winter barley!")