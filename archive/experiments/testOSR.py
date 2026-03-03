import pandas as pd
import numpy as np

print("="*70)
print("OILSEED RAPE: FINDING BEST FEATURES")
print("="*70)

df = pd.read_csv('data/regional_crop_yield_weather_2004_2024.csv')

osr = df[df['Crop'] == 'Oilseed_Rape'].copy()

print(f"\nOSR Dataset:")
print(f"  Observations: {len(osr)}")
print(f"  Yield: {osr['Yield_t_per_ha'].mean():.1f} ± {osr['Yield_t_per_ha'].std():.1f} t/ha")

# Get correlations with all weather features
weather_cols = [col for col in df.columns if col not in 
                ['Year', 'Region', 'Crop', 'Area_hectares', 'Yield_t_per_ha']]

correlations = osr[weather_cols].corrwith(osr['Yield_t_per_ha']).sort_values(ascending=False)

print(f"\n🌟 TOP 15 CORRELATIONS WITH OSR YIELD:")
print(correlations.head(15))

print(f"\n❄️ BOTTOM 5 (Most Negative):")
print(correlations.tail(5))

# Check by region
print(f"\n" + "="*70)
print("CORRELATIONS BY REGION")
print("="*70)

for region in osr['Region'].unique():
    print(f"\n{region}:")
    region_data = osr[osr['Region'] == region]
    region_corr = region_data[weather_cols].corrwith(region_data['Yield_t_per_ha']).sort_values(ascending=False)
    print(region_corr.head(5))