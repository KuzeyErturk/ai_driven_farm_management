import pandas as pd
import numpy as np

print("="*70)
print("OATS: FINDING BEST FEATURES")
print("="*70)

df = pd.read_csv('data/regional_crop_yield_weather_2004_2024.csv')

oats = df[df['Crop'] == 'Oats'].copy()

print(f"\nOats Dataset:")
print(f"  Observations: {len(oats)}")
print(f"  Yield: {oats['Yield_t_per_ha'].mean():.1f} ± {oats['Yield_t_per_ha'].std():.1f} t/ha")

# Train/test stats
train = oats[oats['Year'] <= 2019]
test = oats[oats['Year'] >= 2020]

print(f"\nYield Statistics:")
print(f"  Train: {train['Yield_t_per_ha'].mean():.2f} ± {train['Yield_t_per_ha'].std():.2f} t/ha")
print(f"  Test:  {test['Yield_t_per_ha'].mean():.2f} ± {test['Yield_t_per_ha'].std():.2f} t/ha")
print(f"  Difference: {test['Yield_t_per_ha'].mean() - train['Yield_t_per_ha'].mean():+.2f} t/ha")

# Get correlations with all weather features
weather_cols = [col for col in df.columns if col not in 
                ['Year', 'Region', 'Crop', 'Area_hectares', 'Yield_t_per_ha']]

correlations = oats[weather_cols].corrwith(oats['Yield_t_per_ha']).sort_values(ascending=False)

print(f"\n🌟 TOP 20 CORRELATIONS WITH OATS YIELD:")
print(correlations.head(20))

print(f"\n❄️ BOTTOM 5 (Most Negative):")
print(correlations.tail(5))

# Current features
current = ['Spring_Rain', 'Rain_Deviation_from_Optimal', 'Flowering_Sun']
print(f"\n📊 CURRENT FEATURES (excluding Area):")
for feat in current:
    corr = correlations.get(feat, 0)
    print(f"  {feat}: {corr:+.3f}")

# Check by region
print(f"\n" + "="*70)
print("CORRELATIONS BY REGION")
print("="*70)

for region in oats['Region'].unique():
    print(f"\n{region}:")
    region_data = oats[oats['Region'] == region]
    region_corr = region_data[weather_cols].corrwith(region_data['Yield_t_per_ha']).sort_values(ascending=False)
    print(region_corr.head(5))