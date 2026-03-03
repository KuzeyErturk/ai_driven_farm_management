import pandas as pd

df = pd.read_csv('data/regional_crop_yield_weather_2004_2024.csv')

# Compare correlations
wheat = df[df['Crop'] == 'Wheat']
barley = df[df['Crop'] == 'Barley']

print("WHEAT - Top Correlations with Yield:")
weather_cols = [col for col in df.columns if col not in ['Year', 'Region', 'Crop', 'Area_hectares', 'Yield_t_per_ha']]
wheat_corr = wheat[weather_cols].corrwith(wheat['Yield_t_per_ha']).sort_values(ascending=False)
print(wheat_corr.head(10))

print("\nBARLEY - Top Correlations with Yield:")
barley_corr = barley[weather_cols].corrwith(barley['Yield_t_per_ha']).sort_values(ascending=False)
print(barley_corr.head(10))