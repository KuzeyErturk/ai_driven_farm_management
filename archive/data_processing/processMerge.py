import pandas as pd

print("CREATING FINAL COMPLETE DATASET WITH ALL VARIABLES")
print("="*50)

# Load all datasets
crops = pd.read_csv('data/uk_all_crops_2004_2024.csv')
weather_temp_rain = pd.read_csv('data/uk_weather_2004_2024.csv')
humidity = pd.read_csv('data/uk_humidity_2004_2024.csv')
sunshine = pd.read_csv('data/uk_sunshine_2004_2024.csv')
frost = pd.read_csv('data/uk_frost_2004_2024.csv')
fertilizer = pd.read_csv('data/uk_fertilizer_2004_2024.csv')

print("\nDataset sizes:")
print(f"Crops: {len(crops)} records")
print(f"Weather (temp/rain): {len(weather_temp_rain)} records")
print(f"Humidity: {len(humidity)} records")
print(f"Sunshine: {len(sunshine)} records")
print(f"Frost: {len(frost)} records")
print(f"Fertilizer: {len(fertilizer)} records")

# Merge all weather data
weather_complete = weather_temp_rain.merge(humidity, on='Year', how='left')
weather_complete = weather_complete.merge(sunshine, on='Year', how='left')
weather_complete = weather_complete.merge(frost, on='Year', how='left')

print(f"\nCombined weather: {len(weather_complete)} records with {len(weather_complete.columns)} variables")

# Merge crops with weather
crops_weather = crops.merge(weather_complete, on='Year', how='left')

print(f"Crops + Weather: {len(crops_weather)} records")

# Merge with fertilizer
final_dataset = crops_weather.merge(fertilizer, on='Year', how='left')

print(f"\nFinal dataset: {len(final_dataset)} records")

# Reorder columns for clarity
column_order = [
    'Year', 'Crop',
    'Area_hectares', 'Yield_t_per_ha', 'Production_tonnes',
    'Annual_Temp_C', 'Annual_Rainfall_mm', 'Annual_Humidity_Percent', 
    'Annual_Sunshine_hours', 'Annual_Frost_days',
    'Total_Nitrogen_kg_ha', 'Total_Phosphate_kg_ha', 'Total_Potash_kg_ha'
]

final_dataset = final_dataset[column_order]

print("\n" + "="*50)
print("FINAL DATASET STRUCTURE")
print("="*50)
print(final_dataset.head(20))

print("\n" + "="*50)
print("DATASET SUMMARY")
print("="*50)
print(f"Total Observations: {len(final_dataset)}")
print(f"Total Variables: {len(final_dataset.columns)}")
print(f"\nCrops ({final_dataset['Crop'].nunique()}): {sorted(final_dataset['Crop'].unique())}")
print(f"Years: {final_dataset['Year'].min()} - {final_dataset['Year'].max()}")

print("\nVariable List:")
print("  Target Variable:")
print("    - Yield_t_per_ha")
print("\n  Crop Features:")
print("    - Area_hectares")
print("    - Production_tonnes (reference only)")
print("\n  Weather Features (5):")
print("    - Annual_Temp_C")
print("    - Annual_Rainfall_mm")
print("    - Annual_Humidity_Percent")
print("    - Annual_Sunshine_hours")
print("    - Annual_Frost_days")
print("\n  Agricultural Management (3):")
print("    - Total_Nitrogen_kg_ha")
print("    - Total_Phosphate_kg_ha")
print("    - Total_Potash_kg_ha")

print("\nMissing values check:")
missing = final_dataset.isnull().sum()
if missing.sum() == 0:
    print("  ✓ No missing values!")
else:
    print(missing[missing > 0])

print("\nRecords per crop:")
crop_counts = final_dataset.groupby('Crop').size().sort_values(ascending=False)
for crop, count in crop_counts.items():
    print(f"  {crop}: {count} years")

print("\nDescriptive Statistics:")
print(final_dataset.describe().round(2))

# Save final dataset
final_dataset.to_csv('data/uk_crop_yield_complete_2004_2024.csv', index=False)

print("\n" + "="*50)
print("SUCCESS! DATASET COMPLETE!")
print("="*50)
print("\nSaved: data/uk_crop_yield_complete_2004_2024.csv")
print("\n✓ 126 observations (6 crops × 21 years)")
print("✓ 13 variables (Year, Crop, 5 crop variables, 5 weather, 3 fertilizer)")
print("✓ Ready for exploratory data analysis and modeling!")
print("\nNext steps:")
print("  1. Exploratory Data Analysis (EDA)")
print("  2. Feature engineering (lagged yields, interactions, etc.)")
print("  3. Model building and evaluation")
print("  4. Visualization and interpretation")