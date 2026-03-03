import pandas as pd

print("MERGING ALL UK DATA (2004-2024)")
print("-" * 50)

# Load datasets
yields = pd.read_csv('data/uk_yields_long_2004_2024.csv')
weather = pd.read_csv('data/uk_weather_2004_2024.csv')

print(f"\nYields: {len(yields)} records")
print(f"Weather: {len(weather)} records")

# Merge
uk_complete = yields.merge(weather, on='Year', how='left')

print(f"\nMerged: {len(uk_complete)} records")
print("\nFirst 15 rows:")
print(uk_complete.head(15))

# Check for missing values
print("\nMissing values:")
print(uk_complete.isnull().sum())

# Summary by crop
print("\nRecords per crop:")
print(uk_complete.groupby('Crop').size())

# Save
uk_complete.to_csv('data/uk_complete_2004_2024.csv', index=False)
print("\nSaved: data/uk_complete_2004_2024.csv")

print("\nDataset summary:")
print("-" * 50)
print(f"Years: {uk_complete['Year'].min()} - {uk_complete['Year'].max()}")
print(f"Total years: {uk_complete['Year'].nunique()}")
print(f"Crops: {uk_complete['Crop'].unique()}")
print(f"Total records: {len(uk_complete)}")
print(f"Variables: {uk_complete.columns.tolist()}")

print("\nYield ranges by crop (t/ha):")
for crop in uk_complete['Crop'].unique():
    crop_data = uk_complete[uk_complete['Crop'] == crop]
    print(f"  {crop}: {crop_data['Yield_t_per_ha'].min():.1f} - {crop_data['Yield_t_per_ha'].max():.1f}")

print("\nDataset ready for feature engineering and modeling!")