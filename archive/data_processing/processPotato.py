import pandas as pd

print("PROCESSING POTATO DATA - AREA SOWN (2004-2024)")
print("-" * 50)

# Historical data (2004-2010) - Area Sown
potato_2004_2010 = {
    'Year': [2004, 2005, 2006, 2007, 2008, 2009, 2010],
    'Crop': 'Potato',
    'Area_hectares': [148000, 137000, 140000, 140000, 144000, 144000, 138000],
    'Yield_t_per_ha': [42, 44, 41, 40, 43, 44, 44],
    'Production_tonnes': [6246000, 5979000, 5727000, 5564000, 6132000, 6396000, 6056000]
}

# Current data (2011-2024) - Area Sown
potato_2011_2024 = {
    'Year': [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'Crop': 'Potato',
    'Area_hectares': [146000, 149000, 139000, 141000, 129000, 139000, 145000, 140000, 144000, 142400,
                      137000, 127000, 115000, 118000],
    'Yield_t_per_ha': [43, 31, 42, 42, 44, 39, 43, 36, 37, 39, 37, 40, 41, 46],
    'Production_tonnes': [6310000, 4658000, 5902000, 5911000, 5676000, 5421000, 6235000, 5040000, 
                          5328000, 5553600, 5069000, 5080000, 4715000, 5428000]  # Calculated: Area × Yield
}

hist_df = pd.DataFrame(potato_2004_2010)
curr_df = pd.DataFrame(potato_2011_2024)

potato_complete = pd.concat([hist_df, curr_df], ignore_index=True)

print("\nPotato Data Complete (2004-2024):")
print(potato_complete)

print("\nPotato Statistics:")
print(f"Years: {potato_complete['Year'].min()} - {potato_complete['Year'].max()}")
print(f"Total years: {len(potato_complete)}")
print(f"Yield range: {potato_complete['Yield_t_per_ha'].min():.0f} - {potato_complete['Yield_t_per_ha'].max():.0f} t/ha")
print(f"Mean yield: {potato_complete['Yield_t_per_ha'].mean():.1f} t/ha")

print("\nMissing values:")
print(potato_complete.isnull().sum())

# Save potato data
potato_complete.to_csv('data/uk_potatoes_2004_2024.csv', index=False)
print("\nSaved: data/uk_potatoes_2004_2024.csv")

# Now merge with other crops
print("\n" + "="*50)
print("MERGING WITH OTHER 5 CROPS")
print("="*50)

# Load other crops
other_crops = pd.read_csv('data/uk_crops_complete_2004_2024.csv')

# Combine all 6 crops
all_crops = pd.concat([other_crops, potato_complete], ignore_index=True)

print(f"\nTotal crops: {all_crops['Crop'].nunique()}")
print(f"Crops: {sorted(all_crops['Crop'].unique())}")
print(f"Total records: {len(all_crops)}")

print("\nRecords per crop:")
print(all_crops.groupby('Crop').size())

# Save complete dataset
all_crops.to_csv('data/uk_all_crops_2004_2024.csv', index=False)
print("\nSaved: data/uk_all_crops_2004_2024.csv")

print("\nDATASET COMPLETE!")
print("6 crops × 21 years = 126 observations")