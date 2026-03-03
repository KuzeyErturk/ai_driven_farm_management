import pandas as pd

print("PROCESSING UK AIR FROST DATA (2004-2024)")
print("-" * 50)

# UK air frost data (annual days with air frost)
frost_data = {
    'Year': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014,
             2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'Annual_Frost_days': [46.6, 53.6, 51.7, 39.2, 57.9, 56.8, 93.4, 41.3, 54.9, 71.1, 31.7,
                          43.9, 54.0, 48.3, 54.0, 48.1, 37.7, 62.9, 44.0, 45.1, 36.1]
}

frost_df = pd.DataFrame(frost_data)

print("\nUK Air Frost Data (2004-2024):")
print(frost_df)

print("\nFrost Statistics:")
print(f"Years: {frost_df['Year'].min()} - {frost_df['Year'].max()}")
print(f"Min: {frost_df['Annual_Frost_days'].min():.1f} days")
print(f"Max: {frost_df['Annual_Frost_days'].max():.1f} days")
print(f"Mean: {frost_df['Annual_Frost_days'].mean():.1f} days")
print(f"Std: {frost_df['Annual_Frost_days'].std():.1f} days")

frost_df.to_csv('data/uk_frost_2004_2024.csv', index=False)
print("\nSaved: data/uk_frost_2004_2024.csv")

print("\n" + "="*50)
print("ALL WEATHER DATA COLLECTED!")
print("="*50)
print("\nComplete weather variables:")
print("  ✓ Temperature (°C)")
print("  ✓ Rainfall (mm)")
print("  ✓ Humidity (%)")
print("  ✓ Sunshine (hours)")
print("  ✓ Air Frost (days)")
print("\nReady to merge everything!")