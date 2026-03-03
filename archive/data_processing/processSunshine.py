import pandas as pd

print("PROCESSING UK SUNSHINE DATA (2004-2024)")
print("-" * 50)

# UK sunshine data (annual totals in hours)
sunshine_data = {
    'Year': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014,
             2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'Annual_Sunshine_hours': [1356.8, 1382.2, 1472.4, 1435.8, 1375.4, 1456.2, 1444.7,
                              1397.9, 1330.7, 1410.9, 1416.5, 1445.3, 1417.4, 1369.8,
                              1560.7, 1454.4, 1498.3, 1386.2, 1535.3, 1430.1, 1280.7]
}

sunshine_df = pd.DataFrame(sunshine_data)

print("\nUK Sunshine Data (2004-2024):")
print(sunshine_df)

print("\nSunshine Statistics:")
print(f"Years: {sunshine_df['Year'].min()} - {sunshine_df['Year'].max()}")
print(f"Min: {sunshine_df['Annual_Sunshine_hours'].min():.1f} hours")
print(f"Max: {sunshine_df['Annual_Sunshine_hours'].max():.1f} hours")
print(f"Mean: {sunshine_df['Annual_Sunshine_hours'].mean():.1f} hours")
print(f"Std: {sunshine_df['Annual_Sunshine_hours'].std():.1f} hours")

sunshine_df.to_csv('data/uk_sunshine_2004_2024.csv', index=False)
print("\nSaved: data/uk_sunshine_2004_2024.csv")

print("\n" + "="*50)
print("READY TO MERGE ALL DATA!")
print("="*50)
print("\nYou now have ALL required data:")
print("  ✓ Crops (6 crops, Area, Yield, Production)")
print("  ✓ Temperature (2004-2024)")
print("  ✓ Rainfall (2004-2024)")
print("  ✓ Humidity (2004-2024)")
print("  ✓ Sunshine (2004-2024)")
print("  ✓ Fertilizers (N, P, K, 2004-2024)")
print("\nNext: Merge everything into final dataset!")