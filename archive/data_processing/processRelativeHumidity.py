import pandas as pd


# UK Total humidity data (from regional table + your monthly data)
humidity_data = {
    'Year': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 
             2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'Annual_Humidity_Percent': [81.37, 80.84, 80.25, 81.50, 81.55, 82.15, 81.25, 81.02, 82.80, 80.43,
                                 81.97, 80.23, 81.97, 81.87, 80.13, 81.19, 79.94, 81.40, 78.80, 81.5, 82.3]
}

humidity_df = pd.DataFrame(humidity_data)

print("\nUK Humidity Data (2004-2024):")
print(humidity_df)

print("\nHumidity Statistics:")
print(f"Years: {humidity_df['Year'].min()} - {humidity_df['Year'].max()}")
print(f"Total years: {len(humidity_df)}")
print(f"Min: {humidity_df['Annual_Humidity_Percent'].min():.2f}%")
print(f"Max: {humidity_df['Annual_Humidity_Percent'].max():.2f}%")
print(f"Mean: {humidity_df['Annual_Humidity_Percent'].mean():.2f}%")
print(f"Std: {humidity_df['Annual_Humidity_Percent'].std():.2f}%")

humidity_df.to_csv('data/uk_humidity_2004_2024.csv', index=False)
