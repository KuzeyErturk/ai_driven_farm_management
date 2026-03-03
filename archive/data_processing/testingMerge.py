import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("PROCESSING UK CEREAL YIELDS (2004-2024)")
print("-" * 50)

data = {
    'Year': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
             2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'Wheat': [7.8, 8.0, 8.0, 7.2, 8.3, 7.9, 7.7, 7.7, 6.7, 7.4,
              8.6, 9.0, 7.9, 8.3, 7.8, 8.9, 7.0, 7.8, 8.6, 8.1, 7.3],
    'Barley': [5.8, 5.9, 5.9, 5.7, 6.0, 5.8, 5.7, 5.7, 5.5, 5.8,
               6.4, 6.7, 5.9, 6.1, 5.7, 6.9, 5.9, 6.1, 6.6, 6.1, 5.9],
    'Oats': [5.8, 5.8, 6.0, 5.5, 5.8, 5.8, 5.5, 5.6, 5.1, 5.5,
             6.0, 6.1, 5.8, 5.4, 5.0, 5.9, 4.9, 5.6, 5.7, 5.0, 5.4],
    'Oilseed_Rape': [2.9, 3.2, 3.3, 3.1, 3.3, 3.4, 3.5, 3.9, 3.4, 3.0,
                     3.6, 3.9, 3.1, 3.9, 3.4, 3.3, 2.7, 3.2, 3.7, 3.1, 2.8]
}

df = pd.DataFrame(data)

print("\nData loaded successfully")
print(f"Shape: {df.shape[0]} years x {df.shape[1]} variables")
print(f"Years: {df['Year'].min()} - {df['Year'].max()}")
print(f"Crops: {[col for col in df.columns if col != 'Year']}")

print("\nDATA PREVIEW")
print("-" * 50)
print(df.head(10))

print("\nYIELD STATISTICS (tonnes/hectare)")
print("-" * 50)

for crop in ['Wheat', 'Barley', 'Oats', 'Oilseed_Rape']:
    print(f"\n{crop}:")
    print(f"  Mean:   {df[crop].mean():.2f} t/ha")
    print(f"  Min:    {df[crop].min():.2f} t/ha (Year {df.loc[df[crop].idxmin(), 'Year']:.0f})")
    print(f"  Max:    {df[crop].max():.2f} t/ha (Year {df.loc[df[crop].idxmax(), 'Year']:.0f})")
    print(f"  Std:    {df[crop].std():.2f} t/ha")
    print(f"  Range:  {df[crop].max() - df[crop].min():.2f} t/ha")

df_long = df.melt(id_vars=['Year'], 
                   value_vars=['Wheat', 'Barley', 'Oats', 'Oilseed_Rape'],
                   var_name='Crop', 
                   value_name='Yield_t_per_ha')

print("\nCONVERTED TO LONG FORMAT (ML-READY)")
print("-" * 50)
print(f"Total records: {len(df_long)}")
print(f"\nFirst 15 rows:")
print(df_long.head(15))

df.to_csv('data/uk_yields_wide_2004_2024.csv', index=False)
df_long.to_csv('data/uk_yields_long_2004_2024.csv', index=False)

print("\nSAVED FILES")
print("-" * 50)
print("Saved: data/uk_yields_wide_2004_2024.csv")
print("Saved: data/uk_yields_long_2004_2024.csv")

print("\nNEXT STEPS")
print("-" * 50)
print("1. Download Met Office weather data (2004-2024):")
print("   - Annual temperature")
print("   - Annual rainfall")
print("   - Annual humidity (if available)")
print("2. Get pesticide data (2004-2024) if available")
print("3. Merge everything together")
print("4. Ready for modeling")

fig, ax = plt.subplots(figsize=(14, 6))

for crop in ['Wheat', 'Barley', 'Oats', 'Oilseed_Rape']:
    ax.plot(df['Year'], df[crop], marker='o', linewidth=2, 
            label=crop.replace('_', ' '), markersize=6)

ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Yield (tonnes/hectare)', fontsize=12, fontweight='bold')
ax.set_title('UK Crop Yields 2004-2024', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(2003, 2025)

plt.tight_layout()
plt.savefig('uk_yields_2004_2024_overview.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved: uk_yields_2004_2024_overview.png")