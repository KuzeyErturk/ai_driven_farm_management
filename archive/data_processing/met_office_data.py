import pandas as pd
import numpy as np

# Met Office rainfall data for UK (1990-2013)
# Source: HadUK-Grid 1km gridded climate data
# Last column is annual total (ann)

rainfall_data = {
    'Year': [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
             2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
             2010, 2011, 2012, 2013],
    'annual_rainfall_mm': [1203.6, 1024.9, 1217.3, 1149.4, 1216.6, 1050.4, 935.1,
                           1049.1, 1298.8, 1271.6, 1372.9, 1050.0, 1280.5, 900.6,
                           1207.3, 1079.6, 1171.6, 1192.5, 1292.9, 1207.9, 945.9,
                           1162.7, 1329.6, 1084.0]
}

uk_rainfall = pd.DataFrame(rainfall_data)

print("="*70)
print("UK ANNUAL RAINFALL DATA (1990-2013)")
print("Source: Met Office HadUK-Grid")
print("="*70)

print("\nData:")
print(uk_rainfall)

print("\n" + "="*70)
print("STATISTICS")
print("="*70)
print(f"Years covered: {uk_rainfall['Year'].min()} - {uk_rainfall['Year'].max()}")
print(f"Total years: {len(uk_rainfall)}")
print(f"\nRainfall statistics (mm):")
print(f"  Minimum:  {uk_rainfall['annual_rainfall_mm'].min():.1f} (Year {uk_rainfall.loc[uk_rainfall['annual_rainfall_mm'].idxmin(), 'Year']:.0f})")
print(f"  Maximum:  {uk_rainfall['annual_rainfall_mm'].max():.1f} (Year {uk_rainfall.loc[uk_rainfall['annual_rainfall_mm'].idxmax(), 'Year']:.0f})")
print(f"  Mean:     {uk_rainfall['annual_rainfall_mm'].mean():.1f}")
print(f"  Std Dev:  {uk_rainfall['annual_rainfall_mm'].std():.1f}")
print(f"  Range:    {uk_rainfall['annual_rainfall_mm'].max() - uk_rainfall['annual_rainfall_mm'].min():.1f}")

# Compare to the bad constant data
print("\n" + "="*70)
print("COMPARISON TO OLD DATA")
print("="*70)
print(f"Old data (constant):     1220.0 mm (every year)")
print(f"New data (Met Office):   {uk_rainfall['annual_rainfall_mm'].min():.1f} - {uk_rainfall['annual_rainfall_mm'].max():.1f} mm")
print(f"Coefficient of variation: {(uk_rainfall['annual_rainfall_mm'].std() / uk_rainfall['annual_rainfall_mm'].mean() * 100):.1f}%")
print("\n🎉 NEW DATA SHOWS REAL VARIATION!")

# Identify extreme years
print("\n" + "="*70)
print("NOTABLE YEARS")
print("="*70)
print("\nDriest years:")
driest = uk_rainfall.nsmallest(3, 'annual_rainfall_mm')[['Year', 'annual_rainfall_mm']]
for _, row in driest.iterrows():
    print(f"  {row['Year']:.0f}: {row['annual_rainfall_mm']:.1f} mm")

print("\nWettest years:")
wettest = uk_rainfall.nlargest(3, 'annual_rainfall_mm')[['Year', 'annual_rainfall_mm']]
for _, row in wettest.iterrows():
    print(f"  {row['Year']:.0f}: {row['annual_rainfall_mm']:.1f} mm")

# Save the data
uk_rainfall.to_csv('data/uk_rainfall_metoffice_1990_2013.csv', index=False)
print("\n✅ Saved to: data/uk_rainfall_metoffice_1990_2013.csv")

# Create visualization
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Rainfall over time
ax1 = axes[0]
ax1.plot(uk_rainfall['Year'], uk_rainfall['annual_rainfall_mm'], 
         marker='o', linewidth=2, markersize=6, color='steelblue')
ax1.axhline(y=uk_rainfall['annual_rainfall_mm'].mean(), 
            color='red', linestyle='--', alpha=0.5, label='Mean')
ax1.fill_between(uk_rainfall['Year'], 
                  uk_rainfall['annual_rainfall_mm'], 
                  uk_rainfall['annual_rainfall_mm'].mean(),
                  alpha=0.3)
ax1.set_xlabel('Year')
ax1.set_ylabel('Annual Rainfall (mm)')
ax1.set_title('UK Annual Rainfall 1990-2013 (Met Office Data)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Distribution
ax2 = axes[1]
ax2.hist(uk_rainfall['annual_rainfall_mm'], bins=12, 
         edgecolor='black', color='steelblue', alpha=0.7)
ax2.axvline(x=uk_rainfall['annual_rainfall_mm'].mean(), 
            color='red', linestyle='--', linewidth=2, label='Mean')
ax2.set_xlabel('Annual Rainfall (mm)')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Annual Rainfall')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('uk_rainfall_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved as: uk_rainfall_analysis.png")

print("\n" + "="*70)
print("RAINFALL DATA READY FOR INTEGRATION!")
print("="*70)