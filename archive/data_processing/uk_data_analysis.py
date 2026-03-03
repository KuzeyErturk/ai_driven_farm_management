import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print("="*60)
print("LOADING DATA")
print("="*60)

# Load the CSV file
df = pd.read_csv('data/yield_df.csv')

print("✅ Data loaded successfully!")
print(f"Total dataset shape: {df.shape}")

# Filter for UK data
uk_df = df[df['Area'] == 'United Kingdom']

print("\n" + "="*60)
print("UK DATA OVERVIEW")
print("="*60)
print(f"UK records found: {len(uk_df)}")
print(f"Years covered: {uk_df['Year'].min()} - {uk_df['Year'].max()}")

print("\n" + "="*60)
print("CROPS GROWN IN UK")
print("="*60)
print("\nItems (crops) in UK data:")
print(uk_df['Item'].unique())
print(f"\nTotal number of different crops: {uk_df['Item'].nunique()}")

print("\n" + "="*60)
print("ALL UK DATA")
print("="*60)
print(uk_df)

print("\n" + "="*60)
print("UK DATA STATISTICS")
print("="*60)
print(uk_df.describe())

print("\n" + "="*60)
print("RECORDS PER CROP")
print("="*60)
crop_counts = uk_df['Item'].value_counts()
print(crop_counts)

print("\n" + "="*60)
print("AVERAGE YIELD BY CROP")
print("="*60)
avg_yield_by_crop = uk_df.groupby('Item')['hg/ha_yield'].mean().sort_values(ascending=False)
print("\nAverage yield (hg/ha) for each crop:")
for crop, yield_val in avg_yield_by_crop.items():
    print(f"{crop}: {yield_val:,.2f} hg/ha")

print("\n" + "="*60)
print("UK CLIMATE & PESTICIDE DATA")
print("="*60)
print(f"\nAverage rainfall: {uk_df['average_rain_fall_mm_per_year'].mean():.2f} mm/year")
print(f"Average temperature: {uk_df['avg_temp'].mean():.2f}°C")
print(f"Average pesticide use: {uk_df['pesticides_tonnes'].mean():.2f} tonnes")

# Create visualizations
print("\n" + "="*60)
print("GENERATING PLOTS")
print("="*60)

# Plot 1: Yield trends over time for each crop
fig, ax = plt.subplots(figsize=(12, 6))
for crop in uk_df['Item'].unique():
    crop_data = uk_df[uk_df['Item'] == crop]
    ax.plot(crop_data['Year'], crop_data['hg/ha_yield'], marker='o', label=crop)

ax.set_xlabel('Year')
ax.set_ylabel('Yield (hg/ha)')
ax.set_title('UK Crop Yields Over Time')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('uk_yield_trends.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved as 'uk_yield_trends.png'")

# Plot 2: Average yield comparison
fig, ax = plt.subplots(figsize=(10, 6))
avg_yield_by_crop.plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
ax.set_xlabel('Average Yield (hg/ha)')
ax.set_ylabel('Crop')
ax.set_title('Average Yield by Crop in UK')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('uk_avg_yield_by_crop.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved as 'uk_avg_yield_by_crop.png'")

# Plot 3: Pesticide use over time
fig, ax = plt.subplots(figsize=(10, 6))
pesticide_by_year = uk_df.groupby('Year')['pesticides_tonnes'].mean()
ax.plot(pesticide_by_year.index, pesticide_by_year.values, marker='o', linewidth=2, color='darkgreen')
ax.set_xlabel('Year')
ax.set_ylabel('Pesticides (tonnes)')
ax.set_title('UK Pesticide Use Over Time')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('uk_pesticide_trends.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved as 'uk_pesticide_trends.png'")

# Save UK data to a separate CSV file
uk_df.to_csv('uk_data_only.csv', index=False)
print("\n✅ UK data saved to 'uk_data_only.csv'")

print("\n" + "="*60)
print("UK DATA ANALYSIS COMPLETE!")
print("="*60)
