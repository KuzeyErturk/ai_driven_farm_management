import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("INTEGRATING ALL UK AGRICULTURAL DATA")
print("="*70)

# ==================== LOAD ALL DATA SOURCES ====================
print("\n1. Loading data from all sources...")

# Yield data (from yield_df.csv)
yield_df = pd.read_csv('data/yield_df.csv')
uk_yield = yield_df[yield_df['Area'] == 'United Kingdom'].copy()

# Temperature data (from temp.csv - annual values)
temp_df = pd.read_csv('data/temp.csv')
uk_temp = temp_df[temp_df['country'] == 'United Kingdom'].copy()
uk_temp = uk_temp[(uk_temp['year'] >= 1990) & (uk_temp['year'] <= 2013)]
uk_temp = uk_temp.rename(columns={'year': 'Year'})

# Pesticides data (from pesticides.csv)
pest_df = pd.read_csv('data/pesticides.csv')
uk_pest = pest_df[pest_df['Area'] == 'United Kingdom'].copy()
uk_pest = uk_pest[(uk_pest['Year'] >= 1990) & (uk_pest['Year'] <= 2013)]
uk_pest = uk_pest[['Year', 'Value']].rename(columns={'Value': 'pesticides_tonnes'})

# Rainfall data (Met Office - just processed!)
uk_rainfall = pd.read_csv('data/uk_rainfall_metoffice_1990_2013.csv')

print(f"  ✓ Yield data: {len(uk_yield)} records")
print(f"  ✓ Temperature data: {len(uk_temp)} records")
print(f"  ✓ Pesticides data: {len(uk_pest)} records")
print(f"  ✓ Rainfall data: {len(uk_rainfall)} records")

# ==================== PREPARE YIELD DATA ====================
print("\n2. Processing yield data...")

# Aggregate the 5 records per year-crop to annual level
uk_yield_annual = uk_yield.groupby(['Year', 'Item']).agg({
    'hg/ha_yield': 'mean'  # Average the 5 sub-annual measurements
}).reset_index()

uk_yield_annual = uk_yield_annual.rename(columns={'hg/ha_yield': 'yield_hg_per_ha'})

print(f"  ✓ Aggregated to annual level: {len(uk_yield_annual)} records")
print(f"  ✓ Years: {uk_yield_annual['Year'].min()} - {uk_yield_annual['Year'].max()}")
print(f"  ✓ Crops: {uk_yield_annual['Item'].unique()}")

# ==================== MERGE ALL DATA ====================
print("\n3. Merging all datasets...")

# Start with yield data as base
uk_merged = uk_yield_annual.copy()

# Merge temperature (annual values from temp.csv)
uk_merged = uk_merged.merge(
    uk_temp[['Year', 'avg_temp']], 
    on='Year', 
    how='left',
    suffixes=('', '_from_temp')
)

# Merge rainfall (Met Office data)
uk_merged = uk_merged.merge(
    uk_rainfall[['Year', 'annual_rainfall_mm']], 
    on='Year', 
    how='left'
)

# Merge pesticides
uk_merged = uk_merged.merge(
    uk_pest, 
    on='Year', 
    how='left'
)

print(f"  ✓ Merged dataset: {len(uk_merged)} records")
print(f"  ✓ Columns: {uk_merged.columns.tolist()}")

# ==================== CHECK DATA QUALITY ====================
print("\n4. Data quality check...")

print(f"\nMissing values:")
print(uk_merged.isnull().sum())

print(f"\nData summary:")
print(uk_merged.describe())

# ==================== CREATE ENGINEERED FEATURES ====================
print("\n5. Creating engineered features...")

# Sort by crop and year
uk_merged = uk_merged.sort_values(['Item', 'Year']).reset_index(drop=True)

# Feature 1: Lagged yield (previous year)
uk_merged['yield_lag1'] = uk_merged.groupby('Item')['yield_hg_per_ha'].shift(1)

# Feature 2: Temperature deviation from mean
temp_mean = uk_merged['avg_temp'].mean()
uk_merged['temp_deviation'] = uk_merged['avg_temp'] - temp_mean

# Feature 3: Rainfall deviation from mean
rain_mean = uk_merged['annual_rainfall_mm'].mean()
uk_merged['rainfall_deviation'] = uk_merged['annual_rainfall_mm'] - rain_mean

# Feature 4: Extreme rainfall indicators
rain_std = uk_merged['annual_rainfall_mm'].std()
uk_merged['rainfall_drought'] = (uk_merged['annual_rainfall_mm'] < rain_mean - rain_std).astype(int)
uk_merged['rainfall_flood'] = (uk_merged['annual_rainfall_mm'] > rain_mean + rain_std).astype(int)

# Feature 5: Growing degree days approximation (simplified)
# Assuming base temp of 5°C for crop growth
uk_merged['growing_degree_days'] = (uk_merged['avg_temp'] - 5) * 365

# Feature 6: Year as numeric (for trend)
uk_merged['year_numeric'] = uk_merged['Year'] - uk_merged['Year'].min()

# Feature 7: Crop type binary
uk_merged['is_wheat'] = (uk_merged['Item'] == 'Wheat').astype(int)

# Feature 8: Pesticide change from previous year
uk_merged['pesticides_change'] = uk_merged['pesticides_tonnes'].diff()

# Feature 9: Temperature-rainfall interaction
uk_merged['temp_rain_interaction'] = uk_merged['avg_temp'] * uk_merged['annual_rainfall_mm'] / 1000

print(f"  ✓ Created {len(uk_merged.columns) - 4} additional features")

# ==================== SAVE DATASETS ====================
print("\n6. Saving datasets...")

# Save complete dataset (with missing lagged values)
uk_merged.to_csv('data/uk_complete_dataset.csv', index=False)
print(f"  ✓ Saved complete dataset: data/uk_complete_dataset.csv")

# Save dataset without missing values (ready for ML)
uk_merged_clean = uk_merged.dropna().copy()
uk_merged_clean.to_csv('data/uk_ml_ready.csv', index=False)
print(f"  ✓ Saved ML-ready dataset: data/uk_ml_ready.csv ({len(uk_merged_clean)} records)")

# Save separate files for each crop
for crop in uk_merged_clean['Item'].unique():
    crop_data = uk_merged_clean[uk_merged_clean['Item'] == crop]
    filename = f"data/uk_{crop.lower()}_data.csv"
    crop_data.to_csv(filename, index=False)
    print(f"  ✓ Saved {crop} data: {filename} ({len(crop_data)} records)")

# ==================== SUMMARY STATISTICS ====================
print("\n" + "="*70)
print("FINAL DATASET SUMMARY")
print("="*70)

print(f"\nTotal records: {len(uk_merged_clean)}")
print(f"Years covered: {uk_merged_clean['Year'].min()} - {uk_merged_clean['Year'].max()}")
print(f"Number of years: {uk_merged_clean['Year'].nunique()}")
print(f"Crops: {uk_merged_clean['Item'].unique()}")
print(f"Records per crop: {uk_merged_clean.groupby('Item').size().to_dict()}")

print(f"\nFeatures available:")
for col in uk_merged_clean.columns:
    print(f"  - {col}")

print(f"\n" + "="*70)
print("VARIABLE RANGES")
print("="*70)

print(f"\nYield (hg/ha):")
for crop in uk_merged_clean['Item'].unique():
    crop_data = uk_merged_clean[uk_merged_clean['Item'] == crop]
    print(f"  {crop}: {crop_data['yield_hg_per_ha'].min():.0f} - {crop_data['yield_hg_per_ha'].max():.0f} (mean: {crop_data['yield_hg_per_ha'].mean():.0f})")

print(f"\nTemperature (°C): {uk_merged_clean['avg_temp'].min():.2f} - {uk_merged_clean['avg_temp'].max():.2f}")
print(f"Rainfall (mm): {uk_merged_clean['annual_rainfall_mm'].min():.1f} - {uk_merged_clean['annual_rainfall_mm'].max():.1f}")
print(f"Pesticides (tonnes): {uk_merged_clean['pesticides_tonnes'].min():.0f} - {uk_merged_clean['pesticides_tonnes'].max():.0f}")

# ==================== CREATE VISUALIZATIONS ====================
print("\n7. Creating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('UK Agricultural Data Overview (1990-2013)', fontsize=16, fontweight='bold')

# Plot 1: Yields over time
ax1 = axes[0, 0]
for crop in uk_merged_clean['Item'].unique():
    crop_data = uk_merged_clean[uk_merged_clean['Item'] == crop]
    ax1.plot(crop_data['Year'], crop_data['yield_hg_per_ha'], marker='o', label=crop, linewidth=2)
ax1.set_xlabel('Year')
ax1.set_ylabel('Yield (hg/ha)')
ax1.set_title('Crop Yields Over Time')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Temperature over time
ax2 = axes[0, 1]
ax2.plot(uk_merged_clean[uk_merged_clean['Item'] == 'Wheat']['Year'], 
         uk_merged_clean[uk_merged_clean['Item'] == 'Wheat']['avg_temp'],
         marker='o', color='red', linewidth=2)
ax2.set_xlabel('Year')
ax2.set_ylabel('Temperature (°C)')
ax2.set_title('Annual Temperature')
ax2.grid(True, alpha=0.3)

# Plot 3: Rainfall over time
ax3 = axes[0, 2]
ax3.plot(uk_merged_clean[uk_merged_clean['Item'] == 'Wheat']['Year'],
         uk_merged_clean[uk_merged_clean['Item'] == 'Wheat']['annual_rainfall_mm'],
         marker='o', color='blue', linewidth=2)
ax3.axhline(y=rain_mean, color='red', linestyle='--', alpha=0.5, label='Mean')
ax3.set_xlabel('Year')
ax3.set_ylabel('Rainfall (mm)')
ax3.set_title('Annual Rainfall (Met Office)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Yield vs Temperature
ax4 = axes[1, 0]
for crop in uk_merged_clean['Item'].unique():
    crop_data = uk_merged_clean[uk_merged_clean['Item'] == crop]
    ax4.scatter(crop_data['avg_temp'], crop_data['yield_hg_per_ha'], 
                label=crop, s=100, alpha=0.6)
ax4.set_xlabel('Temperature (°C)')
ax4.set_ylabel('Yield (hg/ha)')
ax4.set_title('Yield vs Temperature')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Yield vs Rainfall
ax5 = axes[1, 1]
for crop in uk_merged_clean['Item'].unique():
    crop_data = uk_merged_clean[uk_merged_clean['Item'] == crop]
    ax5.scatter(crop_data['annual_rainfall_mm'], crop_data['yield_hg_per_ha'],
                label=crop, s=100, alpha=0.6)
ax5.set_xlabel('Rainfall (mm)')
ax5.set_ylabel('Yield (hg/ha)')
ax5.set_title('Yield vs Rainfall')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Correlation heatmap
ax6 = axes[1, 2]
# Select numeric columns for correlation
corr_cols = ['yield_hg_per_ha', 'avg_temp', 'annual_rainfall_mm', 'pesticides_tonnes']
wheat_data = uk_merged_clean[uk_merged_clean['Item'] == 'Wheat'][corr_cols]
corr_matrix = wheat_data.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, ax=ax6, cbar_kws={'shrink': 0.8})
ax6.set_title('Correlation Matrix (Wheat)')

plt.tight_layout()
plt.savefig('uk_data_overview.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: uk_data_overview.png")


print(f"  1. Explore correlations in the data")
print(f"  2. Build baseline prediction models")
print(f"  3. Compare model performance for wheat vs potatoes")