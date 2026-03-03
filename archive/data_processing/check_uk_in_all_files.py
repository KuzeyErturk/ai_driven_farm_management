import pandas as pd
import numpy as np

print("="*70)
print("CHECKING UK DATA IN EACH FILE")
print("="*70)

# ==================== RAINFALL ====================
print("\n" + "="*70)
print("1. RAINFALL.CSV - UK DATA CHECK")
print("="*70)

rainfall_df = pd.read_csv('data/rainfall.csv')
print(f"Column names: {rainfall_df.columns.tolist()}")

# Check for UK (note the space in column name!)
uk_variants = ['United Kingdom', 'UK', 'United Kingdom']
rainfall_df[' Area'] = rainfall_df[' Area'].str.strip()  # Remove spaces

uk_rainfall = rainfall_df[rainfall_df[' Area'].str.contains('United Kingdom', case=False, na=False)]
print(f"\n✅ UK records found: {len(uk_rainfall)}")

if len(uk_rainfall) > 0:
    print(f"Years: {uk_rainfall['Year'].min()} - {uk_rainfall['Year'].max()}")
    print(f"\nFirst 10 UK records:")
    print(uk_rainfall.head(10))
    
    # Check if rainfall varies
    print(f"\nRainfall values:")
    print(f"  Unique values: {uk_rainfall['average_rain_fall_mm_per_year'].nunique()}")
    print(f"  Sample values: {uk_rainfall['average_rain_fall_mm_per_year'].unique()[:10]}")
    
    # Convert to numeric if it's object type
    if uk_rainfall['average_rain_fall_mm_per_year'].dtype == 'object':
        uk_rainfall_clean = uk_rainfall.copy()
        uk_rainfall_clean['average_rain_fall_mm_per_year'] = pd.to_numeric(
            uk_rainfall_clean['average_rain_fall_mm_per_year'], errors='coerce'
        )
        print(f"\nAfter numeric conversion:")
        print(f"  Min: {uk_rainfall_clean['average_rain_fall_mm_per_year'].min()}")
        print(f"  Max: {uk_rainfall_clean['average_rain_fall_mm_per_year'].max()}")
        print(f"  Mean: {uk_rainfall_clean['average_rain_fall_mm_per_year'].mean():.2f}")
        
        if uk_rainfall_clean['average_rain_fall_mm_per_year'].nunique() > 1:
            print("\n🎉 RAINFALL VARIES ACROSS YEARS!")
        else:
            print("\n⚠️  Rainfall is constant")
else:
    print("❌ No UK data found in rainfall.csv")

# ==================== PESTICIDES ====================
print("\n" + "="*70)
print("2. PESTICIDES.CSV - UK DATA CHECK")
print("="*70)

pesticides_df = pd.read_csv('data/pesticides.csv')
uk_pesticides = pesticides_df[pesticides_df['Area'] == 'United Kingdom']

print(f"✅ UK records found: {len(uk_pesticides)}")
if len(uk_pesticides) > 0:
    print(f"Years: {uk_pesticides['Year'].min()} - {uk_pesticides['Year'].max()}")
    print(f"\nFirst 10 UK records:")
    print(uk_pesticides.head(10))
    
    print(f"\nPesticide values:")
    print(f"  Min: {uk_pesticides['Value'].min():.2f}")
    print(f"  Max: {uk_pesticides['Value'].max():.2f}")
    print(f"  Mean: {uk_pesticides['Value'].mean():.2f}")
    print(f"  Unique values: {uk_pesticides['Value'].nunique()}")
    
    if uk_pesticides['Value'].nunique() > 1:
        print("\n✅ PESTICIDES VARY ACROSS YEARS!")

# ==================== TEMPERATURE ====================
print("\n" + "="*70)
print("3. TEMP.CSV - UK DATA CHECK")
print("="*70)

temp_df = pd.read_csv('data/temp.csv')
print(f"Column names: {temp_df.columns.tolist()}")

# Check for UK
uk_temp = temp_df[temp_df['country'].str.contains('United Kingdom', case=False, na=False)]
print(f"\n✅ UK records found: {len(uk_temp)}")

if len(uk_temp) > 0:
    print(f"Years available: {uk_temp['year'].min()} - {uk_temp['year'].max()}")
    
    # Filter for relevant years (1990-2013)
    uk_temp_relevant = uk_temp[(uk_temp['year'] >= 1990) & (uk_temp['year'] <= 2013)]
    print(f"Years 1990-2013: {len(uk_temp_relevant)} records")
    
    if len(uk_temp_relevant) > 0:
        print(f"\nFirst 10 UK records (1990-2013):")
        print(uk_temp_relevant.head(10))
        
        print(f"\nTemperature values:")
        print(f"  Min: {uk_temp_relevant['avg_temp'].min():.2f}°C")
        print(f"  Max: {uk_temp_relevant['avg_temp'].max():.2f}°C")
        print(f"  Mean: {uk_temp_relevant['avg_temp'].mean():.2f}°C")
        print(f"  Unique values: {uk_temp_relevant['avg_temp'].nunique()}")
        
        if uk_temp_relevant['avg_temp'].nunique() > 1:
            print("\n✅ TEMPERATURE VARIES ACROSS YEARS!")
else:
    print("❌ No UK data found in temp.csv")

# ==================== SUMMARY ====================
print("\n" + "="*70)
print("SUMMARY: WHAT CAN WE USE FROM SEPARATE FILES?")
print("="*70)

print("\n1. yield_df.csv:")
print("   ✅ Already merged - use as base")
print("   ✅ Has UK wheat & potato yields")
print("   ⚠️  Rainfall constant (1220.0)")

print("\n2. rainfall.csv:")
if len(uk_rainfall) > 0 and uk_rainfall['average_rain_fall_mm_per_year'].nunique() > 1:
    print("   ✅ Can use - rainfall varies!")
else:
    print("   ❌ Cannot use - no UK data or constant")

print("\n3. pesticides.csv:")
if len(uk_pesticides) > 0:
    print("   ✅ Can use - detailed pesticide data")
    print(f"   📊 Covers years {uk_pesticides['Year'].min()}-{uk_pesticides['Year'].max()}")
else:
    print("   ❌ Cannot use - no UK data")

print("\n4. temp.csv:")
if len(uk_temp) > 0:
    print("   ✅ Can use - annual temperature data")
    print(f"   📊 Very detailed historical data")
else:
    print("   ❌ Cannot use - no UK data")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)