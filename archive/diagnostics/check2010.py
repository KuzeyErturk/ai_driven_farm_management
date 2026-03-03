import pandas as pd

print("="*70)
print("2010 EXTREME YEAR ANALYSIS - Did Weather Signal Show Up?")
print("="*70)

# Load data
df = pd.read_csv('data/uk_crop_yield_complete_2004_2024.csv')

# 2010 Weather: 7.94°C (coldest), 93.4 frost days (double normal!)
print("\n2010 Weather Conditions:")
print("-"*70)
weather_2010 = df[df['Year'] == 2010][['Annual_Temp_C', 'Annual_Rainfall_mm', 
                                        'Annual_Sunshine_hours', 'Annual_Frost_days']].iloc[0]
print(weather_2010)

# Compare to average
print("\nAverage Weather Conditions (All Years):")
print("-"*70)
weather_avg = df[['Annual_Temp_C', 'Annual_Rainfall_mm', 
                   'Annual_Sunshine_hours', 'Annual_Frost_days']].mean()
print(weather_avg)

print("\n2010 Deviation from Average:")
print("-"*70)
deviation = weather_2010 - weather_avg
print(deviation)
print(f"\n🥶 Temperature: {deviation['Annual_Temp_C']:.2f}°C COLDER than average")
print(f"❄️  Frost Days: {deviation['Annual_Frost_days']:.1f} MORE frost days than average")

# Now check yields
print("\n" + "="*70)
print("YIELD ANALYSIS: Did 2010 Crops Suffer?")
print("="*70)

yields_2010 = df[df['Year'] == 2010][['Crop', 'Yield_t_per_ha']].sort_values('Crop')
yields_avg = df.groupby('Crop')['Yield_t_per_ha'].mean().reset_index()
yields_avg.columns = ['Crop', 'Average_Yield']

comparison = yields_2010.merge(yields_avg, on='Crop')
comparison['Difference'] = comparison['Yield_t_per_ha'] - comparison['Average_Yield']
comparison['Percent_Change'] = (comparison['Difference'] / comparison['Average_Yield'] * 100).round(1)

print("\n2010 Yields vs Long-Term Average:")
print("-"*70)
print(comparison[['Crop', 'Yield_t_per_ha', 'Average_Yield', 'Difference', 'Percent_Change']])

# Summary
print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

crashes = comparison[comparison['Percent_Change'] < -5]
if len(crashes) > 0:
    print("\n✓ WEATHER SIGNAL EXISTS!")
    print(f"  {len(crashes)} crops showed yield decline > 5%:")
    for _, row in crashes.iterrows():
        print(f"    • {row['Crop']}: {row['Percent_Change']:.1f}% decline")
    print("\n  → Cold/frost damaged crops as expected")
    print("  → Problem: Annual averaging is HIDING this signal")
    print("  → Solution: Need monthly/seasonal weather data")
else:
    print("\n⚠️ NO YIELD DECLINE DETECTED")
    print("  Despite extreme cold/frost, yields were normal/high")
    print("  Possible explanations:")
    print("    1. UK farmers successfully adapted (irrigation, varieties, etc.)")
    print("    2. Temporal misalignment (wrong year's weather?)")
    print("    3. Cold winter helped vernalization for winter crops")

# Check variance
print("\n" + "="*70)
print("YIELD VARIANCE ANALYSIS")
print("="*70)

variance = df.groupby('Crop')['Yield_t_per_ha'].agg(['mean', 'std', 'min', 'max'])
variance['CV_%'] = (variance['std'] / variance['mean'] * 100).round(1)
variance['Range'] = variance['max'] - variance['min']
variance['Range_%'] = (variance['Range'] / variance['mean'] * 100).round(1)

print("\nYield Variability by Crop:")
print("-"*70)
print(variance[['mean', 'std', 'CV_%', 'min', 'max', 'Range_%']].round(2))

print("\nInterpretation:")
print("-"*70)
avg_cv = variance['CV_%'].mean()
if avg_cv < 5:
    print(f"  ⚠️ Very low variability (CV = {avg_cv:.1f}%)")
    print("  → UK yields are highly stable")
    print("  → Weather has minimal impact OR farmers adapt very well")
    print("  → Hard to predict with weather variables")
elif avg_cv < 10:
    print(f"  ✓ Moderate variability (CV = {avg_cv:.1f}%)")
    print("  → Yields do vary year-to-year")
    print("  → Weather should be predictive with right features")
    print("  → Current weak correlations = data granularity problem")
else:
    print(f"  ✓ High variability (CV = {avg_cv:.1f}%)")
    print("  → Yields vary substantially")
    print("  → Weather MUST be predictive")
    print("  → Weak correlations = serious data problem")

# Check other extreme years
print("\n" + "="*70)
print("OTHER EXTREME YEARS")
print("="*70)

# 2012 - known wet year
print("\n2012 (Wet Summer - Olympics Year):")
yields_2012 = df[df['Year'] == 2012][['Crop', 'Yield_t_per_ha']].merge(yields_avg, on='Crop')
yields_2012['Change_%'] = ((yields_2012['Yield_t_per_ha'] - yields_2012['Average_Yield']) / yields_2012['Average_Yield'] * 100).round(1)
print(yields_2012[['Crop', 'Change_%']].sort_values('Change_%'))

# 2020 - high sunshine year
print("\n2020 (High Sunshine Year - 1498 hours):")
yields_2020 = df[df['Year'] == 2020][['Crop', 'Yield_t_per_ha']].merge(yields_avg, on='Crop')
yields_2020['Change_%'] = ((yields_2020['Yield_t_per_ha'] - yields_2020['Average_Yield']) / yields_2020['Average_Yield'] * 100).round(1)
print(yields_2020[['Crop', 'Change_%']].sort_values('Change_%'))

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print("\nBased on this analysis:")

if len(crashes) > 0:
    print("\n✅ ACTION: Get monthly/seasonal weather data")
    print("   Weather IS affecting yields, but annual averages hide it")
    print("   Download: Monthly temp, rain, sunshine from Met Office")
    print("   Create: Spring_Temp, Summer_Rain, etc.")
    print("   Expected: Correlations will jump to r = 0.4-0.6")
else:
    print("\n🤔 ACTION: Investigate further")
    print("   1. Check if yields vary enough (see CV% above)")
    print("   2. Try different temporal alignment (growing season)")
    print("   3. Consider regional analysis instead of national")
    print("   4. Document UK agriculture resilience as finding")

print("\n" + "="*70)