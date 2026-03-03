import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

print("="*70)
print("DAY 1: PHYSICS-AWARE EDA - DOES THE DATA MAKE SENSE?")
print("="*70)

# Load data
df = pd.read_csv('data/uk_crop_yield_complete_2004_2024.csv')

print(f"\nDataset loaded: {df.shape[0]} observations, {df.shape[1]} variables")
print(f"Crops: {df['Crop'].unique()}")
print(f"Years: {df['Year'].min()} - {df['Year'].max()}")

# ============================================================================
# SECTION 1: DATA QUALITY CHECK
# ============================================================================
print("\n" + "="*70)
print("SECTION 1: DATA QUALITY CHECK")
print("="*70)

print("\n1.1 Missing Values:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("  ✓ No missing values - excellent!")
else:
    print(missing[missing > 0])

print("\n1.2 Outlier Check (Z-score > 3):")
numeric_cols = df.select_dtypes(include=[np.number]).columns
outliers_found = False
for col in numeric_cols:
    if col != 'Year':
        z_scores = np.abs(stats.zscore(df[col]))
        outliers = df[z_scores > 3]
        if len(outliers) > 0:
            outliers_found = True
            print(f"  {col}: {len(outliers)} outliers")
            print(f"    Values: {outliers[[col, 'Year', 'Crop']].to_dict('records')}")

if not outliers_found:
    print("  ✓ No extreme outliers detected")

print("\n1.3 Yield Statistics by Crop:")
yield_stats = df.groupby('Crop')['Yield_t_per_ha'].agg(['mean', 'std', 'min', 'max'])
print(yield_stats.round(2))

# ============================================================================
# SECTION 2: PHYSICS CHECK - DO RELATIONSHIPS MAKE SENSE?
# ============================================================================
print("\n" + "="*70)
print("SECTION 2: PHYSICS CHECK - WEATHER VS YIELD RELATIONSHIPS")
print("="*70)

# Create subplot grid for weather relationships
weather_vars = ['Annual_Temp_C', 'Annual_Rainfall_mm', 'Annual_Sunshine_hours', 
                'Annual_Humidity_Percent', 'Annual_Frost_days']
crops = df['Crop'].unique()

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)

fig, axes = plt.subplots(len(weather_vars), len(crops), figsize=(24, 20))
fig.suptitle('Yield vs Weather: Physics Check for Each Crop', 
             fontsize=16, fontweight='bold', y=0.995)

print("\n2.1 Visual Inspection Results:")
print("-" * 70)

for i, weather_var in enumerate(weather_vars):
    print(f"\n{weather_var}:")
    
    for j, crop in enumerate(crops):
        ax = axes[i, j]
        crop_data = df[df['Crop'] == crop]
        
        # Scatter plot
        ax.scatter(crop_data[weather_var], crop_data['Yield_t_per_ha'], 
                  alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
        
        # Add trend line
        z = np.polyfit(crop_data[weather_var], crop_data['Yield_t_per_ha'], 2)
        p = np.poly1d(z)
        x_trend = np.linspace(crop_data[weather_var].min(), 
                             crop_data[weather_var].max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation
        corr = crop_data[weather_var].corr(crop_data['Yield_t_per_ha'])
        
        # Styling
        ax.set_xlabel(weather_var, fontsize=9)
        if j == 0:
            ax.set_ylabel('Yield (t/ha)', fontsize=9)
        ax.set_title(f'{crop}\nr={corr:.2f}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Print correlation
        if j == 0:  # Print once per weather variable
            print(f"  {crop}: r={corr:.2f}", end="  ")
    print()  # New line after each weather variable

plt.tight_layout()
plt.savefig('plots/day1_physics_check_all_crops.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: plots/day1_physics_check_all_crops.png")
plt.show()

# ============================================================================
# SECTION 3: KEY RELATIONSHIP DIAGNOSTICS
# ============================================================================
print("\n" + "="*70)
print("SECTION 3: KEY RELATIONSHIP DIAGNOSTICS")
print("="*70)

# 3.1 Rainfall - Should show inverted U (optimal zone)
print("\n3.1 RAINFALL RELATIONSHIP CHECK:")
print("Expected: Inverted U-shape (too little = drought, too much = disease)")
print("-" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, crop in enumerate(crops):
    crop_data = df[df['Crop'] == crop]
    ax = axes[idx]
    
    # Scatter with quadratic fit
    ax.scatter(crop_data['Annual_Rainfall_mm'], crop_data['Yield_t_per_ha'], 
              s=100, alpha=0.6, edgecolors='black')
    
    # Fit quadratic
    z = np.polyfit(crop_data['Annual_Rainfall_mm'], crop_data['Yield_t_per_ha'], 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(crop_data['Annual_Rainfall_mm'].min(), 
                          crop_data['Annual_Rainfall_mm'].max(), 100)
    ax.plot(x_smooth, p(x_smooth), 'r-', linewidth=3, label='Quadratic fit')
    
    # Check if inverted U (negative quadratic coefficient)
    if z[0] < 0:
        optimal_rain = -z[1]/(2*z[0])
        ax.axvline(optimal_rain, color='green', linestyle='--', linewidth=2, 
                  label=f'Optimal: {optimal_rain:.0f}mm')
        print(f"  {crop}: INVERTED U detected ✓ (Optimal rainfall ~ {optimal_rain:.0f}mm)")
    else:
        print(f"  {crop}: No inverted U (monotonic relationship)")
    
    ax.set_xlabel('Annual Rainfall (mm)', fontsize=11)
    ax.set_ylabel('Yield (t/ha)', fontsize=11)
    ax.set_title(f'{crop}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/day1_rainfall_optimal_check.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: plots/day1_rainfall_optimal_check.png")
plt.show()

# 3.2 Sunshine - Should be strongly positive
print("\n3.2 SUNSHINE RELATIONSHIP CHECK:")
print("Expected: Strong positive (more sun = more photosynthesis)")
print("-" * 70)

sunshine_corrs = []
for crop in crops:
    crop_data = df[df['Crop'] == crop]
    corr = crop_data['Annual_Sunshine_hours'].corr(crop_data['Yield_t_per_ha'])
    sunshine_corrs.append(corr)
    
    status = "✓ STRONG" if corr > 0.4 else "⚠ WEAK" if corr > 0.2 else "✗ VERY WEAK"
    print(f"  {crop}: r={corr:.3f} {status}")

avg_sunshine_corr = np.mean(sunshine_corrs)
print(f"\n  Average sunshine correlation: {avg_sunshine_corr:.3f}")
if avg_sunshine_corr > 0.4:
    print("  ✓ Sunshine is a strong predictor - good signal!")
else:
    print("  ⚠ Sunshine correlation weaker than expected - investigate further")

# 3.3 Temperature - Should vary by crop
print("\n3.3 TEMPERATURE RELATIONSHIP CHECK:")
print("Expected: Sugar Beet > Wheat > others (heat tolerance varies)")
print("-" * 70)

temp_corrs = {}
for crop in crops:
    crop_data = df[df['Crop'] == crop]
    corr = crop_data['Annual_Temp_C'].corr(crop_data['Yield_t_per_ha'])
    temp_corrs[crop] = corr
    print(f"  {crop}: r={corr:.3f}")

# Check if Sugar Beet shows strongest positive correlation
if 'Sugar_Beet' in temp_corrs:
    if temp_corrs['Sugar_Beet'] > 0.3:
        print("  ✓ Sugar Beet shows positive temp correlation (expected - heat-loving crop)")
    else:
        print("  ⚠ Sugar Beet doesn't show expected positive temp response")

# 3.4 Frost - Should be negative
print("\n3.4 FROST RELATIONSHIP CHECK:")
print("Expected: Negative correlation (frost damages crops)")
print("-" * 70)

for crop in crops:
    crop_data = df[df['Crop'] == crop]
    corr = crop_data['Annual_Frost_days'].corr(crop_data['Yield_t_per_ha'])
    
    status = "✓ Expected" if corr < 0 else "⚠ Unexpected (positive)"
    print(f"  {crop}: r={corr:.3f} {status}")

# ============================================================================
# SECTION 4: CORRELATION MATRIX DEEP DIVE
# ============================================================================
print("\n" + "="*70)
print("SECTION 4: CORRELATION MATRIX ANALYSIS")
print("="*70)

# 4.1 Overall correlation matrix
print("\n4.1 Overall Correlation with Yield:")
print("-" * 70)

# Select numeric features only
feature_cols = ['Area_hectares', 'Annual_Temp_C', 'Annual_Rainfall_mm', 
                'Annual_Humidity_Percent', 'Annual_Sunshine_hours', 
                'Annual_Frost_days', 'Total_Nitrogen_kg_ha', 
                'Total_Phosphate_kg_ha', 'Total_Potash_kg_ha']

correlations = df[feature_cols + ['Yield_t_per_ha']].corr()['Yield_t_per_ha'].drop('Yield_t_per_ha')
correlations = correlations.sort_values(ascending=False)

print("\nTop Positive Correlations:")
print(correlations[correlations > 0].head(5))

print("\nTop Negative Correlations:")
print(correlations[correlations < 0].tail(5))

# Plot correlation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = df[feature_cols + ['Yield_t_per_ha']].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
            cmap='coolwarm', center=0, square=True, linewidths=1,
            cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix (with Yield)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/day1_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: plots/day1_correlation_matrix.png")
plt.show()

# 4.2 Check multicollinearity among predictors
print("\n4.2 Multicollinearity Check (Weather Variables):")
print("Looking for high correlations (|r| > 0.7) between predictors")
print("-" * 70)

weather_corr = df[weather_vars].corr()
high_corr_pairs = []

for i in range(len(weather_vars)):
    for j in range(i+1, len(weather_vars)):
        corr_val = weather_corr.iloc[i, j]
        if abs(corr_val) > 0.7:
            high_corr_pairs.append((weather_vars[i], weather_vars[j], corr_val))
            print(f"  ⚠ {weather_vars[i]} <-> {weather_vars[j]}: r={corr_val:.3f}")

if not high_corr_pairs:
    print("  ✓ No severe multicollinearity detected (all |r| < 0.7)")
else:
    print(f"\n  Warning: {len(high_corr_pairs)} highly correlated pairs found")
    print("  → Consider Ridge/Lasso regression to handle multicollinearity")

# 4.3 Area vs Yield relationship
print("\n4.3 AREA vs YIELD RELATIONSHIP:")
print("Question: Does planted area predict yield?")
print("-" * 70)

area_yield_corr = df['Area_hectares'].corr(df['Yield_t_per_ha'])
print(f"  Overall correlation: r={area_yield_corr:.3f}")

if abs(area_yield_corr) < 0.2:
    print("  → Area does NOT predict yield (r near zero)")
    print("  → Interpretation: Area is market-driven, not weather/quality-driven")
    print("  → Action: Area may not be useful predictor - consider excluding")
elif area_yield_corr > 0.2:
    print("  → Positive correlation: Farmers plant more when they expect good yields")
    print("  → Interpretation: Area captures farmer expectations/forward-looking behavior")
    print("  → Action: Keep Area as predictor - it contains signal")
elif area_yield_corr < -0.2:
    print("  → Negative correlation: More area = lower yield")
    print("  → Interpretation: Expansion onto marginal land reduces average productivity")
    print("  → Action: Keep Area - important economic factor")

# Plot Area vs Yield by crop
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, crop in enumerate(crops):
    crop_data = df[df['Crop'] == crop]
    ax = axes[idx]
    
    corr = crop_data['Area_hectares'].corr(crop_data['Yield_t_per_ha'])
    
    ax.scatter(crop_data['Area_hectares'], crop_data['Yield_t_per_ha'], 
              s=100, alpha=0.6, edgecolors='black')
    
    # Trend line
    z = np.polyfit(crop_data['Area_hectares'], crop_data['Yield_t_per_ha'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(crop_data['Area_hectares'].min(), 
                        crop_data['Area_hectares'].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2)
    
    ax.set_xlabel('Area (hectares)', fontsize=11)
    ax.set_ylabel('Yield (t/ha)', fontsize=11)
    ax.set_title(f'{crop} (r={corr:.2f})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/day1_area_vs_yield.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: plots/day1_area_vs_yield.png")
plt.show()

# 4.4 Fertilizer correlations
print("\n4.4 FERTILIZER vs YIELD RELATIONSHIP:")
print("Question: Do fertilizers predict yield?")
print("-" * 70)

fert_vars = ['Total_Nitrogen_kg_ha', 'Total_Phosphate_kg_ha', 'Total_Potash_kg_ha']
for fert in fert_vars:
    corr = df[fert].corr(df['Yield_t_per_ha'])
    print(f"  {fert}: r={corr:.3f}")

avg_fert_corr = df[fert_vars].corrwith(df['Yield_t_per_ha']).mean()
if abs(avg_fert_corr) < 0.2:
    print(f"\n  ⚠ Weak fertilizer-yield correlation (avg r={avg_fert_corr:.3f})")
    print("  Possible reasons:")
    print("    1. Fertilizer data is aggregate (not crop-specific)")
    print("    2. Farmers already optimize fertilizer → little variation")
    print("    3. Weather dominates yield variation")
    print("  → Fertilizers may not be strong predictors in this dataset")
else:
    print(f"\n  ✓ Moderate fertilizer signal (avg r={avg_fert_corr:.3f})")

# ============================================================================
# SECTION 5: DAY 1 SUMMARY & INSIGHTS
# ============================================================================
print("\n" + "="*70)
print("DAY 1 SUMMARY: KEY FINDINGS")
print("="*70)

print("\n📊 DATA QUALITY:")
print(f"  • {df.shape[0]} observations, {df.shape[1]} variables")
print(f"  • {df['Crop'].nunique()} crops over {df['Year'].nunique()} years")
print(f"  • Missing values: {df.isnull().sum().sum()}")

print("\n🌦 WEATHER RELATIONSHIPS:")
top_3_weather = correlations[correlations.index.isin(weather_vars)].head(3)
print("  Top 3 weather predictors:")
for var, corr_val in top_3_weather.items():
    print(f"    • {var}: r={corr_val:.3f}")

print("\n🌾 CROP-SPECIFIC INSIGHTS:")
print(f"  • Sunshine correlation: {avg_sunshine_corr:.3f} ({'Strong' if avg_sunshine_corr > 0.4 else 'Moderate'})")
print(f"  • Area-yield correlation: {area_yield_corr:.3f} ({'Predictive' if abs(area_yield_corr) > 0.2 else 'Weak'})")
print(f"  • Fertilizer correlation: {avg_fert_corr:.3f} ({'Predictive' if abs(avg_fert_corr) > 0.2 else 'Weak'})")

print("\n⚠️ POTENTIAL ISSUES:")
issues_found = []
if avg_sunshine_corr < 0.3:
    issues_found.append("Sunshine correlation weaker than expected")
if len(high_corr_pairs) > 0:
    issues_found.append(f"Multicollinearity detected ({len(high_corr_pairs)} pairs)")
if abs(avg_fert_corr) < 0.1:
    issues_found.append("Fertilizer variables show very weak signal")

if issues_found:
    for issue in issues_found:
        print(f"  • {issue}")
else:
    print("  • No major data quality issues detected ✓")

print("\n💡 RECOMMENDATIONS FOR DAY 2:")
print("  1. Use Ridge/Lasso regression (handle potential multicollinearity)")
if abs(area_yield_corr) < 0.1:
    print("  2. Consider excluding Area_hectares (weak predictor)")
else:
    print("  2. Keep Area_hectares (shows predictive signal)")
    
if avg_fert_corr < 0.2:
    print("  3. Fertilizer variables may not improve model (aggregate data limitation)")
else:
    print("  3. Include fertilizer variables (moderate signal detected)")

print("  4. Focus on weather features - strongest signal in data")
print("  5. Consider crop-specific models (different crops respond differently)")

print("\n" + "="*70)
print("DAY 1 COMPLETE! 🎉")
print("="*70)
print("\nFiles saved:")
print("  • plots/day1_physics_check_all_crops.png")
print("  • plots/day1_rainfall_optimal_check.png")
print("  • plots/day1_correlation_matrix.png")
print("  • plots/day1_area_vs_yield.png")
print("\nNext: Day 2 - Build baseline models with this knowledge!")