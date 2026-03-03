import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/regional_crop_yield_weather_2004_2024.csv')
barley = df[df['Crop'] == 'Barley'].copy()

# Plot yield over time
fig, ax = plt.subplots(figsize=(12, 6))

for region in barley['Region'].unique():
    region_data = barley[barley['Region'] == region]
    ax.plot(region_data['Year'], region_data['Yield_t_per_ha'], 
            marker='o', label=region)

ax.axvline(x=2019.5, color='red', linestyle='--', linewidth=2, 
           label='Train/Test Split')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Barley Yield (t/ha)', fontsize=12)
ax.set_title('Barley Yield Over Time by Region', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/barley_yield_timeline.png', dpi=300)
print("✓ Saved: plots/barley_yield_timeline.png")
plt.show()

# Check if test years are unusual
train = barley[barley['Year'] <= 2019]
test = barley[barley['Year'] >= 2020]

print("\nBarley Yield Statistics:")
print(f"Train (2004-2019): {train['Yield_t_per_ha'].mean():.2f} ± {train['Yield_t_per_ha'].std():.2f}")
print(f"Test (2020-2024):  {test['Yield_t_per_ha'].mean():.2f} ± {test['Yield_t_per_ha'].std():.2f}")
print(f"Difference: {test['Yield_t_per_ha'].mean() - train['Yield_t_per_ha'].mean():.2f} t/ha")