import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("="*60)
print("LOADING DATA")
print("="*60)

# Load the CSV file
df = pd.read_csv('data/yield_df.csv')

print("✅ Data loaded successfully!")
print(f"Dataset shape: {df.shape}")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n" + "="*60)
print("FIRST 10 ROWS")
print("="*60)
print(df.head(10))

print("\n" + "="*60)
print("COLUMN INFORMATION")
print("="*60)
print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\n" + "="*60)
print("BASIC STATISTICS")
print("="*60)
print(df.describe())

print("\n" + "="*60)
print("MISSING VALUES")
print("="*60)
print(df.isnull().sum())

print("\n" + "="*60)
print("UNIQUE VALUES PER COLUMN")
print("="*60)
for col in df.columns:
    n_unique = df[col].nunique()
    print(f"{col}: {n_unique} unique values")
    
    # If few unique values, show them
    if n_unique < 20:
        print(f"  → {df[col].unique()}")
    print()

print("\n" + "="*60)
print("DATA SAMPLE")
print("="*60)
print(df.sample(5))  # Random 5 rows

# Generate yield distribution plot
plt.figure(figsize=(10, 6))
df['hg/ha_yield'].hist(bins=30, edgecolor='black')
plt.xlabel('Yield (hg/ha)')
plt.ylabel('Frequency')
plt.title('Distribution of Crop Yields')
plt.tight_layout()
plt.savefig('yield_distribution.png')
print("\n✅ Plot saved as 'yield_distribution.png'")
# plt.show()  # Uncomment to display plot

print("\n" + "="*60)
print("EXPLORATION COMPLETE!")
print("="*60)