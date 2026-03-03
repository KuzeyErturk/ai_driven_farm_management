import pandas as pd

df = pd.read_csv('data/yield_df.csv')
uk_df = df[df['Area'] == 'United Kingdom']

print("="*60)
print("COLUMN NAMES")
print("="*60)
print(uk_df.columns.tolist())

print("\n" + "="*60)
print("DATA TYPES")
print("="*60)
print(uk_df.dtypes)

print("\n" + "="*60)
print("COMPLETE FIRST 20 ROWS")
print("="*60)
print(uk_df.head(20))

print("\n" + "="*60)
print("YEAR RANGE")
print("="*60)
print(f"First year: {uk_df['Year'].min()}")
print(f"Last year: {uk_df['Year'].max()}")
print(f"Total years: {uk_df['Year'].nunique()}")
print(f"Years: {sorted(uk_df['Year'].unique())}")

print("\n" + "="*60)
print("RECORDS PER YEAR PER CROP")
print("="*60)
print(uk_df.groupby(['Year', 'Item']).size())

print("\n" + "="*60)
print("UNIQUE VALUES IN EACH COLUMN")
print("="*60)
for col in uk_df.columns:
    n_unique = uk_df[col].nunique()
    print(f"\n{col}: {n_unique} unique values")
    if n_unique <= 20:
        print(f"  Values: {sorted(uk_df[col].unique())}")

print("\n" + "="*60)
print("CHECK FOR MISSING VALUES")
print("="*60)
print(uk_df.isnull().sum())

print("\n" + "="*60)
print("SAMPLE WHEAT DATA")
print("="*60)
wheat_sample = uk_df[uk_df['Item'] == 'Wheat'].head(20)
print(wheat_sample)

print("\n" + "="*60)
print("SAMPLE POTATO DATA")
print("="*60)
potato_sample = uk_df[uk_df['Item'] == 'Potatoes'].head(20)
print(potato_sample)