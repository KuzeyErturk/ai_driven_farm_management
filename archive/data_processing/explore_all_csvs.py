import pandas as pd
import os

print("="*70)
print("EXPLORING ALL CSV FILES")
print("="*70)

csv_files = ['yield_df.csv', 'rainfall.csv', 'pesticides.csv', 'temp.csv']

for csv_file in csv_files:
    filepath = f'data/{csv_file}'
    
    if os.path.exists(filepath):
        print(f"\n{'='*70}")
        print(f"FILE: {csv_file}")
        print("="*70)
        
        df = pd.read_csv(filepath)
        
        print(f"\nShape: {df.shape} (rows: {df.shape[0]}, columns: {df.shape[1]})")
        print(f"\nColumn names:")
        print(df.columns.tolist())
        
        print(f"\nData types:")
        print(df.dtypes)
        
        print(f"\nFirst 10 rows:")
        print(df.head(10))
        
        print(f"\nBasic statistics:")
        print(df.describe())
        
        # Check for UK data
        if 'Area' in df.columns:
            uk_data = df[df['Area'] == 'United Kingdom']
            print(f"\n✅ UK records found: {len(uk_data)}")
            if len(uk_data) > 0:
                print(f"Years: {uk_data['Year'].min()} - {uk_data['Year'].max()}")
                if 'Item' in uk_data.columns:
                    print(f"Items: {uk_data['Item'].unique()}")
        
        print(f"\nMissing values:")
        print(df.isnull().sum())
        
    else:
        print(f"\n❌ File not found: {filepath}")

print("\n" + "="*70)
print("EXPLORATION COMPLETE")
print("="*70)