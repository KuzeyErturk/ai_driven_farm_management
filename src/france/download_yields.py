import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    CROPS, YEAR_START, YEAR_END, PATHS,
    FRANCE_REGIONS, FRANCE_DEPT_TO_REGION, FRANCE_CROP_FILES, FRANCE_DATA_DIR,
    GERMANY_REGIONS, GERMANY_NUTS_TO_REGION, GERMANY_CROP_MAPPING, GERMANY_CSV_PATH,
)


# france data from Schauberger et al. 2022

def load_france_yields():
    all_rows = []

    for crop_name, filename in FRANCE_CROP_FILES.items():
        filepath = os.path.join(FRANCE_DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"    WARNING: Missing {filepath}")
            continue

        df = pd.read_csv(filepath, sep=';')

        # Filter years
        df = df[(df['year'] >= YEAR_START) & (df['year'] <= YEAR_END)]

        df['Region'] = df['department'].map(FRANCE_DEPT_TO_REGION)

        unmapped = df[df['Region'].isna()]['department'].unique()
        if len(unmapped) > 0:
            print(f"    WARNING: Unmapped departments for {crop_name}: {list(unmapped)}")

        df = df.dropna(subset=['Region'])

        df = df[(df['yield'] > 0) & (df['area'] > 0)]

        for (year, region), group in df.groupby(['year', 'Region']):
            total_area = group['area'].sum()
            weighted_yield = (group['yield'] * group['area']).sum() / total_area

            all_rows.append({
                'Year': int(year),
                'Region': region,
                'Crop': crop_name,
                'Area_hectares': int(total_area),
                'Yield_t_per_ha': round(weighted_yield, 3),
            })

        n_depts = df['department'].nunique()
        mean_yield = df['yield'].mean()
        print(f"    {crop_name:<16} {n_depts} depts, mean yield={mean_yield:.2f} t/ha")

    result = pd.DataFrame(all_rows)
    return result

def load_germany_yields():
    if not os.path.exists(GERMANY_CSV_PATH):
        print(f"    WARNING: Missing {GERMANY_CSV_PATH}")
        return pd.DataFrame()

    df = pd.read_csv(GERMANY_CSV_PATH)

    target_crops = set(GERMANY_CROP_MAPPING.keys())
    df = df[df['var'].isin(target_crops)]
    df = df[(df['year'] >= YEAR_START) & (df['year'] <= YEAR_END)]

    df['Region'] = df['nuts_id'].str[:3].map(GERMANY_NUTS_TO_REGION)
    unmapped = df[df['Region'].isna()]['nuts_id'].str[:3].unique()
    if len(unmapped) > 0:
        print(f"    WARNING: Unmapped NUTS prefixes: {list(unmapped)}")
    df = df.dropna(subset=['Region'])

    # Map crop codes to project names
    df['Crop'] = df['var'].map(GERMANY_CROP_MAPPING)

    # Separate yield and area
    yields = df[df['measure'] == 'yield'][['district_no', 'year', 'Region', 'Crop', 'value']].copy()
    yields = yields[yields['value'].apply(lambda x: str(x) != 'NA')]
    yields['value'] = pd.to_numeric(yields['value'])
    yields = yields.rename(columns={'value': 'yield'})
    yields = yields[yields['yield'] > 0]

    areas = df[df['measure'] == 'area'][['district_no', 'year', 'Region', 'Crop', 'value']].copy()
    areas = areas[areas['value'].apply(lambda x: str(x) != 'NA')]
    areas['value'] = pd.to_numeric(areas['value'])
    areas = areas.rename(columns={'value': 'area'})

    area_lookup = {}
    census_years = sorted(areas['year'].unique())
    print(f"    Area census years: {list(census_years)}")

    for _, row in areas.iterrows():
        key = (row['district_no'], row['Crop'], row['year'])
        area_lookup[key] = row['area']

    def get_nearest_area(district_no, crop, year):
        # Try exact year first
        if (district_no, crop, year) in area_lookup:
            return area_lookup[(district_no, crop, year)]
        # Find nearest census year
        best_year = min(census_years, key=lambda cy: abs(cy - year))
        return area_lookup.get((district_no, crop, best_year), np.nan)

    yields['area'] = yields.apply(
        lambda r: get_nearest_area(r['district_no'], r['Crop'], r['year']), axis=1
    )
    yields = yields.dropna(subset=['area'])
    yields = yields[yields['area'] > 0]

    all_rows = []
    for (year, region, crop), group in yields.groupby(['year', 'Region', 'Crop']):
        total_area = group['area'].sum()
        weighted_yield = (group['yield'] * group['area']).sum() / total_area

        all_rows.append({
            'Year': int(year),
            'Region': region,
            'Crop': crop,
            'Area_hectares': int(total_area),
            'Yield_t_per_ha': round(weighted_yield, 3),
        })

    result = pd.DataFrame(all_rows)

    for crop_code, crop_name in GERMANY_CROP_MAPPING.items():
        sub = yields[yields['Crop'] == crop_name]
        n_districts = sub['district_no'].nunique()
        mean_yield = sub['yield'].mean()
        print(f"    {crop_name:<16} {n_districts} districts, mean yield={mean_yield:.2f} t/ha")

    return result

def main():

    os.makedirs(PATHS['france_processed'], exist_ok=True)
    os.makedirs(PATHS['germany_processed'], exist_ok=True)

    france_df = load_france_yields()

    print(f"\n  France summary:")
    print(f"    Shape: {france_df.shape}")
    print(f"    Years: {france_df['Year'].min()}-{france_df['Year'].max()}")
    print(f"    Regions: {sorted(france_df['Region'].unique())}")
    print(f"    Crops: {sorted(france_df['Crop'].unique())}")
    expected = len(CROPS) * len(FRANCE_REGIONS) * (YEAR_END - YEAR_START + 1)
    print(f"    Expected rows: {expected}, Actual: {len(france_df)}")

    print("\n    Yield by crop (t/ha):")
    print(france_df.groupby('Crop')['Yield_t_per_ha'].agg(['mean', 'std']).round(2).to_string())

    france_path = os.path.join(PATHS['france_processed'], 'france_crop_yields_2004_2018.csv')
    france_df.to_csv(france_path, index=False)
    print(f"\n    Saved: {france_path}")

    germany_df = load_germany_yields()

    if len(germany_df) > 0:
        print(f"\n  Germany summary:")
        print(f"    Shape: {germany_df.shape}")
        print(f"    Years: {germany_df['Year'].min()}-{germany_df['Year'].max()}")
        print(f"    Regions: {sorted(germany_df['Region'].unique())}")
        print(f"    Crops: {sorted(germany_df['Crop'].unique())}")
        expected_de = len(CROPS) * len(GERMANY_REGIONS) * (YEAR_END - YEAR_START + 1)
        print(f"    Expected rows: {expected_de}, Actual: {len(germany_df)}")

        print("\n    Yield by crop (t/ha):")
        print(germany_df.groupby('Crop')['Yield_t_per_ha'].agg(['mean', 'std']).round(2).to_string())

        germany_path = os.path.join(PATHS['germany_processed'], 'germany_crop_yields_2004_2018.csv')
        germany_df.to_csv(germany_path, index=False)
        print(f"\n    Saved: {germany_path}")
    return france_df, germany_df


if __name__ == '__main__':
    main()
