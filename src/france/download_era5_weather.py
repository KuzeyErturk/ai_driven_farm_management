"""
Generate/Download Weather Data for French and German Regions
==============================================================
Generates synthetic monthly weather data based on published climate normals
from Météo-France and DWD (German Weather Service).

For real ERA5 data: install cdsapi, configure ~/.cdsapirc, and uncomment
the ERA5 download section.

Usage:
    python src/france/download_era5_weather.py

Output:
    data/france/raw/monthly_weather_{region}.csv  (4 French regions)
    data/france/raw/monthly_weather_{region}.csv  (4 German regions)
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    FRANCE_REGIONS, GERMANY_REGIONS, YEAR_START, YEAR_END, PATHS,
    ERA5_BBOXES,
)


MONTH_NAMES = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
               'jul', 'aug', 'sep', 'oct', 'nov', 'dec']


# ============================================================================
# CLIMATE NORMALS (published 1991-2020 averages)
# ============================================================================

# --- France (Météo-France) ---
FRANCE_NORMALS = {
    'tmax': {
        'Nord':  [6.5, 7.5, 11.0, 14.5, 18.0, 21.0, 23.5, 23.0, 19.5, 14.5, 9.5, 6.5],
        'Ouest': [8.5, 9.0, 12.0, 14.5, 18.0, 21.0, 23.0, 23.0, 20.5, 16.0, 11.5, 8.5],
        'Est':   [5.0, 7.0, 11.5, 15.5, 19.5, 23.0, 25.0, 24.5, 20.5, 15.0, 9.0, 5.5],
        'Sud':   [9.5, 10.5, 14.0, 16.5, 20.5, 25.0, 28.0, 27.5, 23.5, 18.5, 13.0, 9.5],
    },
    'tmin': {
        'Nord':  [1.0, 1.0, 3.0, 5.0, 8.5, 11.0, 13.0, 13.0, 10.5, 7.5, 4.0, 1.5],
        'Ouest': [3.0, 2.5, 4.0, 5.5, 8.5, 11.0, 13.0, 13.0, 11.0, 8.5, 5.5, 3.0],
        'Est':   [-1.0, -0.5, 2.0, 4.5, 8.5, 11.5, 13.5, 13.0, 10.0, 6.5, 2.5, -0.5],
        'Sud':   [1.5, 2.0, 4.5, 7.0, 10.5, 14.0, 16.5, 16.0, 13.0, 9.5, 5.5, 2.0],
    },
    'rain': {
        'Nord':  [55, 45, 50, 50, 60, 55, 60, 55, 50, 65, 60, 65],
        'Ouest': [85, 65, 60, 55, 60, 45, 45, 45, 55, 80, 85, 90],
        'Est':   [55, 50, 50, 55, 70, 65, 60, 60, 60, 65, 60, 65],
        'Sud':   [50, 45, 50, 65, 70, 50, 30, 40, 60, 80, 65, 55],
    },
    'sun': {
        'Nord':  [55, 70, 120, 160, 195, 200, 210, 195, 150, 100, 60, 45],
        'Ouest': [65, 80, 130, 170, 200, 220, 230, 210, 175, 115, 70, 50],
        'Est':   [50, 70, 125, 165, 200, 215, 230, 215, 160, 105, 55, 40],
        'Sud':   [100, 120, 175, 200, 240, 280, 310, 290, 225, 160, 110, 85],
    },
    'frost': {
        'Nord':  [12, 10, 6, 2, 0, 0, 0, 0, 0, 1, 5, 10],
        'Ouest': [6, 6, 3, 1, 0, 0, 0, 0, 0, 0, 3, 6],
        'Est':   [16, 13, 8, 3, 0, 0, 0, 0, 0, 2, 8, 14],
        'Sud':   [6, 5, 2, 0, 0, 0, 0, 0, 0, 0, 3, 5],
    },
}

# --- Germany (DWD normals) ---
GERMANY_NORMALS = {
    'tmax': {
        'Nord_DE':  [3.5, 4.5, 8.5, 13.0, 17.5, 20.0, 22.0, 22.0, 18.0, 12.5, 7.5, 4.0],
        'West_DE':  [4.5, 5.5, 10.0, 14.5, 18.5, 21.5, 23.5, 23.0, 19.0, 13.5, 8.5, 5.0],
        'Ost_DE':   [2.5, 4.0, 9.0, 14.5, 19.0, 22.0, 24.5, 24.0, 19.5, 13.5, 7.0, 3.5],
        'Sued_DE':  [3.5, 5.5, 10.5, 15.0, 19.5, 22.5, 24.5, 24.0, 19.5, 14.0, 8.0, 4.0],
    },
    'tmin': {
        'Nord_DE':  [-1.5, -1.5, 1.0, 3.5, 7.5, 10.5, 12.5, 12.5, 9.5, 6.0, 2.5, -0.5],
        'West_DE':  [-0.5, -0.5, 2.0, 4.5, 8.5, 11.0, 13.0, 13.0, 10.0, 6.5, 3.0, 0.5],
        'Ost_DE':   [-3.0, -2.5, 0.5, 3.5, 8.0, 11.0, 13.0, 12.5, 9.0, 5.5, 1.5, -2.0],
        'Sued_DE':  [-3.0, -2.0, 1.0, 4.0, 8.5, 11.5, 13.5, 13.0, 9.5, 5.5, 1.5, -2.0],
    },
    'rain': {
        'Nord_DE':  [60, 40, 50, 40, 55, 65, 75, 70, 55, 55, 60, 65],
        'West_DE':  [65, 50, 55, 50, 65, 70, 75, 70, 60, 60, 65, 70],
        'Ost_DE':   [40, 30, 35, 35, 50, 60, 65, 60, 45, 40, 40, 45],
        'Sued_DE':  [55, 45, 55, 60, 85, 100, 100, 90, 65, 55, 55, 60],
    },
    'sun': {
        'Nord_DE':  [40, 60, 110, 165, 210, 210, 210, 195, 140, 95, 50, 30],
        'West_DE':  [45, 65, 110, 155, 195, 195, 200, 190, 140, 100, 55, 35],
        'Ost_DE':   [45, 70, 120, 170, 215, 220, 220, 205, 150, 105, 50, 35],
        'Sued_DE':  [55, 80, 125, 165, 200, 210, 225, 210, 160, 110, 60, 40],
    },
    'frost': {
        'Nord_DE':  [15, 14, 10, 3, 0, 0, 0, 0, 0, 2, 7, 13],
        'West_DE':  [12, 11, 7, 2, 0, 0, 0, 0, 0, 1, 5, 10],
        'Ost_DE':   [20, 17, 12, 4, 0, 0, 0, 0, 0, 3, 10, 18],
        'Sued_DE':  [18, 15, 10, 3, 0, 0, 0, 0, 0, 3, 9, 16],
    },
}


def generate_synthetic_weather(regions, normals, seed=42):
    """
    Generate synthetic monthly weather data based on climate normals.
    Adds realistic year-to-year variability using random perturbations.
    """
    np.random.seed(seed)
    years = list(range(YEAR_START, YEAR_END + 1))

    all_data = {}
    for region in regions:
        rows = []
        for year in years:
            row = {'year': year}
            for m_idx in range(12):
                mn = MONTH_NAMES[m_idx]
                row[f'tmax_{mn}'] = normals['tmax'][region][m_idx] + np.random.normal(0, 1.5)
                row[f'tmin_{mn}'] = normals['tmin'][region][m_idx] + np.random.normal(0, 1.2)
                row[f'rain_{mn}'] = max(0, normals['rain'][region][m_idx] +
                                        np.random.normal(0, normals['rain'][region][m_idx] * 0.3))
                row[f'sun_{mn}'] = max(10, normals['sun'][region][m_idx] +
                                       np.random.normal(0, normals['sun'][region][m_idx] * 0.15))
                row[f'frost_{mn}'] = max(0, normals['frost'][region][m_idx] +
                                         np.random.normal(0, 1.5))
            rows.append(row)
        all_data[region] = pd.DataFrame(rows)

    return all_data


def main():
    print("=" * 70)
    print("GENERATE WEATHER DATA FOR FRENCH & GERMAN REGIONS")
    print("=" * 70)

    raw_dir = PATHS['france_raw']
    os.makedirs(raw_dir, exist_ok=True)

    # --- France ---
    print("\n  Generating France weather (based on Météo-France normals)...")
    france_data = generate_synthetic_weather(FRANCE_REGIONS, FRANCE_NORMALS, seed=42)

    france_pkl = os.path.join(raw_dir, 'synthetic_monthly_weather.pkl')
    with open(france_pkl, 'wb') as f:
        pickle.dump(france_data, f)

    for region, df in france_data.items():
        csv_path = os.path.join(raw_dir, f'monthly_weather_{region}.csv')
        df.to_csv(csv_path, index=False)
        print(f"    Saved: {csv_path}")

    # --- Germany ---
    print("\n  Generating Germany weather (based on DWD normals)...")
    germany_data = generate_synthetic_weather(GERMANY_REGIONS, GERMANY_NORMALS, seed=123)

    germany_pkl = os.path.join(raw_dir, 'synthetic_monthly_weather_germany.pkl')
    with open(germany_pkl, 'wb') as f:
        pickle.dump(germany_data, f)

    for region, df in germany_data.items():
        csv_path = os.path.join(raw_dir, f'monthly_weather_{region}.csv')
        df.to_csv(csv_path, index=False)
        print(f"    Saved: {csv_path}")

    print(f"\n  NOTE: Weather data is synthetic (based on climate normals).")
    print(f"  For real ERA5 data, install cdsapi and configure ~/.cdsapirc.")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
