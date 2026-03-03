"""
Process E-OBS Weather Data → Monthly Averages per Macro-Region
================================================================
Extracts daily Tmax, Tmin, Precipitation from E-OBS gridded data,
computes monthly aggregates, and derives frost days from Tmin.

Uses NUTS3 administrative boundaries to spatially average grid cells
within each macro-region (4 French + 4 German regions).

Input:
    data/weather/eobs/{tx,tn,rr}_{1995-2010,2011-2025}.nc
    data/weather/gadm/NUTS_RG_01M_2021_4326_LEVL_3.geojson

Output:
    data/france/raw/eobs_monthly_weather_{region}.csv
    (for each of the 8 macro-regions)

Usage:
    python src/france/process_eobs_weather.py
"""

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    FRANCE_REGIONS, GERMANY_REGIONS,
    YEAR_START, YEAR_END, PATHS
)

EOBS_DIR = os.path.join(PATHS['france_raw'], '..', '..', 'weather', 'eobs')
GADM_PATH = os.path.join(PATHS['france_raw'], '..', '..', 'weather', 'gadm',
                          'NUTS_RG_01M_2021_4326_LEVL_3.geojson')
MONTH_NAMES = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
               'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

# NUTS3 prefix → macro-region mapping
# French NUTS3 codes start with FR, German with DE
NUTS3_TO_REGION = {
    # --- France: Nord ---
    'FR101': 'Nord', 'FR102': 'Nord', 'FR103': 'Nord', 'FR104': 'Nord',
    'FR105': 'Nord', 'FR106': 'Nord', 'FR107': 'Nord', 'FR108': 'Nord',
    'FRE11': 'Nord', 'FRE12': 'Nord',  # Nord, Pas-de-Calais
    'FRE21': 'Nord', 'FRE22': 'Nord', 'FRE23': 'Nord',  # Aisne, Oise, Somme
    'FRD21': 'Nord', 'FRD22': 'Nord',  # Eure, Seine-Maritime

    # --- France: Ouest ---
    'FRD11': 'Ouest', 'FRD12': 'Ouest', 'FRD13': 'Ouest',  # Calvados, Manche, Orne
    'FRH01': 'Ouest', 'FRH02': 'Ouest', 'FRH03': 'Ouest', 'FRH04': 'Ouest',  # Bretagne
    'FRG01': 'Ouest', 'FRG02': 'Ouest', 'FRG03': 'Ouest', 'FRG04': 'Ouest', 'FRG05': 'Ouest',  # Pays de la Loire
    'FRI31': 'Ouest', 'FRI32': 'Ouest', 'FRI33': 'Ouest', 'FRI34': 'Ouest',  # Poitou-Charentes

    # --- France: Est ---
    'FRF21': 'Est', 'FRF22': 'Est', 'FRF23': 'Est', 'FRF24': 'Est',  # Champagne-Ardenne
    'FRF31': 'Est', 'FRF32': 'Est', 'FRF33': 'Est', 'FRF34': 'Est',  # Lorraine
    'FRF11': 'Est', 'FRF12': 'Est',  # Alsace
    'FRC11': 'Est', 'FRC12': 'Est', 'FRC13': 'Est', 'FRC14': 'Est',  # Bourgogne
    'FRC21': 'Est', 'FRC22': 'Est', 'FRC23': 'Est', 'FRC24': 'Est',  # Franche-Comté
    'FRB01': 'Est', 'FRB02': 'Est', 'FRB03': 'Est', 'FRB04': 'Est', 'FRB05': 'Est', 'FRB06': 'Est',  # Centre

    # --- France: Sud ---
    'FRI11': 'Sud', 'FRI12': 'Sud', 'FRI13': 'Sud', 'FRI14': 'Sud', 'FRI15': 'Sud',  # Aquitaine
    'FRI21': 'Sud', 'FRI22': 'Sud', 'FRI23': 'Sud',  # Limousin
    'FRJ21': 'Sud', 'FRJ22': 'Sud', 'FRJ23': 'Sud', 'FRJ24': 'Sud', 'FRJ25': 'Sud',  # Midi-Pyrénées
    'FRJ26': 'Sud', 'FRJ27': 'Sud', 'FRJ28': 'Sud',
    'FRJ11': 'Sud', 'FRJ12': 'Sud', 'FRJ13': 'Sud', 'FRJ14': 'Sud', 'FRJ15': 'Sud',  # Languedoc-Roussillon
    'FRL01': 'Sud', 'FRL02': 'Sud', 'FRL03': 'Sud', 'FRL04': 'Sud', 'FRL05': 'Sud', 'FRL06': 'Sud',  # PACA
    'FRK21': 'Sud', 'FRK22': 'Sud', 'FRK23': 'Sud', 'FRK24': 'Sud', 'FRK25': 'Sud',  # Rhône-Alpes
    'FRK26': 'Sud', 'FRK27': 'Sud', 'FRK28': 'Sud',
    'FRK11': 'Sud', 'FRK12': 'Sud', 'FRK13': 'Sud', 'FRK14': 'Sud',  # Auvergne
    'FRM01': 'Sud', 'FRM02': 'Sud',  # Corsica

    # --- Germany: Nord_DE (Schleswig-Holstein, Niedersachsen, MV, Hamburg, Bremen) ---
    # DE6=Hamburg, DE5=Bremen, DE9=Niedersachsen, DEF=Schleswig-Holstein, DE8=MV
    # --- Germany: West_DE (NRW, Rheinland-Pfalz, Saarland, Hessen) ---
    # DEA=NRW, DEB=Rheinland-Pfalz, DEC=Saarland, DE7=Hessen
    # --- Germany: Ost_DE (Brandenburg, Sachsen-Anhalt, Sachsen, Thüringen, Berlin) ---
    # DE4=Brandenburg, DEE=Sachsen-Anhalt, DED=Sachsen, DEG=Thüringen, DE3=Berlin
    # --- Germany: Sued_DE (Bayern, Baden-Württemberg) ---
    # DE1=Baden-Württemberg, DE2=Bayern
}


def get_germany_region(nuts_id):
    """Map German NUTS3 ID to macro-region using first 3 chars."""
    prefix = nuts_id[:3]
    mapping = {
        'DEF': 'Nord_DE', 'DE9': 'Nord_DE', 'DE8': 'Nord_DE',
        'DE6': 'Nord_DE', 'DE5': 'Nord_DE',
        'DEA': 'West_DE', 'DEB': 'West_DE', 'DEC': 'West_DE', 'DE7': 'West_DE',
        'DE4': 'Ost_DE', 'DEE': 'Ost_DE', 'DED': 'Ost_DE', 'DEG': 'Ost_DE', 'DE3': 'Ost_DE',
        'DE2': 'Sued_DE', 'DE1': 'Sued_DE',
    }
    return mapping.get(prefix)


def build_region_masks(nuts3_gdf, eobs_lats, eobs_lons):
    """
    For each macro-region, find which E-OBS grid cells fall within it.
    Returns dict of {region_name: boolean mask (lat, lon)}.
    """
    print("  Building spatial masks for macro-regions...")

    # Create grid of points
    lon_grid, lat_grid = np.meshgrid(eobs_lons, eobs_lats)

    # Assign each NUTS3 region to a macro-region
    nuts3_gdf = nuts3_gdf.copy()
    macro_regions = []
    for _, row in nuts3_gdf.iterrows():
        nid = row['NUTS_ID']
        if nid in NUTS3_TO_REGION:
            macro_regions.append(NUTS3_TO_REGION[nid])
        elif nid.startswith('DE'):
            macro_regions.append(get_germany_region(nid))
        else:
            macro_regions.append(None)
    nuts3_gdf['macro_region'] = macro_regions

    # Dissolve NUTS3 polygons into macro-regions
    valid = nuts3_gdf[nuts3_gdf['macro_region'].notna()]
    dissolved = valid.dissolve(by='macro_region')

    all_regions = list(FRANCE_REGIONS) + list(GERMANY_REGIONS)
    masks = {}

    for region in all_regions:
        if region not in dissolved.index:
            print(f"    WARNING: No geometry for {region}")
            continue

        geom = dissolved.loc[region, 'geometry']

        # Sample grid points at 0.5° spacing to speed up (still covers region well)
        # Then use contains() for a subset of points
        mask = np.zeros((len(eobs_lats), len(eobs_lons)), dtype=bool)

        # Use vectorized approach: check which grid cell centers are inside the region
        from shapely import contains_xy
        mask = contains_xy(geom, lon_grid, lat_grid)

        n_cells = mask.sum()
        print(f"    {region}: {n_cells} grid cells")
        masks[region] = mask

    return masks


def load_eobs_variable(var_name, time_start, time_end, lat_slice=None, lon_slice=None):
    """Load E-OBS variable, concatenating time chunks. Crops spatially to save memory."""
    datasets = []
    for period in ['1995-2010', '2011-2025']:
        path = os.path.join(EOBS_DIR, f'{var_name}_{period}.nc')
        if os.path.exists(path):
            ds = xr.open_dataset(path)
            ds = ds.sel(time=slice(time_start, time_end))
            if lat_slice:
                ds = ds.sel(latitude=lat_slice)
            if lon_slice:
                ds = ds.sel(longitude=lon_slice)
            if len(ds.time) > 0:
                datasets.append(ds)

    if not datasets:
        raise FileNotFoundError(f"No E-OBS files found for {var_name}")

    if len(datasets) == 1:
        return datasets[0]

    return xr.concat(datasets, dim='time')


def compute_monthly_weather(masks, time_start='2004-01-01', time_end='2016-12-31',
                            lat_slice=None, lon_slice=None):
    """
    Compute monthly weather statistics per macro-region from E-OBS data.
    Returns dict of {region: DataFrame} with monthly Tmax, Tmin, Rain, Frost.
    """
    print("\n  Loading E-OBS data and computing monthly averages...")

    # Load variables (cropped spatially)
    print("    Loading Tmax...")
    ds_tx = load_eobs_variable('tx', time_start, time_end, lat_slice, lon_slice)
    print("    Loading Tmin...")
    ds_tn = load_eobs_variable('tn', time_start, time_end, lat_slice, lon_slice)
    print("    Loading Precipitation...")
    ds_rr = load_eobs_variable('rr', time_start, time_end, lat_slice, lon_slice)

    tx = ds_tx['tx']  # Daily max temperature (°C)
    tn = ds_tn['tn']  # Daily min temperature (°C)
    rr = ds_rr['rr']  # Daily precipitation (mm)

    all_regions = {}

    for region, mask in masks.items():
        print(f"    Processing {region}...")

        # Apply mask using numpy (avoids xarray coordinate alignment issues)
        # mask shape: (n_lat, n_lon), data shape: (n_time, n_lat, n_lon)
        mask3d = mask[np.newaxis, :, :]  # broadcast to (1, n_lat, n_lon)

        tx_vals = tx.values.copy()
        tx_vals[~np.broadcast_to(mask3d, tx_vals.shape)] = np.nan
        tx_region = np.nanmean(tx_vals, axis=(1, 2))

        tn_vals = tn.values.copy()
        tn_vals[~np.broadcast_to(mask3d, tn_vals.shape)] = np.nan
        tn_region = np.nanmean(tn_vals, axis=(1, 2))

        rr_vals = rr.values.copy()
        rr_vals[~np.broadcast_to(mask3d, rr_vals.shape)] = np.nan
        rr_region = np.nanmean(rr_vals, axis=(1, 2))

        # Frost: daily fraction of masked cells with Tmin < 0
        tn_frost = (tn_vals < 0).astype(float)
        tn_frost[~np.broadcast_to(mask3d, tn_frost.shape)] = np.nan
        frost_region = np.nanmean(tn_frost, axis=(1, 2))

        # Get time index from the dataset
        times = pd.DatetimeIndex(tx.time.values)

        # Build daily DataFrame for aggregation
        daily_df = pd.DataFrame({
            'time': times,
            'tmax': tx_region,
            'tmin': tn_region,
            'rain': rr_region,
            'frost_flag': (tn_region < 0).astype(float),
        })
        daily_df['year'] = daily_df['time'].dt.year
        daily_df['month'] = daily_df['time'].dt.month

        # Monthly aggregation
        monthly = daily_df.groupby(['year', 'month']).agg(
            tmax=('tmax', 'mean'),
            tmin=('tmin', 'mean'),
            rain=('rain', 'sum'),       # sum daily means → monthly total (mean of spatial cells)
            frost=('frost_flag', 'sum'),  # count days where region-mean Tmin < 0
        ).reset_index()

        # Build year × month structure
        years = range(YEAR_START, YEAR_END + 1)
        rows = []
        for year in years:
            row = {'year': year}
            year_data = monthly[monthly['year'] == year]
            for m in range(1, 13):
                mn = MONTH_NAMES[m - 1]
                m_data = year_data[year_data['month'] == m]
                if len(m_data) > 0:
                    row[f'tmax_{mn}'] = float(m_data['tmax'].values[0])
                    row[f'tmin_{mn}'] = float(m_data['tmin'].values[0])
                    row[f'rain_{mn}'] = float(m_data['rain'].values[0])
                    row[f'frost_{mn}'] = float(m_data['frost'].values[0])
                else:
                    row[f'tmax_{mn}'] = np.nan
                    row[f'tmin_{mn}'] = np.nan
                    row[f'rain_{mn}'] = np.nan
                    row[f'frost_{mn}'] = np.nan

                # Sunshine: not available from E-OBS (qq requires registration)
                row[f'sun_{mn}'] = np.nan

            rows.append(row)

        all_regions[region] = pd.DataFrame(rows)

    ds_tx.close()
    ds_tn.close()
    ds_rr.close()

    return all_regions


def main():
    print("=" * 70)
    print("PROCESS E-OBS WEATHER → MONTHLY AVERAGES PER REGION")
    print("=" * 70)

    # Load NUTS3 boundaries
    print(f"\n  Loading NUTS3 boundaries: {GADM_PATH}")
    nuts3 = gpd.read_file(GADM_PATH)

    # Filter to France + Germany only
    nuts3_fg = nuts3[nuts3['CNTR_CODE'].isin(['FR', 'DE'])].copy()
    # Exclude overseas territories
    nuts3_fg = nuts3_fg[~nuts3_fg['NUTS_ID'].str.startswith('FRY')]
    print(f"  France+Germany NUTS3 regions: {len(nuts3_fg)}")

    # Load E-OBS coordinates (from any file), cropped to France+Germany bbox
    sample_path = os.path.join(EOBS_DIR, 'tx_1995-2010.nc')
    ds_sample = xr.open_dataset(sample_path)
    lat_sl = slice(42, 55.5)
    lon_sl = slice(-5.5, 15.5)
    lats = ds_sample.latitude.sel(latitude=lat_sl).values
    lons = ds_sample.longitude.sel(longitude=lon_sl).values
    ds_sample.close()
    # Store slices for data loading
    lat_slice = lat_sl
    lon_slice = lon_sl

    print(f"  Grid: {len(lats)} lat x {len(lons)} lon points")

    # Build spatial masks
    masks = build_region_masks(nuts3_fg, lats, lons)

    if not masks:
        print("  ERROR: No masks created. Check NUTS3 mapping.")
        return

    # Compute monthly weather (pass lat/lon slices for cropping)
    monthly_data = compute_monthly_weather(masks, lat_slice=lat_slice, lon_slice=lon_slice)

    # Save
    raw_dir = PATHS['france_raw']
    for region, df in monthly_data.items():
        csv_path = os.path.join(raw_dir, f'eobs_monthly_weather_{region}.csv')
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

        # Print sample for validation
        sample_year = df[df['year'] == 2010].iloc[0]
        print(f"    {region} 2010 Jul: Tmax={sample_year.get('tmax_jul', 'N/A'):.1f}°C, "
              f"Tmin={sample_year.get('tmin_jul', 'N/A'):.1f}°C, "
              f"Rain={sample_year.get('rain_jul', 'N/A'):.0f}mm")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
