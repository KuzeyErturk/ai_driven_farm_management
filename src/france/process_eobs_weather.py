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

# NUTS3 match with macro-region mapping
# French NUTS3 code begin with FR, German with DE
NUTS3_TO_REGION = {
    # Nord ( france ) meaning north region
    'FR101': 'Nord', 'FR102': 'Nord', 'FR103': 'Nord', 'FR104': 'Nord',
    'FR105': 'Nord', 'FR106': 'Nord', 'FR107': 'Nord', 'FR108': 'Nord',
    'FRE11': 'Nord', 'FRE12': 'Nord',
    'FRE21': 'Nord', 'FRE22': 'Nord', 'FRE23': 'Nord',
    'FRD21': 'Nord', 'FRD22': 'Nord', 

    # France: Ouest meaning west
    'FRD11': 'Ouest', 'FRD12': 'Ouest', 'FRD13': 'Ouest',  
    'FRH01': 'Ouest', 'FRH02': 'Ouest', 'FRH03': 'Ouest', 'FRH04': 'Ouest',  
    'FRG01': 'Ouest', 'FRG02': 'Ouest', 'FRG03': 'Ouest', 'FRG04': 'Ouest', 'FRG05': 'Ouest',
    'FRI31': 'Ouest', 'FRI32': 'Ouest', 'FRI33': 'Ouest', 'FRI34': 'Ouest', 

    # France: Est - east   
    'FRF31': 'Est', 'FRF32': 'Est', 'FRF33': 'Est', 'FRF34': 'Est',
    'FRF11': 'Est', 'FRF12': 'Est',
    'FRC11': 'Est', 'FRC12': 'Est', 'FRC13': 'Est', 'FRC14': 'Est',
    'FRC21': 'Est', 'FRC22': 'Est', 'FRC23': 'Est', 'FRC24': 'Est',
    'FRB01': 'Est', 'FRB02': 'Est', 'FRB03': 'Est', 'FRB04': 'Est', 'FRB05': 'Est', 'FRB06': 'Est', 

    # France: Sud    south
    'FRI21': 'Sud', 'FRI22': 'Sud', 'FRI23': 'Sud',
    'FRJ21': 'Sud', 'FRJ22': 'Sud', 'FRJ23': 'Sud', 'FRJ24': 'Sud', 'FRJ25': 'Sud', 
    'FRJ26': 'Sud', 'FRJ27': 'Sud', 'FRJ28': 'Sud',
    'FRJ11': 'Sud', 'FRJ12': 'Sud', 'FRJ13': 'Sud', 'FRJ14': 'Sud', 'FRJ15': 'Sud',  
    'FRL01': 'Sud', 'FRL02': 'Sud', 'FRL03': 'Sud', 'FRL04': 'Sud', 'FRL05': 'Sud', 'FRL06': 'Sud', 
    'FRK21': 'Sud', 'FRK22': 'Sud', 'FRK23': 'Sud', 'FRK24': 'Sud', 'FRK25': 'Sud', 
    'FRK26': 'Sud', 'FRK27': 'Sud', 'FRK28': 'Sud',
    'FRK11': 'Sud', 'FRK12': 'Sud', 'FRK13': 'Sud', 'FRK14': 'Sud', 
    'FRM01': 'Sud', 'FRM02': 'Sud',

}


def get_germany_region(nuts_id):
    # Mapping German regions to match using the first 3 characters
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
    valid = nuts3_gdf[nuts3_gdf['macro_region'].notna()]
    dissolved = valid.dissolve(by='macro_region')

    all_regions = list(FRANCE_REGIONS) + list(GERMANY_REGIONS)
    masks = {}

    for region in all_regions:
        if region not in dissolved.index:
            print(f"{region}") # Error log
            continue

        geom = dissolved.loc[region, 'geometry']

        mask = np.zeros((len(eobs_lats), len(eobs_lons)), dtype=bool)

        from shapely import contains_xy
        mask = contains_xy(geom, lon_grid, lat_grid)

        n_cells = mask.sum()
        print(f"    {region}: {n_cells} grid cells")
        masks[region] = mask

    return masks

# Loading EOBS variables
def load_eobs_variable(var_name, time_start, time_end, lat_slice=None, lon_slice=None):
    datasets = []

    candidates = []
    for period in ['1995-2010', '2011-2025']:
        candidates.append(os.path.join(EOBS_DIR, f'{var_name}_{period}.nc'))
    # Also check for full files
    import glob
    candidates += glob.glob(os.path.join(EOBS_DIR, f'{var_name}_ens_mean_*.nc'))

    for path in candidates:
        if os.path.exists(path) and path not in [d.encoding.get('source', '') for d in datasets]:
            ds = xr.open_dataset(path)
            ds = ds.sel(time=slice(time_start, time_end))
            if lat_slice:
                ds = ds.sel(latitude=lat_slice)
            if lon_slice:
                ds = ds.sel(longitude=lon_slice)
            if len(ds.time) > 0:
                datasets.append(ds)
                if len(ds.time) > 3000:
                    break

    if not datasets:
        raise FileNotFoundError(f"no EOBS file {var_name} in {EOBS_DIR}")

    if len(datasets) == 1:
        return datasets[0]

    return xr.concat(datasets, dim='time')


def compute_monthly_weather(masks, time_start='2004-01-01', time_end='2016-12-31',
                            lat_slice=None, lon_slice=None):
    ds_tx = load_eobs_variable('tx', time_start, time_end, lat_slice, lon_slice)
    ds_tn = load_eobs_variable('tn', time_start, time_end, lat_slice, lon_slice)
    ds_rr = load_eobs_variable('rr', time_start, time_end, lat_slice, lon_slice)

    ds_qq = None
    try:
        ds_qq = load_eobs_variable('qq', time_start, time_end, lat_slice, lon_slice)
    except FileNotFoundError:
        print("no radiation")

    tx = ds_tx['tx']  # daily max temp
    tn = ds_tn['tn']  # daily min temp
    rr = ds_rr['rr']  # daily precipiation ( rainfall)

    all_regions = {}

    for region, mask in masks.items():
        mask3d = mask[np.newaxis, :, :]

        tx_vals = tx.values.copy()
        tx_vals[~np.broadcast_to(mask3d, tx_vals.shape)] = np.nan
        tx_region = np.nanmean(tx_vals, axis=(1, 2))

        tn_vals = tn.values.copy()
        tn_vals[~np.broadcast_to(mask3d, tn_vals.shape)] = np.nan
        tn_region = np.nanmean(tn_vals, axis=(1, 2))

        rr_vals = rr.values.copy()
        rr_vals[~np.broadcast_to(mask3d, rr_vals.shape)] = np.nan
        rr_region = np.nanmean(rr_vals, axis=(1, 2))

        tn_frost = (tn_vals < 0).astype(float)
        tn_frost[~np.broadcast_to(mask3d, tn_frost.shape)] = np.nan
        frost_region = np.nanmean(tn_frost, axis=(1, 2))

        # radiation to sunshine mapping
        if ds_qq is not None:
            qq = ds_qq['qq']
            qq_vals = qq.values.copy()
            qq_vals[~np.broadcast_to(mask3d, qq_vals.shape)] = np.nan
            qq_region = np.nanmean(qq_vals, axis=(1, 2))  # daily mean W/m²
        else:
            qq_region = None

        times = pd.DatetimeIndex(tx.time.values)
        daily_df = pd.DataFrame({
            'time': times,
            'tmax': tx_region,
            'tmin': tn_region,
            'rain': rr_region,
            'frost_flag': (tn_region < 0).astype(float),
        })

        if qq_region is not None:
            daily_df['sunshine'] = qq_region / 25.0

        daily_df['year'] = daily_df['time'].dt.year
        daily_df['month'] = daily_df['time'].dt.month

        agg_dict = {
            'tmax': ('tmax', 'mean'),
            'tmin': ('tmin', 'mean'),
            'rain': ('rain', 'sum'),
            'frost': ('frost_flag', 'sum'),
        }
        if 'sunshine' in daily_df.columns:
            agg_dict['sunshine'] = ('sunshine', 'sum')  # monthly total sunshine hours

        monthly = daily_df.groupby(['year', 'month']).agg(**agg_dict).reset_index()

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
                    if 'sunshine' in m_data.columns:
                        row[f'sun_{mn}'] = float(m_data['sunshine'].values[0])
                    else:
                        row[f'sun_{mn}'] = np.nan
                else:
                    row[f'tmax_{mn}'] = np.nan
                    row[f'tmin_{mn}'] = np.nan
                    row[f'rain_{mn}'] = np.nan
                    row[f'frost_{mn}'] = np.nan
                    row[f'sun_{mn}'] = np.nan

            rows.append(row)

        all_regions[region] = pd.DataFrame(rows)

    ds_tx.close()
    ds_tn.close()
    ds_rr.close()
    if ds_qq is not None:
        ds_qq.close()

    return all_regions


def main():

    print(f"\n  Loading NUTS3 boundaries: {GADM_PATH}")
    nuts3 = gpd.read_file(GADM_PATH)

    # Filter to France + Germany only
    nuts3_fg = nuts3[nuts3['CNTR_CODE'].isin(['FR', 'DE'])].copy()
    # Exclude overseas territories
    nuts3_fg = nuts3_fg[~nuts3_fg['NUTS_ID'].str.startswith('FRY')]
    print(f"  France+Germany NUTS3 regions: {len(nuts3_fg)}")

    sample_path = os.path.join(EOBS_DIR, 'tx_1995-2010.nc')
    ds_sample = xr.open_dataset(sample_path)
    lat_sl = slice(42, 55.5)
    lon_sl = slice(-5.5, 15.5)
    lats = ds_sample.latitude.sel(latitude=lat_sl).values
    lons = ds_sample.longitude.sel(longitude=lon_sl).values
    ds_sample.close()
    lat_slice = lat_sl
    lon_slice = lon_sl

    print(f"  Grid: {len(lats)} lat x {len(lons)} lon points")

    masks = build_region_masks(nuts3_fg, lats, lons)

    if not masks:
        print("  ERROR: No masks created. Check NUTS3 mapping.")
        return

    monthly_data = compute_monthly_weather(masks, lat_slice=lat_slice, lon_slice=lon_slice)

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

if __name__ == '__main__':
    main()
