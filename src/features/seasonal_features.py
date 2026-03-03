import pandas as pd
import numpy as np

print("="*70)
print("SEASONAL WEATHER FEATURE ENGINEERING")
print("="*70)

# ============================================================================
# STEP 1: LOAD MONTHLY WEATHER DATA
# ============================================================================

# Monthly Maximum Temperature
tmax_text = """
year    jan    feb    mar    apr    may    jun    jul    aug    sep    oct    nov    dec
2004    7.1    7.4    9.3   12.3   15.7   18.2   18.6   20.0   17.2   12.4    9.7    8.0
2005    8.1    6.4    9.6   11.9   14.4   18.5   19.4   19.3   17.8   14.6    9.0    7.1
2006    6.4    6.2    7.1   11.4   15.0   19.4   23.3   18.8   19.2   14.9   10.5    8.2
2007    8.7    8.1   10.1   15.2   14.8   17.7   18.0   18.5   16.5   13.9    9.9    7.4
2008    8.1    8.9    8.5   11.1   16.8   17.3   19.4   18.5   16.1   12.2    8.9    6.0
2009    5.7    6.4    9.9   13.3   15.3   18.5   19.2   19.3   17.0   13.8   10.0    5.1
2010    3.6    4.8    8.9   12.7   14.4   19.1   19.6   18.2   16.6   12.9    7.2    2.4
2011    5.8    8.1    9.8   15.7   15.2   17.2   18.7   18.0   17.5   14.5   11.6    7.7
2012    7.5    7.2   12.1   10.2   15.0   15.9   17.9   19.4   15.9   11.6    8.7    6.7
2013    5.6    5.7    5.1   10.4   13.8   17.2   22.2   19.8   16.6   14.1    8.6    8.7
2014    7.4    7.8   10.5   13.2   15.2   18.6   21.2   18.0   18.2   14.3   10.4    7.4
2015    6.6    6.6    9.1   12.9   13.5   17.3   18.6   18.8   16.1   13.7   11.0   10.7
2016    7.3    7.0    8.9   10.6   15.9   17.8   19.3   19.7   18.3   13.4    8.1    8.9
2017    6.7    7.9   10.9   12.2   16.8   18.4   19.3   18.6   16.1   14.3    8.8    7.0
2018    6.7    5.3    6.9   12.0   17.2   19.9   22.6   19.3   16.4   13.3   10.1    8.4
2019    6.2   10.0   10.1   12.9   14.6   17.3   20.7   19.9   17.1   12.2    7.9    7.9
2020    8.3    8.0    9.3   14.2   16.7   18.3   18.3   19.9   17.1   12.4   10.7    6.8
2021    4.9    7.0    9.8   10.9   13.4   18.7   21.1   18.9   18.6   14.0    9.9    7.8
2022    7.7    8.7   11.1   12.6   15.9   18.6   21.3   21.7   17.4   14.9   10.9    5.8
2023    7.3    9.0    9.0   12.0   16.2   21.2   18.9   19.3   19.4   14.0    9.3    8.3
2024    6.6    9.2   10.0   12.1   17.2   17.3   19.0   19.5   16.4   13.8    9.6    8.6
"""

# Monthly Minimum Temperature
tmin_text = """
year    jan    feb    mar    apr    may    jun    jul    aug    sep    oct    nov    dec
2004    1.7    1.3    2.1    4.7    6.5    9.8   10.3   12.2    9.7    6.2    4.3    2.0
2005    2.5    1.0    3.2    3.9    5.7    9.8   11.2   10.4    9.8    8.7    2.2    1.2
2006    1.2    0.7    0.8    3.4    6.5    9.6   12.4   11.1   11.3    8.4    3.9    2.9
2007    3.1    2.1    2.5    5.3    6.5    9.8   10.5   10.2    8.7    6.6    3.8    1.4
2008    2.4    0.9    1.7    3.0    7.5    8.6   11.2   11.8    8.8    5.1    3.5    0.2
2009   -0.1    0.9    2.2    4.5    6.4    9.0   11.1   11.4    9.4    7.0    4.6   -0.8
2010   -1.7   -1.1    1.3    3.2    5.1    9.3   11.6   10.2    9.0    6.0    1.4   -4.1
2011    0.4    2.6    1.8    5.7    6.8    8.2    9.7   10.2   10.0    7.9    5.8    2.0
2012    1.8    1.2    3.4    2.5    6.0    8.7   10.3   11.3    8.0    4.8    2.8    1.0
2013    1.0   -0.2   -0.7    2.2    5.2    8.5   11.9   11.4    8.9    8.3    2.3    2.7
2014    2.1    2.5    2.8    5.3    7.3    9.8   11.5    9.9    9.7    7.8    4.8    1.5
2015    0.8    0.5    1.9    3.0    5.7    8.1   10.2   10.6    7.6    6.3    5.3    5.2
2016    1.8    0.6    1.6    2.5    6.7   10.0   11.3   11.3   10.9    6.2    1.7    2.9
2017    1.0    2.6    3.6    3.8    7.4   10.5   11.0   10.5    9.0    8.1    2.7    1.3
2018    1.4   -0.7    0.7    4.8    6.9    9.8   11.9   11.1    8.4    5.8    4.5    3.2
2019    0.8    2.0    3.5    3.8    5.5    9.2   12.1   11.7    9.1    5.7    2.7    2.4
2020    3.0    2.1    1.9    4.0    6.1    9.8   10.4   11.9    8.6    6.4    4.8    1.9
2021   -0.5    1.1    3.0    0.5    4.9    9.8   12.2   11.1   10.7    7.8    4.0    2.8
2022    1.7    2.5    2.3    3.5    7.8    9.3   11.8   11.6    9.4    8.3    5.5   -0.0
2023    1.5    2.6    2.4    3.7    7.0   10.6   11.1   11.4   11.0    7.6    3.4    3.3
2024    1.0    3.5    3.5    4.6    9.0    8.6   10.6   11.3    8.9    7.2    3.7    3.8
"""

# Monthly Rainfall
rain_text = """
year    jan    feb    mar    apr    may    jun    jul    aug    sep    oct    nov    dec
2004  150.1   74.4   70.1   92.8   51.0   84.8   70.2  159.7  103.1  165.4   75.2  110.6
2005  126.7   69.9   72.7   90.9   72.7   71.7   59.4   85.5   91.9  141.7  115.9   80.6
2006   59.3   65.0  110.1   67.5  112.6   42.0   53.1   92.0  102.5  142.0  153.1  172.5
2007  152.3  110.0   87.4   26.4  114.1  136.1  134.6   85.8   67.0   64.5  100.4  114.0
2008  189.3   81.8  122.6   76.2   47.7   78.7  108.1  135.0  105.8  154.4  105.8   87.5
2009  119.4   59.8   78.9   60.0   81.7   61.9  145.5  114.5   63.8  107.4  214.8  100.3
2010   79.6   74.7   78.7   47.5   38.9   38.4  107.0   97.5  113.3  100.5  122.7   47.1
2011  102.2  113.8   49.5   35.7  100.7   84.7   75.7  105.4  107.4  122.0   99.0  166.7
2012  110.2   59.4   36.7  128.2   65.6  149.0  118.5  111.3  111.8  125.6  135.1  178.3
2013  109.7   59.6   64.4   62.9   91.8   48.5   65.2   72.2   71.0  162.6   90.4  185.6
2014  186.7  167.6   79.7   67.5   99.3   54.7   64.0  138.0   22.6  157.1  123.3  131.3
2015  153.6   79.9   94.9   45.5  109.7   56.0  107.6  104.8   51.9   72.0  171.9  216.3
2016  178.4  110.1   83.8   77.4   61.5   96.0   78.5   85.2   96.3   47.0  104.9   77.2
2017   76.0   93.4   97.8   33.2   57.2  110.4  101.2  102.3  117.2  101.4  107.6  120.8
2018  132.4   68.6  103.9   83.7   47.7   34.6   54.2   83.0  101.5  104.2  121.6  117.6
2019   64.6   71.1  128.9   48.5   63.6  109.0   89.0  132.7  126.1  139.7  119.1  139.7
2020  121.2  213.7   78.6   30.0   32.8  107.5   95.8  122.2   77.3  182.8  104.8  166.9
2021  140.0  105.5   87.1   20.6  120.8   44.7   74.6   66.7   82.5  168.0   80.9  114.5
2022   63.1  152.2   50.5   49.6   77.0   61.8   48.2   51.6  105.0  147.1  167.6  118.2
2023  129.3   45.4  134.5   73.4   39.6   54.7  141.9   89.6  121.4  177.5  121.5  194.5
2024  122.5  144.0  107.9  109.6   83.3   55.2   84.8  104.5  116.6  105.5   93.8  145.3
"""

# Monthly Sunshine
sun_text = """
year    jan    feb    mar    apr    may    jun    jul    aug    sep    oct    nov    dec
2004   42.0   86.9  109.4  124.2  196.5  168.9  152.5  162.4  140.9   82.6   45.1   45.5
2005   47.3   68.8   81.6  146.5  202.4  166.2  161.9  182.1  130.7   71.4   77.4   46.0
2006   45.4   67.1   88.0  162.0  176.0  206.7  249.8  135.5  137.9   84.8   77.1   42.1
2007   50.8   67.2  141.8  203.2  174.1  140.6  160.0  169.5  128.2   99.6   56.7   44.1
2008   38.1  101.3  114.7  153.9  200.4  178.7  166.2  104.1  102.2  109.1   53.1   53.6
2009   51.3   51.3  137.0  159.3  210.1  202.8  169.1  154.3  128.2   83.3   59.4   50.0
2010   53.0   61.6  115.7  181.3  199.1  208.4  138.7  146.6  125.3  104.7   64.0   46.3
2011   49.5   51.2  120.9  202.2  190.5  174.4  169.4  126.5  126.5   90.0   58.6   38.2
2012   57.9   62.4  149.5  126.2  198.2  114.0  136.3  150.5  138.3   90.1   61.5   45.7
2013   36.1   74.8   80.7  166.3  176.2  169.4  247.4  157.6  113.6   77.1   70.0   41.6
2014   42.7   74.8  126.0  144.1  148.8  176.6  221.9  169.5  122.1   82.0   51.1   56.7
2015   58.4   75.6  121.3  211.8  173.8  188.7  160.8  148.1  151.9   90.4   35.5   29.1
2016   37.3   84.5  116.5  160.4  208.1  134.7  156.6  180.8  119.4  104.5   74.3   40.4
2017   54.7   54.7  119.0  158.3  207.6  155.5  168.2  155.1  108.4   72.4   71.0   44.9
2018   49.4   98.1   83.6  131.4  240.6  234.4  234.0  147.1  132.4  111.4   61.8   36.4
2019   50.8   99.1  115.2  168.7  186.9  159.7  173.3  175.1  142.9   88.4   48.1   46.0
2020   45.1   73.4  136.0  223.0  266.9  161.2  144.5  145.4  145.1   66.9   53.3   37.5
2021   44.2   71.4  102.2  223.8  161.4  181.0  190.4  127.1  118.8   80.2   58.6   26.9
2022   63.1   74.1  165.5  167.4  154.2  196.8  178.4  208.5  118.8  104.9   55.4   48.3
2023   62.3   70.4   82.4  157.4  205.9  242.9  141.5  151.1  139.9   84.3   64.7   27.2
2024   59.9   57.2   96.9  124.9  160.1  178.4  154.9  161.3  122.8   89.2   50.9   24.2
"""

# Monthly Frost Days
frost_text = """
year    jan    feb    mar    apr    may    jun    jul    aug    sep    oct    nov    dec
2004   10.3   13.1    7.2    1.2    0.3    0.0    0.0    0.0    0.0    0.5    4.6    9.3
2005    6.1   11.4    8.4    3.0    1.7    0.1    0.0    0.0    0.0    0.2   11.2   11.5
2006   10.7   11.5   13.6    4.8    0.8    0.0    0.0    0.0    0.0    0.1    3.7    6.5
2007    6.0    8.2    5.2    1.3    0.6    0.0    0.0    0.0    0.4    0.8    5.0   11.8
2008    6.7   11.3    8.4    6.5    0.4    0.1    0.0    0.0    0.0    3.1    6.0   15.5
2009   15.3   12.3    6.8    1.5    0.4    0.1    0.0    0.0    0.0    0.6    2.4   17.5
2010   19.0   17.6   11.4    4.3    2.6    0.0    0.0    0.0    0.3    2.5   11.6   24.0
2011   15.1    5.7    8.8    0.5    0.8    0.1    0.0    0.0    0.0    0.4    1.5    8.4
2012    9.9   11.5    3.1    6.4    2.0    0.2    0.0    0.0    0.3    3.1    6.4   11.8
2013   14.2   15.1   19.2    9.3    1.3    0.0    0.0    0.0    0.1    0.3    7.4    4.3
2014    6.0    4.0    5.3    1.3    0.6    0.0    0.0    0.0    0.0    0.3    2.5   11.7
2015   12.8   11.7    7.5    3.9    1.5    0.2    0.0    0.0    0.0    0.4    3.2    2.8
2016    9.8   13.1    8.1    6.6    0.6    0.0    0.0    0.0    0.0    0.5    8.7    6.5
2017   14.0    6.4    4.3    3.0    0.5    0.1    0.0    0.0    0.0    0.6    6.9   12.8
2018    9.8   16.6   12.7    2.6    0.6    0.0    0.0    0.0    0.1    3.2    2.9    5.5
2019   14.2    7.4    3.5    4.4    1.8    0.0    0.0    0.0    0.0    2.2    7.4    7.2
2020    5.3    6.3    7.8    2.9    2.4    0.0    0.0    0.0    0.4    0.2    2.9    9.3
2021   18.5   10.8    5.1   13.8    2.8    0.1    0.0    0.0    0.0    0.3    4.5    7.1
2022   10.2    5.3    6.5    5.8    0.0    0.0    0.0    0.0    0.0    0.1    1.6   14.4
2023   10.7    7.3    8.4    3.8    0.1    0.0    0.0    0.0    0.0    1.0    6.1    7.6
2024   11.8    5.2    4.3    2.7    0.0    0.0    0.0    0.0    0.2    1.0    7.4    3.5
"""

# Parse data
import io

tmax_df = pd.read_csv(io.StringIO(tmax_text), sep=r'\s+')
tmin_df = pd.read_csv(io.StringIO(tmin_text), sep=r'\s+')
rain_df = pd.read_csv(io.StringIO(rain_text), sep=r'\s+')
sun_df = pd.read_csv(io.StringIO(sun_text), sep=r'\s+')
frost_df = pd.read_csv(io.StringIO(frost_text), sep=r'\s+')

print("\n✓ Monthly data loaded successfully!")
print(f"  Years: {tmax_df['year'].min()} - {tmax_df['year'].max()}")

# ============================================================================
# STEP 2: CREATE SEASONAL FEATURES
# ============================================================================

print("\n" + "="*70)
print("CREATING SEASONAL FEATURES")
print("="*70)

months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

seasonal_features = pd.DataFrame()
seasonal_features['Year'] = tmax_df['year']

# TEMPERATURE FEATURES
print("\n1. Temperature Features:")

# Seasonal means
seasonal_features['Winter_Tmax'] = tmax_df[['dec', 'jan', 'feb']].mean(axis=1)
seasonal_features['Spring_Tmax'] = tmax_df[['mar', 'apr', 'may']].mean(axis=1)
seasonal_features['Summer_Tmax'] = tmax_df[['jun', 'jul', 'aug']].mean(axis=1)
seasonal_features['Autumn_Tmax'] = tmax_df[['sep', 'oct', 'nov']].mean(axis=1)

seasonal_features['Winter_Tmin'] = tmin_df[['dec', 'jan', 'feb']].mean(axis=1)
seasonal_features['Spring_Tmin'] = tmin_df[['mar', 'apr', 'may']].mean(axis=1)
seasonal_features['Summer_Tmin'] = tmin_df[['jun', 'jul', 'aug']].mean(axis=1)
seasonal_features['Autumn_Tmin'] = tmin_df[['sep', 'oct', 'nov']].mean(axis=1)

# Mean temperature (for completeness)
seasonal_features['Spring_Temp_Mean'] = (seasonal_features['Spring_Tmax'] + seasonal_features['Spring_Tmin']) / 2
seasonal_features['Summer_Temp_Mean'] = (seasonal_features['Summer_Tmax'] + seasonal_features['Summer_Tmin']) / 2

# Critical period temperatures
seasonal_features['Flowering_Temp'] = tmax_df[['may', 'jun']].mean(axis=1)  # Flowering period
seasonal_features['Grain_Filling_Temp'] = tmax_df[['jun', 'jul']].mean(axis=1)  # Grain filling

# Extreme temperatures
seasonal_features['Summer_Tmax_Peak'] = tmax_df[['jun', 'jul', 'aug']].max(axis=1)
seasonal_features['Spring_Tmin_Coldest'] = tmin_df[['mar', 'apr', 'may']].min(axis=1)

# Diurnal range (stress indicator)
seasonal_features['Spring_Temp_Range'] = seasonal_features['Spring_Tmax'] - seasonal_features['Spring_Tmin']
seasonal_features['Summer_Temp_Range'] = seasonal_features['Summer_Tmax'] - seasonal_features['Summer_Tmin']

# Growing Degree Days (GDD) - simplified (base 0°C for UK crops)
for season, cols in [('Spring', ['mar', 'apr', 'may']), ('Summer', ['jun', 'jul', 'aug'])]:
    gdd = 0
    for month in cols:
        tmax_month = tmax_df[month]
        tmin_month = tmin_df[month]
        days_in_month = 30  # Simplified
        monthly_gdd = days_in_month * np.maximum(0, (tmax_month + tmin_month) / 2 - 0)  # Base 0°C
        gdd += monthly_gdd
    seasonal_features[f'{season}_GDD'] = gdd

print("  ✓ Created 18 temperature features")

# RAINFALL FEATURES
print("\n2. Rainfall Features:")

# Seasonal totals
seasonal_features['Winter_Rain'] = rain_df[['dec', 'jan', 'feb']].sum(axis=1)
seasonal_features['Spring_Rain'] = rain_df[['mar', 'apr', 'may']].sum(axis=1)
seasonal_features['Summer_Rain'] = rain_df[['jun', 'jul', 'aug']].sum(axis=1)
seasonal_features['Autumn_Rain'] = rain_df[['sep', 'oct', 'nov']].sum(axis=1)

# Critical periods
seasonal_features['Planting_Rain'] = rain_df[['mar', 'apr']].sum(axis=1)  # Spring planting
seasonal_features['Grain_Filling_Rain'] = rain_df[['jun', 'jul']].sum(axis=1)  # Grain filling
seasonal_features['Harvest_Rain'] = rain_df[['jul', 'aug']].sum(axis=1)  # Harvest period

# Rainfall deviation from optimal (from Day 1 analysis)
optimal_rainfall = 1180  # Average optimal from Day 1
seasonal_features['Annual_Rain'] = rain_df[months].sum(axis=1)
seasonal_features['Rain_Deviation_from_Optimal'] = np.abs(seasonal_features['Annual_Rain'] - optimal_rainfall)

# Squared terms for inverted-U relationship
seasonal_features['Spring_Rain_Squared'] = seasonal_features['Spring_Rain'] ** 2
seasonal_features['Summer_Rain_Squared'] = seasonal_features['Summer_Rain'] ** 2

print("  ✓ Created 11 rainfall features")

# SUNSHINE FEATURES
print("\n3. Sunshine Features:")

# Seasonal totals
seasonal_features['Winter_Sun'] = sun_df[['dec', 'jan', 'feb']].sum(axis=1)
seasonal_features['Spring_Sun'] = sun_df[['mar', 'apr', 'may']].sum(axis=1)
seasonal_features['Summer_Sun'] = sun_df[['jun', 'jul', 'aug']].sum(axis=1)
seasonal_features['Autumn_Sun'] = sun_df[['sep', 'oct', 'nov']].sum(axis=1)

# Critical periods (photosynthesis during grain filling)
seasonal_features['Grain_Filling_Sun'] = sun_df[['jun', 'jul']].sum(axis=1)
seasonal_features['Flowering_Sun'] = sun_df[['may', 'jun']].sum(axis=1)

# Sunshine efficiency (sunshine per mm of rain)
seasonal_features['Summer_Sun_per_Rain'] = seasonal_features['Summer_Sun'] / (seasonal_features['Summer_Rain'] + 1)  # +1 to avoid division by zero

print("  ✓ Created 7 sunshine features")

# FROST FEATURES
print("\n4. Frost Features:")

# Seasonal totals
seasonal_features['Winter_Frost'] = frost_df[['dec', 'jan', 'feb']].sum(axis=1)
seasonal_features['Spring_Frost'] = frost_df[['mar', 'apr', 'may']].sum(axis=1)
seasonal_features['Autumn_Frost'] = frost_df[['sep', 'oct', 'nov']].sum(axis=1)

# Critical spring frost (damages seedlings)
seasonal_features['Late_Spring_Frost'] = frost_df[['apr', 'may']].sum(axis=1)

print("  ✓ Created 4 frost features")

# INTERACTION FEATURES
print("\n5. Interaction Features:")

# Temperature × Rainfall (drought vs waterlogging)
seasonal_features['Spring_Temp_x_Rain'] = seasonal_features['Spring_Temp_Mean'] * seasonal_features['Spring_Rain']
seasonal_features['Summer_Temp_x_Rain'] = seasonal_features['Summer_Temp_Mean'] * seasonal_features['Summer_Rain']

# Temperature × Sunshine (heat + light synergy)
seasonal_features['Summer_Temp_x_Sun'] = seasonal_features['Summer_Tmax'] * seasonal_features['Summer_Sun']

# Extreme event indicators
seasonal_features['Heat_Stress'] = (seasonal_features['Summer_Tmax_Peak'] > 21).astype(int)  # >21°C is high for UK
seasonal_features['Cold_Spring'] = (seasonal_features['Spring_Tmin_Coldest'] < 2).astype(int)  # Cold spring
seasonal_features['Extreme_Summer_Rain'] = (seasonal_features['Summer_Rain'] > 350).astype(int)  # Wet summer (like 2012)
seasonal_features['Drought_Spring'] = (seasonal_features['Spring_Rain'] < 180).astype(int)  # Dry spring

print("  ✓ Created 7 interaction/extreme features")

# ============================================================================
# STEP 3: SAVE AND SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\nTotal Seasonal Features Created: {len(seasonal_features.columns) - 1}")  # -1 for Year column
print(f"Years covered: {seasonal_features['Year'].min()} - {seasonal_features['Year'].max()}")

print("\nFeature Categories:")
print(f"  • Temperature: 18 features")
print(f"  • Rainfall: 11 features")
print(f"  • Sunshine: 7 features")
print(f"  • Frost: 4 features")
print(f"  • Interactions/Extremes: 7 features")
print(f"  • TOTAL: 47 features")

# Show sample
print("\nSample of Seasonal Features (first 5 years):")
print(seasonal_features.head())

# Save
seasonal_features.to_csv('data/seasonal_weather_features_2004_2024.csv', index=False)
print("\n✓ Saved: data/seasonal_weather_features_2004_2024.csv")

# ============================================================================
# STEP 4: CHECK CORRELATIONS WITH YIELDS
# ============================================================================

print("\n" + "="*70)
print("CORRELATION CHECK: Do Seasonal Features Predict Better?")
print("="*70)

# Load original dataset
original_df = pd.read_csv('data/uk_crop_yield_complete_2004_2024.csv')

# Merge seasonal features
df_with_seasonal = original_df.merge(seasonal_features, on='Year', how='left')

print("\nOriginal Annual Weather Correlations with Yield:")
print("-"*70)
annual_corrs = df_with_seasonal[['Yield_t_per_ha', 'Annual_Temp_C', 'Annual_Rainfall_mm', 
                                  'Annual_Sunshine_hours', 'Annual_Frost_days']].corr()['Yield_t_per_ha'].drop('Yield_t_per_ha')
print(annual_corrs.round(3))

print("\n🔥 NEW Seasonal Feature Correlations with Yield:")
print("-"*70)

# Calculate correlations for seasonal features
seasonal_cols = [col for col in seasonal_features.columns if col != 'Year']
seasonal_corrs = df_with_seasonal[['Yield_t_per_ha'] + seasonal_cols].corr()['Yield_t_per_ha'].drop('Yield_t_per_ha')

# Show top 15 positive and negative
print("\nTop 15 POSITIVE correlations:")
print(seasonal_corrs.nlargest(15))

print("\nTop 15 NEGATIVE correlations:")
print(seasonal_corrs.nsmallest(15))

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

# Compare best seasonal vs best annual
best_annual = annual_corrs.abs().max()
best_seasonal = seasonal_corrs.abs().max()

print(f"\nBest Annual Feature Correlation: {best_annual:.3f}")
print(f"Best Seasonal Feature Correlation: {best_seasonal:.3f}")
print(f"Improvement Factor: {best_seasonal / best_annual:.1f}x")

if best_seasonal > 0.3:
    print("\n✅ SUCCESS! Seasonal features show STRONG correlations!")
    print("   → Weather IS predictive with proper temporal resolution")
    print("   → Ready to build models with these features")
elif best_seasonal > 0.15:
    print("\n✓ IMPROVEMENT! Seasonal features are better than annual")
    print("   → Weather shows moderate predictive signal")
    print("   → Crop-specific models may improve further")
else:
    print("\n⚠️ LIMITED IMPROVEMENT. Possible reasons:")
    print("   1. Need crop-specific seasonal features")
    print("   2. Regional analysis required (national still too coarse)")
    print("   3. Non-linear relationships not captured by correlation")

# ============================================================================
# STEP 5: CREATE FINAL MERGED DATASET
# ============================================================================

print("\n" + "="*70)
print("CREATING FINAL DATASET WITH SEASONAL FEATURES")
print("="*70)

# Merge everything
final_df = original_df.merge(seasonal_features, on='Year', how='left')

print(f"\nFinal Dataset Shape: {final_df.shape}")
print(f"Total Variables: {final_df.shape[1]}")
print(f"  - Original: 13")
print(f"  - Seasonal Weather: 47")
print(f"  - Total: {final_df.shape[1]}")

# Show columns
print("\nColumn List:")
print(final_df.columns.tolist())

# Check for missing values
print("\nMissing Values Check:")
missing = final_df.isnull().sum()
if missing.sum() == 0:
    print("  ✓ No missing values!")
else:
    print(missing[missing > 0])

# Save final dataset
final_df.to_csv('data/uk_crop_yield_with_seasonal_features_2004_2024.csv', index=False)
print("\n✓ Saved: data/uk_crop_yield_with_seasonal_features_2004_2024.csv")

# ============================================================================
# STEP 6: SPECIFIC CROP ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("CROP-SPECIFIC SEASONAL CORRELATIONS")
print("="*70)

crops = final_df['Crop'].unique()

for crop in crops:
    crop_data = final_df[final_df['Crop'] == crop]
    
    # Calculate correlations for this crop
    crop_seasonal_corrs = crop_data[['Yield_t_per_ha'] + seasonal_cols].corr()['Yield_t_per_ha'].drop('Yield_t_per_ha')
    
    print(f"\n{crop}:")
    print(f"  Top 3 Positive:")
    for idx, (feature, corr) in enumerate(crop_seasonal_corrs.nlargest(3).items(), 1):
        print(f"    {idx}. {feature}: r={corr:.3f}")
    
    print(f"  Top 3 Negative:")
    for idx, (feature, corr) in enumerate(crop_seasonal_corrs.nsmallest(3).items(), 1):
        print(f"    {idx}. {feature}: r={corr:.3f}")

print("\n" + "="*70)
print("🎉 SEASONAL FEATURE ENGINEERING COMPLETE!")
print("="*70)

print("\nWhat We've Created:")
print("  ✓ 47 seasonal weather features")
print("  ✓ Temperature: means, extremes, GDD, ranges")
print("  ✓ Rainfall: seasonal totals, critical periods, squared terms")
print("  ✓ Sunshine: seasonal totals, critical periods")
print("  ✓ Frost: seasonal counts, late spring frost")
print("  ✓ Interactions: temp×rain, temp×sun, extreme events")
print("  ✓ Crop-specific correlations calculated")

print("\nNext Steps:")
print("  1. Review correlation results above")
print("  2. Select top 10-15 features for modeling")
print("  3. Build crop-specific models (Day 2)")
print("  4. Compare: Annual vs Seasonal feature performance")

print("\nExpected Results:")
print("  • Annual features: R² ~ 0.3-0.4")
print("  • Seasonal features: R² ~ 0.5-0.7 (or better!)")
print("  • This validates that temporal resolution matters!")

print("\n" + "="*70)