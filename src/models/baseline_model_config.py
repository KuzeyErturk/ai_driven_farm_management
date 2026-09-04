# Parameters of baseline model - Random Forest
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 8,
    'min_samples_split': 3,
    'random_state': 42
}

# Train/Test split
TRAIN_YEARS = list(range(2004, 2020))  # 2004-2019
TEST_YEARS = list(range(2020, 2025))   # 2020-2024

# Feature configurations per crop
CROP_FEATURES = {
    'Wheat': [
        'Area_hectares',
        'Winter_Sun',
        'Summer_Sun',
        'Extreme_Summer_Rain',
        'Grain_Filling_Rain'
    ],
    'Winter_Barley': [
        'Area_hectares',
        'Spring_Tmax',
        'Winter_Sun',
        'Planting_Rain',
        'Winter_Frost'
    ],
    'Spring_Barley': [
        'Area_hectares',
        'Spring_Rain',
        'Spring_Tmax',
        'Summer_Sun'
    ],
    'Oats': [
        'Area_hectares',
        'Spring_Rain',
        'Rain_Deviation_from_Optimal',
        'Flowering_Sun'
    ],
    'OSR': [
        'Area_hectares',
        'Summer_Tmin',
        'Planting_Rain',
        'Winter_Frost'
    ]
}

# Data sources
DATA_SOURCES = {
    'Wheat': 'data/processed/regional_crop_yield_weather_2004_2024.csv',
    'Winter_Barley': 'data/processed/winter_barley_with_weather.csv',
    'Spring_Barley': 'data/processed/spring_barley_with_weather.csv',
    'Oats': 'data/processed/regional_crop_yield_weather_2004_2024.csv',
    'OSR': 'data/processed/regional_crop_yield_weather_2004_2024.csv',
}

# Baseline R² scores (RF, max_depth=8, test split 2020-2024)
BASELINE_RESULTS = {
    'Wheat':         {'test_r2': 0.485, 'loocv_r2': 0.355, 'test_rmse': 0.598},
    'Winter_Barley': {'test_r2': 0.408, 'loocv_r2': 0.231, 'test_rmse': 0.491},
    'Spring_Barley': {'test_r2': -0.153, 'loocv_r2': 0.129, 'test_rmse': 0.692},
    'Oats':          {'test_r2': 0.188, 'loocv_r2': 0.194, 'test_rmse': 0.609},
    'OSR':           {'test_r2': 0.091, 'loocv_r2': 0.041, 'test_rmse': 0.524},
}

# Some improvements in some crops
IMPROVED_RESULTS = {
    'Wheat':         {'model': 'RandomForest(max_depth=8)',  'test_r2': 0.485, 'loocv_r2': 0.355, 'test_rmse': 0.598},
    'Winter_Barley': {'model': 'RandomForest(max_depth=5)',  'test_r2': 0.418, 'loocv_r2': 0.236, 'test_rmse': 0.487},
    'Spring_Barley': {'model': 'Ridge(alpha=10)',            'test_r2': -0.186, 'loocv_r2': 0.221, 'test_rmse': 0.702},
    'Oats':          {'model': 'RandomForest(max_depth=3)',  'test_r2': 0.188, 'loocv_r2': 0.247, 'test_rmse': 0.609},
    'OSR':           {'model': 'SVR(kernel=linear)',         'test_r2': 0.064, 'loocv_r2': 0.196, 'test_rmse': 0.531},
}

CROP_NAME_IN_DATA = {
    'Wheat': 'Wheat',
    'Oats': 'Oats',
    'OSR': 'Oilseed_Rape',
}

