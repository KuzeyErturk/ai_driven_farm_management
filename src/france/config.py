"""
Cross-Country Comparison Configuration
========================================
Maps French départements and German districts to macro-regions,
defines crop name mappings, and provides shared constants.

Data sources:
- France: Schauberger et al. (2022), DOI: 10.5880/PIK.2021.001
- Germany: OpenAgrar district-level dataset (2024)
- UK: Project's own regional dataset
"""

import os

# ---------------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

PATHS = {
    'france_raw': os.path.join(PROJECT_ROOT, 'data', 'france', 'raw'),
    'france_processed': os.path.join(PROJECT_ROOT, 'data', 'france', 'processed'),
    'germany_processed': os.path.join(PROJECT_ROOT, 'data', 'germany', 'processed'),
    'uk_processed': os.path.join(PROJECT_ROOT, 'data', 'processed'),
    'pooled': os.path.join(PROJECT_ROOT, 'data', 'pooled'),
    'plots': os.path.join(PROJECT_ROOT, 'plots'),
    'models': os.path.join(PROJECT_ROOT, 'src', 'models'),
}

# ---------------------------------------------------------------------------
# SHARED CONSTANTS
# ---------------------------------------------------------------------------

CROPS = ['Wheat', 'Winter_Barley', 'Spring_Barley', 'Oats', 'Oilseed_Rape']

YEAR_START = 2004
YEAR_END = 2016  # Common overlap: France ends 2016 in GDHY, Germany goes to 2021

# Optimal annual rainfall (mm) — for Rain_Deviation_from_Optimal
UK_OPTIMAL_RAINFALL = 1180
FRANCE_OPTIMAL_RAINFALL = 750
GERMANY_OPTIMAL_RAINFALL = 700

# ---------------------------------------------------------------------------
# FRANCE: Schauberger et al. (2022) département → 4 macro-regions
# ---------------------------------------------------------------------------
# Nord  = Northern cereal belt (Picardie, Nord-PdC, Ile-de-France, Haute-Normandie)
# Ouest = Atlantic/oceanic (Bretagne, Pays de la Loire, Basse-Normandie, Poitou-Charentes)
# Est   = Continental (Champagne-Ardenne, Lorraine, Alsace, Bourgogne, Franche-Comté, Centre)
# Sud   = Southern (Aquitaine, Midi-Pyrénées, Languedoc-Roussillon, PACA, Rhône-Alpes, Auvergne)

FRANCE_REGIONS = ['Nord', 'Ouest', 'Est', 'Sud']

# Mapping uses département NAMES as they appear in Schauberger dataset
FRANCE_DEPT_TO_REGION = {
    # --- Nord (Northern cereal belt) ---
    'Aisne': 'Nord', 'Nord': 'Nord', 'Oise': 'Nord', 'Pas_de_Calais': 'Nord', 'Somme': 'Nord',
    'Paris': 'Nord', 'Seine_et_Marne': 'Nord', 'Yvelines': 'Nord', 'Essonne': 'Nord',
    'Hauts_de_Seine': 'Nord', 'Seine_Saint_Denis': 'Nord', 'Val_de_Marne': 'Nord',
    'Val_d_Oise': 'Nord', 'Seine_SeineOise': 'Nord',
    'Eure': 'Nord', 'Seine_Maritime': 'Nord',

    # --- Ouest (Atlantic/oceanic) ---
    'Calvados': 'Ouest', 'Manche': 'Ouest', 'Orne': 'Ouest',
    'Cotes_d_Armor': 'Ouest', 'Finistere': 'Ouest', 'Ille_et_Vilaine': 'Ouest', 'Morbihan': 'Ouest',
    'Loire_Atlantique': 'Ouest', 'Maine_et_Loire': 'Ouest', 'Mayenne': 'Ouest',
    'Sarthe': 'Ouest', 'Vendee': 'Ouest',
    'Charente': 'Ouest', 'Charente_Maritime': 'Ouest', 'Deux_Sevres': 'Ouest', 'Vienne': 'Ouest',

    # --- Est (Continental) ---
    'Ardennes': 'Est', 'Aube': 'Est', 'Marne': 'Est', 'Haute_Marne': 'Est',
    'Meurthe_et_Moselle': 'Est', 'Meuse': 'Est', 'Moselle': 'Est', 'Vosges': 'Est',
    'Bas_Rhin': 'Est', 'Haut_Rhin': 'Est',
    'Cote_d_Or': 'Est', 'Nievre': 'Est', 'Saone_et_Loire': 'Est', 'Yonne': 'Est',
    'Doubs': 'Est', 'Jura': 'Est', 'Haute_Saone': 'Est', 'Territoire_de_Belfort': 'Est',
    'Cher': 'Est', 'Eure_et_Loir': 'Est', 'Indre': 'Est', 'Indre_et_Loire': 'Est',
    'Loir_et_Cher': 'Est', 'Loiret': 'Est',

    # --- Sud (Southern) ---
    'Dordogne': 'Sud', 'Gironde': 'Sud', 'Landes': 'Sud', 'Lot_et_Garonne': 'Sud',
    'Pyrenees_Atlantiques': 'Sud',
    'Ariege': 'Sud', 'Aveyron': 'Sud', 'Haute_Garonne': 'Sud', 'Gers': 'Sud', 'Lot': 'Sud',
    'Hautes_pyrenees': 'Sud', 'Tarn': 'Sud', 'Tarn_et_Garonne': 'Sud',
    'Aude': 'Sud', 'Gard': 'Sud', 'Herault': 'Sud', 'Lozere': 'Sud', 'Pyrenees_Orientales': 'Sud',
    'Alpes_de_Haute_Provence': 'Sud', 'Hautes_Alpes': 'Sud', 'Alpes_Maritimes': 'Sud',
    'Bouches_du_Rhone': 'Sud', 'Var': 'Sud', 'Vaucluse': 'Sud',
    'Ain': 'Sud', 'Ardeche': 'Sud', 'Drome': 'Sud', 'Isere': 'Sud', 'Loire': 'Sud',
    'Rhone': 'Sud', 'Savoie': 'Sud', 'Haute_Savoie': 'Sud',
    'Allier': 'Sud', 'Cantal': 'Sud', 'Haute_Loire': 'Sud', 'Puy_de_Dome': 'Sud',
    'Correze': 'Sud', 'Creuse': 'Sud', 'Haute_Vienne': 'Sud',
    # Corsica
    'Corse_du_Sud': 'Sud', 'Haute_Corse': 'Sud',
}

# Schauberger file names → project crop names
FRANCE_CROP_FILES = {
    'Wheat':         'wheat_total_data_1900-2018_FILTERED.txt',
    'Winter_Barley': 'barley_winter_data_1900-2018_FILTERED.txt',
    'Spring_Barley': 'barley_spring_data_1900-2018_FILTERED.txt',
    'Oats':          'oats_total_data_1900-2018_FILTERED.txt',
    'Oilseed_Rape':  'rape_winter_data_1900-2018_FILTERED.txt',
}

FRANCE_DATA_DIR = os.path.join(
    PATHS['france_raw'],
    'franceDOI-Schauberger',
    '2021-001_Schauberger-et-al_Data_FILTERED'
)

# ---------------------------------------------------------------------------
# GERMANY: OpenAgrar district-level dataset
# ---------------------------------------------------------------------------
# 16 Bundesländer grouped into 4 macro-regions (climate zones)
# Nord  = Northern lowland (Schleswig-Holstein, Niedersachsen, Mecklenburg-Vorpommern,
#         Hamburg, Bremen)
# West  = Western (Nordrhein-Westfalen, Rheinland-Pfalz, Saarland, Hessen)
# Ost   = Eastern (Brandenburg, Sachsen-Anhalt, Sachsen, Thüringen, Berlin)
# Sued  = Southern (Bayern, Baden-Württemberg)

GERMANY_REGIONS = ['Nord_DE', 'West_DE', 'Ost_DE', 'Sued_DE']

# NUTS ID prefix → macro-region (first 3 chars of nuts_id identify the Bundesland)
GERMANY_NUTS_TO_REGION = {
    'DEF': 'Nord_DE',  # Schleswig-Holstein
    'DE9': 'Nord_DE',  # Niedersachsen
    'DE8': 'Nord_DE',  # Mecklenburg-Vorpommern
    'DE6': 'Nord_DE',  # Hamburg
    'DE5': 'Nord_DE',  # Bremen

    'DEA': 'West_DE',  # Nordrhein-Westfalen
    'DEB': 'West_DE',  # Rheinland-Pfalz
    'DEC': 'West_DE',  # Saarland
    'DE7': 'West_DE',  # Hessen

    'DE4': 'Ost_DE',   # Brandenburg
    'DEE': 'Ost_DE',   # Sachsen-Anhalt
    'DED': 'Ost_DE',   # Sachsen
    'DEG': 'Ost_DE',   # Thüringen
    'DE3': 'Ost_DE',   # Berlin

    'DE2': 'Sued_DE',  # Bayern
    'DE1': 'Sued_DE',  # Baden-Württemberg
}

# Germany CSV crop codes → project crop names
GERMANY_CROP_MAPPING = {
    'ww':    'Wheat',
    'wb':    'Winter_Barley',
    'sb':    'Spring_Barley',
    'oats':  'Oats',
    'wrape': 'Oilseed_Rape',
}

GERMANY_CSV_PATH = os.path.join(PATHS['france_raw'], 'Germany.csv')

# ---------------------------------------------------------------------------
# ERA5 bounding boxes [North, West, South, East] per French region
# (for future use if downloading real ERA5 data)
# ---------------------------------------------------------------------------
ERA5_BBOXES = {
    'Nord':  [51.0, 0.5, 48.5, 3.5],
    'Ouest': [48.8, -5.0, 45.5, 0.5],
    'Est':   [49.5, 1.5, 46.0, 7.5],
    'Sud':   [46.0, -1.5, 42.5, 7.8],
}
