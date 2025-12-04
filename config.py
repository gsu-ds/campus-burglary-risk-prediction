from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent

# Data Directory
DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
RAW_APD_DIR = RAW_DIR / "apd"
RAW_SHP_DIR = RAW_DIR / "shapefiles"

INTERIM_DIR = DATA_DIR / "interim" / "apd"
PROCESSED_DIR = DATA_DIR / "processed" / "apd"
EXTERNAL_DIR = DATA_DIR / "external"

# Data Panel Paths
DATA_SPARSE = PROCESSED_DIR / "npu_sparse_panel.parquet"
DATA_DENSE  = PROCESSED_DIR / "npu_dense_panel.parquet"
DATA_TARGET = PROCESSED_DIR / "target_crimes.parquet"

# Weather Data Paths
HOURLY_WEATHER_CSV = EXTERNAL_DIR / "atlanta_hourly_weather_2021_to_current.csv"
DAILY_WEATHER_CSV = EXTERNAL_DIR / "atlanta_daily_weather_2021_to_current.csv"

HOURLY_WEATHER_PARQUET = EXTERNAL_DIR / "atlanta_hourly_weather_2021_to_current.parquet"
DAILY_WEATHER_PARQUET = EXTERNAL_DIR / "atlanta_daily_weather_2021_to_current.parquet"

# Shapefiles
APD_ZONE_SHP = RAW_SHP_DIR / "apd_zone_2019_sf" / "apd_police_zones_2019.shp"
CAMPUS_SHP = RAW_SHP_DIR / "area_landmark_2024_sf" / "ga_census_landmarks_2023.shp"
NEIGHBORHOOD_SHP = RAW_SHP_DIR / "atl_neighborhood_sf" / "atl_neighborhoods.shp"
NPU_SHP = RAW_SHP_DIR / "atl_npu_sf" / "atl_npu_boundaries.shp"
CITIES_SHP = RAW_SHP_DIR / "census_boundary_2024_sf" / "ga_census_places_2024.shp"

# Artifacts
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
WANDB_OUTPUT_DIR = ARTIFACTS_DIR / "wandb"
CATBOOST_OUTPUT_DIR = ARTIFACTS_DIR / "catboost_info"
ASSORTED_ARTIFACTS = ARTIFACTS_DIR / "assorted"

# Reports
REPORTS_DIR = PROJECT_ROOT / "reports"

TEST_RESULTS_DIR = REPORTS_DIR / "test_results"
CARDS_DIR = REPORTS_DIR / "cards"

FIGURES_DIR = REPORTS_DIR / "figures"
LEADERBOARD_DIR = FIGURES_DIR / "leaderboard"

# Campus Centers (lat, lon)
SCHOOL_CENTERS = {
    "GSU": (33.7530, -84.3863),
    "GA_Tech": (33.7756, -84.3963),
    "Emory": (33.7925, -84.3239),
    "Clark": (33.7533, -84.4124),
    "Spelman": (33.7460, -84.4129),
    "Morehouse": (33.7483, -84.4126),
    "Morehouse_Med": (33.7505, -84.4131),
    "Atlanta_Metro": (33.7145, -84.4020),
    "Atlanta_Tech": (33.7126, -84.4034),
    "SCAD": (33.7997, -84.3920),
    "John_Marshall": (33.7621, -84.3896),
}

# Campus Label Encoding
CAMPUS_ENCODING = {
    "none": 0,
    "GSU": 1,
    "GA_Tech": 2,
    "Emory": 3,
    "Clark": 4,
    "Spelman": 5,
    "Morehouse": 6,
    "Morehouse_Med": 7,
    "Atlanta_Metro": 8,
    "Atlanta_Tech": 9,
    "SCAD": 10,
    "John_Marshall": 11,
}

# Campus Distance Threshold
CAMPUS_DISTANCE_THRESHOLD_M = 2414.016  # ~1.5 miles

# Create folders if missing
INTERIM_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CARDS_DIR.mkdir(parents=True, exist_ok=True)
LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)   
WANDB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CATBOOST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ASSORTED_ARTIFACTS.mkdir(parents=True, exist_ok=True)