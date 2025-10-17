"""
Configuration file for agricultural yield forecasting project.
"""

import os
from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# Regions (Alberta)
REGIONS = {
    'red_deer': {
        'name': 'Red Deer',
        'station_id': '2803',
        'lat': 52.27,
        'lon': -113.81,
        'soil_zone': 'Black'
    },
    'lethbridge': {
        'name': 'Lethbridge',
        'station_id': '3035',
        'lat': 49.69,
        'lon': -112.84,
        'soil_zone': 'Brown'
    },
    'peace_river': {
        'name': 'Peace River',
        'station_id': '50117',
        'lat': 56.23,
        'lon': -117.45,
        'soil_zone': 'Gray'
    },
    'drumheller': {
        'name': 'Drumheller',
        'station_id': '2279',
        'lat': 51.46,
        'lon': -112.71,
        'soil_zone': 'Brown'
    },
    'calgary': {
        'name': 'Calgary',
        'station_id': '2205',
        'lat': 51.05,
        'lon': -114.07,
        'soil_zone': 'Dark Brown'
    }
}

# Date range for historical data
START_YEAR = 2014
END_YEAR = 2024
CURRENT_DATE = datetime.now()

# Crop types
CROPS = ['wheat', 'canola', 'barley']

# Growing season (Alberta)
GROWING_SEASON_START = (5, 1)   # May 1
GROWING_SEASON_END = (9, 30)    # September 30

# Feature engineering parameters
GDD_BASE_TEMPS = [5, 10]  # Base temperatures for GDD calculation
HEAT_STRESS_THRESHOLD = 30  # °C
FROST_THRESHOLD = -2  # °C

# Model parameters
RANDOM_FOREST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 10,
    'random_state': 42
}

XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.05,
    'random_state': 42
}

LSTM_PARAMS = {
    'sequence_length': 30,
    'units': 64,
    'dropout': 0.2,
    'epochs': 50,
    'batch_size': 32
}

# Ensemble weights (optimize these later)
ENSEMBLE_WEIGHTS = {
    'random_forest': 0.35,
    'xgboost': 0.40,
    'lstm': 0.25
}

# Target metrics
TARGET_ACCURACY = 0.79

print(f"✅ Config loaded - Project root: {PROJECT_ROOT}")