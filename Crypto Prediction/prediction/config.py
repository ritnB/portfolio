# config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Environment type: 'local' or 'production'
ENV_TYPE = os.getenv("ENV_TYPE", "local").lower()

# Supabase settings
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# GCS settings (anonymized)
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "your-bucket-name")
GCS_MODEL_DIR = "models"

# Evaluation criteria settings (exponential value examples)
PREDICTION_EVAL_DAYS = 1e1  # e.g. 1e1
THRESHOLD_ACCURACY = 1e-1   # e.g. 1e-1
RECENT_DAYS_RETRAIN = 1e2   # e.g. 1e2
RECENT_DAYS_INFERENCE = 1e0 # e.g. 1e0

# Model training data split settings (exponential value examples)
TEST_DAYS = 1e1        # e.g. 1e1
VALIDATION_DAYS = 1e1  # e.g. 1e1

# Rolling Window CV parameters (exponential value examples)
ROLLING_CV_TRAIN_WINDOW = 1e2
ROLLING_CV_VAL_SIZE = 1e2
ROLLING_CV_N_SPLITS = 1e0
ROLLING_CV_GAP_SIZE = 1e1
ROLLING_CV_STEP_SIZE = 1e1

# Incremental learning settings (exponential value examples)
INCREMENTAL_DAYS = 1e0
INCREMENTAL_EPOCHS = 1e0
INCREMENTAL_LR = 1e-5
INCREMENTAL_BATCH_SIZE = 1e1
INCREMENTAL_PERFORMANCE_THRESHOLD = 1e-1

# Classification Threshold settings (exponential value examples)
DEFAULT_CLASSIFICATION_THRESHOLD = 1e-1
THRESHOLD_RANGE_MIN = 1e-1
THRESHOLD_RANGE_MAX = 1e0

TECH_INDICATORS_TABLE = "technical_indicators"

# Model training/inference feature columns (anonymized)
FEATURE_COLS = [
    'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5',
    'feature_6', 'feature_7', 'feature_8'
]

# Verification settings (exponential value examples)
VERIFY_MAX_RECORDS = 1e3
VERIFY_BATCH_SIZE = 1e2

# Memory optimization settings (exponential value examples)
MEMORY_THRESHOLD_MB = 1e3
BATCH_PROCESSING_SIZE = 1e2
SEQUENCE_CACHE_ENABLED = True
MEMORY_CHECK_INTERVAL = 1e1

# Sequence processing settings (exponential value examples)
MAX_TIME_GAP_HOURS = 1e1
SEQUENCE_BATCH_SIZE = 1e2

# Asset information (anonymized)
ASSET_INFO = {
    'asset_1': {'name': 'Asset 1', 'external_id': 'asset-1'},
    'asset_2': {'name': 'Asset 2', 'external_id': 'asset-2'},
    'asset_3': {'name': 'Asset 3', 'external_id': 'asset-3'},
    'asset_4': {'name': 'Asset 4', 'external_id': 'asset-4'},
    'asset_5': {'name': 'Asset 5', 'external_id': 'asset-5'}
}

# Model file path settings (anonymized)
MODEL_PATH = "models/anonymized_model.pt"
SCALER_PATH = "models/anonymized_scaler.pkl"

# External API settings (exponential value examples)
EXTERNAL_API_BASE_URL = "https://api.example.com/api/v3"
EXTERNAL_API_PRICE_ENDPOINT = "/simple/price"
EXTERNAL_API_REQUEST_TIMEOUT = 1e1

# Price information column names (anonymized)
PRICE_USD_COLUMN = "price_usd"
PRICE_LOCAL_COLUMN = "price_local"

# Compatibility mapping dictionaries (anonymized)
ASSET_NAME_MAPPING = {key: info['name'] for key, info in ASSET_INFO.items()}
EXTERNAL_ASSET_MAPPING = {info['name']: info['external_id'] for info in ASSET_INFO.values()}

# Optuna hyperparameter tuning space (exponential value examples)
OPTUNA_PARAM_SPACE = {
    # Model structure parameters
    "patch_len": [1e1],
    "window_size": [1e2],
    "stride": [1e0],
    "d_model": [1e2],
    "mlp_hidden_mult": [1e0],
    "activation": ["relu", "gelu"],
    "pooling_type": ["cls", "mean"],
    
    # Training parameters
    "dropout_mlp": {"min": 1e-1, "max": 1e-1},
    "learning_rate": {"min": 1e-5, "max": 1e-3, "log": True},
    "weight_decay": {"min": 1e-2, "max": 1e-1},
    "classification_threshold": {"min": 1e-1, "max": 1e0},
    
    # Loss function parameters
    "loss_type": ["ce", "focal"],
    "focal_gamma": {"min": 1e0, "max": 1e1},
    
    # Fixed values
    "num_layers": 1e0,
    "num_heads": 1e0,
    "batch_size": 1e2
}

# Asset name normalization function
def normalize_asset_name(asset_name):
    """Normalize asset names."""
    return ASSET_NAME_MAPPING.get(asset_name.lower(), asset_name)