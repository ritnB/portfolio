# config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Environment type: 'local' or 'production'
ENV_TYPE = os.getenv("ENV_TYPE", "local").lower()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# GCS configuration
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "your-model-bucket")
GCS_MODEL_DIR = "models"

# Evaluation criteria settings (configurable via environment)
PREDICTION_EVAL_DAYS = int(os.getenv("PREDICTION_EVAL_DAYS", "7"))
THRESHOLD_ACCURACY = float(os.getenv("THRESHOLD_ACCURACY", "0.50"))  # Default threshold
RECENT_DAYS = int(os.getenv("RECENT_DAYS", "14"))  # Number of past days to use for prediction

TECH_INDICATORS_TABLE = "technical_indicators"

# Feature columns for model training/inference (configurable)
FEATURE_COLS = [
    'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5',
    'feature_6', 'feature_7', 'feature_8'  # Abstracted feature names
]

# Verification settings
VERIFY_MAX_RECORDS = 5000  # Maximum number of records to check during verification
VERIFY_BATCH_SIZE = 1000   # Batch size for pagination

# Cryptocurrency mapping (sample - actual mapping is proprietary)
COIN_NAME_MAPPING = {
    'sample-coin-1': 'Sample Coin 1',
    'sample-coin-2': 'Sample Coin 2',
    # Additional mappings are proprietary
}

def normalize_coin_name(coin_name):
    """
    Normalize coin name to standardized format.
    Returns original name if not found in mapping.
    Note: Full mapping logic is proprietary.
    """
    if coin_name is None:
        return None
    
    # Basic normalization (proprietary logic abstracted)
    return COIN_NAME_MAPPING.get(coin_name, coin_name)

# Model and scaler path configuration
LOCAL_MODEL_PATH = "models/model_latest.pt"
LOCAL_SCALER_PATH = "models/scaler_latest.pkl"

GCS_MODEL_PATH = "/tmp/latest_model.pt"
GCS_SCALER_PATH = "/tmp/latest_scaler.pkl"

# Use GCS paths unconditionally in production environment
if ENV_TYPE == "production":
    MODEL_PATH = GCS_MODEL_PATH
    SCALER_PATH = GCS_SCALER_PATH
else:
    MODEL_PATH = LOCAL_MODEL_PATH
    SCALER_PATH = LOCAL_SCALER_PATH