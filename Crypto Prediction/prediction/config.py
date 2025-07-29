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
GCS_BUCKET_NAME = "your-bucket-name"  # Placeholder for public portfolio
GCS_MODEL_DIR = "models"

# Evaluation criteria settings
PREDICTION_EVAL_DAYS = 7
THRESHOLD_ACCURACY = 0.30
RECENT_DAYS = 50  # Number of past days to use for prediction

# Model training data split settings
TEST_DAYS = 10        # Recent days for testing (complete holdout)
VALIDATION_DAYS = 10  # Days for validation (Optuna optimization)

# Rolling Window CV parameters (anonymized for portfolio)
ROLLING_CV_TRAIN_WINDOW = 350  # Training window size
ROLLING_CV_VAL_SIZE = 150      # Validation size  
ROLLING_CV_N_SPLITS = 3        # Number of CV splits
ROLLING_CV_GAP_SIZE = 24       # Gap between train and validation
ROLLING_CV_STEP_SIZE = 80      # Step size between folds

# Incremental learning settings (anonymized for portfolio)
INCREMENTAL_DAYS = 7           # Recent days for incremental learning
INCREMENTAL_EPOCHS = 3         # Epochs for incremental learning
INCREMENTAL_LR = 1e-5          # Learning rate for incremental learning
INCREMENTAL_BATCH_SIZE = 16    # Batch size for incremental learning
INCREMENTAL_PERFORMANCE_THRESHOLD = 0.25  # Performance threshold (anonymized)
INCREMENTAL_VALIDATION_RATIO = 0.2        # Validation data ratio (anonymized)
INCREMENTAL_MAX_DEGRADATION = 0.1         # Max performance degradation (anonymized)

TECH_INDICATORS_TABLE = "technical_indicators"

# Feature columns for model training/inference (anonymized for portfolio)
FEATURE_COLS = [
    'feature1', 'feature2', 'feature3', 'feature4',
    'feature5', 'feature6', 'feature7', 'feature8'
]

# Verification settings
VERIFY_MAX_RECORDS = 5000  # Maximum records to check during verification
VERIFY_BATCH_SIZE = 1000   # Batch size for paging

# Asset name mapping dictionary (simplified for portfolio)
ASSET_NAME_MAPPING = {
    'asset1': 'Asset 1',
    'asset2': 'Asset 2',
    'asset3': 'Asset 3',
    'asset4': 'Asset 4',
    'asset5': 'Asset 5',
    'asset6': 'Asset 6',
    'asset7': 'Asset 7',
    'asset8': 'Asset 8',
    'asset9': 'Asset 9',
    'asset10': 'Asset 10'
}

def normalize_asset_name(asset_name):
    """
    Normalize asset name to standardized form.
    Returns original name if not in mapping.
    """
    if asset_name is None:
        return None
    
    # Try exact matching first
    if asset_name in ASSET_NAME_MAPPING:
        return ASSET_NAME_MAPPING[asset_name]
    
    # Try case-insensitive matching
    for key, value in ASSET_NAME_MAPPING.items():
        if key.lower() == asset_name.lower():
            return value
    
    # Return original if not in mapping
    return asset_name

# Model and scaler path branching
LOCAL_MODEL_PATH = "models/patchtst_final_model_20250719.pt"
LOCAL_SCALER_PATH = "models/scaler_standard_20250719.pkl"

GCS_MODEL_PATH = "/tmp/latest_model.pt"
GCS_SCALER_PATH = "/tmp/latest_scaler.pkl"

# Use GCS paths in production environment
if ENV_TYPE == "production":
    MODEL_PATH = GCS_MODEL_PATH
    SCALER_PATH = GCS_SCALER_PATH
else:
    MODEL_PATH = LOCAL_MODEL_PATH
    SCALER_PATH = LOCAL_SCALER_PATH