# config.py (Public-safe version)

import os
import glob

# Supabase credentials (set via environment variables)
SUPABASE_URL = os.getenv("SUPABASE_URL")        # e.g., "https://<your-project>.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY")        # e.g., "<your-secret-key>"

# Table names (used in Supabase)
COMMENTS_TABLE = "comments"
TECH_INDICATORS_TABLE = "technical_indicators"
SENTIMENT_INDICATORS_TABLE = "sentiment_indicators"
COIN_PRICE_SCORES_TABLE = "predictions"

# Sentiment model hyperparameters (intentionally masked)
LEARNING_RATES = [1e8]
BATCH_SIZES = [1e8]
NUM_EPOCHS_LIST = [1e8]
WEIGHT_DECAYS = [1e8]

# Time-series model tuning parameters (masked for public release)
TUNABLE_PARAMS = {
    "learning_rates": [1e8],
    "weight_decays": [1e8],
    "d_models": [1e8]
}

# Model paths (to be set via environment variables; not public)
SENTIMENT_MODEL_PATH = os.getenv("SENTIMENT_MODEL_PATH")
SENTIMENT_TOKENIZER_PATH = os.getenv("SENTIMENT_TOKENIZER_PATH")
TIMESERIES_MODEL_PATH = os.getenv("TIMESERIES_MODEL_PATH")

# Fixed parameters for time-series model (also masked)
FIXED_PARAMS = {
    "batch_size": 1e8,
    "num_layers": 1e8,
    "num_heads": 1e8,
    "d_model": 1e8,
    "recent_days": 1e8,
    "window_size": 1e8,
    "patch_size": 1e8,
    "num_classes": 1e8,
    "input_size": 1e8
}

# Load the most recent scaler file (if any)
default_scaler_pattern = "scaler_standard_*.pkl"
scaler_candidates = glob.glob(default_scaler_pattern)
scaler_candidates.sort(reverse=True)  # Use the latest file

if scaler_candidates:
    SCALER_PATH = scaler_candidates[0]
else:
    raise FileNotFoundError(f"No scaler file matching pattern: {default_scaler_pattern}")
