"""Centralized configuration (heavily redacted for public portfolio).

Sensitive values are loaded from environment variables. Numeric knobs are
generalized to neutral placeholders to avoid disclosing tuned values.
Where ranges are needed, values derive from a generic base rather than
exposing domain-specific choices.
"""

import os
from dotenv import load_dotenv
import json

# Load .env when present (local development only)
load_dotenv()

# Environment type: 'local' or 'production'
ENV_TYPE = os.getenv("ENV_TYPE", "local").lower()

# Supabase (redacted; must be provided via environment)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# GCS (redacted; bucket defaults to placeholder)
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "REDACTED_BUCKET")
GCS_MODEL_DIR = os.getenv("GCS_MODEL_DIR", "models")

# Generic placeholders (override via env if needed)
GENERIC_DAYS = int(os.getenv("GENERIC_DAYS", 1))
GENERIC_RATIO = float(os.getenv("GENERIC_RATIO", 0.5))
GENERIC_THRESHOLD = float(os.getenv("GENERIC_THRESHOLD", 0.5))
GENERIC_N = int(os.getenv("GENERIC_N", 2))
GENERIC_GAP_HOURS = int(os.getenv("GENERIC_GAP_HOURS", 24))
GENERIC_MEMORY_MB = int(os.getenv("GENERIC_MEMORY_MB", 1024))
GENERIC_LR = float(os.getenv("GENERIC_LR", 1e-3))
GENERIC_BATCH = int(os.getenv("GENERIC_BATCH", 32))
GENERIC_TIMEOUT_S = int(os.getenv("GENERIC_TIMEOUT_S", 5))
GENERIC_MIN_DELTA = float(os.getenv("GENERIC_MIN_DELTA", 0.001))

# Evaluation settings (generalized)
PREDICTION_EVAL_DAYS = int(os.getenv("PREDICTION_EVAL_DAYS", GENERIC_DAYS))
THRESHOLD_ACCURACY = float(os.getenv("THRESHOLD_ACCURACY", GENERIC_THRESHOLD))
RECENT_DAYS_RETRAIN = int(os.getenv("RECENT_DAYS_RETRAIN", GENERIC_DAYS))
RECENT_DAYS_INFERENCE = int(os.getenv("RECENT_DAYS_INFERENCE", GENERIC_DAYS))

# Train/test split settings (generalized)
TEST_DAYS = int(os.getenv("TEST_DAYS", GENERIC_DAYS))

# Time-aware CV ratios (generalized; conceptually rolling/cross-validated)
ROLLING_CV_TRAIN_RATIO = float(os.getenv("ROLLING_CV_TRAIN_RATIO", GENERIC_RATIO))
ROLLING_CV_VAL_RATIO = float(os.getenv("ROLLING_CV_VAL_RATIO", GENERIC_RATIO))
ROLLING_CV_STRIDE_RATIO = float(os.getenv("ROLLING_CV_STRIDE_RATIO", GENERIC_RATIO))
ROLLING_CV_N_SPLITS = int(os.getenv("ROLLING_CV_N_SPLITS", GENERIC_N))
ROLLING_CV_GAP_SIZE = int(os.getenv("ROLLING_CV_GAP_SIZE", GENERIC_GAP_HOURS))
SCALER_FIT_RATIO = float(os.getenv("SCALER_FIT_RATIO", GENERIC_RATIO))

# Incremental learning settings (generalized)
INCREMENTAL_DAYS = int(os.getenv("INCREMENTAL_DAYS", GENERIC_DAYS))
INCREMENTAL_EPOCHS = int(os.getenv("INCREMENTAL_EPOCHS", GENERIC_N))
INCREMENTAL_LR = float(os.getenv("INCREMENTAL_LR", GENERIC_LR))
INCREMENTAL_BATCH_SIZE = int(os.getenv("INCREMENTAL_BATCH_SIZE", GENERIC_BATCH))
INCREMENTAL_PERFORMANCE_THRESHOLD = float(os.getenv("INCREMENTAL_PERFORMANCE_THRESHOLD", GENERIC_THRESHOLD))
INCREMENTAL_VALIDATION_RATIO = float(os.getenv("INCREMENTAL_VALIDATION_RATIO", GENERIC_RATIO))
INCREMENTAL_MAX_DEGRADATION = float(os.getenv("INCREMENTAL_MAX_DEGRADATION", GENERIC_RATIO / 10))
INCREMENTAL_MIN_IMPROVEMENT = float(os.getenv("INCREMENTAL_MIN_IMPROVEMENT", GENERIC_MIN_DELTA))

# Classification threshold (generalized)
DEFAULT_CLASSIFICATION_THRESHOLD = float(os.getenv("DEFAULT_CLASSIFICATION_THRESHOLD", GENERIC_THRESHOLD))

# Early Stopping (generalized)
EARLY_STOP_PATIENCE = int(os.getenv("EARLY_STOP_PATIENCE", GENERIC_N))
EARLY_STOP_MAX_LOSS_DIFF = float(os.getenv("EARLY_STOP_MAX_LOSS_DIFF", GENERIC_RATIO))
EARLY_STOP_MIN_F1_IMPROVEMENT = float(os.getenv("EARLY_STOP_MIN_F1_IMPROVEMENT", GENERIC_MIN_DELTA))

# Data sources
TECH_INDICATORS_TABLE = os.getenv("TECH_INDICATORS_TABLE", "technical_indicators")

# Feature columns used by the model (generic placeholders)
FEATURE_COLS = [
    "feature_1", "feature_2", "feature_3", "feature_4",
    "feature_5", "feature_6", "feature_7", "feature_8",
]

# Verification settings (generalized)
VERIFY_MAX_RECORDS = int(os.getenv("VERIFY_MAX_RECORDS", GENERIC_N * 1000))
VERIFY_BATCH_SIZE = int(os.getenv("VERIFY_BATCH_SIZE", GENERIC_N * 500))

# Memory optimization (generalized)
MEMORY_THRESHOLD_MB = int(os.getenv("MEMORY_THRESHOLD_MB", GENERIC_MEMORY_MB))
BATCH_PROCESSING_SIZE = int(os.getenv("BATCH_PROCESSING_SIZE", GENERIC_BATCH * 32))
SEQUENCE_CACHE_ENABLED = os.getenv("SEQUENCE_CACHE_ENABLED", "true").lower() == "true"
MEMORY_CHECK_INTERVAL = int(os.getenv("MEMORY_CHECK_INTERVAL", GENERIC_N * 5))

# Temporal gap handling (generalized)
MAX_TIME_GAP_HOURS = int(os.getenv("MAX_TIME_GAP_HOURS", GENERIC_GAP_HOURS))

# Optimization trials (generalized)
N_TRIALS = int(os.getenv("N_TRIALS", GENERIC_N * 10))

# Time separation / overlap guards (generalized)
PREDICTION_HORIZON_HOURS = int(os.getenv("PREDICTION_HORIZON_HOURS", max(1, GENERIC_GAP_HOURS // 6)))
VAL_TO_TRAIN_REMOVAL_RATIO = float(os.getenv("VAL_TO_TRAIN_REMOVAL_RATIO", float(GENERIC_N)))
FOLD_SIZE_IMBALANCE_THRESHOLD = float(os.getenv("FOLD_SIZE_IMBALANCE_THRESHOLD", 1.0 + GENERIC_RATIO))

# Unified coin info: prefer environment JSON, fall back to minimal public-safe set
_coin_info_env = os.getenv("COIN_INFO_JSON")
if _coin_info_env:
    try:
        COIN_INFO = json.loads(_coin_info_env)
    except Exception:
        COIN_INFO = {
            "bitcoin": {"name": "Bitcoin", "coingecko_id": "bitcoin"},
            "ethereum": {"name": "Ethereum", "coingecko_id": "ethereum"},
        }
else:
    COIN_INFO = {
        "bitcoin": {"name": "Bitcoin", "coingecko_id": "bitcoin"},
        "ethereum": {"name": "Ethereum", "coingecko_id": "ethereum"},
    }

# Derived mappings (for compatibility with existing code)
COIN_NAME_MAPPING = {key: info["name"] for key, info in COIN_INFO.items()}
COINGECKO_COIN_MAPPING = {info["name"]: info["coingecko_id"] for info in COIN_INFO.values()}

# Model artifact paths (redacted defaults; override via env)
MODEL_PATH = os.getenv("MODEL_PATH", "models/patchseq_final_model.pt")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler_standard.pkl")

# External API settings (generalized)
COINGECKO_API_BASE_URL = os.getenv("COINGECKO_API_BASE_URL", "https://api.coingecko.com/api/v3")
COINGECKO_PRICE_ENDPOINT = os.getenv("COINGECKO_PRICE_ENDPOINT", "/simple/price")
COINGECKO_REQUEST_TIMEOUT = int(os.getenv("COINGECKO_REQUEST_TIMEOUT", GENERIC_TIMEOUT_S))

# Price columns
PRICE_USD_COLUMN = os.getenv("PRICE_USD_COLUMN", "price_usd")
PRICE_KRW_COLUMN = os.getenv("PRICE_KRW_COLUMN", "price_krw")

# Optuna hyperparameter search space (generalized via a base)
OPTUNA_BASE = int(os.getenv("OPTUNA_BASE", 16))
OPTUNA_PARAM_SPACE = {
    "patch_len": [OPTUNA_BASE // 2, OPTUNA_BASE],
    "window_size": [OPTUNA_BASE * 2, OPTUNA_BASE * 4],
    "stride": [max(1, OPTUNA_BASE // 4), max(1, OPTUNA_BASE // 2)],
    "d_model": [OPTUNA_BASE * 4, OPTUNA_BASE * 8],
    "mlp_hidden_mult": [1, 2],
    "activation": ["relu", "gelu"],
    "pooling_type": ["cls", "mean"],
    # Training
    "dropout_mlp": {"min": GENERIC_RATIO / 2, "max": GENERIC_RATIO},
    "learning_rate": {"min": GENERIC_LR / 10, "max": GENERIC_LR * 10, "log": True},
    "weight_decay": {"min": GENERIC_RATIO / 100, "max": GENERIC_RATIO / 10},
    # Loss
    "loss_type": ["ce", "focal"],
    "focal_gamma": {"min": 1.5, "max": 2.5},
    # Fixed
    "num_layers": GENERIC_N,
    "num_heads": GENERIC_N * 2,
    "batch_size": GENERIC_BATCH * 2,
}

def normalize_coin_name(coin_name: str) -> str:
    """Normalize coin name using COIN_NAME_MAPPING; fall back to the input value."""
    if not isinstance(coin_name, str):
        return coin_name
    return COIN_NAME_MAPPING.get(coin_name.lower(), coin_name)