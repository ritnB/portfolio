# config.py - Centralized configuration management
import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# 🔑 API Keys & External Services
# =============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Supabase table names
SUPABASE_MARKET_TABLE = os.getenv("SUPABASE_MARKET_TABLE", "technical_indicators")
SUPABASE_COMMUNITY_TABLE = os.getenv("SUPABASE_COMMUNITY_TABLE", "comments")
SUPABASE_PREDICT_TABLE = os.getenv("SUPABASE_PREDICT_TABLE", "predictions")

# Social Media API configuration
SOCIAL_MEDIA_API_KEY = os.getenv("SOCIAL_MEDIA_API_KEY")
SOCIAL_MEDIA_API_URL = os.getenv("SOCIAL_MEDIA_API_URL")
AUTO_POSTING_ENABLED = os.getenv("AUTO_POSTING_ENABLED", "false").lower() == "true"

# =============================================================================
# 🤖 LLM Configuration (Free model support)
# =============================================================================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "huggingface")  # "openai", "huggingface", "ollama"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")  # Default to free model
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.7))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 500))

# Hugging Face configuration
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "facebook/blenderbot-400M-distill")

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")

# Free model fallback configuration (force enable)
FREE_MODEL_FALLBACK = True  # Force use of free models

# =============================================================================
# 📊 Data Analysis Configuration
# =============================================================================
RECENT_DAYS = int(os.getenv("RECENT_DAYS", 3))

# Surge/crash detection thresholds (EMA change rate %)
SURGE_THRESHOLD = float(os.getenv("SURGE_THRESHOLD", 10.0))  # 10%+ surge
CRASH_THRESHOLD = float(os.getenv("CRASH_THRESHOLD", -10.0))  # 10%+ crash

# Prediction accuracy promotion thresholds
ACCURACY_THRESHOLD_3DAY = float(os.getenv("ACCURACY_THRESHOLD_3DAY", 70.0))  # 3-day overall 70%
ACCURACY_THRESHOLD_1DAY = float(os.getenv("ACCURACY_THRESHOLD_1DAY", 80.0))  # 1-day overall 80%

# Maximum number of coins to analyze
MAX_COINS_TO_ANALYZE = int(os.getenv("MAX_COINS_TO_ANALYZE", 10))

# =============================================================================
# ✍️ Content Generation Configuration
# =============================================================================
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 100))  # Within 100 characters
MIN_CONTENT_LENGTH = int(os.getenv("MIN_CONTENT_LENGTH", 30))   # At least 30 characters

# Writing style configuration
CONTENT_STYLE = os.getenv("CONTENT_STYLE", "social")  # social, twitter, formal
USE_EMOJIS = os.getenv("USE_EMOJIS", "true").lower() == "true"

# =============================================================================
# 🎯 Quality Evaluation Criteria
# =============================================================================
# Weight for each criterion (total 100%)
EVAL_WEIGHTS = {
    "data_accuracy": float(os.getenv("EVAL_WEIGHT_DATA_ACCURACY", 40.0)),     # Data accuracy
    "market_relevance": float(os.getenv("EVAL_WEIGHT_MARKET_RELEVANCE", 30.0)), # Market relevance
    "engagement": float(os.getenv("EVAL_WEIGHT_ENGAGEMENT", 20.0)),           # Engagement potential
    "risk_management": float(os.getenv("EVAL_WEIGHT_RISK_MANAGEMENT", 10.0))  # Risk management
}

# Quality pass threshold score (out of 100)
QUALITY_PASS_SCORE = float(os.getenv("QUALITY_PASS_SCORE", 70.0))

# Minimum individual score for each criterion (used in strict mode)
QUALITY_MIN_INDIVIDUAL_SCORE = float(os.getenv("QUALITY_MIN_INDIVIDUAL_SCORE", 70.0))

# Strict evaluation mode
STRICT_EVALUATION = os.getenv("STRICT_EVALUATION", "true").lower() == "true"

# =============================================================================
# 🚨 Risk Management
# =============================================================================
# Forbidden keywords (prevent investment advice)
FORBIDDEN_KEYWORDS = [
    "buy", "sell", "purchase", "recommend", "guaranteed", "certain", 
    "sure thing", "investment advice", "financial advice"
]

# Include disclaimer option
INCLUDE_DISCLAIMER = os.getenv("INCLUDE_DISCLAIMER", "false").lower() == "true"

# =============================================================================
# 🐛 Debugging & Logging
# =============================================================================
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
VERBOSE_LOGGING = os.getenv("VERBOSE_LOGGING", "true").lower() == "true"

# Data caching configuration
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "false").lower() == "true"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 300))  # 5 minutes

# =============================================================================
# 🎨 Social Media Style Configuration
# =============================================================================
SOCIAL_MEDIA_HASHTAGS = os.getenv("HASHTAGS", "#crypto #AI").split()
SOCIAL_MEDIA_TONE = os.getenv("TONE", "casual")  # casual, professional, playful

# =============================================================================
# 🔄 Task Scheduling
# =============================================================================
SCHEDULER_ENABLED = os.getenv("SCHEDULER_ENABLED", "true").lower() == "true"
SCHEDULER_INTERVAL_HOURS = int(os.getenv("SCHEDULER_INTERVAL_HOURS", 12))
