# config.py - LangChain Agent Settings (Public Portfolio Version)
# All sensitive environment variable references and API keys have been redacted for public release.

# Example placeholder configuration for public portfolio:
OPENAI_API_KEY = "[REDACTED]"
SUPABASE_URL = "[REDACTED]"
SUPABASE_KEY = "[REDACTED]"
THREADS_API_KEY = "[REDACTED]"
THREADS_API_URL = "[REDACTED]"
HUGGINGFACE_API_KEY = "[REDACTED]"

# Other configuration values are set to safe defaults for demonstration purposes only.
SUPABASE_MARKET_TABLE = "technical_indicators"
SUPABASE_COMMUNITY_TABLE = "comments"
SUPABASE_PREDICT_TABLE = "predictions"
THREADS_UPLOAD_ENABLED = False
USE_LANGCHAIN_AGENT = True
LLM_PROVIDER = "huggingface"
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 500
HUGGINGFACE_MODEL = "facebook/blenderbot-400M-distill"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama2"
FREE_MODEL_FALLBACK = True
RECENT_DAYS = 3
SURGE_THRESHOLD = 10.0
CRASH_THRESHOLD = -10.0
ACCURACY_THRESHOLD_3DAY = 70.0
ACCURACY_THRESHOLD_1DAY = 80.0
MAX_COINS_TO_ANALYZE = 10
MIN_CONTENT_LENGTH = 10
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 8081
FLASK_DEBUG = False
AGENT_MAX_ITERATIONS = 10
AGENT_VERBOSE = True
DB_PAGE_SIZE = 1000
REQUEST_TIMEOUT = 30
CRYPTO_KEYWORDS = [
    r'surge\\w*', r'crash\\w*', r'pump\\w*', r'dump\\w*', r'bull\\w*', r'bear\\w*',
    r'ATH\\w*', r'all time high\\w*', r'bottom\\w*', r'support\\w*', r'resistance\\w*',
    r'breakout\\w*', r'breakdown\\w*', r'bounce\\w*', r'drop\\w*', r'rally\\w*',
    r'correction\\w*', r'reversal\\w*', r'trend\\w*', r'moon\\w*', r'rocket\\w*'
]
POSITIVE_KEYWORDS = [
    'good', 'great', 'bullish', 'pump', 'moon', 'rocket', 'surge', 'rally', 'strong', 'buy'
]
NEGATIVE_KEYWORDS = [
    'bad', 'bearish', 'dump', 'crash', 'drop', 'weak', 'sell', 'dump', 'bear', 'fall'
]
URGENCY_KEYWORDS = [
    'surge', 'crash', 'pump', 'dump', 'moon', 'rocket', 'ATH', 'bottom', 'breakout', 'breakdown'
]
PREDICTION_CONFIDENCE_THRESHOLD = 80.0
SENTIMENT_TOP_COUNT = 3
HOT_TOPICS_TOP_COUNT = 3
KEYWORDS_TOP_COUNT = 10
TREND_TOP_COUNT = 5
