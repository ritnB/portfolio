# data/supabase_io.py (Public-safe version)

from supabase import create_client
from supabase.client import Client
from config import SUPABASE_URL, SUPABASE_KEY

# Initialize Supabase client
client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def _silent_insert(result: dict, table_name: str):
    """
    Inserts or upserts a result into the given table.
    Conflict resolution logic may vary depending on schema.
    """
    try:
        response = client.table(table_name)\
            .upsert([result])\
            .execute()

        if hasattr(response, "error") and response.error:
            print(f"[Supabase Upsert Failed] Error: {response.error}")
    except Exception as e:
        print(f"[Exception during upsert to {table_name}] {e}")

def save_sentiment_result(timestamp: str, coin: str, sentiment_score: float):
    """
    Saves a single sentiment analysis result.
    """
    _silent_insert({
        "timestamp": timestamp,
        "coin": coin,
        "sentiment_score": sentiment_score
    }, table_name="your_sentiment_table")

def save_prediction(result: dict, table_name: str):
    """
    Saves a result dictionary to the specified table.
    """
    _silent_insert(result, table_name)
