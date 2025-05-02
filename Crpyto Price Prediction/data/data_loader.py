# data/data_loader.py (Public-safe version)

import pandas as pd
from supabase import create_client, Client
from config import (
    SUPABASE_URL,
    SUPABASE_KEY,
    COMMENTS_TABLE,
    TECH_INDICATORS_TABLE,
    SENTIMENT_INDICATORS_TABLE,
)
from datetime import datetime, timedelta

def get_supabase_client() -> Client:
    """
    Create and return a Supabase client instance.
    """
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def load_comments_data(recent_days: int = int(1e8)) -> pd.DataFrame:
    """
    Load comment data from Supabase in paginated batches.
    """
    client = get_supabase_client()
    start_datetime = datetime.combine(
        datetime.utcnow().date() - timedelta(days=recent_days),
        datetime.min.time()
    ).isoformat()

    all_data = []
    batch_size = 1000
    offset = 0

    while True:
        response = client.table(COMMENTS_TABLE)\
            .select("*")\
            .gte("timestamp", start_datetime)\
            .range(offset, offset + batch_size - 1)\
            .execute()

        data = getattr(response, "data", [])
        if not data:
            break
        all_data.extend(data)
        if len(data) < batch_size:
            break
        offset += batch_size

    if not all_data:
        raise Exception("No comment data retrieved from Supabase.")

    return pd.DataFrame(all_data)

def load_technical_indicators(recent_days: int = int(1e8)) -> pd.DataFrame:
    """
    Load technical indicator data from Supabase.
    """
    client = get_supabase_client()
    start_datetime = datetime.combine(
        datetime.utcnow().date() - timedelta(days=recent_days),
        datetime.min.time()
    ).isoformat()

    all_data = []
    batch_size = 1000
    offset = 0

    while True:
        response = client.table(TECH_INDICATORS_TABLE)\
            .select("*")\
            .gte("timestamp", start_datetime)\
            .range(offset, offset + batch_size - 1)\
            .execute()

        data = getattr(response, "data", [])
        if not data:
            break
        all_data.extend(data)
        if len(data) < batch_size:
            break
        offset += batch_size

    if not all_data:
        raise Exception("No technical indicator data retrieved from Supabase.")

    return pd.DataFrame(all_data)

def load_sentiment_indicators(recent_days: int = int(1e8)) -> pd.DataFrame:
    """
    Load sentiment scores from Supabase.
    """
    client = get_supabase_client()
    start_datetime = datetime.combine(
        datetime.utcnow().date() - timedelta(days=recent_days),
        datetime.min.time()
    ).isoformat()

    response = client.table(SENTIMENT_INDICATORS_TABLE)\
        .select("*")\
        .gte("timestamp", start_datetime)\
        .limit(100000)\
        .execute()

    data = getattr(response, "data", [])
    if not data:
        raise Exception("No sentiment indicator data found in Supabase.")
    
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.floor("D")
    return df

def _get_recent_threshold(*args, **kwargs) -> str:
    """
    Placeholder for logic to determine recent data threshold.
    The actual version uses the latest timestamp in the table.
    """
    return datetime.utcnow().isoformat()
