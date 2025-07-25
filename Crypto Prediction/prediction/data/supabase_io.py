import pandas as pd
from datetime import datetime, timedelta
from supabase import create_client
from config import (
    SUPABASE_URL,
    SUPABASE_KEY,
    TECH_INDICATORS_TABLE,
    RECENT_DAYS,
    normalize_coin_name
)

# Create Supabase client
client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================
# Prediction result storage
# ============================
def load_recent_predictions(table_name="predictions", limit=100):
    try:
        response = client.table(table_name)\
            .select("coin,timestamp")\
            .order("timestamp", desc=True)\
            .limit(limit)\
            .execute()
        data = response.__dict__.get("data", [])
        # Store in {(coin, timestamp_str): True} format for duplicate checking
        cached = {(r['coin'], r['timestamp']): True for r in data}
        return cached
    except Exception as e:
        print(f"[❌] Failed to load recent predictions: {e}")
        return {}

def save_prediction(data: dict, cached: dict, table_name="predictions"):
    key = (data['coin'], data['timestamp'])
    if key in cached:
        print(f"[ℹ️] Prediction already exists for {key}, skipping insert.")
        return
    try:
        client.table(table_name).insert(data).execute()
        cached[key] = True  # Add newly saved item to cache
        print(f"[✅] Saved prediction for {key}")
    except Exception as e:
        print(f"[❌] Failed to save prediction to Supabase: {e}")

# ============================
# Technical indicators loading (pagination method)
# ============================
def load_technical_indicators() -> pd.DataFrame:
    start_datetime = datetime.combine(
        datetime.now().date() - timedelta(days=RECENT_DAYS),
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

        data = response.__dict__.get("data", [])
        if not data:
            break

        all_data.extend(data)
        if len(data) < batch_size:
            break
        offset += batch_size

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Apply coin name normalization
    df["coin"] = df["coin"].apply(normalize_coin_name)
    print(f"[🔄] Normalized coin names. Unique coins: {df['coin'].unique()}")
    
    return df

# ============================
# Reference time calculation
# ============================
def get_recent_threshold(table: str) -> str:
    latest_response = client.table(table)\
        .select("timestamp")\
        .order("timestamp", desc=True)\
        .limit(1)\
        .execute()

    data = latest_response.__dict__.get('data', [])
    if not data:
        raise Exception(f"Supabase error retrieving latest timestamp from {table}")

    latest_ts_str = data[0]['timestamp'].rstrip("Z")
    latest_ts = datetime.fromisoformat(latest_ts_str)
    threshold = latest_ts - timedelta(days=RECENT_DAYS)
    return threshold.isoformat()
