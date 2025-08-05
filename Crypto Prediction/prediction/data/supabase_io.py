import pandas as pd
from datetime import datetime, timedelta
from supabase import create_client
from config import (
    SUPABASE_URL,
    SUPABASE_KEY,
    TECH_INDICATORS_TABLE,
    RECENT_DAYS_RETRAIN,
    RECENT_DAYS_INFERENCE,
    normalize_coin_name,
    PRICE_USD_COLUMN
)
from utils.timestamp_utils import safe_parse_timestampz, safe_parse_timestamp_series, normalize_timestamp_for_query

# Create Supabase client
client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================
# 🔹 Save prediction results
# ============================
def load_recent_predictions(table_name="predictions", limit=100):
    try:
        response = client.table(table_name)\
            .select("coin,timestamp")\
            .order("timestamp", desc=True)\
            .limit(limit)\
            .execute()
        data = response.__dict__.get("data", [])
        # For duplicate check, store as {(coin, timestamp_str): True}
        cached = {(r['coin'], r['timestamp']): True for r in data}
        return cached
    except Exception as e:
        print(f"[❌] Failed to load recent predictions: {e}")
        return {}

def save_prediction(data: dict, cached: dict, table_name="predictions"):
    key = (data['coin'], data['timestamp'])
    if key in cached:
        print(f"[ℹ️] Prediction already exists for {key}, skipping insert.")
        return None
    try:
        response = client.table(table_name).insert(data).execute()
        cached[key] = True  # Add newly saved item to cache
        print(f"[✅] Saved prediction for {key}")
        
        # Return the ID of the inserted record
        inserted_data = response.data
        if inserted_data and len(inserted_data) > 0:
            return inserted_data[0].get('id')
        return None
    except Exception as e:
        print(f"[❌] Failed to save prediction to Supabase: {e}")
        return None

# ============================
# 🔹 Load technical indicators (paging)
# ============================
def load_technical_indicators(for_training: bool = False) -> pd.DataFrame:
    """Load technical indicator data compatible with timestampz
    
    Args:
        for_training (bool): If True, for retrain (100 days), if False, for inference (4 days)
    """
    recent_days = RECENT_DAYS_RETRAIN if for_training else RECENT_DAYS_INFERENCE
    start_datetime = normalize_timestamp_for_query(
        datetime.combine(
            datetime.utcnow().date() - timedelta(days=recent_days),
            datetime.min.time()
        )
    )

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
        print(f"[⚠️] No data found in {TECH_INDICATORS_TABLE}.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    print(f"[✅] Loaded {len(all_data):,} technical indicator data")
    
    # ✅ v11_2 compatible timestampz processing
    df["timestamp"] = safe_parse_timestamp_series(df["timestamp"])
    
    # Apply coin name normalization
    df["coin"] = df["coin"].apply(normalize_coin_name)
    unique_coins = df['coin'].nunique()
    print(f"[🔄] Coin name normalization complete. Unique coins: {unique_coins}")
    
    # Data quality check
    null_timestamps = df['timestamp'].isnull().sum()
    if null_timestamps > 0:
        print(f"[⚠️] {null_timestamps} timestamp parsing failed - removed")
        df = df.dropna(subset=['timestamp'])
    
    print(f"[✅] Final data: {len(df):,} records, {unique_coins} coins")
    return df

# ============================
# 🔸 Calculate reference time
# ============================
def get_recent_threshold(table: str) -> str:
    """Calculate reference time compatible with timestampz"""
    latest_response = client.table(table)\
        .select("timestamp")\
        .order("timestamp", desc=True)\
        .limit(1)\
        .execute()

    data = latest_response.__dict__.get('data', [])
    if not data:
        raise Exception(f"Supabase error retrieving latest timestamp from {table}")

    # Safe timestampz processing
    timestamp_raw = data[0]['timestamp']
    latest_ts = safe_parse_timestampz(timestamp_raw)
    
    threshold = latest_ts - timedelta(days=RECENT_DAYS_RETRAIN)
    return normalize_timestamp_for_query(threshold)

# ============================
# 🔹 Update price information
# ============================
def update_prediction_prices(prediction_ids: list, coin_names: list, price_data: dict, table_name="predictions"):
    """
    Batch update price information for prediction records.
    
    Args:
        prediction_ids: List of prediction record IDs to update
        coin_names: List of coin names (same order as prediction_ids)
        price_data: Price data from CoinGecko API
        table_name: Table name
    """
    from utils.price_utils import format_price_update_data
    
    if not prediction_ids or not coin_names:
        print("⚠️ No prediction data to update.")
        return
    
    print(f"💰 Starting price information update: {len(prediction_ids)} predictions")
    
    updated_count = 0
    failed_count = 0
    
    for pred_id, coin_name in zip(prediction_ids, coin_names):
        if pred_id is None:
            print(f"⚠️ {coin_name}: No prediction ID, skipping price update")
            failed_count += 1
            continue
            
        # Format price data
        price_update = format_price_update_data(coin_name, price_data)
        
        if not price_update:
            print(f"⚠️ {coin_name}: No price information")
            failed_count += 1
            continue
        
        try:
            # Supabase update
            client.table(table_name).update(price_update).eq('id', pred_id).execute()
            from config import PRICE_USD_COLUMN
            print(f"✅ {coin_name}: Price update complete (${price_update[PRICE_USD_COLUMN]})")
            updated_count += 1
            
        except Exception as e:
            print(f"❌ {coin_name}: Price update failed - {e}")
            failed_count += 1
    
    print(f"💰 Price update complete: {updated_count} succeeded, {failed_count} failed")
    
    return {
        'updated': updated_count,
        'failed': failed_count,
        'total': len(prediction_ids)
    }


def get_predictions_without_price(hours_back=24, table_name="predictions") -> pd.DataFrame:
    """
    Query recent predictions without price information.
    
    Args:
        hours_back: How many hours back to query predictions
        table_name: Table name
    
    Returns:
        DataFrame of predictions without price information
    """
    cutoff_time = normalize_timestamp_for_query(datetime.utcnow() - timedelta(hours=hours_back))
    
    try:
        response = client.table(table_name)\
            .select("id, coin, timestamp, finalscore, pricetrend")\
            .gte("timestamp", cutoff_time)\
            .is_(PRICE_USD_COLUMN, "null")\
            .order("timestamp", desc=True)\
            .execute()
        
        data = response.data
        if not data:
            print("ℹ️ No predictions without price information.")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        print(f"🔍 Found {len(df)} predictions without price information")
        return df
        
    except Exception as e:
        print(f"❌ Failed to query predictions without price information: {e}")
        return pd.DataFrame()
