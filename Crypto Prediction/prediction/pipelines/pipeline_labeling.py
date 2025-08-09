from datetime import datetime, timedelta
import os
from supabase import create_client
from dotenv import load_dotenv
from config import normalize_coin_name
from utils.timestamp_utils import safe_parse_timestampz, normalize_timestamp_for_query

# Load environment and create Supabase client
load_dotenv()
client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

LABELING_BATCH_SIZE = 1000  # batch size


def normalize_coin_name_labeling(coin_name):
    """Normalize coin name using config.normalize_coin_name."""
    return normalize_coin_name(coin_name)


def get_unlabeled_records_paged(cutoff_time, batch_size=LABELING_BATCH_SIZE):
    """Yield unlabeled records (price_trend is NULL) older than cutoff_time in pages."""
    if isinstance(cutoff_time, datetime):
        cutoff_time_str = cutoff_time.isoformat()
    else:
        cutoff_time_str = cutoff_time
    
    offset = 0
    
    while True:
        resp = client.table("technical_indicators") \
            .select("id, coin, ema, timestamp") \
            .is_("price_trend", "null") \
            .lte("timestamp", cutoff_time_str) \
            .order("id") \
            .range(offset, offset + batch_size - 1) \
            .execute()
        
        data = resp.data
        if not data:
            break
        
        print(f"[📦] Loaded batch: {len(data)} records (offset: {offset})")
        yield data
        
        if len(data) < batch_size:
            break
            
        offset += batch_size


def find_future_ema(coin, timestamp, all_future_data):
    """Find EMA approximately 4 hours after the given timestamp for a coin."""
    target_time = timestamp + timedelta(hours=4)
    
    time_buffer = timedelta(hours=2)
    search_start = target_time - time_buffer
    search_end = target_time + time_buffer
    
    coin_data = all_future_data.get(coin, [])
    
    best_match = None
    min_diff = timedelta.max
    
    for future_record in coin_data:
        future_time = future_record['ts']
        if search_start <= future_time <= search_end:
            diff = abs(future_time - target_time)
            if diff < min_diff:
                min_diff = diff
                best_match = future_record['ema']
                print(f"[⏱️] Found future EMA for {coin}: {future_time} (diff: {diff})")
    
    if best_match is not None:
        print(f"[✅] Best match for {coin}: {best_match} (target: {target_time}, actual: {target_time + min_diff if min_diff < timedelta.max else 'N/A'})")
    
    return best_match


def load_future_data_efficiently(min_timestamp):
    """Load future EMA data efficiently (single pass)."""
    if isinstance(min_timestamp, datetime):
        min_timestamp_str = min_timestamp.isoformat()
    else:
        min_timestamp_str = min_timestamp
    
    print(f"[📊] Loading future data from {min_timestamp_str}...")
    
    future_data = {}
    offset = 0
    batch_size = 2000
    
    while True:
        resp = client.table("technical_indicators") \
            .select("coin, ema, timestamp") \
            .gte("timestamp", min_timestamp_str) \
            .not_.is_("ema", "null") \
            .order("coin, timestamp") \
            .range(offset, offset + batch_size - 1) \
            .execute()
        
        data = resp.data
        if not data:
            break
        
        for record in data:
            try:
                normalized_coin = normalize_coin_name_labeling(record['coin'])
                ts = parse_timestamp_safe(record['timestamp'])
                if ts is None:
                    continue
                
                if normalized_coin not in future_data:
                    future_data[normalized_coin] = []
                
                future_data[normalized_coin].append({
                    'ema': float(record['ema']),
                    'ts': ts
                })
            except Exception as e:
                print(f"[⚠️] Failed to parse future record: {e}")
                continue
        
        print(f"[📦] Loaded future batch: {len(data)} records (offset: {offset})")
        
        if len(data) < batch_size:
            break
            
        offset += batch_size
    
    for coin in future_data:
        future_data[coin].sort(key=lambda x: x['ts'])
    
    total_records = sum(len(records) for records in future_data.values())
    print(f"[✅] Future data loaded: {len(future_data)} coins, {total_records} total records")
    
    return future_data


def parse_timestamp_safe(timestamp_input):
    """Safely parse timestampz (v11_2 compatible)."""
    try:
        return safe_parse_timestampz(timestamp_input)
    except Exception as e:
        print(f"[⚠️] Failed to parse timestampz: {timestamp_input}, error: {e}")
        return None


def run_labeling():
    """Run price_trend labeling pipeline."""
    print("=== Labeling Pipeline Started ===")
    try:
        now = datetime.utcnow()
        initial_cutoff_time = (now - timedelta(hours=4)).isoformat()
        
        print(f"[⏰] Initial cutoff time: {initial_cutoff_time}")
        
        # Load all unlabeled data first to determine the real cutoff time
        all_unlabeled_data = []
        offset = 0
        
        while True:
            resp = client.table("technical_indicators") \
                .select("id, coin, ema, timestamp") \
                .is_("price_trend", "null") \
                .order("timestamp", desc=True) \
                .range(offset, offset + LABELING_BATCH_SIZE - 1) \
                .execute()
            
            data = resp.data
            if not data:
                break
            
            all_unlabeled_data.extend(data)
            
            if len(data) < LABELING_BATCH_SIZE:
                break
                
            offset += LABELING_BATCH_SIZE
        
        if not all_unlabeled_data:
            print("=== No records to label ===")
            return {"message": "No records to label"}, 200
        
        # Determine actual cutoff based on the newest available timestamp
        parsed_timestamps = []
        for record in all_unlabeled_data:
            parsed_ts = parse_timestamp_safe(record['timestamp'])
            if parsed_ts:
                parsed_timestamps.append(parsed_ts)
        
        if not parsed_timestamps:
            print("=== No valid timestamps found ===")
            return {"message": "No valid timestamps found"}, 200
        
        latest_timestamp = max(parsed_timestamps)
        actual_cutoff_time = latest_timestamp - timedelta(hours=5)
        
        print(f"[📅] Latest timestamp in data: {latest_timestamp}")
        print(f"[⏰] Actual cutoff time: {actual_cutoff_time}")
        
        # Filter to rows older than the cutoff
        filtered_data = []
        for record in all_unlabeled_data:
            parsed_ts = parse_timestamp_safe(record['timestamp'])
            if parsed_ts and parsed_ts <= actual_cutoff_time:
                filtered_data.append(record)
        
        print(f"[📋] Total records loaded: {len(all_unlabeled_data)}")
        print(f"[📋] Records to process (after 5h filter): {len(filtered_data)}")
        
        if not filtered_data:
            print("=== No records to label (all records are less than 5 hours old) ===")
            return {"message": "No records to label (all records are less than 5 hours old)"}, 200
        
        # Load future EMA data once based on earliest cutoff
        min_future_timestamp = actual_cutoff_time.isoformat()
        future_data = load_future_data_efficiently(min_future_timestamp)
        
        updated_count = 0
        batch_count = 0
        
        # Process in batches
        batch_size = LABELING_BATCH_SIZE
        for i in range(0, len(filtered_data), batch_size):
            batch = filtered_data[i:i + batch_size]
            batch_count += 1
            batch_updates = []
            
            for record in batch:
                try:
                    original_coin = record['coin']
                    normalized_coin = normalize_coin_name_labeling(original_coin)
                    current_ema = float(record['ema'])
                    timestamp = parse_timestamp_safe(record['timestamp'])
                    if timestamp is None:
                        print(f"[❌] Failed to parse timestamp for record {record.get('id', 'unknown')}")
                        continue
                    
                    future_ema = find_future_ema(normalized_coin, timestamp, future_data)
                    
                    if future_ema is None:
                        if timestamp + timedelta(hours=4) <= now:
                            price_trend = 'down'
                        else:
                            continue
                    elif future_ema > current_ema:
                        price_trend = 'up'
                    elif future_ema < current_ema:
                        price_trend = 'down'
                    else:
                        price_trend = 'neutral'
                    
                    batch_updates.append({
                        'id': record['id'],
                        'price_trend': price_trend
                    })
                    
                except Exception as e:
                    print(f"[❌] Failed to process record {record.get('id', 'unknown')}: {e}")
                    continue
            
            if batch_updates:
                try:
                    for update in batch_updates:
                        client.table("technical_indicators") \
                            .update({"price_trend": update['price_trend']}) \
                            .eq("id", update['id']) \
                            .execute()
                    
                    updated_count += len(batch_updates)
                    print(f"[💾] Batch {batch_count}: Updated {len(batch_updates)} records")
                    
                except Exception as e:
                    print(f"[❌] Batch update failed: {e}")
        
        message = f"Labeling finished. Updated: {updated_count} records"
        print(f"=== Labeling Pipeline Finished. {message} ===")
        return {"message": message}, 200
        
    except Exception as e:
        print(f"[💥 Labeling Pipeline Error] {e}")
        return {"error": str(e)}, 500