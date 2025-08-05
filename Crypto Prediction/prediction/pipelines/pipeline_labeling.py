# pipeline_labeling.py

from datetime import datetime, timedelta
import os
from supabase import create_client
from dotenv import load_dotenv
from config import normalize_asset_name
from utils.timestamp_utils import safe_parse_timestampz, normalize_timestamp_for_query

# Load environment variables and create Database client
load_dotenv()
client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

LABELING_BATCH_SIZE = 1000  # Batch processing size

def normalize_asset_name_labeling(asset_name):
    """
    Normalize asset names (using normalize_asset_name function from config.py)
    """
    return normalize_asset_name(asset_name)

def get_unlabeled_records_paged(cutoff_time, batch_size=LABELING_BATCH_SIZE):
    """
    Get records where price_trend is NULL and before cutoff_time using paging
    
    Args:
        cutoff_time: datetime object or ISO format string
        batch_size: paging batch size
    
    Returns:
        generator: returns data in batches
    """
    # Normalize cutoff_time to ISO format string
    if isinstance(cutoff_time, datetime):
        cutoff_time_str = cutoff_time.isoformat()
    else:
        cutoff_time_str = cutoff_time
    
    offset = 0
    
    while True:
        # Query records where price_trend is NULL and before cutoff_time
        resp = client.table("technical_indicators") \
            .select("id, asset, feature_2, timestamp") \
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
        
        # If fewer records than batch size, no more data
        if len(data) < batch_size:
            break
            
        offset += batch_size

def find_future_feature(asset, timestamp, all_future_data):
    """
    Find specific indicator value 4 hours after a given time for a specific asset (with time range flexibility)
    
    Args:
        asset: normalized asset name
        timestamp: reference time
        all_future_data: pre-loaded future data dictionary
    
    Returns:
        float or None: indicator value 4 hours later
    """
    target_time = timestamp + timedelta(hours=4)
    
    # Time range setting (±2 hours buffer)
    time_buffer = timedelta(hours=2)
    search_start = target_time - time_buffer
    search_end = target_time + time_buffer
    
    # Find closest data within range for the asset
    asset_data = all_future_data.get(asset, [])
    
    best_match = None
    min_diff = timedelta.max
    
    for future_record in asset_data:
        future_time = future_record['ts']
        
        # Check if within search range
        if search_start <= future_time <= search_end:
            diff = abs(future_time - target_time)
            if diff < min_diff:
                min_diff = diff
                best_match = future_record['feature_2']
                print(f"[⏱️] Found future feature for {asset}: {future_time} (diff: {diff})")
    
    if best_match is not None:
        print(f"[✅] Best match for {asset}: {best_match} (target: {target_time}, actual: {target_time + min_diff if min_diff < timedelta.max else 'N/A'})")
    
    return best_match

def load_future_data_efficiently(min_timestamp):
    """
    Load future data efficiently (load only once)
    
    Args:
        min_timestamp: datetime object or ISO format string
    
    Returns:
        dict: organized future data by coin
    """
    # Normalize min_timestamp to ISO format string
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
            .select("asset, feature_2, timestamp") \
            .gte("timestamp", min_timestamp_str) \
            .not_.is_("feature_2", "null") \
            .order("asset, timestamp") \
            .range(offset, offset + batch_size - 1) \
            .execute()
        
        data = resp.data
        if not data:
            break
        
        # Organize data by coin
        for record in data:
            try:
                normalized_asset = normalize_asset_name_labeling(record['asset'])
                ts = parse_timestamp_safe(record['timestamp'])
                if ts is None:
                    continue
                
                if normalized_asset not in future_data:
                    future_data[normalized_asset] = []
                
                future_data[normalized_asset].append({
                    'feature_2': float(record['feature_2']),
                    'ts': ts
                })
            except Exception as e:
                print(f"[⚠️] Failed to parse future record: {e}")
                continue
        
        print(f"[📦] Loaded future batch: {len(data)} records (offset: {offset})")
        
        if len(data) < batch_size:
            break
            
        offset += batch_size
    
    # Sort data by timestamp for each coin
    for asset in future_data:
        future_data[asset].sort(key=lambda x: x['ts'])
    
    total_records = sum(len(records) for records in future_data.values())
    print(f"[✅] Future data loaded: {len(future_data)} assets, {total_records} total records")
    
    return future_data

def parse_timestamp_safe(timestamp_input):
    """
    Safely parse timestampz (compatible with v11_2)
    
    Args:
        timestamp_input: timestampz string or datetime object
    
    Returns:
        datetime or None: parsed datetime object
    """
    try:
        return safe_parse_timestampz(timestamp_input)
    except Exception as e:
        print(f"[⚠️] Failed to parse timestampz: {timestamp_input}, error: {e}")
        return None

def run_labeling():
    """
    Run price_trend labeling pipeline
    """
    print("=== Labeling Pipeline Started ===")
    try:
        now = datetime.utcnow()
        initial_cutoff_time = (now - timedelta(hours=4)).isoformat()
        
        print(f"[⏰] Initial cutoff time: {initial_cutoff_time}")
        
        # First, get all data to determine the actual cutoff time
        all_unlabeled_data = []
        offset = 0
        
        # Get all unlabeled data up to the current time (to find data 4 hours later)
        # Fetch all unlabeled data without timestamp filtering
        
        while True:
            resp = client.table("technical_indicators") \
                .select("id, asset, feature_2, timestamp") \
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
        
        # Find the latest timestamp among the fetched data
        parsed_timestamps = []
        for record in all_unlabeled_data:
            parsed_ts = parse_timestamp_safe(record['timestamp'])
            if parsed_ts:
                parsed_timestamps.append(parsed_ts)
        
        if not parsed_timestamps:
            print("=== No valid timestamps found ===")
            return {"message": "No valid timestamps found"}, 200
        
        latest_timestamp = max(parsed_timestamps)
        
        # Actual cutoff time = 5 hours before the latest timestamp
        actual_cutoff_time = latest_timestamp - timedelta(hours=5)
        
        print(f"[📅] Latest timestamp in data: {latest_timestamp}")
        print(f"[⏰] Actual cutoff time: {actual_cutoff_time}")
        
        # Filter data for records before the actual cutoff time
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
        
        # Load future data based on the earliest cutoff_time (once)
        min_future_timestamp = actual_cutoff_time.isoformat()
        future_data = load_future_data_efficiently(min_future_timestamp)
        
        updated_count = 0
        batch_count = 0
        
        # Process filtered data in batches
        batch_size = LABELING_BATCH_SIZE
        for i in range(0, len(filtered_data), batch_size):
            batch = filtered_data[i:i + batch_size]
            batch_count += 1
            batch_updates = []
            
            for record in batch:
                try:
                    original_asset = record['asset']
                    normalized_asset = normalize_asset_name_labeling(original_asset)
                    current_feature_2 = float(record['feature_2'])
                    timestamp = parse_timestamp_safe(record['timestamp'])
                    if timestamp is None:
                        print(f"[❌] Failed to parse timestamp for record {record.get('id', 'unknown')}")
                        continue
                    
                    # Find future feature
                    future_feature_2 = find_future_feature(normalized_asset, timestamp, future_data)
                    
                    # Price trend determination logic (same as SQL)
                    if future_feature_2 is None:
                        # No data after 4 hours = down
                        if timestamp + timedelta(hours=4) <= now:
                            price_trend = 'down'
                        else:
                            # Still less than 4 hours = do not process
                            continue
                    elif future_feature_2 > current_feature_2:
                        price_trend = 'up'
                    elif future_feature_2 < current_feature_2:
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
            
            # Execute batch updates
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