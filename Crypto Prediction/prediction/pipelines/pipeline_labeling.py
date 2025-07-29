# pipeline_labeling.py

from datetime import datetime, timedelta
import os
from supabase import create_client
from dotenv import load_dotenv
from config import normalize_asset_name

# Load environment variables and create Supabase client
load_dotenv()
client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

LABELING_BATCH_SIZE = 1000  # Batch processing size

def normalize_asset_name_labeling(asset_name):
    """
    Normalize asset name (using normalize_asset_name function from config.py)
    """
    return normalize_asset_name(asset_name)

def get_unlabeled_records_paged(cutoff_time, batch_size=LABELING_BATCH_SIZE):
    """
    Get records with NULL price_trend before cutoff_time using paging
    
    Args:
        cutoff_time: Only records before this time
        batch_size: Paging batch size
    
    Returns:
        generator: Returns data in batches
    """
    offset = 0
    
    while True:
        # Query records with NULL price_trend before cutoff_time
        resp = client.table("technical_indicators") \
            .select("id, asset, ema, timestamp") \
            .is_("price_trend", "null") \
            .lte("timestamp", cutoff_time) \
            .order("id") \
            .range(offset, offset + batch_size - 1) \
            .execute()
        
        data = resp.data
        if not data:
            break
        
        print(f"[📦] Loaded batch: {len(data)} records (offset: {offset})")
        yield data
        
        # If less than batch size, no more data
        if len(data) < batch_size:
            break
            
        offset += batch_size

def find_future_ema(asset, timestamp, all_future_data):
    """
    Find EMA value 4 hours after specific time for specific asset (with time range flexibility)
    
    Args:
        asset: Normalized asset name
        timestamp: Reference time
        all_future_data: Pre-loaded future data dictionary
    
    Returns:
        float or None: EMA value 4 hours later
    """
    target_time = timestamp + timedelta(hours=4)
    
    # Set time range (±30 minutes buffer)
    time_buffer = timedelta(minutes=30)
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
                best_match = future_record['ema']
                print(f"[⏱️] Found future EMA for {asset}: {future_time} (diff: {diff})")
    
    if best_match is not None:
        print(f"[✅] Best match for {asset}: {best_match} (target: {target_time}, actual: {target_time + min_diff if min_diff < timedelta.max else 'N/A'})")
    
    return best_match

def load_future_data_efficiently(min_timestamp):
    """
    Load future data efficiently (load only once)
    
    Args:
        min_timestamp: Minimum timestamp (load only data after this)
    
    Returns:
        dict: Organized future data by asset
    """
    print(f"[📊] Loading future data from {min_timestamp}...")
    
    future_data = {}
    offset = 0
    batch_size = 2000
    
    while True:
        resp = client.table("technical_indicators") \
            .select("asset, ema, timestamp") \
            .gte("timestamp", min_timestamp) \
            .not_.is_("ema", "null") \
            .order("asset, timestamp") \
            .range(offset, offset + batch_size - 1) \
            .execute()
        
        data = resp.data
        if not data:
            break
        
        # Organize data by asset
        for record in data:
            try:
                normalized_asset = normalize_asset_name_labeling(record['asset'])
                ts = parse_timestamp_safe(record['timestamp'])
                if ts is None:
                    continue
                
                if normalized_asset not in future_data:
                    future_data[normalized_asset] = []
                
                future_data[normalized_asset].append({
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
    
    # Sort data by timestamp for each asset
    for asset in future_data:
        future_data[asset].sort(key=lambda x: x['ts'])
    
    total_records = sum(len(records) for records in future_data.values())
    print(f"[✅] Future data loaded: {len(future_data)} assets, {total_records} total records")
    
    return future_data

def parse_timestamp_safe(timestamp_str):
    """
    Safely parse timestamp string (supports multiple formats)
    
    Args:
        timestamp_str: Timestamp string
    
    Returns:
        datetime or None: Parsed datetime object
    """
    if not timestamp_str:
        return None
    
    # Remove Z
    clean_timestamp = timestamp_str.replace('Z', '')
    
    # Try multiple formats
    formats = [
        "%Y-%m-%d %H:%M:%S",   # 2025-07-26 20:16:01
        "%Y-%m-%d %H:%M",      # 2025-07-26 20:16
        "%Y-%m-%dT%H:%M:%S",   # 2025-07-26T20:16:01
        "%Y-%m-%dT%H:%M",      # 2025-07-26T20:16
        "%Y-%m-%d",            # 2025-07-26
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(clean_timestamp, fmt)
        except ValueError:
            continue
    
    # Try ISO format
    try:
        return datetime.fromisoformat(clean_timestamp)
    except ValueError:
        pass
    
    print(f"[⚠️] Failed to parse timestamp: {timestamp_str}")
    return None

def run_labeling():
    """
    Execute price_trend labeling pipeline
    """
    print("=== Labeling Pipeline Started ===")
    try:
        now = datetime.utcnow()
        initial_cutoff_time = (now - timedelta(hours=4)).isoformat()
        
        print(f"[⏰] Initial cutoff time: {initial_cutoff_time}")
        
        # First, get all data to determine the actual cutoff time
        all_unlabeled_data = []
        offset = 0
        
        # Get all unlabeled data up to the current time (to find future data)
        # Fetch all unlabeled data without timestamp filtering
        
        while True:
            resp = client.table("technical_indicators") \
                .select("id, asset, ema, timestamp") \
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
        
        # Filter data based on actual cutoff time
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
                    current_ema = float(record['ema'])
                    timestamp = parse_timestamp_safe(record['timestamp'])
                    if timestamp is None:
                        print(f"[❌] Failed to parse timestamp for record {record.get('id', 'unknown')}")
                        continue
                    
                    # Find 4-hour future EMA
                    future_ema = find_future_ema(normalized_asset, timestamp, future_data)
                    
                    # Price trend determination logic (same as SQL)
                    if future_ema is None:
                        # No data after 4 hours = down
                        if timestamp + timedelta(hours=4) <= now:
                            price_trend = 'down'
                        else:
                            # Still less than 4 hours = do not process
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