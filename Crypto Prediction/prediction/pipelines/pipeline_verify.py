from datetime import datetime, timedelta
import os
from supabase import create_client
from dotenv import load_dotenv
from config import normalize_coin_name, VERIFY_MAX_RECORDS, VERIFY_BATCH_SIZE
from utils.timestamp_utils import safe_parse_timestampz, normalize_timestamp_for_query

# Load environment and create Supabase client
load_dotenv()
client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


def load_technical_indicators_paged(coin=None, max_records=VERIFY_MAX_RECORDS, batch_size=VERIFY_BATCH_SIZE):
    """Load technical_indicators with pagination."""
    all_data = []
    offset = 0
    
    while len(all_data) < max_records:
        query = client.table("technical_indicators") \
            .select("timestamp, price_trend, coin") \
            .order("timestamp", desc=True) \
            .range(offset, offset + batch_size - 1)
        
        if coin:
            query = query.eq("coin", coin)
        
        resp = query.execute()
        
        if not resp.data:
            break
        
        all_data.extend(resp.data)
        
        if len(resp.data) < batch_size:
            break
            
        offset += batch_size
        
        print(f"[📦] Loaded {len(all_data)} records so far...")
    
    if len(all_data) > max_records:
        all_data = all_data[:max_records]
    
    return all_data


def parse_prediction_timestamp(timestamp_input):
    """Parse timestampz from predictions (v11_2 compatible)."""
    try:
        return safe_parse_timestampz(timestamp_input)
    except Exception as e:
        print(f"[⚠️] Failed to parse prediction timestampz: {timestamp_input}, error: {e}")
        return None


def parse_technical_timestamp(timestamp_input):
    """Parse timestampz from technical_indicators (v11_2 compatible)."""
    try:
        return safe_parse_timestampz(timestamp_input)
    except Exception as e:
        print(f"[⚠️] Failed to parse technical timestampz: {timestamp_input}, error: {e}")
        return None


def run_verification():
    print("=== Verification Pipeline Started ===")
    try:
        now = datetime.utcnow()
        cutoff_time_str = (now - timedelta(hours=5)).isoformat()
        cutoff_time_ts = now - timedelta(hours=5)

        # Load predictions in pages of 1000
        predictions = []
        page = 0
        while True:
            resp = client.table("predictions") \
                .select("*") \
                .or_("verified.is.false,verified.is.null") \
                .lte("timestamp", cutoff_time_ts.isoformat() + "Z") \
                .range(page * 1000, (page + 1) * 1000 - 1) \
                .execute()
            
            data = resp.data
            if not data:
                break
            predictions.extend(data)
            page += 1

        if not predictions:
            return {"message": "No predictions to verify"}, 200

        updated = 0
        parse_errors = 0
        batch_updates = []
        
        # Cache normalized coin names
        coin_normalization_cache = {}
        def get_normalized_coin(original_coin):
            if original_coin not in coin_normalization_cache:
                coin_normalization_cache[original_coin] = normalize_coin_name(original_coin)
            return coin_normalization_cache[original_coin]
        
        # Compute prediction time range
        prediction_times = []
        for row in predictions:
            pred_time = parse_prediction_timestamp(row["timestamp"])
            if pred_time:
                prediction_times.append(pred_time)
        
        if not prediction_times:
            return {"message": "No valid prediction timestamps"}, 200
        
        min_pred_time = min(prediction_times)
        max_pred_time = max(prediction_times)
        
        # Load a slightly broader time window for matching
        time_buffer = timedelta(hours=3)
        min_load_time = min_pred_time - time_buffer
        max_load_time = max_pred_time + time_buffer
        
        # minimal logging
        
        all_technical_data = load_technical_indicators_paged(coin=None)
        
        technical_by_coin = {}
        filtered_count = 0
        
        for entry in all_technical_data:
            ts = parse_technical_timestamp(entry["timestamp"])
            if ts is None or ts < min_load_time or ts > max_load_time:
                continue
            
            filtered_count += 1
            normalized_coin = get_normalized_coin(entry["coin"])
            
            if normalized_coin not in technical_by_coin:
                technical_by_coin[normalized_coin] = []
            
            technical_by_coin[normalized_coin].append((ts, entry))
        
        for coin in technical_by_coin:
            technical_by_coin[coin].sort(key=lambda x: x[0])
        
        print(f"[✅] Loaded {filtered_count}/{len(all_technical_data)} technical indicators for {len(technical_by_coin)} coins (time-filtered)")
        
        for row in predictions:
            try:
                original_coin = row["coin"]
                coin = get_normalized_coin(original_coin)
                predicted_trend = row.get("pricetrend")
                
                prediction_time = parse_prediction_timestamp(row["timestamp"])
                if prediction_time is None:
                    parse_errors += 1
                    continue

                print(f"[🔍] Processing {original_coin} -> {coin} - Prediction time: {prediction_time}")

                coin_time_data = technical_by_coin.get(coin, [])
                print(f"[📊] Found {len(coin_time_data)} technical indicators for {coin}")
                
                actual_trend = None
                min_diff = timedelta.max
                valid_entries = len(coin_time_data)
                within_time_range = 0
                has_matching_data = False
                
                time_threshold = timedelta(hours=2)
                search_start = prediction_time - time_threshold
                search_end = prediction_time + time_threshold
                
                for ts, entry in coin_time_data:
                    if ts < search_start:
                        continue
                    if ts > search_end:
                        break
                    
                    diff = abs(ts - prediction_time)
                    within_time_range += 1
                    has_matching_data = True
                    trend = entry.get("price_trend")
                # minimal logging
                    
                    if trend in ["up", "down"] and diff < min_diff:
                        actual_trend = trend
                        min_diff = diff
                        # minimal logging

                print(f"[📈] Valid entries: {valid_entries}, Within 2h: {within_time_range}, Final trend: {actual_trend}")

                if actual_trend is None:
                    if has_matching_data:
                        print(f"[⏸️] Matching data found but price_trend is NULL, skipping verification")
                        continue
                    else:
                        actual_trend = "down"
                        print(f"[⚠️] No matching data found within 2 hours, defaulting to 'down'")
                
                is_correct = (predicted_trend == actual_trend)

                batch_updates.append({
                    "coin": original_coin,
                    "timestamp": row["timestamp"],
                    "actual_trend": actual_trend,
                    "is_correct": is_correct
                })

                updated += 1
                print(f"[📝] Queued {original_coin}: predicted={predicted_trend}, actual={actual_trend}, correct={is_correct}")

            except Exception as row_error:
                print(f"[❌ Verification Error] {row_error}")
                continue

        if batch_updates:
            print(f"[💾] Executing batch updates for {len(batch_updates)} predictions...")
            try:
                for update in batch_updates:
                    client.table("predictions").update({
                        "actual_trend": update["actual_trend"],
                        "is_correct": update["is_correct"],
                        "verified": True
                    }).eq("coin", update["coin"]).eq("timestamp", update["timestamp"]).execute()
                print(f"[✅] Batch updates completed successfully")
            except Exception as e:
                print(f"[❌] Batch updates failed: {e}")

        message = f"Verification finished. Updated: {updated}"
        if parse_errors > 0:
            message += f", Parse errors: {parse_errors}"
            
        print(f"=== Verification Pipeline Finished. {message} ===")
        return {"message": message}, 200

    except Exception as e:
        print(f"[💥 Verification Pipeline Error] {e}")
        return {"error": str(e)}, 500
