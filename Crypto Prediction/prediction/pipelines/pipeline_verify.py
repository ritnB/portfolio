# pipeline_verify.py

import pandas as pd
from datetime import datetime, timedelta
import os
from supabase import create_client
from dotenv import load_dotenv
from config import (
    SUPABASE_URL, SUPABASE_KEY, TECH_INDICATORS_TABLE, 
    VERIFY_MAX_RECORDS, VERIFY_BATCH_SIZE, normalize_coin_name
)

# 환경 변수 로드 및 Supabase 클라이언트 생성
load_dotenv()
client = create_client(SUPABASE_URL, SUPABASE_KEY)

def load_technical_data_paginated(limit=1000, coin=None):
    """
    Load technical indicators data using pagination.
    
    Args:
        limit: Maximum number of records per page
        coin: Filter by specific coin (None for all coins)
        
    Returns:
        list: Technical indicators data list
        
    Note: This is a simplified version for portfolio demonstration.
    """
    all_data = []
    offset = 0
    
    while True:
        query = client.table(TECH_INDICATORS_TABLE).select("*")
        
        # Apply coin filter if specified
        if coin:
            query = query.eq("coin", coin)
        
        response = query.range(offset, offset + limit - 1).execute()
        data = response.data
        
        if not data:
            break
            
        all_data.extend(data)
        
        # Stop if we got less than the batch size (no more data)
        if len(data) < limit:
            break
            
        offset += limit
    
    return all_data

def run_verification():
    """
    Run prediction verification against historical data.
    
    Note: This is a simplified verification process for portfolio demonstration.
    Actual verification contains proprietary accuracy calculation methods.
    """
    print("=== [📊] Verification Pipeline Starting ===")
    
    try:
        # Load recent predictions (paginated approach)
        predictions_data = []
        offset = 0
        
        while len(predictions_data) < VERIFY_MAX_RECORDS:
            batch_size = min(VERIFY_BATCH_SIZE, VERIFY_MAX_RECORDS - len(predictions_data))
            
            resp = client.table("predictions") \
                .select("*") \
                .eq("verified", False) \
                .order("timestamp", desc=True) \
                .range(offset, offset + batch_size - 1) \
                .execute()
            
            batch_data = resp.data
            if not batch_data:
                break
                
            predictions_data.extend(batch_data)
            offset += batch_size
            
            if len(batch_data) < batch_size:
                break
        
        if not predictions_data:
            return {"message": "No unverified predictions found"}, 200
        
        print(f"[📋] Processing {len(predictions_data)} predictions for verification")
        
        verified_count = 0
        accurate_count = 0
        
        for prediction in predictions_data:
            coin = prediction.get("coin")
            pred_timestamp = prediction.get("timestamp")
            predicted_trend = prediction.get("predicted_trend", "unknown")
            
            if not all([coin, pred_timestamp]):
                continue
            
            # Normalize coin name for consistency
            normalized_coin = normalize_coin_name(coin)
            
            # Get historical data for verification (simplified logic)
            verification_result = verify_single_prediction(
                normalized_coin, pred_timestamp, predicted_trend
            )
            
            if verification_result is not None:
                # Update prediction with verification result
                update_data = {
                    "verified": True,
                    "is_correct": verification_result,
                    "verified_at": datetime.utcnow().isoformat()
                }
                
                client.table("predictions") \
                    .update(update_data) \
                    .eq("id", prediction["id"]) \
                    .execute()
                
                verified_count += 1
                if verification_result:
                    accurate_count += 1
        
        accuracy = (accurate_count / verified_count * 100) if verified_count > 0 else 0
        
        result = {
            "message": "Verification completed",
            "total_verified": verified_count,
            "accurate_predictions": accurate_count,
            "accuracy_percentage": round(accuracy, 2)
        }
        
        print(f"[✅] Verification completed: {accurate_count}/{verified_count} accurate ({accuracy:.1f}%)")
        return result, 200
        
    except Exception as e:
        error_result = {"error": f"Verification failed: {str(e)}"}
        print(f"[❌] Verification error: {e}")
        return error_result, 500

def verify_single_prediction(coin, prediction_timestamp, predicted_trend):
    """
    Verify a single prediction against historical data.
    
    Note: This is a simplified verification method for portfolio demonstration.
    Actual verification logic is proprietary.
    """
    try:
        # Parse prediction timestamp
        pred_time = datetime.fromisoformat(prediction_timestamp.replace('Z', ''))
        
        # Look for actual data within a time window (simplified approach)
        start_time = pred_time + timedelta(hours=-2)
        end_time = pred_time + timedelta(hours=2)
        
        # Query historical data (simplified)
        historical_data = load_technical_data_paginated(limit=100, coin=coin)
        
        if not historical_data:
            # Try with normalized coin name
            historical_data = load_technical_data_paginated(limit=1000)
            # Filter for matching coin after normalization
            historical_data = [
                record for record in historical_data 
                if normalize_coin_name(record.get('coin', '')) == coin
            ]
        
        if not historical_data:
            return None
        
        # Find the closest data point within the time window
        closest_record = None
        min_time_diff = float('inf')
        
        for record in historical_data:
            record_time = datetime.fromisoformat(record['timestamp'].replace('Z', ''))
            time_diff = abs((record_time - pred_time).total_seconds())
            
            if start_time <= record_time <= end_time and time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_record = record
        
        if not closest_record:
            return None
        
        # Simplified verification logic (actual method is proprietary)
        actual_trend = closest_record.get('price_trend', 'unknown')
        
        if actual_trend == 'unknown':
            return None
        
        # Convert prediction format if needed
        if isinstance(predicted_trend, int):
            predicted_trend = 'up' if predicted_trend > 1 else 'down'
        
        return actual_trend == predicted_trend
        
    except Exception as e:
        print(f"[⚠️] Verification error for {coin}: {e}")
        return None

if __name__ == "__main__":
    result, status = run_verification()
    print(f"Result: {result}")
    print(f"Status: {status}")
