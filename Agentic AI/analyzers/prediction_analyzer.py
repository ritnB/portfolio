# analyzers/prediction_analyzer.py - AI Prediction Performance Analysis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
from supabase import create_client
from config import (
    SUPABASE_URL, SUPABASE_KEY, SUPABASE_PREDICT_TABLE,
    ACCURACY_THRESHOLD_3DAY, ACCURACY_THRESHOLD_1DAY, PREDICTION_CONFIDENCE_THRESHOLD
)

def analyze_prediction_accuracy(target_coins: list = None) -> dict:
    """Analyze AI prediction accuracy and performance"""
    try:
        # Initialize Supabase client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Get recent prediction data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        # Query prediction data
        response = supabase.table(SUPABASE_PREDICT_TABLE).select("*").gte(
            "timestamp", start_date.isoformat()
        ).lte("timestamp", end_date.isoformat()).execute()
        
        if not response.data:
            return {"error": "No prediction data available"}
        
        # Convert to DataFrame
        df = pd.DataFrame(response.data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter target coins if specified
        if target_coins:
            df = df[df['coin'].isin(target_coins)]
        
        if df.empty:
            return {"error": "No prediction data for specified coins"}
        
        # Overall accuracy
        total_predictions = len(df)
        correct_predictions = len(df[df['is_correct'] == True])
        overall_accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        
        # Daily accuracy
        df['date'] = df['timestamp'].dt.date
        daily_accuracy = df.groupby('date').apply(
            lambda x: (x['is_correct'].sum() / len(x) * 100) if len(x) > 0 else 0
        ).to_dict()
        
        # Coin-specific accuracy
        coin_accuracy = {}
        for coin in df['coin'].unique():
            coin_data = df[df['coin'] == coin]
            if len(coin_data) > 0:
                correct = len(coin_data[coin_data['is_correct'] == True])
                accuracy = (correct / len(coin_data) * 100)
                coin_accuracy[coin] = {
                    "accuracy": round(accuracy, 2),
                    "total_predictions": len(coin_data),
                    "correct_predictions": correct
                }
        
        # Direction-specific accuracy (up/down predictions)
        up_predictions = df[df['predicted_direction'] == 'up']
        down_predictions = df[df['predicted_direction'] == 'down']
        
        up_accuracy = (len(up_predictions[up_predictions['is_correct'] == True]) / len(up_predictions) * 100) if len(up_predictions) > 0 else 0
        down_accuracy = (len(down_predictions[down_predictions['is_correct'] == True]) / len(down_predictions) * 100) if len(down_predictions) > 0 else 0
        
        return {
            "overall_accuracy": round(overall_accuracy, 2),
            "total_predictions": total_predictions,
            "correct_predictions": correct_predictions,
            "daily_accuracy": daily_accuracy,
            "coin_accuracy": coin_accuracy,
            "direction_accuracy": {
                "up_accuracy": round(up_accuracy, 2),
                "down_accuracy": round(down_accuracy, 2),
                "up_predictions": len(up_predictions),
                "down_predictions": len(down_predictions)
            }
        }
        
    except Exception as e:
        logger.error(f"Prediction analysis failed: {e}")
        return {"error": str(e)}

def find_successful_predictions(target_coins: list = None) -> dict:
    """Find successful predictions with high confidence"""
    try:
        # Initialize Supabase client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Get recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)
        
        # Query successful predictions
        response = supabase.table(SUPABASE_PREDICT_TABLE).select("*").gte(
            "timestamp", start_date.isoformat()
        ).lte("timestamp", end_date.isoformat()).eq("is_correct", True).gte(
            "confidence", PREDICTION_CONFIDENCE_THRESHOLD
        ).execute()
        
        if not response.data:
            return {"error": "No successful predictions found"}
        
        # Convert to DataFrame
        df = pd.DataFrame(response.data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter target coins if specified
        if target_coins:
            df = df[df['coin'].isin(target_coins)]
        
        # Match with surge/crash data if available
        successful_predictions = []
        for _, row in df.iterrows():
            prediction = {
                "coin": row['coin'],
                "predicted_direction": row['predicted_direction'],
                "confidence": row['confidence'],
                "timestamp": row['timestamp'].isoformat(),
                "actual_change": row.get('actual_change', 0)
            }
            successful_predictions.append(prediction)
        
        # Sort by confidence
        successful_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            "successful_predictions": successful_predictions,
            "total_successful": len(successful_predictions),
            "average_confidence": np.mean([p['confidence'] for p in successful_predictions]) if successful_predictions else 0
        }
        
    except Exception as e:
        logger.error(f"Successful predictions analysis failed: {e}")
        return {"error": str(e)}

def check_promotion_criteria(accuracy_data: dict) -> dict:
    """Check if prediction accuracy meets promotion criteria"""
    try:
        overall_accuracy = accuracy_data.get("overall_accuracy", 0)
        direction_accuracy = accuracy_data.get("direction_accuracy", {})
        
        # 3-day overall criteria
        meets_3day_criteria = overall_accuracy >= ACCURACY_THRESHOLD_3DAY
        
        # 1-day direction criteria
        up_accuracy = direction_accuracy.get("up_accuracy", 0)
        down_accuracy = direction_accuracy.get("down_accuracy", 0)
        meets_1day_criteria = (up_accuracy >= ACCURACY_THRESHOLD_1DAY or 
                              down_accuracy >= ACCURACY_THRESHOLD_1DAY)
        
        return {
            "meets_3day_criteria": meets_3day_criteria,
            "meets_1day_criteria": meets_1day_criteria,
            "overall_accuracy": overall_accuracy,
            "up_accuracy": up_accuracy,
            "down_accuracy": down_accuracy,
            "promotion_eligible": meets_3day_criteria or meets_1day_criteria
        }
        
    except Exception as e:
        logger.error(f"Promotion criteria check failed: {e}")
        return {"error": str(e)}

def get_prediction_analysis(target_coins: list = None) -> dict:
    """Get comprehensive prediction analysis"""
    try:
        # Analyze prediction accuracy
        accuracy_result = analyze_prediction_accuracy(target_coins)
        
        if "error" in accuracy_result:
            return accuracy_result
        
        # Find successful predictions
        success_result = find_successful_predictions(target_coins)
        
        # Check promotion criteria
        promotion_result = check_promotion_criteria(accuracy_result)
        
        return {
            "accuracy_analysis": accuracy_result,
            "successful_predictions": success_result,
            "promotion_criteria": promotion_result
        }
        
    except Exception as e:
        logger.error(f"Prediction analysis failed: {e}")
        return {"error": str(e)}

def format_prediction_summary(result: dict) -> str:
    """Format prediction analysis results as summary"""
    if "error" in result:
        return f"Prediction analysis failed: {result['error']}"
    
    accuracy_analysis = result.get("accuracy_analysis", {})
    promotion_criteria = result.get("promotion_criteria", {})
    successful_predictions = result.get("successful_predictions", {})
    
    summary_parts = []
    
    # Overall accuracy
    overall_accuracy = accuracy_analysis.get("overall_accuracy", 0)
    summary_parts.append(f"📊 Overall accuracy: {overall_accuracy:.1f}%")
    
    # Direction accuracy
    direction_accuracy = accuracy_analysis.get("direction_accuracy", {})
    up_accuracy = direction_accuracy.get("up_accuracy", 0)
    down_accuracy = direction_accuracy.get("down_accuracy", 0)
    summary_parts.append(f"📈 Up predictions: {up_accuracy:.1f}%")
    summary_parts.append(f"📉 Down predictions: {down_accuracy:.1f}%")
    
    # Promotion eligibility
    if promotion_criteria.get("promotion_eligible", False):
        summary_parts.append("✅ Meets promotion criteria")
    else:
        summary_parts.append("❌ Does not meet promotion criteria")
    
    # Successful predictions
    total_successful = successful_predictions.get("total_successful", 0)
    if total_successful > 0:
        avg_confidence = successful_predictions.get("average_confidence", 0)
        summary_parts.append(f"🎯 Successful predictions: {total_successful} (avg confidence: {avg_confidence:.1f}%)")
    
    return "\n".join(summary_parts) 