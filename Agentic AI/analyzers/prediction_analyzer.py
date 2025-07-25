# analyzers/prediction_analyzer.py - Prediction accuracy analysis
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
from loguru import logger

from supabase import create_client
from config import (
    SUPABASE_URL, SUPABASE_KEY, SUPABASE_PREDICT_TABLE,
    RECENT_DAYS, ACCURACY_THRESHOLD_3DAY, ACCURACY_THRESHOLD_1DAY
)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_prediction_data(days: int = RECENT_DAYS) -> pd.DataFrame:
    """Collect prediction data"""
    since = datetime.utcnow() - timedelta(days=days)
    
    page_size = 1000
    page = 0
    all_records = []
    
    while True:
        start = page * page_size
        end = start + page_size - 1
        
        response = (
            supabase.table(SUPABASE_PREDICT_TABLE)
            .select("coin, timestamp, pricetrend, finalscore, actual_trend, is_correct, verified")
            .gte("timestamp", since.isoformat())
            .eq("verified", True)  # Verified data only
            .order("timestamp", desc=False)
            .range(start, end)
            .execute()
        )
        
        page_data = response.data
        if not page_data:
            break
            
        all_records.extend(page_data)
        page += 1
    
    if not all_records:
        logger.warning("No prediction data available.")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["coin", "timestamp"])
    
    return df


def calculate_accuracy_rates(df: pd.DataFrame) -> Dict:
    """Calculate accuracy rates"""
    if df.empty:
        return {"error": "No prediction data available."}
    
    # Overall accuracy
    total_predictions = len(df)
    correct_predictions = len(df[df["is_correct"] == True])
    overall_accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    
    # Daily accuracy
    df["date"] = df["timestamp"].dt.date
    daily_accuracy = []
    
    for date, day_group in df.groupby("date"):
        day_total = len(day_group)
        day_correct = len(day_group[day_group["is_correct"] == True])
        day_accuracy = (day_correct / day_total * 100) if day_total > 0 else 0
        
        daily_accuracy.append({
            "date": date.isoformat(),
            "total": day_total,
            "correct": day_correct,
            "accuracy": round(day_accuracy, 2)
        })
    
    # Accuracy by coin
    coin_accuracy = []
    for coin, coin_group in df.groupby("coin"):
        coin_total = len(coin_group)
        coin_correct = len(coin_group[coin_group["is_correct"] == True])
        coin_acc = (coin_correct / coin_total * 100) if coin_total > 0 else 0
        
        coin_accuracy.append({
            "coin": coin,
            "total": coin_total,
            "correct": coin_correct,
            "accuracy": round(coin_acc, 2)
        })
    
    # Accuracy by trend prediction (up/down)
    trend_accuracy = {}
    for trend in ["up", "down"]:
        trend_data = df[df["pricetrend"] == trend]
        if not trend_data.empty:
            trend_total = len(trend_data)
            trend_correct = len(trend_data[trend_data["is_correct"] == True])
            trend_acc = (trend_correct / trend_total * 100) if trend_total > 0 else 0
            
            trend_accuracy[trend] = {
                "total": trend_total,
                "correct": trend_correct,
                "accuracy": round(trend_acc, 2)
            }
    
    return {
        "overall": {
            "total": total_predictions,
            "correct": correct_predictions,
            "accuracy": round(overall_accuracy, 2)
        },
        "daily": daily_accuracy,
        "by_coin": sorted(coin_accuracy, key=lambda x: x["accuracy"], reverse=True),
        "by_trend": trend_accuracy
    }


def find_successful_predictions(df: pd.DataFrame, trend_data: Dict = None) -> Dict:
    """Find successful predictions (matching with surge/crash)"""
    if df.empty:
        return {"successful_matches": []}
    
    # Filter only correct predictions
    successful_df = df[df["is_correct"] == True].copy()
    
    matches = []
    
    # Match with surge/crash data if available
    if trend_data and not successful_df.empty:
        # Match surge predictions
        for surge_coin in trend_data.get("surge", []):
            coin_predictions = successful_df[
                (successful_df["coin"] == surge_coin["coin"]) &
                (successful_df["pricetrend"] == "up")
            ]
            
            for _, pred in coin_predictions.iterrows():
                matches.append({
                    "type": "surge_prediction",
                    "coin": pred["coin"],
                    "predicted_trend": pred["pricetrend"],
                    "actual_trend": pred["actual_trend"],
                    "confidence": pred["finalscore"],
                    "timestamp": pred["timestamp"].isoformat(),
                    "trend_change": surge_coin["change_rate"]
                })
        
        # Match crash predictions
        for crash_coin in trend_data.get("crash", []):
            coin_predictions = successful_df[
                (successful_df["coin"] == crash_coin["coin"]) &
                (successful_df["pricetrend"] == "down")
            ]
            
            for _, pred in coin_predictions.iterrows():
                matches.append({
                    "type": "crash_prediction",
                    "coin": pred["coin"],
                    "predicted_trend": pred["pricetrend"],
                    "actual_trend": pred["actual_trend"],
                    "confidence": pred["finalscore"],
                    "timestamp": pred["timestamp"].isoformat(),
                    "trend_change": crash_coin["change_rate"]
                })
    
    # High confidence successful predictions (based on finalscore)
    high_confidence = successful_df[abs(successful_df["finalscore"]) >= 80].copy()
    
    high_conf_list = []
    for _, pred in high_confidence.iterrows():
        high_conf_list.append({
            "coin": pred["coin"],
            "predicted_trend": pred["pricetrend"],
            "confidence": pred["finalscore"],
            "timestamp": pred["timestamp"].isoformat()
        })
    
    return {
        "successful_matches": matches,
        "high_confidence_hits": high_conf_list
    }


def check_promotion_criteria(accuracy_data: Dict) -> Dict:
    """Check promotion criteria"""
    overall_acc = accuracy_data.get("overall", {}).get("accuracy", 0)
    daily_accs = accuracy_data.get("daily", [])
    
    # Check 3-day overall accuracy
    meets_3day_criteria = overall_acc >= ACCURACY_THRESHOLD_3DAY
    
    # Check daily accuracy (all days must meet criteria)
    daily_meets_criteria = []
    for day_data in daily_accs:
        meets_1day = day_data["accuracy"] >= ACCURACY_THRESHOLD_1DAY
        daily_meets_criteria.append({
            "date": day_data["date"],
            "accuracy": day_data["accuracy"],
            "meets_criteria": meets_1day
        })
    
    all_days_meet_criteria = all(day["meets_criteria"] for day in daily_meets_criteria)
    
    return {
        "can_promote_3day": meets_3day_criteria,
        "can_promote_1day": all_days_meet_criteria,
        "overall_accuracy": overall_acc,
        "daily_breakdown": daily_meets_criteria,
        "criteria": {
            "3day_threshold": ACCURACY_THRESHOLD_3DAY,
            "1day_threshold": ACCURACY_THRESHOLD_1DAY
        }
    }


def analyze_prediction_performance() -> Dict:
    """Comprehensive prediction performance analysis"""
    try:
        logger.info("🎯 Starting prediction performance analysis...")
        
        # Collect prediction data
        pred_df = get_prediction_data()
        if pred_df.empty:
            return {"error": "No prediction data available."}
        
        # Calculate accuracy rates
        accuracy_data = calculate_accuracy_rates(pred_df)
        
        # Check promotion criteria
        promotion_check = check_promotion_criteria(accuracy_data)
        
        # Find successful predictions
        successful_preds = find_successful_predictions(pred_df)
        
        logger.success(f"✅ Prediction analysis completed: overall accuracy {accuracy_data['overall']['accuracy']}%")
        
        return {
            "accuracy": accuracy_data,
            "promotion": promotion_check,
            "successful_predictions": successful_preds,
            "total_predictions": len(pred_df)
        }
        
    except Exception as e:
        logger.error(f"❌ Prediction analysis failed: {e}")
        return {"error": str(e)}


def format_prediction_summary(analysis_result: Dict) -> str:
    """Format prediction analysis results to text"""
    if "error" in analysis_result:
        return f"Prediction analysis error: {analysis_result['error']}"
    
    accuracy = analysis_result.get("accuracy", {})
    promotion = analysis_result.get("promotion", {})
    
    overall_acc = accuracy.get("overall", {}).get("accuracy", 0)
    
    summary_parts = [f"🎯 Prediction accuracy: {overall_acc}%"]
    
    # Promotion eligibility
    if promotion.get("can_promote_3day"):
        summary_parts.append(f"✅ 3-day overall accuracy criteria met ({ACCURACY_THRESHOLD_3DAY}%+ achieved)")
    
    if promotion.get("can_promote_1day"):
        summary_parts.append(f"✅ All daily accuracy criteria met ({ACCURACY_THRESHOLD_1DAY}%+ each day)")
    
    # Top performing coins
    top_coins = accuracy.get("by_coin", [])[:3]
    if top_coins:
        coin_list = [f"{coin['coin']} {coin['accuracy']}%" for coin in top_coins]
        summary_parts.append(f"🏆 Top performers: {', '.join(coin_list)}")
    
    return "\n".join(summary_parts) 