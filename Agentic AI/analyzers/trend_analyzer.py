# analyzers/trend_analyzer.py - EMA-based surge/crash analysis
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
from loguru import logger

from supabase import create_client
from config import (
    SUPABASE_URL, SUPABASE_KEY, SUPABASE_MARKET_TABLE,
    RECENT_DAYS, SURGE_THRESHOLD, CRASH_THRESHOLD, MAX_COINS_TO_ANALYZE
)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_ema_data(days: int = RECENT_DAYS) -> pd.DataFrame:
    """Collect EMA data"""
    since = datetime.utcnow() - timedelta(days=days)
    
    page_size = 1000
    page = 0
    all_records = []
    
    while True:
        start = page * page_size
        end = start + page_size - 1
        
        response = (
            supabase.table(SUPABASE_MARKET_TABLE)
            .select("coin, timestamp, ema")
            .gte("timestamp", since.isoformat())
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
        logger.warning("No EMA data available.")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["coin", "timestamp"])
    
    return df


def calculate_ema_change_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate EMA change rate"""
    if df.empty:
        return df
    
    # Calculate change rate by coin
    df = df.copy()
    df["ema_pct_change"] = df.groupby("coin")["ema"].pct_change() * 100
    
    # Remove first value (change rate cannot be calculated)
    df = df.dropna(subset=["ema_pct_change"])
    
    return df


def detect_surge_crash(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """Detect surge/crash coins"""
    if df.empty:
        return {"surge": [], "crash": []}
    
    # Surge coins (above threshold increase)
    surge_mask = df["ema_pct_change"] >= SURGE_THRESHOLD
    surge_coins = df[surge_mask].copy()
    
    # Crash coins (below threshold decrease)
    crash_mask = df["ema_pct_change"] <= CRASH_THRESHOLD
    crash_coins = df[crash_mask].copy()
    
    def format_coin_data(coin_df: pd.DataFrame) -> List[Dict]:
        result = []
        for _, row in coin_df.iterrows():
            result.append({
                "coin": row["coin"],
                "timestamp": row["timestamp"].isoformat(),
                "ema": row["ema"],
                "change_rate": round(row["ema_pct_change"], 2)
            })
        return result
    
    return {
        "surge": format_coin_data(surge_coins),
        "crash": format_coin_data(crash_coins)
    }


def get_top_trending_coins(trend_data: Dict, limit: int = MAX_COINS_TO_ANALYZE) -> Dict:
    """Extract coins with the largest change rates"""
    
    # Sort surge coins (by largest change rate)
    surge_sorted = sorted(
        trend_data["surge"], 
        key=lambda x: x["change_rate"], 
        reverse=True
    )[:limit]
    
    # Sort crash coins (by largest absolute change rate)
    crash_sorted = sorted(
        trend_data["crash"], 
        key=lambda x: abs(x["change_rate"]), 
        reverse=True
    )[:limit]
    
    return {
        "surge": surge_sorted,
        "crash": crash_sorted
    }


def analyze_coin_trends() -> Dict:
    """Comprehensive coin trend analysis"""
    try:
        logger.info("📊 Starting coin trend analysis...")
        
        # Collect EMA data
        ema_df = get_ema_data()
        if ema_df.empty:
            return {"error": "No EMA data available."}
        
        # Calculate change rates
        ema_df = calculate_ema_change_rate(ema_df)
        
        # Detect surge/crash
        trend_data = detect_surge_crash(ema_df)
        
        # Extract top coins
        top_trends = get_top_trending_coins(trend_data)
        
        # Statistics information
        stats = {
            "total_coins_analyzed": ema_df["coin"].nunique(),
            "surge_coins_count": len(trend_data["surge"]),
            "crash_coins_count": len(trend_data["crash"]),
            "analysis_period_days": RECENT_DAYS,
            "surge_threshold": SURGE_THRESHOLD,
            "crash_threshold": CRASH_THRESHOLD
        }
        
        logger.success(f"✅ Trend analysis completed: {stats['surge_coins_count']} surge, {stats['crash_coins_count']} crash")
        
        return {
            "trends": top_trends,
            "stats": stats,
            "raw_data": trend_data
        }
        
    except Exception as e:
        logger.error(f"❌ Trend analysis failed: {e}")
        return {"error": str(e)}


def format_trend_summary(analysis_result: Dict) -> str:
    """Format trend analysis results to text"""
    if "error" in analysis_result:
        return f"Trend analysis error: {analysis_result['error']}"
    
    trends = analysis_result.get("trends", {})
    stats = analysis_result.get("stats", {})
    
    summary_parts = []
    
    # Surge coins
    if trends.get("surge"):
        surge_list = []
        for coin_data in trends["surge"][:3]:  # Top 3 only
            surge_list.append(f"{coin_data['coin']} (+{coin_data['change_rate']}%)")
        summary_parts.append(f"🚀 Surge: {', '.join(surge_list)}")
    
    # Crash coins
    if trends.get("crash"):
        crash_list = []
        for coin_data in trends["crash"][:3]:  # Top 3 only
            crash_list.append(f"{coin_data['coin']} ({coin_data['change_rate']}%)")
        summary_parts.append(f"📉 Crash: {', '.join(crash_list)}")
    
    if not summary_parts:
        return f"📊 No significant changes in the last {stats.get('analysis_period_days', RECENT_DAYS)} days"
    
    return "\n".join(summary_parts) 