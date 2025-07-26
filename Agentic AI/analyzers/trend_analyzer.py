# analyzers/trend_analyzer.py - EMA-based Surge/Crash Analysis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
from supabase import create_client
from config import (
    SUPABASE_URL, SUPABASE_KEY, SUPABASE_MARKET_TABLE,
    SURGE_THRESHOLD, CRASH_THRESHOLD, RECENT_DAYS, TREND_TOP_COUNT
)

def calculate_ema(prices: list, period: int = 12) -> list:
    """Calculate Exponential Moving Average"""
    if len(prices) < period:
        return prices
    
    ema_values = []
    multiplier = 2 / (period + 1)
    
    # First EMA is SMA
    sma = sum(prices[:period]) / period
    ema_values.append(sma)
    
    # Calculate EMA for remaining values
    for price in prices[period:]:
        ema = (price * multiplier) + (ema_values[-1] * (1 - multiplier))
        ema_values.append(ema)
    
    return ema_values

def analyze_trends(target_coins: list = None) -> dict:
    """Analyze cryptocurrency market trends using EMA"""
    try:
        # Initialize Supabase client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Get recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=RECENT_DAYS)
        
        # Query market data
        response = supabase.table(SUPABASE_MARKET_TABLE).select("*").gte(
            "timestamp", start_date.isoformat()
        ).lte("timestamp", end_date.isoformat()).execute()
        
        if not response.data:
            return {"error": "No market data available"}
        
        # Convert to DataFrame
        df = pd.DataFrame(response.data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['coin', 'timestamp'])
        
        # Filter target coins if specified
        if target_coins:
            df = df[df['coin'].isin(target_coins)]
        
        # Get unique coins
        coins = df['coin'].unique()
        
        surge_coins = []
        crash_coins = []
        
        for coin in coins:
            coin_data = df[df['coin'] == coin].copy()
            
            if len(coin_data) < 2:
                continue
            
            # Calculate EMA
            prices = coin_data['price'].tolist()
            ema_values = calculate_ema(prices)
            
            # Calculate change rate
            if len(ema_values) >= 2:
                current_ema = ema_values[-1]
                previous_ema = ema_values[-2]
                change_rate = ((current_ema - previous_ema) / previous_ema) * 100
                
                coin_info = {
                    "coin": coin,
                    "current_price": prices[-1],
                    "ema": current_ema,
                    "change_rate": change_rate,
                    "timestamp": coin_data['timestamp'].iloc[-1].isoformat()
                }
                
                # Categorize as surge or crash
                if change_rate >= SURGE_THRESHOLD:
                    surge_coins.append(coin_info)
                elif change_rate <= CRASH_THRESHOLD:
                    crash_coins.append(coin_info)
        
        # Sort by change rate
        top_surge = sorted(surge_coins, key=lambda x: x["change_rate"], reverse=True)[:TREND_TOP_COUNT]
        top_crash = sorted(crash_coins, key=lambda x: abs(x["change_rate"]), reverse=True)[:TREND_TOP_COUNT]
        
        return {
            "surge_coins": top_surge,
            "crash_coins": top_crash,
            "total_analyzed": len(coins),
            "surge_count": len(surge_coins),
            "crash_count": len(crash_coins)
        }
        
    except Exception as e:
        logger.error(f"Trend analysis failed: {e}")
        return {"error": str(e)}

def get_trend_analysis(target_coins: list = None) -> dict:
    """Get comprehensive trend analysis"""
    try:
        # Collect EMA data
        result = analyze_trends(target_coins)
        
        if "error" in result:
            return result
        
        # Calculate change rates
        surge_coins = result.get("surge_coins", [])
        crash_coins = result.get("crash_coins", [])
        
        # Detect surges and crashes
        trending_coins = surge_coins + crash_coins
        
        # Top trending coins
        top_trending = sorted(trending_coins, key=lambda x: abs(x["change_rate"]), reverse=True)[:TREND_TOP_COUNT]
        
        # Statistics
        stats = {
            "total_analyzed": result.get("total_analyzed", 0),
            "surge_count": len(surge_coins),
            "crash_count": len(crash_coins),
            "trending_count": len(trending_coins)
        }
        
        return {
            "surge_coins": surge_coins,
            "crash_coins": crash_coins,
            "top_trending": top_trending,
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Trend analysis failed: {e}")
        return {"error": str(e)}

def format_trend_summary(result: dict) -> str:
    """Format trend analysis results as summary"""
    if "error" in result:
        return f"Trend analysis failed: {result['error']}"
    
    surge_coins = result.get("surge_coins", [])
    crash_coins = result.get("crash_coins", [])
    stats = result.get("statistics", {})
    
    summary_parts = []
    
    # Surge coins
    if surge_coins:
        summary_parts.append("🚀 Surging coins:")
        for coin in surge_coins[:3]:
            summary_parts.append(f"  • {coin['coin']}: +{coin['change_rate']:.1f}%")
    
    # Crash coins
    if crash_coins:
        summary_parts.append("📉 Crashing coins:")
        for coin in crash_coins[:3]:
            summary_parts.append(f"  • {coin['coin']}: {coin['change_rate']:.1f}%")
    
    # Statistics
    summary_parts.append(f"📊 Analyzed {stats.get('total_analyzed', 0)} coins")
    summary_parts.append(f"  • Surging: {stats.get('surge_count', 0)}")
    summary_parts.append(f"  • Crashing: {stats.get('crash_count', 0)}")
    
    return "\n".join(summary_parts) 