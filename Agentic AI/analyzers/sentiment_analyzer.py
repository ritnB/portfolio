# analyzers/sentiment_analyzer.py - Community sentiment analysis
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import re
from loguru import logger

from supabase import create_client
from config import (
    SUPABASE_URL, SUPABASE_KEY, SUPABASE_COMMUNITY_TABLE,
    RECENT_DAYS, MAX_COINS_TO_ANALYZE
)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_community_data(days: int = RECENT_DAYS, target_coins: List[str] = None) -> pd.DataFrame:
    """Collect community comment data"""
    since = datetime.utcnow() - timedelta(days=days)
    
    page_size = 1000
    page = 0
    all_records = []
    
    while True:
        start = page * page_size
        end = start + page_size - 1
        
        query = (
            supabase.table(SUPABASE_COMMUNITY_TABLE)
            .select("coin, timestamp, text, sentimental")
            .gte("timestamp", since.isoformat())
            .order("timestamp", desc=False)
            .range(start, end)
        )
        
        # Filter specific coins
        if target_coins:
            query = query.in_("coin", target_coins)
        
        response = query.execute()
        page_data = response.data
        
        if not page_data:
            break
            
        all_records.extend(page_data)
        page += 1
    
    if not all_records:
        logger.warning("No community data available.")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["coin", "timestamp"])
    
    return df


def extract_trending_topics(df: pd.DataFrame) -> Dict:
    """Extract trending topics"""
    if df.empty:
        return {"topics": [], "keywords": []}
    
    # Text preprocessing
    all_text = " ".join(df["text"].dropna().astype(str))
    
    # Simple keyword extraction (regex-based)
    # Cryptocurrency-related keywords (Korean patterns)
    crypto_keywords = [
        r'급등\w*', r'급락\w*', r'상승\w*', r'하락\w*', r'폭등\w*', r'폭락\w*',
        r'ATH\w*', r'신고가\w*', r'바닥\w*', r'저점\w*', r'고점\w*',
        r'강세\w*', r'약세\w*', r'불장\w*', r'곰장\w*', r'횡보\w*',
        r'돌파\w*', r'지지\w*', r'저항\w*', r'반등\w*', r'반락\w*'
    ]
    
    found_keywords = []
    for pattern in crypto_keywords:
        matches = re.findall(pattern, all_text, re.IGNORECASE)
        found_keywords.extend(matches)
    
    # Calculate keyword frequency
    keyword_counts = {}
    for keyword in found_keywords:
        keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
    
    # Extract top keywords
    top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Coin mention frequency
    coin_mentions = df["coin"].value_counts().head(10).to_dict()
    
    return {
        "keywords": [{"word": word, "count": count} for word, count in top_keywords],
        "coin_mentions": coin_mentions,
        "total_comments": len(df)
    }


def analyze_coin_sentiment(df: pd.DataFrame, target_coins: List[str] = None) -> Dict:
    """Analyze sentiment by coin"""
    if df.empty:
        return {"coin_sentiments": []}
    
    coins_to_analyze = target_coins if target_coins else df["coin"].unique()[:MAX_COINS_TO_ANALYZE]
    
    coin_sentiments = []
    
    for coin in coins_to_analyze:
        coin_data = df[df["coin"] == coin]
        
        if coin_data.empty:
            continue
        
        # Basic statistics
        total_comments = len(coin_data)
        
        # Sentiment analysis (using sentimental column)
        sentiment_counts = coin_data["sentimental"].value_counts().to_dict()
        
        # Recent comments
        recent_comments = coin_data.tail(5)["text"].tolist()
        
        # Positive/negative keyword count (Korean keywords)
        positive_keywords = ['좋다', '상승', '급등', '강세', '불장', '호재', '대박']
        negative_keywords = ['나쁘다', '하락', '급락', '약세', '곰장', '악재', '망했']
        
        all_text = " ".join(coin_data["text"].dropna().astype(str))
        positive_count = sum(all_text.lower().count(word) for word in positive_keywords)
        negative_count = sum(all_text.lower().count(word) for word in negative_keywords)
        
        # Calculate sentiment score (-1 ~ 1)
        total_sentiment_words = positive_count + negative_count
        sentiment_score = 0
        if total_sentiment_words > 0:
            sentiment_score = (positive_count - negative_count) / total_sentiment_words
        
        coin_sentiments.append({
            "coin": coin,
            "total_comments": total_comments,
            "sentiment_score": round(sentiment_score, 2),
            "positive_signals": positive_count,
            "negative_signals": negative_count,
            "recent_comments": recent_comments[-3:],  # Last 3 only
            "sentiment_distribution": sentiment_counts
        })
    
    # Sort by sentiment score
    coin_sentiments.sort(key=lambda x: x["sentiment_score"], reverse=True)
    
    return {"coin_sentiments": coin_sentiments}


def find_hot_discussions(df: pd.DataFrame) -> Dict:
    """Find hot discussions"""
    if df.empty:
        return {"hot_topics": []}
    
    # Coins with high comment counts
    active_coins = df["coin"].value_counts().head(5)
    
    hot_topics = []
    
    for coin, comment_count in active_coins.items():
        coin_data = df[df["coin"] == coin]
        
        # Extract keywords from recent comments
        recent_texts = coin_data.tail(10)["text"].tolist()
        combined_text = " ".join([str(text) for text in recent_texts if text])
        
        # Check urgency keywords (Korean)
        urgency_keywords = ['급등', '급락', '폭등', '폭락', '대박', '망했', '신고가', '바닥']
        urgency_count = sum(combined_text.count(keyword) for keyword in urgency_keywords)
        
        if urgency_count > 0:  # Only if urgency keywords exist
            hot_topics.append({
                "coin": coin,
                "comment_count": comment_count,
                "urgency_signals": urgency_count,
                "sample_comments": recent_texts[-2:],  # Last 2 comments
                "activity_level": "high" if comment_count > 10 else "medium"
            })
    
    # Sort by urgency signals
    hot_topics.sort(key=lambda x: x["urgency_signals"], reverse=True)
    
    return {"hot_topics": hot_topics}


def analyze_community_sentiment(target_coins: List[str] = None) -> Dict:
    """Comprehensive community sentiment analysis"""
    try:
        logger.info("💬 Starting community sentiment analysis...")
        
        # Collect community data
        comm_df = get_community_data(target_coins=target_coins)
        if comm_df.empty:
            return {"error": "No community data available."}
        
        # Extract trending topics
        trending = extract_trending_topics(comm_df)
        
        # Analyze sentiment by coin
        sentiments = analyze_coin_sentiment(comm_df, target_coins)
        
        # Find hot discussions
        hot_discussions = find_hot_discussions(comm_df)
        
        logger.success(f"✅ Sentiment analysis completed: {len(sentiments['coin_sentiments'])} coins analyzed")
        
        return {
            "trending": trending,
            "sentiments": sentiments,
            "hot_discussions": hot_discussions,
            "analysis_period_days": RECENT_DAYS
        }
        
    except Exception as e:
        logger.error(f"❌ Sentiment analysis failed: {e}")
        return {"error": str(e)}


def format_sentiment_summary(analysis_result: Dict, target_coins: List[str] = None) -> str:
    """Format sentiment analysis results to text"""
    if "error" in analysis_result:
        return f"Sentiment analysis error: {analysis_result['error']}"
    
    sentiments = analysis_result.get("sentiments", {}).get("coin_sentiments", [])
    hot_topics = analysis_result.get("hot_discussions", {}).get("hot_topics", [])
    trending = analysis_result.get("trending", {})
    
    summary_parts = []
    
    # Top sentiment score coins
    if sentiments:
        top_positive = [s for s in sentiments if s["sentiment_score"] > 0][:3]
        if top_positive:
            pos_list = [f"{s['coin']} ({s['sentiment_score']:+.2f})" for s in top_positive]
            summary_parts.append(f"😊 Positive: {', '.join(pos_list)}")
        
        top_negative = [s for s in sentiments if s["sentiment_score"] < 0][:3]
        if top_negative:
            neg_list = [f"{s['coin']} ({s['sentiment_score']:+.2f})" for s in top_negative]
            summary_parts.append(f"😰 Negative: {', '.join(neg_list)}")
    
    # Hot discussion coins
    if hot_topics:
        hot_list = [f"{topic['coin']} ({topic['comment_count']} comments)" for topic in hot_topics[:3]]
        summary_parts.append(f"🔥 Hot discussions: {', '.join(hot_list)}")
    
    # Trending keywords
    top_keywords = trending.get("keywords", [])[:3]
    if top_keywords:
        keyword_list = [f"{kw['word']} ({kw['count']} times)" for kw in top_keywords]
        summary_parts.append(f"📈 Keywords: {', '.join(keyword_list)}")
    
    if not summary_parts:
        return f"💬 No significant community reactions in the last {analysis_result.get('analysis_period_days', RECENT_DAYS)} days"
    
    return "\n".join(summary_parts) 