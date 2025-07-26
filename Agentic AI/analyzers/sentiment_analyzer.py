# analyzers/sentiment_analyzer.py - Community Sentiment Analysis
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from collections import Counter
from loguru import logger
from supabase import create_client
from config import (
    SUPABASE_URL, SUPABASE_KEY, SUPABASE_COMMUNITY_TABLE,
    CRYPTO_KEYWORDS, POSITIVE_KEYWORDS, NEGATIVE_KEYWORDS, URGENCY_KEYWORDS,
    SENTIMENT_TOP_COUNT, HOT_TOPICS_TOP_COUNT, KEYWORDS_TOP_COUNT, DB_PAGE_SIZE
)

def extract_keywords(text: str) -> list:
    """Extract cryptocurrency-related keywords from text"""
    if not text:
        return []
    
    # Text preprocessing
    text = text.lower()
    
    # Simple keyword extraction (regex-based)
    # Cryptocurrency-related keywords (English)
    found_keywords = []
    for pattern in CRYPTO_KEYWORDS:
        matches = re.findall(pattern, text)
        found_keywords.extend(matches)
    
    # Count keyword frequency
    keyword_counts = {}
    for keyword in found_keywords:
        keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
    
    # Extract top keywords
    top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:KEYWORDS_TOP_COUNT]
    
    return {
        "keywords": [{"word": word, "count": count} for word, count in top_keywords],
        "total_keywords": len(found_keywords)
    }

def analyze_coin_sentiment(comments: list) -> dict:
    """Analyze sentiment for a specific coin"""
    if not comments:
        return {"sentiment_score": 0, "comment_count": 0}
    
    # Basic statistics
    comment_count = len(comments)
    
    # Sentiment analysis (using sentimental column)
    sentiment_scores = [comment.get("sentimental", 0) for comment in comments if comment.get("sentimental") is not None]
    
    # Recent comments
    recent_comments = sorted(comments, key=lambda x: x.get("timestamp", ""), reverse=True)
    
    # Positive/negative keyword count (English)
    all_text = " ".join([comment.get("content", "") for comment in comments])
    positive_count = sum(all_text.lower().count(word) for word in POSITIVE_KEYWORDS)
    negative_count = sum(all_text.lower().count(word) for word in NEGATIVE_KEYWORDS)
    
    # Calculate sentiment score (-1 to 1)
    if sentiment_scores:
        avg_sentiment = np.mean(sentiment_scores)
    else:
        # Fallback to keyword-based sentiment
        total_keywords = positive_count + negative_count
        if total_keywords > 0:
            avg_sentiment = (positive_count - negative_count) / total_keywords
        else:
            avg_sentiment = 0
    
    return {
        "sentiment_score": round(avg_sentiment, 3),
        "comment_count": comment_count,
        "positive_keywords": positive_count,
        "negative_keywords": negative_count,
        "recent_comments": recent_comments[-3:],  # Last 3 only
        "sentiment_scores": sentiment_scores
    }

def find_hot_discussions(comments: list) -> dict:
    """Find hot discussion topics"""
    if not comments:
        return {"hot_topics": [], "urgency_signals": []}
    
    # Active coins (by comment count)
    coin_counts = Counter([comment.get("coin", "unknown") for comment in comments])
    active_coins = coin_counts.most_common(5)
    
    # Extract keywords from recent comments
    recent_texts = [comment.get("content", "") for comment in comments[-10:]]
    combined_text = " ".join(recent_texts).lower()
    
    # Check urgency keywords (English)
    urgency_count = sum(combined_text.count(keyword) for keyword in URGENCY_KEYWORDS)
    
    hot_topics = []
    if urgency_count > 0:  # Only if urgency keywords are present
        hot_topics.append({
            "topic": "Market urgency detected",
            "urgency_signals": urgency_count,
            "sample_comments": recent_texts[-2:],  # Last 2 comments
            "type": "urgency"
        })
    
    # Sort by urgency signal
    hot_topics.sort(key=lambda x: x["urgency_signals"], reverse=True)
    
    return {
        "hot_topics": hot_topics,
        "active_coins": [{"coin": coin, "count": count} for coin, count in active_coins]
    }

def analyze_community_sentiment(target_coins: list = None) -> dict:
    """Analyze community sentiment and mood"""
    try:
        # Initialize Supabase client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Get recent community data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)
        
        # Query community data
        response = supabase.table(SUPABASE_COMMUNITY_TABLE).select("*").gte(
            "timestamp", start_date.isoformat()
        ).lte("timestamp", end_date.isoformat()).execute()
        
        if not response.data:
            return {"error": "No community data available"}
        
        # Convert to DataFrame
        df = pd.DataFrame(response.data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter target coins if specified
        if target_coins:
            df = df[df['coin'].isin(target_coins)]
        
        # Extract trending topics
        trending = extract_keywords(" ".join(df['content'].fillna("")))
        
        # Analyze sentiment by coin
        coin_sentiments = []
        for coin in df['coin'].unique():
            coin_comments = df[df['coin'] == coin].to_dict('records')
            sentiment = analyze_coin_sentiment(coin_comments)
            sentiment['coin'] = coin
            coin_sentiments.append(sentiment)
        
        # Find hot discussions
        hot_discussions = find_hot_discussions(df.to_dict('records'))
        
        # Sort by sentiment score
        coin_sentiments.sort(key=lambda x: x["sentiment_score"], reverse=True)
        
        return {
            "sentiments": {
                "coin_sentiments": coin_sentiments[:SENTIMENT_TOP_COUNT],
                "total_coins": len(coin_sentiments)
            },
            "trending": trending,
            "hot_discussions": hot_discussions
        }
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return {"error": str(e)}

def format_sentiment_summary(result: dict) -> str:
    """Format sentiment analysis results as summary"""
    if "error" in result:
        return f"Sentiment analysis failed: {result['error']}"
    
    sentiments = result.get("sentiments", {})
    coin_sentiments = sentiments.get("coin_sentiments", [])
    trending = result.get("trending", {})
    hot_discussions = result.get("hot_discussions", {})
    
    summary_parts = []
    
    # Top sentiment score coins
    if coin_sentiments:
        summary_parts.append("😊 Top sentiment coins:")
        for coin in coin_sentiments[:3]:
            summary_parts.append(f"  • {coin['coin']}: {coin['sentiment_score']:.3f}")
    
    # Hot discussion coins
    hot_topics = hot_discussions.get("hot_topics", [])
    if hot_topics:
        summary_parts.append("🔥 Hot discussions:")
        for topic in hot_topics[:2]:
            summary_parts.append(f"  • {topic['topic']}")
    
    # Trending keywords
    top_keywords = trending.get("keywords", [])[:HOT_TOPICS_TOP_COUNT]
    if top_keywords:
        keyword_list = [f"{kw['word']} ({kw['count']}x)" for kw in top_keywords]
        summary_parts.append(f"📈 Keywords: {', '.join(keyword_list)}")
    
    return "\n".join(summary_parts) 