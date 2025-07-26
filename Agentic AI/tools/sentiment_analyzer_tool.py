# tools/sentiment_analyzer_tool.py - Sentiment Analysis Tool
from langchain.tools import BaseTool
from typing import Optional
import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzers.sentiment_analyzer import analyze_sentiment

class SentimentAnalyzerTool(BaseTool):
    name = "sentiment_analyzer"
    description = """
    Analyze community sentiment and mood for cryptocurrencies.
    Use this tool to understand market sentiment and community reactions.
    
    Input: Target coins to analyze (optional, defaults to top coins)
    Output: Sentiment analysis results including trending topics and keywords
    
    This tool helps identify:
    - Community mood and sentiment trends
    - Hot topics and trending keywords
    - Coins with high community engagement
    - Urgency signals in discussions
    """
    
    def _run(self, target_coins: Optional[str] = None) -> str:
        """Analyze community sentiment"""
        try:
            # Parse target coins
            coins = None
            if target_coins:
                coins = [coin.strip() for coin in target_coins.split(',')]
            
            result = analyze_sentiment(target_coins=coins)
            return str(result)
            
        except Exception as e:
            return f"Sentiment analysis failed: {str(e)}"
    
    def _arun(self, target_coins: Optional[str] = None) -> str:
        """Async version of sentiment analysis"""
        return self._run(target_coins) 