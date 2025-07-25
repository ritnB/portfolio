# tests/test_tools.py - Tools tests
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from analyzers.trend_analyzer import analyze_coin_trends, detect_surge_crash
from analyzers.prediction_analyzer import analyze_prediction_performance, calculate_accuracy_rates
from analyzers.sentiment_analyzer import analyze_community_sentiment
from tools.threads_poster import upload_to_threads


@pytest.mark.skip(reason="Requires actual Supabase connection. For integration testing")
def test_analyze_coin_trends_integration():
    """Trend analysis integration test (actual database connection)"""
    result = analyze_coin_trends()
    
    assert isinstance(result, dict)
    if "error" not in result:
        assert "trends" in result
        assert "stats" in result
        assert isinstance(result["trends"], dict)


@patch('analyzers.trend_analyzer.get_ema_data')
def test_analyze_coin_trends_with_mock_data(mock_get_data):
    """Trend analysis test using mock data"""
    
    # Generate mock data
    mock_df = pd.DataFrame({
        'coin': ['Bitcoin', 'Bitcoin', 'Ethereum', 'Ethereum'],
        'timestamp': pd.to_datetime(['2025-01-20', '2025-01-21', '2025-01-20', '2025-01-21']),
        'ema': [50000, 55000, 3000, 2700]  # Bitcoin 10% rise, Ethereum 10% fall
    })
    
    mock_get_data.return_value = mock_df
    
    result = analyze_coin_trends()
    
    assert "trends" in result
    trends = result["trends"]
    
    # Check surge coins (Bitcoin)
    surge_coins = [coin["coin"] for coin in trends.get("surge", [])]
    assert "Bitcoin" in surge_coins
    
    # Check crash coins (Ethereum)
    crash_coins = [coin["coin"] for coin in trends.get("crash", [])]
    assert "Ethereum" in crash_coins


def test_detect_surge_crash():
    """Surge/crash detection function test"""
    
    # Test data
    test_df = pd.DataFrame({
        'coin': ['Bitcoin', 'Ethereum', 'Cardano'],
        'timestamp': pd.to_datetime(['2025-01-21'] * 3),
        'ema': [50000, 3000, 1.5],
        'ema_pct_change': [15.0, -12.0, 5.0]  # surge, crash, normal
    })
    
    result = detect_surge_crash(test_df)
    
    assert isinstance(result, dict)
    assert "surge" in result
    assert "crash" in result
    
    # Check if Bitcoin is in surge list
    surge_coins = [coin["coin"] for coin in result["surge"]]
    assert "Bitcoin" in surge_coins
    
    # Check if Ethereum is in crash list
    crash_coins = [coin["coin"] for coin in result["crash"]]
    assert "Ethereum" in crash_coins


@patch('analyzers.prediction_analyzer.get_prediction_data')
def test_calculate_accuracy_rates_with_mock(mock_get_data):
    """Accuracy rate calculation test using mock data"""
    
    # Mock prediction data
    mock_df = pd.DataFrame({
        'coin': ['Bitcoin', 'Bitcoin', 'Ethereum', 'Ethereum'],
        'timestamp': pd.to_datetime(['2025-01-20', '2025-01-21', '2025-01-20', '2025-01-21']),
        'pricetrend': ['up', 'down', 'up', 'down'],
        'is_correct': [True, True, False, True],
        'finalscore': [85.5, -72.3, 45.2, -88.1]
    })
    
    mock_get_data.return_value = mock_df
    
    result = calculate_accuracy_rates(mock_df)
    
    assert "overall" in result
    assert result["overall"]["accuracy"] == 75.0  # 3/4 = 75%
    
    assert "by_coin" in result
    coin_accuracies = {coin["coin"]: coin["accuracy"] for coin in result["by_coin"]}
    assert coin_accuracies["Bitcoin"] == 100.0  # 2/2
    assert coin_accuracies["Ethereum"] == 50.0  # 1/2


@patch('analyzers.sentiment_analyzer.get_community_data')
def test_analyze_community_sentiment_with_mock(mock_get_data):
    """Sentiment analysis test using mock data"""
    
    # Mock community data
    mock_df = pd.DataFrame({
        'coin': ['Bitcoin', 'Bitcoin', 'Ethereum'],
        'timestamp': pd.to_datetime(['2025-01-21'] * 3),
        'text': ['Bitcoin surge amazing!', 'Good upward trend', 'Ethereum decline concern'],
        'sentimental': ['positive', 'positive', 'negative']
    })
    
    mock_get_data.return_value = mock_df
    
    result = analyze_community_sentiment()
    
    assert "sentiments" in result
    sentiments = result["sentiments"]["coin_sentiments"]
    
    # Check if Bitcoin sentiment score is positive
    bitcoin_sentiment = next((s for s in sentiments if s["coin"] == "Bitcoin"), None)
    assert bitcoin_sentiment is not None
    assert bitcoin_sentiment["sentiment_score"] > 0


@patch('tools.threads_poster.requests.post')
def test_upload_to_threads_success(mock_post):
    """Social media upload success test"""
    
    # Mock success response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_post.return_value = mock_response
    
    result = upload_to_threads("test content")
    
    assert "successful" in result


@patch('tools.threads_poster.requests.post')
def test_upload_to_threads_failure(mock_post):
    """Social media upload failure test"""
    
    # Mock failure response
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = "Bad Request"
    mock_post.return_value = mock_response
    
    result = upload_to_threads("test content")
    
    assert "failed" in result
    assert "400" in result


def test_upload_to_threads_disabled():
    """Social media upload disabled state test"""
    
    with patch('tools.threads_poster.AUTO_POSTING_ENABLED', False):
        result = upload_to_threads("test content")
        
        assert "Disabled" in result
