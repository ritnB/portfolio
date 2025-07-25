# tests/test_chain.py - Chain tests
import pytest
from unittest.mock import patch, MagicMock

from chains.thread_chain import generate_thread_content
from strategies.content_generator import ContentGenerator, generate_targeted_content


@pytest.mark.skip(reason="This test includes actual LLM calls. Cost caution")
def test_generate_thread_content_legacy():
    """Legacy chain function test (compatibility maintained)"""
    
    # Sample input values (correct parameter names)
    market_ema_trend = "Bitcoin: 2025-01-20: 50000, 2025-01-21: 55000"
    prediction_info = "Bitcoin prediction accuracy: 85%"
    community_info = "Bitcoin-related positive comments increasing"

    result = generate_thread_content(
        market_ema_trend=market_ema_trend,
        prediction_info=prediction_info,
        community_info=community_info
    )

    assert isinstance(result, str)
    assert len(result.strip()) > 0
    # Length limit checks handled by new system


@patch('llm.generate_text')
def test_content_generator_surge_content(mock_llm):
    """Surge content generation test"""
    
    mock_llm.return_value = "🚀 Bitcoin 15% surge! EMA analysis shows strong upward momentum detected #crypto"
    
    generator = ContentGenerator()
    coin_data = {"coin": "Bitcoin", "change_rate": 15.5}
    
    result = generator.generate_surge_content(coin_data, prediction_match=True)
    
    assert isinstance(result, str)
    assert "Bitcoin" in result
    mock_llm.assert_called_once()


@patch('llm.generate_text')
def test_content_generator_crash_content(mock_llm):
    """Crash content generation test"""
    
    mock_llm.return_value = "📉 Ethereum 12% decline detected. EMA-based analysis confirms correction zone entry #crypto"
    
    generator = ContentGenerator()
    coin_data = {"coin": "Ethereum", "change_rate": -12.0}
    
    result = generator.generate_crash_content(coin_data, prediction_match=False)
    
    assert isinstance(result, str)
    assert "Ethereum" in result
    mock_llm.assert_called_once()


@patch('llm.generate_text')
def test_content_generator_accuracy_promotion(mock_llm):
    """Accuracy promotion content generation test"""
    
    mock_llm.return_value = "🎯 AI prediction model achieved 85% 3-day accuracy! Data-driven analysis reliability proven #AI"
    
    generator = ContentGenerator()
    accuracy_data = {
        "overall": {"accuracy": 85.5},
        "by_coin": [
            {"coin": "Bitcoin", "accuracy": 90},
            {"coin": "Ethereum", "accuracy": 80}
        ]
    }
    
    result = generator.generate_accuracy_promotion(accuracy_data)
    
    assert isinstance(result, str)
    assert "85" in result or "accuracy" in result
    mock_llm.assert_called_once()


@patch('strategies.content_generator.ContentGenerator.generate_surge_content')
@patch('strategies.content_generator.ContentGenerator.generate_accuracy_promotion')
def test_generate_targeted_content_integration(mock_accuracy, mock_surge):
    """Targeted content generation integration test"""
    
    # Set mock return values
    mock_surge.return_value = "🚀 Bitcoin surge content"
    mock_accuracy.return_value = "🎯 Accuracy promotion content"
    
    # Analysis results mock data
    analysis_results = {
        "trend_analysis": {
            "trends": {"surge": [{"coin": "Bitcoin", "change_rate": 15}], "crash": []}
        },
        "prediction_analysis": {
            "promotion": {"can_promote_3day": True},
            "successful_predictions": {
                "successful_matches": [{
                    "type": "surge_prediction",
                    "coin": "Bitcoin",
                    "trend_change": 15.0
                }]
            },
            "accuracy": {"overall": {"accuracy": 85}}
        },
        "sentiment_analysis": {
            "hot_discussions": {"hot_topics": []}
        }
    }
    
    result = generate_targeted_content(analysis_results)
    
    assert "recommended" in result
    assert "contents" in result
    assert result["recommended"] is not None
    assert len(result["contents"]) > 0
    
    # Check if high-priority content was recommended
    recommended = result["recommended"]
    assert recommended["type"] in ["surge_with_prediction", "accuracy_promotion"]


def test_generate_targeted_content_no_data():
    """Content generation test when no data is available"""
    
    # Empty analysis results
    analysis_results = {
        "trend_analysis": {"trends": {"surge": [], "crash": []}},
        "prediction_analysis": {
            "promotion": {"can_promote_3day": False},
            "successful_predictions": {"successful_matches": []}
        },
        "sentiment_analysis": {
            "hot_discussions": {"hot_topics": []}
        }
    }
    
    with patch('strategies.content_generator.ContentGenerator.generate_comprehensive_analysis') as mock_comprehensive:
        mock_comprehensive.return_value = "📊 Comprehensive analysis content"
        
        result = generate_targeted_content(analysis_results)
        
        assert "recommended" in result
        assert result["recommended"]["type"] == "comprehensive_analysis"
        mock_comprehensive.assert_called_once()
