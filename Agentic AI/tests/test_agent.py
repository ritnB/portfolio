# tests/test_agent.py - Agent tests
import pytest
from unittest.mock import patch, MagicMock
from agents.agent import run_agent


@pytest.mark.skip(reason="Requires actual database and LLM API calls. For integration testing")
def test_run_agent_end_to_end():
    """Full agent execution test (actual API calls)"""
    result = run_agent()

    assert isinstance(result, dict)
    assert "status" in result
    assert "content" in result
    
    # Status check
    valid_statuses = ["uploaded", "blocked", "error", "upload_failed"]
    assert result["status"] in valid_statuses


@patch('analyzers.trend_analyzer.analyze_coin_trends')
@patch('analyzers.prediction_analyzer.analyze_prediction_performance')
@patch('analyzers.sentiment_analyzer.analyze_community_sentiment')
@patch('strategies.content_generator.generate_targeted_content')
@patch('evaluation.thread_quality_eval.evaluate_thread_output')
@patch('tools.threads_poster.upload_to_threads')
def test_run_agent_with_mocks(
    mock_upload, mock_eval, mock_content, mock_sentiment, mock_prediction, mock_trend
):
    """Agent test using mocks"""
    
    # Mock data setup
    mock_trend.return_value = {
        "trends": {"surge": [{"coin": "Bitcoin", "change_rate": 15.5}], "crash": []},
        "stats": {"total_coins_analyzed": 10}
    }
    
    mock_prediction.return_value = {
        "accuracy": {"overall": {"accuracy": 75.5}},
        "promotion": {"can_promote_3day": True},
        "successful_predictions": {"successful_matches": []}
    }
    
    mock_sentiment.return_value = {
        "sentiments": {"coin_sentiments": []},
        "hot_discussions": {"hot_topics": []}
    }
    
    mock_content.return_value = {
        "recommended": {
            "type": "surge_with_prediction",
            "content": "🚀 Bitcoin surge detected! EMA-based analysis shows 15.5% increase recorded. #crypto #AI",
            "priority": 1
        },
        "contents": []
    }
    
    mock_eval.return_value = {
        "total_score": 85.5,
        "passes_evaluation": True,
        "individual_scores": {}
    }
    
    mock_upload.return_value = "✅ Social media upload successful"
    
    # Execute test
    result = run_agent()
    
    # Verify results
    assert result["status"] == "uploaded"
    assert "Bitcoin" in result["content"]
    assert result["content_type"] == "surge_with_prediction"
    
    # Verify mock calls
    mock_trend.assert_called_once()
    mock_prediction.assert_called_once()
    mock_sentiment.assert_called_once()
    mock_content.assert_called_once()
    mock_eval.assert_called_once()
    mock_upload.assert_called_once()


@patch('analyzers.trend_analyzer.analyze_coin_trends')
def test_run_agent_trend_analysis_failure(mock_trend):
    """Error handling test when trend analysis fails"""
    mock_trend.return_value = {"error": "Database connection failed"}
    
    result = run_agent()
    
    assert result["status"] == "error"
    assert result["error_type"] == "trend_analysis_failed"
    assert "Database connection failed" in result["error_message"]


@patch('analyzers.trend_analyzer.analyze_coin_trends')
@patch('analyzers.prediction_analyzer.analyze_prediction_performance')
@patch('analyzers.sentiment_analyzer.analyze_community_sentiment')
@patch('strategies.content_generator.generate_targeted_content')
@patch('strategies.promotion_strategy.PromotionStrategy')
def test_run_agent_content_safety_block(
    mock_strategy_class, mock_content, mock_sentiment, mock_prediction, mock_trend
):
    """Test case where content is blocked due to safety issues"""
    
    # Mock strategy instance
    mock_strategy = MagicMock()
    mock_strategy_class.return_value = mock_strategy
    
    # Normal analysis results
    mock_trend.return_value = {"trends": {"surge": [], "crash": []}}
    mock_prediction.return_value = {"accuracy": {"overall": {"accuracy": 70}}}
    mock_sentiment.return_value = {"sentiments": {"coin_sentiments": []}}
    
    # Generate dangerous content
    mock_content.return_value = {
        "recommended": {
            "content": "You should definitely buy Bitcoin! It will surely go up!",
            "type": "dangerous_content"
        }
    }
    
    # Safety check finds issues
    mock_strategy.check_content_safety.return_value = (False, ["Investment advice suspected: 'buy'"])
    mock_strategy.evaluate_promotion_opportunity.return_value = {"strategy": "minimal"}
    
    result = run_agent()
    
    assert result["status"] == "blocked"
    assert result["block_reason"] == "safety_issues"
    assert "buy" in str(result["block_details"])