# agents/agent.py - Refactored AI agent

from loguru import logger
from typing import Dict, Any

from analyzers import analyze_coin_trends, analyze_prediction_performance, analyze_community_sentiment
from strategies import generate_targeted_content, PromotionStrategy
from evaluation.thread_quality_eval import evaluate_thread_output
from tools.threads_poster import upload_to_threads
from config import DEBUG_MODE, VERBOSE_LOGGING


def run_agent() -> dict:
    """Main AI agent execution"""
    try:
        logger.info("🚀 AI Agent started")
        
        # =============================================================================
        # 📊 Data Analysis Phase
        # =============================================================================
        
        # 1. Trend analysis (surge/crash)
        logger.info("📊 Starting trend analysis...")
        trend_analysis = analyze_coin_trends()
        
        if "error" in trend_analysis:
            logger.error(f"Trend analysis failed: {trend_analysis['error']}")
            return _create_error_response("trend_analysis_failed", trend_analysis["error"])
        
        # 2. Prediction performance analysis (accuracy)
        logger.info("🎯 Starting prediction performance analysis...")
        prediction_analysis = analyze_prediction_performance()
        
        if "error" in prediction_analysis:
            logger.error(f"Prediction analysis failed: {prediction_analysis['error']}")
            return _create_error_response("prediction_analysis_failed", prediction_analysis["error"])
        
        # 3. Community sentiment analysis
        logger.info("💬 Starting community sentiment analysis...")
        
        # Target trending coins for sentiment analysis
        trending_coins = []
        if "trends" in trend_analysis:
            trending_coins.extend([coin["coin"] for coin in trend_analysis["trends"].get("surge", [])])
            trending_coins.extend([coin["coin"] for coin in trend_analysis["trends"].get("crash", [])])
        
        sentiment_analysis = analyze_community_sentiment(target_coins=trending_coins[:10] if trending_coins else None)
        
        if "error" in sentiment_analysis:
            logger.warning(f"Sentiment analysis failed (continuing execution): {sentiment_analysis['error']}")
            sentiment_analysis = {"sentiments": {"coin_sentiments": []}, "hot_discussions": {"hot_topics": []}}
        
        # =============================================================================
        # 🎯 Promotion Strategy Evaluation
        # =============================================================================
        
        logger.info("🎯 Evaluating promotion strategy...")
        promotion_strategy = PromotionStrategy()
        promotion_evaluation = promotion_strategy.evaluate_promotion_opportunity(
            trend_analysis, prediction_analysis, sentiment_analysis
        )
        
        if DEBUG_MODE:
            logger.debug("Promotion strategy evaluation results:")
            logger.debug(promotion_strategy.generate_promotion_report(promotion_evaluation))
        
        # =============================================================================
        # ✍️ Content Generation
        # =============================================================================
        
        logger.info("✍️ Starting content generation...")
        
        analysis_results = {
            "trend_analysis": trend_analysis,
            "prediction_analysis": prediction_analysis,
            "sentiment_analysis": sentiment_analysis,
            "promotion_evaluation": promotion_evaluation
        }
        
        content_result = generate_targeted_content(analysis_results)
        
        if "error" in content_result:
            logger.error(f"Content generation failed: {content_result['error']}")
            return _create_error_response("content_generation_failed", content_result["error"])
        
        if not content_result.get("recommended"):
            logger.warning("No content generated.")
            return _create_error_response("no_content_generated", "No recommended content available.")
        
        recommended_content = content_result["recommended"]
        generated_text = recommended_content["content"]
        
        # =============================================================================
        # 🔍 Quality Evaluation
        # =============================================================================
        
        logger.info("🔍 Starting quality evaluation...")
        
        # Safety check
        is_safe, safety_issues = promotion_strategy.check_content_safety(generated_text)
        
        if not is_safe:
            logger.warning(f"Content safety issues: {safety_issues}")
            return _create_blocked_response(generated_text, "safety_issues", safety_issues)
        
        # LLM-based quality evaluation
        eval_result = evaluate_thread_output(
            generated_text=generated_text,
            analysis_data=analysis_results
        )
        
        # Output evaluation results
        if VERBOSE_LOGGING:
            logger.info("📊 Quality evaluation results:")
            for key, value in eval_result.items():
                if isinstance(value, (str, int, float)):
                    logger.info(f" - {key}: {value}")
                elif isinstance(value, dict):
                    logger.info(f" - {key}: {value}")
        
        # Check quality criteria
        should_block = _should_block_content(eval_result)
        
        if should_block:
            logger.warning("❌ Upload blocked due to quality standards not met")
            return _create_blocked_response(generated_text, "quality_failed", eval_result)
        
        # =============================================================================
        # 📤 Social Media Upload
        # =============================================================================
        
        logger.info("📤 Attempting social media upload...")
        upload_result = upload_to_threads(generated_text)
        
        # Return final results
        final_result = {
            "status": "uploaded" if "success" in upload_result.lower() else "upload_failed",
            "content": generated_text.strip(),
            "content_type": recommended_content["type"],
            "upload_result": upload_result,
            "analysis_summary": _create_analysis_summary(analysis_results),
            "evaluation": eval_result,
            "promotion_strategy": promotion_evaluation["strategy"]
        }
        
        logger.success("✅ AI Agent execution completed")
        return final_result
        
    except Exception as e:
        logger.error(f"❌ Exception occurred during AI Agent execution: {e}")
        return _create_error_response("unexpected_error", str(e))


def _should_block_content(eval_result: Dict) -> bool:
    """Determine whether to block content based on quality evaluation results"""
    from config import QUALITY_PASS_SCORE, STRICT_EVALUATION, QUALITY_MIN_INDIVIDUAL_SCORE
    
    # Check overall score
    total_score = eval_result.get("total_score", 0)
    
    if STRICT_EVALUATION:
        # Strict mode: all individual criteria must also pass
        individual_scores = eval_result.get("individual_scores", {})
        
        # Check minimum score for each criterion
        for criterion, score in individual_scores.items():
            if score < QUALITY_MIN_INDIVIDUAL_SCORE:
                logger.warning(f"Criterion failed: {criterion} = {score} < {QUALITY_MIN_INDIVIDUAL_SCORE}")
                return True
    
    # Overall score below threshold
    if total_score < QUALITY_PASS_SCORE:
        logger.warning(f"Overall score below threshold: {total_score} < {QUALITY_PASS_SCORE}")
        return True
    
    return False


def _create_analysis_summary(analysis_results: Dict) -> Dict:
    """Generate analysis results summary"""
    trend_data = analysis_results.get("trend_analysis", {})
    prediction_data = analysis_results.get("prediction_analysis", {})
    sentiment_data = analysis_results.get("sentiment_analysis", {})
    
    summary = {
        "trend_summary": "",
        "prediction_summary": "",
        "sentiment_summary": ""
    }
    
    # Trend summary
    if "trends" in trend_data:
        trends = trend_data["trends"]
        surge_count = len(trends.get("surge", []))
        crash_count = len(trends.get("crash", []))
        summary["trend_summary"] = f"Detected {surge_count} surge, {crash_count} crash coins"
    
    # Prediction summary
    if "accuracy" in prediction_data:
        accuracy = prediction_data["accuracy"]["overall"]["accuracy"]
        summary["prediction_summary"] = f"AI prediction accuracy {accuracy}%"
    
    # Sentiment summary
    hot_topics = sentiment_data.get("hot_discussions", {}).get("hot_topics", [])
    summary["sentiment_summary"] = f"Community hot topics: {len(hot_topics)}"
    
    return summary


def _create_error_response(error_type: str, error_message: str) -> Dict:
    """Generate error response"""
    return {
        "status": "error",
        "error_type": error_type,
        "error_message": error_message,
        "content": f"❌ Error occurred: {error_message}",
        "timestamp": logger.info("Error timestamp recorded")
    }


def _create_blocked_response(content: str, block_reason: str, details: Any) -> Dict:
    """Generate blocked response"""
    return {
        "status": "blocked",
        "block_reason": block_reason,
        "content": content.strip(),
        "block_details": details,
        "message": f"[Quality Failed] Generated content did not meet quality standards and was not uploaded.\n\n{content}"
    }