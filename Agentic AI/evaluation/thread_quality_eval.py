# evaluation/thread_quality_eval.py - Quality evaluation system
from typing import Dict, Any
from loguru import logger

from llm import generate_text
from config import (
    EVAL_WEIGHTS, QUALITY_PASS_SCORE, STRICT_EVALUATION,
    MAX_CONTENT_LENGTH, MIN_CONTENT_LENGTH, FORBIDDEN_KEYWORDS,
    QUALITY_MIN_INDIVIDUAL_SCORE
)


def evaluate_thread_output(generated_text: str, analysis_data: Dict) -> Dict:
    """Quality evaluation system"""
    try:
        # =============================================================================
        # 🔢 Basic Metrics Evaluation
        # =============================================================================
        
        basic_metrics = _evaluate_basic_metrics(generated_text)
        
        # =============================================================================
        # 🤖 LLM-based Evaluation
        # =============================================================================
        
        llm_evaluation = _evaluate_with_llm(generated_text, analysis_data)
        
        # =============================================================================
        # 🛡️ Safety Evaluation
        # =============================================================================
        
        safety_evaluation = _evaluate_safety(generated_text)
        
        # =============================================================================
        # 📊 Final Score Calculation
        # =============================================================================
        
        final_score = _calculate_final_score(basic_metrics, llm_evaluation, safety_evaluation)
        
        return {
            "total_score": final_score["total"],
            "individual_scores": final_score["individual"],
            "basic_metrics": basic_metrics,
            "llm_evaluation": llm_evaluation,
            "safety_evaluation": safety_evaluation,
            "pass_threshold": QUALITY_PASS_SCORE,
            "passes_evaluation": final_score["total"] >= QUALITY_PASS_SCORE
        }
        
    except Exception as e:
        logger.error(f"Evaluation system error: {e}")
        return {
            "total_score": 0,
            "error": str(e),
            "passes_evaluation": False
        }


def _evaluate_basic_metrics(text: str) -> Dict:
    """Evaluate basic metrics"""
    metrics = {}
    
    # Length evaluation
    text_length = len(text)
    if MIN_CONTENT_LENGTH <= text_length <= MAX_CONTENT_LENGTH:
        metrics["length_score"] = 100
    elif text_length < MIN_CONTENT_LENGTH:
        metrics["length_score"] = max(0, (text_length / MIN_CONTENT_LENGTH) * 100)
    else:
        excess_ratio = (text_length - MAX_CONTENT_LENGTH) / MAX_CONTENT_LENGTH
        metrics["length_score"] = max(0, 100 - (excess_ratio * 50))
    
    # Structure evaluation (simple heuristics)
    has_emoji = any(ord(char) > 127 for char in text)  # Include emoji/special characters
    has_hashtag = '#' in text
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    
    structure_score = 0
    if has_emoji: structure_score += 25
    if has_hashtag: structure_score += 25
    if 1 <= sentence_count <= 3: structure_score += 50
    
    metrics["structure_score"] = structure_score
    
    # Readability evaluation (simple metrics)
    avg_sentence_length = text_length / max(sentence_count, 1)
    if 10 <= avg_sentence_length <= 50:
        metrics["readability_score"] = 100
    else:
        metrics["readability_score"] = max(0, 100 - abs(avg_sentence_length - 30) * 2)
    
    return metrics


def _evaluate_with_llm(text: str, analysis_data: Dict) -> Dict:
    """LLM-based evaluation"""
    
    # Summarize analysis data
    trend_summary = analysis_data.get("trend_analysis", {}).get("stats", {})
    prediction_summary = analysis_data.get("prediction_analysis", {}).get("accuracy", {})
    
    prompt = f"""
Please evaluate the following cryptocurrency-related post based on 4 criteria:

Generated text:
"{text}"

Analysis data summary:
- Trend analysis: {trend_summary}
- Prediction performance: {prediction_summary}

Evaluation criteria (each out of 100 points):

1. **Data Accuracy** (40%): Is the provided analysis data accurately reflected in the text?
2. **Market Relevance** (30%): Is it highly relevant to current market conditions?
3. **Engagement** (20%): Is it attractive content that can capture readers' interest?
4. **Risk Management** (10%): Does it avoid investment advice or exaggerated expressions?

Please provide a score (0-100) and brief reason for each criterion.

Answer in the following JSON format:
{{
    "data_accuracy": 85,
    "market_relevance": 90,
    "engagement": 75,
    "risk_management": 95,
    "comments": {{
        "data_accuracy": "Analysis data well reflected",
        "market_relevance": "Matches current market trends",
        "engagement": "Interesting but could be more impactful",
        "risk_management": "Fact-based without investment advice"
    }}
}}
"""
    
    try:
        response = generate_text(prompt)
        
        # JSON 파싱 시도
        import json
        import re
        
        # Extract JSON part only
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            try:
                result = json.loads(json_str)
                
                # Validate required fields
                required_fields = ["data_accuracy", "market_relevance", "engagement", "risk_management"]
                if all(field in result for field in required_fields):
                    # Validate score range (0-100)
                    for field in required_fields:
                        if not isinstance(result[field], (int, float)) or not (0 <= result[field] <= 100):
                            logger.warning(f"Invalid score range: {field} = {result[field]}")
                            return _get_default_llm_evaluation()
                    return result
                else:
                    logger.warning(f"Missing required fields: {set(required_fields) - set(result.keys())}")
                    return _get_default_llm_evaluation()
                    
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing error: {e}")
                return _get_default_llm_evaluation()
        else:
            logger.warning("Could not find JSON format in LLM response")
            return _get_default_llm_evaluation()
            
    except Exception as e:
        logger.warning(f"LLM evaluation failed: {e}")
        return _get_default_llm_evaluation()


def _evaluate_safety(text: str) -> Dict:
    """Safety evaluation"""
    safety_score = 100
    issues = []
    
    # Check forbidden keywords
    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in text:
            safety_score -= 20
            issues.append(f"Forbidden keyword included: {keyword}")
    
    # Check exaggerated expressions
    exaggerated_words = ["무조건", "확실", "절대", "100%"]
    for word in exaggerated_words:
        if word in text:
            safety_score -= 10
            issues.append(f"Exaggerated expression: {word}")
    
    return {
        "safety_score": max(0, safety_score),
        "issues": issues
    }


def _calculate_final_score(basic_metrics: Dict, llm_eval: Dict, safety_eval: Dict) -> Dict:
    """Calculate final score"""
    
    # Basic metrics score (20%)
    basic_avg = sum(basic_metrics.values()) / len(basic_metrics)
    basic_weighted = basic_avg * 0.2
    
    # LLM evaluation score (70%)
    llm_scores = {
        "data_accuracy": llm_eval.get("data_accuracy", 50),
        "market_relevance": llm_eval.get("market_relevance", 50),
        "engagement": llm_eval.get("engagement", 50),
        "risk_management": llm_eval.get("risk_management", 50)
    }
    
    llm_weighted = 0
    for criterion, weight in EVAL_WEIGHTS.items():
        score = llm_scores.get(criterion, 50)
        llm_weighted += score * (weight / 100) * 0.7
    
    # Safety score (10%)
    safety_weighted = safety_eval["safety_score"] * 0.1
    
    # Final score
    total_score = round(basic_weighted + llm_weighted + safety_weighted, 1)
    
    return {
        "total": total_score,
        "individual": {
            "basic_metrics": round(basic_weighted, 1),
            "llm_evaluation": round(llm_weighted, 1),
            "safety": round(safety_weighted, 1),
            **llm_scores
        }
    }


def _get_default_llm_evaluation() -> Dict:
    """Default values when LLM evaluation fails"""
    return {
        "data_accuracy": 50,
        "market_relevance": 50,
        "engagement": 50,
        "risk_management": 50,
        "comments": {
            "data_accuracy": "Evaluation unavailable",
            "market_relevance": "Evaluation unavailable", 
            "engagement": "Evaluation unavailable",
            "risk_management": "Evaluation unavailable"
        }
    }