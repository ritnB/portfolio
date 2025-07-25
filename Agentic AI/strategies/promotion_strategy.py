# strategies/promotion_strategy.py - Promotion strategy
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from loguru import logger

from config import (
    ACCURACY_THRESHOLD_3DAY, ACCURACY_THRESHOLD_1DAY,
    SURGE_THRESHOLD, CRASH_THRESHOLD, FORBIDDEN_KEYWORDS
)


class PromotionStrategy:
    """Promotion strategy decision class"""
    
    def __init__(self):
        self.accuracy_3day_threshold = ACCURACY_THRESHOLD_3DAY
        self.accuracy_1day_threshold = ACCURACY_THRESHOLD_1DAY
        self.surge_threshold = SURGE_THRESHOLD
        self.crash_threshold = CRASH_THRESHOLD
        self.forbidden_keywords = FORBIDDEN_KEYWORDS
    
    def should_promote_accuracy(self, prediction_analysis: Dict) -> Tuple[bool, str]:
        """Determine whether to promote accuracy"""
        promotion_data = prediction_analysis.get("promotion", {})
        
        can_promote_3day = promotion_data.get("can_promote_3day", False)
        can_promote_1day = promotion_data.get("can_promote_1day", False)
        overall_accuracy = promotion_data.get("overall_accuracy", 0)
        
        if can_promote_3day and can_promote_1day:
            return True, f"All criteria met (3-day: {overall_accuracy}%, daily: all days {self.accuracy_1day_threshold}%+)"
        elif can_promote_3day:
            return True, f"3-day overall criteria met ({overall_accuracy}% >= {self.accuracy_3day_threshold}%)"
        elif can_promote_1day:
            return True, f"Daily criteria met (all days {self.accuracy_1day_threshold}%+)"
        else:
            return False, f"Criteria not met (current: {overall_accuracy}%, required: 3-day {self.accuracy_3day_threshold}% or daily {self.accuracy_1day_threshold}%)"
    
    def should_promote_trend_prediction(self, trend_analysis: Dict, prediction_analysis: Dict) -> List[Dict]:
        """Find trend prediction success targets for promotion"""
        successful_matches = prediction_analysis.get("successful_predictions", {}).get("successful_matches", [])
        
        promotable_matches = []
        
        for match in successful_matches:
            coin = match["coin"]
            match_type = match["type"]
            confidence = abs(match.get("confidence", 0))
            trend_change = abs(match.get("trend_change", 0))
            
            # Check promotion criteria
            should_promote = False
            reason = ""
            
            if match_type == "surge_prediction":
                if trend_change >= self.surge_threshold and confidence >= 70:
                    should_promote = True
                    reason = f"Surge prediction success ({trend_change:+.1f}%, confidence {confidence:.1f})"
            
            elif match_type == "crash_prediction":
                if trend_change >= abs(self.crash_threshold) and confidence >= 70:
                    should_promote = True
                    reason = f"Crash prediction success ({trend_change:+.1f}%, confidence {confidence:.1f})"
            
            if should_promote:
                promotable_matches.append({
                    "coin": coin,
                    "type": match_type,
                    "confidence": confidence,
                    "trend_change": trend_change,
                    "reason": reason,
                    "timestamp": match.get("timestamp"),
                    "priority": self._calculate_promotion_priority(match_type, trend_change, confidence)
                })
        
        # Sort by priority
        promotable_matches.sort(key=lambda x: x["priority"], reverse=True)
        
        return promotable_matches
    
    def _calculate_promotion_priority(self, match_type: str, trend_change: float, confidence: float) -> float:
        """Calculate promotion priority"""
        base_score = 0
        
        # Change rate score (higher score for larger changes)
        change_score = min(abs(trend_change) / 20, 1.0) * 40  # Max 40 points
        
        # Confidence score
        confidence_score = min(confidence / 100, 1.0) * 30  # Max 30 points
        
        # Type-based weight
        type_score = 20 if match_type == "surge_prediction" else 15  # Surge gets more attention
        
        # Time bonus (higher score for more recent)
        time_score = 10  # Base 10 points
        
        return base_score + change_score + confidence_score + type_score + time_score
    
    def check_content_safety(self, content: str) -> Tuple[bool, List[str]]:
        """Check content safety"""
        issues = []
        
        # Check forbidden keywords
        for keyword in self.forbidden_keywords:
            if keyword in content:
                issues.append(f"Forbidden keyword included: '{keyword}'")
        
        # Check exaggerated expressions
        exaggerated_patterns = ["무조건", "확실", "100%", "guaranteed", "절대"]
        for pattern in exaggerated_patterns:
            if pattern in content:
                issues.append(f"Exaggerated expression used: '{pattern}'")
        
        # Check investment advice patterns
        investment_advice_patterns = ["사세요", "파세요", "투자하세요", "buy", "sell"]
        for pattern in investment_advice_patterns:
            if pattern in content:
                issues.append(f"Investment advice suspected: '{pattern}'")
        
        is_safe = len(issues) == 0
        return is_safe, issues
    
    def evaluate_promotion_opportunity(self, 
                                     trend_analysis: Dict, 
                                     prediction_analysis: Dict, 
                                     sentiment_analysis: Dict) -> Dict:
        """Comprehensive promotion opportunity evaluation"""
        
        # 1. Accuracy promotion opportunity
        accuracy_promotion, accuracy_reason = self.should_promote_accuracy(prediction_analysis)
        
        # 2. Trend prediction success promotion opportunity
        trend_promotions = self.should_promote_trend_prediction(trend_analysis, prediction_analysis)
        
        # 3. Community interest evaluation
        hot_topics = sentiment_analysis.get("hot_discussions", {}).get("hot_topics", [])
        community_interest_score = len(hot_topics) * 10  # Number of hot topics * 10
        
        # 4. Calculate total promotion score
        total_score = 0
        
        if accuracy_promotion:
            total_score += 50
        
        total_score += len(trend_promotions) * 30
        total_score += min(community_interest_score, 20)
        
        # 5. Determine promotion strategy
        strategy = self._determine_promotion_strategy(total_score, accuracy_promotion, trend_promotions)
        
        return {
            "total_score": total_score,
            "strategy": strategy,
            "accuracy_promotion": {
                "should_promote": accuracy_promotion,
                "reason": accuracy_reason
            },
            "trend_promotions": trend_promotions,
            "community_interest": {
                "score": community_interest_score,
                "hot_topics_count": len(hot_topics)
            },
            "recommended_actions": self._get_recommended_actions(strategy, accuracy_promotion, trend_promotions)
        }
    
    def _determine_promotion_strategy(self, score: int, has_accuracy: bool, trend_matches: List) -> str:
        """Determine promotion strategy"""
        if score >= 100:
            return "aggressive"  # Aggressive promotion
        elif score >= 50:
            return "moderate"    # Moderate promotion
        elif score >= 30:
            return "conservative"  # Conservative promotion
        else:
            return "minimal"     # Minimal promotion
    
    def _get_recommended_actions(self, strategy: str, has_accuracy: bool, trend_matches: List) -> List[str]:
        """Recommended actions by strategy"""
        actions = []
        
        if strategy == "aggressive":
            actions.append("Create main promotional content and publish immediately")
            if has_accuracy:
                actions.append("Emphasize accuracy performance")
            if trend_matches:
                actions.append("Specifically mention prediction success cases")
        
        elif strategy == "moderate":
            actions.append("Create balanced informational content")
            if has_accuracy:
                actions.append("Mention performance with humble tone")
        
        elif strategy == "conservative":
            actions.append("Simple fact-based updates")
            actions.append("Avoid excessive confidence expressions")
        
        else:  # minimal
            actions.append("Provide basic market information only")
            actions.append("Minimize promotional elements")
        
        return actions
    
    def generate_promotion_report(self, evaluation: Dict) -> str:
        """Generate promotion evaluation report"""
        strategy = evaluation["strategy"]
        score = evaluation["total_score"]
        
        report_parts = [
            f"🎯 Promotion Strategy Evaluation Results",
            f"Total Score: {score} points",
            f"Recommended Strategy: {strategy.upper()}",
            ""
        ]
        
        # Accuracy promotion status
        accuracy = evaluation["accuracy_promotion"]
        report_parts.append(f"📊 Accuracy Promotion: {'✅ Possible' if accuracy['should_promote'] else '❌ Not possible'}")
        report_parts.append(f"   Reason: {accuracy['reason']}")
        report_parts.append("")
        
        # Trend prediction success status
        trend_promotions = evaluation["trend_promotions"]
        report_parts.append(f"🎯 Prediction Success Promotion: {len(trend_promotions)} cases")
        for i, promo in enumerate(trend_promotions[:3], 1):
            report_parts.append(f"   {i}. {promo['coin']}: {promo['reason']}")
        
        # Recommended actions
        actions = evaluation["recommended_actions"]
        if actions:
            report_parts.append("")
            report_parts.append("📋 Recommended Actions:")
            for action in actions:
                report_parts.append(f"   • {action}")
        
        return "\n".join(report_parts) 