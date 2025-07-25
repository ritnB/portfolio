# strategies/content_generator.py - Content generation strategies
from typing import Dict, List, Optional
from loguru import logger

from llm import generate_text
from config import (
    MAX_CONTENT_LENGTH, MIN_CONTENT_LENGTH, USE_EMOJIS, 
    CONTENT_STYLE, SOCIAL_MEDIA_HASHTAGS, SOCIAL_MEDIA_TONE
)

def safe_prompt(text: str) -> str:
    """Utility for safe prompt generation"""
    try:
        # Safely encode to UTF-8
        return text.encode('utf-8', errors='replace').decode('utf-8')
    except:
        # Replace with ASCII-safe version if there's an issue
        import re
        # Simplify complex special characters
        text = re.sub(r'[^\w\s가-힣.,!?%-]', '', text)
        return text


class ContentGenerator:
    """Content generation strategy class"""
    
    def __init__(self):
        self.max_length = MAX_CONTENT_LENGTH
        self.min_length = MIN_CONTENT_LENGTH
        self.use_emojis = USE_EMOJIS
        self.style = CONTENT_STYLE
        self.hashtags = SOCIAL_MEDIA_HASHTAGS
        self.tone = SOCIAL_MEDIA_TONE
    
    def _get_style_instructions(self) -> str:
        """Return style-specific guidelines"""
        style_guides = {
            "social": "Write in short and engaging style suitable for social media platforms",
            "twitter": "Write in concise and impactful style suitable for Twitter", 
            "formal": "Write in formal and professional tone"
        }
        
        tone_guides = {
            "casual": "friendly and comfortable tone",
            "professional": "professional and trustworthy tone",
            "playful": "fun and energetic tone"
        }
        
        return f"{style_guides.get(self.style, '')} - {tone_guides.get(self.tone, '')}"
    
    def _build_base_prompt(self) -> str:
        """Build base prompt"""
        prompt_parts = [
            f"Based on cryptocurrency AI analysis results, please write a short post within {self.max_length} characters.",
            f"Writing style: {self._get_style_instructions()}",
            ""
        ]
        
        if self.use_emojis:
            prompt_parts.append("Use appropriate emojis to make the content more attractive.")
        
        prompt_parts.extend([
            "Important rules:",
            "1. Never provide investment advice (no words like 'buy', 'sell', 'recommend')",
            "2. Only provide data-based facts",
            "3. Avoid exaggerated expressions",
            f"4. Write between {self.min_length} and {self.max_length} characters",
            "5. Write in target language",
            ""
        ])
        
        return "\n".join(prompt_parts)
    
    def generate_surge_content(self, coin_data: Dict, prediction_match: bool = False) -> str:
        """Generate surge coin content"""
        coin = coin_data["coin"]
        change_rate = coin_data["change_rate"]
        
        prompt = self._build_base_prompt()
        prompt += f"""
Analysis Information:
- Coin: {coin}
- EMA Change Rate: {change_rate}%
- Surge situation detected
"""
        
        if prediction_match:
            prompt += "- AI prediction was accurate\n"
        
        prompt += f"""
Based on the above information, please write a post announcing the surge situation of {coin}.
Write concisely based on facts, but deliver it in an interesting way."""
        
        if self.hashtags:
            prompt += f"\nHashtags: {' '.join(self.hashtags)}"
        
        return generate_text(prompt)
    
    def generate_crash_content(self, coin_data: Dict, prediction_match: bool = False) -> str:
        """Generate crash coin content"""
        coin = coin_data["coin"]
        change_rate = coin_data["change_rate"]
        
        prompt = self._build_base_prompt()
        prompt += f"""
Analysis Information:
- Coin: {coin}
- EMA Change Rate: {change_rate}%
- Crash situation detected
"""
        
        if prediction_match:
            prompt += "- AI prediction was accurate\n"
        
        prompt += f"""
Based on the above information, please write a post announcing the crash situation of {coin}.
Write concisely based on facts, but do not create excessive anxiety."""
        
        if self.hashtags:
            prompt += f"\nHashtags: {' '.join(self.hashtags)}"
        
        return generate_text(prompt)
    
    def generate_accuracy_promotion(self, accuracy_data: Dict) -> str:
        """Generate accuracy promotion content"""
        overall_acc = accuracy_data.get("overall", {}).get("accuracy", 0)
        
        prompt = self._build_base_prompt()
        prompt += f"""
AI Prediction Performance Information:
- Overall Accuracy: {overall_acc}%
- Analysis Period: Recent 3 days
- Excellent performance achieved by prediction model
"""
        
        # 상위 성과 코인 추가
        top_coins = accuracy_data.get("by_coin", [])[:3]
        if top_coins:
            prompt += "\nTop Performance Coins:\n"
            for coin in top_coins:
                prompt += f"- {coin['coin']}: {coin['accuracy']}%\n"
        
        prompt += f"""
Based on the above performance, please write a post announcing the excellent performance of AI prediction model.
Write humbly but with confidence."""
        
        if self.hashtags:
            prompt += f"\nHashtags: {' '.join(self.hashtags)}"
        
        return generate_text(prompt)
    
    def generate_community_trend(self, sentiment_data: Dict, hot_topics: List[Dict]) -> str:
        """Generate community trend content"""
        prompt = self._build_base_prompt()
        prompt += "Community Analysis Information:\n"
        
        # 감정 분석 결과
        sentiments = sentiment_data.get("coin_sentiments", [])
        if sentiments:
            top_positive = [s for s in sentiments if s["sentiment_score"] > 0][:2]
            if top_positive:
                prompt += "Coins with Positive Reactions:\n"
                for s in top_positive:
                    prompt += f"- {s['coin']} (sentiment score: {s['sentiment_score']:+.2f})\n"
        
        # 뜨거운 토론 코인
        if hot_topics:
            prompt += "\nHot Discussion Coins:\n"
            for topic in hot_topics[:2]:
                prompt += f"- {topic['coin']} ({topic['comment_count']} comments)\n"
        
        prompt += f"""
Based on the above community analysis results, please write a post announcing current cryptocurrency community interests and trends.
Deliver objectively and interestingly."""
        
        if self.hashtags:
            prompt += f"\nHashtags: {' '.join(self.hashtags)}"
        
        return generate_text(prompt)
    
    def generate_comprehensive_analysis(self, 
                                      trend_data: Dict, 
                                      prediction_data: Dict, 
                                      sentiment_data: Dict) -> str:
        """종합 분석 콘텐츠 생성"""
        prompt = self._build_base_prompt()
        prompt += "Comprehensive Analysis Information:\n\n"
        
        # 트렌드 요약
        trends = trend_data.get("trends", {})
        if trends.get("surge"):
            surge_coin = trends["surge"][0]
            prompt += f"📈 Surge: {surge_coin['coin']} (+{surge_coin['change_rate']}%)\n"
        
        if trends.get("crash"):
            crash_coin = trends["crash"][0] 
            prompt += f"📉 Crash: {crash_coin['coin']} ({crash_coin['change_rate']}%)\n"
        
        # 예측 성과
        accuracy = prediction_data.get("accuracy", {}).get("overall", {}).get("accuracy", 0)
        if accuracy > 0:
            prompt += f"🎯 AI Prediction Accuracy: {accuracy}%\n"
        
        # 커뮤니티 반응
        hot_topics = sentiment_data.get("hot_discussions", {}).get("hot_topics", [])
        if hot_topics:
            hot_coin = hot_topics[0]
            prompt += f"🔥 Community Focus: {hot_coin['coin']}\n"
        
        prompt += f"""
Based on the above comprehensive analysis results, please write a post announcing the overall situation of the current cryptocurrency market.
Deliver concisely but comprehensively."""
        
        if self.hashtags:
            prompt += f"\nHashtags: {' '.join(self.hashtags)}"
        
        return generate_text(prompt)


def generate_targeted_content(analysis_results: Dict) -> Dict:
    """Generate targeted content based on analysis results"""
    generator = ContentGenerator()
    
    trend_data = analysis_results.get("trend_analysis", {})
    prediction_data = analysis_results.get("prediction_analysis", {})
    sentiment_data = analysis_results.get("sentiment_analysis", {})
    
    contents = []
    
    try:
        # 1. 급등/급락 + 예측 매칭 콘텐츠 (최우선)
        successful_matches = prediction_data.get("successful_predictions", {}).get("successful_matches", [])
        
        for match in successful_matches[:2]:  # 최대 2개
            if match["type"] == "surge_prediction":
                content = generator.generate_surge_content(
                    {"coin": match["coin"], "change_rate": match["trend_change"]}, 
                    prediction_match=True
                )
                contents.append({
                    "type": "surge_with_prediction",
                    "content": content,
                    "priority": 1,
                    "coin": match["coin"]
                })
            elif match["type"] == "crash_prediction":
                content = generator.generate_crash_content(
                    {"coin": match["coin"], "change_rate": match["trend_change"]}, 
                    prediction_match=True
                )
                contents.append({
                    "type": "crash_with_prediction", 
                    "content": content,
                    "priority": 1,
                    "coin": match["coin"]
                })
        
        # 2. 적중률 홍보 콘텐츠
        promotion = prediction_data.get("promotion", {})
        if promotion.get("can_promote_3day") or promotion.get("can_promote_1day"):
            content = generator.generate_accuracy_promotion(prediction_data.get("accuracy", {}))
            contents.append({
                "type": "accuracy_promotion",
                "content": content, 
                "priority": 2
            })
        
        # 3. 커뮤니티 트렌드 콘텐츠
        hot_topics = sentiment_data.get("hot_discussions", {}).get("hot_topics", [])
        if hot_topics:
            content = generator.generate_community_trend(sentiment_data, hot_topics)
            contents.append({
                "type": "community_trend",
                "content": content,
                "priority": 3
            })
        
        # 4. 종합 분석 콘텐츠 (fallback)
        if not contents:
            content = generator.generate_comprehensive_analysis(trend_data, prediction_data, sentiment_data)
            contents.append({
                "type": "comprehensive_analysis",
                "content": content,
                "priority": 4
            })
        
        # 우선순위별 정렬
        contents.sort(key=lambda x: x["priority"])
        
        logger.success(f"✅ Content generation completed: {len(contents)} pieces")
        
        return {
            "contents": contents,
            "recommended": contents[0] if contents else None,
            "total_generated": len(contents)
        }
        
    except Exception as e:
        logger.error(f"❌ Content generation failed: {e}")
        return {"error": str(e), "contents": [], "recommended": None} 