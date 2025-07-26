# tools/trend_analyzer_tool.py - Trend Analysis Tool
from langchain.tools import BaseTool
from typing import Optional
from pydantic import BaseModel, Field
from loguru import logger
from analyzers.trend_analyzer import analyze_coin_trends
from config import TREND_TOP_COUNT

class TrendAnalyzerInput(BaseModel):
    force_analysis: bool = Field(default=False, description="트렌드 분석 강제 실행.")

class TrendAnalyzerTool(BaseTool):
    name = "trend_analyzer"
    description = (
        "암호화폐 시장 트렌드(급등/급락)를 분석합니다. "
        "다음과 같은 경우에 이 도구를 사용하세요: "
        "1) 어떤 코인들이 크게 급등하거나 급락하고 있는지 탐지해야 할 때 "
        "2) 가격 변동을 바탕으로 시장 코멘터리를 생성하고 싶을 때 "
        "3) 변동성이 큰 시장 상황을 식별해야 할 때 "
        "4) 트렌딩 코인에 대한 콘텐츠를 만들고 싶을 때 "
        "5) 시장 분석의 첫 번째 단계로 시장 동향을 파악해야 할 때 "
        "반환: 급등/급락 코인 목록과 가격/거래량 데이터, 시장 감정 요약."
    )
    args_schema = TrendAnalyzerInput

    def _run(self, force_analysis: bool = False) -> str:
        try:
            logger.info("[Tool] 트렌드 분석기 실행 중...")
            result = analyze_coin_trends()
            if "error" in result:
                return f"트렌드 분석 실패: {result['error']}"
            
            trends = result.get("trends", {})
            surge_coins = trends.get("surge", [])
            crash_coins = trends.get("crash", [])
            
            summary = f"[트렌드 분석]\n급등 코인: {len(surge_coins)}개\n"
            if surge_coins:
                summary += "주요 급등 코인:\n"
                for coin in surge_coins[:TREND_TOP_COUNT]:
                    summary += f"- {coin['coin']}: {coin.get('change_24h', 'N/A')}%\n"
            
            summary += f"\n급락 코인: {len(crash_coins)}개\n"
            if crash_coins:
                summary += "주요 급락 코인:\n"
                for coin in crash_coins[:TREND_TOP_COUNT]:
                    summary += f"- {coin['coin']}: {coin.get('change_24h', 'N/A')}%\n"
            
            market_summary = result.get("market_summary", {})
            if market_summary:
                summary += f"\n시장 감정: {market_summary.get('overall_sentiment', 'N/A')}"
            
            return summary
        except Exception as e:
            return f"트렌드 분석 오류: {str(e)}"

    def _arun(self, force_analysis: bool = False) -> str:
        return self._run(force_analysis) 