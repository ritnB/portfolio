# tools/prediction_analyzer_tool.py
from langchain.tools import BaseTool
from typing import Optional
from pydantic import BaseModel, Field
from loguru import logger
from analyzers.prediction_analyzer import analyze_prediction_performance

class PredictionAnalyzerInput(BaseModel):
    force_analysis: bool = Field(default=False, description="예측 분석 강제 실행.")

class PredictionAnalyzerTool(BaseTool):
    name = "prediction_analyzer"
    description = (
        "AI 예측 모델의 성능과 적중률을 분석합니다. "
        "다음과 같은 경우에 이 도구를 사용하세요: "
        "1) AI 예측의 정확도를 확인하고 싶을 때 "
        "2) 예측 성능을 홍보할 만한 내용이 있는지 확인해야 할 때 "
        "3) AI 모델의 신뢰성을 평가하고 싶을 때 "
        "4) 예측 결과에 대한 콘텐츠를 만들고 싶을 때 "
        "5) 시장 분석의 두 번째 단계로 AI 예측 성능을 검증해야 할 때 "
        "반환: 예측 적중률 통계, 성능 지표, 신뢰도 평가."
    )
    args_schema = PredictionAnalyzerInput

    def _run(self, force_analysis: bool = False) -> str:
        try:
            logger.info("[Tool] 예측 분석기 실행 중...")
            result = analyze_prediction_performance()
            if "error" in result:
                return f"예측 분석 실패: {result['error']}"
            accuracy = result.get("accuracy", {})
            overall = accuracy.get("overall", {})
            summary = f"[예측 분석]\n전체 적중률: {overall.get('accuracy', 'N/A')}%\n"
            if "daily" in accuracy:
                daily = accuracy["daily"]
                summary += f"1일 적중률: {daily.get('accuracy', 'N/A')}%\n"
            if "weekly" in accuracy:
                weekly = accuracy["weekly"]
                summary += f"7일 적중률: {weekly.get('accuracy', 'N/A')}%\n"
            performance = result.get("performance", {})
            if performance:
                summary += f"예측 신뢰도: {performance.get('confidence', 'N/A')}%\n"
            return summary
        except Exception as e:
            return f"예측 분석 오류: {str(e)}"

    def _arun(self, force_analysis: bool = False) -> str:
        return self._run(force_analysis) 