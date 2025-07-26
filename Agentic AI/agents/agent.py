# agents/agent.py - AI Agent (LangChain 전용)

from loguru import logger
from typing import Dict, Any
from config import USE_LANGCHAIN_AGENT

# LangChain Agent 전용
from .langchain_agent import run_langchain_agent


def run_agent() -> dict:
    """메인 AI 에이전트 실행"""
    
    logger.info("[Agent] LangChain Agent 실행")
    return run_langchain_agent()