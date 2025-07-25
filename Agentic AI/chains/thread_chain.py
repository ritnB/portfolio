# chains/thread_chain.py - 수정된 체인
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI  # ✅ 올바른 import
from config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE

with open("prompts/thread_prompt.txt", "r", encoding="utf-8") as f:
    prompt_template = f.read()

prompt = PromptTemplate(
    input_variables=["market_ema_trend", "market_prediction_info", "community_info"],
    template=prompt_template,
)


def generate_thread_content(
    market_ema_trend: str,
    prediction_info: str,
    community_info: str,
) -> str:
    """기존 호환성을 위한 함수 (Deprecated)"""
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=OPENAI_API_KEY
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    result = chain.run({
        "market_ema_trend": market_ema_trend,
        "market_prediction_info": prediction_info,
        "community_info": community_info
    })

    return result.strip()