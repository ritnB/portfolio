# agents/langchain_agent.py - LangChain Agent with Memory and Quality Control
from langchain.agents import AgentExecutor, create_plan_and_execute_agent
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage
from langchain.tools import BaseTool
from langchain_community.llms import HuggingFaceHub
from langchain_openai import ChatOpenAI
from loguru import logger
import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.trend_analyzer_tool import TrendAnalyzerTool
from tools.sentiment_analyzer_tool import SentimentAnalyzerTool
from tools.prediction_analyzer_tool import PredictionAnalyzerTool
from tools.threads_poster_tool import ThreadsPosterTool
from llm.providers import get_llm_provider
from config import AGENT_MAX_ITERATIONS, AGENT_VERBOSE

def load_prompt_from_file(filename: str) -> str:
    """Load prompt from file"""
    try:
        prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", filename)
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Failed to load prompt from {filename}: {e}")
        return ""

def create_langchain_agent() -> AgentExecutor:
    """Create LangChain Plan-And-Execute Agent with Memory"""
    try:
        # Load prompts from files
        system_prompt = load_prompt_from_file("agent_system_prompt.txt")
        planner_prompt = load_prompt_from_file("planner_prompt.txt")
        executor_prompt = load_prompt_from_file("executor_prompt.txt")
        
        # Get LLM provider
        llm = get_llm_provider()
        
        # Create tools
        tools = [
            TrendAnalyzerTool(),
            SentimentAnalyzerTool(),
            PredictionAnalyzerTool(),
            ThreadsPosterTool()
        ]
        
        # Create memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create prompt templates
        from langchain.prompts import PromptTemplate
        
        system_prompt_template = PromptTemplate(
            input_variables=[],
            template=system_prompt
        )
        
        planner_prompt_template = PromptTemplate(
            input_variables=["input", "chat_history"],
            template=planner_prompt
        )
        
        executor_prompt_template = PromptTemplate(
            input_variables=["input", "chat_history", "agent_scratchpad"],
            template=executor_prompt
        )
        
        # Create agent
        agent = create_plan_and_execute_agent(
            llm=llm,
            tools=tools,
            system_prompt=system_prompt_template,
            planner_prompt=planner_prompt_template,
            executor_prompt=executor_prompt_template,
            verbose=AGENT_VERBOSE,
            max_iterations=AGENT_MAX_ITERATIONS
        )
        
        # Create agent executor with memory
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=AGENT_VERBOSE,
            max_iterations=AGENT_MAX_ITERATIONS,
            handle_parsing_errors=True
        )
        
        logger.info("LangChain Agent created successfully")
        return agent_executor
        
    except Exception as e:
        logger.error(f"Failed to create LangChain agent: {e}")
        raise

def run_agent_with_memory(agent: AgentExecutor, task: str) -> str:
    """Run agent with memory for consistent execution"""
    try:
        # Define task to execute
        result = agent.run(task)
        return result
        
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        return f"Agent execution failed: {str(e)}" 