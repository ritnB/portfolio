# tools/threads_poster_tool.py - Threads Upload Tool
from langchain.tools import BaseTool
from typing import Optional
import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.threads_poster import upload_to_threads

class ThreadsPosterTool(BaseTool):
    name = "threads_poster"
    description = """
    Upload high-quality crypto market analysis content to social media (Threads).
    Use this tool only when content quality score is 28+ points (0.7+).
    
    Input: The content to post (should be engaging, factual, and under 200 characters)
    Output: Upload status message
    
    Guidelines:
    - Only post content that meets quality standards
    - Focus on factual market insights, not investment advice
    - Include relevant emojis and keep content concise
    """
    
    def _run(self, content: str) -> str:
        """Upload content to Threads"""
        return upload_to_threads(content)
    
    def _arun(self, content: str) -> str:
        """Async version of upload"""
        return self._run(content)

# Check success status (case-insensitive) 