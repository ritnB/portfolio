import sys
import os
import locale

# System encoding setup
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform.startswith('win'):
    os.environ['PYTHONLEGACYWINDOWSSTDIO'] = '1'

# UTF-8 encoding setup
if hasattr(sys, '_getframe'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Locale setup
try:
    locale.setlocale(locale.LC_ALL, 'C.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        pass

# app.py - Flask Server (Cloud Run Endpoint)
from flask import Flask, request, jsonify
import os
import sys

# Set environment variables for Windows compatibility
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONLEGACYWINDOWSSTDIO'] = '1'

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.langchain_agent import create_langchain_agent
from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG
from loguru import logger

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Portfolio Agent is running"})

@app.route('/analyze', methods=['POST'])
def analyze_market():
    """Market analysis endpoint"""
    try:
        # Execute agent
        agent = create_langchain_agent()
        result = agent.run("Analyze the current cryptocurrency market and generate high-quality content.")
        
        return jsonify({
            "status": "success",
            "result": result
        })
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with project information"""
    return jsonify({
        "name": "Portfolio Agent",
        "description": "AI-powered cryptocurrency market analysis agent",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze"
        }
    })

if __name__ == '__main__':
    logger.info(f"Starting Portfolio Agent server on {FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)