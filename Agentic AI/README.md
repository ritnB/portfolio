# 🤖 Cryptocurrency Analysis & Auto-Posting Bot

A sophisticated AI-powered system that analyzes cryptocurrency market data and automatically generates social media content based on market trends, prediction accuracy, and community sentiment.

## 📋 Project Overview

This Flask-based application runs on cloud infrastructure to perform comprehensive cryptocurrency market analysis every 12 hours. It leverages multiple data sources and AI models to generate intelligent, data-driven social media posts about cryptocurrency trends.

## 🏗️ Architecture

```
crypto-bot/
├── 🚀 app.py                          # Flask server & API endpoints
├── ⚙️ config.py                       # Centralized configuration management
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Container deployment configuration
│
├── 🤖 agents/
│   └── agent.py                       # Main AI agent orchestration logic
│
├── 🔍 analyzers/                      # Market analysis modules
│   ├── __init__.py
│   ├── trend_analyzer.py             # EMA-based surge/crash analysis
│   ├── prediction_analyzer.py        # AI prediction accuracy analysis
│   └── sentiment_analyzer.py         # Community sentiment analysis
│
├── 🎯 strategies/                     # Content generation strategies
│   ├── __init__.py
│   ├── content_generator.py          # Context-aware content creation
│   └── promotion_strategy.py         # Promotional content strategy
│
├── 🤖 llm/                           # LLM Provider abstraction
│   ├── __init__.py
│   └── providers.py                  # OpenAI/HuggingFace/Ollama support
│
├── 🔗 chains/
│   └── thread_chain.py               # LangChain integration
│
├── 🛠️ tools/
│   └── threads_poster.py             # Social media API integration
│
├── 📊 evaluation/
│   └── thread_quality_eval.py       # Content quality assessment system
│
├── 📝 prompts/
│   └── thread_prompt.txt             # Optimized LLM prompts
│
└── 🧪 tests/
    ├── test_agent.py                 # Mock-based agent testing
    ├── test_tools.py                 # Analyzer & tool tests
    └── test_chain.py                 # Content generation tests
```

## 🔧 Key Features

### 📊 Multi-Layer Analysis System
- **Trend Analysis**: EMA-based surge/crash detection (10%+ price movements)
- **Prediction Performance**: AI model accuracy analysis (3-day 70%, 1-day 80% thresholds)
- **Community Sentiment**: Coin-specific comment sentiment analysis and hot topic extraction

### 🎯 Smart Content Strategy
- **Priority-Based Generation**: Surge/crash + prediction accuracy → accuracy promotion → community trends
- **Safety Measures**: Investment advice prevention, exaggeration filtering
- **Platform Optimization**: Character limits, emoji integration, hashtag optimization

### 🤖 Multi-Model LLM Support
- **Fallback System**: Ollama → HuggingFace → OpenAI priority chain
- **Provider Abstraction**: Easy model switching via configuration
- **Cost Optimization**: Free model prioritization

### 🔍 Advanced Quality Evaluation
- **Multi-Metric Assessment**: Basic metrics + LLM evaluation + safety checks
- **Weighted Scoring**: Data accuracy (40%) + market relevance (30%) + engagement (20%) + risk (10%)
- **Strict Standards**: Configurable quality thresholds

## ⚙️ Technology Stack

- **Backend**: Python Flask, LangChain
- **AI/ML**: OpenAI GPT, HuggingFace Transformers, Ollama
- **Database**: Supabase (PostgreSQL)
- **Deployment**: Docker, Cloud Run
- **Testing**: Pytest with mock-based testing
- **Monitoring**: Loguru logging system

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- Docker (for deployment)
- API keys for chosen LLM providers

### Local Development
```bash
# Clone repository
git clone <repository-url>
cd crypto-bot

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp env.example .env
# Edit .env file with your configuration

# Run development server
python app.py

# Test the agent
curl -X POST http://localhost:8081/agent
```

### Testing
```bash
# Run all tests (mock-based, no API costs)
pytest tests/

# Run specific test suite
pytest tests/test_agent.py::test_run_agent_with_mocks

# Integration tests (actual API calls - costs apply)
pytest tests/ -k "integration"
```

### Docker Deployment
```bash
# Build container
docker build -t crypto-analysis-bot .

# Deploy to cloud platform
# (Example for Google Cloud Run)
gcloud run deploy crypto-bot \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

## ⚙️ Configuration

### Environment Variables
```bash
# 🔑 API Keys
OPENAI_API_KEY=your_openai_key_here
HUGGINGFACE_API_KEY=your_huggingface_key_here
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
SOCIAL_MEDIA_API_KEY=your_social_api_key

# 🤖 LLM Configuration
LLM_PROVIDER=openai                    # openai, huggingface, ollama
LLM_MODEL=gpt-4o-mini                  # Model name
FREE_MODEL_FALLBACK=true               # Use free models first

# 📊 Analysis Thresholds
SURGE_THRESHOLD=10.0                   # Surge detection threshold (%)
CRASH_THRESHOLD=-10.0                  # Crash detection threshold (%)
ACCURACY_THRESHOLD_3DAY=70.0           # 3-day accuracy promotion threshold
ACCURACY_THRESHOLD_1DAY=80.0           # 1-day accuracy promotion threshold

# ✍️ Content Settings
MAX_CONTENT_LENGTH=100                 # Maximum character count
CONTENT_STYLE=social                   # Content style preference
USE_EMOJIS=true                       # Enable emoji usage

# 🎯 Quality Control
QUALITY_PASS_SCORE=70.0               # Quality threshold score
STRICT_EVALUATION=true                # Enable strict evaluation mode

# 🔄 Scheduling
AUTO_POSTING_ENABLED=false            # Enable/disable auto-posting
SCHEDULER_INTERVAL_HOURS=12           # Execution interval
```

## 📈 Workflow

1. **📊 Data Collection**: Gather 3-day historical EMA, predictions, and community data
2. **🔍 Analysis Execution**: Run surge/crash, accuracy, and sentiment analysis in parallel
3. **🎯 Strategy Decision**: Evaluate promotional opportunities and content priorities
4. **✍️ Content Generation**: Create context-appropriate optimized content
5. **🔍 Quality Verification**: Multi-layer quality assessment system
6. **📤 Auto-Upload**: Automatic posting upon quality standard compliance

## 🛡️ Safety Features

- **Investment Advice Prevention**: Automatic filtering of financial advice keywords
- **Exaggeration Removal**: Filter words like "guaranteed", "certain", etc.
- **Risk Management**: Optional disclaimer inclusion
- **Quality Gates**: Automatic blocking for substandard content
- **Error Recovery**: Step-by-step fallback handling

## 📊 API Endpoints

### Start Analysis
```bash
POST /agent
Content-Type: application/json

Response:
{
  "status": "uploaded|blocked|error",
  "content": "Generated content text",
  "content_type": "surge_with_prediction|accuracy_promotion|community_trend",
  "analysis_summary": {...},
  "evaluation": {...}
}
```

### Health Check
```bash
GET /
Response: "Crypto Analysis Bot is running"
```

## 🔧 Extensibility

- **New Analyzers**: Add modules to `analyzers/` directory
- **Content Strategies**: Extend `strategies/` with new approaches
- **LLM Models**: Add providers to `llm/providers.py`
- **Quality Criteria**: Adjust weights in `config.py`

## 📝 Development Notes

- All configurations controllable via environment variables
- Mock-based testing for cost-free development
- UTF-8 optimized output handling
- Cloud platform logging support
- Cost optimization through free model prioritization

## 📄 License

This project is intended for educational and portfolio demonstration purposes.

## 🤝 Contributing

This is a portfolio project. For questions or discussions about the implementation, please feel free to reach out. 