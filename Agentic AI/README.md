# Portfolio Agent - AI-Powered Crypto Market Analysis

A LangChain-based AI agent that analyzes cryptocurrency market data and generates social media content. This project demonstrates the implementation of a Plan-And-Execute agent architecture with memory capabilities.

## 🏗️ Project Structure

```
portfolio-agent/
├── app.py                          # Flask server
├── config.py                       # Configuration management
├── requirements.txt                # Dependencies
├── Dockerfile                      # Docker deployment
├── agents/                         # AI agents
│   ├── agent.py                    # Main agent
│   └── langchain_agent.py         # LangChain PlanAndExecute Agent
├── analyzers/                      # Data analysis modules
│   ├── trend_analyzer.py          # Market trend analysis
│   ├── prediction_analyzer.py     # Prediction accuracy analysis
│   └── sentiment_analyzer.py      # Community sentiment analysis
├── tools/                          # LangChain Tools
│   ├── trend_analyzer_tool.py     # Trend analysis tool
│   ├── prediction_analyzer_tool.py # Prediction analysis tool
│   ├── sentiment_analyzer_tool.py # Sentiment analysis tool
│   ├── threads_poster_tool.py     # Social media posting tool
│   └── threads_poster.py          # Posting module
├── llm/                           # LLM providers
│   └── providers.py               # LLM provider implementations
└── prompts/                       # Prompt templates
    ├── agent_system_prompt.txt    # System prompt
    ├── planner_prompt.txt         # Planner prompt
    ├── executor_prompt.txt        # Executor prompt
    ├── content_generator_prompt.txt # Content generation
    └── quality_evaluator_prompt.txt # Quality evaluation
```

## 🔄 Architecture

### Plan-And-Execute + Memory
- **Planner**: Creates comprehensive analysis strategies
- **Executor**: Follows plans step-by-step using available tools
- **Memory**: Remembers previous actions for consistency

### Quality-Based Content Generation
- Content quality evaluation (40-point scale)
- Automatic regeneration for low-quality content
- Posting only high-quality content (28+ points)

## 🚀 Features

- **Market Trend Analysis**: Detects surging and crashing cryptocurrencies
- **Prediction Analysis**: Evaluates AI prediction accuracy
- **Sentiment Analysis**: Analyzes community sentiment and mood
- **Content Generation**: Creates engaging social media posts
- **Quality Evaluation**: Ensures content meets safety and accuracy standards
- **Social Media Integration**: Posts high-quality content automatically

## 🛠️ Technologies

- **LangChain**: Agent framework with Plan-And-Execute architecture
- **Flask**: Web server for API endpoints
- **Python**: Core programming language
- **Docker**: Containerization for deployment

## 📋 Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🔧 Configuration

The project uses a configuration system that supports multiple LLM providers and analysis parameters. Sensitive configuration details have been redacted for public release.

## 📝 Notes

- This is a portfolio project demonstrating AI agent implementation
- Sensitive configuration and prompt details have been redacted
- The project showcases LangChain's Plan-And-Execute pattern with memory
- Content focuses on factual market insights, not investment advice 