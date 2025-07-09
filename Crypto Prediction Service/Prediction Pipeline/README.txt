antMeter/
├── .dockerignore
├── .env
├── config.py
├── Dockerfile
├── main.py
├── requirements.txt
├── data/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocess.py
│   └── supabase_io.py
├── debug_tools/
│   └── debug_tools.py
├── inference/
│   ├── sentiment_inference.py
│   └── timeseries_inference.py
└── models/
    ├── sentiment_model_hf
    ├── sentiment_tokenizer_hf
    ├── timeseries_model.pt
    ├── scaler_standard.pkl
    ├── sentiment_model.py
    └── timeseries_model.py

Model Training: v6_modelTuning.ipynb
---

## 🔄 End-to-End Data ETL & Inference Flow

### 🗂️ 1. Data Ingestion
- **External Crawler** (handled by another contributor) collects:
  - Public comments and posts (e.g., from Reddit, community forums)
  - Technical indicators (e.g., MACD, RSI) from external APIs
- These are stored in Supabase tables: `comments`, `technical_indicators`

### 🧼 2. ETL & Preprocessing (`data/`)
- `data_loader.py`:
  - Loads paginated Supabase data (batch size 1000, offset logic)
  - Applies `gte(timestamp, start_date)` filters based on `recent_days` (masked in public version)
- `preprocess.py`:
  - Normalizes coin names using a mapping dictionary
  - Fills missing values, drops unused columns (e.g., `price_trend`)
  - Generates sliding windows for time-series model
- All preprocessing is done before inference and shares logic across training & inference

### 🤖 3. Model Inference
- `sentiment_inference.py`:
  - Tokenizes comment texts using a pretrained HuggingFace tokenizer
  - Scores sentiment using a fine-tuned BERT-based model
  - Aggregates scores by date/coin and saves to Supabase (`your_sentiment_table`)
- `timeseries_inference.py`:
  - Uses preprocessed sentiment + technical indicators
  - Applies sliding windows per coin
  - Predicts directional price trend using PatchTST (Transformer)
  - Aggregated result saved to Supabase (`your_predictions_table`)

### 💾 4. Data Persistence
- Results are upserted via Supabase’s REST API using the official Python SDK
- Conflict resolution logic (e.g., `timestamp + coin`) is omitted in public version
- Results are later consumed by a visualization frontend (not included here)

---


