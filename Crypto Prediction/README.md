✨ Antmeter (Redacted) — Production‑Style ML Timeseries Pipeline

A portfolio‑friendly, security‑sanitized showcase of an end‑to‑end ML pipeline for timeseries classification. It highlights engineering practices (pipelines, model training, inference, verification, incremental learning) while redacting sensitive details.

🧭 Table of Contents
- 🌟 Highlights
- 🧱 Architecture at a Glance
- 📂 Project Structure
- ⚙️ Quickstart
- 🐳 Docker
- 🌐 API Endpoints
- 🔐 Security & Redaction
- 🧪 Config Philosophy
- 🔧 Environment Variables

🌟 Highlights
- 🧠 Generic patch‑based sequence classifier (CLS/mean pooling) with optional FocalLoss
- 🔁 Time‑aware CV (leakage‑aware rolling splits) and consistent preprocessing
- ♻️ Incremental learning with validation gating and safety checks
- 📈 Batched post‑inference enrichment (e.g., price updates) via external API
- ☁️ Artifact management through cloud storage (model/scaler discovery)
- 🧰 Clean module boundaries, cache hygiene, and memory‑safe utilities

🧱 Architecture at a Glance
- Data I/O: Load technical indicators from a backend (e.g., Supabase)
- Training: Optuna search → final training → artifacts (model/scaler)
- Inference: Download latest artifacts → window generation → prediction → persistence
- Verification: Time‑aligned validation of predictions against observed outcomes
- Incremental: Lightweight fine‑tuning with performance guards → re‑upload artifacts

📂 Project Structure
```
project_root/
├── app.py                       # Flask entrypoint (API)
├── config.py                    # Centralized config (env‑driven, redacted)
├── Dockerfile                   # Container build
├── requirements.txt             # Dependencies
├── data/
│   ├── preprocess.py            # Sliding windows, datasets, CV helpers
│   ├── supabase_io.py           # Backend I/O utils
│   └── __init__.py
├── inference/
│   └── timeseries_inference.py  # Inference + persistence + enrichment
├── models/
│   └── timeseries_model.py      # PatchSequenceModel + checkpoint loader
├── pipelines/
│   ├── pipeline_timeseries.py   # Orchestrates inference pipeline
│   ├── pipeline_verify.py       # Validates past predictions
│   ├── pipeline_retrain.py      # Threshold‑based retraining trigger
│   ├── pipeline_incremental.py  # Incremental learning cycle
│   └── pipeline_labeling.py     # Label generation with future lookup
├── trainers/
│   └── train_patchtst.py        # Training with Optuna
├── utils/
│   ├── gcs_utils.py             # Cloud storage helpers
│   ├── memory_utils.py          # Memory hygiene helpers
│   ├── price_utils.py           # External price API helpers
│   ├── timestamp_utils.py       # Timestamp parsing utilities
│   └── training_utils.py        # Training callbacks/utilities
└── debug_tools/
    └── debug_tools.py           # Simple diagnostics printers
```

⚙️ Quickstart
1) Prerequisites
   - Python 3.10+
   - `pip install -r requirements.txt`

2) Environment
   - Create a `.env` with the required variables (see section below)

3) Run the API
   - Local (gunicorn):
     ```bash
     gunicorn --timeout 3600 --bind 0.0.0.0:8080 app:app
     ```
   - Health check:
     ```bash
     curl http://localhost:8080/
     ```

🐳 Docker
- Build
  ```bash
  docker build -t antmeter .
  ```
- Run (with env file)
  ```bash
  docker run -p 8080:8080 --env-file .env antmeter
  ```

🌐 API Endpoints
- GET `/`            → Health
- POST `/timeseries` → Run timeseries inference (downloads latest artifacts)
- POST `/verify`     → Verify historical predictions
- POST `/retrain`    → Trigger retraining if accuracy falls below threshold
- POST `/incremental`→ Run incremental update with validation gating
- POST `/labeling`   → Generate labels with future‑aware lookup

🔐 Security & Redaction
- No secrets or identifiers are committed. All sensitive values are env‑driven.
- Cloud buckets/keys/URLs are redacted and must be provided via environment.
- Coin mappings are minimized; provide your own via `COIN_INFO_JSON` when needed.
- `.gitignore` includes `.env`, `secrets/`, and common sensitive patterns.

🧪 Config Philosophy
- `config.py` centralizes all configuration with strong sanitization:
  - Generic placeholders (e.g., `GENERIC_*`) replace tuned numeric values.
  - Ranges and thresholds are expressed as neutral defaults.
  - Optuna search space scales from a generic base (`OPTUNA_BASE`).
  - Everything is overrideable via environment variables for real deployments.

🔧 Environment Variables (subset)
- Required (examples)
  - `SUPABASE_URL`, `SUPABASE_KEY`                      # backend access
  - `GCS_BUCKET_NAME`, `GCS_MODEL_DIR`                  # artifact storage
- Optional (generalized defaults applied if absent)
  - `MODEL_PATH`, `SCALER_PATH`
  - `COIN_INFO_JSON`                                    # custom coin mapping JSON
  - `PREDICTION_EVAL_DAYS`, `THRESHOLD_ACCURACY`
  - `INCREMENTAL_*` (e.g., `INCREMENTAL_EPOCHS`, `INCREMENTAL_LR`, etc.)
  - `ROLLING_CV_*`, `N_TRIALS`, `MAX_TIME_GAP_HOURS`
  - External API knobs (e.g., `COINGECKO_*`)

💬 Notes
- This repository is intentionally sanitized for public presentation. Replace placeholders with real values in your private environment.
- The pipeline modules are designed to be composable; feel free to swap the data source, model head, or storage backend.