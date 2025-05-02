# 🧠 Crypto Trend Forecasting & Monitoring

A full-stack project for analyzing and monitoring cryptocurrency market trends using machine learning and automated dashboards.

---

## 🔹 1. `Crypto Price Forecasting/` – ML Training Pipeline

**Purpose:**  
Train models to predict crypto market trends using sentiment and technical indicators.

**Highlights:**  
- Sentiment classification with BERTweet  
- Time-series forecasting with PatchTST  
- EDA-based text augmentation  
- Early stopping with F1 & loss threshold  
- Model outputs: HuggingFace + PyTorch (.pt)

---

## 🔹 2. `Automated Data Processing Status Dashboard/` – Dashboard API

**Purpose:**  
Automatically update Google Sheets dashboards with processed data for monitoring.

**Highlights:**  
- Flask API to sync Supabase → Google Sheets  
- Shows data processing status (daily)  
- Deployable to GCP Cloud Run + Cloud Scheduler  
- Secure `.env` and credential handling

---

## ⚙️ Tech Stack

- **ML/NLP**: PyTorch, Transformers (BERTweet), HuggingFace, scikit-learn
- **Time Series**: PatchTST, custom windowed dataset, CrossEntropy loss
- **Data Handling**: pandas, numpy, joblib, StandardScaler
- **Backend/API**: Flask, Supabase (Python SDK), gspread, Google Sheets API
- **Deployment**: Docker, Gunicorn, GCP Cloud Run, GCP Cloud Scheduler
- **Automation**: Scheduled inference and dashboard updates

---

## 📁 Structure
portfoilo/
├── antMeter/ # ML training pipeline
├── antmeter_visualization/ # Dashboard backend API
└── README.md # ← this file


---

## 👤 Author

Mincheol Shin  
🔗 [LinkedIn](https://www.linkedin.com/in/min-shin-9a8797340/)
🔗 [GitHub](https://github.com/ritnB)
