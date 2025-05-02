# 🧠 antMeter – Crypto Trend Forecasting & Monitoring

A full-stack project for analyzing and monitoring cryptocurrency market trends using machine learning and automated dashboards.

---

## 🔹 1. `antMeter/` – ML Training Pipeline

**Purpose:**  
Train models to predict crypto market trends using sentiment and technical indicators.

**Highlights:**  
- Sentiment classification with BERTweet  
- Time-series forecasting with PatchTST  
- EDA-based text augmentation  
- Early stopping with F1 & loss threshold  
- Model outputs: HuggingFace + PyTorch (.pt)

---

## 🔹 2. `antmeter_visualization/` – Dashboard API

**Purpose:**  
Automatically update Google Sheets dashboards with processed data for monitoring.

**Highlights:**  
- Flask API to sync Supabase → Google Sheets  
- Shows data processing status (daily)  
- Deployable to GCP Cloud Run + Cloud Scheduler  
- Secure `.env` and credential handling

---

## ⚙️ Stack

- **ML**: PyTorch, HuggingFace Transformers  
- **Backend**: Flask, gspread, Supabase  
- **Deployment**: Docker, GCP Cloud Run  
- **Scheduling**: GCP Cloud Scheduler  

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
