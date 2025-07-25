# 📊 antMeter Visualization API

This is a lightweight Flask-based API that updates Google Sheets dashboards with real-time cryptocurrency sentiment and technical indicator data fetched from a Supabase database.

It is part of the `antMeter` system, a full-stack crypto analytics platform combining machine learning prediction and business intelligence reporting.

---

## 🚀 Features

- Retrieves data from Supabase (e.g., `sentiment_indicators`, `technical_indicators`)
- Updates Google Sheets dynamically via the Sheets API
- Supports automated updates via GCP Cloud Scheduler
- Containerized for GCP Cloud Run deployment

---

## 📂 Folder Structure

.
├── main.py # Flask API server
├── requirements.txt # Python dependencies
├── Dockerfile # For GCP deployment
├── .env.example # Environment variable template
└── credentials.json # (ignored) GCP service account file


---

## 📌 Environment Setup

Create a `.env` file based on the `.env.example` provided:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-key
SPREADSHEET_ID=your-google-sheet-id
TABLE_SHEET_MAP=sentiment_indicators:Sheet1,technical_indicators:Sheet2
DAY_WINDOW=7
GOOGLE_CREDENTIALS_PATH=credentials.json

🧪 API Endpoint
POST /update
Fetches rows from each Supabase table (within the given DAY_WINDOW) and updates corresponding Google Sheets tabs.

Returns:
{
  "status": "complete",
  "results": {
    "sentiment_indicators": "120 rows written",
    "technical_indicators": "119 rows written"
  }
}

🐳 Run with Docker
docker build -t antmeter-visual-api .
docker run -p 8080:8080 \
  --env-file .env \
  -v $(pwd)/credentials.json:/app/credentials.json \
  antmeter-visual-api

☁️ Deploy to Google Cloud Run
gcloud run deploy antmeter-visual-api \
  --source . \
  --region asia-northeast1 \
  --set-env-vars "SUPABASE_URL=...,SUPABASE_KEY=..." \
  --service-account=your-sa@your-project.iam.gserviceaccount.com

🛡 Notes
credentials.json and .env are excluded via .gitignore

👤 Author
Mincheol Shin