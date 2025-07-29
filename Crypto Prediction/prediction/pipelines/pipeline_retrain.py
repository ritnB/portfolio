# pipeline_retrain.py

import os
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from supabase import create_client
from dotenv import load_dotenv
import joblib
import torch

from trainers.train_patchtst import train_patchtst_model
from utils.gcs_utils import upload_to_gcs
from config import THRESHOLD_ACCURACY  # Modified part

# Environment variables and Supabase client setup
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
client = create_client(SUPABASE_URL, SUPABASE_KEY)

# GCS configuration
GCS_BUCKET_NAME = "your-bucket-name"  # Placeholder for public portfolio
GCS_MODEL_DIR = "models"

def run_retraining_pipeline():
    print("=== [📉] Retraining Pipeline Starting ===")

    try:
        recent_days = 7

        # Load recent verified predictions
        now = datetime.utcnow()
        cutoff = now - timedelta(days=recent_days)

        resp = client.table("predictions") \
            .select("timestamp, is_correct, verified") \
            .eq("verified", True) \
            .gte("timestamp", cutoff.isoformat()) \
            .execute()

        records = resp.data
        if not records:
            print("⛔ No recent prediction data. Exiting.")
            return {"message": "No recent verified predictions."}, 200

        df = pd.DataFrame(records)
        df['is_correct'] = df['is_correct'].astype(bool)
        correct_ratio = df['is_correct'].mean()

        print(f"✅ Recent {recent_days} days accuracy: {correct_ratio:.2%}")

        if correct_ratio > THRESHOLD_ACCURACY:  # Modified part
            print(f"🎯 Good accuracy ({correct_ratio:.2%}) → No retraining needed. Exiting.")
            return {"message": f"✅ Accuracy OK ({correct_ratio:.2%}), no retraining needed."}, 200
        print(f"🚨 Low accuracy → Starting retraining")

        # Perform model training
        model_artifacts = train_patchtst_model()

        today_str = datetime.utcnow().strftime("%Y%m%d")
        model_name = f"patchtst_final_model_{today_str}.pt"
        scaler_name = f"scaler_standard_{today_str}.pkl"

        # Upload to GCS
        upload_to_gcs(model_artifacts["model_path"], GCS_BUCKET_NAME, f"{GCS_MODEL_DIR}/{model_name}")
        upload_to_gcs(model_artifacts["scaler_path"], GCS_BUCKET_NAME, f"{GCS_MODEL_DIR}/{scaler_name}")
        upload_to_gcs(model_artifacts["log_path"], GCS_BUCKET_NAME, f"{GCS_MODEL_DIR}/logs/{today_str}.log")

        print(f"✅ Model and logs uploaded to GCS successfully")
        return {"message": f"✅ Retraining done. Model uploaded to GCS."}, 200

    except Exception as e:
        print(f"[💥] Retraining failed: {e}")
        return {"error": str(e)}, 500

if __name__ == "__main__":
    run_retraining_pipeline()
