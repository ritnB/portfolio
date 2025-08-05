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
from pipelines.pipeline_incremental import run_incremental_learning
from utils.gcs_utils import upload_to_gcs
from utils.timestamp_utils import normalize_timestamp_for_query
from utils.memory_utils import safe_memory_cleanup, monitor_memory_usage
from config import THRESHOLD_ACCURACY

# Environment variables and Supabase client setup
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
client = create_client(SUPABASE_URL, SUPABASE_KEY)

# GCS settings (anonymized)
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "your-bucket-name")
GCS_MODEL_DIR = "models"

def calculate_performance_drop(current_performance, previous_performance=None):
    """Calculate the degree of performance drop."""
    if previous_performance is None:
        # Use default if no previous performance
        previous_performance = 0.5  # Assume average performance
    
    return previous_performance - current_performance

def run_retraining_pipeline():
    """v11_2 compatible retraining pipeline"""
    print("=== [📉] v11_2 Compatible Retraining Pipeline Starting ===")
    print(f"📊 Initial memory: {monitor_memory_usage():.1f}MB")

    try:
        recent_days = 7

        # Load recent verified predictions (timestampz compatible)
        now = datetime.utcnow()
        cutoff = now - timedelta(days=recent_days)
        cutoff_str = normalize_timestamp_for_query(cutoff)

        resp = client.table("predictions") \
            .select("timestamp, is_correct, verified") \
            .eq("verified", True) \
            .gte("timestamp", cutoff_str) \
            .execute()

        records = resp.data
        if not records:
            print("⛔ No recent prediction data. Exiting.")
            return {"message": "No recent verified predictions."}, 200

        df = pd.DataFrame(records)
        df['is_correct'] = df['is_correct'].astype(bool)
        current_performance = df['is_correct'].mean()

        print(f"✅ Accuracy for last {recent_days} days: {current_performance:.2%}")

        # Calculate performance drop (use DB or default for previous performance)
        performance_drop = calculate_performance_drop(current_performance)
        print(f"📊 Performance drop: {performance_drop:.2%}")

        # Branch by performance drop
        if current_performance > THRESHOLD_ACCURACY:
            print(f"🎯 Accuracy is good ({current_performance:.2%}) → No retraining needed. Exiting.")
            return {"message": f"✅ Accuracy OK ({current_performance:.2%}), no retraining needed."}, 200
        
        # Memory check
        safe_memory_cleanup()
        
        # Gradual performance drop (< 15%)
        if performance_drop < 0.15:
            print(f"🔄 Gradual performance drop detected → Trying incremental learning")
            try:
                result, status = run_incremental_learning()
                if status == 200:
                    print("✅ Incremental learning successful")
                    return {"message": "Incremental learning completed successfully."}, 200
                else:
                    print("⚠️ Incremental learning failed → Proceeding to full retrain")
            except Exception as e:
                print(f"❌ Incremental learning error: {e} → Proceeding to full retrain")
                safe_memory_cleanup()
        
        # Severe performance drop (>= 15%) or incremental learning failed
        print(f"🚨 Severe performance drop ({current_performance:.2%}) → Starting v11_2 compatible full retrain")

        # v11_2 compatible model training
        print("🔄 Starting v11_2 compatible model training...")
        model_artifacts = train_patchtst_model()
        
        if not model_artifacts:
            print("❌ Model training failed")
            return {"error": "Model training failed"}, 500

        today_str = datetime.utcnow().strftime("%Y%m%d")
        model_name = f"patchtst_final_model_{today_str}.pt"
        scaler_name = f"scaler_standard_{today_str}.pkl"
        
        print(f"📤 Starting GCS upload: {model_name}")

        # Upload to GCS (v11_2 compatible model)
        try:
            upload_to_gcs(model_artifacts["model_path"], GCS_BUCKET_NAME, f"{GCS_MODEL_DIR}/{model_name}")
            upload_to_gcs(model_artifacts["scaler_path"], GCS_BUCKET_NAME, f"{GCS_MODEL_DIR}/{scaler_name}")
            upload_to_gcs(model_artifacts["log_path"], GCS_BUCKET_NAME, f"{GCS_MODEL_DIR}/logs/{today_str}.log")
            
            print(f"✅ v11_2 model GCS upload complete")
            print(f"  - Model: {model_name}")
            print(f"  - Scaler: {scaler_name}")
            print(f"  - Best F1: {model_artifacts.get('best_f1', 'N/A')}")
            
            # Memory cleanup
            safe_memory_cleanup()
            print(f"📊 Final memory: {monitor_memory_usage():.1f}MB")
            
            return {
                "message": f"✅ v11_2 Retraining completed. Model uploaded to GCS.",
                "model_name": model_name,
                "best_f1": model_artifacts.get('best_f1', 0.0)
            }, 200
            
        except Exception as upload_error:
            print(f"❌ GCS upload failed: {upload_error}")
            return {"error": f"GCS upload failed: {upload_error}"}, 500

    except Exception as e:
        print(f"[💥] v11_2 retraining failed: {e}")
        # Memory cleanup on error
        safe_memory_cleanup()
        return {"error": str(e)}, 500

if __name__ == "__main__":
    run_retraining_pipeline()
