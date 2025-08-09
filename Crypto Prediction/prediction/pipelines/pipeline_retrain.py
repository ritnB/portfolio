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
from utils.training_utils import clear_fold_scaler_cache
from config import THRESHOLD_ACCURACY, GCS_BUCKET_NAME, GCS_MODEL_DIR

# Environment and Supabase client
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
client = create_client(SUPABASE_URL, SUPABASE_KEY)


def calculate_performance_drop(current_performance, previous_performance=None):
    """Compute performance drop compared to a previous baseline."""
    if previous_performance is None:
        previous_performance = 0.5  # assume mid performance when unknown
    return previous_performance - current_performance


def run_retraining_pipeline():
    """Retraining pipeline (v11_2 compatible)."""
    print("=== [📉] Retraining Pipeline (v11_2 compatible) Starting ===")
    print(f"📊 Initial memory: {monitor_memory_usage():.1f}MB")

    try:
        recent_days = 7

        # Load recent verified predictions
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
            print("⛔ No recent verified predictions. Exiting.")
            clear_fold_scaler_cache()
            return {"message": "No recent verified predictions."}, 200

        df = pd.DataFrame(records)
        df["is_correct"] = df["is_correct"].astype(bool)
        current_performance = df["is_correct"].mean()

        # minimal logging

        # Calculate performance drop (baseline may come from DB in prod)
        performance_drop = calculate_performance_drop(current_performance)
        # minimal logging

        # If accuracy is above threshold, skip retraining
        if current_performance > THRESHOLD_ACCURACY:
            print("🎯 Accuracy OK → No retraining.")
            clear_fold_scaler_cache()
            return {"message": f"✅ Accuracy OK ({current_performance:.2%}), no retraining needed."}, 200

        # Memory hygiene
        safe_memory_cleanup()

        # Mild degradation (< 15%): try incremental learning first
        if performance_drop < 0.15:
            print("🔄 Mild degradation detected → Trying incremental learning")
            try:
                result, status = run_incremental_learning()
                if status == 200:
                    print("✅ Incremental learning succeeded")
                    clear_fold_scaler_cache()
                    return {"message": "Incremental learning completed successfully."}, 200
                else:
                    print("⚠️ Incremental learning failed → Proceeding to full retraining")
            except Exception as e:
                print(f"❌ Incremental learning error: {e} → Proceeding to full retraining")
                safe_memory_cleanup()
                clear_fold_scaler_cache()

        # Severe degradation or incremental failed → full retrain
        print(f"🚨 Severe degradation ({current_performance:.2%}) → Starting full retraining (v11_2 compatible)")

        # Train model
        print("🔄 Training model (v11_2 compatible)...")
        model_artifacts = train_patchtst_model()

        if not model_artifacts:
            print("❌ Model training failed")
            return {"error": "Model training failed"}, 500

        today_str = datetime.utcnow().strftime("%Y%m%d")
        model_name = f"patchseq_final_model_{today_str}.pt"
        scaler_name = f"scaler_standard_{today_str}.pkl"

        print(f"📤 Uploading to GCS: {model_name}")

        try:
            upload_to_gcs(model_artifacts["model_path"], GCS_BUCKET_NAME, f"{GCS_MODEL_DIR}/{model_name}")
            upload_to_gcs(model_artifacts["scaler_path"], GCS_BUCKET_NAME, f"{GCS_MODEL_DIR}/{scaler_name}")
            upload_to_gcs(model_artifacts["log_path"], GCS_BUCKET_NAME, f"{GCS_MODEL_DIR}/logs/{today_str}.log")

            print("✅ Upload to GCS completed")
            print(f"  - model: {model_name}")
            print(f"  - scaler: {scaler_name}")
            print(f"  - Best F1: {model_artifacts.get('best_f1', 'N/A')}")

            safe_memory_cleanup()
            clear_fold_scaler_cache()
            print(f"📊 Final memory: {monitor_memory_usage():.1f}MB")

            return {
                "message": "✅ Retraining completed. Model uploaded to GCS.",
                "model_name": model_name,
                "best_f1": model_artifacts.get("best_f1", 0.0),
            }, 200

        except Exception as upload_error:
            print(f"❌ GCS upload failed: {upload_error}")
            safe_memory_cleanup()
            clear_fold_scaler_cache()
            return {"error": f"GCS upload failed: {upload_error}"}, 500

    except Exception as e:
        print(f"[💥] Retraining failed: {e}")
        safe_memory_cleanup()
        clear_fold_scaler_cache()
        return {"error": str(e)}, 500


if __name__ == "__main__":
    run_retraining_pipeline()
