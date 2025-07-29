# pipeline_timeseries.py

from inference.timeseries_inference import run_timeseries_pipeline
from data.supabase_io import load_technical_indicators

# Additional imports
from utils.gcs_utils import get_latest_model_filename, download_from_gcs
from config import GCS_BUCKET_NAME

# Download latest model and scaler from GCS
def download_latest_model_and_scaler():
    print("=== [☁️] Downloading latest model and scaler from GCS ===")
    
    # 1. Try incremental learning model first
    incremental_model = get_latest_model_filename(GCS_BUCKET_NAME, prefix="models/patchtst_incremental_")
    if incremental_model:
        print("📈 Using incremental learning model")
        download_from_gcs(GCS_BUCKET_NAME, incremental_model, "/tmp/latest_model.pt")
        scaler_blob = incremental_model.replace("patchtst_incremental_", "scaler_incremental_").replace(".pt", ".pkl")
        download_from_gcs(GCS_BUCKET_NAME, scaler_blob, "/tmp/latest_scaler.pkl")
        return
    
    # 2. Try complete retraining model
    model_blob = get_latest_model_filename(GCS_BUCKET_NAME, prefix="models/patchtst_final_model_")
    if model_blob:
        print("🔄 Using complete retraining model")
        download_from_gcs(GCS_BUCKET_NAME, model_blob, "/tmp/latest_model.pt")
        scaler_blob = model_blob.replace("patchtst_final_model_", "scaler_standard_").replace(".pt", ".pkl")
        download_from_gcs(GCS_BUCKET_NAME, scaler_blob, "/tmp/latest_scaler.pkl")
    else:
        print("[⚠️] Could not find latest model file.")

def main():
    print("=== [🚀] Timeseries Pipeline Starting ===")

    # Download latest model/scaler from GCS
    download_latest_model_and_scaler()

    # Load technical indicator data from Supabase
    df = load_technical_indicators()
    if df.empty:
        print("No technical indicator data found.")
        return

    # Run pipeline
    run_timeseries_pipeline(df, use_latest_only=True)

    print("=== [✅] Timeseries Pipeline Finished ===")

if __name__ == "__main__":
    main()
