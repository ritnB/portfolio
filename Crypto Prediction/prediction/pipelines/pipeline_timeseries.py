from inference.timeseries_inference import run_timeseries_pipeline
from data.supabase_io import load_technical_indicators
from utils.gcs_utils import get_latest_model_filename, download_from_gcs
from config import GCS_BUCKET_NAME


def download_latest_model_and_scaler():
    print("=== [☁️] Downloading latest model and scaler from GCS ===")
    
    incremental_model = get_latest_model_filename(GCS_BUCKET_NAME, prefix="models/patchseq_incremental_")
    if incremental_model:
        print("📈 Using incremental model")
        download_from_gcs(GCS_BUCKET_NAME, incremental_model, "/tmp/latest_model.pt")
        scaler_blob = incremental_model.replace("patchseq_incremental_", "scaler_incremental_").replace(".pt", ".pkl")
        download_from_gcs(GCS_BUCKET_NAME, scaler_blob, "/tmp/latest_scaler.pkl")
        return
    
    model_blob = get_latest_model_filename(GCS_BUCKET_NAME, prefix="models/patchseq_final_model_")
    if model_blob:
        print("🔄 Using fully retrained model")
        download_from_gcs(GCS_BUCKET_NAME, model_blob, "/tmp/latest_model.pt")
        scaler_blob = model_blob.replace("patchseq_final_model_", "scaler_standard_").replace(".pt", ".pkl")
        download_from_gcs(GCS_BUCKET_NAME, scaler_blob, "/tmp/latest_scaler.pkl")
    else:
        print("[⚠️] Could not find latest model in GCS")


def main():
    print("=== [🚀] Timeseries Pipeline Starting ===")

    download_latest_model_and_scaler()

    # Load technical indicators from Supabase
    df = load_technical_indicators()
    if df.empty:
        print("No technical indicator data found.")
        return

    # Run pipeline
    run_timeseries_pipeline(df, use_latest_only=True)

    print("=== [✅] Timeseries Pipeline Finished ===")


if __name__ == "__main__":
    main()
