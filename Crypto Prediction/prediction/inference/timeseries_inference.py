# inference/timeseries_inference.py

import torch
import numpy as np
import pandas as pd
from datetime import datetime
import joblib

from config import MODEL_PATH, SCALER_PATH, normalize_asset_name, FEATURE_COLS
from models.timeseries_model import load_model_from_checkpoint
from data.preprocess import generate_sliding_windows, aggregate_asset_scores
from data.supabase_io import save_prediction, load_recent_predictions

# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Scaler loader function
def load_scaler():
    return joblib.load(SCALER_PATH)

# Load and prepare trained model
def load_trained_model():
    model = load_model_from_checkpoint(MODEL_PATH)
    model.to(device)
    model.eval()
    
    # Extract model args (for window_size, etc.)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model_args = checkpoint["model_args"]
    
    return model, model_args

# Inference for a single window
def run_timeseries_inference(model, input_features, asset=None):
    input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        logits = output["logits"]
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    finalscore = float((probs[1] - probs[0]) * 100)
    prediction = {
        "asset": asset,
        "probabilities": probs.tolist(),
        "finalscore": finalscore,
        "pricetrend": int(np.argmax(probs) + 1)
    }
    return prediction

# Run the full pipeline
def run_timeseries_pipeline(merged_df, feature_cols=None, use_latest_only=True):
    if feature_cols is None:
        # Use the same feature columns as training code
        feature_cols = FEATURE_COLS

    model, model_args = load_trained_model()
    scaler = load_scaler()
    window_size = model_args["window_size"]
    cached_predictions = load_recent_predictions()

    assets = merged_df["asset"].unique()
    for original_asset in assets:
        # Normalize asset name
        normalized_asset = normalize_asset_name(original_asset)
        asset_df = merged_df[merged_df["asset"] == original_asset].copy()
        print(f"\n[🔍 Asset] {original_asset} -> {normalized_asset} - {len(asset_df)} rows")

        if len(asset_df) < window_size:
            continue

        try:
            asset_df[feature_cols] = scaler.transform(asset_df[feature_cols])
        except Exception as e:
            print(f"[❌ Scaling Error] {original_asset}: {e}")
            continue

        windows = generate_sliding_windows(asset_df, window_size, feature_cols)
        if not windows:
            continue

        if use_latest_only:
            latest_window = windows[-1]
            try:
                result = run_timeseries_inference(model, latest_window, asset=normalized_asset)
                if result.get("finalscore") is not None:
                    final_result = {
                        "asset": normalized_asset,  # Use normalized asset name
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "finalscore": round(result["finalscore"], 4),
                        "pricetrend": "up" if result["finalscore"] > 0 else "down"
                    }
                    save_prediction(final_result, cached=cached_predictions, table_name="predictions")
            except Exception as e:
                print(f"[❌ Inference Error] {original_asset}: {e}")
        else:
            window_scores = []
            for idx, window in enumerate(windows):
                try:
                    result = run_timeseries_inference(model, window, asset=normalized_asset)
                    if result.get("finalscore") is not None:
                        window_scores.append(result["finalscore"])
                except Exception as e:
                    print(f"[❌ Inference Error] {original_asset} (window {idx}): {e}")
            if window_scores:
                agg_result = aggregate_asset_scores(window_scores)
                final_result = {
                    "asset": normalized_asset,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    **agg_result
                }
                save_prediction(final_result, cached=cached_predictions, table_name="predictions")
