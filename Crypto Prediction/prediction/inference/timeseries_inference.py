# inference/timeseries_inference.py

import torch
import numpy as np
import pandas as pd
from datetime import datetime
import joblib

from config import MODEL_PATH, SCALER_PATH, normalize_coin_name, FEATURE_COLS
from models.timeseries_model import load_model_from_checkpoint
from data.preprocess import create_sliding_windows
from data.supabase_io import save_prediction, load_recent_predictions

# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Scaler loader function
def load_scaler():
    return joblib.load(SCALER_PATH)

# Model loading and preparation
def load_trained_model():
    model = load_model_from_checkpoint(MODEL_PATH)
    model.to(device)
    model.eval()
    
    # Extract model args (for necessary info like window_size)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model_args = checkpoint["model_args"]
    
    return model, model_args

# Inference for single window
def run_inference(model, input_features, coin=None):
    """
    Run inference on a single window of features.
    
    Note: Actual prediction logic contains proprietary algorithms.
    """
    input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        logits = output["logits"]
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    # Simplified scoring (actual scoring logic is proprietary)
    confidence_score = float((probs[1] - probs[0]) * 100)
    prediction = {
        "coin": coin,
        "probabilities": probs.tolist(),
        "confidence_score": confidence_score,
        "predicted_trend": int(np.argmax(probs) + 1)
    }
    return prediction

# Execute prediction pipeline
def run_prediction_pipeline(merged_df, feature_cols=None, use_latest_only=True):
    """
    Execute the complete prediction pipeline.
    
    Note: This is a simplified version for portfolio demonstration.
    Actual pipeline contains proprietary optimization and filtering logic.
    """
    if feature_cols is None:
        # Use same feature columns as training code
        feature_cols = FEATURE_COLS

    model, model_args = load_trained_model()
    scaler = load_scaler()
    window_size = model_args["window_size"]
    cached_predictions = load_recent_predictions()

    coins = merged_df["coin"].unique()
    for original_coin in coins:
        # Normalize coin name
        normalized_coin = normalize_coin_name(original_coin)
        coin_df = merged_df[merged_df["coin"] == original_coin].copy()
        print(f"\n[🔍 Processing] {original_coin} -> {normalized_coin} - {len(coin_df)} rows")

        if len(coin_df) < window_size:
            continue

        try:
            # Apply feature scaling
            coin_df[feature_cols] = scaler.transform(coin_df[feature_cols])
        except Exception as e:
            print(f"[❌ Scaling Error] {original_coin}: {e}")
            continue

        # Generate prediction windows (proprietary logic abstracted)
        windows = create_sliding_windows(coin_df, window_size, feature_cols)
        if not windows:
            continue

        predictions = []
        for window in windows:
            pred = run_inference(model, window, normalized_coin)
            predictions.append(pred)

        if not predictions:
            continue

        # Aggregate predictions (proprietary aggregation logic abstracted)
        final_prediction = aggregate_predictions(predictions, normalized_coin)
        
        # Save prediction with timestamp
        prediction_data = {
            "coin": normalized_coin,
            "confidence_score": final_prediction["confidence_score"],
            "predicted_trend": final_prediction["predicted_trend"],
            "timestamp": datetime.utcnow().isoformat(),
            "model_version": "simplified_demo",
            "verified": False
        }

        save_prediction(prediction_data, cached_predictions)
        print(f"[✅ Saved] {normalized_coin}: {final_prediction['predicted_trend']} (confidence: {final_prediction['confidence_score']:.2f})")


def aggregate_predictions(predictions, coin):
    """
    Aggregate multiple predictions for a single coin.
    
    Note: Actual aggregation algorithm is proprietary.
    """
    # Simplified aggregation (actual logic is proprietary)
    if not predictions:
        return {"confidence_score": 0.0, "predicted_trend": 1}
    
    # Basic averaging (actual method is more sophisticated)
    avg_confidence = np.mean([p["confidence_score"] for p in predictions])
    avg_trend = np.mean([p["predicted_trend"] for p in predictions])
    
    return {
        "coin": coin,
        "confidence_score": round(avg_confidence, 2),
        "predicted_trend": int(round(avg_trend))
    }


# Legacy function name for backward compatibility
def run_timeseries_inference(model, input_features, coin=None):
    return run_inference(model, input_features, coin)

# Legacy function name for backward compatibility  
def run_timeseries_pipeline(merged_df, feature_cols=None, use_latest_only=True):
    return run_prediction_pipeline(merged_df, feature_cols, use_latest_only)
