# inference/timeseries_inference.py

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import datetime
import joblib

from config import MODEL_PATH, SCALER_PATH, normalize_coin_name, FEATURE_COLS, DEFAULT_CLASSIFICATION_THRESHOLD
from models.timeseries_model import load_model_from_checkpoint
from data.preprocess import generate_sliding_windows, aggregate_coin_scores
from data.supabase_io import save_prediction, load_recent_predictions, update_prediction_prices
from utils.price_utils import fetch_batch_prices

# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Scaler loader function (prefer GCS download)
def load_scaler():
    # Use latest scaler downloaded from GCS if available
    gcs_scaler_path = "/tmp/latest_scaler.pkl"
    if os.path.exists(gcs_scaler_path):
        print(f"📥 Using latest scaler from GCS: {gcs_scaler_path}")
        return joblib.load(gcs_scaler_path)
    
    # Fallback: use default path from config
    print(f"⚠️ Using default scaler: {SCALER_PATH}")
    return joblib.load(SCALER_PATH)

# Model loader and preparation (prefer GCS download)
def load_trained_model():
    # Use latest model downloaded from GCS if available
    gcs_model_path = "/tmp/latest_model.pt"
    if os.path.exists(gcs_model_path):
        print(f"📥 Using latest model from GCS: {gcs_model_path}")
        model_path = gcs_model_path
    else:
        # Fallback: use default path from config
        print(f"⚠️ Using default model: {MODEL_PATH}")
        model_path = MODEL_PATH
    
    # Load with original model_args including v11_2 settings
    model, model_args = load_model_from_checkpoint(model_path, return_args=True)
    model.to(device)
    model.eval()
    
    return model, model_args

# Inference for a single window (threshold-based)
def run_timeseries_inference(model, input_features, threshold=None, coin=None):
    if threshold is None:
        threshold = DEFAULT_CLASSIFICATION_THRESHOLD
        
    input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        logits = output["logits"]
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    # 🎯 Threshold-based prediction
    class_1_prob = probs[1]  # Probability of class 1 (up)
    predicted_class = 1 if class_1_prob > threshold else 0
    
    finalscore = float((probs[1] - probs[0]) * 100)
    prediction = {
        "coin": coin,
        "probabilities": probs.tolist(),
        "finalscore": finalscore,
        "pricetrend": predicted_class + 1,  # 1=down, 2=up (keep original format)
        "threshold": threshold
    }
    return prediction

# Run the entire pipeline
def run_timeseries_pipeline(merged_df, feature_cols=None, use_latest_only=True):
    if feature_cols is None:
        # Use the same feature columns as training
        feature_cols = FEATURE_COLS

    model, model_args = load_trained_model()
    scaler = load_scaler()
    window_size = model_args.get("window_size", 64)  # v11_2 compatible
    threshold = model_args.get("classification_threshold", DEFAULT_CLASSIFICATION_THRESHOLD)  # 🎯 Load threshold from model
    cached_predictions = load_recent_predictions()
    
    print(f"🎯 Using Classification Threshold: {threshold:.3f}")

    # 🎯 Collect data for batch price update
    predicted_coins = []
    prediction_ids = []
    
    coins = merged_df["coin"].unique()
    for original_coin in coins:
        # Normalize coin name
        normalized_coin = normalize_coin_name(original_coin)
        coin_df = merged_df[merged_df["coin"] == original_coin].copy()
        print(f"\n[🔍 Coin] {original_coin} -> {normalized_coin} - {len(coin_df)} rows")

        if len(coin_df) < window_size:
            continue

        try:
            coin_df[feature_cols] = scaler.transform(coin_df[feature_cols])
        except Exception as e:
            print(f"[❌ Scaling Error] {original_coin}: {e}")
            continue

        windows = generate_sliding_windows(coin_df, window_size, feature_cols)
        if not windows:
            continue

        if use_latest_only:
            latest_window = windows[-1]
            try:
                result = run_timeseries_inference(model, latest_window, threshold=threshold, coin=normalized_coin)
                if result.get("finalscore") is not None:
                    final_result = {
                        "coin": normalized_coin,  # Use normalized coin name
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "finalscore": round(result["finalscore"], 4),
                        "pricetrend": "up" if result["pricetrend"] == 2 else "down"  # 🎯 Use threshold-based prediction
                    }
                    prediction_id = save_prediction(final_result, cached=cached_predictions, table_name="predictions")
                    
                    # 🎯 Collect info for batch update
                    if prediction_id:
                        predicted_coins.append(normalized_coin)
                        prediction_ids.append(prediction_id)
            except Exception as e:
                print(f"[❌ Inference Error] {original_coin}: {e}")
        else:
            window_results = []
            for idx, window in enumerate(windows):
                try:
                    result = run_timeseries_inference(model, window, threshold=threshold, coin=normalized_coin)
                    if result.get("finalscore") is not None:
                        window_results.append(result)
                except Exception as e:
                    print(f"[❌ Inference Error] {original_coin} - Window {idx+1}: {e}")

            if not window_results:
                continue

            # Aggregate results from multiple windows (threshold-based)
            window_scores = [r["finalscore"] for r in window_results]
            window_predictions = [r["pricetrend"] for r in window_results]
            
            aggregated = aggregate_coin_scores(window_scores)
            # Use majority vote of threshold-based predictions
            up_votes = sum(1 for p in window_predictions if p == 2)
            down_votes = sum(1 for p in window_predictions if p == 1)
            majority_prediction = "up" if up_votes > down_votes else "down"
            
            final_result = {
                "coin": normalized_coin,  # Use normalized coin name
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "finalscore": aggregated["finalscore"],
                "pricetrend": majority_prediction  # 🎯 Use majority of threshold-based predictions
            }

            try:
                prediction_id = save_prediction(final_result, cached=cached_predictions, table_name="predictions")
                
                # 🎯 Collect info for batch update
                if prediction_id:
                    predicted_coins.append(normalized_coin)
                    prediction_ids.append(prediction_id)
            except Exception as e:
                print(f"[💥 Save Failed] {original_coin}: {e}")

    print("=== Predictions Saved, Starting Price Updates ===")
    
    # 🎯 Batch price update
    if predicted_coins and prediction_ids:
        try:
            print(f"💰 Coins to update prices for: {len(predicted_coins)} coins")
            
            # Query prices from CoinGecko API
            price_data = fetch_batch_prices(predicted_coins)
            
            if price_data:
                # Update price info in Supabase
                update_result = update_prediction_prices(prediction_ids, predicted_coins, price_data)
                print(f"✅ Price update result: {update_result}")
            else:
                print("⚠️ Failed to fetch price data - predictions are saved normally")
                
        except Exception as e:
            print(f"❌ Error during price update (predictions are saved normally): {e}")
    else:
        print("ℹ️ No predictions to update.")

    print("=== Timeseries Pipeline Finished ===")
