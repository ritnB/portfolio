# inference/timeseries_inference.py (Public-safe version)

import torch
import numpy as np
import pandas as pd
from datetime import datetime

from config import TIMESERIES_MODEL_PATH, FIXED_PARAMS
from data.preprocess import generate_sliding_windows, aggregate_coin_scores
from data.supabase_io import save_prediction
from models.timeseries_model import load_model

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_timeseries_inference(model, input_features, coin=None):
    """
    Run inference on a single sliding window of features.
    """
    input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        logits = output["logits"]
        probs = torch.softmax(logits, dim=1).squeeze()
        probs = probs.cpu().numpy()

    finalscore = float((probs[1] - probs[0]) * 100)

    return {
        "coin": coin,
        "probabilities": probs.tolist(),
        "finalscore": finalscore,
        "pricetrend": int(np.argmax(probs) + 1)
    }

def run_timeseries_pipeline(merged_df, hyperparams=None, feature_cols=None):
    """
    Run the full time-series prediction pipeline and store results.
    """
    if feature_cols is None:
        feature_cols = [
            "sma", "ema", "macd", "macd_signal", "macd_diff",
            "rsi", "stochastic", "mfi", "cci", "sentiment_score"
        ]
    window_size = int(1e8)

    model = load_model(
        input_size=int(1e8),
        d_model=int(1e8),
        num_layers=int(1e8),
        num_heads=int(1e8),
        window_size=window_size,
        num_classes=int(1e8),
        patch_size=int(1e8)
    )

    checkpoint = torch.load(TIMESERIES_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    for coin in merged_df["coin"].unique():
        coin_df = merged_df[merged_df["coin"] == coin].copy()
        if len(coin_df) < window_size:
            continue

        windows = generate_sliding_windows(coin_df, window_size, feature_cols)
        if not windows:
            continue

        window_scores = []
        for window in windows:
            result = run_timeseries_inference(model, window, coin=coin)
            if result.get("finalscore") is not None:
                window_scores.append(result["finalscore"])

        if not window_scores:
            continue

        aggregated = aggregate_coin_scores(window_scores)
        final_result = {
            "coin": coin,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
            **aggregated
        }
        save_prediction(final_result, table_name="your_predictions_table")

    print("=== Timeseries Pipeline Finished ===")
