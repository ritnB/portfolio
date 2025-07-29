import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def generate_sliding_windows(df, window_size, feature_cols):
    """
    Generate sliding windows from asset-specific DataFrame.
    Each window is a numpy array of shape (window_size, feature_dim).
    """
    df_sorted = df.sort_values("timestamp")
    X = df_sorted[feature_cols].values.astype(np.float32)
    if len(X) < window_size:
        return []

    windows = []
    for i in range(len(X) - window_size + 1):
        window = X[i : i + window_size]
        if np.isnan(window).any():
            continue  # Remove windows containing NaN
        windows.append(window)
    return windows

def aggregate_asset_scores(scores):
    avg_score = float(np.mean(scores))
    pricetrend = "up" if avg_score > 0 else "down"
    return {
        "finalscore": round(avg_score, 4),
        "avgscore": round(avg_score, 4),
        "pricetrend": pricetrend
    }

class AssetTimeSeriesDataset(Dataset):
    def __init__(self, df, window_size, feature_cols):
        self.samples = []
        for asset, grp in df.groupby('asset'):
            grp_sorted = grp.sort_values('timestamp')
            X = grp_sorted[feature_cols].values.astype(np.float32)
            y = grp_sorted['label'].values.astype(np.int64)
            for i in range(len(X) - window_size + 1):
                window_X = X[i:i+window_size]
                window_y = y[i+window_size-1]
                self.samples.append((window_X, window_y, asset))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y, asset = self.samples[idx]
        return {
            "x": torch.tensor(X),
            "labels": torch.tensor(y),
            "asset": asset
        }
