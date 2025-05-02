# data/preprocess.py (Public-safe version)

import pandas as pd
import numpy as np

# Mapping dictionary to unify coin name variants
MAPPING_DICT = {
    "bitcoin": "Bitcoin",
    "wrapped-bitcoin": "Bitcoin",
    "ethereum": "Ethereum",
    # ... (mapping trimmed for public release)
}

def update_coin(coin: str) -> str:
    """
    Normalize the coin name using a predefined mapping dictionary.
    """
    if pd.isna(coin):
        return coin
    coin_clean = str(coin).strip().lower()
    return MAPPING_DICT.get(coin_clean, coin)

def apply_coin_mapping(df: pd.DataFrame, coin_column: str = "coin") -> pd.DataFrame:
    """
    Apply coin normalization to a DataFrame column.
    """
    df[coin_column] = df[coin_column].apply(update_coin)
    return df

def drop_non_common_coins(ti_df: pd.DataFrame, comments_df: pd.DataFrame, coin_column: str = "coin") -> pd.DataFrame:
    """
    Filter out rows in ti_df that don't have matching coins in comments_df.
    """
    comment_coin_set = set(comments_df[coin_column].dropna().unique())
    return ti_df[ti_df[coin_column].isin(comment_coin_set)].copy()

def fill_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop non-essential columns and replace NaNs with 0.
    """
    if "price_trend" in df.columns:
        df = df.drop(columns=["price_trend"])
    return df.fillna(0)

def generate_sliding_windows(df: pd.DataFrame, window_size: int, feature_cols: list) -> list:
    """
    Generate sliding windows of time-series features.
    Each window is a (window_size, feature_dim) NumPy array.
    """
    df_sorted = df.sort_values("timestamp")
    data = df_sorted[feature_cols].to_numpy()
    return [data[i:i+window_size] for i in range(len(data) - window_size + 1)]

def aggregate_coin_scores(scores: list) -> dict:
    """
    Aggregate multiple window-based prediction scores into a final decision.
    """
    if not scores:
        return {"finalscore": None, "pricetrend": "no_data"}
    avg_score = np.mean(scores)
    pricetrend = "up" if avg_score >= 0 else "down"
    return {"finalscore": avg_score, "pricetrend": pricetrend}
