import numpy as np
import pandas as pd
import torch
import time
from datetime import timedelta
from torch.utils.data import Dataset
from typing import List, Dict, Any
from utils.memory_utils import safe_memory_cleanup, monitor_memory_usage, check_memory_limit

# ======================== Sequence Caching System ========================

# Global caches
_cached_sequences: Dict[str, List] = {}
_cached_train_data = None
_cached_test_data = None

def get_sequences(data, window_size: int, force_regenerate: bool = False, cache_key: str = "default"):
    """Cache and return sequences per window_size (v11_2 compatible)."""
    global _cached_sequences
    
    full_cache_key = f"{cache_key}_{window_size}"
    
    # Generate if forced or missing in cache
    if force_regenerate or full_cache_key not in _cached_sequences:
        start_time = time.time()
        print(f"🔄 {cache_key} window_size={window_size} generating sequences...")
        sequences = batch_sequence_processing(data, window_size)
        _cached_sequences[full_cache_key] = sequences
        elapsed = time.time() - start_time
        print(f"✅ {cache_key} window_size={window_size} cached sequences ({len(sequences):,} items, {elapsed:.1f}s)")
    else:
        print(f"♻️ {cache_key} window_size={window_size} using cached sequences ({len(_cached_sequences[full_cache_key]):,} items)")
    
    return _cached_sequences[full_cache_key]

def clear_sequence_cache():
    """Clear sequence cache."""
    global _cached_sequences
    cache_count = len(_cached_sequences)
    _cached_sequences.clear()
    print(f"🧹 Cleared sequence cache ({cache_count} entries removed)")

def print_cache_status():
    """Print cache status."""
    print(f"📊 Sequence cache status:")
    for key, sequences in _cached_sequences.items():
        print(f"  - {key}: {len(sequences):,} sequences")
    print(f"  - total {len(_cached_sequences)} cache entries")

def safe_parse_timestamp(timestamp_series):
    """Parse timestamps robustly across formats (v11_2 compatible)."""
    try:
        # Already datetime dtype
        if pd.api.types.is_datetime64_any_dtype(timestamp_series):
            return timestamp_series
        # String dtype
        elif timestamp_series.dtype == 'object':
            try:
                return pd.to_datetime(timestamp_series, format='ISO')
            except:
                try:
                    return pd.to_datetime(timestamp_series, infer_datetime_format=True)
                except:
                    return pd.to_datetime(timestamp_series, format='mixed')
        else:
            # Numeric timestamp
            return pd.to_datetime(timestamp_series, unit='s')
    except Exception as e:
        print(f"⚠️ Timestamp parse error: {e}")
        return pd.to_datetime(timestamp_series, errors='coerce')

def split_by_time_gap(df, max_gap_hours=24):
    """Split sequences when time gap exceeds max_gap_hours (v11_2 compatible)."""
    split_data = []
    
    for coin, coin_df in df.groupby('coin'):
        coin_df_sorted = coin_df.sort_values('timestamp').reset_index(drop=True)
        
        if len(coin_df_sorted) <= 1:
            if len(coin_df_sorted) == 1:
                split_data.append(coin_df_sorted)
            continue
        
        sequences = []
        current_start = 0
        
        for i in range(1, len(coin_df_sorted)):
            time_diff = coin_df_sorted.iloc[i]['timestamp'] - coin_df_sorted.iloc[i-1]['timestamp']
            
            if time_diff > pd.Timedelta(hours=max_gap_hours):
                sequences.append(coin_df_sorted.iloc[current_start:i].copy())
                current_start = i
        
        sequences.append(coin_df_sorted.iloc[current_start:].copy())
        split_data.extend(sequences)
    
    return split_data

def batch_sequence_processing(split_data, window_size, batch_size=1000):
    """Generate sequences with memory-efficient batching (v11_2 compatible)."""
    all_sequences = []
    
    # Compute dataset-wide stats
    total_instances = sum(len(chunk) for chunk in split_data if len(chunk) >= window_size)
    potential_sequences = sum(len(chunk) - window_size + 1 for chunk in split_data if len(chunk) >= window_size)
    
    processed_chunks = 0
    total_chunks = len([chunk for chunk in split_data if len(chunk) >= window_size])
    
    print(f"🔄 Sequence generation started (window_size={window_size}):")
    print(f"  - total instances: {total_instances:,}")
    print(f"  - potential sequences: {potential_sequences:,}")
    print(f"  - chunks to process: {total_chunks}")
    
    for chunk in split_data:
        if len(chunk) >= window_size:
            processed_chunks += 1
            
            # Process per batch
            for i in range(0, len(chunk), batch_size):
                batch_end = min(i + batch_size, len(chunk))
                batch_data = chunk.iloc[i:batch_end]
                
                # Overlap handling (window_size-1)
                if i > 0:
                    overlap_start = max(0, i - (window_size - 1))
                    overlap_data = chunk.iloc[overlap_start:i]
                    batch_data = pd.concat([overlap_data, batch_data], ignore_index=True)
                
                # Create sequences via sliding window
                for j in range(len(batch_data) - window_size + 1):
                    sequence = batch_data.iloc[j:j+window_size]
                    all_sequences.append(sequence)
    
    # Final stats
    if all_sequences:
        print(f"✅ Sequence generation complete:")
        print(f"  - generated sequences: {len(all_sequences):,}")
        print(f"  - potential sequences: {potential_sequences:,}")
        print(f"  - data utilization: {len(all_sequences)/potential_sequences*100:.1f}%")
        print(f"  - processed chunks: {processed_chunks}/{total_chunks}")
    else:
        print("⚠️ Sequence generation failed: insufficient data")
    
    return all_sequences

def generate_sliding_windows(df, window_size, feature_cols, max_time_gap_hours=24):
    """
    Generate sliding windows per coin considering temporal continuity.
    Each window is a numpy array of shape (window_size, feature_dim).
    (kept for v11_2 compatibility)
    """
    # v11_2 style processing
    split_data = split_by_time_gap(df, max_gap_hours=max_time_gap_hours)
    sequences = batch_sequence_processing(split_data, window_size)
    
    # Convert to legacy numpy list
    all_windows = []
    for sequence in sequences:
        if len(sequence) >= window_size:
            X = sequence[feature_cols].values.astype(np.float32)
            if not np.isnan(X).any():
                all_windows.append(X)
    
    return all_windows

def aggregate_coin_scores(scores):
    """Aggregate scores across windows for a coin (legacy compatibility)."""
    avg_score = float(np.mean(scores))
    pricetrend = "up" if avg_score > 0 else "down"
    return {
        "finalscore": round(avg_score, 4),
        "avgscore": round(avg_score, 4),
        "pricetrend": pricetrend
    }

def rolling_window_cv_split(sequences, n_splits=3):
    """Overlapping rolling-window CV split (v11_2 compatible)."""
    if not sequences:
        return []
    
    import random
    from config import ROLLING_CV_TRAIN_RATIO, ROLLING_CV_VAL_RATIO, ROLLING_CV_STRIDE_RATIO
    
    # Sort by time and break ties randomly
    sequences.sort(key=lambda x: (x['timestamp'].min(), random.random()))
    
    total_sequences = len(sequences)
    
    # Compute fold sizes (with overlap)
    train_window_size = int(total_sequences * ROLLING_CV_TRAIN_RATIO)
    val_window_size = int(total_sequences * ROLLING_CV_VAL_RATIO)
    stride = int(total_sequences * ROLLING_CV_STRIDE_RATIO)
    
    cv_folds = []
    for i in range(n_splits):
        # Train window start
        train_start = i * stride
        train_end = min(train_start + train_window_size, total_sequences)
        
        # Validation window starts right after train window
        val_start = train_end
        val_end = min(val_start + val_window_size, total_sequences)
        
        # Append only valid splits
        if train_end > train_start and val_end > val_start:
            train_sequences = sequences[train_start:train_end]
            val_sequences = sequences[val_start:val_end]
            
            if train_sequences and val_sequences:
                cv_folds.append({
                    'train': train_sequences,
                    'validation': val_sequences
                })
    
    return cv_folds

class CoinTimeSeriesDataset(Dataset):
    """Time series dataset (sequence-based), v11_2 compatible."""
    def __init__(self, sequences, feature_cols):
        self.samples = []
        
        for sequence in sequences:
            if len(sequence) >= 1:  # require at least one instance
                X = sequence[feature_cols].values.astype(np.float32)
                y = sequence['label'].values.astype(np.int64)
                
                # Use the label of the last instance in the sequence
                self.samples.append((X, y[-1]))
        
        # print(f"📊 {len(self.samples)} sequence samples")  # avoid duplicate logs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        return {"x": torch.tensor(X), "labels": torch.tensor(y)}


class CompatibleCoinTimeSeriesDataset(Dataset):
    """Legacy dataset kept for backward compatibility with existing code."""
    def __init__(self, df, window_size, feature_cols, max_time_gap_hours=24):
        self.samples = []
        self.max_time_gap = pd.Timedelta(hours=max_time_gap_hours)
        
        # Create sequences (v11_2 style)
        split_data = split_by_time_gap(df, max_gap_hours=max_time_gap_hours)
        sequences = batch_sequence_processing(split_data, window_size)
        
        # Convert to legacy format
        for seq_idx, sequence in enumerate(sequences):
            if len(sequence) >= window_size:
                X = sequence[feature_cols].values.astype(np.float32)
                y = sequence['label'].values.astype(np.int64)
                
                # Create sliding windows
                for i in range(len(X) - window_size + 1):
                    window_X = X[i:i+window_size]
                    window_y = y[i+window_size-1]
                    coin_info = sequence['coin'].iloc[0] if 'coin' in sequence.columns else f"seq_{seq_idx}"
                    self.samples.append((window_X, window_y, coin_info))
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y, coin = self.samples[idx]
        return {
            "x": torch.tensor(X),
            "labels": torch.tensor(y),
            "coin": coin
        }
