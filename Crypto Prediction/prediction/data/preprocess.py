import numpy as np
import pandas as pd
import torch
import time
from datetime import timedelta
from torch.utils.data import Dataset
from typing import List, Dict, Any
from utils.memory_utils import safe_memory_cleanup, monitor_memory_usage, check_memory_limit

# ======================== Sequence Caching System ========================

# Global cache variables
_cached_sequences: Dict[str, List] = {}
_cached_train_data = None
_cached_test_data = None

def get_sequences(data, window_size: int, force_regenerate: bool = False, cache_key: str = "default"):
    """Return cached sequences by window_size (anonymized)"""
    global _cached_sequences
    
    full_cache_key = f"{cache_key}_{window_size}"
    
    # Force regeneration or if not in cache
    if force_regenerate or full_cache_key not in _cached_sequences:
        start_time = time.time()
        print(f"🔄 Generating sequences for {cache_key} window_size={window_size}...")
        sequences = batch_sequence_processing(data, window_size)
        _cached_sequences[full_cache_key] = sequences
        elapsed = time.time() - start_time
        print(f"✅ {cache_key} window_size={window_size} sequence cache completed ({len(sequences):,} sequences, {elapsed:.1f}s)")
    else:
        print(f"♻️ Using cached sequences for {cache_key} window_size={window_size} ({len(_cached_sequences[full_cache_key]):,} sequences)")
    
    return _cached_sequences[full_cache_key]

def clear_sequence_cache():
    """Clear sequence cache"""
    global _cached_sequences
    cache_count = len(_cached_sequences)
    _cached_sequences.clear()
    print(f"🧹 Sequence cache cleared ({cache_count} items removed)")

def print_cache_status():
    """Print cache status"""
    print(f"📊 Sequence cache status:")
    for key, sequences in _cached_sequences.items():
        print(f"  - {key}: {len(sequences):,} sequences")
    print(f"  - Total {len(_cached_sequences)} cache items")

def safe_parse_timestamp(timestamp_series):
    """Parse all timestamp formats (anonymized)"""
    try:
        # Already datetime type
        if pd.api.types.is_datetime64_any_dtype(timestamp_series):
            return timestamp_series
        # String type
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
        print(f"⚠️ Timestamp parsing error: {e}")
        return pd.to_datetime(timestamp_series, errors='coerce')

def split_by_time_gap(df, max_gap_hours=24):
    """Split at points with more than 24 hours difference (anonymized)"""
    split_data = []
    
    for asset, asset_df in df.groupby('asset'):
        asset_df_sorted = asset_df.sort_values('timestamp').reset_index(drop=True)
        
        if len(asset_df_sorted) <= 1:
            if len(asset_df_sorted) == 1:
                split_data.append(asset_df_sorted)
            continue
        
        sequences = []
        current_start = 0
        
        for i in range(1, len(asset_df_sorted)):
            time_diff = asset_df_sorted.iloc[i]['timestamp'] - asset_df_sorted.iloc[i-1]['timestamp']
            
            if time_diff > pd.Timedelta(hours=max_gap_hours):
                sequences.append(asset_df_sorted.iloc[current_start:i].copy())
                current_start = i
        
        sequences.append(asset_df_sorted.iloc[current_start:].copy())
        split_data.extend(sequences)
    
    return split_data

def batch_sequence_processing(split_data, window_size, batch_size=1000):
    """Generate sequences with memory-efficient batch processing (anonymized)"""
    all_sequences = []
    
    # Calculate total statistics
    total_instances = sum(len(chunk) for chunk in split_data if len(chunk) >= window_size)
    potential_sequences = sum(len(chunk) - window_size + 1 for chunk in split_data if len(chunk) >= window_size)
    
    processed_chunks = 0
    total_chunks = len([chunk for chunk in split_data if len(chunk) >= window_size])
    
    print(f"🔄 Starting sequence generation (window_size={window_size}):")
    print(f"  - Total instances: {total_instances:,}")
    print(f"  - Potential sequences: {potential_sequences:,}")
    print(f"  - Chunks to process: {total_chunks}")
    
    for chunk in split_data:
        if len(chunk) >= window_size:
            processed_chunks += 1
            
            # Process by batch
            for i in range(0, len(chunk), batch_size):
                batch_end = min(i + batch_size, len(chunk))
                batch_data = chunk.iloc[i:batch_end]
                
                # Handle overlap (window_size-1)
                if i > 0:
                    overlap_start = max(0, i - (window_size - 1))
                    overlap_data = chunk.iloc[overlap_start:i]
                    batch_data = pd.concat([overlap_data, batch_data], ignore_index=True)
                
                # Generate sequences using sliding window
                for j in range(len(batch_data) - window_size + 1):
                    sequence = batch_data.iloc[j:j+window_size]
                    all_sequences.append(sequence)
    
    # Final statistics output
    if all_sequences:
        print(f"✅ Sequence generation completed:")
        print(f"  - Generated sequences: {len(all_sequences):,}")
        print(f"  - Potential sequences: {potential_sequences:,}")
        print(f"  - Data utilization: {len(all_sequences)/potential_sequences*100:.1f}%")
        print(f"  - Processed chunks: {processed_chunks}/{total_chunks}")
    else:
        print("⚠️ Sequence generation failed: insufficient data")
    
    return all_sequences

def generate_sliding_windows(df, window_size, feature_cols, max_time_gap_hours=24):
    """
    Generates sliding windows from a DataFrame for each coin, considering time continuity.
    Each window is a numpy array of shape (window_size, feature_dim).
    (Compatibility with v11_2)
    """
    # Process using v11_2 method
    split_data = split_by_time_gap(df, max_gap_hours=max_time_gap_hours)
    sequences = batch_sequence_processing(split_data, window_size)
    
    # Convert to original format (list of numpy arrays)
    all_windows = []
    for sequence in sequences:
        if len(sequence) >= window_size:
            X = sequence[feature_cols].values.astype(np.float32)
            if not np.isnan(X).any():
                all_windows.append(X)
    
    return all_windows

def aggregate_coin_scores(scores):
    """Aggregate coin scores (compatibility with existing code)"""
    avg_score = float(np.mean(scores))
    pricetrend = "up" if avg_score > 0 else "down"
    return {
        "finalscore": round(avg_score, 4),
        "avgscore": round(avg_score, 4),
        "pricetrend": pricetrend
    }

def rolling_window_cv_split(sequences, n_splits=3):
    """Rolling Window CV split (compatibility with v11_2)"""
    if not sequences:
        return []
    
    import random
    # Sort by time and shuffle within the same timestamp
    sequences.sort(key=lambda x: (x['timestamp'].min(), random.random()))
    
    total_sequences = len(sequences)
    
    # Calculate size for each fold (use a larger window for overlap)
    train_window_size = int(total_sequences * 0.6)  # 60% for training
    val_window_size = int(total_sequences * 0.2)    # 20% for validation
    stride = int(total_sequences * 0.2)             # 20% stride between folds
    
    cv_folds = []
    for i in range(n_splits):
        # Train window start
        train_start = i * stride
        train_end = min(train_start + train_window_size, total_sequences)
        
        # Validation window starts right after train
        val_start = train_end
        val_end = min(val_start + val_window_size, total_sequences)
        
        # Add only if valid data exists
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
    """v11_2 compatible time series dataset (sequence-based)"""
    def __init__(self, sequences, feature_cols):
        self.samples = []
        
        for sequence in sequences:
            if len(sequence) >= 1:  # Minimum 1 instance
                X = sequence[feature_cols].values.astype(np.float32)
                y = sequence['label'].values.astype(np.int64)
                
                # Use the label of the last instance in the sequence
                self.samples.append((X, y[-1]))
        
        # print(f"📊 {len(self.samples)} sequence samples generated")  # Avoid duplicate output

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        return {"x": torch.tensor(X), "labels": torch.tensor(y)}


class CompatibleCoinTimeSeriesDataset(Dataset):
    """Legacy dataset for compatibility with existing code"""
    def __init__(self, df, window_size, feature_cols, max_time_gap_hours=24):
        self.samples = []
        self.max_time_gap = pd.Timedelta(hours=max_time_gap_hours)
        
        # Generate sequences using v11_2 method
        split_data = split_by_time_gap(df, max_gap_hours=max_time_gap_hours)
        sequences = batch_sequence_processing(split_data, window_size)
        
        # Convert to original format
        for seq_idx, sequence in enumerate(sequences):
            if len(sequence) >= window_size:
                X = sequence[feature_cols].values.astype(np.float32)
                y = sequence['label'].values.astype(np.int64)
                
                # Generate sliding windows
                for i in range(len(X) - window_size + 1):
                    window_X = X[i:i+window_size]
                    window_y = y[i+window_size-1]
                    coin_info = sequence['asset'].iloc[0] if 'asset' in sequence.columns else f"seq_{seq_idx}"
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
