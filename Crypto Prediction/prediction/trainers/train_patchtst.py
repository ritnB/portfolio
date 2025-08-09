import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from datetime import datetime, timedelta
from collections import Counter
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from transformers import Trainer, TrainingArguments, TrainerCallback

import optuna
from optuna.pruners import MedianPruner

from data.supabase_io import load_technical_indicators
from utils.memory_utils import safe_memory_cleanup, monitor_memory_usage, check_memory_limit
from config import TEST_DAYS, DEFAULT_CLASSIFICATION_THRESHOLD, FEATURE_COLS

# v11_4 constants
N_TRIALS = 30
MEMORY_THRESHOLD_MB = 8000
PREDICTION_HORIZON_HOURS = 4
VAL_TO_TRAIN_REMOVAL_RATIO = 3
FOLD_SIZE_IMBALANCE_THRESHOLD = 1.5
CLASSIFICATION_THRESHOLD = DEFAULT_CLASSIFICATION_THRESHOLD

# Global caches
_cached_train_data = None
_cached_test_data = None
_cached_sequences = {}
_cached_scalers = {}
_cached_processed_data = {}

# Seed everything
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ======================== v11_4 classes ========================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class PatchTST(nn.Module):
    def __init__(self, input_size, d_model, num_layers, num_heads,
                 patch_size, window_size, num_classes,
                 dropout=0.0, pooling_type='cls', mlp_hidden_mult=2,
                 activation='relu', stride=None):

        super().__init__()

        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.pooling_type = pooling_type.lower()

        assert window_size >= patch_size, "window_size must be >= patch_size"
        assert (window_size - patch_size) % self.stride == 0, "patches must align evenly"

        self.num_patches = 1 + (window_size - patch_size) // self.stride

        # Patch embedding
        self.input_proj = nn.Linear(input_size * patch_size, d_model)

        # CLS token
        if self.pooling_type == 'cls':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Position embedding
        self.pos_embedding = nn.Parameter(torch.randn(
            1, self.num_patches + (1 if self.pooling_type == 'cls' else 0), d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Activation
        self.act = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU()
        }[activation]

        # MLP Head
        hidden_dim = d_model * mlp_hidden_mult
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_loss = None

    def forward(self, x, labels=None):
        B, W, D = x.shape
        P = self.patch_size
        S = self.stride

        # ① Create sliding patches
        x = x.unfold(dimension=1, size=P, step=S)
        x = x.contiguous().view(B, -1, P * D)

        # ② Linear projection
        x = self.input_proj(x)

        # ③ Insert CLS token
        if self.pooling_type == 'cls':
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_token, x], dim=1)

        # ④ Add positional embedding
        x = x + self.pos_embedding[:, :x.size(1), :]

        # ⑤ Transformer encoder
        x = self.transformer(x)

        # ⑥ Pooling
        if self.pooling_type == 'cls':
            x = x[:, 0, :]
        else:
            x = x.mean(dim=1)

        # ⑦ Classification
        logits = self.mlp(x)

        # ⑧ Loss
        if labels is not None:
            if self.focal_loss is not None:
                loss = self.focal_loss(logits, labels)
            else:
                loss = self.ce_loss(logits, labels)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}

class CoinTimeSeriesDataset(Dataset):
    def __init__(self, sequences, feature_cols):
        self.samples = []
        
        for sequence in sequences:
            if len(sequence) >= 1:
                X = sequence[feature_cols].values.astype(np.float32)
                y = sequence['label'].values.astype(np.int64)
                self.samples.append((X, y[-1]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        return {"x": torch.tensor(X), "labels": torch.tensor(y)}

class EnhancedEarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=5, max_loss_diff=0.05, min_f1_improvement=0.0005):
        self.patience = patience
        self.max_loss_diff = max_loss_diff
        self.min_f1_improvement = min_f1_improvement
        self.wait = 0
        self.best_f1 = None
        self.last_train_loss = None
        self.epochs_without_improvement = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            self.last_train_loss = logs["loss"]

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        eval_f1 = metrics.get("eval_weighted_f1", None)
        eval_loss = metrics.get("eval_loss", None)

        if eval_f1 is None or eval_loss is None or self.last_train_loss is None:
            return control

        # Overfitting detection
        loss_diff = eval_loss - self.last_train_loss
        if loss_diff > self.max_loss_diff:
            print(f"⏹️ Enhanced Early stopping: severe overfitting detected")
            control.should_training_stop = True
            return control

        # F1 improvement check
        if self.best_f1 is None:
            self.best_f1 = eval_f1
            self.wait = 0
            self.epochs_without_improvement = 0
        else:
            improvement = eval_f1 - self.best_f1
            
            if improvement > self.min_f1_improvement:
                self.best_f1 = eval_f1
                self.wait = 0
                self.epochs_without_improvement = 0
            else:
                self.wait += 1
                self.epochs_without_improvement += 1
                
                if self.wait >= self.patience:
                    print(f"⏹️ Enhanced Early stopping: no F1 improvement")
                    control.should_training_stop = True
                    return control

        return control

# ======================== v11_4 data processing helpers ========================

def safe_parse_timestamp(timestamp_series):
    """Parse timestamps robustly across formats."""
    try:
        if pd.api.types.is_datetime64_any_dtype(timestamp_series):
            return timestamp_series
        elif timestamp_series.dtype == 'object':
            try:
                return pd.to_datetime(timestamp_series, format='ISO')
            except:
                try:
                    return pd.to_datetime(timestamp_series, infer_datetime_format=True)
                except:
                    return pd.to_datetime(timestamp_series, format='mixed')
        else:
            return pd.to_datetime(timestamp_series, unit='s')
    except Exception as e:
        print(f"⚠️ Timestamp parse error: {e}")
        return pd.to_datetime(timestamp_series, errors='coerce')

def split_by_time_gap(df, max_gap_hours=24):
    """Split sequences when time gap exceeds max_gap_hours."""
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
    """Generate sequences with memory-efficient batching."""
    all_sequences = []
    
    total_instances = sum(len(chunk) for chunk in split_data if len(chunk) >= window_size)
    potential_sequences = sum(len(chunk) - window_size + 1 for chunk in split_data if len(chunk) >= window_size)
    
    print(f"🔄 Sequence generation started (window_size={window_size}):")
    print(f"  - total instances: {total_instances:,}")
    print(f"  - potential sequences: {potential_sequences:,}")
    
    for chunk in split_data:
        if len(chunk) >= window_size:
            for i in range(0, len(chunk), batch_size):
                batch_end = min(i + batch_size, len(chunk))
                batch_data = chunk.iloc[i:batch_end]
                
                if i > 0:
                    overlap_start = max(0, i - (window_size - 1))
                    overlap_data = chunk.iloc[overlap_start:i]
                    batch_data = pd.concat([overlap_data, batch_data], ignore_index=True)
                
                for j in range(len(batch_data) - window_size + 1):
                    sequence = batch_data.iloc[j:j+window_size]
                    all_sequences.append(sequence)
    
    if all_sequences:
        print(f"✅ Sequences generated: {len(all_sequences):,}")
    else:
        print("⚠️ Sequence generation failed: insufficient data")
    
    return all_sequences

def rolling_window_cv_split(sequences, n_splits=3):
    """Rolling Window CV with leakage prevention."""
    if not sequences:
        return []
    
    sequences.sort(key=lambda x: (x['timestamp'].min(), random.random()))
    
    total_sequences = len(sequences)
    
    train_window_size = int(total_sequences * 0.55)
    val_window_size = int(total_sequences * 0.15)
    stride = int(total_sequences * 0.15)
    
    cv_folds = []
    for i in range(n_splits):
        train_start = i * stride
        train_end = min(train_start + train_window_size, total_sequences)
        
        val_start = train_end
        val_end = min(val_start + val_window_size, total_sequences)
        
        if train_end > train_start and val_end > val_start:
            train_sequences = sequences[train_start:train_end]
            val_sequences = sequences[val_start:val_end]
            
            if train_sequences and val_sequences:
                cleaned_train, cleaned_val = remove_temporal_overlap(
                    train_sequences, val_sequences, 
                    gap_hours=PREDICTION_HORIZON_HOURS
                )
                
                if cleaned_train and cleaned_val:
                    cv_folds.append({
                        'train': cleaned_train,
                        'validation': cleaned_val
                    })
    
    return cv_folds

def remove_temporal_overlap(train_sequences, val_sequences, gap_hours=4):
    """Dynamically remove train/val sequences near boundaries to avoid overlap."""
    from datetime import timedelta
    
    val_min_time = min(seq['timestamp'].min() for seq in val_sequences)
    gap_threshold = val_min_time - timedelta(hours=gap_hours)
    
    overlapping_train = []
    overlapping_val = []
    
    for seq in train_sequences:
        if seq['timestamp'].max() > gap_threshold:
            overlapping_train.append(seq)
    
    train_max_time = max(seq['timestamp'].max() for seq in train_sequences)
    for seq in val_sequences:
        if seq['timestamp'].min() < train_max_time + timedelta(hours=gap_hours):
            overlapping_val.append(seq)
    
    val_to_remove = len(overlapping_val)
    train_to_remove = min(val_to_remove * VAL_TO_TRAIN_REMOVAL_RATIO, len(overlapping_train))
    
    overlapping_train.sort(key=lambda x: x['timestamp'].max(), reverse=True)
    overlapping_val.sort(key=lambda x: x['timestamp'].min())
    
    train_to_remove_set = set(id(seq) for seq in overlapping_train[:train_to_remove])
    val_to_remove_set = set(id(seq) for seq in overlapping_val[:val_to_remove])
    
    cleaned_train = [seq for seq in train_sequences if id(seq) not in train_to_remove_set]
    cleaned_val = [seq for seq in val_sequences if id(seq) not in val_to_remove_set]
    
    return cleaned_train, cleaned_val

# ======================== v11_4 caching ========================

def get_processed_data(window_size, force_regenerate=False):
    """Cache and return processed data for a given window_size."""
    global _cached_processed_data
    
    if force_regenerate or window_size not in _cached_processed_data:
        print(f"🔄 Generating processed data for window_size={window_size}...")
        
        sequences = batch_sequence_processing(_cached_train_data, window_size)
        if not sequences:
            raise optuna.exceptions.TrialPruned()
        
        cv_folds = rolling_window_cv_split(sequences)
        if not cv_folds:
            raise optuna.exceptions.TrialPruned()
        
        if len(cv_folds) != 3:
            print(f"⚠️ Insufficient folds {len(cv_folds)}/3 - skipping window_size={window_size}")
            raise optuna.exceptions.TrialPruned()
        
        fold_sizes = [len(fold['train']) + len(fold['validation']) for fold in cv_folds]
        if fold_sizes and max(fold_sizes) / min(fold_sizes) > FOLD_SIZE_IMBALANCE_THRESHOLD:
            print(f"⚠️ Fold size imbalance - skipping window_size={window_size}")
            raise optuna.exceptions.TrialPruned()
        
        scalers = {}
        for fold_idx, fold_data in enumerate(cv_folds):
            scaler = StandardScaler()
            train_sequences = fold_data['train']
            
            for seq in train_sequences:
                scaler.partial_fit(seq[FEATURE_COLS])
            
            scalers[fold_idx] = scaler
        
        _cached_processed_data[window_size] = {
            'cv_folds': cv_folds,
            'scalers': scalers
        }
        
        print(f"✅ window_size={window_size} processed cache ready ({len(cv_folds)} folds)")
    else:
        print(f"♻️ Using cached processed data for window_size={window_size}")
    
    return _cached_processed_data[window_size]

def clear_cache():
    """Clear all caches."""
    global _cached_sequences, _cached_scalers, _cached_processed_data
    _cached_sequences.clear()
    _cached_scalers.clear()
    _cached_processed_data.clear()
    print("🧹 Caches cleared")

# ======================== v11_4 Optuna objective ========================

def objective(trial):
    """Optuna objective for v11_4 training."""
    print(f"\n🔄 Trial {trial.number} starting...")
    
    if check_memory_limit():
        print(f"⏹️ Trial {trial.number} pruned (memory limit)")
        raise optuna.exceptions.TrialPruned()
    
    # v11_4 hyperparameters
    patch_len = trial.suggest_categorical("patch_len", [4, 8, 16, 32, 48, 64])
    window_size = trial.suggest_categorical("window_size", [16, 32, 64, 96, 128])
    stride = trial.suggest_categorical("stride", [2, 4, 8, 16, 32, 48, 64])
    
    # Pruning guard
    if (
        window_size <= patch_len
        or stride > patch_len
        or (window_size - patch_len) % stride != 0
        or (1 + (window_size - patch_len) // stride) < 3
        or stride > window_size // 4
        or patch_len > window_size // 2
    ):
        print(f"⏹️ Trial {trial.number} pruned (parameter constraints)")
        raise optuna.exceptions.TrialPruned()
    
    # Remaining hyperparameters
    d_model = trial.suggest_categorical("d_model", [64, 96, 128, 256])
    mlp_mult = trial.suggest_categorical("mlp_hidden_mult", [1, 2, 3])
    dropout = trial.suggest_float("dropout_mlp", 0.0, 0.3)
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.05)
    activation = trial.suggest_categorical("activation", ["relu", "gelu"])
    pooling_type = trial.suggest_categorical("pooling_type", ["cls", "mean"])
    loss_type = trial.suggest_categorical("loss_type", ["ce", "focal"])
    focal_gamma = trial.suggest_float("focal_gamma", 1.5, 2.5) if loss_type == "focal" else None
    
    # Constants
    threshold = CLASSIFICATION_THRESHOLD
    num_layers = 2
    num_heads = 4
    batch_size = 128
    feature_cols = FEATURE_COLS
    
    # 완성본 데이터 생성
    processed_data = get_processed_data(window_size)
    cv_folds = processed_data['cv_folds']
    fold_scalers = processed_data['scalers']
    
    cv_scores = []
    
    for fold_idx, fold_data in enumerate(reversed(cv_folds)):
        print(f"  Fold {fold_idx+1}/{len(cv_folds)}", end="")
        train_sequences = fold_data['train']
        val_sequences = fold_data['validation']
        
        fold_scaler = fold_scalers[fold_idx]
        
        # Apply scaling
        train_sequences_scaled = []
        for seq in train_sequences:
            seq_scaled = seq.copy()
            seq_scaled[feature_cols] = fold_scaler.transform(seq[feature_cols])
            train_sequences_scaled.append(seq_scaled)
        
        val_sequences_scaled = []
        for seq in val_sequences:
            seq_scaled = seq.copy()
            seq_scaled[feature_cols] = fold_scaler.transform(seq[feature_cols])
            val_sequences_scaled.append(seq_scaled)
        
        # Create datasets
        train_dataset = CoinTimeSeriesDataset(train_sequences_scaled, feature_cols)
        val_dataset = CoinTimeSeriesDataset(val_sequences_scaled, feature_cols)
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            continue

        model = PatchTST(
            input_size=len(feature_cols),
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            patch_size=patch_len,
            window_size=window_size,
            stride=stride,
            num_classes=2,
            dropout=dropout,
            pooling_type=pooling_type,
            mlp_hidden_mult=mlp_mult,
            activation=activation
        )

        training_args = TrainingArguments(
            output_dir=f"./tmp_trial_{trial.number}_fold_{fold_idx}",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=32,
            eval_strategy="epoch",
            save_strategy="no",
            num_train_epochs=50,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            logging_steps=999999,
            disable_tqdm=True,
            report_to=[],
            logging_dir=None,
            log_level="error",
            log_level_replica="error",
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            logging_first_step=False,
            logging_strategy="no",
            prediction_loss_only=False,
            load_best_model_at_end=False,
            eval_delay=0,
            dataloader_num_workers=2,
            no_cuda=False
        )

        def compute_metrics_with_threshold(p):
            try:
                probs = F.softmax(torch.tensor(p.predictions), dim=1)
                class_1_probs = probs[:, 1].numpy()
                preds = (class_1_probs > threshold).astype(int)
                
                metrics = {
                    "eval_weighted_f1": f1_score(p.label_ids, preds, average="weighted", zero_division=0),
                    "eval_weighted_precision": precision_score(p.label_ids, preds, average="weighted", zero_division=0),
                    "eval_weighted_recall": recall_score(p.label_ids, preds, average="weighted", zero_division=0)
                }
                
                return metrics
            except Exception as e:
                return {
                    "eval_weighted_f1": 0.0,
                    "eval_weighted_precision": 0.0,
                    "eval_weighted_recall": 0.0
                }

        callbacks = [EnhancedEarlyStoppingCallback(patience=5, max_loss_diff=0.05, min_f1_improvement=0.0005)]
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics_with_threshold,
            callbacks=callbacks
        )

        model.focal_loss = FocalLoss(gamma=focal_gamma) if loss_type == "focal" else None
        
        try:
            import sys
            from io import StringIO
            
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                trainer.train()
                eval_metrics = trainer.evaluate()
            finally:
                sys.stdout = old_stdout
            
            if 'eval_weighted_f1' in eval_metrics:
                cv_scores.append(eval_metrics)
                print(f" ✓", end="")
                
                if len(cv_scores) >= 2:
                    current_mean_f1 = np.mean([score["eval_weighted_f1"] for score in cv_scores])
                    trial.report(current_mean_f1, step=fold_idx)
                    
                    if trial.should_prune():
                        print(f" ⏹️ Trial {trial.number} pruned")
                        raise optuna.exceptions.TrialPruned()
            else:
                print(f" ✗", end="")
                continue
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            print(f" ✗", end="")
            continue
        finally:
            try:
                del model, trainer, train_dataset, val_dataset
                safe_memory_cleanup()
            except Exception:
                pass
    
    if not cv_scores:
        raise optuna.exceptions.TrialPruned()
    
    mean_f1 = np.mean([score["eval_weighted_f1"] for score in cv_scores])
    mean_precision = np.mean([score["eval_weighted_precision"] for score in cv_scores])
    mean_recall = np.mean([score["eval_weighted_recall"] for score in cv_scores])
    
        print(f" → F1={mean_f1:.3f} | P={mean_precision:.3f} | R={mean_recall:.3f}")
    
    return mean_f1

# ======================== 메인 훈련 함수 ========================

def train_patchtst_model():
    """Full training pipeline with Optuna (v11_4)."""
    global _cached_train_data, _cached_test_data
    
    print("🚀 v11_4 방식 모델 훈련 시작...")
    
    # 데이터 로딩 및 전처리
    print("🔄 데이터 로딩 및 전처리 시작...")
    
    df = load_technical_indicators(for_training=True)
    
    # timestamp 처리
    df['timestamp'] = safe_parse_timestamp(df['timestamp'])
    
    # 데이터 정제
    df = df.dropna()
    df = df[df['price_trend'].isin(['up', 'down'])].copy()
    df['label'] = (df['price_trend'] == 'up').astype(int)
    
    # 훈련/테스트 데이터 분할
    max_ts = df['timestamp'].max()
    test_cutoff = max_ts - timedelta(days=TEST_DAYS)
    
    train_df = df[df['timestamp'] < test_cutoff].copy()
    test_df = df[df['timestamp'] >= test_cutoff].copy()
    
    print(f"📊 데이터 분할 통계:")
    print(f"  - 훈련: {len(train_df):,}개")
    print(f"  - 테스트: {len(test_df):,}개")
    
    # 24시간 분할 전처리 (캐싱용)
    train_split_data = split_by_time_gap(train_df, max_gap_hours=24)
    test_split_data = split_by_time_gap(test_df, max_gap_hours=24)
    
    _cached_train_data = train_split_data
    _cached_test_data = test_split_data
    
    print(f"✅ 전처리 완료!")
    
    # Optuna Study 생성
    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # 최적화 실행
    print(f"\n🚀 Optuna 최적화 시작 ({N_TRIALS} trials)...")
    
    with tqdm(total=N_TRIALS, desc="Optuna Trials", unit="trial") as pbar:
        def objective_with_progress(trial):
            result = objective(trial)
            pbar.update(1)
            
            try:
                best_f1 = f"{study.best_value:.3f}" if len([t for t in study.trials if t.state.is_finished()]) > 0 else "N/A"
            except:
                best_f1 = "N/A"
                
            pbar.set_postfix({
                'Best F1': best_f1,
                'Current': f"{result:.3f}" if result is not None else "Failed"
            })
            return result
        
        study.optimize(objective_with_progress, n_trials=N_TRIALS)
    
    # 결과 출력
    print(f"\n🎯 Optimization complete!")
    print(f"  - best F1: {study.best_value:.3f}")
    print(f"  - best params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    
    # 최적 파라미터로 최종 모델 훈련
    print(f"\n🔧 Training final model with best params...")
    best_params = study.best_params
    
    # 전체 훈련 데이터로 시퀀스 생성
    final_sequences = batch_sequence_processing(_cached_train_data, best_params['window_size'])
    
    if not final_sequences:
        print("❌ Final training failed: could not generate sequences")
        return None
    
    # 전체 훈련 데이터로 스케일러 생성
    print("🔄 Building final scaler...")
    final_scaler = StandardScaler()
    
    for seq in final_sequences:
        final_scaler.partial_fit(seq[FEATURE_COLS])
    
    # 모든 시퀀스에 스케일링 적용
    final_sequences_scaled = []
    for seq in final_sequences:
        seq_scaled = seq.copy()
        seq_scaled[FEATURE_COLS] = final_scaler.transform(seq[FEATURE_COLS])
        final_sequences_scaled.append(seq_scaled)
    print("✅ Scaling complete for final model")
    
    # 최종 모델 생성 및 훈련
    final_model = PatchSequenceModel(
        input_size=len(FEATURE_COLS),
        d_model=best_params['d_model'],
        num_layers=2,
        num_heads=4,
        patch_size=best_params['patch_len'],
        window_size=best_params['window_size'],
        stride=best_params['stride'],
        num_classes=2,
        dropout=best_params['dropout_mlp'],
        pooling_type=best_params['pooling_type'],
        mlp_hidden_mult=best_params['mlp_hidden_mult'],
        activation=best_params['activation']
    )
    
    # Focal Loss 설정
    if best_params['loss_type'] == 'focal':
        final_model.focal_loss = FocalLoss(gamma=best_params['focal_gamma'])
    
    # 최종 데이터셋 생성
    final_dataset = CoinTimeSeriesDataset(final_sequences_scaled, FEATURE_COLS)
    
    # 최종 훈련 설정
    final_training_args = TrainingArguments(
        output_dir="./final_model_training",
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        eval_strategy="no",
        save_strategy="no",
        num_train_epochs=100,
        learning_rate=best_params['learning_rate'],
        weight_decay=best_params['weight_decay'],
        logging_steps=999999,
        disable_tqdm=True,
        report_to=[],
        log_level="error",
        log_level_replica="error",
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        logging_first_step=False,
        logging_strategy="no",
        dataloader_num_workers=2,
        no_cuda=False
    )
    
    # Final training
    final_trainer = Trainer(
        model=final_model,
        args=final_training_args,
        train_dataset=final_dataset,
        callbacks=[EnhancedEarlyStoppingCallback()]
    )
    
    print("🔄 Training final model...")
    final_trainer.train()
    print("✅ Final model training complete!")
    
    # 파일 저장
    today_str = datetime.utcnow().strftime("%Y%m%d")
    save_dir = f"/tmp/model_{today_str}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save scaler
    scaler_path = os.path.join(save_dir, f"scaler_standard_{today_str}.pkl")
    joblib.dump(final_scaler, scaler_path)
    print(f"✅ Scaler saved: {scaler_path}")
    
    # Save model
    model_path = os.path.join(save_dir, f"patchseq_final_model_{today_str}.pt")
    
    model_args = {
        'input_size': len(FEATURE_COLS),
        'd_model': best_params['d_model'],
        'num_layers': 2,
        'num_heads': 4,
        'patch_size': best_params['patch_len'],
        'window_size': best_params['window_size'],
        'stride': best_params['stride'],
        'num_classes': 2,
        'dropout': best_params['dropout_mlp'],
        'pooling_type': best_params['pooling_type'],
        'mlp_hidden_mult': best_params['mlp_hidden_mult'],
        'activation': best_params['activation'],
        'learning_rate': best_params['learning_rate'],
        'weight_decay': best_params['weight_decay'],
        'loss_type': best_params.get('loss_type', 'ce'),
        'focal_gamma': best_params.get('focal_gamma', None) if best_params.get('loss_type') == 'focal' else None,
        'classification_threshold': CLASSIFICATION_THRESHOLD
    }
    
    try:
        torch.save({
            'model_class': 'PatchSequenceModel',
            'model_args': model_args,
            'state_dict': final_model.state_dict()
        }, model_path)
        print(f"✅ Model saved: {model_path}")
    except Exception as e:
        print(f"❌ Model save failed: {e}")
        return None
    
    # Save training log
    log_path = os.path.join(save_dir, f"training_log_{today_str}.txt")
    with open(log_path, "w") as f:
        f.write(f"Best F1 Score: {study.best_value:.4f}\n")
        f.write("Best Params:\n")
        for k, v in best_params.items():
            f.write(f"  {k}: {v}\n")
    
    # 메모리 정리
    try:
        del final_trainer, final_dataset, final_sequences_scaled
        safe_memory_cleanup()
    except Exception:
        pass
    
    # 캐시 정리
    clear_cache()
    
    return {
        "model_path": model_path,
        "scaler_path": scaler_path,
        "log_path": log_path,
        "best_f1": study.best_value,
        "best_params": best_params
    }