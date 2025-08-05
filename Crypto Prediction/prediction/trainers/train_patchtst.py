# ✅ train_patchtst.py (v10_2 compatible modification)

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import joblib
from datetime import datetime, timedelta

from torch.utils.data import Subset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler
from transformers import Trainer, TrainingArguments
from optuna.trial import TrialState

from models.timeseries_model import PatchTST, FocalLoss
from data.preprocess import CoinTimeSeriesDataset, get_sequences, split_by_time_gap, batch_sequence_processing, safe_parse_timestamp
from data.supabase_io import load_technical_indicators
from utils.training_utils import objective, EnhancedEarlyStoppingCallback, create_optuna_study, safe_model_save
from utils.memory_utils import safe_memory_cleanup, monitor_memory_usage, check_memory_limit
# Settings from config
from config import TEST_DAYS, VALIDATION_DAYS, DEFAULT_CLASSIFICATION_THRESHOLD, FEATURE_COLS

def train_patchtst_model():
    """v11_2 compatible model training pipeline"""
    from optuna import create_study
    from optuna.pruners import MedianPruner

    print("🚀 v11_2 compatible model training started...")
    print(f"📊 Initial memory: {monitor_memory_usage():.1f}MB")

    # ==== v11_2 Data Load and Preprocessing ====
    df = load_technical_indicators(for_training=True)  # Load data for retrain
    
    # v11_2 compatible timestamp processing
    df['timestamp'] = safe_parse_timestamp(df['timestamp'])
    
    # Data cleaning
    df = df.dropna()
    df = df[df['price_trend'].isin(['up', 'down'])].copy()
    df['label'] = df['price_trend'].map({'down': 0, 'up': 1})

    # v11_2 compatible feature columns (same as processing)
    feature_cols = FEATURE_COLS

    # ==== v11_2 Two-stage time-based data split ====
    max_ts = df['timestamp'].max()
    test_cutoff = max_ts - timedelta(days=TEST_DAYS)  # Only last 10 days for test

    train_df = df[df['timestamp'] < test_cutoff].copy()   # All others (for training)
    test_df = df[df['timestamp'] >= test_cutoff].copy()   # For test

    # Keep as raw data (scaling is done in CV)
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    
    print("✅ v11_2: Only test is separated, all others used for rolling CV")

    today_str = datetime.utcnow().strftime("%Y%m%d")
    save_dir = f"/tmp/model_{today_str}"
    os.makedirs(save_dir, exist_ok=True)

    # Data split statistics
    print(f"📊 Data split stats:")
    print(f"  - Total: {len(df):,}")
    print(f"  - Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  - Test: {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Memory check
    if check_memory_limit():
        print("⚠️ Memory warning")
        safe_memory_cleanup()

    # ==== v11_2 compatible Optuna tuning ====
    study = create_study(
        direction="maximize", 
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        study_name="patchtst_holdout_validation"
    )
    
    if study is None:
        print("⚠️ Optuna disabled - training with default parameters")
        # Default parameter settings
        best_params = {
            'patch_len': 8, 'window_size': 64, 'stride': 4,
            'd_model': 128, 'mlp_hidden_mult': 2, 'dropout_mlp': 0.2,
            'learning_rate': 1e-4, 'weight_decay': 0.03,
            'activation': 'relu', 'pooling_type': 'cls',
            'loss_type': 'ce', 'classification_threshold': 0.5
        }
        best_value = 0.0
    else:
        print(f"🎯 Starting Optuna optimization (20 trials)...")
        study.optimize(lambda trial: objective(trial, train_df, feature_cols), n_trials=20)
        best_params = study.best_params
        best_value = study.best_value

    params = best_params

    print(f"\n🎯 Optimization results:")
    print(f" - Best F1 Score: {best_value:.4f}")
    print(f" - Best Params:")
    for k, v in params.items():
        print(f"    {k}: {v}")

    # ==== v11_2 Final Model Training ====
    window_size = params['window_size']
    threshold = params.get('classification_threshold', DEFAULT_CLASSIFICATION_THRESHOLD)
    
    print(f"\n🔄 v11_2 Final Model Training Start...")
    print(f"  - window_size: {window_size}")
    print(f"  - threshold: {threshold}")
    
    # Use all train_df for final model training
    combined_train_df = train_df.copy()
    print(f"  - Final training data: {len(combined_train_df):,}")
    
    # v11_2 sequence generation
    combined_split_data = split_by_time_gap(combined_train_df, max_gap_hours=24)
    combined_sequences = batch_sequence_processing(combined_split_data, window_size)
    
    if not combined_sequences:
        print("❌ Final model training failed: sequence generation failed")
        return None
    
    # Train scaler (v11_2 method)
    final_scaler = StandardScaler()
    base_sequences = combined_sequences[:int(len(combined_sequences) * 0.6)]
    
    for seq in base_sequences:
        final_scaler.partial_fit(seq[feature_cols])
    
    # Apply scaling to all sequences
    scaled_sequences = []
    for seq in combined_sequences:
        seq_scaled = seq.copy()
        seq_scaled[feature_cols] = final_scaler.transform(seq[feature_cols])
        scaled_sequences.append(seq_scaled)
    
    # v11_2 compatible dataset creation
    train_dataset = CoinTimeSeriesDataset(scaled_sequences, feature_cols)
    print(f"  - Final dataset: {len(train_dataset):,} samples")
    
    # Save scaler
    scaler_path = os.path.join(save_dir, f"scaler_standard_{today_str}.pkl")
    joblib.dump(final_scaler, scaler_path)
    print(f"✅ StandardScaler saved: {scaler_path}")

    # For retraining, split train/val (use part of combined data for validation)
    val_ratio = 0.1
    split_idx = int(len(train_dataset) * (1 - val_ratio))
    tr_ds = Subset(train_dataset, list(range(0, split_idx)))
    vl_ds = Subset(train_dataset, list(range(split_idx, len(train_dataset))))

    final_model = PatchTST(
        input_size=len(feature_cols),
        d_model=params['d_model'],
        num_layers=2,
        num_heads=4,
        patch_size=params['patch_len'],
        window_size=window_size,
        stride=params['stride'],
        num_classes=2,
        dropout=params['dropout_mlp'],
        pooling_type=params['pooling_type'],
        mlp_hidden_mult=params['mlp_hidden_mult'],
        activation=params['activation']
    )
    if params['loss_type'] == 'focal':
        final_model.focal_loss = FocalLoss(gamma=params.get('focal_gamma', 2.0))

    final_args = TrainingArguments(
        output_dir=f"{save_dir}/best_patchtst_model",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=1000,
        learning_rate=params['learning_rate'],
        weight_decay=params['weight_decay'],
        logging_steps=50,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="eval_weighted_f1"
    )

    # 🎯 Custom compute_metrics function with threshold (for final training)
    def compute_metrics_with_threshold_final(p):
        # Calculate probabilities with softmax
        probs = F.softmax(torch.tensor(p.predictions), dim=1)
        class_1_probs = probs[:, 1].numpy()  # Probability of class 1 (up)
        
        # Apply threshold
        preds = (class_1_probs > threshold).astype(int)
        
        return {
            "eval_weighted_f1": f1_score(p.label_ids, preds, average="weighted")
        }

    final_trainer = Trainer(
        model=final_model,
        args=final_args,
        train_dataset=tr_ds,
        eval_dataset=vl_ds,
        compute_metrics=compute_metrics_with_threshold_final,
        callbacks=[EnhancedEarlyStoppingCallback(patience=8, max_loss_diff=0.05)]  # Enhanced Early Stopping
    )

    # 🔁 Start retraining
    final_trainer.train()

    # ==== Test dataset evaluation (v11_2 method) ====
    final_model.eval()
    
    # Build test dataset (v11_2 method)
    test_split_data = split_by_time_gap(test_df, max_gap_hours=24)
    test_sequences = batch_sequence_processing(test_split_data, window_size)
    
    # Apply scaling to test sequences
    scaled_test_sequences = []
    for seq in test_sequences:
        seq_scaled = seq.copy()
        seq_scaled[feature_cols] = final_scaler.transform(seq[feature_cols])
        scaled_test_sequences.append(seq_scaled)
    
    test_dataset = CoinTimeSeriesDataset(scaled_test_sequences, feature_cols)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_preds, all_labels = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_model.to(device)

    # Evaluation loop (apply threshold)
    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)               # Move to GPU
            labels = batch['labels'].to(device)     # Move to GPU

            outputs = final_model(x)
            logits = outputs['logits']
            
            # Prediction with threshold
            probs = F.softmax(logits, dim=1)
            class_1_probs = probs[:, 1]  # Probability of class 1 (up)
            preds = (class_1_probs > threshold).long()

            all_preds.extend(preds.cpu().numpy())       # Move to CPU and convert to numpy
            all_labels.extend(labels.cpu().numpy())     # Move to CPU and convert to numpy

    # 🧾 Print evaluation metrics
    print("\n📊 Test Set Evaluation Metrics:")
    print(classification_report(all_labels, all_preds, digits=4))

    f1 = f1_score(all_labels, all_preds, average="weighted")
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    print(f"✅ Weighted F1 Score: {f1:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall:    {recall:.4f}")

    # ==== v11_2 compatible model save ====
    model_path = os.path.join(save_dir, f"patchtst_final_model_{today_str}.pt")
    
    # v11_2 fully compatible model_args
    model_args = {
        'input_size': len(feature_cols),
        'd_model': params['d_model'],
        'num_layers': 2,
        'num_heads': 4,
        'patch_size': params['patch_len'],
        'window_size': window_size,
        'stride': params['stride'],
        'num_classes': 2,
        'dropout': params['dropout_mlp'],
        'pooling_type': params['pooling_type'],
        'mlp_hidden_mult': params['mlp_hidden_mult'],
        'activation': params['activation'],
        'learning_rate': params['learning_rate'],
        'weight_decay': params['weight_decay'],
        'loss_type': params.get('loss_type', 'ce'),
        'focal_gamma': params.get('focal_gamma', None) if params.get('loss_type') == 'focal' else None,
        'classification_threshold': threshold  # v11_2 compatible threshold
    }
    
    # v11_2 compatible safe model save
    safe_model_save(final_model, model_path, model_args)

    # Save log
    log_path = os.path.join(save_dir, f"training_log_{today_str}.txt")
    with open(log_path, "w") as f:
        f.write("\n".join([
            f"🎯 Best Trial Results:",
            f"Best F1 Score: {best_value:.4f}",
            f"Best Params:",
            *[f"  {k}: {v}" for k, v in params.items()],
            f"",
            f"📊 Test Set Results:",
            f"Weighted F1 Score: {f1:.4f}",
            f"Precision: {precision:.4f}",
            f"Recall: {recall:.4f}"
        ]))

    print(f"\n✅ v11_2 compatible model saved: {model_path}")
    print(f"📊 Final memory: {monitor_memory_usage():.1f}MB")
    
    # Memory cleanup
    safe_memory_cleanup()

    return {
        "model_path": model_path,
        "scaler_path": scaler_path,
        "log_path": log_path,
        "best_f1": best_value,
        "best_params": params
    }
