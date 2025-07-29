# train_patchtst.py (v10_2 compatible modifications)

import os
import numpy as np
import pandas as pd
import torch
import joblib
from datetime import datetime, timedelta

from torch.utils.data import Subset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler
from transformers import Trainer, TrainingArguments
from optuna.trial import TrialState

from models.timeseries_model import PatchTST, FocalLoss
from data.preprocess import AssetTimeSeriesDataset
from data.supabase_io import load_technical_indicators
from utils.training_utils import objective, CombinedEarlyStoppingCallback
# Configuration from config
from config import TEST_DAYS, VALIDATION_DAYS

def train_patchtst_model():
    from optuna import create_study
    from optuna.pruners import MedianPruner

    # ==== Data loading and preprocessing (v10_1 style) ====
    df = load_technical_indicators()  # Asset mapping applied automatically
    df = df[df['price_trend'].isin(['up', 'down'])].copy()
    df['label'] = df['price_trend'].map({'down': 0, 'up': 1})

    feature_cols = [
        'sma', 'ema', 'macd', 'macd_signal', 'macd_diff',
        'rsi', 'stochastic', 'cci'
    ]

    # ==== 3-stage time-based data splitting (v10_2 method) ====
    max_ts = df['timestamp'].max()
    test_cutoff = max_ts - timedelta(days=TEST_DAYS)                    # Recent days (test)
    val_cutoff = max_ts - timedelta(days=TEST_DAYS + VALIDATION_DAYS)   # Previous days (validation)

    train_df = df[df['timestamp'] < val_cutoff].copy()                                          # Remaining (training)
    val_df = df[(df['timestamp'] >= val_cutoff) & (df['timestamp'] < test_cutoff)].copy()      # Validation
    test_df = df[df['timestamp'] >= test_cutoff].copy()                                         # Test

    # ==== Scaling performed independently in CV and final training ====
    # Keep in raw data state (no scaling)
    train_df = train_df.dropna()
    val_df = val_df.dropna()
    test_df = test_df.dropna()
    
    print("⚠️ Scaling performed independently in each CV fold and final training.")

    today_str = datetime.utcnow().strftime("%Y%m%d")
    save_dir = f"/tmp/model_{today_str}"
    os.makedirs(save_dir, exist_ok=True)

    # ==== Optuna tuning (v11_2 Holdout validation) ====
    study = create_study(
        direction="maximize", 
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        study_name="patchtst_holdout_validation"
    )
    study.optimize(lambda trial: objective(trial, train_df, val_df, feature_cols), n_trials=20)

    best_trial = study.best_trial
    params = best_trial.params

    print(f"\n🎯 Best Trial Results:")
    print(f" - Best F1 Score: {best_trial.value:.4f}")
    print(f" - Best Params:")
    for k, v in best_trial.params.items():
        print(f"    {k}: {v}")

    # ==== Final model training (v11_2 method) ====
    window_size = params['window_size']
    
    # Combine Train + Validation data for final model retraining
    combined_train_df = pd.concat([train_df, val_df], ignore_index=True)
    
    # Final model scaling
    final_scaler = StandardScaler()
    scaled_combined = final_scaler.fit_transform(combined_train_df[feature_cols])
    scaled_combined_df = combined_train_df.copy()
    scaled_combined_df[feature_cols] = scaled_combined
    
    train_dataset = AssetTimeSeriesDataset(scaled_combined_df, window_size, feature_cols)
    
    # Save scaler
    scaler_path = os.path.join(save_dir, f"scaler_standard_{today_str}.pkl")
    joblib.dump(final_scaler, scaler_path)
    print(f"✅ StandardScaler saved: {scaler_path}")

    # Retraining train/val split (use part of combined data for validation)
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

    final_trainer = Trainer(
        model=final_model,
        args=final_args,
        train_dataset=tr_ds,
        eval_dataset=vl_ds,
        compute_metrics=lambda p: {
            "eval_weighted_f1": f1_score(p.label_ids, np.argmax(p.predictions, axis=1), average="weighted")
        },
        callbacks=[CombinedEarlyStoppingCallback(patience=10, max_loss_diff=0.05)]
    )

    # Start retraining
    final_trainer.train()

    # ==== Test dataset evaluation (v11_2 method) ====
    final_model.eval()
    
    # Test dataset composition (using final_scaler)
    scaled_test = final_scaler.transform(test_df[feature_cols])
    scaled_test_df = test_df.copy()
    scaled_test_df[feature_cols] = scaled_test
    
    test_dataset = AssetTimeSeriesDataset(scaled_test_df, window_size, feature_cols)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_preds, all_labels = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_model.to(device)

    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)
            labels = batch['labels'].to(device)

            outputs = final_model(x)
            logits = outputs['logits']
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Print evaluation metrics
    print("\n📊 Test Set Evaluation Metrics:")
    print(classification_report(all_labels, all_preds, digits=4))

    f1 = f1_score(all_labels, all_preds, average="weighted")
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    print(f"✅ Weighted F1 Score: {f1:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall:    {recall:.4f}")

    # ==== Model saving (v11_2 method) ====
    model_path = os.path.join(save_dir, f"patchtst_final_model_{today_str}.pt")
    
    # Model args configuration (same as v11_2)
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
        'loss_type': params.get('loss_type', 'ce'),
        'focal_gamma': params.get('focal_gamma', None) if params.get('loss_type') == 'focal' else None
    }
    
    torch.save({
        'model_class': 'PatchTST',
        'model_args': model_args,
        'state_dict': final_model.state_dict()
    }, model_path)

    # Log saving
    log_path = os.path.join(save_dir, f"training_log_{today_str}.txt")
    with open(log_path, "w") as f:
        f.write("\n".join([
            f"🎯 Best Trial Results:",
            f"Best F1 Score: {best_trial.value:.4f}",
            f"Best Params:",
            *[f"  {k}: {v}" for k, v in best_trial.params.items()],
            f"",
            f"📊 Test Set Results:",
            f"Weighted F1 Score: {f1:.4f}",
            f"Precision: {precision:.4f}",
            f"Recall: {recall:.4f}"
        ]))

    print(f"\n✅ Final model saved (for service): {model_path}")

    return {
        "model_path": model_path,
        "scaler_path": scaler_path,
        "log_path": log_path
    }
