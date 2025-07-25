# ✅ train_patchtst.py (v10_2 호환 수정)

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

from models.timeseries_model import TimeSeriesTransformer, ModelConfig
from data.preprocess import CoinTimeSeriesDataset
from data.supabase_io import load_technical_indicators
from utils.training_utils import objective, CombinedEarlyStoppingCallback, setup_training_environment

# Configuration constants (actual values are proprietary)
TEST_DAYS = 30
VALIDATION_DAYS = 30

def train_patchtst_model():
    """
    Train the time series prediction model.
    
    Note: This is a simplified version for portfolio demonstration.
    Actual training pipeline contains proprietary optimizations and strategies.
    """
    from optuna import create_study
    from optuna.pruners import MedianPruner
    
    setup_training_environment()

    # Data loading and preprocessing
    df = load_technical_indicators()  # Coin mapping automatically applied
    df = df[df['price_trend'].isin(['up', 'down'])].copy()
    df['label'] = df['price_trend'].map({'down': 0, 'up': 1})

    # Use abstracted feature columns
    from config import FEATURE_COLS
    feature_cols = FEATURE_COLS

    # 3-stage time-based data splitting
    max_ts = df['timestamp'].max()
    test_cutoff = max_ts - timedelta(days=TEST_DAYS)
    val_cutoff = max_ts - timedelta(days=TEST_DAYS + VALIDATION_DAYS)

    train_df = df[df['timestamp'] < val_cutoff].copy()
    val_df = df[(df['timestamp'] >= val_cutoff) & (df['timestamp'] < test_cutoff)].copy()
    test_df = df[df['timestamp'] >= test_cutoff].copy()

    # Feature scaling
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_df[feature_cols])
    scaled_val = scaler.transform(val_df[feature_cols])
    scaled_test = scaler.transform(test_df[feature_cols])

    train_df = train_df.copy()
    val_df = val_df.copy() 
    test_df = test_df.copy()
    train_df[feature_cols] = scaled_train
    val_df[feature_cols] = scaled_val
    test_df[feature_cols] = scaled_test

    train_df = train_df.dropna()
    val_df = val_df.dropna()
    test_df = test_df.dropna()

    today_str = datetime.utcnow().strftime("%Y%m%d")
    save_dir = f"/tmp/model_{today_str}"
    os.makedirs(save_dir, exist_ok=True)

    scaler_path = os.path.join(save_dir, f"scaler_{today_str}.pkl")
    joblib.dump(scaler, scaler_path)

    # Simplified hyperparameter tuning (actual search space is proprietary)
    study = create_study(
        direction="maximize", 
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=0),
        study_name="timeseries_model_tuning"
    )
    
    # Reduced trials for demo (actual optimization uses more trials)
    study.optimize(lambda trial: objective(trial, train_df, val_df, feature_cols), n_trials=5)

    best_trial = study.best_trial
    params = best_trial.params

    print(f"\n🎯 Best Trial Results:")
    print(f" - Best F1 Score: {best_trial.value:.4f}")
    print(f" - Best Window Size: {params['window_size']}")

    # Final model training with simplified configuration
    window_size = params['window_size']
    
    # Combine Train + Validation data for final model retraining
    combined_train_df = pd.concat([train_df, val_df], ignore_index=True)
    train_dataset = CoinTimeSeriesDataset(combined_train_df, window_size, feature_cols)

    # Split for retraining
    val_ratio = 0.1
    split_idx = int(len(train_dataset) * (1 - val_ratio))
    tr_ds = Subset(train_dataset, list(range(0, split_idx)))
    vl_ds = Subset(train_dataset, list(range(split_idx, len(train_dataset))))

    # Create final model
    config = ModelConfig(
        n_features=len(feature_cols),
        n_classes=2,
        window_size=window_size
    )
    final_model = TimeSeriesTransformer(config)

    # Training configuration (simplified)
    final_training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=10,
        per_device_train_batch_size=params.get('batch_size', 32),
        learning_rate=params.get('learning_rate', 1e-4),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        no_cuda=False if torch.cuda.is_available() else True,
    )

    final_trainer = Trainer(
        model=final_model,
        args=final_training_args,
        train_dataset=tr_ds,
        eval_dataset=vl_ds,
        callbacks=[CombinedEarlyStoppingCallback(patience=5)]
    )

    # Start retraining
    final_trainer.train()

    # Test dataset evaluation
    final_model.eval()
    test_dataset = CoinTimeSeriesDataset(test_df, window_size, feature_cols)
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
    print(f"✅ Recall: {recall:.4f}")

    # Model saving with simplified parameters
    model_path = os.path.join(save_dir, f"model_{today_str}.pt")
    torch.save({
        "state_dict": final_model.state_dict(),
        "model_args": {
            "n_features": len(feature_cols),
            "n_classes": 2,
            "window_size": window_size,
        }
    }, model_path)

    log_path = os.path.join(save_dir, f"training_log_{today_str}.txt")
    with open(log_path, "w") as f:
        f.write("\n".join([
            f"📊 Test Set Results:",
            f"Weighted F1 Score: {f1:.4f}",
            f"Precision: {precision:.4f}",
            f"Recall: {recall:.4f}"
        ]))

    print(f"\n✅ Final model saved successfully (for service): {model_path}")

    return {
        "model_path": model_path,
        "scaler_path": scaler_path,
        "log_path": log_path
    }
