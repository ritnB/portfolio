import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from transformers import Trainer, TrainingArguments, TrainerCallback

from models.timeseries_model import PatchTST, FocalLoss
from data.preprocess import AssetTimeSeriesDataset
from config import (ROLLING_CV_TRAIN_WINDOW, ROLLING_CV_VAL_SIZE, ROLLING_CV_N_SPLITS, 
                   ROLLING_CV_GAP_SIZE, ROLLING_CV_STEP_SIZE)


class CombinedEarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=10, max_loss_diff=0.3):
        self.patience = patience
        self.max_loss_diff = max_loss_diff
        self.wait = 0
        self.best_f1 = None
        self.last_train_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            self.last_train_loss = logs["loss"]

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        eval_f1 = metrics.get("eval_weighted_f1", None)
        eval_loss = metrics.get("eval_loss", None)

        if eval_f1 is None or eval_loss is None or self.last_train_loss is None:
            return control

        loss_diff = eval_loss - self.last_train_loss
        if loss_diff > self.max_loss_diff:
            print(f"⏹️ Early stopping: Overfitting detected (val_loss - train_loss = {loss_diff:.4f})")
            control.should_training_stop = True
            return control

        if self.best_f1 is None or eval_f1 > self.best_f1:
            self.best_f1 = eval_f1
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"⏹️ Early stopping: No F1 improvement for {self.patience} epochs")
                control.should_training_stop = True

        return control

# Rolling Window CV function
def rolling_window_cv_split(combined_df, train_window_size=ROLLING_CV_TRAIN_WINDOW, 
                           val_size=ROLLING_CV_VAL_SIZE, n_splits=ROLLING_CV_N_SPLITS, 
                           gap_size=ROLLING_CV_GAP_SIZE, step_size=ROLLING_CV_STEP_SIZE):
    """
    Rolling Window Cross Validation for time series data
    """
    cv_folds = []
    
    for asset, asset_df in combined_df.groupby('asset'):
        asset_df_sorted = asset_df.sort_values('timestamp').reset_index(drop=True)
        
        # Exclude assets with insufficient data
        if len(asset_df_sorted) < train_window_size + val_size + gap_size + (n_splits-1) * step_size:
            print(f"[⚠️] {asset}: Insufficient data ({len(asset_df_sorted)} records) - excluded from CV")
            continue
            
        asset_folds = []
        for i in range(n_splits):
            start_idx = i * step_size
            train_end_idx = start_idx + train_window_size
            gap_end_idx = train_end_idx + gap_size
            val_end_idx = gap_end_idx + val_size
            
            if val_end_idx <= len(asset_df_sorted):
                fold_train = asset_df_sorted.iloc[start_idx:train_end_idx]
                fold_val = asset_df_sorted.iloc[gap_end_idx:val_end_idx]
                asset_folds.append((fold_train, fold_val))
        
        cv_folds.extend(asset_folds)
    
    return cv_folds

def objective(trial, train_df, val_df, feature_cols):
    # Hyperparameter search ranges (anonymized for portfolio)
    patch_len   = trial.suggest_categorical("patch_len", [8, 16, 32])
    window_size = trial.suggest_categorical("window_size", [64, 128])
    stride      = trial.suggest_categorical("stride", [4, 8])

    # Pruning conditions
    if (
        window_size <= patch_len
        or stride > patch_len
        or (window_size - patch_len) % stride != 0
        or (1 + (window_size - patch_len) // stride) < 3
    ):
        raise optuna.exceptions.TrialPruned()

    d_model       = trial.suggest_categorical("d_model", [64, 128])
    mlp_mult      = trial.suggest_categorical("mlp_hidden_mult", [1, 2])
    dropout       = trial.suggest_float("dropout_mlp", 0.05, 0.15)
    learning_rate = trial.suggest_float("learning_rate", 5e-5, 3e-4, log=True)
    weight_decay  = trial.suggest_float("weight_decay", 0.0, 1e-2)
    activation    = trial.suggest_categorical("activation", ["relu", "gelu"])
    pooling_type  = trial.suggest_categorical("pooling_type", ["cls", "mean"])
    loss_type     = trial.suggest_categorical("loss_type", ["ce", "focal"])
    focal_gamma   = trial.suggest_float("focal_gamma", 1.5, 2.5) if loss_type == "focal" else None

    # Rolling Window CV
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    cv_folds = rolling_window_cv_split(combined_df)
    
    if not cv_folds:
        print("⚠️ All assets excluded from CV. Using holdout validation.")
        # Fallback scaling
        fold_scaler = StandardScaler()
        scaled_train = fold_scaler.fit_transform(train_df[feature_cols])
        scaled_val = fold_scaler.transform(val_df[feature_cols])
        
        fold_train_df = train_df.copy()
        fold_val_df = val_df.copy()
        fold_train_df[feature_cols] = scaled_train
        fold_val_df[feature_cols] = scaled_val
        
        train_dataset = AssetTimeSeriesDataset(fold_train_df, window_size, feature_cols)
        val_dataset = AssetTimeSeriesDataset(fold_val_df, window_size, feature_cols)
        cv_folds = [(fold_train_df, fold_val_df)]
    
    cv_scores = []
    
    for fold_idx, (fold_train_df, fold_val_df) in enumerate(cv_folds):
        # Independent scaling for each fold
        fold_scaler = StandardScaler()
        scaled_fold_train = fold_scaler.fit_transform(fold_train_df[feature_cols])
        scaled_fold_val = fold_scaler.transform(fold_val_df[feature_cols])
        
        # Create new DataFrames with scaled data
        scaled_train_df = fold_train_df.copy()
        scaled_val_df = fold_val_df.copy()
        scaled_train_df[feature_cols] = scaled_fold_train
        scaled_val_df[feature_cols] = scaled_fold_val
        
        train_dataset = AssetTimeSeriesDataset(scaled_train_df, window_size, feature_cols)
        val_dataset = AssetTimeSeriesDataset(scaled_val_df, window_size, feature_cols)
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            continue

        model = PatchTST(
            input_size=len(feature_cols),
            d_model=d_model,
            num_layers=2,
            num_heads=4,
            patch_size=patch_len,
            window_size=window_size,
            stride=stride,
            num_classes=2,
            dropout=dropout,
            pooling_type=pooling_type,
            mlp_hidden_mult=mlp_mult,
            activation=activation
        )

        args = TrainingArguments(
            output_dir=f"./tmp_trial_{trial.number}_fold_{fold_idx}",
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="no",
            num_train_epochs=1000,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            logging_steps=50,
            disable_tqdm=True,
            report_to=[]
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=lambda p: {
                "eval_weighted_f1": f1_score(
                    p.label_ids, np.argmax(p.predictions, axis=1), average="weighted"
                )
            },
            callbacks=[CombinedEarlyStoppingCallback(patience=10, max_loss_diff=0.05)]
        )

        model.focal_loss = FocalLoss(gamma=focal_gamma) if loss_type == "focal" else None
        trainer.train()
        eval_metrics = trainer.evaluate()
        cv_scores.append(eval_metrics["eval_weighted_f1"])
        
        print(f"Fold {fold_idx+1} F1: {eval_metrics['eval_weighted_f1']:.4f}")
    
    if not cv_scores:
        raise optuna.exceptions.TrialPruned()
    
    mean_f1 = np.mean(cv_scores)
    print(f"CV Mean F1: {mean_f1:.4f}")
    return mean_f1
