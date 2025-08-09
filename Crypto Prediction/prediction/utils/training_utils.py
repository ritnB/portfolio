"""Training utilities and callbacks (v11_2-compatible)."""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import Trainer, TrainingArguments, TrainerCallback
from torch.utils.data import Subset

# Optuna
try:
    import optuna
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not installed. Hyperparameter tuning disabled")

from models.timeseries_model import PatchSequenceModel, FocalLoss
from data.preprocess import CoinTimeSeriesDataset, get_sequences, rolling_window_cv_split
from utils.memory_utils import safe_memory_cleanup, monitor_memory_usage, check_memory_limit
from config import OPTUNA_PARAM_SPACE

# Fold-wise scaler cache
_fold_scaler_cache = {}  # {(window_size, fold_idx): scaler}


def get_cached_fold_scaler(window_size, fold_idx, train_sequences, feature_cols, force_regenerate=False):
    """Get or build a cached StandardScaler for a given fold."""
    global _fold_scaler_cache
    
    cache_key = (window_size, fold_idx)
    
    # Build when forced or missing in cache
    if force_regenerate or cache_key not in _fold_scaler_cache:
        print(f"🔄 Building scaler (window_size={window_size}, fold={fold_idx})...")
        scaler = StandardScaler()
        
        # Fit on train sequences only
        for seq in train_sequences:
            scaler.partial_fit(seq[feature_cols])
        
        _fold_scaler_cache[cache_key] = scaler
        print(f"✅ Scaler cached (window_size={window_size}, fold={fold_idx})")
    else:
        print(f"♻️ Using cached scaler (window_size={window_size}, fold={fold_idx})")
    
    return _fold_scaler_cache[cache_key]


def clear_fold_scaler_cache():
    """Clear fold-wise scaler cache."""
    global _fold_scaler_cache
    cache_count = len(_fold_scaler_cache)
    _fold_scaler_cache.clear()
    print(f"🧹 Cleared scaler cache ({cache_count} entries removed)")


class EnhancedEarlyStoppingCallback(TrainerCallback):
    """Enhanced Early Stopping Callback (v11_2)."""
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

        # No improvement streak
        if self.epochs_without_improvement >= 10:
            print(f"⏹️ Enhanced Early stopping: no improvement for 10 evaluations")
            control.should_training_stop = True
            return control

        return control


class OptunaPruningCallback(TrainerCallback):
    """Optuna Pruning Callback (v11_2)."""
    def __init__(self, trial):
        self.trial = trial
        self.last_reported_step = -1
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and state.global_step > self.last_reported_step:
            eval_f1 = metrics.get("eval_weighted_f1", 0)
            self.trial.report(eval_f1, step=state.global_step)
            self.last_reported_step = state.global_step
            
            if self.trial.should_prune():
                print(f"⏹️ Optuna Pruning: Trial {self.trial.number} pruned")
                control.should_training_stop = True
                if OPTUNA_AVAILABLE:
                    raise optuna.exceptions.TrialPruned()


def objective(trial, train_df, feature_cols):
    """Optuna objective (holdout validation), v11_2 compatible."""
    trial_start_time = time.time()
    
    print(f"\n🔄 Trial {trial.number} starting... (memory: {monitor_memory_usage():.1f}MB)")
    
    # Memory check
    if check_memory_limit():
        print(f"⏹️ Trial {trial.number} pruned (memory limit)")
        if OPTUNA_AVAILABLE:
            raise optuna.exceptions.TrialPruned()
        else:
            return 0.0
    
    # Get hyperparameter space from config
    params = OPTUNA_PARAM_SPACE
    
    # Hyperparameter definitions
    patch_len = trial.suggest_categorical("patch_len", params["patch_len"])
    window_size = trial.suggest_categorical("window_size", params["window_size"])
    stride = trial.suggest_categorical("stride", params["stride"])
    
    # Early pruning constraints
    if (
        window_size <= patch_len
        or stride > patch_len
        or (window_size - patch_len) % stride != 0
        or (1 + (window_size - patch_len) // stride) < 3
        or stride > window_size // 4
        or patch_len > window_size // 2
    ):
        print(f"⏹️ Trial {trial.number} pruned (parameter constraints)")
        if OPTUNA_AVAILABLE:
            raise optuna.exceptions.TrialPruned()
        else:
            return 0.0
    
    # Remaining hyperparameters
    d_model = trial.suggest_categorical("d_model", params["d_model"])
    mlp_mult = trial.suggest_categorical("mlp_hidden_mult", params["mlp_hidden_mult"])
    
    # Ranged parameters
    dropout_spec = params["dropout_mlp"]
    dropout = trial.suggest_float("dropout_mlp", dropout_spec["min"], dropout_spec["max"])
    
    lr_spec = params["learning_rate"]
    learning_rate = trial.suggest_float("learning_rate", lr_spec["min"], lr_spec["max"], log=lr_spec.get("log", False))
    
    wd_spec = params["weight_decay"]
    weight_decay = trial.suggest_float("weight_decay", wd_spec["min"], wd_spec["max"])
    
    threshold = 0.5  # fixed threshold
    
    # Categorical parameters
    activation = trial.suggest_categorical("activation", params["activation"])
    pooling_type = trial.suggest_categorical("pooling_type", params["pooling_type"])
    loss_type = trial.suggest_categorical("loss_type", params["loss_type"])
    
    # Focal loss parameter
    if loss_type == "focal":
        gamma_spec = params["focal_gamma"]
        focal_gamma = trial.suggest_float("focal_gamma", gamma_spec["min"], gamma_spec["max"])
    else:
        focal_gamma = None
    
    # Constants
    num_layers = params["num_layers"]
    num_heads = params["num_heads"]
    batch_size = params["batch_size"]
    
    try:
        # Apply rolling CV (v11_2 style)
        from data.preprocess import split_by_time_gap, batch_sequence_processing
        
        # Use entire train_df for rolling CV
        combined_split_data = split_by_time_gap(train_df, max_gap_hours=24)
        combined_sequences = batch_sequence_processing(combined_split_data, window_size)
        
        if not combined_sequences:
            print("⚠️ Failed to generate sequences")
            if OPTUNA_AVAILABLE:
                raise optuna.exceptions.TrialPruned()
            else:
                return 0.0
        
        # Rolling Window CV split (original sequences)
        cv_folds = rolling_window_cv_split(combined_sequences)
        
        if not cv_folds:
            print("⚠️ Failed to create CV folds")
            if OPTUNA_AVAILABLE:
                raise optuna.exceptions.TrialPruned()
            else:
                return 0.0
        
        cv_scores = []
        
        # Evaluate from the latest folds first
        for fold_idx, fold_data in enumerate(reversed(cv_folds)):
            train_sequences = fold_data['train']
            val_sequences = fold_data['validation']
            
            # Cache scaler per fold and apply
            fold_scaler = get_cached_fold_scaler(window_size, fold_idx, train_sequences, feature_cols)
            
            # Apply scaling to train sequences
            train_sequences_scaled = []
            for seq in train_sequences:
                seq_scaled = seq.copy()
                seq_scaled[feature_cols] = fold_scaler.transform(seq[feature_cols])
                train_sequences_scaled.append(seq_scaled)
            
            # Apply scaling to validation sequences
            val_sequences_scaled = []
            for seq in val_sequences:
                seq_scaled = seq.copy()
                seq_scaled[feature_cols] = fold_scaler.transform(seq[feature_cols])
                val_sequences_scaled.append(seq_scaled)
            
            # Create datasets (scaled sequences)
            train_dataset = CoinTimeSeriesDataset(train_sequences_scaled, feature_cols)
            val_dataset = CoinTimeSeriesDataset(val_sequences_scaled, feature_cols)
            
            if len(train_dataset) == 0 or len(val_dataset) == 0:
                continue

            # Create model (new per fold)
            model = PatchSequenceModel(
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

            # Training Arguments
            training_args = TrainingArguments(
            output_dir=f"./tmp_trial_{trial.number}_fold_{fold_idx}",
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=32,
            eval_strategy="epoch",
                save_strategy="no",
                num_train_epochs=50,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            logging_steps=999999,  # suppressed by design
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

            # Pruning callback (first fold only)
            callbacks = [OptunaPruningCallback(trial)] if fold_idx == 0 else []
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics_with_threshold,
                callbacks=callbacks
            )

            # Set FocalLoss if selected
            model.focal_loss = FocalLoss(gamma=focal_gamma) if loss_type == "focal" else None
            
            try:
                # Train (suppress verbose output)
                import sys
                from io import StringIO
                
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                
                try:
                    trainer.train()
                    eval_metrics = trainer.evaluate()
                finally:
                    sys.stdout = old_stdout
                
                # Collect CV scores
                if 'eval_weighted_f1' in eval_metrics:
                    cv_scores.append(eval_metrics)
                    print(f" Fold{fold_idx+1}={eval_metrics['eval_weighted_f1']:.3f}", end="")
                else:
                    continue
                    
            except optuna.exceptions.TrialPruned:
                raise
            except Exception as e:
                print(f" Fold{fold_idx+1}=Error", end="")
                continue
            finally:
                # Cleanup per fold
                try:
                    del model, trainer, train_dataset, val_dataset
                    safe_memory_cleanup()
                except:
                    pass
        
        if not cv_scores:
            if OPTUNA_AVAILABLE:
                raise optuna.exceptions.TrialPruned()
            else:
                return 0.0
        
        # Compute mean F1
        mean_f1 = np.mean([score["eval_weighted_f1"] for score in cv_scores])
        total_time = time.time() - trial_start_time
        
        print(f" → CV_F1={mean_f1:.3f} | time={total_time:.0f}s")
        
        return mean_f1
        
    except Exception as e:
        if OPTUNA_AVAILABLE and isinstance(e, optuna.exceptions.TrialPruned):
            raise
        print(f"⚠️ Trial {trial.number} error: {type(e).__name__}")
        return 0.0
    finally:
        # Memory cleanup (also happens per fold)
        safe_memory_cleanup()


def create_optuna_study(n_trials: int = 20):
    """Create and return an Optuna Study."""
    if not OPTUNA_AVAILABLE:
        print("❌ Optuna not installed; skipping hyperparameter tuning.")
        return None
    
    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    return study


def safe_model_save(model, path, model_args):
    """Safely save model (v11_2 compatible format)."""
    try:
        # Cleanup before saving
        safe_memory_cleanup()
        
        torch.save({
            'model_class': 'PatchSequenceModel',
            'model_args': model_args,
            'state_dict': model.state_dict()
        }, path)
        
        print(f"✅ Model saved: {path}")
        
    except Exception as e:
        print(f"❌ Model save failed: {e}")
        # Try backup path
        backup_path = path.replace('.pt', '_backup.pt')
        try:
            torch.save({
                'model_class': 'PatchSequenceModel',
                'model_args': model_args,
                'state_dict': model.state_dict()
            }, backup_path)
            print(f"✅ Backup model saved: {backup_path}")
        except Exception as backup_error:
            print(f"❌ Backup save failed: {backup_error}")


def load_and_preprocess_data(feature_cols, test_days=10):
    """Load and preprocess data (legacy compatibility)."""
    from data.supabase_io import load_technical_indicators
    from datetime import timedelta
    
    print("🔄 Loading and preprocessing data...")
    
    # Memory check
    if check_memory_limit():
        print("⚠️ Aborting due to insufficient memory")
        return None, None
    
    try:
        # Load data
        df = load_technical_indicators(for_training=True)
        
        # Basic preprocessing
        df = df.dropna()
        df = df[df['price_trend'].isin(['up', 'down'])].copy()
        df['label'] = (df['price_trend'] == 'up').astype(int)
        
        # Train/Test split
        max_ts = df['timestamp'].max()
        test_cutoff = max_ts - timedelta(days=test_days)
        
        train_df = df[df['timestamp'] < test_cutoff].copy()
        test_df = df[df['timestamp'] >= test_cutoff].copy()
        
        print(f"✅ Preprocessing complete!")
        print(f"  - train: {len(train_df):,}")
        print(f"  - test: {len(test_df):,}")
        
        return train_df, test_df
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None, None