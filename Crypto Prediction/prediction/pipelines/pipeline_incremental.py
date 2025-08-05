# pipeline_incremental.py

import os
from datetime import datetime, timedelta
import torch
import torch.optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score

from models.timeseries_model import load_model_from_checkpoint
from data.supabase_io import load_technical_indicators
from data.preprocess import AssetTimeSeriesDataset, split_by_time_gap, batch_sequence_processing
from utils.gcs_utils import upload_to_gcs, download_latest_model_and_scaler, verify_model_compatibility
from utils.memory_utils import safe_memory_cleanup, monitor_memory_usage, check_memory_limit
from config import (GCS_BUCKET_NAME, GCS_MODEL_DIR, FEATURE_COLS,
                   INCREMENTAL_DAYS, INCREMENTAL_EPOCHS, INCREMENTAL_LR, INCREMENTAL_BATCH_SIZE,
                   INCREMENTAL_PERFORMANCE_THRESHOLD)

# Incremental learning safety settings (anonymized)
INCREMENTAL_VALIDATION_RATIO = 0.2        # Validation data ratio
INCREMENTAL_MAX_DEGRADATION = 0.05        # Maximum performance degradation allowance
INCREMENTAL_MIN_IMPROVEMENT = 0.01        # Minimum performance improvement requirement

def load_model_and_scaler():
    """Load latest model and scaler from Cloud Storage"""
    print("🔄 Downloading compatible model from Cloud Storage...")
    
    # Temporary file paths
    model_path = "/tmp/current_model.pt"
    scaler_path = "/tmp/current_scaler.pkl"
    
    # Download compatible model/scaler and check integrity
    model_success, scaler_success = download_latest_model_and_scaler(
        GCS_BUCKET_NAME, model_path, scaler_path
    )
    
    if not model_success:
        raise Exception("Compatible model not found in Cloud Storage.")
    
    if not scaler_success:
        raise Exception("Compatible scaler not found in Cloud Storage.")
    
    print("✅ Compatible model and scaler loaded successfully")
    return model_path, scaler_path

def load_incremental_data():
    """Load recent data for incremental learning"""
    try:
        # Load recent data for incremental learning
        
        print(f"🔄 Loading recent {INCREMENTAL_DAYS} days of data...")
        
        # Load compatible data (including trained timestampz processing)
        df = load_technical_indicators(for_training=True)
        
        if df.empty:
            print("⚠️ No data available for incremental learning")
            return df
        
        # Compatible preprocessing
        df = df.dropna()
        df = df[df['price_trend'].isin(['up', 'down'])].copy()
        df['label'] = df['price_trend'].map({'down': 0, 'up': 1})
        
        print(f"✅ Incremental learning data: {len(df):,} records")
        return df
        
    finally:
        # Incremental learning completed
        pass

def evaluate_model_performance(model, test_data, scaler, window_size):
    """Evaluate compatible model performance"""
    if test_data.empty:
        return 0.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Compatible sequence generation
    split_data = split_by_time_gap(test_data, max_gap_hours=24)
    sequences = batch_sequence_processing(split_data, window_size)
    
    if not sequences:
        return 0.0
    
    # Apply scaling
    scaled_sequences = []
    for seq in sequences:
        seq_scaled = seq.copy()
        seq_scaled[FEATURE_COLS] = scaler.transform(seq[FEATURE_COLS])
        scaled_sequences.append(seq_scaled)
    
    # Create compatible dataset
    dataset = AssetTimeSeriesDataset(scaled_sequences, FEATURE_COLS)
    if len(dataset) == 0:
        return 0.0
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch['x'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(x)
            logits = outputs['logits']
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    if not all_preds:
        return 0.0
    
    return f1_score(all_labels, all_preds, average="weighted")

def incremental_update_with_validation(model, new_data, scaler, window_size):
    """Update model with validation included"""
    if new_data.empty:
        return model, scaler, False
    
    print(f"🔄 Starting incremental learning (Memory: {monitor_memory_usage():.1f}MB)")
    
    # Memory check
    if check_memory_limit():
        print("⚠️ Incremental learning aborted due to insufficient memory.")
        return model, scaler, False
    
    # Data splitting (train/validation)
    split_idx = int(len(new_data) * (1 - INCREMENTAL_VALIDATION_RATIO))
    train_data = new_data.iloc[:split_idx]
    val_data = new_data.iloc[split_idx:]
    
    if train_data.empty or val_data.empty:
        print("⚠️ Insufficient data for incremental learning.")
        return model, scaler, False
    
    print(f"  - Training data: {len(train_data):,}")
    print(f"  - Validation data: {len(val_data):,}")
    
    # Measure original model performance
    original_performance = evaluate_model_performance(model, val_data, scaler, window_size)
    print(f"  - Original model performance: {original_performance:.4f}")
    
    # Performance threshold check
    if original_performance < INCREMENTAL_PERFORMANCE_THRESHOLD:
        print(f"⚠️ Original model performance below threshold ({INCREMENTAL_PERFORMANCE_THRESHOLD})")
        return model, scaler, False
    
    # Device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Prepare compatible data
    train_split_data = split_by_time_gap(train_data, max_gap_hours=24)
    train_sequences = batch_sequence_processing(train_split_data, window_size)
    
    if not train_sequences:
        print("⚠️ Failed to generate training sequences")
        return model, scaler, False
    
    # Apply scaling
    scaled_train_sequences = []
    for seq in train_sequences:
        seq_scaled = seq.copy()
        seq_scaled[FEATURE_COLS] = scaler.transform(seq[FEATURE_COLS])
        scaled_train_sequences.append(seq_scaled)
    
    # Create compatible dataset
    dataset = AssetTimeSeriesDataset(scaled_train_sequences, FEATURE_COLS)
    if len(dataset) == 0:
        print("⚠️ Failed to create dataset")
        return model, scaler, False
    
    print(f"  - Training samples: {len(dataset):,}")
    
    # Incremental learning
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=INCREMENTAL_LR)
    
    dataloader = DataLoader(dataset, batch_size=INCREMENTAL_BATCH_SIZE, shuffle=True)
    
    for epoch in range(INCREMENTAL_EPOCHS):
        total_loss = 0
        for batch in dataloader:
            x = batch['x'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(x, labels)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"    Epoch {epoch+1}/{INCREMENTAL_EPOCHS}: Loss {avg_loss:.4f}")
        
        # Memory cleanup
        if (epoch + 1) % 2 == 0:
            safe_memory_cleanup()
    
    # Measure updated model performance
    updated_performance = evaluate_model_performance(model, val_data, scaler, window_size)
    print(f"  - Updated model performance: {updated_performance:.4f}")
    
    # Performance verification
    performance_improvement = updated_performance - original_performance
    print(f"  - Performance change: {performance_improvement:+.4f}")
    
    # Validation conditions
    if updated_performance < INCREMENTAL_PERFORMANCE_THRESHOLD:
        print(f"⚠️ Performance below threshold: {updated_performance:.4f}")
        return model, scaler, False
    
    if performance_improvement < -INCREMENTAL_MAX_DEGRADATION:
        print(f"⚠️ Performance degradation too large: {performance_improvement:.4f}")
        return model, scaler, False
    
    if performance_improvement < INCREMENTAL_MIN_IMPROVEMENT:
        print(f"⚠️ Performance improvement too small: {performance_improvement:.4f}")
        return model, scaler, False
    
    print(f"✅ Incremental learning successful: {original_performance:.4f} → {updated_performance:.4f}")
    return model, scaler, True

def run_incremental_learning():
    """Run the incremental learning pipeline"""
    print("=== [🔄] Incremental Learning Pipeline Starting ===")
    print(f"📊 Initial memory: {monitor_memory_usage():.1f}MB")
    
    try:
        # 1. Load current model and scaler (including compatible settings)
        model_path, scaler_path = load_model_and_scaler()
        model, model_args = load_model_from_checkpoint(model_path, return_args=True)
        scaler = joblib.load(scaler_path)
        
        # 2. Load recent data
        new_data = load_incremental_data()
        if new_data.empty:
            print("⚠️ No new data available.")
            return {"message": "No new data available."}, 200
        
        print(f"📊 Incremental learning data: {len(new_data):,} samples (last {INCREMENTAL_DAYS} days)")
        
        # Memory check
        if check_memory_limit():
            print("⚠️ Incremental learning aborted due to insufficient memory.")
            return {"message": "Incremental learning aborted: insufficient memory."}, 200
        
        # 3. Update model
        window_size = model_args.get("window_size", 64)
        
        print(f"📊 Model settings:")
        print(f"  - window_size: {window_size}")
        print(f"  - threshold: {model_args.get('classification_threshold', 0.5)}")
        
        updated_model, updated_scaler, success = incremental_update_with_validation(
            model, new_data, scaler, window_size
        )
        
        if not success:
            print("❌ Incremental learning failed: performance validation failed.")
            safe_memory_cleanup()
            return {"message": "Incremental learning failed: performance validation failed."}, 200
        
        # 4. Save updated model
        today_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        updated_model_path = f"/tmp/incremental_model_{today_str}.pt"
        updated_scaler_path = f"/tmp/incremental_scaler_{today_str}.pkl"
        
        # Save model (preserve existing model_args)
        incremental_model_args = model_args.copy()
        incremental_model_args['training_type'] = 'incremental'
        incremental_model_args['incremental_timestamp'] = datetime.utcnow().isoformat()
        
        torch.save({
            'model_class': 'PatchTST',
            'model_args': incremental_model_args,
            'state_dict': updated_model.state_dict()
        }, updated_model_path)
        
        joblib.dump(updated_scaler, updated_scaler_path)
        
        # 5. Upload to GCS
        model_name = f"patchtst_incremental_{today_str}.pt"
        scaler_name = f"scaler_incremental_{today_str}.pkl"
        
        try:
            upload_to_gcs(updated_model_path, GCS_BUCKET_NAME, f"{GCS_MODEL_DIR}/{model_name}")
            upload_to_gcs(updated_scaler_path, GCS_BUCKET_NAME, f"{GCS_MODEL_DIR}/{scaler_name}")
            
            print(f"✅ Incremental learning completed: {model_name}")
            print(f"📊 Final memory: {monitor_memory_usage():.1f}MB")
            
            # Memory cleanup
            safe_memory_cleanup()
            
            return {
                "message": f"Incremental learning completed: {model_name}",
                "model_name": model_name
            }, 200
        except Exception as upload_error:
            print(f"❌ GCS upload failed: {upload_error}")
            return {"error": f"GCS upload failed: {upload_error}"}, 500
        
    except Exception as e:
        print(f"[💥] Incremental learning failed: {e}")
        # Memory cleanup on error
        safe_memory_cleanup()
        return {"error": str(e)}, 500

if __name__ == "__main__":
    run_incremental_learning() 