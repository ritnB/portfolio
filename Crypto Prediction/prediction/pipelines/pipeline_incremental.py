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
from data.preprocess import CoinTimeSeriesDataset
from utils.gcs_utils import upload_to_gcs, download_from_gcs, get_latest_model_filename
from config import (GCS_BUCKET_NAME, GCS_MODEL_DIR, FEATURE_COLS,
                   INCREMENTAL_DAYS, INCREMENTAL_EPOCHS, INCREMENTAL_LR, INCREMENTAL_BATCH_SIZE,
                   INCREMENTAL_PERFORMANCE_THRESHOLD, INCREMENTAL_VALIDATION_RATIO, INCREMENTAL_MAX_DEGRADATION)

def load_model_and_scaler():
    """Load latest model and scaler from GCS"""
    model_blob = get_latest_model_filename(GCS_BUCKET_NAME, prefix=f"{GCS_MODEL_DIR}/patchtst_final_model_")
    if not model_blob:
        raise Exception("Could not find model in GCS.")
    
    # Download to temporary files
    model_path = "/tmp/current_model.pt"
    scaler_path = "/tmp/current_scaler.pkl"
    
    download_from_gcs(GCS_BUCKET_NAME, model_blob, model_path)
    
    # Estimate scaler filename
    scaler_blob = model_blob.replace("patchtst_final_model_", "scaler_standard_").replace(".pt", ".pkl")
    download_from_gcs(GCS_BUCKET_NAME, scaler_blob, scaler_path)
    
    return model_path, scaler_path

def load_incremental_data():
    """Load recent few days data (for simple incremental learning)"""
    # Use existing function but temporarily adjust RECENT_DAYS
    from config import RECENT_DAYS
    
    original_recent_days = RECENT_DAYS
    try:
        # Load only recent 7 days data
        import config
        config.RECENT_DAYS = INCREMENTAL_DAYS
        
        # Use existing function
        df = load_technical_indicators()
        
        # Additional preprocessing
        df = df[df['price_trend'].isin(['up', 'down'])].copy()
        df['label'] = df['price_trend'].map({'down': 0, 'up': 1})
        
        return df
    finally:
        # Restore original value
        config.RECENT_DAYS = original_recent_days

def evaluate_model_performance(model, test_data, scaler, window_size):
    """Evaluate model performance"""
    if test_data.empty:
        return 0.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Scale test data
    test_data_scaled = test_data.copy()
    test_data_scaled[FEATURE_COLS] = scaler.transform(test_data[FEATURE_COLS])
    
    # Create dataset
    dataset = CoinTimeSeriesDataset(test_data_scaled, window_size, FEATURE_COLS)
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
    """Incremental model update including validation"""
    if new_data.empty:
        return model, scaler, False
    
    # Data splitting (train/validation)
    split_idx = int(len(new_data) * (1 - INCREMENTAL_VALIDATION_RATIO))
    train_data = new_data.iloc[:split_idx]
    val_data = new_data.iloc[split_idx:]
    
    if train_data.empty or val_data.empty:
        return model, scaler, False
    
    # Measure original model performance
    original_performance = evaluate_model_performance(model, val_data, scaler, window_size)
    print(f"Original model performance: {original_performance:.4f}")
    
    # Device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Scale new data (independent scaler)
    train_scaler = joblib.load("/tmp/current_scaler.pkl")  # Copy existing scaler
    train_data_scaled = train_data.copy()
    train_data_scaled[FEATURE_COLS] = train_scaler.transform(train_data[FEATURE_COLS])
    
    # Create dataset
    dataset = CoinTimeSeriesDataset(train_data_scaled, window_size, FEATURE_COLS)
    if len(dataset) == 0:
        return model, scaler, False
    
    # Incremental training
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
        
        print(f"Incremental epoch {epoch+1}/{INCREMENTAL_EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")
    
    # Measure updated model performance
    updated_performance = evaluate_model_performance(model, val_data, scaler, window_size)
    print(f"Updated model performance: {updated_performance:.4f}")
    
    # Performance verification
    performance_improvement = updated_performance - original_performance
    print(f"Performance change: {performance_improvement:+.4f}")
    
    # Performance threshold check
    if updated_performance < INCREMENTAL_PERFORMANCE_THRESHOLD:
        print(f"⚠️ Performance below threshold ({INCREMENTAL_PERFORMANCE_THRESHOLD}): {updated_performance:.4f}")
        return model, scaler, False
    
    # Performance degradation check
    if performance_improvement < -INCREMENTAL_MAX_DEGRADATION:
        print(f"⚠️ Performance degradation too large: {performance_improvement:.4f}")
        return model, scaler, False
    
    print(f"✅ Incremental learning successful: Performance {original_performance:.4f} → {updated_performance:.4f}")
    return model, scaler, True

def run_incremental_learning():
    """Run incremental learning pipeline"""
    print("=== [🔄] Incremental Learning Pipeline Starting ===")
    
    try:
        # 1. Load current model and scaler
        model_path, scaler_path = load_model_and_scaler()
        model = load_model_from_checkpoint(model_path)
        scaler = joblib.load(scaler_path)
        
        # 2. Load recent data
        new_data = load_incremental_data()
        if new_data.empty:
            print("No new data available.")
            return {"message": "No new data available."}, 200
        
        print(f"Incremental learning data: {len(new_data)} samples (last {INCREMENTAL_DAYS} days)")
        
        # 3. Update model (including validation)
        checkpoint = torch.load(model_path, map_location='cpu')
        window_size = checkpoint["model_args"]["window_size"]
        
        updated_model, updated_scaler, success = incremental_update_with_validation(
            model, new_data, scaler, window_size
        )
        
        if not success:
            print("❌ Incremental learning failed: performance validation failed.")
            return {"message": "Incremental learning failed: performance validation failed."}, 200
        
        # 4. Save updated model
        today_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        updated_model_path = f"/tmp/incremental_model_{today_str}.pt"
        updated_scaler_path = f"/tmp/incremental_scaler_{today_str}.pkl"
        
        torch.save({
            'model_class': 'PatchTST',
            'model_args': checkpoint["model_args"],
            'state_dict': updated_model.state_dict(),
            'training_type': 'incremental'
        }, updated_model_path)
        
        joblib.dump(updated_scaler, updated_scaler_path)
        
        # 5. Upload to GCS
        model_name = f"patchtst_incremental_{today_str}.pt"
        scaler_name = f"scaler_incremental_{today_str}.pkl"
        
        upload_to_gcs(updated_model_path, GCS_BUCKET_NAME, f"{GCS_MODEL_DIR}/{model_name}")
        upload_to_gcs(updated_scaler_path, GCS_BUCKET_NAME, f"{GCS_MODEL_DIR}/{scaler_name}")
        
        print(f"✅ Incremental learning completed: {model_name}")
        return {"message": f"Incremental learning completed: {model_name}"}, 200
        
    except Exception as e:
        print(f"[💥] Incremental learning failed: {e}")
        return {"error": str(e)}, 500

if __name__ == "__main__":
    run_incremental_learning() 