import os
from datetime import datetime
from typing import Optional

from google.cloud import storage
from google.oauth2 import service_account


def get_storage_client():
    """
    When running locally, if GOOGLE_APPLICATION_CREDENTIALS is set,
    use the JSON key file for authentication and return an authenticated storage.Client.
    In GCP Cloud Run etc., use default authentication method.
    """
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if cred_path and os.path.isfile(cred_path):
        credentials = service_account.Credentials.from_service_account_file(cred_path)
        return storage.Client(credentials=credentials)
    else:
        return storage.Client()


def upload_to_gcs(local_path: str, bucket_name: str, gcs_path: str):
    """
    Upload local file to GCS.
    """
    try:
        client = get_storage_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        print(f"✅ GCS upload successful: gs://{bucket_name}/{gcs_path}")
    except Exception as e:
        print(f"[❌] GCS upload failed: {e}")


def download_from_gcs(bucket_name: str, gcs_path: str, local_path: str):
    """
    Download file from GCS to local path.
    """
    try:
        client = get_storage_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.download_to_filename(local_path)
        print(f"📥 GCS download successful: gs://{bucket_name}/{gcs_path} → {local_path}")
    except Exception as e:
        print(f"[❌] GCS download failed: {e}")


def get_latest_model_filename(bucket_name: str, prefix: str = "models/patchtst_final_model_") -> Optional[str]:
    """
    Return the latest PatchTST model filename from GCS bucket.
    prefix can be "models/patchtst_final_model_" or "models/patchtst_incremental_" etc.

    Returns:
        Latest filename (e.g., models/patchtst_final_model_20250721.pt)
        None if not found
    """
    try:
        client = get_storage_client()
        blobs = client.list_blobs(bucket_name, prefix=prefix)

        latest_date = None
        latest_blob_name = None

        for blob in blobs:
            name = os.path.basename(blob.name)
            try:
                # Support various model types (final, incremental, etc.)
                if "patchtst_final_model_" in name:
                    date_str = name.replace("patchtst_final_model_", "").replace(".pt", "")
                elif "patchtst_incremental_" in name:
                    date_str = name.replace("patchtst_incremental_", "").replace(".pt", "")
                else:
                    # Try to extract date using general pattern
                    import re
                    date_match = re.search(r'(\d{8})', name)
                    if date_match:
                        date_str = date_match.group(1)
                    else:
                        continue
                
                date_obj = datetime.strptime(date_str, "%Y%m%d")
                if latest_date is None or date_obj > latest_date:
                    latest_date = date_obj
                    latest_blob_name = blob.name
            except Exception:
                continue

        return latest_blob_name
    except Exception as e:
        print(f"[❌] GCS model list retrieval failed: {e}")
        return None


def get_latest_scaler_filename(bucket_name: str, prefix: str = "models/scaler_standard_") -> Optional[str]:
    """
    Return the latest scaler filename from GCS bucket.
    
    Returns:
        최신 파일명 (예: models/scaler_standard_20250721.pkl)
        없으면 None
    """
    try:
        client = get_storage_client()
        blobs = client.list_blobs(bucket_name, prefix=prefix)

        latest_date = None
        latest_blob_name = None

        for blob in blobs:
            name = os.path.basename(blob.name)
            try:
                # scaler_standard_YYYYMMDD.pkl 패턴
                if "scaler_standard_" in name:
                    date_str = name.replace("scaler_standard_", "").replace(".pkl", "")
                else:
                    # 일반적인 패턴으로 날짜 추출 시도
                    import re
                    date_match = re.search(r'(\d{8})', name)
                    if date_match:
                        date_str = date_match.group(1)
                    else:
                        continue
                
                date_obj = datetime.strptime(date_str, "%Y%m%d")
                if latest_date is None or date_obj > latest_date:
                    latest_date = date_obj
                    latest_blob_name = blob.name
            except Exception:
                continue

        return latest_blob_name
    except Exception as e:
        print(f"[❌] GCS scaler list retrieval failed: {e}")
        return None


def download_latest_model_and_scaler(bucket_name: str, model_local_path: str = "/tmp/latest_model.pt", 
                                   scaler_local_path: str = "/tmp/latest_scaler.pkl") -> tuple[bool, bool]:
    """
    Download the latest model and scaler from GCS.
    Check for v11_2 compatibility.
    
    Returns:
        (model_success, scaler_success) tuple
    """
    model_success = False
    scaler_success = False
    
    # Download latest model
    latest_model = get_latest_model_filename(bucket_name)
    if latest_model:
        download_from_gcs(bucket_name, latest_model, model_local_path)
        # Check v11_2 compatibility
        model_success = verify_v11_2_model_compatibility(model_local_path)
    else:
        print("⚠️ Could not find latest model in GCS.")
    
    # Download latest scaler
    latest_scaler = get_latest_scaler_filename(bucket_name)
    if latest_scaler:
        download_from_gcs(bucket_name, latest_scaler, scaler_local_path)
        scaler_success = verify_scaler_compatibility(scaler_local_path)
    else:
        print("⚠️ Could not find latest scaler in GCS.")
    
    return model_success, scaler_success


def verify_v11_2_model_compatibility(model_path: str) -> bool:
    """
    Check compatibility with the model format saved by v11_2.
    
    Expected format:
    {
        'model_class': 'PatchTST',
        'model_args': {...},
        'state_dict': {...}
    }
    """
    try:
        import torch
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Check required keys
        required_keys = ['model_class', 'model_args', 'state_dict']
        for key in required_keys:
            if key not in checkpoint:
                print(f"❌ Model compatibility check failed: '{key}' key missing.")
                return False
        
        # Check model class
        if checkpoint['model_class'] != 'PatchTST':
            print(f"❌ Model class mismatch: {checkpoint['model_class']}")
            return False
        
        # Check v11_2 required model_args
        model_args = checkpoint['model_args']
        v11_2_required_args = [
            'input_size', 'd_model', 'num_layers', 'num_heads', 'patch_size', 
            'window_size', 'stride', 'num_classes', 'dropout', 'pooling_type', 
            'mlp_hidden_mult', 'activation', 'classification_threshold'
        ]
        
        missing_args = [arg for arg in v11_2_required_args if arg not in model_args]
        if missing_args:
            print(f"❌ Missing model arguments: {missing_args}")
            return False
        
        print(f"✅ v11_2 model compatibility check complete")
        print(f"  - window_size: {model_args['window_size']}")
        print(f"  - patch_size: {model_args['patch_size']}")
        print(f"  - stride: {model_args['stride']}")
        print(f"  - classification_threshold: {model_args['classification_threshold']}")
        print(f"  - loss_type: {model_args.get('loss_type', 'ce')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during model compatibility check: {e}")
        return False


def verify_scaler_compatibility(scaler_path: str) -> bool:
    """
    Check scaler file compatibility.
    """
    try:
        import joblib
        from sklearn.preprocessing import StandardScaler
        
        scaler = joblib.load(scaler_path)
        
        # Check StandardScaler type
        if not isinstance(scaler, StandardScaler):
            print(f"❌ Scaler type mismatch: {type(scaler)}")
            return False
        
        # Check trained state
        if not hasattr(scaler, 'mean_') or not hasattr(scaler, 'scale_'):
            print(f"❌ Scaler not trained.")
            return False
        
        feature_count = len(scaler.mean_)
        print(f"✅ Scaler compatibility check complete")
        print(f"  - Feature count: {feature_count}")
        print(f"  - Mean range: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
        print(f"  - Scale range: [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during scaler compatibility check: {e}")
        return False
