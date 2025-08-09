import os
from datetime import datetime
from typing import Optional

from google.cloud import storage
from google.oauth2 import service_account


def get_storage_client():
    """Create a storage client.

    If GOOGLE_APPLICATION_CREDENTIALS is set locally, use that JSON key.
    Otherwise rely on default credentials (e.g., Cloud Run).
    """
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if cred_path and os.path.isfile(cred_path):
        credentials = service_account.Credentials.from_service_account_file(cred_path)
        return storage.Client(credentials=credentials)
    else:
        return storage.Client()


def upload_to_gcs(local_path: str, bucket_name: str, gcs_path: str):
    """Upload a local file to GCS."""
    try:
        client = get_storage_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        print(f"✅ GCS upload success: gs://{bucket_name}/{gcs_path}")
    except Exception as e:
        print(f"[❌] GCS upload failed: {e}")


def download_from_gcs(bucket_name: str, gcs_path: str, local_path: str):
    """Download a file from GCS to a local path."""
    try:
        client = get_storage_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.download_to_filename(local_path)
        print(f"📥 GCS download success: gs://{bucket_name}/{gcs_path} → {local_path}")
    except Exception as e:
        print(f"[❌] GCS download failed: {e}")


def get_latest_model_filename(bucket_name: str, prefix: str = "models/patchseq_final_model_") -> Optional[str]:
    """Return the latest PatchTST model blob name in GCS for a given prefix."""
    try:
        client = get_storage_client()
        blobs = client.list_blobs(bucket_name, prefix=prefix)

        latest_date = None
        latest_blob_name = None

        for blob in blobs:
            name = os.path.basename(blob.name)
            try:
                # Support various model types (final, incremental, etc.)
                if "patchseq_final_model_" in name:
                    date_str = name.replace("patchseq_final_model_", "").replace(".pt", "")
                elif "patchseq_incremental_" in name:
                    date_str = name.replace("patchseq_incremental_", "").replace(".pt", "")
                else:
                    # Fallback: extract YYYYMMDD via regex
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
        print(f"[❌] Failed to list models in GCS: {e}")
        return None


def get_latest_scaler_filename(bucket_name: str, prefix: str = "models/scaler_standard_") -> Optional[str]:
    """Return the latest scaler blob name in GCS for a given prefix."""
    try:
        client = get_storage_client()
        blobs = client.list_blobs(bucket_name, prefix=prefix)

        latest_date = None
        latest_blob_name = None

        for blob in blobs:
            name = os.path.basename(blob.name)
            try:
                # scaler_standard_YYYYMMDD.pkl pattern
                if "scaler_standard_" in name:
                    date_str = name.replace("scaler_standard_", "").replace(".pkl", "")
                else:
                    # Fallback: extract YYYYMMDD via regex
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
        print(f"[❌] Failed to list scalers in GCS: {e}")
        return None


def download_latest_model_and_scaler(bucket_name: str, model_local_path: str = "/tmp/latest_model.pt", 
                                   scaler_local_path: str = "/tmp/latest_scaler.pkl") -> tuple[bool, bool]:
    """Download latest model and scaler from GCS and verify v11_2 compatibility."""
    model_success = False
    scaler_success = False
    
    # Download latest model
    latest_model = get_latest_model_filename(bucket_name)
    if latest_model:
        download_from_gcs(bucket_name, latest_model, model_local_path)
        # Verify v11_2 compatibility
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
    """Verify compatibility of a saved model with v11_2 format.

    Expected format:
    {
        'model_class': 'PatchSequenceModel',
        'model_args': {...},
        'state_dict': {...}
    }
    """
    try:
        import torch
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Required keys
        required_keys = ['model_class', 'model_args', 'state_dict']
        for key in required_keys:
            if key not in checkpoint:
                print(f"❌ Model compatibility check failed: missing key '{key}'")
                return False
        
        # Model class check (generic)
        if checkpoint['model_class'] != 'PatchSequenceModel':
            print(f"❌ Model class mismatch: {checkpoint['model_class']}")
            return False
        
        # v11_2 required model_args
        model_args = checkpoint['model_args']
        v11_2_required_args = [
            'input_size', 'd_model', 'num_layers', 'num_heads', 'patch_size', 
            'window_size', 'stride', 'num_classes', 'dropout', 'pooling_type', 
            'mlp_hidden_mult', 'activation', 'classification_threshold'
        ]
        
        missing_args = [arg for arg in v11_2_required_args if arg not in model_args]
        if missing_args:
            print(f"❌ Missing model_args: {missing_args}")
            return False
        
        print("✅ Model compatibility verified (v11_2)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during model compatibility check: {e}")
        return False


def verify_scaler_compatibility(scaler_path: str) -> bool:
    """Verify scaler type and training state."""
    try:
        import joblib
        from sklearn.preprocessing import StandardScaler
        
        scaler = joblib.load(scaler_path)
        
        # Type check
        if not isinstance(scaler, StandardScaler):
            print(f"❌ Scaler type mismatch: {type(scaler)}")
            return False
        
        # Fitted state check
        if not hasattr(scaler, 'mean_') or not hasattr(scaler, 'scale_'):
            print(f"❌ Scaler is not fitted")
            return False
        
        feature_count = len(scaler.mean_)
        print(f"✅ Scaler compatibility verified")
        print(f"  - feature count: {feature_count}")
        print(f"  - mean range: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
        print(f"  - scale range: [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during scaler compatibility check: {e}")
        return False
