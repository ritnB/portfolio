import os
from datetime import datetime
from typing import Optional

from google.cloud import storage
from google.oauth2 import service_account


def get_storage_client():
    """
    Returns authenticated storage.Client using JSON key file if GOOGLE_APPLICATION_CREDENTIALS 
    is set for local execution. Uses default authentication for GCP Cloud Run.
    """
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if cred_path and os.path.isfile(cred_path):
        credentials = service_account.Credentials.from_service_account_file(cred_path)
        return storage.Client(credentials=credentials)
    else:
        return storage.Client()


def upload_to_gcs(local_path: str, bucket_name: str, gcs_path: str):
    """
    Upload local file to Google Cloud Storage.
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
    Download file from Google Cloud Storage to local path.
    """
    try:
        client = get_storage_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.download_to_filename(local_path)
        print(f"📥 GCS download successful: gs://{bucket_name}/{gcs_path} → {local_path}")
    except Exception as e:
        print(f"[❌] GCS download failed: {e}")


def get_latest_model_filename(bucket_name: str, prefix: str = "models/model_") -> Optional[str]:
    """
    Returns the filename of the latest model in the GCS bucket.
    Prefix should be specified like "models/model_".

    Returns:
        Latest filename (e.g. models/model_20250721.pt)
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
                # Extract date from filename pattern: model_YYYYMMDD.pt
                date_str = name.replace("model_", "").replace(".pt", "")
                date_obj = datetime.strptime(date_str, "%Y%m%d")
                if latest_date is None or date_obj > latest_date:
                    latest_date = date_obj
                    latest_blob_name = blob.name
            except Exception:
                continue

        return latest_blob_name
    except Exception as e:
        print(f"[❌] Failed to retrieve GCS model list: {e}")
        return None
