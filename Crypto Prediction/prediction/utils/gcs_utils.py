import os
from datetime import datetime
from typing import Optional

from google.cloud import storage
from google.oauth2 import service_account


def get_storage_client():
    """
    로컬 실행 시 GOOGLE_APPLICATION_CREDENTIALS가 설정돼 있으면
    해당 JSON 키 파일을 사용해서 인증된 storage.Client 반환.
    GCP Cloud Run 등에서는 기본 인증 방식 사용.
    """
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if cred_path and os.path.isfile(cred_path):
        credentials = service_account.Credentials.from_service_account_file(cred_path)
        return storage.Client(credentials=credentials)
    else:
        return storage.Client()


def upload_to_gcs(local_path: str, bucket_name: str, gcs_path: str):
    """
    로컬 파일을 GCS에 업로드합니다.
    """
    try:
        client = get_storage_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        print(f"✅ GCS 업로드 성공: gs://{bucket_name}/{gcs_path}")
    except Exception as e:
        print(f"[❌] GCS 업로드 실패: {e}")


def download_from_gcs(bucket_name: str, gcs_path: str, local_path: str):
    """
    GCS에서 로컬 경로로 파일을 다운로드합니다.
    """
    try:
        client = get_storage_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.download_to_filename(local_path)
        print(f"📥 GCS 다운로드 성공: gs://{bucket_name}/{gcs_path} → {local_path}")
    except Exception as e:
        print(f"[❌] GCS 다운로드 실패: {e}")


def get_latest_model_filename(bucket_name: str, prefix: str = "models/patchtst_final_model_") -> Optional[str]:
    """
    GCS 버킷에서 가장 최신의 PatchTST 모델 파일명을 반환합니다.
    prefix는 "models/patchtst_final_model_" 과 같이 지정합니다.

    Returns:
        최신 파일명 (예: models/patchtst_final_model_20250721.pt)
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
                date_str = name.replace("patchtst_final_model_", "").replace(".pt", "")
                date_obj = datetime.strptime(date_str, "%Y%m%d")
                if latest_date is None or date_obj > latest_date:
                    latest_date = date_obj
                    latest_blob_name = blob.name
            except Exception:
                continue

        return latest_blob_name
    except Exception as e:
        print(f"[❌] GCS 모델 목록 조회 실패: {e}")
        return None
