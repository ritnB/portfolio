# tools/threads_poster.py
import requests
from config import THREADS_API_KEY, THREADS_API_URL, THREADS_UPLOAD_ENABLED

def upload_to_threads(content: str) -> str:
    if not THREADS_UPLOAD_ENABLED:
        return "[업로드 비활성화됨] 생성된 글:\n" + content

    try:
        response = requests.post(
            THREADS_API_URL,
            headers={"Authorization": f"Bearer {THREADS_API_KEY}"},
            json={"content": content}
        )

        if response.status_code == 200:
            return "✅ Threads 업로드 성공"
        else:
            return f"❌ 업로드 실패: {response.status_code} - {response.text}"

    except Exception as e:
        return f"❌ 업로드 중 예외 발생: {str(e)}"
