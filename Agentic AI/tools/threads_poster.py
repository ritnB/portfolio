# tools/threads_poster.py
import requests
from config import SOCIAL_MEDIA_API_KEY, SOCIAL_MEDIA_API_URL, AUTO_POSTING_ENABLED

def upload_to_threads(content: str) -> str:
    """Upload content to social media platform"""
    if not AUTO_POSTING_ENABLED:
        return "[Upload Disabled] Generated content:\n" + content

    try:
        response = requests.post(
            SOCIAL_MEDIA_API_URL,
            headers={"Authorization": f"Bearer {SOCIAL_MEDIA_API_KEY}"},
            json={"content": content}
        )

        if response.status_code == 200:
            return "✅ Social media upload successful"
        else:
            return f"❌ Upload failed: {response.status_code} - {response.text}"

    except Exception as e:
        return f"❌ Exception occurred during upload: {str(e)}"
