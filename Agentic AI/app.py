import sys
import os
import locale

# ✅ Force system encoding setup (highest priority)
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform.startswith('win'):
    os.environ['PYTHONLEGACYWINDOWSSTDIO'] = '1'

# ✅ Force default encoding to UTF-8
if hasattr(sys, '_getframe'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ✅ Locale setup
try:
    locale.setlocale(locale.LC_ALL, 'C.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        pass  # Continue execution even if locale setup fails

from flask import Flask, request, Response
from agents.agent import run_agent
from loguru import logger
import json
import io

app = Flask(__name__)

@app.route("/agent", methods=["POST"])
def agent_endpoint():
    try:
        logger.info("🟢 /agent request received")
        result = run_agent()
        logger.success("✅ Content generation completed")

        # ✅ Use ensure_ascii=False to prevent Korean text corruption
        return Response(
            json.dumps(result, ensure_ascii=False),
            content_type="application/json; charset=utf-8",
            status=200
        )
    except Exception as e:
        logger.error(f"❌ Error occurred: {e}")
        return Response(
            json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False),
            content_type="application/json; charset=utf-8",
            status=500
        )

@app.route("/", methods=["GET"])
def health_check():
    return "Crypto Analysis Bot is running", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=True)