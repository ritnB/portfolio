from flask import Flask, jsonify, request
from pipelines.pipeline_timeseries import main as run_timeseries
from pipelines.pipeline_verify import run_verification
from pipelines.pipeline_retrain import run_retraining_pipeline

app = Flask(__name__)

@app.route("/timeseries", methods=["POST"])
def trigger_timeseries():
    run_timeseries()
    return jsonify({"message": "✅ Timeseries pipeline triggered"}), 200

@app.route("/verify", methods=["POST"])
def trigger_verification():
    result, status = run_verification()
    return jsonify(result), status

@app.route("/retrain", methods=["POST"])
def trigger_retraining():
    triggered = run_retraining_pipeline()
    if triggered:
        return jsonify({"message": "🔁 Retraining triggered"}), 200
    else:
        return jsonify({"message": "⏸ No retraining needed"}), 204

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"message": "🟢 Service is running"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
