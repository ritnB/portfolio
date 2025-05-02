# main.py (Public-safe Flask API Entry Point with all numeric values masked)

from flask import Flask, request
import warnings
import pandas as pd
import time
import joblib
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Internal project modules
from config import FIXED_PARAMS, SCALER_PATH
from data.data_loader import load_technical_indicators, load_sentiment_indicators
from data.preprocess import apply_coin_mapping, fill_nan
from inference.sentiment_inference import run_sentiment_pipeline
from inference.timeseries_inference import run_timeseries_pipeline
from debug_tools.debug_tools import print_technical_matrix, print_remaining_coins

from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["POST"])
def run_pipeline():
    start_time = time.time()
    warnings.filterwarnings("ignore")

    try:
        print("==== Main Pipeline Started ====", flush=True)

        # Step 1: Run sentiment inference pipeline
        run_sentiment_pipeline()

        # Step 2: Load data
        recent_days = int(1e8)
        ti_df = load_technical_indicators(recent_days)
        sentiment_df = load_sentiment_indicators(recent_days)

        # Step 3: Preprocess technical indicator data
        ti_df = apply_coin_mapping(ti_df, coin_column="coin")
        ti_df["timestamp"] = pd.to_datetime(ti_df["timestamp"])
        ti_df["merge_date"] = ti_df["timestamp"].dt.floor("D")
        sentiment_df["timestamp"] = pd.to_datetime(sentiment_df["timestamp"]).dt.floor("D")
        sentiment_df["sentiment_date"] = sentiment_df["timestamp"]

        # Step 4: Merge datasets
        merged_df = pd.merge(
            ti_df,
            sentiment_df[["sentiment_date", "coin", "sentiment_score"]],
            left_on=["merge_date", "coin"],
            right_on=["sentiment_date", "coin"],
            how="inner"
        ).drop(columns=["merge_date", "sentiment_date"])

        # Step 5: Fill missing values
        merged_df = fill_nan(merged_df)

        # Step 6: Scale technical indicators (example feature names)
        tech_features = [
            "sma", "ema", "macd", "macd_signal", "macd_diff",
            "rsi", "stochastic", "mfi", "cci"
        ]
        scaler_tech = joblib.load(SCALER_PATH)
        merged_df[tech_features] = scaler_tech.transform(merged_df[tech_features])

        # Step 7: Normalize sentiment score
        scaler_sent = StandardScaler()
        merged_df["sentiment_score"] = scaler_sent.fit_transform(merged_df[["sentiment_score"]])
        merged_df["sentiment_score"] *= 1e8

        # Step 8: Run time-series pipeline
        run_timeseries_pipeline(merged_df)

        print("==== Main Pipeline Finished ====", flush=True)

    finally:
        elapsed = time.time() - start_time
        print(f"⏱️ Total execution time: {elapsed:.2f} seconds", flush=True)

    return {"status": "pipeline finished"}, 200
