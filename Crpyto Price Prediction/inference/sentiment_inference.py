# inference/sentiment_inference.py (Public-safe version)

import pandas as pd
import torch
import torch.nn.functional as F
import gc
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data.data_loader import load_comments_data
from data.supabase_io import save_prediction
from config import SENTIMENT_MODEL_PATH, SENTIMENT_TOKENIZER_PATH, FIXED_PARAMS

MAX_LENGTH = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_sentiment_inference(model_path: str, tokenizer_path: str, text: str):
    """
    Perform sentiment inference on a single text string.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).squeeze().tolist()

    # Simplified sentiment score (custom logic hidden)
    sentiment_score = probs[2]  # Use positive class probability directly
    return sentiment_score, probs

def run_sentiment_pipeline():
    """
    Load recent comments, run sentiment analysis, and store the aggregated results.
    """
    print("=== Sentiment Inference Pipeline Started ===")

    comments_df = load_comments_data(recent_days=int(1e8))
    if comments_df.empty:
        print("No comments data found.")
        return

    comments_df["timestamp"] = pd.to_datetime(comments_df["timestamp"])
    comments_df["date"] = comments_df["timestamp"].dt.date

    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_TOKENIZER_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH).to(device)
    model.eval()

    results = []
    for idx, row in comments_df.iterrows():
        text = row.get("text") or row.get("comment", "")
        coin = row["coin"]
        timestamp = row["date"]

        try:
            if not isinstance(text, str) or len(text.strip()) == 0:
                continue
            if len(text) > 1000:
                continue

            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1).squeeze().tolist()

            # Simplified scoring: only positive class probability
            sentiment_score = probs[2]
            results.append({
                "timestamp": timestamp,
                "coin": coin,
                "sentiment_score": sentiment_score
            })

        except Exception as e:
            print(f"Error processing comment {idx+1}: {e}")
            continue

        if idx % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    if not results:
        print("No sentiment scores generated.")
        return

    df = pd.DataFrame(results)
    grouped = df.groupby(["timestamp", "coin"])["sentiment_score"].mean().reset_index()

    for i, row in grouped.iterrows():
        try:
            save_prediction({
                "timestamp": str(row["timestamp"]),
                "coin": row["coin"],
                "sentiment_score": float(row["sentiment_score"])
            }, table_name="your_sentiment_table")
        except Exception as e:
            print(f"Supabase save error: {e}")

    print("=== Sentiment Pipeline Finished ===")
