# debug_tools.py (Public-safe version)

from datetime import datetime, timedelta
import pandas as pd

def get_recent_date_range(recent_days=7):
    return [datetime.utcnow().date() - timedelta(days=i) for i in range(recent_days)][::-1]

def print_sentiment_matrix(comments_df, results, grouped, recent_days=7):
    print("\n========== [📊 Coin × Date Sentiment Matrix] ==========")
    date_range = get_recent_date_range(recent_days)
    coins = sorted(set(comments_df["coin"].unique()).union(
                   [r["coin"] for r in results]).union(
                   grouped["coin"].unique()))

    for coin in coins:
        line = f"{coin:<14s} | "
        for d in date_range:
            has_comment = not comments_df[(comments_df["coin"] == coin) & (comments_df["date"] == d)].empty
            has_result = any(r["coin"] == coin and r["timestamp"] == d for r in results)
            has_grouped = not grouped[(grouped["coin"] == coin) & (grouped["timestamp"] == d)].empty

            if has_grouped:
                mark = "✅"
            elif has_result:
                mark = "🤖"
            elif has_comment:
                mark = "📨"
            else:
                mark = "❌"
            line += f"{mark} "
        print(line)

    print("Legend: ✅ saved | 🤖 scored | 📨 comment only | ❌ nothing")

def print_technical_matrix(tech_df, sentiment_grouped, recent_days=7):
    print("\n========== [📊 Coin × Date Technical/Sentiment Matrix] ==========")
    date_range = get_recent_date_range(recent_days)
    coins = sorted(set(tech_df["coin"].unique()).union(sentiment_grouped["coin"].unique()))

    sentiment_grouped["timestamp"] = pd.to_datetime(sentiment_grouped["timestamp"]).dt.date

    for coin in coins:
        line = f"{coin:<20s} | "
        for d in date_range:
            has_tech = not tech_df[(tech_df["coin"] == coin) & (tech_df["timestamp"].dt.floor("D") == pd.Timestamp(d))].empty
            has_sent = not sentiment_grouped[(sentiment_grouped["coin"] == coin) & (sentiment_grouped["timestamp"] == d)].empty

            if has_tech and has_sent:
                mark = "✅"
            elif has_tech:
                mark = "📉"
            elif has_sent:
                mark = "🧠"
            else:
                mark = "❌"
            line += f"{mark} "
        print(line)

    print("Legend: ✅ both | 📉 only TI | 🧠 only Sentiment | ❌ missing both")

def print_remaining_coins(merged_df):
    print("\n========== [🔍 Remaining Coins Diagnostic] ==========")
    remaining_coins = sorted(merged_df["coin"].unique())
    print(f"Remaining coins after NaN handling: {remaining_coins}")
    print(f"Common coins retained: {remaining_coins}")
    print("====================================================")
