# debug/debug_tools.py

from datetime import datetime, timedelta
import pandas as pd


def get_recent_date_range(recent_days=7):
    return [datetime.utcnow().date() - timedelta(days=i) for i in range(recent_days)][::-1]


def print_technical_matrix(tech_df, recent_days=7):
    """
    Print technical indicator data existence by date
    """
    print("\n========== [📊 Technical Indicator Existence Matrix] ==========")
    date_range = get_recent_date_range(recent_days)
    coins = sorted(tech_df["coin"].unique())

    for coin in coins:
        line = f"{coin:<14s} | "
        for d in date_range:
            has_tech = not tech_df[(tech_df["coin"] == coin) & (tech_df["timestamp"].dt.date == d)].empty
            mark = "✅" if has_tech else "❌"
            line += f"{mark} "
        print(line)

    print("Legend: ✅ data exists | ❌ missing")


def print_remaining_coins(merged_df):
    print("\n========== [🔍 Remaining Coins Diagnosis] ==========")
    remaining_coins = sorted(merged_df["coin"].unique())
    print(f"Number of remaining coins: {len(remaining_coins)}")
    print(f"Coin list: {remaining_coins}")
    print("===========================================")
