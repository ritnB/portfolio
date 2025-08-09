from datetime import datetime, timedelta
import pandas as pd


def get_recent_date_range(recent_days: int = 7):
    """Return a list of recent dates in ascending order."""
    return [datetime.utcnow().date() - timedelta(days=i) for i in range(recent_days)][::-1]


def print_technical_matrix(tech_df: pd.DataFrame, recent_days: int = 7):
    """Print a presence matrix of technical indicators by date and coin."""
    print("\n========== [📊 Technical Indicator Presence Matrix] ==========")
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


def print_remaining_coins(merged_df: pd.DataFrame):
    print("\n========== [🔍 Remaining Coins] ==========")
    remaining_coins = sorted(merged_df["coin"].unique())
    print(f"Count: {len(remaining_coins)}")
    print(f"Coins: {remaining_coins}")
    print("=========================================")
