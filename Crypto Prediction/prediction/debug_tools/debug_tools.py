# debug/debug_tools.py

from datetime import datetime, timedelta
import pandas as pd


def get_recent_date_range(recent_days=7):
    return [datetime.utcnow().date() - timedelta(days=i) for i in range(recent_days)][::-1]


def print_technical_matrix(tech_df, recent_days=7):
    """
    기술지표 데이터 존재 여부를 날짜별로 출력
    """
    print("\n========== [📊 기술지표 존재 매트릭스] ==========")
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
    print("\n========== [🔍 남은 코인 진단] ==========")
    remaining_coins = sorted(merged_df["coin"].unique())
    print(f"남은 코인 수: {len(remaining_coins)}")
    print(f"코인 목록: {remaining_coins}")
    print("===========================================")
