# debug/debug_tools.py

from data.supabase_io import load_technical_indicators
from config import normalize_coin_name
import pandas as pd
from collections import Counter

def analyze_data_coverage():
    """
    Analyze technical indicator data coverage by date.
    
    Note: This is a simplified analysis tool for portfolio demonstration.
    """
    df = load_technical_indicators()
    
    if df.empty:
        print("No data found for analysis")
        return
    
    print("\n========== [📊 Data Coverage Analysis] ==========")
    
    # Group by date and count records
    df['date'] = df['timestamp'].dt.date
    daily_counts = df.groupby('date').size()
    
    print(f"📅 Date range: {daily_counts.index.min()} to {daily_counts.index.max()}")
    print(f"📊 Total records: {len(df)}")
    print(f"🗓️ Days with data: {len(daily_counts)}")
    print(f"📈 Average records per day: {daily_counts.mean():.1f}")
    
    # Show daily distribution
    print("\n📋 Records per day (last 10 days):")
    for date, count in daily_counts.tail(10).items():
        print(f"  {date}: {count} records")

def analyze_coin_distribution():
    """
    Analyze the distribution of coins in the dataset.
    
    Note: This provides basic statistics for portfolio demonstration.
    """
    df = load_technical_indicators()
    
    if df.empty:
        print("No data found for analysis")
        return
    
    print("\n========== [🪙 Coin Analysis] ==========")
    
    # Apply coin normalization
    df['normalized_coin'] = df['coin'].apply(normalize_coin_name)
    
    # Count by normalized coin names
    coin_counts = df['normalized_coin'].value_counts()
    
    print(f"📊 Total unique coins: {len(coin_counts)}")
    print(f"📈 Total records: {len(df)}")
    
    print("\n🏆 Top 10 coins by record count:")
    for coin, count in coin_counts.head(10).items():
        print(f"  {coin}: {count} records")
    
    # Check for coins with low data
    low_data_coins = coin_counts[coin_counts < 100]
    if not low_data_coins.empty:
        print(f"\n⚠️ Coins with < 100 records: {len(low_data_coins)}")
        print(f"   Examples: {list(low_data_coins.head(5).index)}")

def check_data_quality():
    """
    Perform basic data quality checks.
    
    Note: This is a simplified quality check for portfolio demonstration.
    """
    df = load_technical_indicators()
    
    if df.empty:
        print("No data found for analysis")
        return
    
    print("\n========== [🔍 Data Quality Check] ==========")
    
    # Check for missing values
    missing_data = df.isnull().sum()
    print("📋 Missing values by column:")
    for col, missing_count in missing_data.items():
        if missing_count > 0:
            percentage = (missing_count / len(df)) * 100
            print(f"  {col}: {missing_count} ({percentage:.1f}%)")
    
    # Check for duplicate records
    duplicates = df.duplicated(['coin', 'timestamp']).sum()
    print(f"\n🔄 Duplicate records: {duplicates}")
    
    # Check data freshness
    latest_timestamp = df['timestamp'].max()
    print(f"\n⏰ Latest data: {latest_timestamp}")
    
    return {
        "total_records": len(df),
        "missing_values": missing_data.to_dict(),
        "duplicates": duplicates,
        "latest_timestamp": latest_timestamp
    }

def generate_summary_report():
    """
    Generate a comprehensive summary report of the dataset.
    """
    print("========== [📋 Dataset Summary Report] ==========")
    
    # Run all analyses
    analyze_data_coverage()
    analyze_coin_distribution() 
    quality_stats = check_data_quality()
    
    print("\n========== [✅ Analysis Complete] ==========")
    return quality_stats

if __name__ == "__main__":
    generate_summary_report()
