# utils/price_utils.py

import requests
import time
from typing import List, Dict, Optional
from datetime import datetime

from config import (
    EXTERNAL_API_BASE_URL, 
    EXTERNAL_API_PRICE_ENDPOINT, 
    EXTERNAL_API_REQUEST_TIMEOUT,
    ASSET_INFO,
    PRICE_USD_COLUMN,
    PRICE_LOCAL_COLUMN
)


def fetch_batch_prices(coin_names: List[str], retries: int = 3) -> Dict:
    """
    Batch query current prices for predicted coins.
    
    Args:
        coin_names: List of normalized coin names (e.g., ['Ethereum', 'Bitcoin'])
        retries: Number of retries if API call fails
    
    Returns:
        Dictionary of price information
        {
            'ethereum': {'usd': 3443.66, 'krw': 4783351},
            'bitcoin': {'usd': 113545, 'krw': 157718029}
        }
    """
    print(f"🔍 Coins to query prices for: {coin_names}")
    
    # Map coin names to CoinGecko IDs
    coingecko_ids = []
    mapping_info = {}
    
    for coin_name in coin_names:
        # Find coingecko_id directly from COIN_INFO
        coin_info = None
        for key, info in ASSET_INFO.items():
            if info['name'] == coin_name:
                coin_info = info
                break
        
        if coin_info:
            coingecko_id = coin_info['coingecko_id']
            coingecko_ids.append(coingecko_id)
            mapping_info[coingecko_id] = coin_name
        else:
            print(f"⚠️ No CoinGecko mapping: {coin_name}")
    
    if not coingecko_ids:
        print("❌ No coins available for query.")
        return {}
    
    print(f"📡 Calling CoinGecko API: {coingecko_ids}")
    
    # API call
    for attempt in range(retries):
        try:
            # Build API URL
            ids_str = ','.join(coingecko_ids)
            url = f"{EXTERNAL_API_BASE_URL}{EXTERNAL_API_PRICE_ENDPOINT}"
            params = {
                'ids': ids_str,
                'vs_currencies': 'usd,krw'
            }
            
            # Call API
            response = requests.get(
                url, 
                params=params, 
                timeout=EXTERNAL_API_REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            price_data = response.json()
            print(f"✅ Price query successful: {len(price_data)} coins")
            
            # Response log (sample)
            if price_data:
                sample_coin = list(price_data.keys())[0]
                sample_price = price_data[sample_coin]
                print(f"   Example - {sample_coin}: ${sample_price.get('usd', 'N/A')}")
            
            return price_data
            
        except requests.exceptions.RequestException as e:
            print(f"❌ API call failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"⏳ Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("❌ All retries failed")
                return {}
        
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return {}


def get_price_for_coin(coin_name: str, price_data: Dict) -> Optional[Dict]:
    """
    Extract price information for a specific coin.
    
    Args:
        coin_name: Normalized coin name
        price_data: Data returned by fetch_batch_prices
    
    Returns:
        {'usd': float, 'krw': float} or None
    """
    # Find coingecko_id from COIN_INFO
    coingecko_id = None
    for info in ASSET_INFO.values():
        if info['name'] == coin_name:
            coingecko_id = info['coingecko_id']
            break
    
    if not coingecko_id:
        return None
    
    return price_data.get(coingecko_id)


def format_price_update_data(coin_name: str, price_data: Dict) -> Optional[Dict]:
    """
    Format price data for Supabase update.
    
    Args:
        coin_name: Coin name
        price_data: Price data from CoinGecko API response
    
    Returns:
        Dictionary for Supabase update
    """
    coin_price = get_price_for_coin(coin_name, price_data)
    if not coin_price:
        return None
    
    return {
        PRICE_USD_COLUMN: coin_price.get('usd'),
        PRICE_LOCAL_COLUMN: coin_price.get('krw')
    }


def test_coingecko_api():
    """Test CoinGecko API connection"""
    print("🧪 Testing CoinGecko API connection...")
    
    # Get actual coin names from COIN_INFO
    test_coins = ['Ethereum', 'Litecoin', 'Cardano']
    result = fetch_batch_prices(test_coins)
    
    if result:
        print("✅ API connection successful!")
        for coin in test_coins:
            price = get_price_for_coin(coin, result)
            if price:
                print(f"   {coin}: ${price.get('usd', 'N/A')}")
            else:
                print(f"   {coin}: Price information not available")
        
        print(f"\n📊 Integrated structure info:")
        print(f"   Number of supported coins: {len(ASSET_INFO)}")
        print(f"   Price columns: {PRICE_USD_COLUMN}, {PRICE_LOCAL_COLUMN}")
    else:
        print("❌ API connection failed")
    
    return result


if __name__ == "__main__":
    # Run test
    test_coingecko_api()