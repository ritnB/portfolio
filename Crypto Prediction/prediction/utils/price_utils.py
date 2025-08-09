import requests
import time
from typing import List, Dict, Optional
from datetime import datetime

from config import (
    COINGECKO_API_BASE_URL, 
    COINGECKO_PRICE_ENDPOINT, 
    COINGECKO_REQUEST_TIMEOUT,
    COIN_INFO,
    PRICE_USD_COLUMN,
    PRICE_KRW_COLUMN
)


def fetch_batch_prices(coin_names: List[str], retries: int = 3) -> Dict:
    """Fetch current prices for the given coins in a single API call.

    Args:
        coin_names: normalized coin names (e.g., ["Ethereum", "Bitcoin"])
        retries: retry attempts on API failure

    Returns:
        dict mapping coingecko_id to price payload
    """
    print(f"🔍 Fetching prices for: {coin_names}")
    
    # Map coin names to CoinGecko IDs
    coingecko_ids = []
    mapping_info = {}
    
    for coin_name in coin_names:
        coin_info = None
        for key, info in COIN_INFO.items():
            if info['name'] == coin_name:
                coin_info = info
                break
        
        if coin_info:
            coingecko_id = coin_info['coingecko_id']
            coingecko_ids.append(coingecko_id)
            mapping_info[coingecko_id] = coin_name
        else:
            print(f"⚠️ No CoinGecko mapping for: {coin_name}")
    
    if not coingecko_ids:
        print("❌ No mappable coins provided.")
        return {}
    
    print(f"📡 CoinGecko API call for: {coingecko_ids}")
    
    for attempt in range(retries):
        try:
            ids_str = ','.join(coingecko_ids)
            url = f"{COINGECKO_API_BASE_URL}{COINGECKO_PRICE_ENDPOINT}"
            params = {
                'ids': ids_str,
                'vs_currencies': 'usd,krw'
            }
            
            response = requests.get(
                url, 
                params=params, 
                timeout=COINGECKO_REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            price_data = response.json()
            print(f"✅ Price fetch success for {len(price_data)} coins")
            
            if price_data:
                sample_coin = list(price_data.keys())[0]
                sample_price = price_data[sample_coin]
                print(f"   Sample - {sample_coin}: ${sample_price.get('usd', 'N/A')}")
            
            return price_data
            
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                print(f"⏳ Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print("❌ All retries failed")
                return {}
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return {}
    
    return {}


def get_price_for_coin(coin_name: str, price_data: Dict) -> Optional[Dict]:
    """Extract price info for a given coin name."""
    coingecko_id = None
    for info in COIN_INFO.values():
        if info['name'] == coin_name:
            coingecko_id = info['coingecko_id']
            break
    
    if not coingecko_id:
        return None
    
    return price_data.get(coingecko_id)


def format_price_update_data(coin_name: str, price_data: Dict) -> Optional[Dict]:
    """Format price payload for Supabase update."""
    coin_price = get_price_for_coin(coin_name, price_data)
    if not coin_price:
        return None
    
    return {
        PRICE_USD_COLUMN: coin_price.get('usd'),
        PRICE_KRW_COLUMN: coin_price.get('krw')
    }


def test_coingecko_api():
    """Quick connectivity test to CoinGecko API."""
    print("🧪 Testing CoinGecko API...")
    
    test_coins = ['Ethereum', 'Bitcoin']
    result = fetch_batch_prices(test_coins)
    
    if result:
        print("✅ API reachable!")
        for coin in test_coins:
            price = get_price_for_coin(coin, result)
            if price:
                print(f"   {coin}: ${price.get('usd', 'N/A')}")
            else:
                print(f"   {coin}: no price data")
        
        print(f"\n📊 Meta:")
        print(f"   supported coins in config: {len(COIN_INFO)}")
        print(f"   price columns: {PRICE_USD_COLUMN}, {PRICE_KRW_COLUMN}")
    else:
        print("❌ API unreachable")
    
    return result


if __name__ == "__main__":
    test_coingecko_api()