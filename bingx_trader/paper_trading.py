"""
Paper Trading Simulation
Gets real prices from BingX without placing real orders
"""
import requests
import time
from typing import Dict, Optional

BINANCE_TICKER_URL = "https://fapi.binance.com/fapi/v1/ticker/price"

_last_prices = {}

def get_simulated_price(symbol: str, max_retries: int = 3) -> float:
    if '-' in symbol:
        symbol_formatted = symbol
    else:
        symbol_formatted = symbol.replace('USDT', '-USDT').replace('USDC', '-USDC')
    
    url = "https://open-api.bingx.com/openApi/swap/v2/quote/price"
    params = {'symbol': symbol_formatted}
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get('code') == 0:
                price = float(data['data']['price'])
                _last_prices[symbol] = price
                return price
            else:
                raise Exception(f"BingX API error: {data}")
        
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait_time = 0.5 * (attempt + 1)
                print(f"⚠️  Timeout getting price for {symbol}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                if symbol in _last_prices:
                    print(f"⚠️  Using cached price for {symbol}: ${_last_prices[symbol]:.4f}")
                    return _last_prices[symbol]
                else:
                    print(f"❌ Failed to get price for {symbol} after {max_retries} attempts")
                    raise
        
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 0.5 * (attempt + 1)
                print(f"⚠️  Error getting price for {symbol}: {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                if symbol in _last_prices:
                    print(f"⚠️  Using cached price for {symbol}: ${_last_prices[symbol]:.4f}")
                    return _last_prices[symbol]
                else:
                    print(f"⚠️  BingX price failed for {symbol}: {e}, falling back to Binance")
                    try:
                        response = requests.get(BINANCE_TICKER_URL, params={'symbol': symbol}, timeout=5)
                        response.raise_for_status()
                        price = float(response.json().get('price'))
                        _last_prices[symbol] = price
                        return price
                    except Exception as fallback_err:
                        print(f"❌ Error getting price for {symbol}: {fallback_err}")
                        raise
    
    raise Exception(f"Failed to get price for {symbol} after {max_retries} attempts")
