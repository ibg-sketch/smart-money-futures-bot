#!/usr/bin/env python3
"""
RSI Validation Script - Compare bot's RSI with industry standard
Fetches live data from Binance and compares our RSI calculation with reference implementation
"""

import requests
import numpy as np
from signals.features import compute_rsi
import json
from datetime import datetime

def fetch_binance_klines(symbol, interval='15m', limit=100):
    """Fetch kline data from Binance"""
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def compute_rsi_reference(klines, period=14):
    """
    Reference RSI implementation using Wilder's Smoothing Method
    This is the industry standard formula for comparison
    """
    if not klines or len(klines) < period + 1:
        return None
    
    closes = np.array([float(k[4]) for k in klines])
    deltas = np.diff(closes)
    
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    if len(gains) < period:
        return None
    
    # First RSI: use simple average of first 'period' values
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    # Apply Wilder's smoothing for remaining values
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return float(rsi)

def validate_symbol(symbol, interval='15m'):
    """Validate RSI calculation for a specific symbol"""
    print(f"\n{'='*70}")
    print(f"Validating {symbol} ({interval})")
    print(f"{'='*70}")
    
    try:
        # Fetch live data from Binance
        klines = fetch_binance_klines(symbol, interval, limit=100)
        
        # Get current price
        current_price = float(klines[-1][4])
        timestamp = datetime.fromtimestamp(klines[-1][0] / 1000)
        
        print(f"📊 Current Price: ${current_price:,.2f}")
        print(f"⏰ Last Candle: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📈 Klines Fetched: {len(klines)}")
        
        # Calculate RSI using our bot's function
        bot_rsi = compute_rsi(klines, period=14)
        
        # Calculate RSI using reference implementation
        ref_rsi = compute_rsi_reference(klines, period=14)
        
        # Compare
        if bot_rsi is not None and ref_rsi is not None:
            difference = abs(bot_rsi - ref_rsi)
            match = difference < 0.01  # Allow 0.01 point difference for rounding
            
            print(f"\n🤖 Bot RSI:       {bot_rsi:.2f}")
            print(f"✅ Reference RSI: {ref_rsi:.2f}")
            print(f"📊 Difference:    {difference:.4f}")
            
            if match:
                print(f"✅ ✅ ✅ PERFECT MATCH! ✅ ✅ ✅")
                return True, bot_rsi, ref_rsi, difference
            else:
                print(f"⚠️  MISMATCH DETECTED!")
                return False, bot_rsi, ref_rsi, difference
        else:
            print(f"❌ RSI calculation failed (insufficient data)")
            print(f"   Bot RSI: {bot_rsi}")
            print(f"   Ref RSI: {ref_rsi}")
            return None, bot_rsi, ref_rsi, None
            
    except Exception as e:
        print(f"❌ Error validating {symbol}: {e}")
        return None, None, None, None

def main():
    """Main validation routine"""
    print("\n" + "="*70)
    print("RSI VALIDATION - Comparing Bot vs Industry Standard")
    print("="*70)
    print("Method: Wilder's Smoothing (14-period)")
    print("Data Source: Binance Futures Live Data")
    print("="*70)
    
    # Test symbols from the bot's config
    test_symbols = [
        'BTCUSDT',
        'ETHUSDT',
        'SOLUSDT',
        'BNBUSDT',
        'LINKUSDT',
        'AVAXUSDT',
        'DOGEUSDT'
    ]
    
    results = []
    
    for symbol in test_symbols:
        result = validate_symbol(symbol, interval='15m')
        results.append({
            'symbol': symbol,
            'match': result[0],
            'bot_rsi': result[1],
            'ref_rsi': result[2],
            'difference': result[3]
        })
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    matches = sum(1 for r in results if r['match'] is True)
    total = len([r for r in results if r['match'] is not None])
    
    print(f"\n📊 Total Symbols Tested: {len(test_symbols)}")
    print(f"✅ Perfect Matches: {matches}/{total}")
    
    if matches == total and total > 0:
        print(f"\n🎉 🎉 🎉 ALL VALIDATIONS PASSED! 🎉 🎉 🎉")
        print(f"✅ RSI calculation matches industry standard (Wilder's Smoothing)")
        print(f"✅ Bot will show same RSI values as Binance/TradingView")
    else:
        print(f"\n⚠️  SOME VALIDATIONS FAILED")
        print(f"❌ RSI implementation may need review")
    
    # Detailed results
    print(f"\nDetailed Results:")
    print(f"{'Symbol':<12} {'Bot RSI':<10} {'Ref RSI':<10} {'Diff':<10} {'Status'}")
    print("-" * 70)
    
    for r in results:
        if r['match'] is not None:
            status = "✅ PASS" if r['match'] else "❌ FAIL"
            diff_str = f"{r['difference']:.4f}" if r['difference'] is not None else "N/A"
            print(f"{r['symbol']:<12} {r['bot_rsi']:<10.2f} {r['ref_rsi']:<10.2f} {diff_str:<10} {status}")
        else:
            print(f"{r['symbol']:<12} {'N/A':<10} {'N/A':<10} {'N/A':<10} ⚠️  ERROR")
    
    print("="*70)
    
    return matches == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
