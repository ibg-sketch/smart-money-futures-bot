"""
UIF Feature Engine - Main Runner

Collects 4 local indicators (ADX14, PSAR, Momentum5, VolAccel) every 5 minutes.
Read-only preparation for ML feature engineering - no signal logic changes.
"""

import os
import sys
import time
import yaml
import pandas as pd
import requests
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from services.uif_feature_engine.calculators import calculate_all_indicators
from services.uif_feature_engine.snapshot import write_snapshot
from services.uif_feature_engine.writer import UIFWriter
from services.uif_feature_engine.health import print_status


CONFIG_PATH = 'config.yaml'


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        sys.exit(1)


def fetch_ohlcv(symbol: str, limit: int = 60) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data from CryptoCompare API (free, no regional restrictions).
    
    Args:
        symbol: Trading pair (e.g., BTCUSDT)
        limit: Number of 5-minute candles to fetch
    
    Returns:
        DataFrame with OHLCV data or None on error
    """
    try:
        # Extract base currency (BTC from BTCUSDT)
        base_currency = symbol.replace('USDT', '')
        
        url = "https://min-api.cryptocompare.com/data/v2/histominute"
        params = {
            'fsym': base_currency,
            'tsym': 'USD',
            'limit': limit,
            'aggregate': 5  # 5-minute candles
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('Response') != 'Success' or 'Data' not in data:
            return None
        
        candles = data['Data']['Data']
        
        df = pd.DataFrame(candles)
        
        # Map CryptoCompare fields to our format
        df = df.rename(columns={
            'time': 'timestamp',
            'volumefrom': 'volume',
            'volumeto': 'quoteVolume'
        })
        
        # Keep only needed columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quoteVolume']]
        
        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume', 'quoteVolume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    except Exception as e:
        print(f"[ERROR] Failed to fetch OHLCV for {symbol}: {e}")
        return None


def process_symbol(symbol: str, writer: UIFWriter) -> Optional[Dict[str, Any]]:
    """
    Process single symbol: fetch OHLCV and calculate indicators.
    
    Args:
        symbol: Trading pair
        writer: CSV writer instance
    
    Returns:
        Dict with indicator values or None on error
    """
    start_time = time.time()
    
    try:
        # Fetch OHLCV
        df = fetch_ohlcv(symbol, limit=60)
        
        if df is None:
            writer.append(symbol, {}, 0, "OHLCV_FETCH_FAILED")
            return None
        
        if len(df) < 30:
            writer.append(symbol, {}, 0, f"INSUFFICIENT_DATA:rows={len(df)}")
            return None
        
        # Calculate indicators
        indicators = calculate_all_indicators(df)
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Check for None values
        missing = [k for k, v in indicators.items() if v is None]
        
        # If all indicators are None, mark as calculation failure
        if all(v is None for v in indicators.values()):
            writer.append(symbol, {}, latency_ms, "CALC_ALL_FAILED")
            return None
        
        # If some indicators are None, log but continue
        errors = f"PARTIAL:{','.join(missing)}" if missing else ""
        
        # Log to CSV
        writer.append(symbol, indicators, latency_ms, errors)
        
        # Prepare snapshot entry (use 0 for None values)
        result = {
            'adx14': indicators.get('adx14') or 0.0,
            'psar_state': indicators.get('psar_state') or 0,
            'momentum5': indicators.get('momentum5') or 0.0,
            'vol_accel': indicators.get('vol_accel') or 0.0,
            'updated': int(time.time())
        }
        
        return result
    
    except Exception as e:
        error_msg = str(e)[:80]
        writer.append(symbol, {}, 0, f"ERROR:{error_msg}")
        return None


def run_collection_cycle(symbols: list, writer: UIFWriter) -> bool:
    """
    Run single collection cycle for all symbols.
    
    Args:
        symbols: List of trading pairs
        writer: CSV writer instance
    
    Returns:
        True if cycle completed successfully
    """
    print(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] Starting UIF collection cycle")
    
    snapshot_data = {}
    success_count = 0
    
    for symbol in symbols:
        result = process_symbol(symbol, writer)
        
        if result:
            snapshot_data[symbol] = result
            success_count += 1
            print(f"  ✓ {symbol}: ADX={result['adx14']}, PSAR={result['psar_state']}, "
                  f"Mom={result['momentum5']}, VolA={result['vol_accel']}")
        else:
            print(f"  ✗ {symbol}: Failed")
    
    # Write snapshot
    if snapshot_data:
        write_snapshot(snapshot_data)
        print(f"[INFO] Snapshot updated with {success_count}/{len(symbols)} symbols")
    
    return success_count > 0


def main():
    """Main runner loop."""
    print("="*60)
    print("UIF Feature Engine - Local Indicators Collection")
    print("="*60)
    
    # Load config
    config = load_config()
    
    # Check feature flag
    if not config.get('feature_flags', {}).get('enable_uif_engine', False):
        print("[WARN] enable_uif_engine flag is disabled in config.yaml")
        print("[INFO] Service will run in dry-run mode")
    
    symbols = config.get('symbols', [])
    if not symbols:
        print("[ERROR] No symbols configured")
        sys.exit(1)
    
    print(f"[INFO] Monitoring {len(symbols)} symbols: {', '.join(symbols)}")
    
    # Initialize CSV writer
    writer = UIFWriter(
        rotate_mb=config.get('data_feeds', {}).get('sinks', {}).get('rotate_mb', 200),
        keep_files=config.get('data_feeds', {}).get('sinks', {}).get('keep_files', 14)
    )
    
    # Get interval
    interval_sec = config.get('uif_engine', {}).get('interval_sec', 300)  # Default 5 minutes
    print(f"[INFO] Collection interval: {interval_sec}s")
    
    # Initial health check
    print("\n[INFO] Initial health status:")
    print_status()
    
    print("\n[INFO] Starting collection loop...")
    
    # Main loop
    while True:
        try:
            run_collection_cycle(symbols, writer)
            
            print(f"[INFO] Sleeping {interval_sec}s until next cycle...")
            time.sleep(interval_sec)
        
        except KeyboardInterrupt:
            print("\n[INFO] Shutdown requested")
            break
        
        except Exception as e:
            print(f"[ERROR] Cycle failed: {e}")
            print("[INFO] Retrying in 60s...")
            time.sleep(60)
    
    print("[INFO] UIF Feature Engine stopped")


if __name__ == "__main__":
    main()
