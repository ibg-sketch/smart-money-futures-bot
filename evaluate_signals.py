#!/usr/bin/env python3
"""
Signal Effectiveness Evaluator
Analyzes signals from signals_log.csv and calculates actual performance
by checking if targets were reached during the ENTIRE maximum period.
"""

import csv
import time
import requests
import yaml
from datetime import datetime, timedelta, timezone
from collections import defaultdict

def load_config():
    """Load configuration from config.yaml"""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load config.yaml: {e}")
        return None

def get_max_duration_minutes(symbol, config):
    """
    Get the maximum check duration for a coin based on its strategy.
    
    Intraday coins (YFI, LUMIA, ANIME): 720 minutes (12 hours max)
    Scalping coins: 60 minutes (max range)
    """
    intraday_coins = ['YFIUSDT', 'LUMIAUSDT', 'ANIMEUSDT']
    
    if symbol in intraday_coins:
        return 720  # 12 hours for intraday strategy
    else:
        return 60   # 60 minutes for scalping strategy

def fetch_price_range_during_period(symbol, timestamp, duration_minutes):
    """
    Fetch ALL candles during the entire target period and find:
    - Highest price reached
    - Lowest price reached  
    - Final close price
    
    This checks if target was hit AT ANY POINT during the period.
    """
    try:
        signal_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        check_time = signal_time + timedelta(minutes=duration_minutes)
        
        # If check_time is in the future, we can't evaluate yet
        now = datetime.now(timezone.utc)
        if check_time > now:
            return None, "PENDING"
        
        # Binance endpoint for klines
        url = "https://fapi.binance.com/fapi/v1/klines"
        
        # Calculate time range
        start_time = int(signal_time.timestamp() * 1000)
        end_time = int(check_time.timestamp() * 1000)
        
        # Choose appropriate interval based on duration
        # For long periods (intraday), use 15m candles to reduce data
        # For short periods (scalping), use 1m candles for precision
        if duration_minutes >= 240:  # 4+ hours
            interval = '15m'
            limit = 1000  # Max allowed
        elif duration_minutes >= 60:
            interval = '5m'
            limit = 1000
        else:
            interval = '1m'
            limit = min(duration_minutes + 5, 1000)
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            return None, "NO_DATA"
        
        # Find the highest high and lowest low across ALL candles
        # Data format: [open_time, open, high, low, close, volume, ...]
        highest_price = max(float(candle[2]) for candle in data)  # High prices
        lowest_price = min(float(candle[3]) for candle in data)   # Low prices
        final_close = float(data[-1][4])  # Last close price
        
        return {
            'highest': highest_price,
            'lowest': lowest_price,
            'close': final_close,
            'candles_checked': len(data)
        }, "SUCCESS"
        
    except Exception as e:
        return None, f"ERROR: {e}"

def get_coin_targets(symbol, config):
    """Get target percentages for a coin from config"""
    if not config or 'coin_configs' not in config:
        return [0.4, 0.7]  # Default
    
    coin_config = config['coin_configs'].get(symbol, config.get('default_coin', {}))
    return coin_config.get('targets', [0.4, 0.7])

def evaluate_signal(row, config):
    """
    Evaluate a single signal from the CSV.
    Checks if target was reached during the ENTIRE maximum period.
    """
    timestamp = row['timestamp']
    symbol = row['symbol']
    verdict = row['verdict']
    confidence = float(row['confidence'])
    entry_price = float(row['entry_price'])
    
    # Get maximum duration for this coin's strategy
    duration_minutes = get_max_duration_minutes(symbol, config)
    
    # Get target percentages from config
    targets = get_coin_targets(symbol, config)
    min_target_pct = targets[0]
    max_target_pct = targets[1]
    
    # Calculate target prices
    if verdict == 'BUY':
        min_target = entry_price * (1 + min_target_pct / 100)
        max_target = entry_price * (1 + max_target_pct / 100)
    else:  # SELL
        min_target = entry_price * (1 - max_target_pct / 100)
        max_target = entry_price * (1 - min_target_pct / 100)
    
    # Fetch price range during entire period
    price_data, status = fetch_price_range_during_period(symbol, timestamp, duration_minutes)
    
    if status != "SUCCESS":
        return {
            'symbol': symbol,
            'timestamp': timestamp,
            'verdict': verdict,
            'entry_price': entry_price,
            'confidence': confidence,
            'min_target': min_target,
            'max_target': max_target,
            'duration_min': duration_minutes,
            'status': status,
            'result': 'N/A',
            'profit': 0
        }
    
    if price_data is None:
        return {
            'symbol': symbol,
            'timestamp': timestamp,
            'verdict': verdict,
            'entry_price': entry_price,
            'confidence': confidence,
            'min_target': min_target,
            'max_target': max_target,
            'duration_min': duration_minutes,
            'status': 'ERROR',
            'result': 'N/A',
            'profit': 0
        }
    
    highest = price_data['highest']
    lowest = price_data['lowest']
    close = price_data['close']
    candles_checked = price_data['candles_checked']
    
    # Determine if target was hit during the period
    if verdict == 'BUY':
        # Check if highest price reached minimum target
        if highest >= min_target:
            if highest >= max_target:
                result = 'BIG_WIN'
                profit = (max_target - entry_price) / entry_price * 100
            else:
                result = 'WIN'
                profit = (min_target - entry_price) / entry_price * 100
        else:
            result = 'LOSS'
            profit = (close - entry_price) / entry_price * 100
    else:  # SELL
        # Check if lowest price reached maximum target (lower for sell)
        if lowest <= max_target:
            if lowest <= min_target:
                result = 'BIG_WIN'
                profit = (entry_price - min_target) / entry_price * 100
            else:
                result = 'WIN'
                profit = (entry_price - max_target) / entry_price * 100
        else:
            result = 'LOSS'
            profit = (entry_price - close) / entry_price * 100
    
    return {
        'symbol': symbol,
        'timestamp': timestamp,
        'verdict': verdict,
        'entry_price': entry_price,
        'confidence': confidence,
        'min_target': min_target,
        'max_target': max_target,
        'duration_min': duration_minutes,
        'highest': highest,
        'lowest': lowest,
        'close': close,
        'candles_checked': candles_checked,
        'status': status,
        'result': result,
        'profit': profit
    }

def main():
    print("=" * 80)
    print("SIGNAL EFFECTIVENESS EVALUATOR")
    print("=" * 80)
    print()
    
    # Load configuration
    config = load_config()
    if config:
        print("✅ Loaded config.yaml - using coin-specific strategies")
    else:
        print("⚠️  Using default configuration")
    print()
    
    # Read signals log
    try:
        with open('signals_log.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            signals = list(reader)
    except FileNotFoundError:
        print("❌ Error: signals_log.csv not found!")
        print("   Generate some signals first by running the bot.")
        return
    
    if not signals:
        print("⚠️  No signals found in signals_log.csv")
        return
    
    print(f"📊 Found {len(signals)} signals to evaluate")
    print()
    print("⏳ Fetching price data from Binance...")
    print("   • Intraday coins (YFI, LUMIA, ANIME): Checking 12-hour window")
    print("   • Scalping coins: Checking 60-minute window")
    print()
    
    # Evaluate each signal
    results = []
    for i, signal in enumerate(signals):
        result = evaluate_signal(signal, config)
        results.append(result)
        
        # Show progress every 10 signals
        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{len(signals)} signals...")
        
        # Rate limit: 0.5 seconds between requests (safe for Binance)
        if i < len(signals) - 1:
            time.sleep(0.5)
    
    # Calculate statistics
    total = len(results)
    evaluable = [r for r in results if r['status'] == 'SUCCESS']
    pending = [r for r in results if r['status'] == 'PENDING']
    
    wins = [r for r in evaluable if r['result'] in ['WIN', 'BIG_WIN']]
    big_wins = [r for r in evaluable if r['result'] == 'BIG_WIN']
    losses = [r for r in evaluable if r['result'] == 'LOSS']
    
    # Stats by symbol
    by_symbol = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total': 0, 'profit': 0})
    for r in evaluable:
        by_symbol[r['symbol']]['total'] += 1
        by_symbol[r['symbol']]['profit'] += r['profit']
        if r['result'] in ['WIN', 'BIG_WIN']:
            by_symbol[r['symbol']]['wins'] += 1
        else:
            by_symbol[r['symbol']]['losses'] += 1
    
    # Stats by confidence
    by_confidence = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total': 0})
    for r in evaluable:
        conf_level = f"{int(r['confidence'] * 100)}%"
        by_confidence[conf_level]['total'] += 1
        if r['result'] in ['WIN', 'BIG_WIN']:
            by_confidence[conf_level]['wins'] += 1
        else:
            by_confidence[conf_level]['losses'] += 1
    
    # Stats by strategy
    intraday_results = [r for r in evaluable if r['symbol'] in ['YFIUSDT', 'LUMIAUSDT', 'ANIMEUSDT']]
    scalping_results = [r for r in evaluable if r['symbol'] not in ['YFIUSDT', 'LUMIAUSDT', 'ANIMEUSDT']]
    
    # Print results
    print("=" * 80)
    print("📈 OVERALL PERFORMANCE")
    print("=" * 80)
    print(f"Total Signals:        {total}")
    print(f"Evaluable:            {len(evaluable)} ({len(evaluable)/total*100:.1f}%)")
    print(f"Pending (too recent): {len(pending)}")
    print()
    
    if evaluable:
        win_rate = len(wins) / len(evaluable) * 100
        avg_profit = sum(r['profit'] for r in evaluable) / len(evaluable)
        avg_win_profit = sum(r['profit'] for r in wins) / len(wins) if wins else 0
        avg_loss = sum(r['profit'] for r in losses) / len(losses) if losses else 0
        
        print(f"✅ Wins:              {len(wins)} ({win_rate:.1f}%)")
        print(f"⚡ Big Wins:          {len(big_wins)} ({len(big_wins)/len(evaluable)*100:.1f}%)")
        print(f"❌ Losses:            {len(losses)} ({len(losses)/len(evaluable)*100:.1f}%)")
        print()
        print(f"💰 Average Profit:    {avg_profit:+.2f}%")
        print(f"📊 Avg Win Profit:    {avg_win_profit:+.2f}%")
        print(f"📉 Avg Loss:          {avg_loss:+.2f}%")
        print()
        
        # Performance rating
        if win_rate >= 75:
            rating = "🏆 EXCELLENT"
        elif win_rate >= 65:
            rating = "✅ VERY GOOD"
        elif win_rate >= 55:
            rating = "👍 GOOD"
        elif win_rate >= 45:
            rating = "⚠️  AVERAGE"
        else:
            rating = "❌ POOR"
        
        print(f"Rating: {rating}")
        print()
    
    # Strategy comparison
    if intraday_results or scalping_results:
        print("=" * 80)
        print("📊 PERFORMANCE BY STRATEGY")
        print("=" * 80)
        
        if scalping_results:
            scalp_wins = [r for r in scalping_results if r['result'] in ['WIN', 'BIG_WIN']]
            scalp_rate = len(scalp_wins) / len(scalping_results) * 100
            scalp_profit = sum(r['profit'] for r in scalping_results) / len(scalping_results)
            print(f"Scalping (60-min):    {len(scalp_wins)}/{len(scalping_results)} wins ({scalp_rate:.1f}%) | Avg: {scalp_profit:+.2f}%")
        
        if intraday_results:
            intra_wins = [r for r in intraday_results if r['result'] in ['WIN', 'BIG_WIN']]
            intra_rate = len(intra_wins) / len(intraday_results) * 100
            intra_profit = sum(r['profit'] for r in intraday_results) / len(intraday_results)
            print(f"Intraday (12-hour):   {len(intra_wins)}/{len(intraday_results)} wins ({intra_rate:.1f}%) | Avg: {intra_profit:+.2f}%")
        print()
    
    # By symbol
    if by_symbol:
        print("=" * 80)
        print("📊 PERFORMANCE BY SYMBOL")
        print("=" * 80)
        for symbol in sorted(by_symbol.keys()):
            stats = by_symbol[symbol]
            win_rate = stats['wins'] / stats['total'] * 100 if stats['total'] > 0 else 0
            avg_profit = stats['profit'] / stats['total'] if stats['total'] > 0 else 0
            duration = "12h" if symbol in ['YFIUSDT', 'LUMIAUSDT', 'ANIMEUSDT'] else "60min"
            print(f"{symbol:12} {stats['wins']:2}/{stats['total']:2} wins ({win_rate:5.1f}%) | Avg: {avg_profit:+.2f}% | ⏱ {duration}")
        print()
    
    # By confidence
    if by_confidence:
        print("=" * 80)
        print("📊 PERFORMANCE BY CONFIDENCE")
        print("=" * 80)
        for conf in sorted(by_confidence.keys()):
            stats = by_confidence[conf]
            win_rate = stats['wins'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"{conf:4} confidence  {stats['wins']:2}/{stats['total']:2} wins ({win_rate:5.1f}%)")
        print()
    
    # Recent signals detail
    print("=" * 80)
    print("📋 RECENT SIGNALS (Last 10)")
    print("=" * 80)
    for r in results[-10:]:
        if r['status'] == 'SUCCESS':
            emoji = "✅" if r['result'] in ['WIN', 'BIG_WIN'] else "❌"
            duration_str = f"{r['duration_min']}min" if r['duration_min'] < 120 else f"{r['duration_min']//60}h"
            print(f"{emoji} {r['timestamp']} | {r['symbol']:8} {r['verdict']:4} @ ${r['entry_price']:.2f} → ${r['close']:.2f} | {r['result']:7} {r['profit']:+6.2f}% | ⏱ {duration_str}")
        elif r['status'] == 'PENDING':
            print(f"⏳ {r['timestamp']} | {r['symbol']:8} {r['verdict']:4} @ ${r['entry_price']:.2f} | PENDING (too recent)")
        else:
            print(f"⚠️  {r['timestamp']} | {r['symbol']:8} {r['verdict']:4} @ ${r['entry_price']:.2f} | {r['status']}")
    print()
    
    print("=" * 80)
    print("✅ Evaluation complete!")
    print("=" * 80)
    print()
    print("Note: This evaluator checks if targets were reached AT ANY POINT")
    print("      during the maximum target period for each coin:")
    print("      • Intraday coins: Entire 12-hour window")
    print("      • Scalping coins: Entire 60-minute window")

if __name__ == '__main__':
    main()
