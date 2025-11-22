"""
Daily Basic Report Generator for Smart Money Signal Bot (MVP)

Computes per-symbol metrics from analysis_log.csv and effectiveness_log.csv:
- profit_factor: Sum of wins / Sum of losses (reward-to-risk ratio)
- win_rate: Wins / (Wins + Losses + Cancelled) 
- blocked_share: % of potential signals blocked by dev_sigma filter
- avg_ttl_min: Average signal duration in minutes

Sends compact table to Telegram and prints to console.
"""

import csv
from collections import defaultdict
from datetime import datetime, time, timezone
from telegram_utils import send_telegram_message


def compute_daily_metrics(analysis_log_path='analysis_log.csv', effectiveness_log_path='effectiveness_log.csv'):
    """
    Compute per-symbol daily metrics from logs.
    
    Returns:
        dict: {symbol: {profit_factor, win_rate, blocked_share, avg_ttl_min}}
    """
    # Get today's date range (00:00 to 23:59 UTC)
    today = datetime.now(timezone.utc).date()
    today_start = datetime.combine(today, time.min, tzinfo=timezone.utc)
    today_end = datetime.combine(today, time.max, tzinfo=timezone.utc)
    
    # Data structures for metrics
    symbol_data = defaultdict(lambda: {
        'blocked_count': 0,
        'total_verdicts': 0,  # BUY + SELL + blocked
        'ttl_sum': 0.0,
        'ttl_count': 0,
        'wins': 0,
        'losses': 0,
        'cancelled': 0,
        'win_pnl': 0.0,
        'loss_pnl': 0.0
    })
    
    # Parse analysis_log.csv for blocked and TTL data
    try:
        with open(analysis_log_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = datetime.datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
                    if not (today_start <= ts <= today_end):
                        continue
                    
                    symbol = row['symbol']
                    verdict = row['verdict']
                    
                    # Track TTL (for BUY/SELL only)
                    if verdict in ['BUY', 'SELL']:
                        ttl = float(row.get('ttl_minutes', 0))
                        if ttl > 0:
                            symbol_data[symbol]['ttl_sum'] += ttl
                            symbol_data[symbol]['ttl_count'] += 1
                    
                    # Track blocked signals
                    blocked = int(row.get('dev_sigma_blocked', 0))
                    if blocked == 1:
                        symbol_data[symbol]['blocked_count'] += 1
                    
                    # Count total potential verdicts (BUY, SELL, or blocked)
                    if verdict in ['BUY', 'SELL'] or blocked == 1:
                        symbol_data[symbol]['total_verdicts'] += 1
                        
                except (ValueError, KeyError) as e:
                    continue
    except FileNotFoundError:
        print(f'[WARN] Daily report: {analysis_log_path} not found')
    
    # Parse effectiveness_log.csv for win/loss data
    try:
        with open(effectiveness_log_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts_sent = datetime.datetime.strptime(row['timestamp_sent'], '%Y-%m-%d %H:%M:%S')
                    if not (today_start <= ts_sent <= today_end):
                        continue
                    
                    symbol = row['symbol']
                    outcome = row['outcome']
                    pnl_pct = float(row.get('pnl_pct', 0.0))
                    
                    if outcome == 'win':
                        symbol_data[symbol]['wins'] += 1
                        symbol_data[symbol]['win_pnl'] += abs(pnl_pct)
                    elif outcome == 'loss':
                        symbol_data[symbol]['losses'] += 1
                        symbol_data[symbol]['loss_pnl'] += abs(pnl_pct)
                    elif outcome == 'cancelled':
                        symbol_data[symbol]['cancelled'] += 1
                        
                except (ValueError, KeyError) as e:
                    continue
    except FileNotFoundError:
        print(f'[WARN] Daily report: {effectiveness_log_path} not found')
    
    # Compute metrics per symbol
    metrics = {}
    for symbol, data in symbol_data.items():
        # Profit factor: sum(wins) / sum(losses)
        profit_factor = data['win_pnl'] / data['loss_pnl'] if data['loss_pnl'] > 0 else (
            999.0 if data['win_pnl'] > 0 else 0.0
        )
        
        # Win rate: wins / (wins + losses + cancelled)
        total_signals = data['wins'] + data['losses'] + data['cancelled']
        win_rate = data['wins'] / total_signals if total_signals > 0 else 0.0
        
        # Blocked share: blocked / (blocked + actual signals)
        blocked_share = data['blocked_count'] / data['total_verdicts'] if data['total_verdicts'] > 0 else 0.0
        
        # Average TTL
        avg_ttl_min = data['ttl_sum'] / data['ttl_count'] if data['ttl_count'] > 0 else 0.0
        
        metrics[symbol] = {
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'blocked_share': blocked_share,
            'avg_ttl_min': avg_ttl_min,
            'signal_count': total_signals
        }
    
    return metrics


def format_daily_report_table(metrics):
    """
    Format metrics as compact ASCII table for Telegram (HTML).
    
    Args:
        metrics: dict from compute_daily_metrics()
    
    Returns:
        str: HTML-formatted table
    """
    if not metrics:
        return "<b>📊 Daily Basic Report (MVP)</b>\n<i>No signals generated today</i>"
    
    # Sort by symbol
    sorted_symbols = sorted(metrics.keys())
    
    lines = []
    lines.append("<b>📊 Daily Basic Report (MVP - UTC)</b>")
    lines.append(f"<code>{'Symbol':<10} {'PF':>5} {'WR%':>5} {'Block%':>6} {'TTL':>4} {'#':>3}</code>")
    lines.append("<code>" + "─" * 42 + "</code>")
    
    for symbol in sorted_symbols:
        m = metrics[symbol]
        pf = m['profit_factor']
        wr = m['win_rate'] * 100
        blocked = m['blocked_share'] * 100
        ttl = m['avg_ttl_min']
        count = m['signal_count']
        
        # Format profit factor (cap display at 99.9)
        pf_str = f"{min(pf, 99.9):.1f}" if pf < 999 else "∞"
        
        lines.append(
            f"<code>{symbol:<10} {pf_str:>5} {wr:>5.1f} {blocked:>6.1f} {ttl:>4.0f} {count:>3}</code>"
        )
    
    lines.append("<code>" + "─" * 42 + "</code>")
    lines.append("<i>PF=Profit Factor, WR=Win Rate, #=Signal Count</i>")
    
    return "\n".join(lines)


def send_daily_report():
    """
    Compute daily metrics, format table, send to Telegram, and print to console.
    MVP version - no quality gates, no Brier scores.
    """
    print('[DAILY_REPORT] Generating daily basic report (MVP)...')
    
    # Compute daily metrics
    metrics = compute_daily_metrics()
    
    # Format report
    report = format_daily_report_table(metrics)
    
    # Print to console
    print('\n' + '=' * 60)
    print(report.replace('<b>', '').replace('</b>', '').replace('<code>', '').replace('</code>', '').replace('<i>', '').replace('</i>', ''))
    print('=' * 60 + '\n')
    
    # Send to Telegram
    try:
        send_telegram_message(report)
        print('[DAILY_REPORT] Sent to Telegram successfully')
    except Exception as e:
        print(f'[DAILY_REPORT] Failed to send to Telegram: {e}')


if __name__ == '__main__':
    # Test report generation
    send_daily_report()
