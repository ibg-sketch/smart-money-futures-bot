"""
Brier Score & Reliability Curve Computation
Evaluates confidence calibration for daily KPI reporting.
"""
import csv
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np

def compute_brier_score_per_symbol(lookback_days=1):
    """
    Compute Brier score and reliability curve (10 bins) per symbol.
    
    Args:
        lookback_days: Number of days to analyze (default: 1 for daily report)
    
    Returns:
        Dict of {symbol: {brier_score, reliability_bins}}
    """
    # Read effectiveness log to get signal outcomes
    try:
        with open('effectiveness_log.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            signals = list(reader)
    except:
        print('[BRIER] No effectiveness_log.csv found')
        return {}
    
    # Filter to last N days
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    
    # Group signals by symbol
    symbol_data = defaultdict(list)
    
    for signal in signals:
        try:
            # Parse timestamp
            timestamp_str = signal.get('timestamp_sent', '')
            if not timestamp_str:
                continue
            
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            
            # Skip if outside lookback window
            if timestamp < cutoff_date:
                continue
            
            symbol = signal.get('symbol', '')
            confidence = float(signal.get('confidence', 0))
            outcome = signal.get('outcome', '').lower()  # Normalize to lowercase
            
            # Skip if no outcome yet
            if outcome not in ['win', 'loss', 'cancelled']:
                continue
            
            # Convert outcome to binary (1 for win, 0 for loss/cancelled)
            outcome_binary = 1.0 if outcome == 'win' else 0.0
            
            symbol_data[symbol].append({
                'confidence': confidence,
                'outcome': outcome_binary
            })
        except Exception as e:
            continue
    
    # Compute Brier score and reliability curve for each symbol
    results = {}
    
    for symbol, data in symbol_data.items():
        if len(data) < 3:  # Need at least 3 signals for meaningful stats
            continue
        
        confidences = np.array([d['confidence'] for d in data])
        outcomes = np.array([d['outcome'] for d in data])
        
        # Compute Brier score: mean((forecast - outcome)^2)
        brier_score = np.mean((confidences - outcomes) ** 2)
        
        # Compute reliability curve (10 bins)
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        reliability_bins = []
        for i in range(n_bins):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            
            # Find signals in this bin
            mask = (confidences >= bin_start) & (confidences < bin_end)
            if i == n_bins - 1:  # Last bin includes upper edge
                mask = (confidences >= bin_start) & (confidences <= bin_end)
            
            bin_confidences = confidences[mask]
            bin_outcomes = outcomes[mask]
            
            if len(bin_outcomes) == 0:
                continue
            
            # Calculate bin statistics
            mean_confidence = np.mean(bin_confidences)
            mean_outcome = np.mean(bin_outcomes)  # Actual win rate
            count = len(bin_outcomes)
            
            reliability_bins.append({
                'bin_start': bin_start,
                'bin_end': bin_end,
                'mean_confidence': mean_confidence,
                'mean_outcome': mean_outcome,
                'count': count
            })
        
        results[symbol] = {
            'brier_score': brier_score,
            'reliability_bins': reliability_bins,
            'n_signals': len(data)
        }
    
    return results

def format_brier_for_telegram(brier_results):
    """
    Format Brier score results for compact Telegram display.
    
    Args:
        brier_results: Dict from compute_brier_score_per_symbol()
    
    Returns:
        String with formatted Brier score info
    """
    if not brier_results:
        return ''
    
    lines = []
    lines.append('\n📊 <b>Confidence Calibration (24h)</b>\n')
    
    for symbol in sorted(brier_results.keys()):
        data = brier_results[symbol]
        brier = data['brier_score']
        n_signals = data['n_signals']
        
        # Brier score interpretation:
        # 0.00-0.05: Excellent
        # 0.05-0.10: Good
        # 0.10-0.15: Fair
        # 0.15+: Poor
        if brier < 0.05:
            quality = '🟢'
        elif brier < 0.10:
            quality = '🟡'
        elif brier < 0.15:
            quality = '🟠'
        else:
            quality = '🔴'
        
        # Format reliability bins compactly
        bins = data['reliability_bins']
        if len(bins) > 0:
            # Show only bins with significant data (>2 signals)
            significant_bins = [b for b in bins if b['count'] > 2]
            
            # Find worst calibrated bin (largest gap between confidence and outcome)
            if significant_bins:
                worst_bin = max(significant_bins, key=lambda b: abs(b['mean_confidence'] - b['mean_outcome']))
                gap = worst_bin['mean_confidence'] - worst_bin['mean_outcome']
                gap_str = f'+{gap:.0%}' if gap > 0 else f'{gap:.0%}'
                
                line = f'{quality} <code>{symbol:8}</code> Brier: <code>{brier:.3f}</code> ({n_signals:2}sig) '
                line += f'Worst: {worst_bin["mean_confidence"]:.0%}→{worst_bin["mean_outcome"]:.0%} ({gap_str})'
            else:
                line = f'{quality} <code>{symbol:8}</code> Brier: <code>{brier:.3f}</code> ({n_signals:2}sig)'
        else:
            line = f'{quality} <code>{symbol:8}</code> Brier: <code>{brier:.3f}</code> ({n_signals:2}sig)'
        
        lines.append(line)
    
    return '\n'.join(lines)
