import os
import time
from typing import Dict, Optional

import pandas as pd

_CALIBRATION_CACHE: Dict[str, Dict[str, float]] = {}
_CALIBRATION_TTL = 300


def load_effectiveness_log():
    """Load effectiveness_log.csv and return as pandas DataFrame, or None if failed."""
    log_file = 'effectiveness_log.csv'

    if not os.path.exists(log_file):
        return None

    try:
        df = pd.read_csv(log_file)

        required_cols = ['symbol', 'verdict', 'confidence', 'result']
        if not all(col in df.columns for col in required_cols):
            return None

        return df
    except Exception:
        return None


def calibrate_confidence(symbol, verdict, formula_confidence, config=None):
    """Calibrate formula-based confidence using empirical win rates."""
    global _CALIBRATION_CACHE

    cache_key = f"{symbol}_{verdict}"
    now = time.time()
    if cache_key in _CALIBRATION_CACHE:
        cached_data = _CALIBRATION_CACHE[cache_key]
        if now - cached_data['timestamp'] < _CALIBRATION_TTL:
            return cached_data['value']

    df = load_effectiveness_log()
    if df is None:
        return formula_confidence

    recent = df[(df['symbol'] == symbol) & (df['verdict'] == verdict)]

    if recent.empty:
        return formula_confidence

    recent_sorted = recent.sort_values(by='timestamp', ascending=False)
    recent_lookback = recent_sorted.head(50)

    total = len(recent_lookback)
    if total == 0:
        return formula_confidence

    wins = len(recent_lookback[recent_lookback['result'] == 'WIN'])
    empirical_win_rate = wins / total if total > 0 else 0.0

    min_pct = config.get('min_confidence', 0.70) if config else 0.70
    max_bound = 0.80 if verdict == 'SELL' else 0.65

    if empirical_win_rate < min_pct:
        empirical_win_rate = min_pct

    if total < 10:
        formula_weight = 0.90
        empirical_weight = 0.10
    elif total < 30:
        formula_weight = 0.70
        empirical_weight = 0.30
    elif total < 50:
        formula_weight = 0.50
        empirical_weight = 0.50
    else:
        formula_weight = 0.40
        empirical_weight = 0.60

    calibrated = (formula_weight * formula_confidence) + (empirical_weight * empirical_win_rate)
    calibrated = min(max_bound, calibrated)

    _CALIBRATION_CACHE[cache_key] = {'value': calibrated, 'timestamp': now}
    return calibrated


def aggregate_recent_analysis(symbol, minutes=5):
    """Aggregate the last N minutes of analysis data from analysis_log.csv."""
    log_file = 'analysis_log.csv'

    if not os.path.exists(log_file):
        return None

    try:
        df = pd.read_csv(log_file)

        if 'timestamp' not in df.columns:
            return None

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        now = pd.Timestamp.now()
        lookback_start = now - pd.Timedelta(minutes=minutes)

        recent_data = df[
            (df['symbol'] == symbol)
            & (df['timestamp'] > lookback_start)
            & (df['timestamp'] <= now)
        ]

        if len(recent_data) < 2:
            return None

        aggregated = {
            'cvd_mean': recent_data['cvd'].mean(),
            'cvd_std': recent_data['cvd'].std() if len(recent_data) > 1 else 0,
            'cvd_max': recent_data['cvd'].max(),
            'cvd_min': recent_data['cvd'].min(),
            'oi_change_mean': recent_data['oi_change'].mean() if 'oi_change' in recent_data.columns else 0,
            'oi_change_pct_mean': recent_data['oi_change_pct'].mean() if 'oi_change_pct' in recent_data.columns else 0,
            'price_vs_vwap_pct_mean': recent_data['price_vs_vwap_pct'].mean() if 'price_vs_vwap_pct' in recent_data.columns else 0,
            'price_vs_vwap_pct_std': recent_data['price_vs_vwap_pct'].std() if 'price_vs_vwap_pct' in recent_data.columns and len(recent_data) > 1 else 0,
            'rsi_mean': recent_data['rsi'].mean() if 'rsi' in recent_data.columns else 50,
            'rsi_min': recent_data['rsi'].min() if 'rsi' in recent_data.columns else 50,
            'rsi_max': recent_data['rsi'].max() if 'rsi' in recent_data.columns else 50,
            'volume_sum': recent_data['volume'].sum() if 'volume' in recent_data.columns else 0,
            'volume_median': recent_data['volume_median'].mean() if 'volume_median' in recent_data.columns else 1,
            'data_points': len(recent_data),
            'timeframe_minutes': minutes,
        }

        return aggregated

    except Exception:
        return None


__all__ = ['calibrate_confidence', 'load_effectiveness_log', 'aggregate_recent_analysis']
