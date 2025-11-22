import os
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

COINALYZE_API = 'https://api.coinalyze.net/v1'
COINALYZE_KEY = os.getenv('COINALYZE_API_KEY')


class DataFeedUnavailable(RuntimeError):
    """Raised when upstream market data cannot be reached (network/proxy issues)."""


# Simple in-memory cache to reduce API calls (2.5-minute TTL)
_API_CACHE: Dict[str, Tuple[object, float]] = {}
_CACHE_TTL = 150  # seconds

# Rate-limited warning timestamps (UIF-30: prevent log spam)
_WARN_LOG_TIMESTAMPS: Dict[str, float] = {}


def _warn_once_per_minute(key: str, message: str) -> None:
    """Log warning message at most once per minute to prevent spam."""
    now = time.time()
    last_warn = _WARN_LOG_TIMESTAMPS.get(key, 0)
    if now - last_warn >= 60:
        print(f"[WARN] {message}")
        _WARN_LOG_TIMESTAMPS[key] = now


def _get(u, p=None, t=20, retries=5):
    """GET request with retry logic for 429 rate limit errors."""
    p = p or {}
    if COINALYZE_KEY:
        p['api_key'] = COINALYZE_KEY

    for attempt in range(retries):
        try:
            r = requests.get(u, params=p, timeout=t)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.ProxyError as e:
            # Explicitly surface proxy blocks (e.g., CONNECT 403) so the caller can alert
            _warn_once_per_minute('proxy_block', f'Proxy blocked data feed: {e}')
            raise DataFeedUnavailable(f'Proxy blocked access to {u}: {e}')
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            _warn_once_per_minute('conn_error', f'Data feed connection error: {e}')
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise DataFeedUnavailable(f'Data feed unavailable: {e}')
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and attempt < retries - 1:
                wait_time = (2 ** attempt) * 1.5
                time.sleep(wait_time)
                continue
            raise

    raise Exception(f"Failed after {retries} retries")


def _symbol_to_coinalyze(s: str) -> str:
    symbol_map = {
        'BTCUSDT': 'BTCUSDT_PERP.A',
        'ETHUSDT': 'ETHUSDT_PERP.A',
        'BNBUSDT': 'BNBUSDT_PERP.A',
        'SOLUSDT': 'SOLUSDT_PERP.A',
        'AVAXUSDT': 'AVAXUSDT_PERP.A',
        'DOGEUSDT': 'DOGEUSDT_PERP.A',
        'LINKUSDT': 'LINKUSDT_PERP.A',
        'YFIUSDT': 'YFIUSDT_PERP.A',
        'LUMIAUSDT': 'LUMIAUSDT_PERP.A',
        'ANIMEUSDT': 'ANIMEUSDT_PERP.A'
    }
    return symbol_map.get(s, f"{s}_PERP.A")


def fetch_klines(s, i, l=200):
    """Fetch klines with 1-minute caching to reduce API calls."""
    cache_key = f"klines_{s}_{i}_{l}"
    now = time.time()

    if cache_key in _API_CACHE:
        cached_data, cached_time = _API_CACHE[cache_key]
        if now - cached_time < _CACHE_TTL:
            return cached_data

    interval_map = {'1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min', '30m': '30min', '1h': '1hour', '2h': '2hour', '4h': '4hour', '6h': '6hour', '12h': '12hour', '1d': 'daily'}
    iv = interval_map.get(i, '15min')
    to_ts = int(time.time())
    minutes = {'1min': 1, '3min': 3, '5min': 5, '15min': 15, '30min': 30, '1hour': 60, '2hour': 120, '4hour': 240, '6hour': 360, '12hour': 720, 'daily': 1440}
    from_ts = to_ts - (l * minutes.get(iv, 15) * 60)
    sym = _symbol_to_coinalyze(s)
    data = _get(f"{COINALYZE_API}/ohlcv-history", {'symbols': sym, 'interval': iv, 'from': from_ts, 'to': to_ts})
    if not data or not isinstance(data, list) or not data[0].get('history'):
        result = []
    else:
        hist = data[0]['history']
        result = [[int(h['t']) * 1000, float(h['o']), float(h['h']), float(h['l']), float(h['c']), float(h.get('v', 0)), int(h['t']) * 1000] for h in hist]

    _API_CACHE[cache_key] = (result, now)
    return result


def fetch_agg_trades(s, l=1000):
    return []


def fetch_open_interest(s):
    """Fetch OI with 1-minute caching to reduce API calls."""
    cache_key = f"oi_{s}"
    now = time.time()

    if cache_key in _API_CACHE:
        cached_data, cached_time = _API_CACHE[cache_key]
        if now - cached_time < _CACHE_TTL:
            return cached_data

    sym = _symbol_to_coinalyze(s)
    data = _get(f"{COINALYZE_API}/open-interest", {'symbols': sym, 'convert_to_usd': 'true'})
    if data and isinstance(data, list) and len(data) > 0:
        result = float(data[0].get('value', 0))
    else:
        result = 0.0

    _API_CACHE[cache_key] = (result, now)
    return result


def fetch_open_interest_hist(s, p='5min', l=30):
    """Fetch OI history with 1-minute caching to reduce API calls."""
    cache_key = f"oi_hist_{s}_{p}_{l}"
    now = time.time()

    if cache_key in _API_CACHE:
        cached_data, cached_time = _API_CACHE[cache_key]
        if now - cached_time < _CACHE_TTL:
            return cached_data

    try:
        sym = _symbol_to_coinalyze(s)
        to_ts = int(time.time())
        from_ts = to_ts - (l * 5 * 60)
        data = _get(f"{COINALYZE_API}/open-interest-history", {'symbols': sym, 'interval': '5min', 'from': from_ts, 'to': to_ts, 'convert_to_usd': 'true'})
        if data and isinstance(data, list) and data[0].get('history'):
            result = [float(h['c']) for h in data[0]['history']]
        else:
            result = []
    except Exception:
        result = []

    _API_CACHE[cache_key] = (result, now)
    return result


def fetch_liquidations(s, st=None, en=None, l=1000):
    """Read liquidation data from liquidation_service data file."""
    try:
        import json
        from pathlib import Path
        liq_file = Path('liquidation_data.json')
        if not liq_file.exists():
            return {'long_count': 0, 'short_count': 0, 'long_usd': 0.0, 'short_usd': 0.0}

        with open(liq_file, 'r') as f:
            data = json.load(f)

        last_update = data.get('last_update', 0)
        if time.time() - last_update > 300:
            return {'long_count': 0, 'short_count': 0, 'long_usd': 0.0, 'short_usd': 0.0}

        liquidations = data.get('liquidations', {})
        return liquidations.get(s, {'long_count': 0, 'short_count': 0, 'long_usd': 0.0, 'short_usd': 0.0})
    except Exception:
        return {'long_count': 0, 'short_count': 0, 'long_usd': 0.0, 'short_usd': 0.0}


# TAAPI removed - using local VWAP calculation only (more reliable and free)

def fetch_basis(symbol):
    """Fetch basis_pct from feeds_snapshot.json (written by Data Feeds Service)."""
    global _API_CACHE
    cache_key = 'basis_snapshot'
    now = time.time()

    if cache_key in _API_CACHE:
        cached_snapshot, cached_time = _API_CACHE[cache_key]
        if now - cached_time < 5:
            snapshot = cached_snapshot
        else:
            snapshot = None
    else:
        snapshot = None

    if snapshot is None:
        try:
            import json
            from pathlib import Path
            snapshot_file = Path('feeds_snapshot.json')
            if not snapshot_file.exists():
                _warn_once_per_minute('basis_snapshot_missing', 'feeds_snapshot.json missing - basis_pct unavailable')
                return None, None

            with open(snapshot_file, 'r') as f:
                snapshot = json.load(f)
            _API_CACHE[cache_key] = (snapshot, now)
        except Exception:
            _warn_once_per_minute('basis_snapshot_read_error', 'Failed to read feeds_snapshot.json - basis_pct unavailable')
            return None, None

    try:
        basis_data = snapshot.get('basis_pct', {})
        meta = snapshot.get('_meta', {})
        last_update = meta.get('last_update', 0)
        age_sec = int(time.time() - last_update) if last_update else None

        if age_sec is not None and age_sec > 120:
            _warn_once_per_minute(f'basis_stale_{symbol}', f'Basis data stale for {symbol} (age={age_sec}s)')
            return None, age_sec

        symbol_basis = basis_data.get(symbol)
        if symbol_basis is None:
            return None, age_sec

        return float(symbol_basis), age_sec
    except Exception:
        _warn_once_per_minute('basis_extract_error', 'Failed to extract basis_pct from snapshot')
        return None, None


def fetch_uif_snapshot(symbol):
    """Fetch UIF features from feeds_snapshot.json (UIF-12)."""
    global _API_CACHE
    cache_key = 'uif_snapshot'
    now = time.time()

    if cache_key in _API_CACHE:
        cached_snapshot, cached_time = _API_CACHE[cache_key]
        if now - cached_time < 5:
            snapshot = cached_snapshot
        else:
            snapshot = None
    else:
        snapshot = None

    if snapshot is None:
        try:
            import json
            from pathlib import Path
            snapshot_file = Path('feeds_snapshot.json')
            if not snapshot_file.exists():
                _warn_once_per_minute('uif_snapshot_missing', 'feeds_snapshot.json missing - UIF data unavailable')
                return None, None

            with open(snapshot_file, 'r') as f:
                snapshot = json.load(f)
            _API_CACHE[cache_key] = (snapshot, now)
        except Exception:
            _warn_once_per_minute('uif_snapshot_read_error', 'Failed to read feeds_snapshot.json - UIF data unavailable')
            return None, None

    try:
        uif_data = snapshot.get('uif_features', {})
        meta = snapshot.get('_meta', {})
        last_update = meta.get('last_update', 0)
        age_sec = int(time.time() - last_update) if last_update else None

        if age_sec is not None and age_sec > 120:
            _warn_once_per_minute(f'uif_stale_{symbol}', f'UIF data stale for {symbol} (age={age_sec}s)')
            return None, age_sec

        symbol_data = uif_data.get(symbol)
        if symbol_data is None:
            return None, age_sec

        return symbol_data, age_sec
    except Exception:
        _warn_once_per_minute('uif_extract_error', 'Failed to extract UIF features from snapshot')
        return None, None


def fetch_funding_rate(symbol):
    try:
        base = symbol.replace('USDT', '')
        okx_symbol = f'{base}-USDT-SWAP'
        url = 'https://www.okx.com/api/v5/public/funding-rate'
        params = {'instId': okx_symbol}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        if data.get('code') == '0' and data.get('data'):
            funding_rate = float(data['data'][0]['fundingRate'])
            return funding_rate
        else:
            print(f"[FUNDING ERROR] {symbol}: OKX API returned code={data.get('code')}, msg={data.get('msg')}")
            return 0.0
    except Exception as e:
        print(f"[FUNDING ERROR] {symbol}: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0.0


def compute_cvd(symbol, lookback_ms):
    """Read CVD (Cumulative Volume Delta) from cvd_service data file."""
    try:
        import json
        from pathlib import Path
        cvd_file = Path('cvd_data.json')
        if not cvd_file.exists():
            return 0.0

        with open(cvd_file, 'r') as f:
            data = json.load(f)

        last_update = data.get('last_update', 0)
        if time.time() - last_update > 300:
            return 0.0

        cvd_values = data.get('cvd', {})
        return float(cvd_values.get(symbol, 0.0))
    except Exception:
        return 0.0


def validate_signal_momentum_fresh(symbol, verdict, original_cvd):
    """
    CRITICAL PRE-BROADCAST VALIDATION: Bypass cache and verify signal direction still valid.

    This prevents sending signals based on stale data when market has already reversed.
    """
    try:
        fresh_cvd = compute_cvd(symbol, lookback_ms=1000)

        if verdict == 'BUY':
            if fresh_cvd < -abs(original_cvd) * 0.5:
                print(f'[VALIDATION FAILED] {symbol} BUY: CVD reversed from {original_cvd:,.0f} to {fresh_cvd:,.0f}')
                return False
        else:
            if fresh_cvd > abs(original_cvd) * 0.5:
                print(f'[VALIDATION FAILED] {symbol} SELL: CVD reversed from {original_cvd:,.0f} to {fresh_cvd:,.0f}')
                return False

        cache_key = f"klines_{symbol}_15m_200"
        if cache_key in _API_CACHE:
            del _API_CACHE[cache_key]

        fresh_klines = fetch_klines(symbol, '15m', 200)

        if not fresh_klines or len(fresh_klines) == 0:
            print(f'[VALIDATION WARNING] {symbol}: Could not fetch fresh klines, proceeding cautiously')
            return True

        fresh_price = fresh_klines[-1][4]
        print(f'[VALIDATION OK] {symbol} {verdict}: Fresh CVD={fresh_cvd:,.0f}, Price=${fresh_price:,.2f}')
        return True

    except Exception as e:
        print(f'[VALIDATION ERROR] {symbol}: {e} - Aborting signal to be safe')
        return False


def compute_volume_spike(kl, w, mult):
    vs = [float(k[5]) * float(k[4]) for k in kl[-w:]]
    if len(vs) < 5:
        return False, 0.0, 0.0
    med = float(np.median(vs[:-1]))
    return vs[-1] >= med * mult, vs[-1], med


def compute_vwap_sigma(kl, w, use_quote_volume=True, use_typical_price=True):
    """Calculate VWAP and weighted standard deviation (sigma)."""
    CLOSE_IDX = 4
    BASEVOL_IDX = 5
    QUOTEVOL_IDX = 7

    if not kl or len(kl) < 1:
        return 0.0, 0.0

    recent = kl[-w:] if len(kl) >= w else kl

    if use_typical_price:
        c = np.array(
            [
                (float(k[2]) + float(k[3]) + float(k[CLOSE_IDX])) / 3.0
                for k in recent
            ],
            dtype=float,
        )
    else:
        c = np.array([float(k[CLOSE_IDX]) for k in recent], dtype=float)

    if use_quote_volume and len(kl[0]) > QUOTEVOL_IDX:
        v = np.array([max(float(k[QUOTEVOL_IDX]), 1e-8) for k in recent], dtype=float)
    else:
        v = np.array([max(float(k[BASEVOL_IDX]), 1e-8) for k in recent], dtype=float)

    vol_sum = v.sum()
    if vol_sum <= 0 or len(c) == 0:
        vw = float(c.mean()) if len(c) > 0 else 0.0
        sigma = 0.0
        return vw, sigma

    vw = float((c * v).sum() / vol_sum)

    variance = float(((c - vw) ** 2 * v).sum() / vol_sum)
    sigma = float(np.sqrt(max(variance, 0.0)))

    return vw, sigma


def compute_rsi(klines, period=14):
    if klines is None or len(klines) < period + 1:
        return None

    closes = [float(k[4]) for k in klines]
    deltas = np.diff(closes)

    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0

    for delta in deltas[period:]:
        up = (up * (period - 1) + max(delta, 0)) / period
        down = (down * (period - 1) + max(-delta, 0)) / period
        rs = up / down if down != 0 else 0

    rsi = 100 - (100 / (1 + rs)) if down != 0 else 100
    return rsi


def compute_ema(closes, period):
    if closes is None or len(closes) < period:
        return None

    ema = closes[0]
    multiplier = 2 / (period + 1)

    for price in closes[1:]:
        ema = (price - ema) * multiplier + ema

    return ema


def compute_ema_crossover(klines, short_period=5, long_period=20):
    closes = [float(k[4]) for k in klines]

    ema_short = compute_ema(closes, short_period)
    ema_long = compute_ema(closes, long_period)

    if ema_short is None or ema_long is None:
        return None, None, False, False

    prev_short = compute_ema(closes[:-1], short_period)
    prev_long = compute_ema(closes[:-1], long_period)

    if prev_short is None or prev_long is None:
        return ema_short, ema_long, False, False

    cross_up = prev_short < prev_long and ema_short > ema_long
    cross_down = prev_short > prev_long and ema_short < ema_long

    return ema_short, ema_long, cross_up, cross_down


def detect_regime_hybrid(price, vwap, ema_short, ema_long, dev_sigma, adx, comp):
    if price is None or vwap is None:
        return 'neutral'

    strong_trend = adx is not None and adx >= 50
    moderate_trend = adx is not None and adx >= 25

    if price > vwap and ema_short and ema_long and ema_short > ema_long:
        if dev_sigma > 2.5 or strong_trend:
            regime = 'strong_bull'
        elif dev_sigma > 1.5 or moderate_trend:
            regime = 'bull_warning'
        else:
            regime = 'bull'
    elif price < vwap and ema_short and ema_long and ema_short < ema_long:
        if dev_sigma > 2.5 or strong_trend:
            regime = 'strong_bear'
        elif dev_sigma > 1.5 or moderate_trend:
            regime = 'bear_warning'
        else:
            regime = 'bear'
    else:
        regime = 'sideways'

    comp['regime'] = regime
    return regime


def compute_adx(klines, period=14):
    if klines is None or len(klines) < period + 1:
        return None

    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    closes = [float(k[4]) for k in klines]

    tr_list = []
    for i in range(1, len(klines)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
        tr_list.append(tr)

    tr_sma = np.mean(tr_list[-period:]) if len(tr_list) >= period else None
    if tr_sma is None or tr_sma == 0:
        return None

    plus_dm_list = []
    minus_dm_list = []
    for i in range(1, len(klines)):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]

        plus_dm = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm = down_move if down_move > up_move and down_move > 0 else 0

        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)

    if len(plus_dm_list) < period or len(minus_dm_list) < period:
        return None

    plus_di = (np.mean(plus_dm_list[-period:]) / tr_sma) * 100
    minus_di = (np.mean(minus_dm_list[-period:]) / tr_sma) * 100

    if plus_di + minus_di == 0:
        return None

    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = np.mean([dx for _ in range(period)])

    return adx


def detect_strict_vwap_cross(klines, vwap, lookback=2):
    if klines is None or len(klines) < lookback + 1:
        return False, False

    closes = [float(k[4]) for k in klines[-(lookback + 1):]]
    above = closes[-1] > vwap and closes[-2] <= vwap
    below = closes[-1] < vwap and closes[-2] >= vwap

    return above, below


def summarize_liquidations(liq_data, lookback_ms):
    if liq_data is None:
        return {'long_count': 0, 'short_count': 0, 'long_usd': 0.0, 'short_usd': 0.0}

    long_usd = float(liq_data.get('long_usd', 0.0))
    short_usd = float(liq_data.get('short_usd', 0.0))
    long_count = int(liq_data.get('long_count', 0))
    short_count = int(liq_data.get('short_count', 0))

    ratio = long_usd / (short_usd + 1e-8)
    long_dominance = ratio > 1.2
    short_dominance = ratio < 0.8

    return {
        'long_usd': long_usd,
        'short_usd': short_usd,
        'long_count': long_count,
        'short_count': short_count,
        'ratio': ratio,
        'long_dominance': long_dominance,
        'short_dominance': short_dominance
    }


def calculate_atr(klines, period=14):
    if klines is None or len(klines) < period + 1:
        return None

    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    closes = [float(k[4]) for k in klines]

    trs: List[float] = []
    for i in range(1, len(klines)):
        high_low = highs[i] - lows[i]
        high_close_prev = abs(highs[i] - closes[i - 1])
        low_close_prev = abs(lows[i] - closes[i - 1])
        tr = max(high_low, high_close_prev, low_close_prev)
        trs.append(tr)

    if len(trs) < period:
        return None

    atr = sum(trs[-period:]) / period
    return atr


def calculate_volatility_based_interval(symbol, atr, price):
    if atr is None or price == 0:
        return '15m'

    atr_pct = atr / price * 100

    if atr_pct > 1.5:
        return '5m'
    elif atr_pct > 1.0:
        return '15m'
    elif atr_pct > 0.5:
        return '30m'
    else:
        return '1h'


__all__ = [
    'COINALYZE_API',
    'COINALYZE_KEY',
    'fetch_klines',
    'fetch_agg_trades',
    'fetch_open_interest',
    'fetch_open_interest_hist',
    'fetch_liquidations',
    'fetch_basis',
    'fetch_uif_snapshot',
    'fetch_funding_rate',
    'compute_cvd',
    'compute_volume_spike',
    'compute_vwap_sigma',
    'compute_rsi',
    'compute_ema',
    'compute_ema_crossover',
    'detect_regime_hybrid',
    'compute_adx',
    'detect_strict_vwap_cross',
    'summarize_liquidations',
    'calculate_atr',
    'calculate_volatility_based_interval',
    'validate_signal_momentum_fresh',
]
