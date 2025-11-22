import time, requests, numpy as np, os, pandas as pd
from dotenv import load_dotenv
load_dotenv()
COINALYZE_API='https://api.coinalyze.net/v1'
COINALYZE_KEY=os.getenv('COINALYZE_API_KEY')

# Order Flow Indicators (Nov 15, 2025)
from order_flow_indicators import calculate_bid_ask_aggression, detect_psychological_levels

# Simple in-memory cache to reduce API calls (2.5-minute TTL)
_API_CACHE = {}
_CACHE_TTL = 150  # seconds

# Confidence calibration cache (reload every 5 minutes)
_CALIBRATION_CACHE = {}
_CALIBRATION_TTL = 300  # 5 minutes

# Rate-limited warning timestamps (UIF-30: prevent log spam)
_WARN_LOG_TIMESTAMPS = {}

def _warn_once_per_minute(key, message):
    """Log warning message at most once per minute to prevent spam"""
    now = time.time()
    last_warn = _WARN_LOG_TIMESTAMPS.get(key, 0)
    if now - last_warn >= 60:
        print(f"[WARN] {message}")
        _WARN_LOG_TIMESTAMPS[key] = now

def _get(u,p=None,t=20,retries=5):
    """GET request with retry logic for 429 rate limit errors"""
    p=p or {}
    if COINALYZE_KEY: p['api_key']=COINALYZE_KEY
    
    for attempt in range(retries):
        try:
            r=requests.get(u,params=p,timeout=t)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            # Handle 429 rate limit with longer exponential backoff
            if e.response.status_code == 429 and attempt < retries - 1:
                wait_time = (2 ** attempt) * 1.5  # 1.5s, 3s, 6s, 12s, 24s
                time.sleep(wait_time)
                continue
            raise
    
    # Should never reach here, but just in case
    raise Exception(f"Failed after {retries} retries")

def _symbol_to_coinalyze(s):
    symbol_map={
        'BTCUSDT':'BTCUSDT_PERP.A',
        'ETHUSDT':'ETHUSDT_PERP.A',
        'BNBUSDT':'BNBUSDT_PERP.A',
        'SOLUSDT':'SOLUSDT_PERP.A',
        'AVAXUSDT':'AVAXUSDT_PERP.A',
        'DOGEUSDT':'DOGEUSDT_PERP.A',
        'LINKUSDT':'LINKUSDT_PERP.A',
        'YFIUSDT':'YFIUSDT_PERP.A',
        'LUMIAUSDT':'LUMIAUSDT_PERP.A',
        'ANIMEUSDT':'ANIMEUSDT_PERP.A'
    }
    return symbol_map.get(s, f"{s}_PERP.A")

def fetch_klines(s,i,l=200):
    """Fetch klines with 1-minute caching to reduce API calls"""
    cache_key = f"klines_{s}_{i}_{l}"
    now = time.time()
    
    # Check cache first
    if cache_key in _API_CACHE:
        cached_data, cached_time = _API_CACHE[cache_key]
        if now - cached_time < _CACHE_TTL:
            return cached_data
    
    # Cache miss or expired - fetch from API
    interval_map={'1m':'1min','3m':'3min','5m':'5min','15m':'15min','30m':'30min','1h':'1hour','2h':'2hour','4h':'4hour','6h':'6hour','12h':'12hour','1d':'daily'}
    iv=interval_map.get(i,'15min')
    to_ts=int(time.time())
    minutes={'1min':1,'3min':3,'5min':5,'15min':15,'30min':30,'1hour':60,'2hour':120,'4hour':240,'6hour':360,'12hour':720,'daily':1440}
    from_ts=to_ts-(l*minutes.get(iv,15)*60)
    sym=_symbol_to_coinalyze(s)
    data=_get(f"{COINALYZE_API}/ohlcv-history",{'symbols':sym,'interval':iv,'from':from_ts,'to':to_ts})
    if not data or not isinstance(data,list) or not data[0].get('history'): 
        result = []
    else:
        hist=data[0]['history']
        result = [[int(h['t'])*1000,float(h['o']),float(h['h']),float(h['l']),float(h['c']),float(h.get('v',0)),int(h['t'])*1000] for h in hist]
    
    # Cache the result
    _API_CACHE[cache_key] = (result, now)
    return result

def fetch_agg_trades(s,l=1000):
    return []

def fetch_open_interest(s):
    """Fetch OI with 1-minute caching to reduce API calls"""
    cache_key = f"oi_{s}"
    now = time.time()
    
    # Check cache first
    if cache_key in _API_CACHE:
        cached_data, cached_time = _API_CACHE[cache_key]
        if now - cached_time < _CACHE_TTL:
            return cached_data
    
    # Cache miss or expired - fetch from API
    sym=_symbol_to_coinalyze(s)
    data=_get(f"{COINALYZE_API}/open-interest",{'symbols':sym,'convert_to_usd':'true'})
    if data and isinstance(data,list) and len(data)>0:
        result = float(data[0].get('value',0))
    else:
        result = 0.0
    
    # Cache the result
    _API_CACHE[cache_key] = (result, now)
    return result

def fetch_open_interest_hist(s,p='5min',l=30):
    """Fetch OI history with 1-minute caching to reduce API calls"""
    cache_key = f"oi_hist_{s}_{p}_{l}"
    now = time.time()
    
    # Check cache first
    if cache_key in _API_CACHE:
        cached_data, cached_time = _API_CACHE[cache_key]
        if now - cached_time < _CACHE_TTL:
            return cached_data
    
    # Cache miss or expired - fetch from API
    try:
        sym=_symbol_to_coinalyze(s)
        to_ts=int(time.time())
        from_ts=to_ts-(l*5*60)
        data=_get(f"{COINALYZE_API}/open-interest-history",{'symbols':sym,'interval':'5min','from':from_ts,'to':to_ts,'convert_to_usd':'true'})
        if data and isinstance(data,list) and data[0].get('history'):
            result = [float(h['c']) for h in data[0]['history']]
        else:
            result = []
    except:
        result = []
    
    # Cache the result
    _API_CACHE[cache_key] = (result, now)
    return result

def fetch_liquidations(s,st=None,en=None,l=1000):
    """
    Read liquidation data from liquidation_service data file.
    Returns liquidation summary from Binance WebSocket stream.
    """
    try:
        import json
        from pathlib import Path
        liq_file = Path('liquidation_data.json')
        if not liq_file.exists():
            return {'long_count': 0, 'short_count': 0, 'long_usd': 0.0, 'short_usd': 0.0}
        
        with open(liq_file, 'r') as f:
            data = json.load(f)
        
        # Check if data is stale (older than 5 minutes)
        last_update = data.get('last_update', 0)
        if time.time() - last_update > 300:
            return {'long_count': 0, 'short_count': 0, 'long_usd': 0.0, 'short_usd': 0.0}
        
        # Return liquidation summary for this symbol
        liquidations = data.get('liquidations', {})
        return liquidations.get(s, {'long_count': 0, 'short_count': 0, 'long_usd': 0.0, 'short_usd': 0.0})
    except Exception as e:
        return {'long_count': 0, 'short_count': 0, 'long_usd': 0.0, 'short_usd': 0.0}

# TAAPI removed - using local VWAP calculation only (more reliable and free)

def fetch_basis(symbol):
    """
    Fetch basis_pct from feeds_snapshot.json (written by Data Feeds Service).
    Implements 5-second cache and freshness validation.
    
    Returns:
        tuple: (basis_pct: float|None, age_sec: int|None)
               - (None, None) if snapshot missing/corrupt
               - (None, age_sec) if stale (age > 120s)
               - (basis_pct, age_sec) if fresh and valid
    """
    global _API_CACHE
    cache_key = 'basis_snapshot'
    now = time.time()
    
    # Check 5-second cache
    if cache_key in _API_CACHE:
        cached_snapshot, cached_time = _API_CACHE[cache_key]
        if now - cached_time < 5:  # 5-second cache
            # Use cached snapshot
            snapshot = cached_snapshot
        else:
            # Cache expired, reload
            snapshot = None
    else:
        snapshot = None
    
    # Load snapshot if not cached
    if snapshot is None:
        try:
            import json
            from pathlib import Path
            snapshot_path = Path('data/feeds_snapshot.json')
            
            if not snapshot_path.exists():
                _warn_once_per_minute('basis_snapshot_missing', 
                    f"Basis snapshot not found: {snapshot_path}")
                return None, None
            
            with open(snapshot_path, 'r') as f:
                snapshot = json.load(f)
            
            # Cache for 5 seconds
            _API_CACHE[cache_key] = (snapshot, now)
            
        except Exception as e:
            _warn_once_per_minute('basis_snapshot_read_error', 
                f"Failed to read basis snapshot: {e}")
            return None, None
    
    # Extract symbol data
    try:
        symbol_data = snapshot.get('symbols', {}).get(symbol)
        if not symbol_data:
            return None, None
        
        basis_pct = symbol_data.get('basis_pct')
        updated_ts = symbol_data.get('updated', 0)
        
        # Calculate age
        age_sec = int(now - updated_ts) if updated_ts > 0 else 999
        
        # Check freshness (120s = 4 cycles of 30s)
        if age_sec > 120:
            _warn_once_per_minute(f'basis_stale_{symbol}', 
                f"Basis data stale for {symbol}: {age_sec}s old (threshold: 120s)")
            return None, age_sec
        
        # Return valid data
        return basis_pct, age_sec
        
    except Exception as e:
        _warn_once_per_minute('basis_extract_error', 
            f"Failed to extract basis for {symbol}: {e}")
        return None, None

def fetch_uif_snapshot(symbol):
    """
    Fetch UIF features from uif_snapshot.json (written by UIF Feature Engine).
    Implements 5-second cache and freshness validation.
    
    Returns 8 UIF features:
        - adx14: ADX(14) directional strength
        - psar: Parabolic SAR state (+1 bullish, -1 bearish)
        - momentum5: 5-minute price momentum
        - vol_accel: Volume acceleration
        - zcvd: Normalized CVD (from analysis_log placeholder)
        - doi_pct: OI change percentage (from analysis_log placeholder)
        - dev_sigma: VWAP deviation sigma (from analysis_log placeholder)
        - rsi_dist: RSI distance from 50 (from analysis_log placeholder)
    
    Returns:
        tuple: (uif_dict: dict|None, age_sec: int|None)
               - (None, None) if snapshot missing/corrupt
               - (None, age_sec) if stale (age > 120s)
               - (uif_dict, age_sec) if fresh and valid
    """
    global _API_CACHE
    cache_key = 'uif_snapshot'
    now = time.time()
    
    # Check 5-second cache
    if cache_key in _API_CACHE:
        cached_snapshot, cached_time = _API_CACHE[cache_key]
        if now - cached_time < 5:
            snapshot = cached_snapshot
        else:
            snapshot = None
    else:
        snapshot = None
    
    # Load snapshot if not cached
    if snapshot is None:
        try:
            import json
            from pathlib import Path
            snapshot_path = Path('data/uif_snapshot.json')
            
            if not snapshot_path.exists():
                _warn_once_per_minute('uif_snapshot_missing',
                    f"UIF snapshot not found: {snapshot_path}")
                return None, None
            
            with open(snapshot_path, 'r') as f:
                snapshot = json.load(f)
            
            # Cache for 5 seconds
            _API_CACHE[cache_key] = (snapshot, now)
            
        except Exception as e:
            _warn_once_per_minute('uif_snapshot_read_error',
                f"Failed to read UIF snapshot: {e}")
            return None, None
    
    # Extract symbol data
    try:
        symbol_data = snapshot.get('symbols', {}).get(symbol)
        if not symbol_data:
            return None, None
        
        updated_ts = symbol_data.get('updated', 0)
        
        # Calculate age
        age_sec = int(now - updated_ts) if updated_ts > 0 else 999
        
        # Check freshness (120s = 24 cycles of 5m)
        if age_sec > 120:
            _warn_once_per_minute(f'uif_stale_{symbol}',
                f"UIF data stale for {symbol}: {age_sec}s old (threshold: 120s)")
            return None, age_sec
        
        # Extract 4 UIF features (others will be placeholders from analysis_log)
        uif_dict = {
            'adx14': symbol_data.get('adx14', 0.0),
            'psar': symbol_data.get('psar_state', 0),
            'momentum5': symbol_data.get('momentum5', 0.0),
            'vol_accel': symbol_data.get('vol_accel', 0.0),
            # Placeholders for features calculated elsewhere
            'zcvd': 0.0,
            'doi_pct': 0.0,
            'dev_sigma': 0.0,
            'rsi_dist': 0.0
        }
        
        return uif_dict, age_sec
        
    except Exception as e:
        _warn_once_per_minute('uif_extract_error',
            f"Failed to extract UIF for {symbol}: {e}")
        return None, None

def fetch_funding_rate(symbol):
    """
    Fetch current funding rate using OKX API (geolocation-free alternative to Binance).
    
    Funding Rate interpretation:
    - Positive rate: Longs pay shorts (market is bullish-biased) → Contrarian SELL signal
    - Negative rate: Shorts pay longs (market is bearish-biased) → Contrarian BUY signal
    - Typical range: -0.05% to +0.05%
    - Extreme levels: >0.10% or <-0.10% indicate strong bias
    
    Returns:
        float: Current funding rate (e.g., 0.0001 = 0.01%)
    """
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
    """
    Read CVD (Cumulative Volume Delta) from cvd_service data file.
    Returns 0.0 if cvd_service is not running or data is unavailable.
    """
    try:
        import json
        from pathlib import Path
        cvd_file = Path('cvd_data.json')
        if not cvd_file.exists():
            return 0.0
        
        with open(cvd_file, 'r') as f:
            data = json.load(f)
        
        # Check if data is stale (older than 5 minutes)
        last_update = data.get('last_update', 0)
        if time.time() - last_update > 300:
            return 0.0
        
        # Return CVD for this symbol
        cvd_values = data.get('cvd', {})
        return float(cvd_values.get(symbol, 0.0))
    except Exception as e:
        return 0.0

def validate_signal_momentum_fresh(symbol, verdict, original_cvd):
    """
    CRITICAL PRE-BROADCAST VALIDATION: Bypass cache and verify signal direction still valid.
    
    This prevents sending signals based on stale data when market has already reversed.
    
    Args:
        symbol: Trading symbol
        verdict: 'BUY' or 'SELL' 
        original_cvd: CVD value when signal was originally generated
        
    Returns:
        True if signal is still valid, False if momentum has reversed
    """
    try:
        # Get FRESH CVD from file (no cache - cvd_service updates every 5 seconds)
        fresh_cvd = compute_cvd(symbol, lookback_ms=1000)
        
        # Check if CVD has reversed direction
        if verdict == 'BUY':
            # BUY signal requires positive momentum
            # Reject if CVD turned significantly negative
            if fresh_cvd < -abs(original_cvd) * 0.5:  # Reversed by >50%
                print(f'[VALIDATION FAILED] {symbol} BUY: CVD reversed from {original_cvd:,.0f} to {fresh_cvd:,.0f}')
                return False
        else:  # SELL
            # SELL signal requires negative momentum  
            # Reject if CVD turned significantly positive
            if fresh_cvd > abs(original_cvd) * 0.5:  # Reversed by >50%
                print(f'[VALIDATION FAILED] {symbol} SELL: CVD reversed from {original_cvd:,.0f} to {fresh_cvd:,.0f}')
                return False
        
        # Also bypass klines cache to get fresh price (clear cache for this symbol)
        cache_key = f"klines_{symbol}_15m_200"
        if cache_key in _API_CACHE:
            del _API_CACHE[cache_key]
        
        # Fetch fresh klines (will hit API, no cache)
        fresh_klines = fetch_klines(symbol, '15m', 200)
        
        if not fresh_klines or len(fresh_klines) == 0:
            print(f'[VALIDATION WARNING] {symbol}: Could not fetch fresh klines, proceeding cautiously')
            # Don't block on klines failure - CVD check is more critical
            return True
        
        fresh_price = fresh_klines[-1][4]  # Last close price
        
        # Log the fresh validation check
        print(f'[VALIDATION OK] {symbol} {verdict}: Fresh CVD={fresh_cvd:,.0f}, Price=${fresh_price:,.2f}')
        return True
        
    except Exception as e:
        print(f'[VALIDATION ERROR] {symbol}: {e} - Aborting signal to be safe')
        # On error, reject signal to be conservative
        return False

def compute_volume_spike(kl, w, mult):
    # Calculate volume in USDT (volume * close price)
    vs=[float(k[5]) * float(k[4]) for k in kl[-w:]]
    if len(vs)<5: return (False,0.0,0.0)
    med=float(np.median(vs[:-1])); return (vs[-1]>=med*mult, vs[-1], med)

def compute_vwap_sigma(kl, w, use_quote_volume=True, use_typical_price=True):
    """
    Calculate VWAP and weighted standard deviation (sigma).
    
    Professional implementation using:
    - Quote volume (USDT) instead of base volume for stability across assets
    - Typical price ((H+L+C)/3) to reduce noise from single tick extremes
    - Weighted variance around VWAP (not simple price std)
    
    Args:
        kl: List of kline data [open_time, open, high, low, close, base_vol, close_time, quote_vol, ...]
        w: Window size (number of candles)
        use_quote_volume: Use dollar volume (index 7) vs base volume (index 5). Default True.
        use_typical_price: Use (H+L+C)/3 vs Close only. Default True.
    
    Returns:
        tuple: (vwap, sigma) where sigma is weighted std around VWAP
    """
    CLOSE_IDX = 4
    BASEVOL_IDX = 5
    QUOTEVOL_IDX = 7
    
    if not kl or len(kl) < 1:
        return 0.0, 0.0
    
    recent = kl[-w:] if len(kl) >= w else kl
    
    # Price: typical price (H+L+C)/3 or close only
    if use_typical_price:
        c = np.array([
            (float(k[2]) + float(k[3]) + float(k[CLOSE_IDX])) / 3.0
            for k in recent
        ], dtype=float)
    else:
        c = np.array([float(k[CLOSE_IDX]) for k in recent], dtype=float)
    
    # Volume: quote volume (USDT) or base volume (BTC/ETH)
    if use_quote_volume and len(kl[0]) > QUOTEVOL_IDX:
        v = np.array([max(float(k[QUOTEVOL_IDX]), 1e-8) for k in recent], dtype=float)
    else:
        v = np.array([max(float(k[BASEVOL_IDX]), 1e-8) for k in recent], dtype=float)
    
    vol_sum = v.sum()
    if vol_sum <= 0 or len(c) == 0:
        vw = float(c.mean()) if len(c) > 0 else 0.0
        sigma = 0.0
        return vw, sigma
    
    # VWAP = Σ(price × volume) / Σ(volume)
    vw = float((c * v).sum() / vol_sum)
    
    # Weighted variance around VWAP (not simple std!)
    # σ² = Σ(volume × (price - VWAP)²) / Σ(volume)
    var = ((v * (c - vw) ** 2).sum() / vol_sum)
    sigma = float(np.sqrt(var))
    
    return vw, sigma

def compute_rsi(klines, period=14):
    """
    Calculate RSI (Relative Strength Index) using Wilder's Smoothing Method.
    
    This matches the industry standard used by Binance, TradingView, and all major platforms.
    
    RSI interpretation:
    - RSI > 70: Overbought (potential SELL signal)
    - RSI < 30: Oversold (potential BUY signal)
    - RSI 30-70: Neutral zone
    
    Method:
    - First average: Simple Moving Average of gains/losses over period
    - Subsequent: Wilder's smoothing: (prev_avg * (period-1) + current) / period
    
    Args:
        klines: List of kline data
        period: RSI period (default 14)
    
    Returns:
        float: RSI value (0-100), or None if insufficient data
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

def compute_ema(closes, period):
    """
    Calculate Exponential Moving Average with proper exponential smoothing.
    
    Uses SMA as seed, then applies EMA formula recursively across full series.
    
    Args:
        closes: Array of closing prices (full series)
        period: EMA period
    
    Returns:
        float: Current EMA value
    """
    if len(closes) < period:
        return None
    
    closes = np.array(closes)
    multiplier = 2 / (period + 1)
    
    # Seed with SMA of first 'period' values
    ema = float(closes[:period].mean())
    
    # Apply EMA formula recursively to remaining values
    for price in closes[period:]:
        ema = (price - ema) * multiplier + ema
    
    return float(ema)

def compute_ema_crossover(klines, short_period=5, long_period=20):
    """
    Calculate EMA crossover signals with proper exponential smoothing.
    
    IMPORTANT: Passes full price history to compute_ema for accurate EMA calculation.
    
    EMA Crossover interpretation:
    - 5-EMA crosses ABOVE 20-EMA: Bullish (BUY signal)
    - 5-EMA crosses BELOW 20-EMA: Bearish (SELL signal)
    
    Args:
        klines: List of kline data
        short_period: Short EMA period (default 5)
        long_period: Long EMA period (default 20)
    
    Returns:
        tuple: (ema_short, ema_long, cross_up, cross_down)
    """
    if not klines or len(klines) < long_period + 2:
        return (None, None, False, False)
    
    closes = [float(k[4]) for k in klines]
    
    # Calculate current EMAs using FULL price series
    ema_short = compute_ema(closes, short_period)
    ema_long = compute_ema(closes, long_period)
    
    # Calculate previous EMAs using series up to -1 (excluding last candle)
    ema_short_prev = compute_ema(closes[:-1], short_period)
    ema_long_prev = compute_ema(closes[:-1], long_period)
    
    if None in [ema_short, ema_long, ema_short_prev, ema_long_prev]:
        return (ema_short, ema_long, False, False)
    
    # Detect crossovers
    cross_up = (ema_short_prev <= ema_long_prev) and (ema_short > ema_long)
    cross_down = (ema_short_prev >= ema_long_prev) and (ema_short < ema_long)
    
    return (ema_short, ema_long, cross_up, cross_down)

def detect_regime_hybrid(price, vwap, ema_short, ema_long, dev_sigma, adx, comp):
    """
    HYBRID EMA+VWAP regime detection for faster reversal detection.
    
    Combines EMA (fast reaction) with VWAP (institutional reference) to detect:
    - strong_bear/strong_bull: Both EMA and VWAP agree (high confidence)
    - bear_warning/bull_warning: EMA sees reversal, VWAP hasn't confirmed yet (EARLY signal)
    - neutral/sideways: No clear trend or conflicting signals
    
    Key innovation: bear_warning/bull_warning modes allow early entry on reversals
    without waiting for full VWAP confirmation, improving signal timing by 2-4 minutes.
    
    Args:
        price: Current close price
        vwap: VWAP value
        ema_short: Short EMA (5-bar)
        ema_long: Long EMA (20-bar)
        dev_sigma: Price deviation from VWAP in standard deviations
        adx: Average Directional Index (trend strength)
        comp: Components dict with Price_below_VWAP, Price_above_VWAP, EMA_bearish, etc.
    
    Returns:
        str: Regime name (strong_bear, bear_warning, strong_bull, bull_warning, neutral, sideways)
    """
    # EMA trend detection
    ema_bearish = comp.get('EMA_bearish', False)  # EMA_short < EMA_long
    ema_bullish = (ema_short is not None and ema_long is not None and ema_short > ema_long)
    
    # VWAP position
    price_below_vwap = comp.get('Price_below_VWAP', False)
    price_above_vwap = comp.get('Price_above_VWAP', False)
    
    # EMA convergence - EMAs close together indicates potential reversal
    ema_converging = False
    if ema_short and ema_long:
        ema_diff_pct = abs(ema_short - ema_long) / ema_long
        ema_converging = (ema_diff_pct < 0.003)  # <0.3% difference = converging
    
    # Price position relative to EMA_short (faster than VWAP)
    price_below_ema = (price < ema_short) if ema_short else False
    price_above_ema = (price > ema_short) if ema_short else False
    
    # REGIME DETECTION LOGIC:
    # Priority order: sideways (weak trend) > warning regimes (early signals) > strong regimes > neutral
    # Warning regimes checked first because they're more specific (require multiple conditions)
    
    # SIDEWAYS: ADX indicates weak trend (check first - overrides other signals)
    if adx is not None and adx < 20:
        return 'sideways'
    
    # BEAR WARNING: Early bearish signal before VWAP confirms (check before strong_bear)
    # Case 1: Price dropped below EMA while in bullish EMA trend + EMAs converging (reversal starting)
    # Case 2: EMA bearish but price still above VWAP (VWAP lagging)
    elif (price_below_ema and ema_bullish and ema_converging) or \
         (ema_bearish and price_above_vwap):
        return 'bear_warning'
    
    # BULL WARNING: Early bullish signal before VWAP confirms (check before strong_bull)
    # Case 1: Price above EMA while in bearish EMA trend + EMAs converging (reversal starting)
    # Case 2: EMA bullish but price still below VWAP (VWAP lagging)
    elif (price_above_ema and ema_bearish and ema_converging) or \
         (ema_bullish and price_below_vwap):
        return 'bull_warning'
    
    # STRONG BEAR: EMA AND VWAP both confirm bearish trend
    elif ema_bearish and price_below_vwap:
        return 'strong_bear'
    
    # STRONG BULL: EMA AND VWAP both confirm bullish trend
    elif ema_bullish and price_above_vwap:
        return 'strong_bull'
    
    # NEUTRAL: All other cases (conflicting signals or insufficient data)
    else:
        return 'neutral'

def compute_adx(klines, period=14):
    """
    Calculate Average Directional Index (ADX) - measures trend strength.
    
    ADX interpretation:
    - ADX > 25: Strong trend (use trend-following strategies)
    - ADX < 25: Weak/no trend (use mean-reversion strategies)
    - ADX > 50: Very strong trend
    - ADX < 20: Very weak trend
    
    Args:
        klines: List of kline data
        period: ADX period (default 14)
    
    Returns:
        float: ADX value (0-100), or None if insufficient data
    """
    min_required = period * 3
    if not klines or len(klines) < min_required:
        return None
    
    try:
        highs = np.array([float(k[2]) for k in klines])
        lows = np.array([float(k[3]) for k in klines])
        closes = np.array([float(k[4]) for k in klines])
        
        high_diff = np.diff(highs)
        low_diff = -np.diff(lows)
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        high_low = highs[1:] - lows[1:]
        high_close = np.abs(highs[1:] - closes[:-1])
        low_close = np.abs(lows[1:] - closes[:-1])
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        
        true_range = np.maximum(true_range, 1e-10)
        
        def wilder_smooth(data, period):
            if len(data) < period:
                return np.array([])
            data = np.array(data)
            data = np.where(np.isnan(data) | np.isinf(data), 0, data)
            smoothed = []
            sma = np.mean(data[:period])
            if np.isnan(sma) or np.isinf(sma):
                sma = 0.0
            smoothed.append(sma)
            for i in range(period, len(data)):
                smoothed_value = (smoothed[-1] * (period - 1) + data[i]) / period
                if np.isnan(smoothed_value) or np.isinf(smoothed_value):
                    smoothed_value = smoothed[-1]
                smoothed.append(smoothed_value)
            return np.array(smoothed)
        
        atr = wilder_smooth(true_range, period)
        smoothed_plus_dm = wilder_smooth(plus_dm, period)
        smoothed_minus_dm = wilder_smooth(minus_dm, period)
        
        if len(atr) == 0 or len(smoothed_plus_dm) == 0 or len(smoothed_minus_dm) == 0:
            return None
        
        min_len = min(len(atr), len(smoothed_plus_dm), len(smoothed_minus_dm))
        atr = atr[:min_len]
        smoothed_plus_dm = smoothed_plus_dm[:min_len]
        smoothed_minus_dm = smoothed_minus_dm[:min_len]
        
        atr = np.maximum(atr, 1e-10)
        
        plus_di = 100 * smoothed_plus_dm / atr
        minus_di = 100 * smoothed_minus_dm / atr
        
        di_sum = plus_di + minus_di
        di_sum = np.maximum(di_sum, 1e-10)
        
        dx = 100 * np.abs(plus_di - minus_di) / di_sum
        
        dx = np.where(np.isnan(dx) | np.isinf(dx), 0, dx)
        
        adx_values = wilder_smooth(dx, period)
        
        if len(adx_values) == 0:
            return None
        
        last_adx = float(adx_values[-1])
        
        if np.isnan(last_adx) or np.isinf(last_adx):
            return 0.0
        
        return max(0.0, min(100.0, last_adx))
    
    except Exception as e:
        return None

def detect_strict_vwap_cross(klines, vwap, lookback=2):
    """
    Detect strict two-point VWAP cross (previous vs current candle).
    
    This ensures we only trigger on actual crossovers, not just price being above/below VWAP.
    
    Args:
        klines: List of kline data
        vwap: VWAP value
        lookback: Number of candles to check (default 2 for strict two-point)
    
    Returns:
        tuple: (vwap_cross_up, vwap_cross_down)
    """
    if not klines or len(klines) < 2 or vwap is None or vwap <= 0:
        return (False, False)
    
    prev_close = float(klines[-2][4])
    curr_close = float(klines[-1][4])
    
    # Strict two-point cross: previous candle on one side, current candle on opposite side
    vwap_cross_up = (prev_close < vwap) and (curr_close >= vwap)
    vwap_cross_down = (prev_close > vwap) and (curr_close <= vwap)
    
    return (vwap_cross_up, vwap_cross_down)

def summarize_liquidations(liq_data, lookback_ms):
    """
    Return liquidation data from liquidation service.
    liq_data is already aggregated from liquidation_service.py
    """
    if isinstance(liq_data, dict) and 'long_count' in liq_data:
        return liq_data
    return {'long_count': 0, 'short_count': 0, 'long_usd': 0.0, 'short_usd': 0.0}

def calculate_confluence_score(components, weights, direction='BUY'):
    """
    PROFESSIONAL CONFLUENCE-BASED SCORING (matches $10M+ trading firms).
    
    Instead of summing all weighted indicators, this checks for CONFLUENCE:
    - PRIMARY SIGNALS (must have 2+ aligned): CVD, OI, VWAP/Price
    - FILTERS (must ALL pass): RSI extremes preferred, adequate volume, EMA trend confirmation
    - DIRECTIONAL BLOCKING: Opposing indicators (OI_up for SELL, OI_down for BUY) block signals
    
    NOTE: Volume removed from primary signals (negative -0.48 correlation with wins)
    
    This prevents weak signals from passing by requiring strong alignment across
    multiple independent indicators.
    
    Args:
        components: Dict of boolean component indicators
        weights: Dict of weights per indicator (used for confidence scaling)
        direction: 'BUY' or 'SELL'
    
    Returns:
        Tuple: (has_signal, aligned_count, max_possible, aligned_indicators)
    """
    aligned_indicators = []
    
    if direction == 'BUY':
        # NO HARD VETO: Confluence logic will handle opposing indicators
        # If CVD is negative, it simply won't contribute to primary_checks
        # Signal can still be valid if OI + VWAP align (2 of 3 confluence)
        
        # PRIMARY SIGNALS (need 2+ of 3: CVD, OI, VWAP)
        primary_checks = []
        
        # 1. CVD Signal: Positive CVD = buying pressure
        # If CVD_neg (opposite), it simply doesn't contribute - no hard block
        if components.get('CVD_pos'):
            primary_checks.append(('CVD', weights.get('cvd', 1.0)))
            aligned_indicators.append('CVD+')
        
        # 2. OI Signal: Rising OI = new positions opening
        if components.get('OI_up'):
            primary_checks.append(('OI', weights.get('oi', 1.0)))
            aligned_indicators.append('OI↑')
        
        # 3. VWAP/Price Signal: Price below VWAP (mean reversion BUY) OR crossing up
        if components.get('Price_below_VWAP') or components.get('VWAP_cross_up'):
            primary_checks.append(('VWAP', weights.get('vwap', 1.0)))
            aligned_indicators.append('VWAP')
        
        # 4. Volume Signal: REMOVED from confluence (negative -0.48 correlation with wins)
        # Volume spike still tracked but not required for signal generation
        
        # FILTERS (all must pass to avoid bad trades)
        filter_checks = []
        
        # Filter 1: RSI Oversold (STRICT - data shows 100% WR but only 4 samples)
        # Accept oversold conditions (RSI < 30) OR neutral RSI (30-70)
        # Backtest: RSI < 30 = 100% WR (4W-0L), but blocks 89.5% of BUY signals
        rsi_ok = components.get('RSI_oversold', False) or not components.get('RSI_overbought', False)
        filter_checks.append(('RSI_OK', rsi_ok))
        
        # Filter 2: EMA trend confirmation (SOFTENED - allow strong CVD+OI to override)
        # Only block if EMA_cross_down AND we don't have strong CVD+OI confluence
        # This allows BUY signals during early reversals when CVD/OI lead EMA
        has_strong_confluence = components.get('CVD_pos') and components.get('OI_up')
        ema_ok = not components.get('EMA_cross_down', False) or has_strong_confluence
        filter_checks.append(('EMA_OK', ema_ok))
        if components.get('EMA_cross_up'):
            aligned_indicators.append('EMA↗')
        
        # Filter 3: Volume - MVP SIMPLIFICATION: Always pass (CVD+OI+VWAP are more reliable)
        # Reason: Volume check blocks signals when bot runs between candles (current candle incomplete)
        # Pattern mining showed volume has negative correlation with wins anyway
        volume_adequate = True  # Simplified for MVP stability
        filter_checks.append(('Volume_OK', volume_adequate))
        
    else:  # SELL direction
        # NO HARD VETO: Confluence logic will handle opposing indicators
        # If CVD is positive, it simply won't contribute to primary_checks
        # Signal can still be valid if OI + VWAP align (2 of 3 confluence)
        
        # PRIMARY SIGNALS (need 2+ of 3: CVD, OI, VWAP)
        primary_checks = []
        
        # 1. CVD Signal: Negative CVD = selling pressure
        # If CVD_pos (opposite), it simply doesn't contribute - no hard block
        if components.get('CVD_neg'):
            primary_checks.append(('CVD', weights.get('cvd', 1.0)))
            aligned_indicators.append('CVD-')
        
        # 2. OI Signal: Accept EITHER OI↑ (new shorts) OR OI↓ (position closing)
        # Both can accompany downward moves in different market conditions
        if components.get('OI_up'):
            primary_checks.append(('OI', weights.get('oi', 1.0)))
            aligned_indicators.append('OI↑')
        elif components.get('OI_down'):
            primary_checks.append(('OI', weights.get('oi', 1.0)))
            aligned_indicators.append('OI↓')
        
        # 3. VWAP/Price Signal: Price below VWAP OR crossing down (bearish conditions)
        if components.get('Price_below_VWAP') or components.get('VWAP_cross_down'):
            primary_checks.append(('VWAP', weights.get('vwap', 1.0)))
            aligned_indicators.append('VWAP')
        
        # FILTERS (all must pass)
        filter_checks = []
        
        # Filter 1: RSI Filter - accept any RSI for SELL
        rsi_ok = True
        filter_checks.append(('RSI_OK', rsi_ok))
        
        # Filter 2: EMA trend confirmation
        ema_ok = not components.get('EMA_cross_up', False)  # Avoid selling into uptrend
        filter_checks.append(('EMA_OK', ema_ok))
        if components.get('EMA_cross_down'):
            aligned_indicators.append('EMA↘')
        
        # Filter 3: Volume - MVP SIMPLIFICATION: Always pass (CVD+OI+VWAP are more reliable)
        # Reason: Volume check blocks signals when bot runs between candles (current candle incomplete)
        # Pattern mining showed DOWN movements occur on QUIET volume (blocking is counterproductive)
        volume_adequate = True  # Simplified for MVP stability
        filter_checks.append(('Volume_OK', volume_adequate))
    
    # Calculate confluence
    primary_aligned = len(primary_checks)
    primary_total = 3  # Only 3 primary signals (CVD, OI, VWAP) - volume removed
    all_filters_pass = all(f[1] for f in filter_checks)
    
    # Calculate weighted score (for confidence scaling)
    weighted_score = sum(weight for _, weight in primary_checks)
    max_possible_score = sum(weights.get(k, 1.0) for k in ['cvd', 'oi', 'vwap'])  # Removed 'volume'
    
    # Signal requirements: 2+ primary signals aligned AND all filters pass (relaxed from 3+)
    has_signal = (primary_aligned >= 2) and all_filters_pass
    
    return (has_signal, weighted_score, max_possible_score, aligned_indicators)

def calculate_magnitude_score(cvd, cvd_threshold, oi_change, oi_current, volume_last, volume_median, 
                               price, vwap, atr, direction='BUY'):
    """
    DEPRECATED: This function created double-counting of CVD/OI/Volume.
    Now returns 1.0 (neutral) to avoid duplicating market strength factors.
    
    Market strength (CVD/OI/Volume) is now handled ONLY in multiplier calculation
    in calculate_price_targets() to prevent over-inflation of targets.
    
    TODO: In future, replace with orthogonal factors:
    - ATR expansion (volatility trending up/down)
    - Price momentum (rate of change)
    - Trend strength (ADX-based)
    
    For now, returns 1.0 to eliminate double-counting bug.
    """
    # Return neutral multiplier - market strength handled elsewhere
    return 1.0

def calculate_weighted_score(components, weights, direction='BUY'):
    """
    Calculate weighted score using SIGNED weights based on direction.
    CRITICAL FIX: Each indicator contributes +weight if it supports the direction,
    or -weight if it opposes the direction. This prevents conflicting indicators
    from producing high-confidence signals.
    
    Score = Σ(+weight for supporting indicators, -weight for opposing indicators)
    
    Args:
        components: Dict of boolean component indicators
        weights: Dict of weights per indicator
        direction: 'BUY' or 'SELL' (determines sign of weight)
    
    Returns:
        Weighted score (float, can be negative if opposing indicators dominate)
    """
    score = 0.0
    
    if direction == 'BUY':
        # CVD contribution: positive CVD supports BUY, negative CVD opposes
        if components.get('CVD_pos'):
            score += weights.get('cvd', 1.0)
        elif components.get('CVD_neg'):
            score -= weights.get('cvd', 1.0)
        
        # OI contribution: rising OI supports BUY, falling OI opposes
        # DATA-DRIVEN (Nov 4, 2025): OI has strongest correlation with success (+0.371)
        # Therefore, double the weight for both BUY and SELL signals
        oi_weight = weights.get('oi', 1.0) * 2.0  # 2x multiplier for all directions
        if components.get('OI_up'):
            score += oi_weight
        elif components.get('OI_down'):
            score -= oi_weight
        
        # VWAP contribution: price below VWAP or crossing up supports BUY
        if components.get('VWAP_cross_up') or components.get('Price_below_VWAP'):
            score += weights.get('vwap', 1.0)
        elif components.get('VWAP_cross_down') or components.get('Price_above_VWAP'):
            score -= weights.get('vwap', 1.0)
        
        # Volume contribution: spike supports both directions (neutral for now)
        if components.get('Vol_spike'):
            score += weights.get('volume', 1.0)
        
        # Liquidations contribution: shorts liquidating supports BUY
        if components.get('Liq_short'):
            score += weights.get('liquidations', 1.0)
        elif components.get('Liq_long'):
            score -= weights.get('liquidations', 1.0)
        
        # Funding Rate contribution: negative rate (shorts pay longs) supports BUY
        if components.get('Funding_negative'):
            score += weights.get('funding', 1.0)
        elif components.get('Funding_positive'):
            score -= weights.get('funding', 1.0)
        
        # RSI contribution: oversold (RSI < 30) supports BUY, overbought (RSI > 70) opposes
        if components.get('RSI_oversold'):
            score += weights.get('rsi', 1.0)
        elif components.get('RSI_overbought'):
            score -= weights.get('rsi', 1.0)
        
        # EMA crossover contribution: cross up supports BUY, cross down opposes
        if components.get('EMA_cross_up'):
            score += weights.get('ema', 1.0)
        elif components.get('EMA_cross_down'):
            score -= weights.get('ema', 1.0)
        
        # Basis_pct contribution: negative basis (discount) supports BUY, positive basis (premium) opposes
        # UIF-30 integration (Nov 11, 2025): AUROC=0.839, Lift=1.79x
        # CONTRARIAN SIGN HANDLING: Invert sign so negative basis_pct (discount) adds to BUY score
        basis_score_raw = components.get('basis_score', 0.0)
        if basis_score_raw != 0.0:
            # Example: basis_pct=-0.10 → basis_score_raw=-0.01 → score -= (-0.01) = +0.01 (bullish ✓)
            #          basis_pct=+0.10 → basis_score_raw=+0.01 → score -= (+0.01) = -0.01 (bearish ✓)
            score -= basis_score_raw  # Invert sign for contrarian behavior
    
    else:  # SELL direction
        # CVD contribution: negative CVD supports SELL, positive CVD opposes
        if components.get('CVD_neg'):
            score += weights.get('cvd', 1.0)
        elif components.get('CVD_pos'):
            score -= weights.get('cvd', 1.0)
        
        # OI contribution: falling OI supports SELL, rising OI opposes
        # DATA-DRIVEN (Nov 4, 2025): OI has strongest correlation with success (+0.371)
        # Therefore, double the weight for both BUY and SELL signals
        oi_weight = weights.get('oi', 1.0) * 2.0  # 2x multiplier for all directions
        if components.get('OI_down'):
            score += oi_weight
        elif components.get('OI_up'):
            score -= oi_weight
        
        # VWAP contribution: price above VWAP or crossing down supports SELL
        if components.get('VWAP_cross_down') or components.get('Price_above_VWAP'):
            score += weights.get('vwap', 1.0)
        elif components.get('VWAP_cross_up') or components.get('Price_below_VWAP'):
            score -= weights.get('vwap', 1.0)
        
        # Volume contribution: spike supports both directions (neutral for now)
        if components.get('Vol_spike'):
            score += weights.get('volume', 1.0)
        
        # Liquidations contribution: longs liquidating supports SELL
        if components.get('Liq_long'):
            score += weights.get('liquidations', 1.0)
        elif components.get('Liq_short'):
            score -= weights.get('liquidations', 1.0)
        
        # Funding Rate contribution: positive rate (longs pay shorts) supports SELL
        if components.get('Funding_positive'):
            score += weights.get('funding', 1.0)
        elif components.get('Funding_negative'):
            score -= weights.get('funding', 1.0)
        
        # RSI contribution: overbought (RSI > 70) supports SELL, oversold (RSI < 30) opposes
        # DATA-DRIVEN (Nov 4, 2025): RSI has very strong correlation with SELL profit (+0.554, p<0.001)
        # Therefore, double the weight for SELL signals
        rsi_weight = weights.get('rsi', 1.0) * 2.0  # 2x multiplier for SELL
        if components.get('RSI_overbought'):
            score += rsi_weight
        elif components.get('RSI_oversold'):
            score -= rsi_weight
        
        # EMA crossover contribution: cross down supports SELL, cross up opposes
        if components.get('EMA_cross_down'):
            score += weights.get('ema', 1.0)
        elif components.get('EMA_cross_up'):
            score -= weights.get('ema', 1.0)
        
        # Basis_pct contribution: positive basis (premium) supports SELL, negative basis (discount) opposes
        # UIF-30 integration (Nov 11, 2025): AUROC=0.839, Lift=1.79x
        # CONTRARIAN SIGN HANDLING: Keep sign so positive basis_pct (premium) adds to SELL score
        basis_score_raw = components.get('basis_score', 0.0)
        if basis_score_raw != 0.0:
            # Example: basis_pct=+0.10 → basis_score_raw=+0.01 → score += (+0.01) = +0.01 (bearish ✓)
            #          basis_pct=-0.10 → basis_score_raw=-0.01 → score += (-0.01) = -0.01 (bullish opposes ✓)
            score += basis_score_raw  # Keep sign for contrarian behavior
    
    return score

def calculate_confidence_from_score(score, min_score, max_score, min_confidence=0.70, verdict='BUY'):
    """
    Calculate confidence based on weighted score using RECALIBRATED formula (Nov 3, 2025).
    
    RECALIBRATION RATIONALE (based on 346 signals over 7 days):
    - OLD formula: confidence 70-95% → actual WR 32-73% (deviation -30 to -50%)
    - BUY signals: actual WR 32.5% (was promised 70-95%)
    - SELL signals: actual WR 50.0% (was promised 70-95%)
    
    NEW REALISTIC RANGES (calibrated to empirical data):
    - BUY:  [0.25, 0.65] (reflects actual 32.5% WR, with headroom for strong signals)
    - SELL: [0.40, 0.80] (reflects actual 50.0% WR, with headroom for strong signals)
    
    This ensures confidence values match realistic win rate expectations.
    
    Args:
        score: Actual weighted score
        min_score: Minimum score threshold (e.g., 60% of max_score)
        max_score: Maximum possible score (sum of all weights)
        min_confidence: Minimum confidence level (default 0.70, ignored - using calibrated ranges)
        verdict: 'BUY' or 'SELL' (determines confidence range)
    
    Returns:
        Confidence level calibrated to actual performance:
        - BUY: 0.25 to 0.65
        - SELL: 0.40 to 0.80
    """
    if score < min_score:
        return 0.0  # No signal
    
    if max_score <= min_score:
        # Edge case: use minimum confidence for direction
        return 0.25 if verdict == 'BUY' else 0.40
    
    # Calculate score margin (0.0 to 1.0)
    score_margin = (score - min_score) / (max_score - min_score)
    
    # Apply direction-specific calibrated ranges
    if verdict == 'SELL':
        # SELL: 40% to 80% (range of 40%)
        # Reflects empirical SELL WR of 50% with headroom for strong signals
        confidence = 0.40 + (score_margin * 0.40)
        return max(0.40, min(0.80, confidence))
    else:
        # BUY: 25% to 65% (range of 40%)
        # Reflects empirical BUY WR of 32.5% with headroom for strong signals
        confidence = 0.25 + (score_margin * 0.40)
        return max(0.25, min(0.65, confidence))

def apply_oi_multiplier(confidence, oi_change, verdict='BUY'):
    """
    Apply OI-based multiplier to confidence (Nov 4, 2025 data-driven fix).
    
    RATIONALE: Analysis of 64 signals showed that OI Change is the strongest
    predictor of success (correlation +0.371), yet it was underweighted in
    the score formula. Signals with OI < 0 incorrectly received high confidence.
    
    DATA FINDINGS:
    - High confidence (70-80%) but LOSS: avg OI = -4.8M (negative!)
    - Low confidence (50-70%) but WIN: avg OI = +928k (positive!)
    - OI > 0: WR 93.5%, Profit +0.78%
    - OI ≤ 0: WR 78.1%, Profit +0.33%
    
    This multiplier corrects the inverse confidence correlation from -0.372
    to positive (+0.003 tested), while preserving calibrated ranges.
    
    Args:
        confidence: Base confidence from calibrate_confidence()
        oi_change: OI change value (absolute, not percentage)
        verdict: 'BUY' or 'SELL' (determines final clamping range)
    
    Returns:
        Adjusted confidence with OI multiplier applied and clamped to ranges
    """
    if oi_change is None:
        return confidence
    
    # Calculate OI multiplier
    if oi_change > 0:
        # Positive OI: boost up to 1.2x
        # Scale: 0 → 1.0, 50M → 1.2
        multiplier = 1.0 + min(0.2, oi_change / 50_000_000)
    else:
        # Negative OI: penalty to 0.8x
        multiplier = 0.8
    
    # Apply multiplier
    adjusted_confidence = confidence * multiplier
    
    # Clamp to direction-specific ranges (maintain recalibrated bounds)
    if verdict == 'SELL':
        return max(0.40, min(0.80, adjusted_confidence))
    else:  # BUY
        return max(0.25, min(0.65, adjusted_confidence))

def load_coin_config(symbol, config_dict):
    """
    Load configuration for specific coin from config dictionary.
    Falls back to default_coin if symbol not found.
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        config_dict: Full configuration dictionary from config.yaml
    
    Returns:
        Dict with coin-specific weights, min_score, and targets
    """
    coin_configs = config_dict.get('coin_configs', {})
    default_config = config_dict.get('default_coin', {
        'weights': {'cvd': 1.0, 'oi': 2.0, 'vwap': 1.0, 'volume': 1.0, 'liquidations': 1.0},  # OI=2.0 (Nov 4, 2025 data-driven)
        'targets': [0.4, 0.7]
    })
    
    # Get coin-specific config or default
    coin_cfg = coin_configs.get(symbol, default_config)
    
    # Calculate max possible score (sum of all weights)
    weights = coin_cfg.get('weights', default_config['weights'])
    max_score = sum(weights.values())
    
    # ASYMMETRIC MIN_SCORE: Return both BUY/SELL thresholds for correct logging
    # Legacy fallback for backwards compatibility
    legacy_min_score_pct = coin_cfg.get('min_score_pct', config_dict.get('min_score_pct', 0.70))
    min_score_pct_buy = coin_cfg.get('min_score_pct_buy', legacy_min_score_pct)
    min_score_pct_sell = coin_cfg.get('min_score_pct_sell', legacy_min_score_pct)
    
    # Use average for logging (actual filtering uses direction-specific thresholds)
    avg_min_score_pct = (min_score_pct_buy + min_score_pct_sell) / 2.0
    min_score = max_score * avg_min_score_pct
    
    return {
        'weights': weights,
        'min_score': min_score,
        'max_score': max_score,
        'min_score_pct_buy': min_score_pct_buy,
        'min_score_pct_sell': min_score_pct_sell,
        'targets': coin_cfg.get('targets', default_config['targets']),
        'cvd_threshold': coin_cfg.get('cvd_threshold', 1000000)
    }

def calibrate_confidence(symbol, verdict, formula_confidence, config=None):
    """
    Calibrate formula-based confidence using empirical win rates from effectiveness_log.csv.
    
    Blends:
    - Formula confidence (current method using weighted score)
    - Empirical win rate (actual historical performance for similar signals)
    
    Weighting adapts based on sample size:
    - Low data (< 10 samples): 90% formula, 10% empirical
    - Medium data (10-30 samples): 70% formula, 30% empirical
    - High data (30-50 samples): 50% formula, 50% empirical
    - Very high data (50+ samples): 40% formula, 60% empirical
    
    Args:
        symbol: Trading pair symbol
        verdict: 'BUY' or 'SELL'
        formula_confidence: Raw confidence from weighted score calculation
        config: Full config dict (optional, for future enhancements)
    
    Returns:
        Calibrated confidence (float, 0.0-1.0)
    """
    
    # If no verdict or invalid confidence, return as-is
    if verdict == 'NO_TRADE' or formula_confidence <= 0:
        return formula_confidence
    
    # Check cache first
    cache_key = 'effectiveness_data'
    now = time.time()
    
    if cache_key in _CALIBRATION_CACHE:
        cached_data, cached_time = _CALIBRATION_CACHE[cache_key]
        if now - cached_time < _CALIBRATION_TTL:
            df = cached_data
        else:
            # Cache expired, reload
            df = load_effectiveness_log()
            _CALIBRATION_CACHE[cache_key] = (df, now)
    else:
        # First load
        df = load_effectiveness_log()
        _CALIBRATION_CACHE[cache_key] = (df, now)
    
    # If no data or failed to load, return formula confidence
    if df is None or len(df) == 0:
        return formula_confidence
    
    # Filter for similar signals (same symbol + verdict)
    similar = df[(df['symbol'] == symbol) & (df['verdict'] == verdict)]
    
    if len(similar) == 0:
        # No similar signals, return formula confidence
        return formula_confidence
    
    # Define confidence buckets (±10% around formula confidence)
    # RECALIBRATED: Use direction-specific min/max bounds
    if verdict == 'SELL':
        # SELL range: 0.40 to 0.80
        min_bound = 0.40
        max_bound = 0.80
    else:
        # BUY range: 0.25 to 0.65
        min_bound = 0.25
        max_bound = 0.65
    
    conf_min = max(min_bound, formula_confidence - 0.10)
    conf_max = min(max_bound, formula_confidence + 0.10)
    
    # Filter for similar confidence range
    bucket = similar[(similar['confidence'] >= conf_min) & (similar['confidence'] <= conf_max)]
    
    if len(bucket) == 0:
        # No signals in this confidence range, use all similar signals as fallback
        bucket = similar
    
    # Calculate empirical win rate
    wins = len(bucket[bucket['result'] == 'WIN'])
    total = len(bucket)
    empirical_win_rate = wins / total if total > 0 else formula_confidence
    
    # Adaptive weighting based on sample size
    if total < 10:
        # Low data: trust formula heavily
        formula_weight = 0.90
        empirical_weight = 0.10
    elif total < 30:
        # Medium data: balanced but still formula-heavy
        formula_weight = 0.70
        empirical_weight = 0.30
    elif total < 50:
        # High data: equal weighting
        formula_weight = 0.50
        empirical_weight = 0.50
    else:
        # Very high data: trust empirical more
        formula_weight = 0.40
        empirical_weight = 0.60
    
    # Blend confidences
    calibrated = (formula_weight * formula_confidence) + (empirical_weight * empirical_win_rate)
    
    # Clamp to direction-specific max bound (RECALIBRATED)
    # SELL: max 0.80, BUY: max 0.65
    calibrated = min(max_bound, calibrated)
    
    return calibrated

def load_effectiveness_log():
    """Load effectiveness_log.csv and return as pandas DataFrame, or None if failed."""
    
    log_file = 'effectiveness_log.csv'
    
    if not os.path.exists(log_file):
        return None
    
    try:
        df = pd.read_csv(log_file)
        
        # Ensure required columns exist
        required_cols = ['symbol', 'verdict', 'confidence', 'result']
        if not all(col in df.columns for col in required_cols):
            return None
        
        return df
    except Exception as e:
        return None

def aggregate_recent_analysis(symbol, minutes=5):
    """
    Aggregate the last N minutes of analysis data from analysis_log.csv
    Implements 5-minute lookback window based on optimizer results (81.4% accuracy)
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        minutes: Lookback window in minutes (default 5)
    
    Returns:
        Dict with aggregated indicator values, or None if insufficient data
    """
    log_file = 'analysis_log.csv'
    
    if not os.path.exists(log_file):
        return None
    
    try:
        # Read analysis log
        df = pd.read_csv(log_file)
        
        # Ensure timestamp column exists and convert
        if 'timestamp' not in df.columns:
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter for this symbol and last N minutes
        now = pd.Timestamp.now()
        lookback_start = now - pd.Timedelta(minutes=minutes)
        
        recent_data = df[
            (df['symbol'] == symbol) &
            (df['timestamp'] > lookback_start) &
            (df['timestamp'] <= now)
        ]
        
        # Need at least 2 data points for meaningful aggregation
        if len(recent_data) < 2:
            return None
        
        # Aggregate indicators
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
            'timeframe_minutes': minutes
        }
        
        return aggregated
        
    except Exception as e:
        # Silently fail and return None - will fall back to instant values
        return None

def decide_signal(symbol, interval, config=None, lookback_minutes=15, vwap_window=30, volume_spike_mult=0.5, min_components=3, use_aggregation=False, aggregation_minutes=5):
    """
    Generate trading signal using weighted scoring system.
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Timeframe (e.g., '15m')
        config: Configuration dict with coin-specific weights (optional)
        lookback_minutes: Minutes of historical data
        vwap_window: VWAP calculation window
        volume_spike_mult: Volume spike threshold multiplier
        min_components: Legacy parameter (ignored in weighted mode)
        use_aggregation: Use N-minute lookback aggregation from analysis_log (default False)
        aggregation_minutes: Lookback window for aggregation (default 5, based on optimizer)
    
    Returns:
        Dict with signal details including weighted score and confidence
    
    Note:
        When use_aggregation=True, aggregates last N minutes of analysis data
        for trend-based decisions (81.4% accuracy from optimizer vs instant values)
    """
    # Load coin-specific configuration
    if config:
        coin_cfg = load_coin_config(symbol, config)
    else:
        # Fallback to legacy mode if no config provided
        coin_cfg = {
            'weights': {'cvd': 1.0, 'oi': 2.0, 'vwap': 1.0, 'volume': 1.0, 'liquidations': 1.0},  # OI=2.0 (Nov 4, 2025)
            'min_score': 2.0,  # Legacy: 2 components
            'max_score': 5.0,
            'targets': [0.4, 0.7],
            'cvd_threshold': 1000000
        }
    
    lb=lookback_minutes*60*1000
    kl=fetch_klines(symbol, interval, max(vwap_window,60)); 
    if not kl or len(kl)<2: 
        return {
            'symbol':symbol,'interval':interval,'last_close':0,'vwap_ref':0,'cvd':0,
            'oi_now':0,'oi_prev':0,'oi_change':0,
            'liq_summary':{'long_count':0,'short_count':0,'long_usd':0,'short_usd':0},
            'volume':{'last':0,'median':0,'spike':False},'components':{},'verdict':'NO_TRADE',
            'confidence':0.0,'score':0.0,'min_score':0.0,'max_score':0.0
        }
    
    # Try to use aggregated data if enabled
    agg_data = None
    if use_aggregation:
        agg_data = aggregate_recent_analysis(symbol, aggregation_minutes)
    
    # Fetch indicator data (use aggregated if available, otherwise fetch instant)
    last=float(kl[-1][4]); vwap,vwap_sigma=compute_vwap_sigma(kl, vwap_window)  # Local VWAP calculation with weighted sigma
    
    # SAFETY CHECK #1: If vwap_sigma < 1e-3, set dev_sigma to 0 (no boost)
    # Reasoning: Extremely low sigma indicates data quality issues or abnormal market conditions
    if vwap_sigma < 1e-3:
        dev_sigma = 0.0
    else:
        # Calculate deviation from VWAP in standard deviations (sigma)
        # dev_sigma > 2.0: Price far from VWAP (strong signal potential)
        # dev_sigma < 0.3: Price near VWAP (weak signal, consolidation)
        dev_sigma = abs((last - vwap) / vwap_sigma)
    
    # SAFETY CHECK #2: Calculate quote volume percentile for boost guard-rail
    # If last-bar quote_volume < P10 of last 100 bars, skip boost
    quote_vol_pctl = 0.0
    boost_safe_by_volume = True
    if len(kl) > 0 and len(kl[0]) > 7:  # Ensure quote volume exists
        last_quote_vol = float(kl[-1][7])  # Last bar quote volume
        lookback_bars = min(100, len(kl))
        recent_quote_vols = [float(k[7]) for k in kl[-lookback_bars:] if len(k) > 7]
        
        if recent_quote_vols:
            # Calculate percentile of last bar's volume
            sorted_vols = sorted(recent_quote_vols)
            rank = sum(1 for v in sorted_vols if v <= last_quote_vol)
            quote_vol_pctl = rank / len(sorted_vols) * 100
            
            # Skip boost if volume is below 10th percentile (thin volume)
            if quote_vol_pctl < 10.0:
                boost_safe_by_volume = False
    
    if agg_data:
        # Use aggregated values from lookback window
        cvd = agg_data['cvd_mean']
        d_oi = agg_data['oi_change_mean']
        oi_change_pct = agg_data['oi_change_pct_mean']
        rsi = agg_data['rsi_mean']
        vl = agg_data['volume_sum'] / agg_data['data_points']  # Avg volume per data point
        vm = agg_data['volume_median']
        sp = (vl / vm) >= volume_spike_mult if vm > 0 else False
        # OI values for backward compatibility
        oi = 0  # Not available in aggregated data
        oip = None
    else:
        # Fetch instant values (fallback or when aggregation disabled)
        cvd=compute_cvd(symbol, lb)
        
        # Add 1.0s delay before first Coinalyze API call to prevent burst
        # This spreads 3 API calls over 2s instead of <0.1s (reducing burst from 180/min to 20/min)
        time.sleep(1.0)
        oi=fetch_open_interest(symbol)
        
        # Add 1.0s delay between Coinalyze API calls to prevent rate limit (40/min)
        # Total delays: 1.0s + 1.0s = 2s per symbol (adds 22s to full cycle, ~88s total)
        time.sleep(1.0)
        oih=fetch_open_interest_hist(symbol,'5min',12)
        
        oip=oih[-2] if len(oih)>=2 else None
        d_oi=(oi-oip) if oip is not None else 0.0
        oi_change_pct = (d_oi / oip * 100) if oip and oip > 0 else 0.0
        sp,vl,vm=compute_volume_spike(kl, min(30,len(kl)), volume_spike_mult)
        rsi=compute_rsi(kl, period=14)
    
    # Price-based indicators (always from klines, not aggregated)
    liq=fetch_liquidations(symbol)
    
    # Use strict two-point VWAP cross detection
    vwap_cross_up, vwap_cross_down = detect_strict_vwap_cross(kl, vwap)
    
    funding_rate=fetch_funding_rate(symbol)
    ema_short,ema_long,ema_cross_up,ema_cross_down=compute_ema_crossover(kl, short_period=5, long_period=20)
    
    # Calculate ADX for trend strength detection
    try:
        adx = compute_adx(kl, period=14)
    except Exception as e:
        print(f"[ADX ERROR] {symbol}: ADX calculation failed: {e}")
        import traceback
        traceback.print_exc()
        adx = None
    
    # Build component dictionary
    # ASYMMETRIC VOLUME FILTER: DOWN movements happen on QUIET volume (pattern mining analysis)
    # BUY: volume_weak if < 35% median (UP movements avg 0.65x median)
    # SELL: volume_weak if < 25% median (DOWN movements avg 0.57x median, 60.9% occur <0.3x)
    vol_weak_buy = (vl < vm * 0.35) if vm > 0 else False
    vol_weak_sell = (vl < vm * 0.25) if vm > 0 else False
    
    # Get CVD threshold from config (minimum CVD magnitude required for signal)
    cvd_threshold = coin_cfg.get('cvd_threshold', 1000000)
    
    # UIF-30: Fetch basis_pct from Data Feeds Service snapshot (gated by feature flag)
    # IMPORTANT: Must be after coin_cfg is loaded to access weights
    basis_pct = None
    basis_age_sec = None
    basis_score_component = 0.0
    enable_basis = config and config.get('feature_flags', {}).get('enable_basis_in_scoring', False) if config else False
    
    if enable_basis:
        basis_pct_raw, basis_age_sec = fetch_basis(symbol)
        if basis_pct_raw is not None and basis_age_sec is not None and basis_age_sec <= 120:
            # basis_pct_raw is already in percentage (e.g., -0.0461 for -0.0461%)
            # Clamp to ±0.30% to prevent extreme outliers
            basis_pct = max(-0.30, min(0.30, basis_pct_raw))
            
            # Calculate weighted contribution: basis_pct * weight
            # Sign handling: Negative basis_pct (discount) = bullish signal
            #                Positive basis_pct (premium) = bearish signal
            basis_weight = coin_cfg['weights'].get('basis_pct', 0.10)
            basis_score_component = basis_pct * basis_weight
    
    # UIF-12: Fetch UIF features from UIF Feature Engine snapshot (gated by feature flag)
    # DIAGNOSTIC ONLY: All weights are 0.0 for telemetry collection phase
    uif_features = None
    uif_age_sec = None
    uif_score_components = {}
    enable_uif = config and config.get('feature_flags', {}).get('enable_uif_in_scoring', False) if config else False
    
    if enable_uif:
        uif_data, uif_age_sec = fetch_uif_snapshot(symbol)
        if uif_data is not None and uif_age_sec is not None and uif_age_sec <= 120:
            uif_features = uif_data
            
            # Calculate score components for 8 UIF features with weights from config
            # All weights default to 0.0 for diagnostic phase (no scoring impact)
            weights = coin_cfg.get('weights', {})
            
            # 4 UIF Engine features
            uif_score_components['adx14'] = uif_data.get('adx14', 0.0) * weights.get('adx14', 0.0)
            uif_score_components['psar'] = uif_data.get('psar', 0) * weights.get('psar', 0.0)
            uif_score_components['momentum5'] = uif_data.get('momentum5', 0.0) * weights.get('momentum5', 0.0)
            uif_score_components['vol_accel'] = uif_data.get('vol_accel', 0.0) * weights.get('vol_accel', 0.0)
            
            # 4 placeholder features (calculated from existing data in analysis_log)
            # zcvd: normalized CVD
            uif_score_components['zcvd'] = (cvd / abs(vl) if vl != 0 else 0.0) * weights.get('zcvd', 0.0)
            # doi_pct: OI change percentage (already calculated as oi_change_pct)
            uif_score_components['doi_pct'] = oi_change_pct * weights.get('doi_pct', 0.0)
            # dev_sigma: VWAP deviation in sigma units (already calculated)
            uif_score_components['dev_sigma'] = dev_sigma * weights.get('dev_sigma', 0.0)
            # rsi_dist: RSI distance from neutral 50
            uif_score_components['rsi_dist'] = (abs(rsi - 50) if rsi is not None else 0.0) * weights.get('rsi_dist', 0.0)
    
    # ORDER FLOW INDICATORS (Nov 15, 2025)
    # Calculate order flow microstructure signals (bid-ask aggression + psychological levels)
    # DIAGNOSTIC ONLY: All weights start at 0.0 for initial telemetry collection
    of_ba_aggression = None
    of_psych_levels = None
    of_score_components = {}
    enable_order_flow = config and config.get('feature_flags', {}).get('enable_order_flow', False) if config else False
    
    if enable_order_flow:
        # 1. BID-ASK AGGRESSION RATIO
        # Measures buying vs selling pressure from CVD changes
        # BA ratio > 2.0 = aggressive buying (bullish)
        # BA ratio < 0.5 = aggressive selling (bearish)
        try:
            of_ba_aggression = calculate_bid_ask_aggression(symbol, lookback_minutes=5)
            weights = coin_cfg.get('weights', {})
            
            # Calculate weighted score component (zero by default for diagnostic phase)
            # Positive ba_ratio boost for BUY, negative for SELL
            ba_signal_strength = of_ba_aggression.get('strength', 0) / 100.0  # Normalize to 0-1
            ba_direction = 1.0 if of_ba_aggression.get('signal') == 'BULLISH' else -1.0 if of_ba_aggression.get('signal') == 'BEARISH' else 0.0
            of_score_components['ba_aggression'] = ba_signal_strength * ba_direction * weights.get('ba_aggression', 0.0)
        except Exception as e:
            print(f"[ORDER_FLOW] BA Aggression error for {symbol}: {e}")
            of_ba_aggression = {'signal': 'NEUTRAL', 'strength': 0, 'ba_ratio': 1.0, 'cvd_delta': 0}
            of_score_components['ba_aggression'] = 0.0
        
        # 2. PSYCHOLOGICAL LEVEL PROXIMITY
        # Detects stop-loss/take-profit clusters (round numbers, Fibonacci, extremes)
        # in_danger_zone = True means high risk of stop-hunt volatility
        try:
            # Get recent high/low from last 50 candles for Fibonacci levels
            recent_highs = [float(k[2]) for k in kl[-50:]] if len(kl) >= 50 else [float(k[2]) for k in kl]
            recent_lows = [float(k[3]) for k in kl[-50:]] if len(kl) >= 50 else [float(k[3]) for k in kl]
            recent_high = max(recent_highs) if recent_highs else None
            recent_low = min(recent_lows) if recent_lows else None
            
            of_psych_levels = detect_psychological_levels(
                symbol=symbol,
                current_price=last,
                recent_high=recent_high,
                recent_low=recent_low,
                proximity_threshold=0.003  # 0.3% proximity
            )
            
            # Calculate weighted score component (zero by default)
            # High risk_score reduces confidence (reduces position size near stop clusters)
            risk_penalty = of_psych_levels.get('risk_score', 0) / 100.0  # Normalize to 0-1
            of_score_components['psych_level_risk'] = -risk_penalty * weights.get('psych_level_risk', 0.0)
        except Exception as e:
            print(f"[ORDER_FLOW] Psychological Level error for {symbol}: {e}")
            of_psych_levels = {'in_danger_zone': False, 'risk_score': 0, 'nearest_level': 0.0}
            of_score_components['psych_level_risk'] = 0.0
    
    # OI threshold calculation
    # LOWERED TO 0.02% based on pattern mining: real movements occur at -0.02% to +0.03%
    # Previous 0.05% threshold blocked 32.1% of valid movements
    if agg_data:
        # When using aggregated data, use percentage-based threshold
        # OI components: Only set if percentage exceeds 0.02%
        oi_up = oi_change_pct > 0.02
        oi_down = oi_change_pct < -0.02
    else:
        # Calculate minimum OI change (0.02% of current OI or $10K, whichever is larger)
        # Lowered from 0.05% to 0.02% based on pattern mining analysis
        # BTC example: 0.02% = $1.6M (realistic for 5-15min intervals with 1%+ price moves)
        min_oi_change = max(abs(oi) * 0.0002, 10000)
        oi_up = d_oi > min_oi_change
        oi_down = d_oi < -min_oi_change
    
    # Calculate price vs VWAP percentage
    price_vs_vwap_pct = ((last - vwap) / vwap * 100) if vwap > 0 else 0
    
    comp={
        # CVD components: Only set if magnitude exceeds threshold
        'CVD_pos': cvd > cvd_threshold,
        'CVD_neg': cvd < -cvd_threshold,
        # OI components: Set based on whether using aggregated or instant data
        'OI_up': oi_up,
        'OI_down': oi_down,
        # VWAP components: Strict two-point cross + position
        'VWAP_cross_up':vwap_cross_up,
        'VWAP_cross_down':vwap_cross_down,
        'Price_below_VWAP':last < vwap,
        'Price_above_VWAP':last > vwap,
        # Volume components (ASYMMETRIC: separate thresholds for BUY/SELL)
        'Vol_spike':sp,
        'Vol_weak_buy':vol_weak_buy,    # <35% median blocks BUY
        'Vol_weak_sell':vol_weak_sell,  # <25% median blocks SELL
        # Liquidation components
        'Liq_long':liq['long_count']>liq['short_count'],
        'Liq_short':liq['short_count']>liq['long_count'],
        # Funding rate components
        'Funding_positive':funding_rate>0,
        'Funding_negative':funding_rate<0,
        # RSI components
        'RSI_overbought':(rsi is not None and rsi > 70),
        'RSI_oversold':(rsi is not None and rsi < 30),
        # EMA components
        'EMA_cross_up':ema_cross_up,
        'EMA_cross_down':ema_cross_down,
        'EMA_bearish':(ema_short is not None and ema_long is not None and ema_short < ema_long),
        # ADX components (trend strength)
        'ADX_strong_trend':(adx is not None and adx > 25),
        'ADX_very_strong':(adx is not None and adx > 50)
    }
    
    # UIF-30: Add basis_score component if available (omit entirely when None/stale for graceful fallback)
    if basis_score_component != 0.0:
        comp['basis_score'] = basis_score_component
    
    # UIF-12: Add UIF score components if available (diagnostic logging only, all weights = 0.0)
    if uif_score_components:
        for feature_name, score_value in uif_score_components.items():
            comp[f'uif_{feature_name}_score'] = score_value
    
    # ORDER FLOW: Add Order Flow score components if available (diagnostic logging only, all weights = 0.0)
    if of_score_components:
        for feature_name, score_value in of_score_components.items():
            comp[f'of_{feature_name}_score'] = score_value
    
    # PROFESSIONAL CONFLUENCE-BASED SCORING (replaces weighted sum approach)
    # Check BUY and SELL confluence separately
    buy_signal, buy_score, buy_max, buy_aligned = calculate_confluence_score(comp, coin_cfg['weights'], direction='BUY')
    sell_signal, sell_score, sell_max, sell_aligned = calculate_confluence_score(comp, coin_cfg['weights'], direction='SELL')
    
    # DEV_SIGMA FILTER: Professional institutional-grade VWAP deviation filter
    # Read per-symbol thresholds from config (with fallback to defaults)
    dev_sigma_config = coin_cfg.get('dev_sigma_thresholds', {
        'block_below': 0.30,
        'boost_above': 2.0,
        'optimized_set': 'B_default'
    })
    block_threshold = dev_sigma_config.get('block_below', 0.30)
    boost_threshold = dev_sigma_config.get('boost_above', 2.0)
    ab_set_used = dev_sigma_config.get('optimized_set', 'B_default')
    
    # Check if signal should be blocked (price too close to VWAP = consolidation)
    dev_sigma_blocked = 0
    if dev_sigma < block_threshold and (buy_signal or sell_signal):
        # Block signals in consolidation zone
        # Reasoning: Low profit potential, high noise
        buy_signal = False
        sell_signal = False
        buy_score = 0.0
        sell_score = 0.0
        dev_sigma_blocked = 1
    
    # Determine verdict based on confluence signals
    min_confidence = config.get('min_confidence', 0.70) if config else 0.70
    
    # ASYMMETRIC MIN_SCORE: Get coin-specific thresholds for BUY and SELL separately
    # Pattern mining analysis: BUY needs lower threshold (×0.85), SELL higher (×0.90)
    # Fallback to legacy min_score_pct if asymmetric values not configured
    legacy_min_score_pct = coin_cfg.get('min_score_pct', 0.80)
    min_score_pct_buy = coin_cfg.get('min_score_pct_buy', legacy_min_score_pct)
    min_score_pct_sell = coin_cfg.get('min_score_pct_sell', legacy_min_score_pct)
    
    if buy_signal and not sell_signal:
        # Use aligned score for confidence calculation
        formula_conf = calculate_confidence_from_score(buy_score, buy_max * 0.6, buy_max, min_confidence, verdict='BUY')
        # Apply confidence calibration using empirical win rates
        conf = calibrate_confidence(symbol, 'BUY', formula_conf, config)
        # Apply OI multiplier (Nov 4, 2025 data-driven fix)
        conf = apply_oi_multiplier(conf, d_oi, verdict='BUY')
        
        # BOOST GUARD-RAILS: Apply boost ONLY if all safety checks pass
        # Order: block → base_confidence → boost
        dev_sigma_boost = 0.0
        boost_applied = 0
        if dev_sigma_blocked == 0 and boost_safe_by_volume:
            # Calculate smooth dev_sigma boost for extreme deviations
            # Formula: boost = max(0.0, min(0.20, (dev_sigma - 1.0) / 3.0))
            dev_sigma_boost = max(0.0, min(0.20, (dev_sigma - 1.0) / 3.0))
            if dev_sigma_boost > 0.0:
                conf = min(conf * (1.0 + dev_sigma_boost), 0.65)  # Apply boost (capped at BUY max 65%)
                boost_applied = 1
        
        # CRITICAL FIX: Only set verdict to BUY if score meets threshold
        # ASYMMETRIC: Use min_score_pct_buy (lower threshold for UP movements)
        score_pct = (buy_score / buy_max) if buy_max > 0 else 0
        if score_pct >= min_score_pct_buy:
            v = 'BUY'
            score = buy_score
            aligned_indicators = buy_aligned
        else:
            v = 'NO_TRADE'
            conf = 0.0
            score = buy_score
            aligned_indicators = []
    elif sell_signal and not buy_signal:
        formula_conf = calculate_confidence_from_score(sell_score, sell_max * 0.6, sell_max, min_confidence, verdict='SELL')
        # Apply confidence calibration using empirical win rates
        conf = calibrate_confidence(symbol, 'SELL', formula_conf, config)
        # Apply OI multiplier (Nov 4, 2025 data-driven fix)
        conf = apply_oi_multiplier(conf, d_oi, verdict='SELL')
        
        # BOOST GUARD-RAILS: Apply boost ONLY if all safety checks pass
        # Order: block → base_confidence → boost
        dev_sigma_boost = 0.0
        boost_applied = 0
        if dev_sigma_blocked == 0 and boost_safe_by_volume:
            # Calculate smooth dev_sigma boost for extreme deviations
            # Formula: boost = max(0.0, min(0.20, (dev_sigma - 1.0) / 3.0))
            dev_sigma_boost = max(0.0, min(0.20, (dev_sigma - 1.0) / 3.0))
            if dev_sigma_boost > 0.0:
                conf = min(conf * (1.0 + dev_sigma_boost), 0.80)  # Apply boost (capped at SELL max 80%)
                boost_applied = 1
        
        # CRITICAL FIX: Only set verdict to SELL if score meets threshold
        # ASYMMETRIC: Use min_score_pct_sell (higher threshold for DOWN movements)
        score_pct = (sell_score / sell_max) if sell_max > 0 else 0
        if score_pct >= min_score_pct_sell:
            v = 'SELL'
            score = sell_score
            aligned_indicators = sell_aligned
        else:
            v = 'NO_TRADE'
            conf = 0.0
            score = sell_score
            aligned_indicators = []
    elif buy_signal and sell_signal:
        # Both signals present - choose stronger one based on weighted score
        if buy_score > sell_score:
            formula_conf = calculate_confidence_from_score(buy_score, buy_max * 0.6, buy_max, min_confidence, verdict='BUY')
            conf = calibrate_confidence(symbol, 'BUY', formula_conf, config)
            # Apply OI multiplier (Nov 4, 2025 data-driven fix)
            conf = apply_oi_multiplier(conf, d_oi, verdict='BUY')
            
            # BOOST GUARD-RAILS: Apply boost ONLY if all safety checks pass
            dev_sigma_boost = 0.0
            boost_applied = 0
            if dev_sigma_blocked == 0 and boost_safe_by_volume:
                dev_sigma_boost = max(0.0, min(0.20, (dev_sigma - 1.0) / 3.0))
                if dev_sigma_boost > 0.0:
                    conf = min(conf * (1.0 + dev_sigma_boost), 0.65)  # Apply boost (capped at BUY max 65%)
                    boost_applied = 1
            
            # CRITICAL FIX: Only set verdict to BUY if score meets threshold
            # ASYMMETRIC: Use min_score_pct_buy (lower threshold for UP movements)
            score_pct = (buy_score / buy_max) if buy_max > 0 else 0
            if score_pct >= min_score_pct_buy:
                v = 'BUY'
                score = buy_score
                aligned_indicators = buy_aligned
            else:
                v = 'NO_TRADE'
                conf = 0.0
                score = buy_score
                aligned_indicators = []
        else:
            formula_conf = calculate_confidence_from_score(sell_score, sell_max * 0.6, sell_max, min_confidence, verdict='SELL')
            conf = calibrate_confidence(symbol, 'SELL', formula_conf, config)
            # Apply OI multiplier (Nov 4, 2025 data-driven fix)
            conf = apply_oi_multiplier(conf, d_oi, verdict='SELL')
            
            # BOOST GUARD-RAILS: Apply boost ONLY if all safety checks pass
            dev_sigma_boost = 0.0
            boost_applied = 0
            if dev_sigma_blocked == 0 and boost_safe_by_volume:
                dev_sigma_boost = max(0.0, min(0.20, (dev_sigma - 1.0) / 3.0))
                if dev_sigma_boost > 0.0:
                    conf = min(conf * (1.0 + dev_sigma_boost), 0.80)  # Apply boost (capped at SELL max 80%)
                    boost_applied = 1
            
            # CRITICAL FIX: Only set verdict to SELL if score meets threshold
            # ASYMMETRIC: Use min_score_pct_sell (higher threshold for DOWN movements)
            score_pct = (sell_score / sell_max) if sell_max > 0 else 0
            if score_pct >= min_score_pct_sell:
                v = 'SELL'
                score = sell_score
                aligned_indicators = sell_aligned
            else:
                v = 'NO_TRADE'
                conf = 0.0
                score = sell_score
                aligned_indicators = []
    else:
        v = 'NO_TRADE'
        conf = 0.0
        score = max(buy_score, sell_score)
        aligned_indicators = []
        # Set default values for boost tracking
        dev_sigma_boost = 0.0
        boost_applied = 0
    
    atr = calculate_atr(kl, period=14)
    
    # For backward compatibility, calculate min/max scores
    # ASYMMETRIC: Use appropriate threshold based on verdict
    # FIX: Use max of both directions to prevent blocking when one direction returns 0
    # (e.g., BUY blocked by CVD<0 returns max=0, but SELL is valid with max=2.85)
    max_score = max(buy_max, sell_max)
    min_score_pct_used = min_score_pct_buy if v == 'BUY' else (min_score_pct_sell if v == 'SELL' else legacy_min_score_pct)
    min_score = max_score * min_score_pct_used
    
    # HYBRID EMA+VWAP regime detection (new system)
    # Detects: strong_bear, bear_warning, strong_bull, bull_warning, neutral, sideways
    # bear_warning/bull_warning enable early entry on reversals (2-4 minutes faster than old system)
    regime = detect_regime_hybrid(last, vwap, ema_short, ema_long, dev_sigma, adx, comp)
    
    return {
        'symbol':symbol,
        'interval':interval,
        'last_close':last,
        'vwap_ref':vwap,
        'cvd':cvd,
        'oi_now':oi,
        'oi_prev':oip,
        'oi_change':d_oi,
        'oi_change_pct':oi_change_pct,
        'liq_summary':liq,
        'volume':{'last':vl,'median':vm,'spike':sp},
        'components':comp,
        'verdict':v,
        'confidence':round(conf,2),
        'score':round(score,2),
        'min_score':round(min_score,2),
        'max_score':round(max_score,2),
        'aligned_indicators':aligned_indicators if v != 'NO_TRADE' else [],
        'coin_config':coin_cfg,  # Include coin config for target calculations
        'klines':kl,  # Include klines for ATR calculation
        'funding_rate':funding_rate,
        'rsi':rsi,
        'ema_short':ema_short,
        'ema_long':ema_long,
        'atr':atr,
        # New fields for enhanced logging
        'regime':regime,
        'vwap_cross_up':vwap_cross_up,
        'vwap_cross_down':vwap_cross_down,
        'ema_cross_up':ema_cross_up,
        'ema_cross_down':ema_cross_down,
        'price_vs_vwap_pct':round(price_vs_vwap_pct, 2),
        'adx':adx,
        # VWAP sigma fields (professional institutional metrics)
        'vwap_sigma':round(vwap_sigma, 2),
        'dev_sigma':round(dev_sigma, 2),
        # Dev_sigma filter tracking fields
        'dev_sigma_blocked':dev_sigma_blocked,
        'dev_sigma_boost':round(dev_sigma_boost, 4),
        'ab_set_used':ab_set_used,
        # Boost guard-rail tracking fields
        'quote_vol_pctl':round(quote_vol_pctl, 1),
        'boost_applied':boost_applied,
        # UIF-30: Basis_pct integration tracking fields
        'basis_pct':round(basis_pct, 4) if basis_pct is not None else None,
        'basis_age_sec':basis_age_sec,
        'basis_score_component':round(basis_score_component, 4),
        # Diagnostic: interaction tracking for analysis
        'basis_x_funding':round(basis_pct * funding_rate, 6) if (basis_pct is not None and funding_rate is not None) else None,
        'basis_x_zcvd':round(basis_pct * (cvd / abs(cvd) if cvd != 0 else 0), 4) if basis_pct is not None else None,
        # UIF-12: UIF features integration tracking fields (diagnostic only, weights = 0.0)
        'uif_age_sec':uif_age_sec,
        'uif_score_components':{k: round(v, 6) for k, v in uif_score_components.items()},  # Dict for main.py extraction
        # Raw UIF feature values (for analysis_log.csv diagnostic logging)
        'adx14':round(uif_features.get('adx14', 0.0), 2) if uif_features else 0.0,
        'psar':uif_features.get('psar', 0) if uif_features else 0,
        'momentum5':round(uif_features.get('momentum5', 0.0), 4) if uif_features else 0.0,
        'vol_accel':round(uif_features.get('vol_accel', 0.0), 4) if uif_features else 0.0,
        # Placeholder features calculated from existing data
        'zcvd':round((cvd / abs(vl) if vl != 0 else 0.0), 4),
        'doi_pct':round(oi_change_pct, 2),
        'dev_sigma_uif':round(dev_sigma, 2),
        'rsi_dist':round((abs(rsi - 50) if rsi is not None else 0.0), 2),
        # ORDER FLOW: Order Flow indicators integration tracking (Nov 15, 2025)
        'of_score_components':{k: round(v, 6) for k, v in of_score_components.items()},  # Dict for main.py extraction
        # Raw Order Flow feature values
        'of_ba_ratio':round(of_ba_aggression.get('ba_ratio', 1.0), 3) if of_ba_aggression else 1.0,
        'of_ba_signal':of_ba_aggression.get('signal', 'NEUTRAL') if of_ba_aggression else 'NEUTRAL',
        'of_ba_strength':of_ba_aggression.get('strength', 0) if of_ba_aggression else 0,
        'of_psych_risk':of_psych_levels.get('risk_score', 0) if of_psych_levels else 0,
        'of_in_danger_zone':of_psych_levels.get('in_danger_zone', False) if of_psych_levels else False,
        'of_nearest_level':round(of_psych_levels.get('nearest_level', 0.0), 2) if of_psych_levels else 0.0
    }

def _human_int(n):
    try: return f"{int(round(float(n))):,}".replace(',',' ')
    except: return str(n)

def calculate_atr(klines, period=14):
    """
    Calculate Average True Range (ATR) - measures market volatility.
    Returns ATR as a price value (not percentage).
    
    Requires period+1 candles (e.g., 15 for period=14) to compute period true ranges.
    """
    if not klines or len(klines) < period + 1:
        return None
    
    true_ranges = []
    # Start from index 1 to compare with previous candle (index 0)
    for i in range(1, len(klines)):
        high = float(klines[i][2])
        low = float(klines[i][3])
        prev_close = float(klines[i-1][4])
        
        # True Range = max of:
        # 1. Current high - current low
        # 2. abs(current high - previous close)
        # 3. abs(current low - previous close)
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        true_ranges.append(tr)
    
    # ATR = average of last N true ranges (exactly period values)
    if len(true_ranges) >= period:
        return sum(true_ranges[-period:]) / period
    else:
        # Not enough data for full period, use what we have
        return sum(true_ranges) / len(true_ranges) if true_ranges else None

def calculate_volatility_based_interval(symbol, atr, price):
    """
    Calculate base TTL interval from recent volatility (ATR).
    Returns base interval in minutes.
    
    Strategy:
    - High volatility (fast-moving) = shorter base interval
    - Low volatility (slow-moving) = longer base interval
    - Tier-1 scalpers (BTC/ETH): 12-18min base
    - Mid-cap alts (SOL/BNB/LINK/AVAX): 20-35min base  
    - High-vol coins (DOGE): 25-45min base
    - Intraday (YFI/LUMIA/ANIME): 360min base (6h) with ±50% flex
    """
    intraday_coins = ['YFIUSDT', 'LUMIAUSDT', 'ANIMEUSDT']
    
    # Intraday coins: 6h base with flexibility
    if symbol in intraday_coins:
        return 360  # 6 hours base
    
    # Scalping coins: derive from ATR
    if atr is None or atr <= 0 or price <= 0:
        # Fallback to symbol-specific defaults
        tier1_scalpers = ['BTCUSDT', 'ETHUSDT']
        if symbol in tier1_scalpers:
            return 15
        else:
            return 25
    
    # Calculate ATR as percentage of price
    atr_pct = (atr / price) * 100
    
    # Map ATR% to base interval (inverse relationship: high volatility = shorter interval)
    # Typical ATR%: 0.3-0.8% for majors, 0.8-2.0% for alts
    tier1_scalpers = ['BTCUSDT', 'ETHUSDT']
    midcap_alts = ['SOLUSDT', 'BNBUSDT', 'LINKUSDT', 'AVAXUSDT']
    
    if symbol in tier1_scalpers:
        # BTC/ETH: 12-18min range
        if atr_pct >= 0.5:
            return 12  # High volatility = faster moves
        elif atr_pct >= 0.3:
            return 15  # Medium volatility
        else:
            return 18  # Low volatility = needs more time
    
    elif symbol in midcap_alts:
        # SOL/BNB/LINK/AVAX: 20-35min range
        if atr_pct >= 1.0:
            return 20
        elif atr_pct >= 0.6:
            return 25
        else:
            return 35
    
    else:
        # DOGE and others: 25-45min range
        if atr_pct >= 1.5:
            return 25
        elif atr_pct >= 1.0:
            return 35
        else:
            return 45

def calculate_dynamic_ttl(symbol, atr, price, volume_ratio, oi_change, oi_current, cvd, cvd_threshold, verdict, multiplier):
    """
    Calculate dynamic Time-To-Live (TTL) for trading signals.
    
    Strategy:
    - Base interval derived from volatility (ATR)
    - Dynamic multiplier from volume (0.4), OI (0.35), CVD (0.25)
    - Asymmetric logic: extend TTL only if OI+CVD align structurally, otherwise shorten
    - Strong signals = shorter TTL (fast moves) unless OI structure confirms continuation
    - Guardrails: cap at 2.5× base, floor at 0.75× base
    
    Returns (ttl_minutes, ttl_multiplier, base_interval_minutes)
    """
    # Step 1: Get volatility-based base interval
    base_interval = calculate_volatility_based_interval(symbol, atr, price)
    
    # Step 2: Calculate TTL multiplier components
    
    # Volume component (0.4 weight): high volume shortens TTL (fast catalyst)
    volume_boost = 0.0
    if volume_ratio > 1.0:
        # Volume spike = move happens faster = shorter TTL
        # Inverse: higher volume ratio = negative boost (shortens duration)
        volume_boost = -min((volume_ratio - 1.0) * 0.25, 0.4)  # Cap at -0.4 (shortens)
    else:
        # Low volume = needs more time = positive boost
        volume_boost = min((1.0 - volume_ratio) * 0.3, 0.2)  # Cap at +0.2 (lengthens)
    
    # OI component (0.35 weight): growing OI lengthens TTL (structural conviction)
    oi_boost = 0.0
    if oi_current > 0:
        oi_change_pct = abs(oi_change / oi_current)
        # Check if OI supports signal direction (growing OI for continuation)
        oi_aligned = (verdict == 'BUY' and oi_change > 0) or (verdict == 'SELL' and oi_change < 0)
        
        if oi_aligned and oi_change_pct > 0.001:  # >0.1% OI change aligned
            # Structural support = extend TTL
            oi_boost = min(oi_change_pct * 50, 0.35)  # Cap at +0.35
        elif oi_change_pct > 0.002:  # >0.2% OI change against
            # OI fighting signal = shorten TTL
            oi_boost = -min(oi_change_pct * 30, 0.2)  # Cap at -0.2
    
    # CVD component (0.25 weight): strong directional flow
    cvd_boost = 0.0
    cvd_aligned = (verdict == 'BUY' and cvd > 0) or (verdict == 'SELL' and cvd < 0)
    if cvd_aligned and cvd_threshold > 0:
        cvd_strength = min(abs(cvd) / cvd_threshold, 1.5)  # Normalize
        if cvd_strength > 1.0:
            # Very strong CVD = fast move = shorten TTL
            cvd_boost = -min((cvd_strength - 1.0) * 0.2, 0.25)
        else:
            # Moderate CVD = normal duration
            cvd_boost = 0.0
    
    # Step 3: Calculate final TTL multiplier with asymmetric logic
    # If both OI and CVD aligned = extend (structural confirmation)
    # Otherwise = shorten (fast move expected)
    if oi_boost > 0 and abs(cvd) >= cvd_threshold * 0.5:
        # Structural support from both OI and CVD = extend TTL
        ttl_multiplier = 1.0 + abs(volume_boost) * 0.3 + oi_boost + abs(cvd_boost) * 0.3
    else:
        # Fast catalyst or weak structure = shorten TTL
        ttl_multiplier = 1.0 + volume_boost + oi_boost + cvd_boost
    
    # Step 4: Apply guardrails (floor: 0.75×, cap: 2.5×)
    ttl_multiplier = max(0.75, min(2.5, ttl_multiplier))
    
    # Step 5: Calculate final TTL in minutes
    ttl_minutes = int(base_interval * ttl_multiplier)
    
    # Step 6: Apply hard caps to prevent extreme durations
    # DATA-DRIVEN CAPS: Analysis shows optimal range is 10-30 min for scalping
    # Too short (<10min) = insufficient time for targets
    # Too long (>30min) = increased risk, lower win rates
    intraday_coins = ['YFIUSDT', 'LUMIAUSDT', 'ANIMEUSDT']
    if symbol in intraday_coins:
        min_ttl = 60   # Intraday minimum: 1 hour
        max_ttl = 120  # Intraday maximum: 2 hours
    else:
        min_ttl = 10   # Scalping minimum: 10 minutes (increased from 5)
        max_ttl = 30   # Scalping maximum: 30 minutes (NEW HARD CAP)
    
    ttl_minutes = max(min_ttl, min(max_ttl, ttl_minutes))
    
    return (ttl_minutes, ttl_multiplier, base_interval)

def calculate_price_targets(price, confidence, cvd, symbol, coin_config=None, klines=None, volume_data=None, oi_change=0, verdict='BUY', vwap=None):
    """
    DATA-DRIVEN target calculation using ATR + market strength multipliers.
    NOW WITH DYNAMIC TTL based on volatility and market conditions.
    NOW WITH DIRECTIONAL VWAP logic (mean-reversion vs trend-following).
    
    - Scalping coins: ATR-based targets adjusted by volume/CVD/OI/VWAP strength
    - Intraday coins: Fixed config-based targets for positional trades
    
    Returns (min_target, max_target, duration_str, move_pct_str, multiplier, strength_icon, strength_label, ttl_minutes, base_interval)
    """
    # INTRADAY/POSITIONAL COINS - Use config-based targets for longer holds
    intraday_coins = ['YFIUSDT', 'LUMIAUSDT', 'ANIMEUSDT']
    
    if symbol in intraday_coins and coin_config and 'targets' in coin_config:
        # Use config-defined targets for intraday strategy
        targets = coin_config['targets']
        final_min = targets[0] if len(targets) > 0 else 1.5
        final_max = targets[1] if len(targets) > 1 else 3.0
        
        # Calculate dynamic TTL for intraday coins too
        atr_intraday = None
        if klines and len(klines) >= 15:
            atr_intraday = calculate_atr(klines, period=14)
        
        volume_ratio = 1.0
        if volume_data:
            volume_last = volume_data.get('last', 0)
            volume_median = volume_data.get('median', 1)
            volume_ratio = volume_last / volume_median if volume_median > 0 else 1.0
        
        oi_current = volume_data.get('oi_current', 1_000_000_000) if volume_data else 1_000_000_000
        cvd_threshold = coin_config.get('cvd_threshold', 120_000)
        multiplier = 1.3  # Fixed for intraday target calculation
        
        ttl_minutes, ttl_multiplier, base_interval = calculate_dynamic_ttl(
            symbol, atr_intraday, price, volume_ratio, oi_change, oi_current, cvd, cvd_threshold, verdict, multiplier
        )
        
        # Format duration string
        if ttl_minutes >= 720:
            duration = f"{ttl_minutes // 60}h"  # Show in hours
        elif ttl_minutes >= 60:
            hours = ttl_minutes // 60
            minutes = ttl_minutes % 60
            duration = f"{hours}h{minutes}m" if minutes > 0 else f"{hours}h"
        else:
            duration = f"{ttl_minutes}min"
        
        strength_icon = "📊"
        strength_label = "Intraday"
        return (final_min, final_max, duration, f"{final_min:.1f}-{final_max:.1f}%", multiplier, strength_icon, strength_label, ttl_minutes, base_interval)
    
    # SCALPING COINS - ATR-based data-driven targets
    
    # Step 1: Calculate ATR baseline (actual market volatility)
    atr = None
    if klines and len(klines) >= 15:  # Need period+1 candles for period-14 ATR
        atr = calculate_atr(klines, period=14)
    
    # Fallback if ATR calculation fails: use default conservative targets
    if atr is None or atr <= 0:
        base_target_pct = 0.4  # Conservative 0.4% baseline
    else:
        # Convert ATR to percentage
        base_target_pct = (atr / price) * 100
        # Clamp to reasonable range (0.2% - 2.0%)
        base_target_pct = max(0.2, min(2.0, base_target_pct))
    
    # Step 1b: Apply symbol-specific ATR multiplier (data-driven from effectiveness analysis)
    # This scales ATR targets based on actual P80 price movements observed historically
    atr_multiplier = coin_config.get('atr_multiplier', 1.0) if coin_config else 1.0
    base_target_pct = base_target_pct * atr_multiplier
    
    # Step 2: Apply market strength multipliers (more sensitive thresholds)
    multiplier = 1.0
    
    # Volume spike multiplier (more sensitive: 1.3x and 1.6x instead of 1.5x and 2.0x)
    # ADDED: Stronger downward adjustments for weak volume
    if volume_data:
        volume_last = volume_data.get('last', 0)
        volume_median = volume_data.get('median', 1)
        if volume_median > 0:
            volume_ratio = volume_last / volume_median
            if volume_ratio > 1.6:
                multiplier *= 1.3  # Strong volume = bigger move expected
            elif volume_ratio > 1.3:
                multiplier *= 1.15
            elif volume_ratio < 0.7:  # Weak volume
                multiplier *= 0.75  # Reduce target - low conviction
            elif volume_ratio < 0.5:  # Very weak volume
                multiplier *= 0.6  # Strongly reduce target
    
    # CVD strength multiplier (DIRECTION-AWARE: only apply when CVD supports signal)
    # ADDED: Stronger downward adjustments for weak/opposing CVD
    cvd_threshold = coin_config.get('cvd_threshold', 1_000_000) if coin_config else 1_000_000
    cvd_abs = abs(cvd)
    
    # Check if CVD direction matches signal verdict
    cvd_supports_signal = (verdict == 'BUY' and cvd > 0) or (verdict == 'SELL' and cvd < 0)
    
    if cvd_supports_signal:
        if cvd_abs >= cvd_threshold * 1.5:  # More sensitive: 1.5x instead of 2x
            multiplier *= 1.4  # Very strong directional flow
        elif cvd_abs >= cvd_threshold * 0.5:  # More sensitive: 0.5x instead of 1x
            multiplier *= 1.2  # Strong directional flow
        elif cvd_abs < cvd_threshold * 0.3:  # Weak CVD
            multiplier *= 0.85  # Reduce target
    else:
        # CVD opposes signal - this is dangerous, reduce target
        if cvd_abs >= cvd_threshold * 0.5:
            multiplier *= 0.7  # Strong opposing CVD - significantly reduce
    
    # OI change multiplier (relative to current OI, not hardcoded threshold)
    # Get current OI from volume_data or use a reasonable estimate
    if oi_change != 0:
        oi_change_pct = abs(oi_change) / max(abs(oi_change) * 100, 1_000_000_000)  # Rough estimate
        # More sensitive: 0.3% OI change is significant
        if oi_change_pct > 0.5 or abs(oi_change) > 5_000_000:  # Lower threshold: 5M instead of 10M
            multiplier *= 1.15
        elif abs(oi_change) > 2_000_000:  # Additional tier for moderate changes
            multiplier *= 1.08
    
    # VWAP multiplier (DIRECTIONAL: mean-reversion vs trend-following)
    # FIX: Previously abs(price - vwap) always boosted, even when dangerous (overextended)
    # NOW: Direction-aware logic prevents boosting overextended positions
    if vwap is not None and vwap > 0 and price > 0:
        vwap_dev_pct = abs(price - vwap) / vwap * 100
        
        if verdict == 'BUY':
            # BUY signals: boost when price BELOW vwap (mean reversion potential)
            if price < vwap and vwap_dev_pct > 1.0:
                multiplier *= 1.15  # Strong mean reversion setup
            elif price < vwap and vwap_dev_pct > 0.5:
                multiplier *= 1.08  # Moderate mean reversion
            # Reduce when price ABOVE vwap (potential overextension)
            elif price > vwap and vwap_dev_pct > 1.0:
                multiplier *= 0.85  # Overextended - reduce target
        
        else:  # SELL signals
            # SELL signals: boost when price ABOVE vwap (mean reversion potential)
            if price > vwap and vwap_dev_pct > 1.0:
                multiplier *= 1.15  # Strong mean reversion setup
            elif price > vwap and vwap_dev_pct > 0.5:
                multiplier *= 1.08  # Moderate mean reversion
            # Reduce when price BELOW vwap (potential overextension)
            elif price < vwap and vwap_dev_pct > 1.0:
                multiplier *= 0.85  # Overextended - reduce target
    
    # Step 2b: REMOVED magnitude_mult to eliminate double-counting
    # Previously: multiplier (CVD/OI/Volume) × magnitude_mult (same CVD/OI/Volume) = over-inflation
    # Now: Use ONLY multiplier for market strength
    # magnitude_mult deprecated and returns 1.0 (see calculate_magnitude_score)
    magnitude_mult = 1.0  # Neutral - no longer double-counting
    
    # No longer combining - just use multiplier directly
    combined_multiplier = multiplier  # Simplified: magnitude_mult always 1.0
    
    # Step 2c: Cap combined multiplier to prevent over-extension
    # DATA-DRIVEN CAP (from 863 signal analysis):
    # - Q1 (multiplier ≤1.2): +0.40% profit, 62.9% WR ✅
    # - Q4 (multiplier >1.5): +0.17% profit, 28.6% WR ❌
    # Conclusion: Lower cap = better performance (prevents over-aggressive targets)
    intraday_coins = ['YFIUSDT', 'LUMIAUSDT', 'ANIMEUSDT', 'DOGEUSDT']  # DOGEUSDT treated as intraday
    max_multiplier = 1.8 if symbol in intraday_coins else 1.3  # Reduced from 2.0/1.6 to 1.8/1.3
    combined_multiplier = min(combined_multiplier, max_multiplier)
    
    # Use combined_multiplier for final calculations instead of just multiplier
    multiplier = combined_multiplier
    
    # Step 3: Calculate dynamic TTL based on market conditions
    volume_ratio = 1.0
    if volume_data:
        volume_last = volume_data.get('last', 0)
        volume_median = volume_data.get('median', 1)
        volume_ratio = volume_last / volume_median if volume_median > 0 else 1.0
    
    oi_current = volume_data.get('oi_current', 1_000_000_000) if volume_data else 1_000_000_000
    
    ttl_minutes, ttl_multiplier, base_interval = calculate_dynamic_ttl(
        symbol, atr, price, volume_ratio, oi_change, oi_current, cvd, cvd_threshold, verdict, multiplier
    )
    
    # Step 4: Determine strength label based on TTL (shorter = stronger)
    if ttl_minutes <= base_interval * 0.85:
        strength_icon = "🔥"
        strength_label = "Very Strong"
    elif ttl_minutes <= base_interval * 1.1:
        strength_icon = "⚡"
        strength_label = "Strong"
    else:
        strength_icon = "⏱"
        strength_label = "Extended"
    
    # Format duration string
    if ttl_minutes >= 60:
        hours = ttl_minutes // 60
        minutes = ttl_minutes % 60
        duration = f"{hours}h{minutes}m" if minutes > 0 else f"{hours}h"
    else:
        duration = f"{ttl_minutes}min"
    
    # Step 5: Calculate final targets
    # Min = 50% of ATR baseline × multiplier
    # Max = 100% of ATR baseline × multiplier
    final_min = base_target_pct * 0.5 * multiplier
    final_max = base_target_pct * 1.0 * multiplier
    
    # Ensure minimum targets (prevent too-small targets)
    final_min = max(0.2, final_min)
    final_max = max(0.4, final_max)
    
    return (final_min, final_max, duration, f"{final_min:.1f}-{final_max:.1f}%", multiplier, strength_icon, strength_label, ttl_minutes, base_interval)

def format_signal_telegram(s: dict)->str:
    liq=s['liq_summary']; arr='🟢' if s['verdict']=='BUY' else ('🔴' if s['verdict']=='SELL' else '⚪️')
    confidence = int(s['confidence']*100)
    
    # Determine signal quality based on confidence
    if s['verdict'] == 'BUY':
        if confidence >= 60:
            quality = "🟢 Excellent"
        elif confidence >= 50:
            quality = "🟡 Good"
        elif confidence >= 40:
            quality = "🟠 Fair"
        else:
            quality = "🔴 Weak"
    else:  # SELL
        if confidence >= 75:
            quality = "🟢 Excellent"
        elif confidence >= 65:
            quality = "🟡 Good"
        elif confidence >= 55:
            quality = "🟠 Fair"
        else:
            quality = "🔴 Weak"
    
    # Header (without interval)
    out=[f"{arr} <b>{s['symbol']}</b> — <b>{s['verdict']}</b>"]
    
    # Add market regime indicator (compact)
    regime = s.get('regime', 'unknown')
    regime_icons = {
        'strong_bull': '🐂',
        'bull_warning': '⚠️🐂',
        'neutral': '➡️',
        'sideways': '↔️',
        'bear_warning': '⚠️🐻',
        'strong_bear': '🐻'
    }
    regime_icon = regime_icons.get(regime, '❓')
    
    # Compact header line with confidence, quality, and regime
    out.append(f"{quality} {confidence}% | {regime_icon} {regime.replace('_', ' ').title()}")
    
    # Determine decimal places based on price
    # < $1: 4 decimals (e.g., DOGE: 0.1974)
    # $1-$10: 4 decimals (e.g., XRP: 2.4100, TRX: 0.3200)
    # $10-$100: 2 decimals (e.g., LINK: 17.50)
    # $100+: 2 decimals (e.g., BTC: 109,332.00)
    price = s['last_close']
    if price < 10:
        decimals = 4
    else:
        decimals = 2
    
    # === BLOCK 1: ENTRY & TARGET ===
    out.append("")  # Separator
    
    # Price with VWAP comparison
    vwap = s.get('vwap_ref')
    if vwap and vwap > 0:
        vwap_diff_pct = ((price - vwap) / vwap) * 100
        vwap_indicator = "↗️" if vwap_diff_pct > 0 else "↘️"
        out.append(f"💰 Entry: <code>{price:.{decimals}f}</code> {vwap_indicator} VWAP ({vwap_diff_pct:+.2f}%)")
    else:
        out.append(f"💰 Entry: <code>{price:.{decimals}f}</code>")
    
    # Add price targets for BUY/SELL signals
    if s['verdict'] in ['BUY', 'SELL']:
        min_pct, max_pct, duration, move_str, multiplier, strength_icon, strength_label, ttl_minutes, base_interval = calculate_price_targets(
            s['last_close'], 
            s['confidence'], 
            s['cvd'], 
            s['symbol'], 
            s.get('coin_config'),
            klines=s.get('klines'),
            volume_data=s.get('volume'),
            oi_change=s.get('oi_change', 0),
            verdict=s['verdict'],  # Pass verdict for direction-aware CVD multiplier
            vwap=s.get('vwap_ref')  # Pass VWAP for directional mean-reversion logic
        )
        
        # Store TTL and base_interval in signal dict for tracking and logging
        s['ttl_minutes'] = ttl_minutes
        s['base_interval'] = base_interval
        s['target_pct'] = max_pct  # Store target % for AI comparison
        
        # Calculate actual price targets
        if s['verdict'] == 'BUY':
            target_min = s['last_close'] * (1 + min_pct / 100)
            target_max = s['last_close'] * (1 + max_pct / 100)
            out.append(f"🎯 Target: <code>{target_min:.{decimals}f} - {target_max:.{decimals}f}</code> ({move_str})")
        else:  # SELL
            target_min = s['last_close'] * (1 - min_pct / 100)
            target_max = s['last_close'] * (1 - max_pct / 100)
            out.append(f"🎯 Target: <code>{target_min:.{decimals}f} - {target_max:.{decimals}f}</code> ({move_str})")
        
        # Market strength and duration
        out.append(f"{strength_icon} {strength_label} ({multiplier:.2f}x) | ⏱ {duration}")
    
    # Calculate volume percentage change vs median
    vol_last = s['volume']['last']
    vol_median = s['volume']['median']
    vol_pct_change = ((vol_last - vol_median) / vol_median * 100) if vol_median > 0 else 0
    vol_pct_str = f"{vol_pct_change:+.0f}%" if vol_median > 0 else ""
    
    # Get active components for bold formatting
    comp = s.get('components', {})
    
    # Check which indicators are supporting this signal
    cvd_active = comp.get('CVD_pos', False) or comp.get('CVD_neg', False)
    
    # VWAP should only be bold when supporting the signal direction
    # Bot uses MEAN REVERSION: BUY when price < VWAP, SELL when price > VWAP
    if s['verdict'] == 'BUY':
        vwap_active = comp.get('VWAP_cross_up', False) or comp.get('Price_below_VWAP', False)
    elif s['verdict'] == 'SELL':
        vwap_active = comp.get('VWAP_cross_down', False) or comp.get('Price_above_VWAP', False)
    else:
        vwap_active = False
    
    oi_active = comp.get('OI_up', False) or comp.get('OI_down', False)
    vol_active = comp.get('Vol_spike', False)
    
    # Liquidations should only be bold when supporting the signal direction
    # BUY: More shorts liquidated (forced buying) = Bullish
    # SELL: More longs liquidated (forced selling) = Bearish
    if s['verdict'] == 'BUY':
        liq_active = comp.get('Liq_short', False)
    elif s['verdict'] == 'SELL':
        liq_active = comp.get('Liq_long', False)
    else:
        liq_active = False
    
    # === GROUPED INDICATORS DISPLAY (8 total) ===
    
    # Extract additional indicator values from signal dict (not components)
    rsi = s.get('rsi')
    ema_short = s.get('ema_short')
    ema_long = s.get('ema_long')
    atr = s.get('atr')
    
    # Check which additional indicators are active
    rsi_active = comp.get('RSI_oversold', False) or comp.get('RSI_overbought', False)
    ema_active = comp.get('EMA_cross_up', False) or comp.get('EMA_cross_down', False)
    
    # === BLOCK 2: FLOW INDICATORS (CVD, OI with direction) ===
    out.append("")  # Separator
    
    # OI Direction indicator
    oi_change = s.get('oi_change', 0)
    oi_direction = "📈" if oi_change > 0 else "📉" if oi_change < 0 else "➡️"
    
    # CVD with direction
    cvd = s.get('cvd', 0)

    # Есть ли вообще осмысленные данные по CVD?
    # 0 или почти 0 трактуем как "по сути нет сигнала по потоку ордеров"
    has_cvd_data = isinstance(cvd, (int, float)) and abs(cvd) >= 1

    cvd_direction = "🟢" if cvd > 0 else "🔴" if cvd < 0 else "⚪️"

    if not has_cvd_data:
        # Нет нормального CVD → показываем N/A, чтобы не путать с "реальным нулём"
        cvd_text = "CVD: N/A"
        if oi_active:
            # OI важен → подсветим его, а CVD оставим как есть
            cvd_oi_line = f"{cvd_text} | <b>{oi_direction} OI: {_human_int(oi_change)}</b>"
        else:
            cvd_oi_line = f"{cvd_text} | {oi_direction} OI: {_human_int(oi_change)}"
    else:
        if cvd_active and oi_active:
            cvd_oi_line = f"<b>{cvd_direction} CVD: {_human_int(cvd)} | {oi_direction} OI: {_human_int(oi_change)}</b>"
        elif cvd_active:
            cvd_oi_line = f"<b>{cvd_direction} CVD: {_human_int(cvd)}</b> | {oi_direction} OI: {_human_int(oi_change)}"
        elif oi_active:
            cvd_oi_line = f"{cvd_direction} CVD: {_human_int(cvd)} | <b>{oi_direction} OI: {_human_int(oi_change)}</b>"
        else:
            cvd_oi_line = f"{cvd_direction} CVD: {_human_int(cvd)} | {oi_direction} OI: {_human_int(oi_change)}"

    
    # === BLOCK 3: TECHNICAL INDICATORS (compact) ===
    # EMA trend
    if ema_short is not None and ema_long is not None:
        ema_trend = "↗️" if ema_short > ema_long else "↘️"
        ema_line = f"<b>{ema_trend} EMA</b>" if ema_active else f"{ema_trend} EMA"
    else:
        ema_line = "EMA: N/A"
    
    # RSI compact
    if rsi is not None:
        if rsi >= 70:
            rsi_icon = "🔴"
        elif rsi <= 30:
            rsi_icon = "🟢"
        else:
            rsi_icon = "⚪️"
        rsi_line = f"<b>{rsi_icon} RSI {rsi:.0f}</b>" if rsi_active else f"{rsi_icon} RSI {rsi:.0f}"
    else:
        rsi_line = "RSI: N/A"
    
    # Volume compact
    vol_spike_indicator = "⚡️" if s['volume']['spike'] else ""
    if vol_active:
        vol_line = f"<b>{vol_spike_indicator}Vol {vol_pct_str}</b>"
    else:
        vol_line = f"{vol_spike_indicator}Vol {vol_pct_str}" if vol_pct_str else "Vol: N/A"
    
    # Combine technical indicators in one line
    tech_line = f"{ema_line} | {rsi_line} | {vol_line}"
    
    # Liquidations - only if significant
    total_liq_usd = liq['long_usd'] + liq['short_usd']
    if total_liq_usd > 10000 or liq_active:
        if liq_active:
            liq_line = f"<b>💸 Liq: {liq['long_count']}L/{liq['short_count']}S</b>"
        else:
            liq_line = f"💸 Liq: {liq['long_count']}L/{liq['short_count']}S"
    else:
        liq_line = None
    
    # Build compact grouped output
    out.append(cvd_oi_line)
    out.append(tech_line)
    if liq_line:
        out.append(liq_line)
    
    if s['verdict']=='NO_TRADE': out.append('Reason: <i>conditions not aligned</i>')
    return '\n'.join(out)
