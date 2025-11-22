import uuid
from typing import Dict, Tuple

import numpy as np
from order_flow_indicators import calculate_bid_ask_aggression, detect_psychological_levels

from .calibration import aggregate_recent_analysis, calibrate_confidence
from .features import (
    calculate_atr,
    calculate_volatility_based_interval,
    compute_adx,
    compute_cvd,
    compute_ema_crossover,
    compute_rsi,
    compute_volume_spike,
    compute_vwap_sigma,
    detect_regime_hybrid,
    detect_strict_vwap_cross,
    fetch_basis,
    fetch_funding_rate,
    fetch_klines,
    fetch_liquidations,
    fetch_open_interest,
    fetch_uif_snapshot,
    summarize_liquidations,
)


# Exported API
__all__ = ["decide_signal", "calculate_price_targets"]


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
