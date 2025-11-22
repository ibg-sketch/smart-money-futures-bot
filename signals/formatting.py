from .scoring import calculate_price_targets

def _human_int(n):
    try: return f"{int(round(float(n))):,}".replace(',',' ')
    except: return str(n)


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

__all__ = ['format_signal_telegram']