import time, yaml, os, csv, json, fcntl, sys, atexit, uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from signals import decide_signal
from signals.formatting import format_signal_telegram
from signals.scoring import calculate_price_targets
from signals.features import DataFeedUnavailable
from telegram_utils import send_telegram_message
from signal_tracker import ActiveSignalsManager, log_cancelled_signal, format_effectiveness_report
from services.ai_analyst.runner import AIAnalystService
load_dotenv()

LOG_FILE='analysis_log.csv'
SIGNAL_FILE='signals_log.csv'
TRACKING_FILE='sent_signals.json'
ACTIVE_SIGNALS_FILE='active_signals.json'
PID_FILE='signal_bot.pid'

# Global AI Analyst instance (initialized in main)
ai_analyst = None

# Throttle noisy alerts when market data is unavailable
_LAST_FEED_ALERT_TS = 0
_FEED_ALERT_COOLDOWN = 30 * 60  # seconds


def alert_data_feed_unavailable(error_message: str) -> None:
    """Send a single Telegram alert when upstream market data is unreachable."""
    global _LAST_FEED_ALERT_TS
    now = time.time()
    if now - _LAST_FEED_ALERT_TS < _FEED_ALERT_COOLDOWN:
        return

    _LAST_FEED_ALERT_TS = now
    try:
        send_telegram_message(
            "⚠️ <b>Data feed unavailable</b>\n"
            "Signals are paused because market data could not be reached.\n"
            f"Last error: {error_message}\n"
            "Check network / proxy access to coinalyze.net and Binance."
        )
    except Exception as e:
        print(f"[ALERT WARN] Failed to send data feed alert: {e}")

def acquire_lock():
    """Ensure only one instance of the bot is running"""
    try:
        pid_file = open(PID_FILE, 'w')
        fcntl.flock(pid_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        pid_file.write(str(os.getpid()))
        pid_file.flush()
        
        def cleanup():
            try:
                fcntl.flock(pid_file.fileno(), fcntl.LOCK_UN)
                pid_file.close()
                if os.path.exists(PID_FILE):
                    os.remove(PID_FILE)
            except:
                pass
        
        atexit.register(cleanup)
        return pid_file
    except IOError:
        print('[ERROR] Another instance of Smart Money Signal Bot is already running!')
        print('[ERROR] Only one instance is allowed to prevent duplicate signals.')
        print('[ERROR] If you are sure no other instance is running, delete signal_bot.pid and try again.')
        sys.exit(1)

# Two-bar confirmation buffer for SELL signals: {symbol: {regime: deque([score_prev, score_curr], maxlen=2)}}
# HYBRID REGIME SUPPORT: Tracks scores separately for each regime (strong_bear, bear_warning, strong_bull, bull_warning, neutral, sideways)
# Buffers are cleared when regime changes to prevent cross-regime confirmation
SELL_CONFIRM_STATE = defaultdict(lambda: defaultdict(lambda: deque(maxlen=2)))

def init_logs():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE,'w',encoding='utf-8',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['timestamp','symbol','interval','verdict','confidence','score','min_score','max_score','price','vwap','price_vs_vwap_pct','cvd','oi','oi_change','oi_change_pct','volume','volume_median','volume_spike','liq_long_count','liq_short_count','liq_long_usd','liq_short_usd','liq_ratio','funding_rate','rsi','ema_short','ema_long','atr','ttl_minutes','base_interval','regime','vwap_cross_up','vwap_cross_down','ema_cross_up','ema_cross_down','adx','confirm2_passed','vwap_sigma','dev_sigma','dev_sigma_blocked','dev_sigma_boost','ab_set_used','quote_vol_pctl','boost_applied','gate_action','min_score_delta','sell_enabled','basis_pct','basis_age_sec','basis_score_component','adx14','adx14_score_component','psar','psar_score_component','momentum5','momentum5_score_component','vol_accel','vol_accel_score_component','zcvd','zcvd_score_component','doi_pct','doi_pct_score_component','dev_sigma_uif','dev_sigma_uif_score_component','rsi_dist','rsi_dist_score_component'])
    if not os.path.exists(SIGNAL_FILE):
        with open(SIGNAL_FILE,'w',encoding='utf-8',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['timestamp','symbol','interval','verdict','confidence','score','min_score','max_score','entry_price','vwap','oi','oi_change','volume_spike','liq_long','liq_short','components','ttl_minutes','target_min','target_max','signal_id'])

def load_sent_signals():
    """Load tracking of previously sent signals (append-only list)"""
    if not os.path.exists(TRACKING_FILE):
        return []
    try:
        with open(TRACKING_FILE, 'r') as f:
            data = json.load(f)
            # Handle both old dict format and new list format
            if isinstance(data, dict):
                # Convert old dict format to list
                return list(data.values())
            return data
    except:
        return []

def save_sent_signals(tracking):
    """Save tracking of sent signals"""
    try:
        with open(TRACKING_FILE, 'w') as f:
            json.dump(tracking, f, indent=2)
    except Exception as e:
        print(f'[WARN] Failed to save tracking: {e}')

def load_active_signals():
    """Load active signals being tracked for effectiveness with file locking"""
    if not os.path.exists(ACTIVE_SIGNALS_FILE):
        return []
    try:
        with open(ACTIVE_SIGNALS_FILE, 'r') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                data = json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return data
    except (json.JSONDecodeError, ValueError) as e:
        print(f'[WARN] JSON decode error in active signals: {e}')
        return []
    except Exception as e:
        print(f'[WARN] Error loading active signals: {e}')
        return []

def save_active_signals(signals):
    """
    Save active signals with atomic write protected by exclusive lock.
    CRITICAL: Must coordinate with signal_tracker.py to prevent lost updates.
    """
    import tempfile
    from pathlib import Path
    
    # Create lock file if it doesn't exist
    lock_file_path = ACTIVE_SIGNALS_FILE + '.lock'
    Path(lock_file_path).touch(exist_ok=True)
    
    try:
        # Acquire exclusive lock to prevent concurrent read-modify-write
        with open(lock_file_path, 'r') as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                # Write to temp file first
                temp_fd, temp_path = tempfile.mkstemp(
                    dir=os.path.dirname(ACTIVE_SIGNALS_FILE) or '.', 
                    prefix='.active_signals_tmp_', 
                    suffix='.json'
                )
                try:
                    with os.fdopen(temp_fd, 'w') as f:
                        json.dump(signals, f, indent=2)
                        f.flush()
                        os.fsync(f.fileno())
                    
                    # Atomic rename overwrites destination (POSIX guarantees atomicity)
                    os.replace(temp_path, ACTIVE_SIGNALS_FILE)
                except Exception as e:
                    # Clean up temp file if write/rename fails
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    raise e
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        print(f'[WARN] Failed to save active signals: {e}')

def register_signal_for_tracking(res, cfg, telegram_msg_id=None):
    """Register a signal for real-time effectiveness tracking"""
    try:
        min_pct, max_pct, duration, move_str, multiplier, strength_icon, strength_label, ttl_minutes, base_interval = calculate_price_targets(
            res['last_close'],
            res['confidence'],
            res['cvd'],
            res['symbol'],
            res.get('coin_config'),
            klines=res.get('klines'),
            volume_data=res.get('volume'),
            oi_change=res.get('oi_change', 0),
            verdict=res['verdict']
        )
        
        if res['verdict'] == 'BUY':
            target_min = res['last_close'] * (1 + min_pct / 100)
            target_max = res['last_close'] * (1 + max_pct / 100)
        else:
            target_min = res['last_close'] * (1 - max_pct / 100)
            target_max = res['last_close'] * (1 - min_pct / 100)
        
        # Use dynamic TTL from calculate_price_targets
        duration_minutes = ttl_minutes
        
        signal_data = {
            'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': res['symbol'],
            'verdict': res['verdict'],
            'confidence': res['confidence'],
            'entry_price': res['last_close'],
            'target_min': target_min,
            'target_max': target_max,
            'duration_minutes': duration_minutes,
            'market_strength': multiplier,
            'highest_reached': res['last_close'],
            'lowest_reached': res['last_close'],
            'telegram_msg_id': telegram_msg_id,
            'rsi': res.get('rsi'),
            'ema_short': res.get('ema_short'),
            'ema_long': res.get('ema_long'),
            'adx': res.get('adx'),
            'funding_rate': res.get('funding_rate'),
            'signal_id': res.get('signal_id', ''),  # CRITICAL: signal_id for reliable tracking lookup
            'regime': res.get('regime', 'neutral')  # CRITICAL: regime for cancellation logic
        }
        
        # Use context manager for atomic read-modify-write
        with ActiveSignalsManager(ACTIVE_SIGNALS_FILE) as active_signals:
            active_signals.append(signal_data)
        
        print(f'[TRACK] Registered {res["symbol"]} {res["verdict"]} for effectiveness tracking (duration: {duration_minutes}min)')
        
    except Exception as e:
        print(f'[WARN] Failed to register signal for tracking: {e}')

def append_analysis_log(res: dict):
    try:
        ts=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'); liq=res.get('liq_summary',{}); vol=res.get('volume',{})
        price=res.get('last_close',0); vwap=res.get('vwap_ref',0); oi=res.get('oi_now',0); oip=res.get('oi_prev',0)
        price_vs_vwap=res.get('price_vs_vwap_pct', round(((price/vwap)-1)*100,2) if vwap>0 else 0)
        oi_chg=res.get('oi_change',0); oi_chg_pct=round((oi_chg/oip)*100,2) if oip and oip>0 else 0
        liq_ratio=round(liq.get('short_count',0)/max(liq.get('long_count',1),1),2)
        funding_rate=res.get('funding_rate',0); rsi=res.get('rsi') if res.get('rsi') is not None else 0
        ema_short=res.get('ema_short') if res.get('ema_short') is not None else 0
        ema_long=res.get('ema_long') if res.get('ema_long') is not None else 0
        atr=res.get('atr') if res.get('atr') is not None else 0
        ttl_minutes=res.get('ttl_minutes',0)
        base_interval=res.get('base_interval',0)
        # New fields for enhanced analysis
        regime=res.get('regime','neutral')
        vwap_cross_up=1 if res.get('vwap_cross_up', False) else 0
        vwap_cross_down=1 if res.get('vwap_cross_down', False) else 0
        ema_cross_up=1 if res.get('ema_cross_up', False) else 0
        ema_cross_down=1 if res.get('ema_cross_down', False) else 0
        adx=res.get('adx') if res.get('adx') is not None else 0
        # Two-bar confirmation field (1/0 for SELL evaluations, NA for BUY/NO_TRADE)
        confirm2_passed=res.get('confirm2_passed', 'NA')
        # VWAP sigma fields (professional institutional metrics)
        vwap_sigma=res.get('vwap_sigma', 0)
        dev_sigma=res.get('dev_sigma', 0)
        # Dev_sigma filter tracking fields
        dev_sigma_blocked=res.get('dev_sigma_blocked', 0)
        dev_sigma_boost=res.get('dev_sigma_boost', 0.0)
        ab_set_used=res.get('ab_set_used', 'B_default')
        # Boost guard-rail tracking fields
        quote_vol_pctl=res.get('quote_vol_pctl', 0.0)
        boost_applied=res.get('boost_applied', 0)
        # Quality gate tracking fields
        gate_action=res.get('gate_action', 'none')
        min_score_delta=res.get('min_score_delta', 0.0)
        sell_enabled=res.get('sell_enabled', 1)
        # UIF-30: Basis tracking fields (from Data Feeds Service snapshot)
        basis_pct=res.get('basis_pct')
        basis_age_sec=res.get('basis_age_sec')
        basis_score_component=res.get('basis_score_component', 0.0)
        # UIF-30: 8 UIF features (raw values + score components)
        uif_components=res.get('uif_score_components', {})
        adx14=res.get('adx14', 0.0)
        adx14_score_component=uif_components.get('adx14', 0.0)
        psar=res.get('psar', 0)
        psar_score_component=uif_components.get('psar', 0.0)
        momentum5=res.get('momentum5', 0.0)
        momentum5_score_component=uif_components.get('momentum5', 0.0)
        vol_accel=res.get('vol_accel', 0.0)
        vol_accel_score_component=uif_components.get('vol_accel', 0.0)
        zcvd=res.get('zcvd', 0.0)
        zcvd_score_component=uif_components.get('zcvd', 0.0)
        doi_pct=res.get('doi_pct', 0.0)
        doi_pct_score_component=uif_components.get('doi_pct', 0.0)
        dev_sigma_uif=res.get('dev_sigma_uif', 0.0)
        dev_sigma_uif_score_component=uif_components.get('dev_sigma_uif', 0.0)
        rsi_dist=res.get('rsi_dist', 0.0)
        rsi_dist_score_component=uif_components.get('rsi_dist', 0.0)
        with open(LOG_FILE,'a',encoding='utf-8',newline='') as f:
            writer=csv.writer(f)
            writer.writerow([ts,res.get('symbol'),res.get('interval'),res.get('verdict'),res.get('confidence'),res.get('score',0),res.get('min_score',0),res.get('max_score',0),price,vwap,price_vs_vwap,res.get('cvd',0),oi,oi_chg,oi_chg_pct,vol.get('last',0),vol.get('median',0),vol.get('spike',False),liq.get('long_count',0),liq.get('short_count',0),liq.get('long_usd',0),liq.get('short_usd',0),liq_ratio,funding_rate,rsi,ema_short,ema_long,atr,ttl_minutes,base_interval,regime,vwap_cross_up,vwap_cross_down,ema_cross_up,ema_cross_down,adx,confirm2_passed,vwap_sigma,dev_sigma,dev_sigma_blocked,dev_sigma_boost,ab_set_used,quote_vol_pctl,boost_applied,gate_action,min_score_delta,sell_enabled,basis_pct,basis_age_sec,basis_score_component,adx14,adx14_score_component,psar,psar_score_component,momentum5,momentum5_score_component,vol_accel,vol_accel_score_component,zcvd,zcvd_score_component,doi_pct,doi_pct_score_component,dev_sigma_uif,dev_sigma_uif_score_component,rsi_dist,rsi_dist_score_component])
    except Exception as e: print(f'[WARN] analysis log failed: {e}')

def append_signal_log(res: dict):
    if res.get('verdict')=='NO_TRADE': return
    try:
        # Generate signal_id if not already present (for timestamp consistency with sent_signals.json)
        if 'signal_id' not in res:
            res['signal_id'] = uuid.uuid4().hex
        
        ts=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'); liq=res.get('liq_summary',{}); comp=res.get('components',{})
        comp_str='|'.join([k for k,v in comp.items() if v])
        ttl_minutes=res.get('ttl_minutes',0)
        
        # DEDUPLICATION: Check last entry to avoid writing duplicate signals
        # This prevents the same signal from being logged twice in consecutive rows
        try:
            if os.path.exists(SIGNAL_FILE):
                with open(SIGNAL_FILE, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if len(lines) > 1:  # Has header + at least one data row
                        last_line = lines[-1].strip()
                        if last_line:
                            parts = last_line.split(',')
                            if len(parts) >= 4:
                                last_ts, last_sym, last_interval, last_verdict = parts[0], parts[1], parts[2], parts[3]
                                current_key = (ts, res.get('symbol'), res.get('interval'), res.get('verdict'))
                                last_key = (last_ts, last_sym, last_interval, last_verdict)
                                
                                if current_key == last_key:
                                    print(f'[SIGNAL LOG] Skipping duplicate: {res.get("symbol")} {res.get("verdict")} @ {ts}')
                                    return
        except Exception as e:
            # If dedup check fails, log warning but continue with write
            print(f'[WARN] Dedup check failed: {e}')
        
        # Calculate target_min and target_max for trading signals (matches register_signal_for_tracking logic)
        try:
            min_pct, max_pct, _, _, _, _, _, _, _ = calculate_price_targets(
                res['last_close'],
                res['confidence'],
                res['cvd'],
                res['symbol'],
                res.get('coin_config'),
                klines=res.get('klines'),
                volume_data=res.get('volume'),
                oi_change=res.get('oi_change', 0),
                verdict=res['verdict'],
                vwap=res.get('vwap_ref')
            )
        except Exception as e:
            # If calculate_price_targets fails, use default values based on confidence
            print(f'[WARN] calculate_price_targets failed for {res.get("symbol")}: {e}')
            # Default targets: 0.3-0.7% for BUY, 0.2-0.4% for SELL (conservative)
            if res['verdict'] == 'BUY':
                min_pct = 0.3
                max_pct = 0.7
            else:
                min_pct = 0.2
                max_pct = 0.4
        
        if res['verdict'] == 'BUY':
            target_min = res['last_close'] * (1 + min_pct / 100)
            target_max = res['last_close'] * (1 + max_pct / 100)
        else:
            # SELL: target_min is MORE aggressive (lower price, larger % down)
            target_min = res['last_close'] * (1 - max_pct / 100)
            target_max = res['last_close'] * (1 - min_pct / 100)
        
        with open(SIGNAL_FILE,'a',encoding='utf-8',newline='') as f:
            writer=csv.writer(f)
            writer.writerow([ts,res.get('symbol'),res.get('interval'),res.get('verdict'),res.get('confidence'),res.get('score',0),res.get('min_score',0),res.get('max_score',0),res.get('last_close'),res.get('vwap_ref'),res.get('oi_now'),res.get('oi_change'),res.get('volume',{}).get('spike',False),liq.get('long_count',0),liq.get('short_count',0),comp_str,ttl_minutes,target_min,target_max,res.get('signal_id', '')])
    except Exception as e: print(f'[WARN] signal log failed: {e}')

def check_cancellation(symbol, res, tracking, cfg):
    """
    Check if an active signal should be cancelled based on market conditions.
    
    Cancellation criteria:
    1. Confidence drops below 30% (absolute threshold)
    2. Opposite signal with ≥70% confidence
    3. Regime change to opposite direction:
       - BUY signal: Bull/Strong Bull → Bear/Strong Bear
       - SELL signal: Bear/Strong Bear → Bull/Strong Bull
       - Any signal: Neutral → Strong Bull/Strong Bear (exit from sideways)
    
    Args:
        symbol: Trading symbol
        res: Current signal analysis result
        tracking: Tracking data (deprecated, using active_signals.json instead)
        cfg: Configuration
    
    Returns:
        True if signal should be cancelled, False otherwise
    """
    # Find active signal for this symbol with matching verdict
    active_signal = None
    try:
        with ActiveSignalsManager(ACTIVE_SIGNALS_FILE) as active_signals:
            for sig in active_signals:
                # Match by symbol AND verdict (BUY vs SELL)
                if sig['symbol'] == symbol and sig['verdict'] == res.get('verdict', 'NO_TRADE'):
                    active_signal = sig
                    break
    except Exception as e:
        print(f'[CANCEL WARN] Failed to load active signals: {e}')
        return False
    
    # No active signal to cancel (or different verdict)
    if not active_signal:
        return False
    
    # Skip cancellation check if current verdict is NO_TRADE
    # (only check when we have a real signal to compare against)
    if res.get('verdict') == 'NO_TRADE':
        return False
    
    # Extract data from active signal
    original_confidence = active_signal['confidence']
    original_verdict = active_signal['verdict']
    
    # Get regime from stored signal data (if available, otherwise use current)
    original_regime = active_signal.get('regime', 'neutral')
    
    # Extract current analysis data
    current_confidence = res.get('confidence', 0)
    current_verdict = res.get('verdict', 'NO_TRADE')
    current_regime = res.get('regime', 'neutral')
    
    cancellation_reason = None
    
    # CRITERION 1: Confidence drops BELOW 30% (absolute threshold)
    if current_confidence < 30:
        cancellation_reason = f"Confidence упала до {current_confidence:.0f}% (&lt; 30%)"
    
    # CRITERION 2: Opposite signal with ≥70% confidence
    elif current_verdict != 'NO_TRADE' and current_verdict != original_verdict:
        if current_confidence >= 70:
            cancellation_reason = f"Противоположный сигнал {current_verdict} с {current_confidence:.0f}% confidence (&ge; 70%)"
    
    # CRITERION 3: Regime change to opposite direction
    else:
        # Define regime categories
        bull_regimes = ['bull', 'strong_bull', 'bull_warning', 'bull_trend']
        bear_regimes = ['bear', 'strong_bear', 'bear_warning', 'bear_trend']
        strong_regimes = ['strong_bull', 'strong_bear']
        
        # For BUY signals: check if regime changed from bullish to bearish
        if original_verdict == 'BUY':
            # Was in bullish regime, now in bearish
            if original_regime in bull_regimes and current_regime in bear_regimes:
                cancellation_reason = f"Regime change {original_regime} → {current_regime} (against BUY)"
            # Was neutral, now ANY bearish (exit from sideways to Bear/Strong Bear)
            elif original_regime in ['neutral', 'sideways'] and current_regime in bear_regimes:
                cancellation_reason = f"Exit from neutral to {current_regime} (against BUY)"
        
        # For SELL signals: check if regime changed from bearish to bullish
        elif original_verdict == 'SELL':
            # Was in bearish regime, now in bullish
            if original_regime in bear_regimes and current_regime in bull_regimes:
                cancellation_reason = f"Regime change {original_regime} → {current_regime} (against SELL)"
            # Was neutral, now ANY bullish (exit from sideways to Bull/Strong Bull)
            elif original_regime in ['neutral', 'sideways'] and current_regime in bull_regimes:
                cancellation_reason = f"Exit from neutral to {current_regime} (against SELL)"
    
    # If cancellation triggered, log and send notification
    if cancellation_reason:
        print(f'[CANCELLATION] {symbol} {original_verdict} ({original_confidence:.0f}%): {cancellation_reason}')
        
        # Log cancelled signal to effectiveness_log.csv
        try:
            result_data = log_cancelled_signal(active_signal)
            
            # Send Telegram notification about cancellation
            from telegram_utils import send_cancellation_notification
            telegram_msg_id = active_signal.get('telegram_msg_id')
            print(f'[CANCEL TELEGRAM] telegram_msg_id={telegram_msg_id}')
            if telegram_msg_id:
                print(f'[CANCEL TELEGRAM] Sending cancellation notification for {symbol} {original_verdict}...')
                msg_id = send_cancellation_notification(
                    active_signal,
                    result_data,
                    cancellation_reason,
                    reply_to_message_id=telegram_msg_id
                )
                if msg_id:
                    print(f'[CANCEL TELEGRAM] ✅ Sent (msg_id: {msg_id})')
                else:
                    print(f'[CANCEL TELEGRAM] ❌ Failed to send')
            else:
                print(f'[CANCEL TELEGRAM] ⚠️ No telegram_msg_id found')
            
            # Remove from active signals
            with ActiveSignalsManager(ACTIVE_SIGNALS_FILE) as active_signals:
                active_signals[:] = [s for s in active_signals if s['symbol'] != symbol]
            
        except Exception as e:
            print(f'[CANCEL ERROR] Failed to process cancellation: {e}')
        
        return True
    
    return False

def run_once(cfg, tracking, gate_results=None):
    """Run one iteration of signal analysis for all symbols with optional quality gate results"""
    symbols=cfg['symbols']; interval=cfg.get('interval','15m'); lookback=int(cfg.get('lookback_minutes',15)); vwap_window=int(cfg.get('vwap_window',30)); volume_spike_mult=float(cfg.get('volume_spike_mult',1.6)); min_components=int(cfg.get('min_components',2))
    if gate_results is None:
        gate_results = {}
    feed_failures = 0
    last_feed_error = ''
    for idx, sym in enumerate(symbols):
        try:
            # Pass full config to decide_signal for weighted scoring
            start_time = time.time()
            res=decide_signal(sym, interval, config=cfg, lookback_minutes=lookback, vwap_window=vwap_window, volume_spike_mult=volume_spike_mult, min_components=min_components)
            
            # SHADOW MODE: Log dual-formula evaluation for A/B comparison (non-blocking)
            from shadow_integration import evaluate_with_shadow_mode
            res = evaluate_with_shadow_mode(res, start_time)
            
            # TWO-BAR CONFIRMATION FOR SELL ONLY (Don't touch BUY logic)
            current_regime = res.get('regime', 'neutral')
            current_score = res.get('score', 0)
            original_verdict = res.get('verdict', 'NO_TRADE')
            
            # Use min_score as threshold (already calculated from max_score * min_score_pct in decide_signal)
            threshold = res.get('min_score', 0)
            
            # Apply two-bar confirmation gate for SELL signals only
            # HYBRID REGIME SUPPORT:
            # - strong_bear, bear_warning: Apply 2-bar confirmation (bearish regimes)
            # - strong_bull, bull_warning, sideways: Block SELL entirely (bullish/neutral regimes)
            # - neutral: Apply 2-bar confirmation
            if original_verdict == 'SELL':
                # Define regime categories
                bearish_regimes = ('strong_bear', 'bear_warning', 'bear_trend')  # backwards compat: bear_trend
                bullish_regimes = ('strong_bull', 'bull_warning', 'bull_trend')  # backwards compat: bull_trend
                
                # Block SELL in bullish regimes
                if current_regime in bullish_regimes or current_regime == 'sideways':
                    res['confirm2_passed'] = 0
                    res['verdict'] = 'NO_TRADE'
                    res['confidence'] = 0.0
                    print(f'[SELL BLOCKED] 🚫 {sym} {current_regime}: SELL not allowed in {current_regime}')
                # Apply 2-bar confirmation in bearish or neutral regimes
                else:
                    # Get buffer for this symbol
                    regime_buffer = SELL_CONFIRM_STATE[sym]
                    
                    # Check if regime changed - if so, clear old regime buffers
                    for old_regime in list(regime_buffer.keys()):
                        if old_regime != current_regime:
                            regime_buffer[old_regime].clear()
                    
                    # Get score buffer for current regime
                    score_buffer = regime_buffer[current_regime]
                    
                    # Check if we have previous score in same regime (use -1 to get most recent previous bar)
                    if len(score_buffer) >= 1 and score_buffer[-1] is not None:
                        prev_score = score_buffer[-1]  # Most recent previous bar (immediate predecessor)
                        # Both current and previous must exceed threshold
                        if current_score > threshold and prev_score > threshold:
                            res['confirm2_passed'] = 1
                            print(f'[SELL CONFIRM2] ✅ {sym} {current_regime}: scores prev={prev_score:.2f} curr={current_score:.2f} both > {threshold:.2f}')
                        else:
                            res['confirm2_passed'] = 0
                            res['verdict'] = 'NO_TRADE'  # Block SELL signal
                            res['confidence'] = 0.0
                            print(f'[SELL BLOCKED] ❌ {sym} {current_regime}: scores prev={prev_score:.2f} curr={current_score:.2f} (need both > {threshold:.2f})')
                    else:
                        # First bar in this regime - store but don't emit
                        res['confirm2_passed'] = 0
                        res['verdict'] = 'NO_TRADE'
                        res['confidence'] = 0.0
                        print(f'[SELL BLOCKED] ⏸️  {sym} {current_regime}: first bar, score={current_score:.2f} (need 2 consecutive > {threshold:.2f})')
                    
                    # Update buffer with current score (append to right, maxlen=2 auto-pops left)
                    score_buffer.append(current_score)
            else:
                # BUY or NO_TRADE - no confirmation needed
                res['confirm2_passed'] = 'NA'
            
            # Add quality gate results if available
            if sym in gate_results:
                res['gate_action'] = gate_results[sym]['gate_action']
                res['min_score_delta'] = gate_results[sym]['min_score_delta']
                res['sell_enabled'] = gate_results[sym]['sell_enabled']
            
            text=format_signal_telegram(res); print(text)
            # CRITICAL: Only log analysis (not signals) before Telegram send
            append_analysis_log(res)
            
            # Check if we should cancel a previous signal
            cancelled = check_cancellation(sym, res, tracking, cfg)
            
            # Send new signal if valid (and not just cancelled)
            if res['verdict']!='NO_TRADE' and not cancelled:
                # CRITICAL: Validate fresh data before broadcasting to prevent sending signals when market has reversed
                from signals.features import validate_signal_momentum_fresh
                is_still_valid = validate_signal_momentum_fresh(
                    symbol=sym,
                    verdict=res['verdict'],
                    original_cvd=res.get('cvd', 0)
                )
                
                if not is_still_valid:
                    print(f'[SIGNAL BLOCKED] {sym} {res["verdict"]}: Market momentum reversed during signal generation - NOT sending to Telegram')
                    continue
                
                message_id = send_telegram_message(text)
                
                # CRITICAL FIX: Only track signal AND log to CSV if Telegram succeeded
                # This prevents signal_tracker from adding signals with telegram_msg_id=0
                # which breaks reply-to functionality
                if message_id:
                    # NOW write to signals_log.csv (AFTER Telegram success)
                    append_signal_log(res)
                    
                    # Track this signal for potential future cancellation (append-only list)
                    tracking.append({
                        'symbol': sym,
                        'message_id': message_id,
                        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                        'verdict': res['verdict'],
                        'confidence': res.get('confidence', 0),
                        'entry_price': res.get('last_close', 0),
                        'signal_id': res.get('signal_id', '')  # CRITICAL: signal_id for reply-to lookup
                    })
                    save_sent_signals(tracking)
                    
                    # Register for real-time effectiveness tracking (with message_id for reply-to alerts)
                    register_signal_for_tracking(res, cfg, telegram_msg_id=message_id)
                    
                    # AI Analyst: Generate market context as reply to signal
                    if ai_analyst and ai_analyst.enabled:
                        try:
                            features = {
                                'price': res.get('last_close', 0),
                                'vwap': res.get('vwap_ref', 0),
                                'vwap_dist_pct': ((res.get('last_close', 0) - res.get('vwap_ref', 0)) / res.get('vwap_ref', 1)) * 100 if res.get('vwap_ref') else 0,
                                'dev_sigma': res.get('dev_sigma', 0),
                                'cvd': res.get('cvd', 0),
                                'oi_change': res.get('oi_change', 0),
                                'funding_rate': res.get('funding_rate', 0),
                                'basis_pct': res.get('basis_pct', 0),
                                'rsi': res.get('rsi', 0),
                                'ema_trend': res.get('ema_trend', 'unknown'),
                                'regime': res.get('regime', 'unknown'),
                                'volume_spike': res.get('volume', {}).get('spike_pct', 0),
                                'liq_long': res.get('liq_summary', {}).get('long_count', 0),
                                'liq_short': res.get('liq_summary', {}).get('short_count', 0),
                                'adx14': res.get('adx14', 0),
                                'psar': res.get('psar', 0),
                                'momentum5': res.get('momentum5', 0),
                                'vol_accel': res.get('vol_accel', 0),
                                'ttl_minutes': res.get('ttl_minutes', 0),
                                'target_pct': res.get('target_pct', 0)
                            }
                            ai_analyst.generate_signal_context(
                                symbol=sym,
                                verdict=res['verdict'],
                                confidence=res.get('confidence', 0),
                                features=features,
                                signal_id=res.get('signal_id', ''),
                                message_id=message_id
                            )
                        except Exception as e:
                            print(f'[AI ANALYST] Failed to generate context: {e}')
                else:
                    print(f'[TELEGRAM FAIL] {sym} {res["verdict"]}: Signal generation completed but Telegram send failed - NOT logged to CSV or tracked')
            
            # Add delay between symbols to avoid API rate limiting (skip delay after last symbol)
            # Set to 6 seconds for 11 symbols (66s total = ~33 calls per minute, safely under 40/min limit)
            # Coinalyze free tier: 40 API calls/minute, each symbol makes ~3 API calls
            if idx < len(symbols) - 1:
                time.sleep(6.0)
        except DataFeedUnavailable as e:
            feed_failures += 1
            last_feed_error = str(e)
            print(f"[FEED DOWN] {sym}: {e}")
        except Exception as e: print(f"[ERR] {sym}: {e}")

    # If all symbols failed due to market data outages, alert once and skip signal push
    if feed_failures == len(symbols) and feed_failures > 0:
        alert_data_feed_unavailable(last_feed_error)

def main():
    global ai_analyst
    acquire_lock()
    init_logs(); cfg=yaml.safe_load(open('config.yaml','r',encoding='utf-8'))
    tracking = load_sent_signals()
    
    # Initialize AI Analyst
    try:
        ai_analyst = AIAnalystService('config.yaml')
        if ai_analyst.enabled:
            print('[INFO] AI Analyst: ENABLED')
        else:
            print('[INFO] AI Analyst: DISABLED (check config or API key)')
    except Exception as e:
        print(f'[WARN] AI Analyst initialization failed: {e}')
        ai_analyst = None
    
    if cfg.get('run_once',False): 
        run_once(cfg, tracking)
        return
    
    # Get daily report settings from config (MVP freeze)
    report_config = cfg.get('report', {})
    daily_time_utc = report_config.get('time_utc', '23:59')
    quality_gates_enabled = report_config.get('quality_gates_enabled', False)
    brier_enabled = report_config.get('brier_enabled', False)
    
    interval_min=1  # Signal generation interval (different from candle timeframe)
    print(f'[INFO] Smart Money Futures Signal Bot (MVP) started. Signal Generation: every {interval_min} min | Candle timeframe: {cfg.get("interval")}')
    print(f'[INFO] Monitoring {len(cfg["symbols"])} symbols: {", ".join(cfg["symbols"])}')
    print(f'[INFO] Logs: {LOG_FILE} (all analysis), {SIGNAL_FILE} (signals only)')
    print(f'[INFO] Tracking: {TRACKING_FILE} (sent signals for cancellation)')
    print(f'[INFO] Hourly effectiveness reports: DISABLED (handled by Signal Tracker)')
    print(f'[INFO] Daily basic reports: ENABLED (UTC {daily_time_utc})')
    print(f'[INFO] Quality gates: {"ENABLED" if quality_gates_enabled else "DISABLED (MVP freeze)"}')
    print(f'[INFO] Brier scores: {"ENABLED" if brier_enabled else "DISABLED (MVP freeze)"}')
    
    # Track last daily report date (initialize to yesterday to enable first run)
    last_daily_report_date = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    
    while True:
        run_once(cfg, tracking)
        
        # NOTE: Hourly effectiveness report is now handled by signal_tracker.py to avoid duplicates
        
        # Check if it's time to send daily basic report (MVP - no quality gates/brier)
        current_utc = datetime.now(timezone.utc)
        current_date = current_utc.date()
        current_hour = current_utc.hour
        current_minute = current_utc.minute
        
        # Parse report time from config (default 18:59 UTC for MVP)
        report_hour, report_minute = map(int, daily_time_utc.split(':'))
        
        # Send daily report at configured time (once per day)
        if current_date != last_daily_report_date and current_hour == report_hour and current_minute >= report_minute:
            try:
                from daily_report import send_daily_report
                send_daily_report()
                last_daily_report_date = current_date
                print(f'[DAILY_REPORT] ✅ Daily basic report sent for {current_date}')
            except Exception as e:
                print(f'[DAILY_REPORT ERROR] Failed to send daily report: {e}')
        
        now=int(time.time())
        target=((now//(interval_min*60))+1)*(interval_min*60)
        sleep_s=max(10,target-now)
        print(f'[INFO] Next run in {int(sleep_s)}s...')
        time.sleep(sleep_s)

if __name__=='__main__': main()
