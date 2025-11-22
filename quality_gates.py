"""
Daily Quality Gates & Per-Symbol Auto-Tuning (Regime-Aware)

Applies rule-based adjustments to config.yaml based on daily KPI metrics
computed per-regime (bear_trend, sideways) for improved precision.

Features:
- Per-regime gate decisions (not global per symbol)
- Canary limit: max 2 SELL disables per day
- Shadow effect analysis before min_score increases
- Dry-run mode for simulation
- Alert-based suggestions for zero signals and high blocking
"""
import yaml
import json
import os
import sys
import csv
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from collections import defaultdict
from telegram_utils import send_telegram_message

HISTORY_FILE = 'quality_gates_history.json'

def load_history():
    """Load historical metrics for consecutive-day tracking"""
    if not os.path.exists(HISTORY_FILE):
        return {}
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_history(history):
    """Save historical metrics"""
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f'[QUALITY_GATES] Failed to save history: {e}')

def backup_config(config):
    """Create timestamped backup of config.yaml"""
    try:
        # Ensure configs/ directory exists
        Path('configs').mkdir(exist_ok=True)
        
        # Create backup filename with date
        date_str = datetime.now(timezone.utc).strftime('%Y%m%d')
        backup_path = f'configs/config_{date_str}.yaml'
        
        # Write backup
        with open(backup_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f'[QUALITY_GATES] Config backup saved to {backup_path}')
        return backup_path
    except Exception as e:
        print(f'[QUALITY_GATES] Failed to backup config: {e}')
        return None

def compute_regime_metrics(analysis_log_path='analysis_log.csv', effectiveness_log_path='effectiveness_log.csv'):
    """
    Compute metrics per (symbol, regime) combination from logs.
    
    Returns:
        dict: {symbol: {regime: {profit_factor, win_rate, blocked_share, signals_total, ...}}}
    """
    # Get today's date range (00:00 to 23:59 UTC)
    today = datetime.now(timezone.utc).date()
    today_start = datetime.combine(today, time.min, tzinfo=timezone.utc)
    today_end = datetime.combine(today, time.max, tzinfo=timezone.utc)
    
    # Data structures for regime-specific metrics
    regime_data = defaultdict(lambda: defaultdict(lambda: {
        'blocked_count': 0,
        'total_verdicts': 0,  # BUY + SELL + blocked
        'wins': 0,
        'losses': 0,
        'cancelled': 0,
        'win_pnl': 0.0,
        'loss_pnl': 0.0
    }))
    
    # Map signal IDs to (symbol, regime) for effectiveness matching
    signal_to_regime = {}
    
    # Parse analysis_log.csv for regime and blocked data
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
                    regime = row.get('regime', 'unknown')
                    
                    # Normalize regime to bear_trend or sideways
                    if regime in ['bear_trend', 'sideways']:
                        pass  # Keep as is
                    else:
                        regime = 'other'  # Group bull_trend, neutral, unknown
                    
                    # Track blocked signals
                    blocked = int(row.get('dev_sigma_blocked', 0))
                    if blocked == 1:
                        regime_data[symbol][regime]['blocked_count'] += 1
                    
                    # Count total potential verdicts (BUY, SELL, or blocked)
                    if verdict in ['BUY', 'SELL'] or blocked == 1:
                        regime_data[symbol][regime]['total_verdicts'] += 1
                    
                    # Track signal ID for regime matching
                    if verdict in ['BUY', 'SELL']:
                        signal_id = f"{symbol}_{ts.strftime('%Y%m%d_%H%M%S')}"
                        signal_to_regime[signal_id] = (symbol, regime)
                        
                except (ValueError, KeyError) as e:
                    continue
    except FileNotFoundError:
        print(f'[WARN] Quality gates: {analysis_log_path} not found')
    
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
                    
                    # Try to match to regime (fallback to 'other' if not found)
                    signal_id = f"{symbol}_{ts_sent.strftime('%Y%m%d_%H%M%S')}"
                    _, regime = signal_to_regime.get(signal_id, (symbol, 'other'))
                    
                    if outcome == 'win':
                        regime_data[symbol][regime]['wins'] += 1
                        regime_data[symbol][regime]['win_pnl'] += abs(pnl_pct)
                    elif outcome == 'loss':
                        regime_data[symbol][regime]['losses'] += 1
                        regime_data[symbol][regime]['loss_pnl'] += abs(pnl_pct)
                    elif outcome == 'cancelled':
                        regime_data[symbol][regime]['cancelled'] += 1
                        
                except (ValueError, KeyError) as e:
                    continue
    except FileNotFoundError:
        print(f'[WARN] Quality gates: {effectiveness_log_path} not found')
    
    # Compute metrics per (symbol, regime)
    metrics = defaultdict(dict)
    for symbol, regimes in regime_data.items():
        for regime, data in regimes.items():
            # Profit factor: sum(wins) / sum(losses)
            profit_factor = data['win_pnl'] / data['loss_pnl'] if data['loss_pnl'] > 0 else (
                999.0 if data['win_pnl'] > 0 else 0.0
            )
            
            # Win rate: wins / (wins + losses + cancelled)
            total_signals = data['wins'] + data['losses'] + data['cancelled']
            win_rate = data['wins'] / total_signals if total_signals > 0 else 0.0
            
            # Blocked share: blocked / (blocked + actual signals)
            blocked_share = data['blocked_count'] / data['total_verdicts'] if data['total_verdicts'] > 0 else 0.0
            
            metrics[symbol][regime] = {
                'profit_factor': profit_factor,
                'win_rate': win_rate,
                'blocked_share': blocked_share,
                'signals_total': total_signals
            }
    
    return metrics

def compute_shadow_effect(symbol, new_min_score, analysis_log_path='analysis_log.csv', effectiveness_log_path='effectiveness_log.csv'):
    """
    Compute shadow effect of increasing min_score: how many signals would be filtered
    and what the new PF estimate would be.
    
    Args:
        symbol: Trading symbol
        new_min_score: Proposed new min_score threshold
        
    Returns:
        dict: {shadow_filtered: int, shadow_pf: float}
    """
    # Get yesterday's date range
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    yesterday_start = datetime.datetime.combine(yesterday, datetime.time.min)
    yesterday_end = datetime.datetime.combine(yesterday, datetime.time.max)
    
    # Collect yesterday's signals with scores
    signals = []
    try:
        with open(analysis_log_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = datetime.datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
                    if not (yesterday_start <= ts <= yesterday_end):
                        continue
                    
                    if row['symbol'] != symbol:
                        continue
                    
                    verdict = row['verdict']
                    if verdict not in ['BUY', 'SELL']:
                        continue
                    
                    # CRITICAL FIX: Use score (not confidence) for min_score comparison
                    # min_score_pct operates on score field (values like 1.9, 2.4)
                    # confidence is in 0-1 range (0.82, 0.75, etc.)
                    score = float(row.get('score', 0.0))
                    confidence = float(row['confidence'])
                    signal_id = f"{symbol}_{ts.strftime('%Y%m%d_%H%M%S')}"
                    
                    signals.append({
                        'signal_id': signal_id,
                        'score': score,
                        'confidence': confidence
                    })
                    
                except (ValueError, KeyError):
                    continue
    except FileNotFoundError:
        return {'shadow_filtered': 0, 'shadow_pf': 0.0}
    
    # Match with effectiveness outcomes
    signal_outcomes = {}
    try:
        with open(effectiveness_log_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts_sent = datetime.datetime.strptime(row['timestamp_sent'], '%Y-%m-%d %H:%M:%S')
                    if not (yesterday_start <= ts_sent <= yesterday_end):
                        continue
                    
                    if row['symbol'] != symbol:
                        continue
                    
                    signal_id = f"{symbol}_{ts_sent.strftime('%Y%m%d_%H%M%S')}"
                    outcome = row['outcome']
                    pnl_pct = float(row.get('pnl_pct', 0.0))
                    
                    signal_outcomes[signal_id] = {
                        'outcome': outcome,
                        'pnl_pct': pnl_pct
                    }
                    
                except (ValueError, KeyError):
                    continue
    except FileNotFoundError:
        return {'shadow_filtered': 0, 'shadow_pf': 0.0}
    
    # Compute shadow metrics
    # CRITICAL: The verdict decision uses confidence >= min_score_pct (see smart_signal.py:1394, 1421)
    # So we compare confidence (0-1 range) against new_min_score (also 0-1 range percentage)
    
    filtered_count = 0
    new_win_pnl = 0.0
    new_loss_pnl = 0.0
    
    for sig in signals:
        signal_id = sig['signal_id']
        confidence = sig['confidence']
        
        # Would this signal pass the new threshold?
        # Compare confidence against new min_score_pct threshold
        if confidence < new_min_score:
            filtered_count += 1
            continue
        
        # Include in new PF calculation
        if signal_id in signal_outcomes:
            outcome_data = signal_outcomes[signal_id]
            outcome = outcome_data['outcome']
            pnl = abs(outcome_data['pnl_pct'])
            
            if outcome == 'win':
                new_win_pnl += pnl
            elif outcome == 'loss':
                new_loss_pnl += pnl
    
    # Calculate new PF
    shadow_pf = new_win_pnl / new_loss_pnl if new_loss_pnl > 0 else (
        999.0 if new_win_pnl > 0 else 0.0
    )
    
    return {
        'shadow_filtered': filtered_count,
        'shadow_pf': shadow_pf
    }

def apply_quality_gates(regime_metrics, dry_run=False):
    """
    Apply quality gates and auto-tuning rules based on per-regime daily KPI metrics.
    
    Args:
        regime_metrics: Dict of {symbol: {regime: {profit_factor, win_rate, ...}}}
        dry_run: If True, simulate without writing config
    
    Returns:
        Dict of {symbol: {regime: {gate_action, min_score_delta, sell_enabled, ...}}}
    """
    # Load config
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f'[QUALITY_GATES] Failed to load config: {e}')
        return {}
    
    # Create immutable backup BEFORE any mutations
    import copy
    original_config = copy.deepcopy(config)
    
    # Load history
    history = load_history()
    today = datetime.now(timezone.utc).date().isoformat()
    
    # Track SELL disables today (canary limit)
    disabled_today_count = 0
    
    # Initialize results
    results = defaultdict(dict)
    config_changed = False
    alerts = []
    suggestions = []
    
    for symbol in regime_metrics.keys():
        if symbol not in config.get('symbols', []):
            continue
        
        # Get coin config
        coin_cfg = config.get('coin_configs', {}).get(symbol, {})
        
        # Initialize history for this symbol if needed
        if symbol not in history:
            history[symbol] = {}
        
        # Process each regime separately
        for regime, metrics in regime_metrics[symbol].items():
            # Skip 'other' regime for gate decisions (not enough bearish/sideways data)
            if regime == 'other':
                continue
            
            # Initialize history for this (symbol, regime)
            regime_key = regime
            if regime_key not in history[symbol]:
                history[symbol][regime_key] = {
                    'profit_factor_history': [],
                    'blocked_share_history': []
                }
            
            # Initialize result for this (symbol, regime)
            result = {
                'gate_action': '',
                'min_score_delta': 0.0,
                'sell_enabled': 1 if coin_cfg.get('sell_enabled', True) else 0,
                'gate_scope': 'per_regime',
                'disabled_today_count': disabled_today_count,
                'shadow_filtered': 0,
                'shadow_pf': 0.0,
                'gate_shadow_effect': ''
            }
            
            actions = []
            
            # Get metrics
            profit_factor = metrics.get('profit_factor', 0.0)
            win_rate = metrics.get('win_rate', 0.0)
            blocked_share = metrics.get('blocked_share', 0.0)
            signals_total = metrics.get('signals_total', 0)
            
            # === ALERT 1: Zero signals warning ===
            if signals_total == 0:
                alert_msg = f'⚠️ <b>ZERO SIGNALS ALERT</b>\n\n'
                alert_msg += f'Symbol: <b>{symbol}</b>\n'
                alert_msg += f'Regime: <b>{regime}</b>\n'
                alert_msg += f'Issue: No signals generated today\n'
                alert_msg += f'Suggestion: Check market conditions or reduce min_score threshold'
                suggestions.append(alert_msg)
            
            # === RULE 1: If profit_factor < 1.10 OR win_rate < 0.35: increase min_score by +0.05 (max +0.15) ===
            if profit_factor < 1.10 or win_rate < 0.35:
                # Get current min_score (check both scalping and intraday)
                current_min_scalp = coin_cfg.get('min_score_pct', 0.75)
                current_min_intraday = coin_cfg.get('min_score_pct_intraday', 0.85)
                
                # Get total increase applied so far (track in history)
                total_increase = history[symbol][regime_key].get('total_min_score_increase', 0.0)
                
                if total_increase < 0.15:
                    # Apply +0.05 increase
                    delta = min(0.05, 0.15 - total_increase)
                    new_min_scalp = min(current_min_scalp + delta, 0.95)
                    new_min_intraday = min(current_min_intraday + delta, 0.98)
                    
                    # SHADOW EFFECT ANALYSIS: Compute before applying
                    shadow = compute_shadow_effect(symbol, new_min_scalp)
                    result['shadow_filtered'] = shadow['shadow_filtered']
                    result['shadow_pf'] = shadow['shadow_pf']
                    result['gate_shadow_effect'] = f"would_filter_{shadow['shadow_filtered']}_signals_pf_{shadow['shadow_pf']:.2f}"
                    
                    if not dry_run:
                        coin_cfg['min_score_pct'] = new_min_scalp
                        coin_cfg['min_score_pct_intraday'] = new_min_intraday
                        
                        history[symbol][regime_key]['total_min_score_increase'] = total_increase + delta
                        config_changed = True
                    
                    result['min_score_delta'] = delta
                    actions.append(f'min_score+{delta:.2f}')
                    
                    print(f'[QUALITY_GATES] {symbol}/{regime}: min_score +{delta:.2f} (shadow: filter {shadow["shadow_filtered"]} signals, new PF={shadow["shadow_pf"]:.2f})')
            
            # === RULE 2: If profit_factor < 1.00 for 2 consecutive days: disable SELL ===
            history[symbol][regime_key]['profit_factor_history'].append({
                'date': today,
                'value': profit_factor
            })
            
            # Keep only last 3 days
            history[symbol][regime_key]['profit_factor_history'] = history[symbol][regime_key]['profit_factor_history'][-3:]
            
            # Check consecutive days
            pf_history = history[symbol][regime_key]['profit_factor_history']
            if len(pf_history) >= 2:
                last_two = pf_history[-2:]
                if all(d['value'] < 1.00 for d in last_two):
                    # Check canary limit
                    if disabled_today_count < 2:
                        # Disable SELL if not already disabled
                        if coin_cfg.get('sell_enabled', True):
                            if not dry_run:
                                coin_cfg['sell_enabled'] = False
                                config_changed = True
                            
                            result['sell_enabled'] = 0
                            disabled_today_count += 1
                            result['disabled_today_count'] = disabled_today_count
                            actions.append('SELL_DISABLED')
                            
                            # Send Telegram alert
                            alert_msg = f'🚨 <b>QUALITY GATE ALERT</b>\n\n'
                            alert_msg += f'Symbol: <b>{symbol}</b>\n'
                            alert_msg += f'Regime: <b>{regime}</b>\n'
                            alert_msg += f'Action: <b>SELL signals DISABLED</b>\n'
                            alert_msg += f'Reason: profit_factor &lt; 1.00 for 2 consecutive days\n'
                            alert_msg += f'Last 2 days: {last_two[-2]["value"]:.2f}, {last_two[-1]["value"]:.2f}'
                            alerts.append(alert_msg)
                    else:
                        # Canary limit exceeded - send warning instead
                        alert_msg = f'🔶 <b>CANARY LIMIT ALERT</b>\n\n'
                        alert_msg += f'Symbol: <b>{symbol}</b>\n'
                        alert_msg += f'Regime: <b>{regime}</b>\n'
                        alert_msg += f'Issue: Would disable SELL but limit reached ({disabled_today_count}/2)\n'
                        alert_msg += f'PF last 2 days: {last_two[-2]["value"]:.2f}, {last_two[-1]["value"]:.2f}\n'
                        alert_msg += f'Action: Manual review recommended'
                        alerts.append(alert_msg)
                        actions.append('CANARY_LIMIT_REACHED')
            
            # === RULE 3: If blocked_share > 0.80 for 3 consecutive days: decrease block_below by 0.05 (floor 0.25) ===
            history[symbol][regime_key]['blocked_share_history'].append({
                'date': today,
                'value': blocked_share
            })
            
            # Keep only last 4 days
            history[symbol][regime_key]['blocked_share_history'] = history[symbol][regime_key]['blocked_share_history'][-4:]
            
            # Check consecutive days
            bs_history = history[symbol][regime_key]['blocked_share_history']
            if len(bs_history) >= 3:
                last_three = bs_history[-3:]
                if all(d['value'] > 0.80 for d in last_three):
                    # Decrease block_below threshold
                    dev_sigma_thresholds = coin_cfg.get('dev_sigma_thresholds', {})
                    current_block = dev_sigma_thresholds.get('block_below', 0.30)
                    
                    if current_block > 0.25:
                        new_block = max(current_block - 0.05, 0.25)
                        
                        if not dry_run:
                            dev_sigma_thresholds['block_below'] = new_block
                            coin_cfg['dev_sigma_thresholds'] = dev_sigma_thresholds
                            config_changed = True
                        
                        actions.append(f'block_below->{new_block:.2f}')
            
            # === ALERT 2: Suggest dev_sigma decrease for 2 consecutive days blocked_share > 0.90 ===
            if len(bs_history) >= 2:
                last_two_bs = bs_history[-2:]
                if all(d['value'] > 0.90 for d in last_two_bs):
                    dev_sigma_thresholds = coin_cfg.get('dev_sigma_thresholds', {})
                    current_block = dev_sigma_thresholds.get('block_below', 0.30)
                    suggested_block = max(current_block - 0.05, 0.25)
                    
                    alert_msg = f'💡 <b>DEV_SIGMA SUGGESTION</b>\n\n'
                    alert_msg += f'Symbol: <b>{symbol}</b>\n'
                    alert_msg += f'Regime: <b>{regime}</b>\n'
                    alert_msg += f'Issue: blocked_share &gt; 90% for 2 days ({last_two_bs[-2]["value"]*100:.0f}%, {last_two_bs[-1]["value"]*100:.0f}%)\n'
                    alert_msg += f'Suggestion: Decrease block_below from {current_block:.2f} to {suggested_block:.2f}\n'
                    alert_msg += f'Note: Not auto-applied, manual review recommended'
                    suggestions.append(alert_msg)
            
            # Update coin config if changed
            if config_changed and symbol in config.get('coin_configs', {}):
                config['coin_configs'][symbol] = coin_cfg
            
            # Set gate action
            result['gate_action'] = ','.join(actions) if actions else 'none'
            results[symbol][regime] = result
    
    # Save updated config if changes were made
    if config_changed and not dry_run:
        try:
            # Backup ORIGINAL config (before mutations)
            backup_config(original_config)
            
            # Write updated config
            with open('config.yaml', 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            action_count = sum(1 for sym_results in results.values() for r in sym_results.values() if r['gate_action'] != 'none')
            print(f'[QUALITY_GATES] Config updated with {action_count} gate actions')
        except Exception as e:
            print(f'[QUALITY_GATES] Failed to save config: {e}')
    
    # Save history (even in dry-run to track consecutive days)
    if not dry_run:
        save_history(history)
    
    # Send alerts if any
    for alert in alerts:
        try:
            if not dry_run:
                send_telegram_message(alert)
            else:
                print(f'[DRY-RUN] Would send alert: {alert[:100]}...')
        except Exception as e:
            print(f'[QUALITY_GATES] Failed to send alert: {e}')
    
    # Send suggestions if any
    for suggestion in suggestions:
        try:
            if not dry_run:
                send_telegram_message(suggestion)
            else:
                print(f'[DRY-RUN] Would send suggestion: {suggestion[:100]}...')
        except Exception as e:
            print(f'[QUALITY_GATES] Failed to send suggestion: {e}')
    
    # Print dry-run summary
    if dry_run:
        print('\n' + '='*60)
        print('DRY-RUN MODE: No changes written to config.yaml')
        print('='*60)
        for symbol, regimes in results.items():
            for regime, result in regimes.items():
                if result['gate_action'] != 'none':
                    print(f'{symbol}/{regime}: {result["gate_action"]}')
                    if result['shadow_filtered'] > 0:
                        print(f'  Shadow effect: would filter {result["shadow_filtered"]} signals, new PF={result["shadow_pf"]:.2f}')
        print('='*60 + '\n')
    
    return results

if __name__ == '__main__':
    # Check for --dry-run flag
    dry_run = '--dry-run' in sys.argv
    
    if dry_run:
        print('[QUALITY_GATES] Running in DRY-RUN mode (no changes will be saved)')
    
    # Compute regime-specific metrics
    regime_metrics = compute_regime_metrics()
    
    # Apply quality gates
    results = apply_quality_gates(regime_metrics, dry_run=dry_run)
    
    # Print results
    print(f'\n[QUALITY_GATES] Processed {len(results)} symbols')
    for symbol, regimes in results.items():
        for regime, result in regimes.items():
            if result['gate_action'] != 'none':
                print(f'  {symbol}/{regime}: {result["gate_action"]}')
