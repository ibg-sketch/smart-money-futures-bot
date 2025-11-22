"""
Alert Queue Manager - Persistent, fault-tolerant alert system
Ensures target zone and final goal alerts are NEVER missed.

Features:
- Persistent queue survives restarts
- Automatic retry with exponential backoff
- Full audit logging
- Thread-safe operations with file locking
"""

import csv
import fcntl
import json
import os
import time
from datetime import datetime, timezone
from telegram_utils import send_telegram_message

ALERTS_QUEUE_FILE = 'alerts_queue.json'
ALERT_LOG_FILE = 'alert_log.csv'
SENT_SIGNALS_FILE = 'sent_signals.json'
MAX_RETRY_ATTEMPTS = 5
RETRY_DELAYS = [10, 30, 60, 300, 900]  # 10s, 30s, 1m, 5m, 15m

def get_telegram_msg_id_by_signal_id(signal_id):
    """
    Lookup telegram_msg_id from sent_signals.json using signal_id.
    This ensures reply-to functionality works even if signal data is reloaded.
    
    Args:
        signal_id: UUID of the signal
        
    Returns:
        int: telegram_msg_id if found, None otherwise
    """
    if not signal_id:
        return None
    
    try:
        if os.path.exists(SENT_SIGNALS_FILE):
            with open(SENT_SIGNALS_FILE, 'r') as f:
                sent_signals = json.load(f)
                
            # Search for signal_id in sent_signals
            for entry in sent_signals:
                if entry.get('signal_id') == signal_id:
                    msg_id = entry.get('message_id')
                    if msg_id and msg_id > 0:
                        return msg_id
        return None
    except Exception as e:
        print(f"[ALERT] Error loading telegram_msg_id for signal_id {signal_id}: {e}")
        return None

class AlertQueueManager:
    """Thread-safe manager for persistent alert queue with file locking"""
    
    def __init__(self, queue_file=ALERTS_QUEUE_FILE):
        self.queue_file = queue_file
        self.lock_file = queue_file + '.lock'
        
        # Ensure lock file exists
        if not os.path.exists(self.lock_file):
            open(self.lock_file, 'w').close()
    
    def __enter__(self):
        """Acquire exclusive lock and load queue"""
        self.lock_fd = open(self.lock_file, 'r')
        fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_EX)
        
        # Load queue
        if os.path.exists(self.queue_file):
            try:
                with open(self.queue_file, 'r') as f:
                    self.queue = json.load(f)
            except:
                self.queue = []
        else:
            self.queue = []
        
        return self.queue
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Save queue and release lock"""
        try:
            # Write to temp file first
            temp_file = self.queue_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(self.queue, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            
            # Atomic rename
            os.replace(temp_file, self.queue_file)
        finally:
            fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
            self.lock_fd.close()

def initialize_alert_log():
    """Initialize alert log CSV if it doesn't exist"""
    if not os.path.exists(ALERT_LOG_FILE):
        with open(ALERT_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'symbol',
                'verdict',
                'alert_type',
                'status',
                'message_id',
                'attempts',
                'error',
                'signal_timestamp',
                'entry_price',
                'target_price'
            ])

def log_alert_attempt(alert_data, status, message_id=None, error=None):
    """Log an alert attempt to CSV for audit trail"""
    try:
        with open(ALERT_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                alert_data['symbol'],
                alert_data['verdict'],
                alert_data['alert_type'],
                status,
                message_id or '',
                alert_data.get('attempts', 0),
                error or '',
                alert_data.get('signal_timestamp', ''),
                alert_data.get('entry_price', 0),
                alert_data.get('target_price', 0)
            ])
    except Exception as e:
        print(f"[ALERT LOG ERROR] Failed to log alert: {e}")

def enqueue_alert(symbol, verdict, alert_type, signal_data, signal_id=None):
    """
    Add an alert to the persistent queue.
    
    Args:
        symbol: Trading pair (e.g., DOGEUSDT)
        verdict: BUY or SELL
        alert_type: 'target_zone' or 'final_goal'
        signal_data: Dict with entry_price, target_min/max, highest/lowest_reached
        signal_id: UUID of original signal (used to lookup telegram_msg_id)
        
    Note: target_price is stored for logging but recalculated at send time
          from active_signals.json to ensure accuracy with latest extremes
    """
    # Lookup telegram_msg_id from sent_signals.json using signal_id
    telegram_msg_id = get_telegram_msg_id_by_signal_id(signal_id) if signal_id else None
    
    alert = {
        'id': f"{symbol}_{alert_type}_{int(time.time()*1000)}",
        'symbol': symbol,
        'verdict': verdict,
        'alert_type': alert_type,
        'signal_timestamp': signal_data.get('timestamp', ''),
        'entry_price': signal_data.get('entry_price', 0),
        'highest_reached': signal_data.get('highest_reached', signal_data.get('entry_price', 0)),
        'lowest_reached': signal_data.get('lowest_reached', signal_data.get('entry_price', 0)),
        'target_min': signal_data.get('target_min', 0),
        'target_max': signal_data.get('target_max', 0),
        'signal_id': signal_id,  # Store signal_id for re-lookup if needed
        'telegram_msg_id': telegram_msg_id,
        'created_at': time.time(),
        'attempts': 0,
        'last_attempt': None,
        'status': 'pending'
    }
    
    with AlertQueueManager() as queue:
        # CRITICAL: Check for duplicates using symbol, alert_type, AND signal_id
        # This prevents re-queuing same alert after restart even if in-memory flags are lost
        existing = [a for a in queue if 
                   a['symbol'] == symbol and 
                   a['alert_type'] == alert_type and 
                   a.get('signal_id') == signal_id and
                   a['status'] in ('pending', 'failed')]
        if existing:
            print(f"[ALERT] {symbol} {alert_type} (signal_id: {signal_id}) already queued, skipping duplicate")
            return False
        
        queue.append(alert)
        print(f"[ALERT] Queued {symbol} {verdict} {alert_type} alert (signal_id: {signal_id}, msg_id: {telegram_msg_id})")
        log_alert_attempt(alert, 'QUEUED')
        return True

def update_alert_extremes(signal_id, highest_reached, lowest_reached):
    """
    Update extremes for pending/failed alerts matching signal_id.
    Called by signal_tracker to keep alert payloads fresh with latest prices.
    
    Args:
        signal_id: UUID of the signal to match
        highest_reached: Latest highest price
        lowest_reached: Latest lowest price
    """
    if not signal_id:
        return
    
    with AlertQueueManager() as queue:
        updated_count = 0
        for alert in queue:
            if (alert.get('signal_id') == signal_id and 
                alert['status'] in ('pending', 'failed')):
                alert['highest_reached'] = highest_reached
                alert['lowest_reached'] = lowest_reached
                updated_count += 1
        
        if updated_count > 0:
            print(f"[ALERT] Updated {updated_count} alert(s) with latest extremes (signal_id: {signal_id[:8]}...)")

def process_alert_queue():
    """
    Process all pending alerts in the queue.
    Attempts to send each alert, with retry logic for failures.
    
    Returns:
        int: Number of alerts successfully sent
    """
    sent_count = 0
    
    with AlertQueueManager() as queue:
        remaining_alerts = []
        
        for alert in queue:
            if alert['status'] != 'pending':
                continue
            
            # Check if we should retry based on last attempt time
            if alert['last_attempt']:
                attempt_num = alert['attempts']
                if attempt_num >= MAX_RETRY_ATTEMPTS:
                    # Max retries reached, mark as failed
                    alert['status'] = 'failed'
                    log_alert_attempt(alert, 'FAILED_MAX_RETRIES', error='Exceeded maximum retry attempts')
                    print(f"[ALERT FAILED] {alert['symbol']} {alert['alert_type']} - Max retries exceeded")
                    remaining_alerts.append(alert)
                    continue
                
                # Calculate next retry time
                retry_delay = RETRY_DELAYS[min(attempt_num, len(RETRY_DELAYS)-1)]
                next_retry_time = alert['last_attempt'] + retry_delay
                
                if time.time() < next_retry_time:
                    # Not time to retry yet
                    remaining_alerts.append(alert)
                    continue
            
            # Attempt to send alert
            success, message_id, error = send_alert(alert)
            
            alert['attempts'] += 1
            alert['last_attempt'] = time.time()
            
            if success:
                alert['status'] = 'sent'
                alert['message_id'] = message_id
                log_alert_attempt(alert, 'SENT', message_id=message_id)
                sent_count += 1
                print(f"[ALERT SENT] {alert['symbol']} {alert['alert_type']} (msg_id: {message_id})")
                # Don't add to remaining - successfully sent alerts are removed from queue
            else:
                # Failed, will retry
                log_alert_attempt(alert, 'RETRY', error=error)
                remaining_alerts.append(alert)
                print(f"[ALERT RETRY] {alert['symbol']} {alert['alert_type']} - Attempt {alert['attempts']}/{MAX_RETRY_ATTEMPTS}")
        
        # Update queue with only pending/failed alerts
        queue.clear()
        queue.extend(remaining_alerts)
    
    return sent_count

def send_alert(alert):
    """
    Send a single alert via Telegram.
    Fetches latest extreme prices from active_signals.json for accuracy.
    Re-lookups telegram_msg_id from signal_id to ensure reply-to works.
    
    Returns:
        tuple: (success: bool, message_id: int|None, error: str|None)
    """
    try:
        symbol = alert['symbol']
        verdict = alert['verdict']
        alert_type = alert['alert_type']
        entry_price = alert['entry_price']
        target_min = alert['target_min']
        target_max = alert['target_max']
        
        # CRITICAL: Re-lookup telegram_msg_id from signal_id
        # This ensures we get the latest telegram_msg_id even if sent_signals.json
        # was updated after the alert was enqueued
        signal_id = alert.get('signal_id')
        telegram_msg_id = get_telegram_msg_id_by_signal_id(signal_id) if signal_id else None
        
        # Fallback to stored telegram_msg_id if lookup fails
        if not telegram_msg_id:
            telegram_msg_id = alert.get('telegram_msg_id')
        
        # Use extreme prices from alert payload
        # These are updated in signal_tracker before every queue processing cycle
        if verdict == 'BUY':
            target_price = alert.get('highest_reached', entry_price)
        else:
            target_price = alert.get('lowest_reached', entry_price)
        
        # Calculate profit
        if verdict == 'BUY':
            profit_pct = ((target_price - entry_price) / entry_price) * 100
        else:
            profit_pct = ((entry_price - target_price) / entry_price) * 100
        
        # Format alert message
        if alert_type == 'target_zone':
            if verdict == 'BUY':
                message = (
                    f"🎯 <b>TARGET ZONE REACHED</b>\n\n"
                    f"<b>{symbol} {verdict}</b>\n"
                    f"✅ Price hit target zone: <b>${target_price:.4f}</b>\n"
                    f"💰 Profit: <b>+{profit_pct:.2f}%</b>\n"
                    f"📍 Target range: ${target_min:.4f} - ${target_max:.4f}\n"
                    f"⏰ Consider taking partial profits"
                )
            else:
                message = (
                    f"🎯 <b>TARGET ZONE REACHED</b>\n\n"
                    f"<b>{symbol} {verdict}</b>\n"
                    f"✅ Price hit target zone: <b>${target_price:.4f}</b>\n"
                    f"💰 Profit: <b>+{profit_pct:.2f}%</b>\n"
                    f"📍 Target range: ${target_max:.4f} - ${target_min:.4f}\n"
                    f"⏰ Consider taking partial profits"
                )
        else:  # final_goal
            message = (
                f"🏆 <b>FINAL GOAL ACHIEVED</b>\n\n"
                f"<b>{symbol} {verdict}</b>\n"
                f"✅ Price hit final target: <b>${target_price:.4f}</b>\n"
                f"💰 Profit: <b>+{profit_pct:.2f}%</b>\n"
                f"🎯 Maximum target reached!\n"
                f"⏰ Recommended: Take profits"
            )
        
        # Send via Telegram with reply-to
        print(f"[ALERT SEND] {symbol} {alert_type} - Using reply_to={telegram_msg_id} (signal_id: {signal_id})")
        msg_id = send_telegram_message(message, reply_to_message_id=telegram_msg_id)
        
        if msg_id:
            return True, msg_id, None
        else:
            return False, None, "Telegram API returned no message_id"
    
    except Exception as e:
        return False, None, str(e)

def get_queue_status():
    """
    Get current alert queue status.
    
    Returns:
        dict: Statistics about pending/sent/failed alerts
    """
    try:
        with AlertQueueManager() as queue:
            pending = [a for a in queue if a['status'] == 'pending']
            failed = [a for a in queue if a['status'] == 'failed']
            
            return {
                'total': len(queue),
                'pending': len(pending),
                'failed': len(failed),
                'pending_alerts': pending
            }
    except:
        return {'total': 0, 'pending': 0, 'failed': 0, 'pending_alerts': []}

# Initialize log on module import
initialize_alert_log()
