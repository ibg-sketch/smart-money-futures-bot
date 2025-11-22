"""
Main Trading Service
Paper Trading Mode
"""
import os
import time
import traceback
from typing import Dict
from datetime import datetime
from .config import TradingConfig, PaperTradingConfig
from .bingx_client import BingXClient
from .position_manager import PositionManager
from .risk_manager import RiskManager
from .signal_reader import SignalReader
from .telegram_notifier import TelegramNotifier
from .trade_logger import TradeLogger
from .cancellation_monitor import CancellationMonitor
from .effectiveness_reporter import format_bingx_effectiveness_report

class TradingService:
    def __init__(self):
        print(f"🚀 Initializing {TradingConfig.EXCHANGE} Trading Service...")
        print(f"Mode: {TradingConfig.MODE}")
        print(f"TP Strategy: {TradingConfig.TP_STRATEGY}")
        
        self.client = BingXClient()
        self.risk_manager = RiskManager()
        self.position_manager = PositionManager(self.client, self.risk_manager)
        self.signal_reader = SignalReader(TradingConfig.SIGNAL_SOURCE)
        self.telegram = TelegramNotifier()
        self.trade_logger = TradeLogger(TradingConfig.TRADES_LOG)
        self.cancellation_monitor = CancellationMonitor('effectiveness_log.csv')
        
        self.running = True
        self.last_hourly_report_hour = -1
        
        print("✅ All components initialized (including cancellation monitor)")
    
    def start(self):
        print(f"\n{'='*60}")
        print(f"🟢 TRADING SERVICE STARTED")
        print(f"{'='*60}")
        print(f"Mode: {TradingConfig.MODE}")
        print(f"Exchange: {TradingConfig.EXCHANGE}")
        print(f"Leverage: {TradingConfig.LEVERAGE}x")
        
        if TradingConfig.MODE == "PAPER" and PaperTradingConfig.ALL_IN_MODE:
            balance = self.risk_manager.get_current_balance()
            print(f"💰 ALL-IN MODE: Trading full balance (${balance:.2f})")
        else:
            print(f"Position Size: ${TradingConfig.POSITION_SIZE_USD}")
        
        print(f"TP Strategy: {TradingConfig.TP_STRATEGY}")
        print(f"Trading Pairs: {', '.join(TradingConfig.TRADING_PAIRS)}")
        print(f"Max Positions: {TradingConfig.MAX_CONCURRENT_POSITIONS}")
        print(f"Min Confidence: {TradingConfig.MIN_CONFIDENCE}%")
        print(f"{'='*60}\n")
        
        if TradingConfig.MODE == "PAPER" and PaperTradingConfig.ALL_IN_MODE:
            balance = self.risk_manager.get_current_balance()
            position_msg = f"<b>💰 ALL-IN MODE:</b> Full balance (${balance:.2f})"
        else:
            position_msg = f"<b>Position Size:</b> ${TradingConfig.POSITION_SIZE_USD}"
        
        self.telegram.send_message(
            f"🟢 <b>TRADING SERVICE STARTED</b>\n\n"
            f"<b>Mode:</b> {TradingConfig.MODE}\n"
            f"<b>Exchange:</b> {TradingConfig.EXCHANGE}\n"
            f"<b>TP Strategy:</b> {TradingConfig.TP_STRATEGY}\n"
            f"<b>Leverage:</b> {TradingConfig.LEVERAGE}x\n"
            f"{position_msg}\n\n"
            f"<b>🔔 Auto-close on signal cancellation: ENABLED</b>\n\n"
            f"<i>Monitoring signals...</i>"
        )
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n⚠️  Shutting down gracefully...")
            self.stop()
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            traceback.print_exc()
            self.telegram.notify_error(f"Fatal error: {str(e)}")
            self.stop()
    
    def _main_loop(self):
        while self.running:
            try:
                self._process_signals()
                
                self._check_cancelled_signals()
                
                self._monitor_positions()
                
                self._send_hourly_report()
                
                time.sleep(TradingConfig.POLL_INTERVAL_SECONDS)
            
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                traceback.print_exc()
                time.sleep(10)
    
    def _process_signals(self):
        signal = self.signal_reader.get_latest_signal()
        
        if signal:
            print(f"\n📊 New signal: {signal['symbol']} {signal['verdict']} ({float(signal['confidence'])*100:.0f}%)")
            
            # Filter by allowed signal types
            if signal['verdict'] not in TradingConfig.ALLOWED_SIGNAL_TYPES:
                print(f"⏭️  Skipping {signal['verdict']} signal (only {', '.join(TradingConfig.ALLOWED_SIGNAL_TYPES)} allowed)")
                return
            
            position = self.position_manager.open_position(signal)
            
            if position:
                print(f"✅ Position opened: {position['symbol']} {position['side']}")
                print(f"   Entry: ${position['entry_price']:.4f}")
                print(f"   TP: ${position['tp_price']:.4f} | SL: ${position['sl_price']:.4f}")
                
                # Send Telegram notification and save message_id
                msg_id = self.telegram.notify_position_opened(
                    symbol=position['symbol'],
                    side=position['side'],
                    entry_price=position['entry_price'],
                    tp_price=position['tp_price'],
                    sl_price=position['sl_price'],
                    size=position['position_size_usd'],
                    confidence=position['confidence'],
                    strategy=position['tp_strategy']
                )
                
                # Update position with telegram_msg_id for reply-to mechanism
                if msg_id:
                    self.position_manager.update_telegram_msg_id(position, msg_id)
                    print(f"   📱 Telegram msg_id saved: {msg_id}")
    
    def _check_cancelled_signals(self):
        """Check for completed signals (WIN, LOSS, CANCELLED) and handle them."""
        completed_signals = self.cancellation_monitor.get_new_cancellations()
        
        if not completed_signals:
            return
        
        positions = self.position_manager.get_active_positions()
        
        if not positions:
            return
        
        for signal in completed_signals:
            result_type = signal['result']
            
            # Format message based on result type
            if result_type == 'CANCELLED':
                event_msg = f"🔔 Signal CANCELLED"
            else:
                event_msg = f"⏱️  TTL EXPIRED ({result_type})"
            
            print(f"\n{event_msg}: {signal['symbol']} {signal['verdict']} "
                  f"({signal['confidence']*100:.0f}%) at {signal['timestamp_cancelled']}")
            
            # Find matching open position
            matched_position = None
            print(f"   🔍 Searching for matching position among {len(positions)} active positions...")
            print(f"      Signal: {signal['symbol']} {signal['verdict']} @ {signal['timestamp_sent']}")
            
            for position in positions:
                print(f"      Checking: {position['symbol']} {position['side']} @ {position.get('signal_timestamp', 'N/A')}")
                if self.cancellation_monitor.match_position_to_cancellation(position, signal):
                    matched_position = position
                    print(f"      ✅ MATCH FOUND!")
                    break
            
            if not matched_position:
                print(f"   ℹ️  No open position found for completed signal")
                pos_list = [f"{p['symbol']} {p['side']} @ {p.get('signal_timestamp', 'N/A')}" for p in positions]
                print(f"      Available positions: {pos_list}")
                continue
            
            # Close position for BOTH CANCELLED signals and TTL EXPIRED signals
            try:
                current_price = self._get_current_price(matched_position['symbol'])
                
                # Determine close reason
                if result_type == 'CANCELLED':
                    close_reason = "Signal Cancelled"
                    print(f"   🔴 Closing position due to signal cancellation...")
                else:
                    close_reason = "TTL Expired"
                    print(f"   🔴 Closing position due to TTL expiry...")
                
                print(f"      Position: {matched_position['symbol']} {matched_position['side']}")
                print(f"      Entry: ${matched_position['entry_price']:.4f} → Exit: ${current_price:.4f}")
                
                trade_result = self.position_manager.close_position(
                    matched_position,
                    current_price,
                    close_reason
                )
                
                self.trade_logger.log_trade(trade_result)
                
                active_count = len(self.position_manager.get_active_positions())
                today_pnl = self._calculate_today_pnl()
                
                print(f"   {'✅' if trade_result['actual_profit_usd'] > 0 else '❌'} Position closed")
                print(f"      Profit: ${trade_result['actual_profit_usd']:+.2f} ({trade_result['actual_profit_pct']:+.2f}%)")
                
                # Different notification based on result type
                if result_type == 'CANCELLED':
                    notification_reason = f"❌ Signal Cancelled"
                else:
                    notification_reason = f"⏱️ TTL Expired ({result_type})"
                
                # Send notification to Telegram with reply-to original message
                current_balance = self.risk_manager.get_current_balance() if TradingConfig.MODE == "PAPER" else None
                self.telegram.notify_position_closed(
                    symbol=matched_position['symbol'],
                    side=matched_position['side'],
                    entry_price=matched_position['entry_price'],
                    exit_price=current_price,
                    profit_usd=trade_result['actual_profit_usd'],
                    profit_pct=trade_result['actual_profit_pct'],
                    duration_minutes=trade_result['duration_minutes'],
                    reason=notification_reason,
                    open_positions=active_count,
                    today_pnl=today_pnl,
                    current_balance=current_balance,
                    reply_to_message_id=matched_position.get('telegram_msg_id')
                )
                
            except Exception as e:
                print(f"   ❌ Error closing position for completed signal: {e}")
                traceback.print_exc()
    
    def _monitor_positions(self):
        positions = self.position_manager.get_active_positions()
        
        if not positions:
            return
        
        for position in positions:
            try:
                current_price = self._get_current_price(position['symbol'])
                
                self.position_manager.update_position_extremes(position, current_price)
                
                should_close, reason = self._check_exit_conditions(position, current_price)
                
                if should_close:
                    trade_result = self.position_manager.close_position(
                        position, current_price, reason
                    )
                    
                    self.trade_logger.log_trade(trade_result)
                    
                    active_count = len(self.position_manager.get_active_positions())
                    today_pnl = self._calculate_today_pnl()
                    
                    print(f"\n{'✅' if trade_result['actual_profit_usd'] > 0 else '❌'} Position closed: {position['symbol']}")
                    print(f"   Profit: ${trade_result['actual_profit_usd']:+.2f} ({trade_result['actual_profit_pct']:+.2f}%)")
                    print(f"   Reason: {reason}")
                    
                    # Send notification with reply-to original message
                    current_balance = self.risk_manager.get_current_balance() if TradingConfig.MODE == "PAPER" else None
                    self.telegram.notify_position_closed(
                        symbol=position['symbol'],
                        side=position['side'],
                        entry_price=position['entry_price'],
                        exit_price=current_price,
                        profit_usd=trade_result['actual_profit_usd'],
                        profit_pct=trade_result['actual_profit_pct'],
                        duration_minutes=trade_result['duration_minutes'],
                        reason=reason,
                        open_positions=active_count,
                        today_pnl=today_pnl,
                        current_balance=current_balance,
                        reply_to_message_id=position.get('telegram_msg_id')
                    )
            
            except Exception as e:
                print(f"⚠️  Error monitoring {position['symbol']}: {e}")
    
    def _get_current_price(self, symbol: str) -> float:
        if TradingConfig.MODE == "PAPER":
            from .paper_trading import get_simulated_price
            return get_simulated_price(symbol)
        else:
            return self.client.get_current_price(symbol)
    
    def _check_exit_conditions(self, position: Dict, current_price: float) -> tuple[bool, str]:
        side = position['side']
        tp_price = position['tp_price']
        sl_price = position['sl_price']
        
        if side == "BUY":
            if current_price >= tp_price:
                return True, "Take-Profit"
            elif current_price <= sl_price:
                return True, "Stop-Loss"
        else:
            if current_price <= tp_price:
                return True, "Take-Profit"
            elif current_price >= sl_price:
                return True, "Stop-Loss"
        
        timestamp_open = datetime.fromisoformat(position['timestamp_open'])
        ttl_minutes = position.get('ttl_minutes', 30)
        current_time = datetime.now()
        
        time_elapsed = (current_time - timestamp_open).total_seconds() / 60
        
        if time_elapsed >= ttl_minutes:
            return True, "TTL Expired"
        
        return False, ""
    
    def _calculate_today_pnl(self) -> float:
        if not os.path.exists(TradingConfig.TRADES_LOG):
            return 0.0
        
        import csv
        from datetime import date
        
        today_profit = 0.0
        
        with open(TradingConfig.TRADES_LOG, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                close_time = datetime.fromisoformat(row['timestamp_close'])
                if close_time.date() == date.today():
                    today_profit += float(row['actual_profit_usd'])
        
        return today_profit
    
    def _send_hourly_report(self):
      """Send hourly effectiveness report to trading channel (if data exists)."""
        now = datetime.now()

        # Only send once per hour and only at minute 0 to avoid spamming
        if now.minute != 0 or self.last_hourly_report_hour == now.hour:
            return

        report = format_bingx_effectiveness_report()

        if not report:
            print("[REPORT] No trades logged yet, skipping hourly effectiveness report")
            self.last_hourly_report_hour = now.hour
            return

        sent_id = self.telegram.send_effectiveness_report(report)
        if sent_id:
            print(f"[REPORT] Hourly effectiveness report sent (msg_id={sent_id})")
        else:
            print("[REPORT WARN] Failed to send hourly effectiveness report")

        self.last_hourly_report_hour = now.hour
    
    def stop(self):
        print("\n🛑 Stopping trading service...")
        self.running = False
        
        positions = self.position_manager.get_active_positions()
        if positions:
            print(f"⚠️  Warning: {len(positions)} positions still open")
        
        self.telegram.send_message(
            f"🔴 <b>TRADING SERVICE STOPPED</b>\n\n"
            f"<b>Open positions:</b> {len(positions)}\n"
            f"<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</i>"
        )
        
        print("✅ Service stopped")

def main():
    import os
    service = TradingService()
    service.start()

if __name__ == "__main__":
    main()
