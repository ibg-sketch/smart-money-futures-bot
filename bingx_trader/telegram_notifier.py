"""
Telegram Notifications for Trading Channel
"""
import os
import requests
from typing import Optional
from .config import TradingConfig
from datetime import datetime

class TelegramNotifier:
    def __init__(self):
        # Provide fallbacks so we never silently drop trading notifications
        # if dedicated trading credentials are missing.
        self.bot_token = TradingConfig.TELEGRAM_BOT_TOKEN or os.getenv('TELEGRAM_BOT_TOKEN')
        self.trading_channel = (
            TradingConfig.TRADING_CHANNEL_ID
            or os.getenv('TRADING_TELEGRAM_CHAT_ID')
            or os.getenv('TELEGRAM_CHAT_ID')
        )

        if not self.bot_token:
            print("[TELEGRAM WARN] Trading bot token is missing; messages will not be sent")
        if not self.trading_channel:
            print("[TELEGRAM WARN] Trading channel ID is missing; messages will not be sent")

        self.base_url = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else None
    
    def send_message(self, message: str, parse_mode: str = "HTML", reply_to_message_id: Optional[int] = None) -> Optional[int]:
        """
        Send message to Telegram channel.
        
        Returns:
            message_id if successful, None otherwise
        """
        if not self.bot_token or not self.trading_channel:
            return None

        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.trading_channel,
                'text': message,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True
            }
            
            # Add reply_to if specified
            if reply_to_message_id:
                payload['reply_to_message_id'] = reply_to_message_id
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('ok'):
                    msg_id = result.get('result', {}).get('message_id')
                    print(f"✅ Telegram message sent successfully (msg_id: {msg_id})")
                    return msg_id
                else:
                    error_code = result.get('error_code', 'unknown')
                    description = result.get('description', 'unknown error')
                    print(f"❌ Telegram API error: [{error_code}] {description}")
                    print(f"   Chat ID: {self.trading_channel}")
                    return None
            else:
                print(f"❌ Telegram HTTP error: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Telegram exception: {e}")
            return None
    
    def notify_position_opened(self, symbol: str, side: str, entry_price: float,
                              tp_price: float, sl_price: float, size: float,
                              confidence: float, strategy: str) -> Optional[int]:
        direction_emoji = "🟢" if side == "BUY" else "🔴"
        direction_text = "LONG" if side == "BUY" else "SHORT"
        
        # Calculate potential profit/loss with leverage
        leverage = TradingConfig.LEVERAGE
        
        # Calculate TP and SL percentages
        if side == "BUY":
            tp_pct = ((tp_price / entry_price) - 1) * 100
            sl_pct = ((sl_price / entry_price) - 1) * 100
        else:
            tp_pct = ((entry_price / tp_price) - 1) * 100
            sl_pct = ((entry_price / sl_price) - 1) * 100
        
        # Calculate potential USD profit/loss
        potential_profit = size * leverage * (tp_pct / 100)
        potential_loss = size * leverage * abs(sl_pct / 100)
        
        # Calculate fees (BingX fee structure)
        # TP: Entry (0.05% taker) + TP exit (0.02% maker) = 0.07%
        # SL: Entry (0.05% taker) + SL exit (0.05% taker) = 0.10%
        position_value = size * leverage
        tp_fee = position_value * (TradingConfig.TAKER_FEE + TradingConfig.MAKER_FEE)
        sl_fee = position_value * (TradingConfig.TAKER_FEE + TradingConfig.TAKER_FEE)
        
        potential_profit -= tp_fee
        potential_loss += sl_fee
        
        # Mode indicator
        mode_emoji = "📝" if TradingConfig.MODE == "PAPER" else "💰"
        mode_text = f"{mode_emoji} <b>{TradingConfig.MODE} MODE</b>"
        
        message = f"""
{direction_emoji} <b>ПОЗИЦИЯ ОТКРЫТА</b> {direction_text}

💱 <b>{symbol}</b>
💰 Entry: <code>${entry_price:.4f}</code>
💵 Size: ${size:.0f} (x{leverage})

🎯 TP: <code>${tp_price:.4f}</code> → 🟢 ${potential_profit:+.2f} ({tp_pct:+.2f}%)
🛑 SL: <code>${sl_price:.4f}</code> → 🔴 ${-potential_loss:.2f} ({sl_pct:.2f}%)

📊 Confidence: {confidence*100:.0f}%
⚙️ Strategy: {strategy}
{mode_text}

<i>🕐 {datetime.now().strftime('%H:%M:%S UTC')}</i>
""".strip()
        
        return self.send_message(message)
    
    def notify_position_closed(self, symbol: str, side: str, entry_price: float,
                              exit_price: float, profit_usd: float, profit_pct: float,
                              duration_minutes: int, reason: str, 
                              open_positions: int, today_pnl: float,
                              current_balance: Optional[float] = None,
                              reply_to_message_id: Optional[int] = None) -> Optional[int]:
        direction_text = "LONG" if side == "BUY" else "SHORT"
        
        # Result emoji based on profit
        if profit_usd > 0:
            result_emoji = "✅"
            profit_color = "🟢"
        elif profit_usd < 0:
            result_emoji = "❌"
            profit_color = "🔴"
        else:
            result_emoji = "⚪️"
            profit_color = "⚪️"
        
        # Reason emoji mapping
        reason_emoji = ""
        if "Take-Profit" in reason:
            reason_emoji = "🎯"
        elif "Stop-Loss" in reason:
            reason_emoji = "🛑"
        elif "Cancelled" in reason or "Cancel" in reason:
            reason_emoji = "🔔"
        elif "TTL" in reason or "Expired" in reason:
            reason_emoji = "⏰"
        
        # Mode indicator
        mode_emoji = "📝" if TradingConfig.MODE == "PAPER" else "💰"
        mode_text = f"{mode_emoji} <b>{TradingConfig.MODE} MODE</b>"
        
        # Today PnL color
        today_color = "🟢" if today_pnl > 0 else "🔴" if today_pnl < 0 else "⚪️"
        
        # Balance info (for paper trading all-in mode)
        balance_line = ""
        if current_balance is not None and TradingConfig.MODE == "PAPER":
            from .config import PaperTradingConfig
            if PaperTradingConfig.ALL_IN_MODE:
                balance_color = "🟢" if current_balance >= PaperTradingConfig.STARTING_BALANCE else "🔴"
                balance_line = f"\n💰 Balance: {balance_color} ${current_balance:.2f}"
        
        message = f"""
{result_emoji} <b>ПОЗИЦИЯ ЗАКРЫТА</b> {direction_text}

💱 <b>{symbol}</b>
📥 Entry: <code>${entry_price:.4f}</code>
📤 Exit: <code>${exit_price:.4f}</code>

💵 PnL: {profit_color} <b>${profit_usd:+.2f}</b> ({profit_pct:+.2f}%)
⏱ Duration: {duration_minutes} min
{reason_emoji} Reason: <i>{reason}</i>

📊 Open: {open_positions} | Today: {today_color} ${today_pnl:+.2f}{balance_line}
{mode_text}

<i>🕐 {datetime.now().strftime('%H:%M:%S UTC')}</i>
""".strip()
        
        return self.send_message(message, reply_to_message_id=reply_to_message_id)
    
    def notify_signal_analysis(self, symbol: str, side: str, signal_time: str,
                               entry_price: float, highest: float, lowest: float,
                               our_exit: float, our_profit: float,
                               would_hit_50: bool, profit_50: float,
                               would_hit_75: bool, profit_75: float) -> Optional[int]:
        direction = "LONG" if side == "BUY" else "SHORT"
        
        best_strategy = "target_min"
        best_profit = our_profit
        
        if would_hit_75 and profit_75 > best_profit:
            best_strategy = "fixed_75"
            best_profit = profit_75
        elif would_hit_50 and profit_50 > best_profit:
            best_strategy = "fixed_50"
            best_profit = profit_50
        
        additional_profit = best_profit - our_profit
        
        message = f"""
📊 <b>SIGNAL ANALYSIS</b>

<b>Symbol:</b> {symbol} {direction}
<b>Signal sent:</b> {signal_time}

<b>Price movement:</b>
├─ Entry: ${entry_price:.4f}
├─ Highest: ${highest:.4f} ({(highest/entry_price - 1)*100:+.2f}%)
└─ Lowest: ${lowest:.4f} ({(lowest/entry_price - 1)*100:+.2f}%)

✅ <b>Our trade (target_min):</b>
   Closed at: ${our_exit:.4f}
   Profit: ${our_profit:+.2f}

📈 <b>Alternative outcomes:</b>
├─ fixed_50 (${entry_price * 1.01:.4f}): ${profit_50:+.2f} {'✅' if would_hit_50 else '❌'}
└─ fixed_75 (${entry_price * 1.015:.4f}): ${profit_75:+.2f} {'✅' if would_hit_75 else '❌'}

💡 <b>Best strategy:</b> {best_strategy} ({additional_profit:+.2f} more)
"""
        
        return self.send_message(message)

    def send_effectiveness_report(self, report_text: str) -> Optional[int]:
        """Send aggregated effectiveness statistics to the trading channel."""
        return self.send_message(report_text, parse_mode="HTML")
    
    def notify_daily_report(self, total_trades: int, wins: int, losses: int,
                           total_profit: float, win_rate: float, roi: float) -> Optional[int]:
        # Mode indicator
        mode_emoji = "📝" if TradingConfig.MODE == "PAPER" else "💰"
        
        message = f"""
📊 <b>DAILY TRADING REPORT</b>

<b>Total trades:</b> {total_trades}
<b>Wins:</b> {wins} | <b>Losses:</b> {losses}
<b>Win Rate:</b> {win_rate:.1f}%

<b>Total Profit:</b> ${total_profit:+.2f}
<b>ROI:</b> {roi:+.1f}%

{mode_emoji} <b>Mode:</b> {TradingConfig.MODE}
<b>Exchange:</b> {TradingConfig.EXCHANGE}

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</i>
"""
        
        return self.send_message(message)
    
    def notify_error(self, error_message: str) -> Optional[int]:
        message = f"""
⚠️ <b>TRADING ERROR</b>

{error_message}

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</i>
"""
        
        return self.send_message(message)
