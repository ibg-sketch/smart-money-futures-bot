"""
Output sinks: CSV logging and Telegram messaging
"""

import csv
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from telegram_utils import send_telegram_message

logger = logging.getLogger(__name__)

CSV_FILE = 'data/ai_reports.csv'
CSV_HEADERS = [
    'ts', 'symbol', 'signal_id', 'verdict', 'bot_confidence', 'ai_confidence',
    'ai_ttl_minutes', 'ai_target_pct', 'features_used', 'tokens_in', 'tokens_out', 'cost_usd',
    'summary_100w', 'reasoning', 'error', 'cached'
]


class OutputSink:
    """Handle CSV logging and Telegram output"""
    
    def __init__(self):
        self._ensure_csv_exists()
    
    def _ensure_csv_exists(self):
        """Create CSV file with headers if it doesn't exist"""
        if not os.path.exists(CSV_FILE):
            with open(CSV_FILE, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                writer.writeheader()
            logger.info(f"Created {CSV_FILE}")
    
    def log_signal_analysis(
        self,
        symbol: str,
        signal_id: str,
        verdict: str,
        bot_confidence: float,
        ai_confidence: Optional[int],
        ai_ttl_minutes: Optional[int],
        ai_target_pct: Optional[float],
        features_used: str,
        ai_result: Dict[str, Any],
        summary: str,
        reasoning: str = ""
    ):
        """
        Log per-signal AI analysis to CSV
        
        Args:
            symbol: Trading symbol
            signal_id: Unique signal identifier
            verdict: BUY/SELL
            bot_confidence: Bot's calculated confidence
            ai_confidence: AI's independent confidence (0-100) or None
            ai_ttl_minutes: AI's recommended TTL in minutes or None
            ai_target_pct: AI's recommended target profit % or None
            features_used: Comma-separated feature names
            ai_result: Result from ai_client.get_completion()
            summary: Truncated summary for CSV
            reasoning: AI's reasoning text
        """
        try:
            row = {
                'ts': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': symbol,
                'signal_id': signal_id,
                'verdict': verdict,
                'bot_confidence': f"{bot_confidence:.2f}",
                'ai_confidence': str(ai_confidence) if ai_confidence is not None else '',
                'ai_ttl_minutes': str(ai_ttl_minutes) if ai_ttl_minutes is not None else '',
                'ai_target_pct': f"{ai_target_pct:.2f}" if ai_target_pct is not None else '',
                'features_used': features_used,
                'tokens_in': ai_result.get('tokens_in', 0),
                'tokens_out': ai_result.get('tokens_out', 0),
                'cost_usd': f"{ai_result.get('cost_usd', 0.0):.6f}",
                'summary_100w': summary,
                'reasoning': reasoning[:200] if reasoning else '',
                'error': ai_result.get('error', ''),
                'cached': 'yes' if ai_result.get('cached', False) else 'no'
            }
            
            with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                writer.writerow(row)
            
            logger.info(f"Logged AI analysis for {symbol} {verdict} to CSV (bot:{bot_confidence:.0f}% ai:{ai_confidence or 'N/A'}% ttl:{ai_ttl_minutes or 'N/A'}min target:{ai_target_pct or 'N/A'}%)")
        
        except Exception as e:
            logger.error(f"Failed to log to CSV: {e}")
    
    def send_signal_context(
        self,
        formatted_text: str,
        reply_to_message_id: Optional[int] = None
    ) -> bool:
        """
        Send AI market context as reply to original signal
        
        Args:
            formatted_text: HTML formatted text
            reply_to_message_id: Message ID to reply to
        
        Returns:
            True if sent successfully
        """
        try:
            result = send_telegram_message(
                formatted_text,
                parse_mode='HTML',
                reply_to_message_id=reply_to_message_id
            )
            
            if result:
                logger.info("Sent AI context to Telegram")
                return True
            else:
                logger.warning("Failed to send AI context to Telegram")
                return False
        
        except Exception as e:
            logger.error(f"Error sending to Telegram: {e}")
            return False
    
    def send_daily_summary(self, formatted_text: str) -> bool:
        """
        Send daily AI summary to Telegram
        
        Args:
            formatted_text: HTML formatted summary
        
        Returns:
            True if sent successfully
        """
        try:
            result = send_telegram_message(formatted_text, parse_mode='HTML')
            
            if result:
                logger.info("Sent daily AI summary to Telegram")
                return True
            else:
                logger.warning("Failed to send daily summary to Telegram")
                return False
        
        except Exception as e:
            logger.error(f"Error sending daily summary: {e}")
            return False
    
    def save_daily_summary_file(self, content: str, date_str: str) -> bool:
        """
        Save daily summary to markdown file
        
        Args:
            content: Summary content
            date_str: Date string (YYYYMMDD)
        
        Returns:
            True if saved successfully
        """
        try:
            filename = f"reports/daily_ai_summary_{date_str}.md"
            
            os.makedirs('reports', exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# AI Daily Summary - {date_str}\n\n")
                f.write(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n")
                f.write("---\n\n")
                f.write(content)
            
            logger.info(f"Saved daily summary to {filename}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save summary file: {e}")
            return False
