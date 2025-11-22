"""
Shadow Mode Logger
Logs dual-formula predictions alongside current system decisions for A/B comparison.
Outputs structured data to CSV for validation before full production release.
"""

import csv
import os
import time
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional

class ShadowLogger:
    """Thread-safe logger for shadow mode predictions and comparisons."""
    
    def __init__(self, log_file: str = "shadow_predictions.csv"):
        """
        Initialize shadow logger.
        
        Args:
            log_file: Path to CSV log file
        """
        self.log_file = log_file
        self.lock = threading.Lock()
        
        # CSV headers matching required schema
        self.headers = [
            # Timestamp and identification
            'ts', 'symbol', 'verdict',
            
            # Dual-formula outputs
            'logit', 'prob', 'should_send',
            
            # Market indicators
            'rsi', 'ema_short', 'ema_long', 'ema_diff_pct',
            'price', 'vwap', 'vwap_dist',
            'volume', 'volume_median', 'volume_ratio', 'atr',
            
            # CVD and OI data
            'cvd', 'cvd_5m', 'oi_change',
            
            # Old system comparison
            'old_system_decision', 'old_system_score',
            
            # Performance metrics
            'latency_ms', 'spread', 'fee_bps', 'slippage_bps',
            
            # Filter results
            'passed_filters', 'filter_reason'
        ]
        
        # Initialize CSV file with headers if it doesn't exist
        self._initialize_log_file()
    
    def _initialize_log_file(self):
        """Create log file with headers if it doesn't exist."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()
    
    def log_prediction(self, data: Dict[str, Any]):
        """
        Log a shadow mode prediction with thread safety.
        
        Args:
            data: Dictionary containing prediction data (must include all required fields)
        """
        with self.lock:
            # Ensure timestamp is present
            if 'ts' not in data:
                data['ts'] = datetime.now(timezone.utc).isoformat()
            
            # Fill in missing fields with None
            row = {header: data.get(header, None) for header in self.headers}
            
            # Append to CSV
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writerow(row)
    
    def log_signal_evaluation(
        self,
        symbol: str,
        verdict: str,
        
        # Dual-formula results
        logit: float,
        prob: float,
        should_send: bool,
        
        # Indicators
        rsi: float,
        ema_short: float,
        ema_long: float,
        price: float,
        vwap: float,
        volume: float,
        volume_median: float,
        atr: float,
        
        # Additional data
        cvd: float = None,
        cvd_5m: float = None,
        oi_change: float = None,
        
        # Old system
        old_decision: str = None,
        old_score: float = None,
        
        # Filters
        passed_filters: bool = False,
        filter_reason: str = None,
        
        # Performance
        latency_ms: float = None,
        spread: float = None,
        fee_bps: float = 10.0,  # Default 0.10% (10 bps)
        slippage_bps: float = 5.0  # Default 0.05% (5 bps)
    ):
        """
        Convenience method to log signal evaluation with calculated fields.
        
        Args:
            symbol: Trading pair symbol
            verdict: Signal direction ('BUY' or 'SELL')
            logit: Raw logit score from dual formula
            prob: Probability score (0-1) after sigmoid
            should_send: Whether signal passed threshold
            rsi: RSI indicator value
            ema_short: Short-term EMA
            ema_long: Long-term EMA
            price: Current price
            vwap: Volume-weighted average price
            volume: Current volume
            volume_median: Median volume
            atr: Average True Range
            cvd: Cumulative Volume Delta
            cvd_5m: 5-minute CVD aggregate
            oi_change: Open Interest change
            old_decision: Decision from current system
            old_score: Score from current system
            passed_filters: Whether signal passed all filters
            filter_reason: Reason for filter rejection (if any)
            latency_ms: Processing latency
            spread: Bid-ask spread
            fee_bps: Trading fee in basis points
            slippage_bps: Expected slippage in basis points
        """
        # Calculate derived fields
        ema_diff_pct = ((ema_short - ema_long) / price) * 100 if price > 0 else 0.0
        vwap_dist = abs((price - vwap) / price) * 100 if price > 0 and vwap > 0 else 0.0
        volume_ratio = volume / volume_median if volume_median > 0 else 1.0
        
        data = {
            'ts': datetime.now(timezone.utc).isoformat(),
            'symbol': symbol,
            'verdict': verdict,
            'logit': round(logit, 4),
            'prob': round(prob, 4),
            'should_send': should_send,
            'rsi': round(rsi, 2),
            'ema_short': round(ema_short, 2),
            'ema_long': round(ema_long, 2),
            'ema_diff_pct': round(ema_diff_pct, 4),
            'price': round(price, 2),
            'vwap': round(vwap, 2) if vwap else None,
            'vwap_dist': round(vwap_dist, 4),
            'volume': round(volume, 0),
            'volume_median': round(volume_median, 0),
            'volume_ratio': round(volume_ratio, 2),
            'atr': round(atr, 4),
            'cvd': round(cvd, 0) if cvd is not None else None,
            'cvd_5m': round(cvd_5m, 0) if cvd_5m is not None else None,
            'oi_change': round(oi_change, 2) if oi_change is not None else None,
            'old_system_decision': old_decision,
            'old_system_score': round(old_score, 2) if old_score is not None else None,
            'latency_ms': round(latency_ms, 2) if latency_ms is not None else None,
            'spread': round(spread, 4) if spread is not None else None,
            'fee_bps': fee_bps,
            'slippage_bps': slippage_bps,
            'passed_filters': passed_filters,
            'filter_reason': filter_reason
        }
        
        self.log_prediction(data)
    
    def get_log_path(self) -> str:
        """Return the path to the log file."""
        return self.log_file


# Global logger instance for easy access
_shadow_logger = None

def get_shadow_logger(log_file: str = "shadow_predictions.csv") -> ShadowLogger:
    """
    Get or create the global shadow logger instance.
    
    Args:
        log_file: Path to CSV log file
    
    Returns:
        ShadowLogger instance
    """
    global _shadow_logger
    if _shadow_logger is None:
        _shadow_logger = ShadowLogger(log_file)
    return _shadow_logger


if __name__ == "__main__":
    # Test the logger
    logger = ShadowLogger("test_shadow_predictions.csv")
    
    print("Testing shadow logger...")
    
    # Log a sample prediction
    logger.log_signal_evaluation(
        symbol="BTCUSDT",
        verdict="BUY",
        logit=-0.5234,
        prob=0.3721,
        should_send=True,
        rsi=55.2,
        ema_short=50100.5,
        ema_long=50000.0,
        price=50000.0,
        vwap=49950.0,
        volume=35_000_000,
        volume_median=40_000_000,
        atr=175.5,
        cvd=2_500_000,
        cvd_5m=1_200_000,
        oi_change=5.2,
        old_decision="BUY",
        old_score=78.5,
        passed_filters=True,
        latency_ms=125.3
    )
    
    logger.log_signal_evaluation(
        symbol="ETHUSDT",
        verdict="SELL",
        logit=-1.2345,
        prob=0.2256,
        should_send=False,
        rsi=68.5,
        ema_short=3000.0,
        ema_long=3010.0,
        price=3005.0,
        vwap=3008.0,
        volume=50_000_000,
        volume_median=40_000_000,
        atr=12.5,
        old_decision="SKIP",
        old_score=45.2,
        passed_filters=False,
        filter_reason="Volume ratio too high (1.25 > 1.0)"
    )
    
    print(f"✅ Test complete. Check {logger.get_log_path()}")
