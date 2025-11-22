"""
Shadow Mode Metrics Reporter
Calculates daily and weekly performance metrics from shadow prediction logs.
Tracks precision, coverage, net expectancy, and generates alerts.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional

import pandas as pd

class MetricsReporter:
    """Analyze shadow mode predictions and calculate performance metrics."""
    
    def __init__(self, log_file: str = "shadow_predictions.csv"):
        """
        Initialize metrics reporter.
        
        Args:
            log_file: Path to shadow predictions CSV log
        """
        self.log_file = log_file
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load shadow predictions from CSV."""
        if os.path.exists(self.log_file):
            self.df = pd.read_csv(self.log_file)
            if len(self.df) > 0:
                self.df['ts'] = pd.to_datetime(self.df['ts'])
        else:
            self.df = pd.DataFrame()
    
    def refresh_data(self):
        """Reload data from CSV (for real-time updates)."""
        self.load_data()
    
    def get_daily_metrics(self, date: Optional[datetime] = None) -> Dict:
        """
        Calculate metrics for a specific day.
        
        Args:
            date: Date to analyze (default: today)
        
        Returns:
            Dictionary with daily metrics
        """
        if self.df is None or len(self.df) == 0:
            return self._empty_metrics()
        
        if date is None:
            date = datetime.now(timezone.utc)
        
        # Filter for specific day
        day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        
        day_df = self.df[(self.df['ts'] >= day_start) & (self.df['ts'] < day_end)]
        
        return self._calculate_metrics(day_df, period="daily")
    
    def get_weekly_metrics(self, week_start: Optional[datetime] = None) -> Dict:
        """
        Calculate metrics for a week.
        
        Args:
            week_start: Start of week (default: 7 days ago)
        
        Returns:
            Dictionary with weekly metrics
        """
        if self.df is None or len(self.df) == 0:
            return self._empty_metrics()
        
        if week_start is None:
            week_start = datetime.now(timezone.utc) - timedelta(days=7)
        
        week_end = week_start + timedelta(days=7)
        
        week_df = self.df[(self.df['ts'] >= week_start) & (self.df['ts'] < week_end)]
        
        return self._calculate_metrics(week_df, period="weekly")
    
    def _calculate_metrics(self, df: pd.DataFrame, period: str = "daily") -> Dict:
        """
        Calculate all metrics for a given dataframe.
        
        Args:
            df: Filtered dataframe
            period: Time period label
        
        Returns:
            Dictionary with calculated metrics
        """
        if len(df) == 0:
            return self._empty_metrics()
        
        # Total evaluations
        total_evaluations = len(df)
        
        # Signals that passed formula threshold
        passed_threshold = df[df['should_send'] == True]
        threshold_count = len(passed_threshold)
        
        # Signals that passed all filters
        passed_filters = df[df['passed_filters'] == True]
        filter_count = len(passed_filters)
        
        # Signals that would be sent (threshold AND filters)
        would_send = df[(df['should_send'] == True) & (df['passed_filters'] == True)]
        send_count = len(would_send)
        
        # Old system comparison
        old_system_signals = df[df['old_system_decision'].notna()]
        old_sent = old_system_signals[old_system_signals['old_system_decision'] != 'SKIP']
        
        # Calculate precision (Note: We can't measure actual win rate in shadow mode yet)
        # This is a placeholder - actual precision requires tracking signal outcomes
        avg_probability = would_send['prob'].mean() if len(would_send) > 0 else 0.0
        
        # Coverage (signals per day/week)
        if period == "daily":
            coverage = send_count  # signals per day
        else:
            coverage = send_count / 7.0  # average signals per day in the week
        
        # Direction breakdown
        long_signals = would_send[would_send['verdict'] == 'BUY']
        short_signals = would_send[would_send['verdict'] == 'SELL']
        
        # Latency metrics
        avg_latency = df['latency_ms'].mean() if 'latency_ms' in df.columns else None
        max_latency = df['latency_ms'].max() if 'latency_ms' in df.columns else None
        
        # Net expectancy (theoretical - based on probability, TP, SL, fees)
        # E = (prob * TP) - ((1-prob) * SL) - (fees + slippage)
        # Using default TP=0.4%, SL=0.3%, fees=0.1%, slippage=0.05%
        tp_pct = 0.40
        sl_pct = 0.30
        fees_pct = 0.15  # 0.10% + 0.05% slippage
        
        if len(would_send) > 0:
            expectancies = []
            for _, row in would_send.iterrows():
                prob = row['prob']
                exp = (prob * tp_pct) - ((1 - prob) * sl_pct) - fees_pct
                expectancies.append(exp)
            avg_expectancy = sum(expectancies) / len(expectancies)
        else:
            avg_expectancy = 0.0
        
        metrics = {
            'period': period,
            'total_evaluations': total_evaluations,
            'passed_threshold': threshold_count,
            'passed_filters': filter_count,
            'would_send_signals': send_count,
            'long_signals': len(long_signals),
            'short_signals': len(short_signals),
            'avg_probability': round(avg_probability, 4),
            'coverage_per_day': round(coverage, 2),
            'old_system_signals': len(old_sent),
            'signal_reduction_pct': round((1 - send_count / len(old_sent)) * 100, 1) if len(old_sent) > 0 else 0.0,
            'avg_latency_ms': round(avg_latency, 2) if avg_latency is not None else None,
            'max_latency_ms': round(max_latency, 2) if max_latency is not None else None,
            'avg_expectancy_pct': round(avg_expectancy, 3)
        }
        
        return metrics
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure."""
        return {
            'period': 'N/A',
            'total_evaluations': 0,
            'passed_threshold': 0,
            'passed_filters': 0,
            'would_send_signals': 0,
            'long_signals': 0,
            'short_signals': 0,
            'avg_probability': 0.0,
            'coverage_per_day': 0.0,
            'old_system_signals': 0,
            'signal_reduction_pct': 0.0,
            'avg_latency_ms': None,
            'max_latency_ms': None,
            'avg_expectancy_pct': 0.0
        }
    
    def check_alerts(self, metrics: Dict) -> List[str]:
        """
        Check for alert conditions.
        
        Alert conditions:
        - Coverage drop > 30%
        - Avg probability < 0.30
        - Max latency > 500ms
        
        Args:
            metrics: Metrics dictionary from get_daily_metrics or get_weekly_metrics
        
        Returns:
            List of alert messages
        """
        alerts = []
        
        # Check coverage drop
        if metrics['signal_reduction_pct'] > 30.0:
            alerts.append(
                f"⚠️ COVERAGE ALERT: Signal reduction {metrics['signal_reduction_pct']:.1f}% > 30%"
            )
        
        # Check average probability (precision proxy)
        if metrics['avg_probability'] < 0.30 and metrics['would_send_signals'] > 0:
            alerts.append(
                f"⚠️ PRECISION ALERT: Avg probability {metrics['avg_probability']:.2%} < 30%"
            )
        
        # Check latency
        if metrics['max_latency_ms'] is not None and metrics['max_latency_ms'] > 500:
            alerts.append(
                f"⚠️ LATENCY ALERT: Max latency {metrics['max_latency_ms']:.0f}ms > 500ms"
            )
        
        # Check expectancy
        if metrics['avg_expectancy_pct'] < 0:
            alerts.append(
                f"⚠️ EXPECTANCY ALERT: Negative net expectancy {metrics['avg_expectancy_pct']:.3f}%"
            )
        
        return alerts
    
    def generate_report(self, period: str = "daily") -> str:
        """
        Generate formatted text report.
        
        Args:
            period: 'daily' or 'weekly'
        
        Returns:
            Formatted report string
        """
        if period == "daily":
            metrics = self.get_daily_metrics()
        else:
            metrics = self.get_weekly_metrics()
        
        alerts = self.check_alerts(metrics)
        
        report = []
        report.append("=" * 80)
        report.append(f"SHADOW MODE {period.upper()} REPORT")
        report.append("=" * 80)
        report.append("")
        
        report.append(f"📊 Signal Metrics:")
        report.append(f"  Total evaluations:    {metrics['total_evaluations']:,}")
        report.append(f"  Passed threshold:     {metrics['passed_threshold']:,}")
        report.append(f"  Passed filters:       {metrics['passed_filters']:,}")
        report.append(f"  Would send signals:   {metrics['would_send_signals']:,}")
        report.append(f"    - LONG signals:     {metrics['long_signals']:,}")
        report.append(f"    - SHORT signals:    {metrics['short_signals']:,}")
        report.append("")
        
        report.append(f"📈 Quality Metrics:")
        report.append(f"  Avg probability:      {metrics['avg_probability']:.2%}")
        report.append(f"  Coverage (sig/day):   {metrics['coverage_per_day']:.1f}")
        report.append(f"  Net expectancy:       {metrics['avg_expectancy_pct']:.3f}%")
        report.append("")
        
        report.append(f"🔄 Comparison with Old System:")
        report.append(f"  Old system signals:   {metrics['old_system_signals']:,}")
        report.append(f"  Signal reduction:     {metrics['signal_reduction_pct']:.1f}%")
        report.append("")
        
        if metrics['avg_latency_ms'] is not None:
            report.append(f"⚡ Performance:")
            report.append(f"  Avg latency:          {metrics['avg_latency_ms']:.1f}ms")
            report.append(f"  Max latency:          {metrics['max_latency_ms']:.1f}ms")
            report.append("")
        
        if alerts:
            report.append("🚨 ALERTS:")
            for alert in alerts:
                report.append(f"  {alert}")
            report.append("")
        else:
            report.append("✅ No alerts - all metrics within acceptable ranges")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_report(self, filename: str = "shadow_report.txt", period: str = "daily"):
        """
        Save report to file.
        
        Args:
            filename: Output filename
            period: 'daily' or 'weekly'
        """
        report = self.generate_report(period)
        with open(filename, 'w') as f:
            f.write(report)
            f.write(f"\nGenerated: {datetime.now(timezone.utc).isoformat()}\n")


if __name__ == "__main__":
    # Test with existing test data
    reporter = MetricsReporter("test_shadow_predictions.csv")
    
    print(reporter.generate_report("daily"))
    
    # Save report
    reporter.save_report("test_shadow_report.txt", "daily")
    print("\n✅ Report saved to test_shadow_report.txt")
