"""
CSV Writer with Log Rotation for UIF Features

Append-only CSV logging to data/uif_log.csv with automatic rotation.
"""

import csv
import os
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional


CSV_PATH = "data/uif_log.csv"
CSV_HEADERS = [
    'timestamp',
    'symbol',
    'adx14',
    'psar_state',
    'momentum5',
    'vol_accel',
    'latency_ms',
    'source_errors'
]


class UIFWriter:
    """Append-only CSV writer with rotation."""
    
    def __init__(self, rotate_mb: int = 200, keep_files: int = 14):
        """
        Initialize UIF CSV writer.
        
        Args:
            rotate_mb: Max file size in MB before rotation
            keep_files: Number of rotated files to keep
        """
        self.csv_path = CSV_PATH
        self.rotate_mb = rotate_mb
        self.keep_files = keep_files
        self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV file with headers if doesn't exist."""
        try:
            os.makedirs("data", exist_ok=True)
            
            if not os.path.exists(self.csv_path):
                with open(self.csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(CSV_HEADERS)
                print(f"[INFO] Created UIF log: {self.csv_path}")
        
        except Exception as e:
            print(f"[ERROR] Failed to init UIF CSV: {e}")
    
    def _should_rotate(self) -> bool:
        """Check if CSV should be rotated."""
        try:
            if not os.path.exists(self.csv_path):
                return False
            
            size_mb = os.path.getsize(self.csv_path) / (1024 * 1024)
            return size_mb >= self.rotate_mb
        
        except Exception:
            return False
    
    def _rotate_csv(self):
        """Rotate CSV file and cleanup old rotations."""
        try:
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            rotated_path = f"{self.csv_path}.{timestamp}"
            
            os.rename(self.csv_path, rotated_path)
            print(f"[INFO] Rotated UIF log to: {rotated_path}")
            
            # Recreate with headers
            self._init_csv()
            
            # Cleanup old files
            self._cleanup_old_rotations()
        
        except Exception as e:
            print(f"[ERROR] Failed to rotate UIF CSV: {e}")
    
    def _cleanup_old_rotations(self):
        """Keep only the most recent N rotated files."""
        try:
            data_dir = os.path.dirname(self.csv_path)
            base_name = os.path.basename(self.csv_path)
            
            rotated_files = [
                f for f in os.listdir(data_dir)
                if f.startswith(base_name + ".")
            ]
            
            rotated_files.sort(reverse=True)
            
            for old_file in rotated_files[self.keep_files:]:
                old_path = os.path.join(data_dir, old_file)
                os.remove(old_path)
                print(f"[INFO] Deleted old rotation: {old_file}")
        
        except Exception as e:
            print(f"[WARN] Failed to cleanup old rotations: {e}")
    
    def append(
        self,
        symbol: str,
        indicators: Dict[str, Optional[float]],
        latency_ms: int = 0,
        source_errors: str = ""
    ):
        """
        Append UIF feature row to CSV.
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            indicators: Dict with adx14, psar_state, momentum5, vol_accel
            latency_ms: Data fetch latency
            source_errors: Any errors encountered during calculation
        """
        try:
            # Rotate if needed
            if self._should_rotate():
                self._rotate_csv()
            
            timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            
            row = [
                timestamp,
                symbol,
                indicators.get('adx14', ''),
                indicators.get('psar_state', ''),
                indicators.get('momentum5', ''),
                indicators.get('vol_accel', ''),
                latency_ms,
                source_errors
            ]
            
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        
        except Exception as e:
            print(f"[ERROR] Failed to append UIF CSV: {e}")
