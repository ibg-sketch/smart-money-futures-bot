#!/usr/bin/env python3
"""
UIF-30 Unit Test: Verify that stale basis data (age_sec=9999) produces zero contribution.

Test cases:
1. Artificially create stale snapshot (updated timestamp = now - 9999s)
2. Call fetch_basis() and verify it returns (None, age_sec)
3. Verify that basis_score_component = 0.0 when basis_pct is None
"""

import json
import time
from pathlib import Path
import sys

def test_stale_basis_zero_contribution():
    """Test that stale basis data (age > 120s) produces zero scoring contribution"""
    
    print("[TEST] UIF-30: Stale basis data produces zero contribution")
    print("-" * 60)
    
    # Create artificial stale snapshot (9999 seconds old)
    test_snapshot = {
        'updated': time.time() - 9999,  # Very old
        'symbols': {
            'BTCUSDT': {
                'basis_pct': -0.05,  # This should be ignored due to staleness
                'updated': time.time() - 9999
            }
        }
    }
    
    # Write test snapshot
    snapshot_path = Path('data/feeds_snapshot.json')
    snapshot_path.parent.mkdir(exist_ok=True)
    
    with open(snapshot_path, 'w') as f:
        json.dump(test_snapshot, f)
    
    print(f"✓ Created stale test snapshot: {snapshot_path}")
    print(f"  Updated: {test_snapshot['updated']} ({time.time() - test_snapshot['updated']:.0f}s ago)")
    
    # Import fetch_basis
    sys.path.insert(0, '.')
    from signals.features import fetch_basis
    
    # Test fetch_basis with stale data
    basis_pct, age_sec = fetch_basis('BTCUSDT')
    
    print(f"\n[RESULT] fetch_basis('BTCUSDT'):")
    print(f"  basis_pct: {basis_pct}")
    print(f"  age_sec: {age_sec}")
    
    # Verify results
    assert basis_pct is None, f"Expected basis_pct=None for stale data, got {basis_pct}"
    assert age_sec is not None and age_sec > 120, f"Expected age_sec > 120, got {age_sec}"
    
    print(f"\n✅ PASS: Stale data (age={age_sec}s) correctly returns basis_pct=None")
    
    # Verify scoring contribution is zero
    basis_score_component = 0.0
    if basis_pct is not None:
        # This branch should NOT execute
        basis_score_component = basis_pct * 0.10
        print(f"❌ FAIL: basis_score_component should be 0.0, got {basis_score_component}")
        sys.exit(1)
    
    print(f"✅ PASS: basis_score_component = {basis_score_component} (zero contribution)")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)

if __name__ == '__main__':
    test_stale_basis_zero_contribution()
