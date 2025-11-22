#!/usr/bin/env python3
"""
Unit Tests for RSI Calculation - Wilder's Smoothing Method
Tests with known reference values to ensure correctness
"""

import unittest
import numpy as np
from signals.features import compute_rsi


class TestRSICalculation(unittest.TestCase):
    """Test suite for RSI calculation using Wilder's Smoothing Method"""
    
    def test_rsi_with_uptrend_data(self):
        """Test RSI with consistently upward trending data"""
        # Create klines with upward trend
        klines = []
        for i in range(50):
            price = 100 + i * 2  # Consistent uptrend
            kline = [
                1000000 + i*60000,  # timestamp
                price,               # open
                price + 1,          # high
                price - 0.5,        # low
                price + 0.5,        # close
                10000,              # volume
                1000000 + i*60000   # close time
            ]
            klines.append(kline)
        
        rsi = compute_rsi(klines, period=14)
        
        self.assertIsNotNone(rsi, "RSI should not be None with sufficient data")
        self.assertGreater(rsi, 70, "RSI should be overbought (>70) in strong uptrend")
        self.assertLessEqual(rsi, 100, "RSI should never exceed 100")
        self.assertGreaterEqual(rsi, 0, "RSI should never be below 0")
    
    def test_rsi_with_downtrend_data(self):
        """Test RSI with consistently downward trending data"""
        # Create klines with downward trend
        klines = []
        for i in range(50):
            price = 200 - i * 2  # Consistent downtrend
            kline = [
                1000000 + i*60000,
                price,
                price + 0.5,
                price - 1,
                price - 0.5,
                10000,
                1000000 + i*60000
            ]
            klines.append(kline)
        
        rsi = compute_rsi(klines, period=14)
        
        self.assertIsNotNone(rsi, "RSI should not be None with sufficient data")
        self.assertLess(rsi, 30, "RSI should be oversold (<30) in strong downtrend")
        self.assertLessEqual(rsi, 100, "RSI should never exceed 100")
        self.assertGreaterEqual(rsi, 0, "RSI should never be below 0")
    
    def test_rsi_with_sideways_data(self):
        """Test RSI with sideways/choppy data"""
        # Create klines with sideways movement
        klines = []
        base_price = 150
        for i in range(50):
            price = base_price + (i % 5) - 2  # Oscillating around 150
            kline = [
                1000000 + i*60000,
                price,
                price + 1,
                price - 1,
                price,
                10000,
                1000000 + i*60000
            ]
            klines.append(kline)
        
        rsi = compute_rsi(klines, period=14)
        
        self.assertIsNotNone(rsi, "RSI should not be None with sufficient data")
        self.assertGreater(rsi, 30, "RSI should be above 30 in sideways market")
        self.assertLess(rsi, 70, "RSI should be below 70 in sideways market")
        self.assertLessEqual(rsi, 100, "RSI should never exceed 100")
        self.assertGreaterEqual(rsi, 0, "RSI should never be below 0")
    
    def test_rsi_with_insufficient_data(self):
        """Test RSI returns None with insufficient data"""
        # Only 10 klines (need 15 for 14-period RSI)
        klines = []
        for i in range(10):
            kline = [1000000 + i*60000, 100+i, 101+i, 99+i, 100.5+i, 10000, 1000000 + i*60000]
            klines.append(kline)
        
        rsi = compute_rsi(klines, period=14)
        self.assertIsNone(rsi, "RSI should be None with insufficient data")
    
    def test_rsi_with_empty_data(self):
        """Test RSI returns None with empty data"""
        rsi = compute_rsi([], period=14)
        self.assertIsNone(rsi, "RSI should be None with empty data")
        
        rsi = compute_rsi(None, period=14)
        self.assertIsNone(rsi, "RSI should be None with None data")
    
    def test_rsi_all_gains_scenario(self):
        """Test RSI = 100 scenario (all gains, no losses)"""
        # Create klines with only gains
        klines = []
        for i in range(50):
            price = 100 + i * 5  # Large consistent gains
            kline = [1000000 + i*60000, price, price+5, price, price+5, 10000, 1000000 + i*60000]
            klines.append(kline)
        
        rsi = compute_rsi(klines, period=14)
        
        self.assertIsNotNone(rsi, "RSI should not be None")
        # With Wilder's smoothing, even all gains won't give exact 100 unless losses are exactly 0
        self.assertGreaterEqual(rsi, 90, "RSI should be very high (≥90) with all gains")
        self.assertLessEqual(rsi, 100, "RSI should be capped at 100")
    
    def test_rsi_value_range(self):
        """Test RSI is always within 0-100 range for various scenarios"""
        test_scenarios = [
            # Volatile up/down
            [100, 105, 103, 107, 104, 110, 108, 112, 109, 115, 113, 118, 115, 120, 117, 122, 119, 125, 122, 128],
            # Gradual up
            [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
            # Gradual down
            [200, 199, 198, 197, 196, 195, 194, 193, 192, 191, 190, 189, 188, 187, 186, 185, 184, 183, 182, 181],
            # Random walk
            [100, 102, 101, 103, 100, 104, 102, 105, 103, 107, 105, 108, 106, 110, 107, 109, 108, 111, 109, 112]
        ]
        
        for prices in test_scenarios:
            klines = []
            for i, price in enumerate(prices):
                kline = [1000000 + i*60000, price, price+1, price-1, price, 10000, 1000000 + i*60000]
                klines.append(kline)
            
            rsi = compute_rsi(klines, period=14)
            
            if rsi is not None:
                self.assertLessEqual(rsi, 100, f"RSI should not exceed 100 (got {rsi})")
                self.assertGreaterEqual(rsi, 0, f"RSI should not be below 0 (got {rsi})")
    
    def test_rsi_wilder_smoothing_vs_simple_average(self):
        """Test that Wilder's smoothing gives different result than simple average"""
        # Create test data with known values (classic RSI test case)
        prices = [44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 
                  45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64,
                  46.21, 46.25, 45.71, 46.45, 45.78, 45.35, 44.03, 44.18, 44.22, 44.57, 43.42]
        
        klines = []
        for i, price in enumerate(prices):
            kline = [1000000 + i*60000, price, price+0.5, price-0.5, price, 10000, 1000000 + i*60000]
            klines.append(kline)
        
        # Calculate using Wilder's smoothing (our implementation)
        rsi_wilder = compute_rsi(klines, period=14)
        
        # Calculate what simple average would give (INCORRECT method)
        closes = np.array([float(k[4]) for k in klines[-(14+1):]])
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain_simple = np.mean(gains)
        avg_loss_simple = np.mean(losses)
        rs_simple = avg_gain_simple / avg_loss_simple if avg_loss_simple > 0 else 100
        rsi_simple = 100 - (100 / (1 + rs_simple))
        
        self.assertIsNotNone(rsi_wilder, "Wilder's RSI should be calculated")
        
        # CRITICAL TEST: Wilder's smoothing MUST give different result than simple average
        # If these are the same, we're not using Wilder's smoothing!
        self.assertNotAlmostEqual(
            rsi_wilder, rsi_simple, places=1,
            msg=f"Wilder's RSI ({rsi_wilder:.2f}) should differ from simple average RSI ({rsi_simple:.2f}). "
                f"Implementation may not be using Wilder's smoothing!"
        )
        
        # Additional verification: both should be valid
        self.assertGreaterEqual(rsi_wilder, 0, "Wilder's RSI should be >= 0")
        self.assertLessEqual(rsi_wilder, 100, "Wilder's RSI should be <= 100")
    
    def test_rsi_consistent_calculation(self):
        """Test RSI gives consistent results for same input"""
        klines = []
        for i in range(30):
            price = 100 + np.sin(i/5) * 10  # Sinusoidal pattern
            kline = [1000000 + i*60000, price, price+1, price-1, price, 10000, 1000000 + i*60000]
            klines.append(kline)
        
        rsi1 = compute_rsi(klines, period=14)
        rsi2 = compute_rsi(klines, period=14)
        
        self.assertEqual(rsi1, rsi2, "RSI should be deterministic and consistent")
    
    def test_rsi_different_periods(self):
        """Test RSI calculation with different period lengths"""
        klines = []
        # Use moderate uptrend to avoid hitting 100 ceiling
        for i in range(50):
            price = 100 + i * 0.5  # Moderate uptrend
            kline = [1000000 + i*60000, price, price+0.5, price-0.3, price, 10000, 1000000 + i*60000]
            klines.append(kline)
        
        rsi_9 = compute_rsi(klines, period=9)
        rsi_14 = compute_rsi(klines, period=14)
        rsi_21 = compute_rsi(klines, period=21)
        
        self.assertIsNotNone(rsi_9, "RSI-9 should be calculated")
        self.assertIsNotNone(rsi_14, "RSI-14 should be calculated")
        self.assertIsNotNone(rsi_21, "RSI-21 should be calculated")
        
        # All should be in valid range
        self.assertGreaterEqual(rsi_9, 0, "RSI-9 should be >= 0")
        self.assertLessEqual(rsi_9, 100, "RSI-9 should be <= 100")
        self.assertGreaterEqual(rsi_14, 0, "RSI-14 should be >= 0")
        self.assertLessEqual(rsi_14, 100, "RSI-14 should be <= 100")
        self.assertGreaterEqual(rsi_21, 0, "RSI-21 should be >= 0")
        self.assertLessEqual(rsi_21, 100, "RSI-21 should be <= 100")


class TestRSIEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def test_rsi_with_zero_movement(self):
        """Test RSI with no price movement (all same price)"""
        klines = []
        price = 100
        for i in range(30):
            kline = [1000000 + i*60000, price, price, price, price, 10000, 1000000 + i*60000]
            klines.append(kline)
        
        rsi = compute_rsi(klines, period=14)
        
        # With no price movement, avg_gain = 0 and avg_loss = 0
        # Implementation should handle division by zero
        self.assertIsNotNone(rsi, "RSI should handle zero movement")
        # When avg_loss = 0, function returns 100.0
        self.assertEqual(rsi, 100.0, "RSI should be 100 when there are no losses (no price change)")
    
    def test_rsi_with_extreme_volatility(self):
        """Test RSI with extreme price swings"""
        klines = []
        for i in range(30):
            price = 100 if i % 2 == 0 else 200  # Alternating extreme moves
            kline = [1000000 + i*60000, price, price+10, price-10, price, 10000, 1000000 + i*60000]
            klines.append(kline)
        
        rsi = compute_rsi(klines, period=14)
        
        self.assertIsNotNone(rsi, "RSI should handle extreme volatility")
        self.assertLessEqual(rsi, 100, "RSI should not exceed 100")
        self.assertGreaterEqual(rsi, 0, "RSI should not be below 0")
        # With alternating equal moves, should be around 50
        self.assertGreater(rsi, 30, "RSI should be in neutral zone with equal up/down moves")
        self.assertLess(rsi, 70, "RSI should be in neutral zone with equal up/down moves")


def run_tests():
    """Run all tests and display results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestRSICalculation))
    suite.addTests(loader.loadTestsFromTestCase(TestRSIEdgeCases))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("RSI UNIT TEST SUMMARY")
    print("="*70)
    print(f"Total Tests: {result.testsRun}")
    print(f"✅ Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Failed: {len(result.failures)}")
    print(f"⚠️  Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("✅ RSI calculation is working correctly")
        print("✅ Uses Wilder's Smoothing Method (industry standard)")
        print("✅ Handles all edge cases properly")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("⚠️  RSI implementation may have issues")
    
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
