#!/usr/bin/env python3
"""
Simple test script to verify the Pattern Recognition Engine is working.
"""

import sys
import os
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logging
from utils.mt5_connector import MT5Connector
from core.data_feeder import DataFeeder
from core.pattern_detector import PatternDetector


def test_mt5_connection():
    """Test MT5 connection."""
    print("=" * 60)
    print("Testing MT5 Connection")
    print("=" * 60)

    mt5 = MT5Connector()

    # Try to connect
    if mt5.connect():
        print("✅ MT5 connection successful")

        # Get account info
        account_info = mt5.get_account_info()
        if account_info:
            print(f"   Account: {account_info.get('login')}")
            print(f"   Balance: {account_info.get('balance'):.2f}")
            print(f"   Currency: {account_info.get('currency')}")

        # Test getting data
        data = mt5.get_rates("EURUSD", "H1", 100)
        if not data.empty:
            print(f"✅ Data retrieved successfully")
            print(f"   Bars: {len(data)}")
            print(f"   Columns: {list(data.columns)}")
            print(f"   Date range: {data.index[0]} to {data.index[-1]}")
        else:
            print("❌ Failed to retrieve data")

        mt5.disconnect()
        return True
    else:
        print("❌ MT5 connection failed")
        print("   Please check:")
        print("   1. MetaTrader 5 is installed and running")
        print("   2. Demo account is created")
        print("   3. Login credentials in .env file are correct")
        return False


def test_data_feeder():
    """Test DataFeeder module."""
    print("\n" + "=" * 60)
    print("Testing DataFeeder")
    print("=" * 60)

    try:
        feeder = DataFeeder(cache_enabled=False)
        data = feeder.get_data("EURUSD", "H1", 200)

        if data.empty:
            print("❌ DataFeeder failed to load data")
            return False

        print("✅ DataFeeder working correctly")
        print(f"   Shape: {data.shape}")
        print(f"   Columns: {list(data.columns)[:10]}...")

        # Check for technical indicators
        required_indicators = ['SMA_20', 'RSI', 'MACD']
        missing = [ind for ind in required_indicators if ind not in data.columns]

        if missing:
            print(f"⚠️  Missing indicators: {missing}")
        else:
            print(f"✅ All technical indicators calculated")

        return True

    except Exception as e:
        print(f"❌ DataFeeder error: {e}")
        return False


def test_pattern_detector():
    """Test PatternDetector module."""
    print("\n" + "=" * 60)
    print("Testing PatternDetector")
    print("=" * 60)

    try:
        # First get some data
        feeder = DataFeeder(cache_enabled=False)
        data = feeder.get_data("EURUSD", "H1", 500)

        if data.empty:
            print("❌ No data for pattern detection")
            return False

        detector = PatternDetector()
        patterns = detector.detect_all_patterns(data, pattern_types=['candlestick'])

        print(f"✅ PatternDetector working correctly")
        print(f"   Patterns found: {len(patterns)}")

        if patterns:
            # Show first few patterns
            for i, pattern in enumerate(patterns[:3]):
                print(f"   Pattern {i + 1}: {pattern.get('pattern_type')} "
                      f"(confidence: {pattern.get('confidence', 0):.2f})")

        return True

    except Exception as e:
        print(f"❌ PatternDetector error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PATTERN RECOGNITION ENGINE - SYSTEM TEST")
    print("=" * 60)

    # Setup logging
    setup_logging(log_level="INFO", log_file="logs/test.log")

    tests = [
        ("MT5 Connection", test_mt5_connection),
        ("DataFeeder", test_data_feeder),
        ("PatternDetector", test_pattern_detector)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n▶ Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:20} {status}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the logs.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

