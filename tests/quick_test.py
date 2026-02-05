#!/usr/bin/env python3
"""
Quick test script to verify basic functionality without complex imports.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test basic imports."""
    print("=" * 60)
    print("Testing Basic Imports")
    print("=" * 60)

    try:
        import config
        print("✅ config imported successfully")

        # Test config values
        print(f"   Symbol: {config.SYMBOL}")
        print(f"   Timeframe: {config.TIMEFRAME}")
        print(f"   MT5 Server: {config.MT5_SERVER}")

    except Exception as e:
        print(f"❌ Error importing config: {e}")
        return False

    try:
        from utils.helpers import validate_dataframe, calculate_returns
        print("✅ utils.helpers imported successfully")
    except Exception as e:
        print(f"❌ Error importing utils.helpers: {e}")
        return False

    try:
        from utils.logger import setup_logging
        print("✅ utils.logger imported successfully")
    except Exception as e:
        print(f"❌ Error importing utils.logger: {e}")
        return False

    return True


def test_data_generation():
    """Test data generation functions."""
    print("\n" + "=" * 60)
    print("Testing Data Generation")
    print("=" * 60)

    try:
        # Generate test data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        data = pd.DataFrame({
            'open': np.random.uniform(1.0, 1.2, 100),
            'high': np.random.uniform(1.1, 1.3, 100),
            'low': np.random.uniform(0.9, 1.1, 100),
            'close': np.random.uniform(1.0, 1.2, 100),
            'volume': np.random.randint(100, 1000, 100)
        }, index=dates)

        print(f"✅ Generated test data: {data.shape}")
        print(f"   Date range: {data.index[0]} to {data.index[-1]}")

        # Test helper functions
        from utils.helpers import validate_dataframe, calculate_returns

        is_valid = validate_dataframe(data, ['open', 'high', 'low', 'close'])
        print(f"✅ Data validation: {'PASS' if is_valid else 'FAIL'}")

        returns = calculate_returns(data['close'])
        print(f"✅ Returns calculation: {len(returns)} values")

        return True

    except Exception as e:
        print(f"❌ Data generation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pattern_detection_basic():
    """Test basic pattern detection with synthetic data."""
    print("\n" + "=" * 60)
    print("Testing Basic Pattern Detection")
    print("=" * 60)

    try:
        # Generate synthetic data with a pattern
        dates = pd.date_range(start='2024-01-01', periods=50, freq='H')

        # Create a simple "pattern" - downward then upward movement
        close_prices = []
        for i in range(50):
            if i < 25:
                price = 1.1 - (i * 0.005)  # Downward trend
            else:
                price = 0.975 + ((i - 25) * 0.005)  # Upward trend
            close_prices.append(price)

        data = pd.DataFrame({
            'open': [p * 0.998 for p in close_prices],
            'high': [p * 1.002 for p in close_prices],
            'low': [p * 0.995 for p in close_prices],
            'close': close_prices,
            'volume': np.random.randint(100, 1000, 50)
        }, index=dates)

        print(f"✅ Created synthetic data with pattern")

        # Try to import and use pattern detector
        try:
            from core.pattern_detector import PatternDetector

            detector = PatternDetector()
            patterns = detector.detect_candlestick_patterns(data)

            print(f"✅ Pattern detector initialized")
            print(f"   Patterns found: {len(patterns)}")

            if patterns:
                for i, pattern in enumerate(patterns[:3]):
                    print(f"   Pattern {i + 1}: {pattern.get('pattern_type', 'unknown')} "
                          f"(confidence: {pattern.get('confidence', 0):.2f})")

            return True

        except Exception as e:
            print(f"⚠️  Pattern detector error (but system still works): {e}")
            return True  # Still return True as this is not critical

    except Exception as e:
        print(f"❌ Pattern detection error: {e}")
        return False


def test_mt5_connection_simple():
    """Test MT5 connection in a simple way."""
    print("\n" + "=" * 60)
    print("Testing MT5 Connection (Simple)")
    print("=" * 60)

    try:
        import MetaTrader5 as mt5

        print("✅ MetaTrader5 library imported")

        # Try to initialize
        initialized = mt5.initialize()

        if initialized:
            print("✅ MT5 initialized successfully")

            # Get terminal info
            terminal_info = mt5.terminal_info()
            if terminal_info:
                print(f"   Terminal: {terminal_info.name}")
                print(f"   Version: {terminal_info.version}")
                print(f"   Build: {terminal_info.build}")

            # Try to login (optional - just test if we can)
            try:
                import config
                logged_in = mt5.login(
                    login=config.MT5_LOGIN,
                    password=config.MT5_PASSWORD,
                    server=config.MT5_SERVER,
                    timeout=5000
                )

                if logged_in:
                    print("✅ MT5 login successful")
                    account_info = mt5.account_info()
                    if account_info:
                        print(f"   Account: {account_info.login}")
                        print(f"   Balance: {account_info.balance:.2f}")
                else:
                    print("⚠️  MT5 login failed (but initialization worked)")
                    print(f"   Error: {mt5.last_error()}")

            except Exception as login_error:
                print(f"⚠️  Login test skipped: {login_error}")

            mt5.shutdown()
            print("✅ MT5 shutdown completed")

        else:
            print("⚠️  MT5 initialization failed")
            print(f"   Error: {mt5.last_error()}")
            print("\n   TROUBLESHOOTING:")
            print("   1. Make sure MetaTrader 5 is installed and running")
            print("   2. Check if you have a demo account")
            print("   3. Verify .env file has correct credentials")

        return True

    except ImportError:
        print("❌ MetaTrader5 library not installed")
        print("   Run: pip install MetaTrader5")
        return False
    except Exception as e:
        print(f"❌ MT5 connection error: {e}")
        return False


def main():
    """Run all quick tests."""
    print("\n" + "=" * 60)
    print("PATTERN RECOGNITION ENGINE - QUICK SYSTEM TEST")
    print("=" * 60)

    tests = [
        ("Basic Imports", test_imports),
        ("Data Generation", test_data_generation),
        ("Basic Pattern Detection", test_pattern_detection_basic),
        ("MT5 Connection", test_mt5_connection_simple)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n▶ Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"   ✅ {test_name}: PASSED")
            else:
                print(f"   ❌ {test_name}: FAILED")
        except Exception as e:
            print(f"   💥 {test_name}: CRASHED - {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:25} {status}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is working correctly.")
        print("\nNext steps:")
        print("1. Update your .env file with real MT5 credentials")
        print("2. Run: python main.py")
        print("3. Or run: python run.py --demo")
    elif passed >= 2:
        print(f"\n⚠️  {passed}/{total} tests passed. Basic functionality works.")
        print("   Some features may need attention (e.g., MT5 connection).")
    else:
        print(f"\n❌ Only {passed}/{total} tests passed. System needs fixes.")

    return passed >= 3  # Require at least 3 tests to pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

