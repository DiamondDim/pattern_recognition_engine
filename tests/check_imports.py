#!/usr/bin/env python3
"""
Check all imports in the project.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_import(module_name, import_name=None):
    """Check if a module can be imported."""
    try:
        if import_name:
            exec(f"from {module_name} import {import_name}")
            print(f"✅ {module_name}.{import_name}")
        else:
            exec(f"import {module_name}")
            print(f"✅ {module_name}")
        return True
    except Exception as e:
        if import_name:
            print(f"❌ {module_name}.{import_name}: {e}")
        else:
            print(f"❌ {module_name}: {e}")
        return False


def main():
    """Check all imports."""
    print("=" * 60)
    print("IMPORT CHECK")
    print("=" * 60)

    imports_to_check = [
        # Core modules
        ("config", None),
        ("utils.helpers", "validate_dataframe"),
        ("utils.helpers", "calculate_returns"),
        ("utils.logger", "setup_logging"),
        ("utils.mt5_connector", "MT5Connector"),
        ("utils.visualization", "plot_patterns"),

        # Core package
        ("core.data_feeder", "DataFeeder"),
        ("core.pattern_detector", "PatternDetector"),
        ("core.pattern_database", "PatternDatabase"),
        ("core.backtesting", "BacktestingEngine"),
        ("core.statistics", "Statistics"),

        # Patterns
        ("patterns.candlestick_patterns", "CandlestickPatterns"),
        ("patterns.geometric_patterns", "GeometricPatterns"),
        ("patterns.harmonic_patterns", "HarmonicPatterns"),
    ]

    results = []

    for module_name, import_name in imports_to_check:
        success = check_import(module_name, import_name)
        results.append(success)

    passed = sum(results)
    total = len(results)

    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed}/{total} imports successful")
    print("=" * 60)

    if passed == total:
        print("🎉 All imports successful!")
        return True
    else:
        print(f"⚠️  {total - passed} imports failed")
        print("\nTROUBLESHOOTING:")
        print("1. Make sure you're in the project root directory")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Check for missing __init__.py files")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

