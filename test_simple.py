#!/usr/bin/env python3
"""
Simple test to check if project works.
"""

import sys
import os

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing Pattern Recognition Engine...")

# Тест 1: Конфигурация
try:
    import config
    print(f"✅ Config: {config.SYMBOL} {config.TIMEFRAME}")
except Exception as e:
    print(f"❌ Config: {e}")

# Тест 2: MT5
try:
    import MetaTrader5 as mt5
    print("✅ MetaTrader5 library")
except Exception as e:
    print(f"❌ MetaTrader5: {e}")

# Тест 3: Основные модули
try:
    from utils.mt5_connector import MT5Connector
    print("✅ MT5Connector")
except Exception as e:
    print(f"❌ MT5Connector: {e}")

try:
    from core.data_feeder import DataFeeder
    print("✅ DataFeeder")
except Exception as e:
    print(f"❌ DataFeeder: {e}")

try:
    from core.pattern_detector import PatternDetector
    print("✅ PatternDetector")
except Exception as e:
    print(f"❌ PatternDetector: {e}")

print("\n" + "=" * 60)
print("If you see at least '✅ Config' and '✅ MetaTrader5 library'")
print("then the basic setup is correct.")
print("=" * 60)

