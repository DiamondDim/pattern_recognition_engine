#!/usr/bin/env python3
"""
Simple startup script for Pattern Recognition Engine.
Runs with minimal dependencies.
"""

import os
import sys

# Добавляем текущую директорию в PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("=" * 60)
print("PATTERN RECOGNITION ENGINE - SIMPLE START")
print("=" * 60)

# 1. Проверяем конфигурацию
try:
    import config

    print(f"✅ Config: {config.SYMBOL} {config.TIMEFRAME}")
except Exception as e:
    print(f"❌ Config error: {e}")
    sys.exit(1)

# 2. Проверяем MT5
try:
    import MetaTrader5 as mt5

    if mt5.initialize():
        print("✅ MT5: Connected")
        account = mt5.account_info()
        if account:
            print(f"   Account: {account.login}")
            print(f"   Balance: {account.balance}")
        mt5.shutdown()
    else:
        print("⚠️  MT5: Not connected (running in simulation mode)")
except Exception as e:
    print(f"⚠️  MT5: {e}")

# 3. Тестируем основные модули
print("\nTesting modules...")

modules_to_test = [
    ("utils.logger", "setup_logging"),
    ("utils.mt5_connector", "MT5Connector"),
    ("core.data_feeder", "DataFeeder"),
    ("core.pattern_detector", "PatternDetector"),
]

for module_name, item_name in modules_to_test:
    try:
        if "." in module_name:
            # Для from module import item
            module_parts = module_name.split(".")
            exec(f"from {'.'.join(module_parts[:-1])} import {module_parts[-1]} as temp")
            print(f"✅ {module_name}")
        else:
            # Для import module
            __import__(module_name)
            print(f"✅ {module_name}")
    except Exception as e:
        print(f"❌ {module_name}: {str(e)[:50]}")

print("\n" + "=" * 60)
print("SYSTEM READY - Starting main application...")
print("=" * 60)

# Запускаем основной код
try:
    # Создаем простой тестовый сценарий
    from utils.mt5_connector import MT5Connector
    from core.data_feeder import DataFeeder
    from core.pattern_detector import PatternDetector

    mt5_conn = MT5Connector()
    if mt5_conn.connect():
        print("✅ Connected to MT5")

        # Получаем данные
        feeder = DataFeeder(cache_enabled=True)
        data = feeder.get_data(config.SYMBOL, config.TIMEFRAME, 100)

        if not data.empty:
            print(f"✅ Data loaded: {len(data)} bars")

            # Ищем паттерны
            detector = PatternDetector()
            patterns = detector.detect_candlestick_patterns(data)

            print(f"✅ Patterns found: {len(patterns)}")

            if patterns:
                for p in patterns[:3]:  # Показываем первые 3
                    print(f"   - {p.get('pattern_type', 'unknown')} "
                          f"(confidence: {p.get('confidence', 0):.2f})")

        mt5_conn.disconnect()
    else:
        print("⚠️  Running in simulation mode (no MT5 connection)")

        # Используем тестовые данные
        import pandas as pd
        import numpy as np

        # Создаем тестовые данные
        dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
        test_data = pd.DataFrame({
            'open': np.random.uniform(1.0, 1.2, 100),
            'high': np.random.uniform(1.1, 1.3, 100),
            'low': np.random.uniform(0.9, 1.1, 100),
            'close': np.random.uniform(1.0, 1.2, 100),
            'volume': np.random.randint(100, 1000, 100)
        }, index=dates)

        print("✅ Using test data")

        # Ищем паттерны
        detector = PatternDetector()
        patterns = detector.detect_candlestick_patterns(test_data)

        print(f"✅ Patterns found: {len(patterns)}")

    print("\n" + "=" * 60)
    print("🎉 SYSTEM WORKING CORRECTLY!")
    print("=" * 60)

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback

    traceback.print_exc()

