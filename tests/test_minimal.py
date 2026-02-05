#!/usr/bin/env python3
"""
Минимальный тест для проверки работоспособности проекта.
Просто запускает основные компоненты без сложных импортов.
"""

import sys
import os

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_config():
    """Тест конфигурации."""
    print("=" * 60)
    print("Тест конфигурации")
    print("=" * 60)

    try:
        import config
        print("✅ config загружен успешно")

        # Проверяем основные параметры
        print(f"   Символ: {config.SYMBOL}")
        print(f"   Таймфрейм: {config.TIMEFRAME}")
        print(f"   MT5 Сервер: {config.MT5_SERVER}")

        return True
    except Exception as e:
        print(f"❌ Ошибка загрузки config: {e}")
        return False


def test_mt5_simple():
    """Простой тест MT5."""
    print("\n" + "=" * 60)
    print("Тест подключения к MT5")
    print("=" * 60)

    try:
        import MetaTrader5 as mt5
        print("✅ MetaTrader5 импортирован")

        # Пробуем подключиться
        if mt5.initialize():
            print("✅ MT5 инициализирован")

            # Получаем информацию о терминале
            terminal_info = mt5.terminal_info()
            if terminal_info:
                print(f"   Терминал: {terminal_info.name}")
                print(f"   Папка данных: {terminal_info.data_path}")
                print(f"   Компания: {terminal_info.company}")

            # Пробуем получить данные
            rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 10)
            if rates is not None:
                print(f"✅ Данные получены: {len(rates)} баров")
            else:
                print("⚠️  Не удалось получить данные")

            mt5.shutdown()
            print("✅ MT5 отключен")
            return True
        else:
            print(f"❌ Ошибка инициализации MT5: {mt5.last_error()}")
            return False

    except Exception as e:
        print(f"❌ Ошибка MT5: {e}")
        return False


def test_core_modules():
    """Тест основных модулей."""
    print("\n" + "=" * 60)
    print("Тест основных модулей")
    print("=" * 60)

    results = []

    # Тест utils.mt5_connector
    try:
        from utils.mt5_connector import MT5Connector
        mt5_conn = MT5Connector()
        print("✅ MT5Connector создан")
        results.append(True)
    except Exception as e:
        print(f"❌ MT5Connector: {e}")
        results.append(False)

    # Тест core.pattern_detector
    try:
        from core.pattern_detector import PatternDetector
        detector = PatternDetector()
        print("✅ PatternDetector создан")
        results.append(True)
    except Exception as e:
        print(f"❌ PatternDetector: {e}")
        results.append(False)

    # Тест core.data_feeder
    try:
        from core.data_feeder import DataFeeder
        feeder = DataFeeder()
        print("✅ DataFeeder создан")
        results.append(True)
    except Exception as e:
        print(f"❌ DataFeeder: {e}")
        results.append(False)

    return all(results)


def test_import_all():
    """Тест импорта всех основных модулей."""
    print("\n" + "=" * 60)
    print("Тест всех импортов")
    print("=" * 60)

    imports_to_test = [
        "config",
        "utils.logger",
        "utils.mt5_connector",
        "utils.helpers",
        "utils.visualization",
        "core.data_feeder",
        "core.pattern_detector",
        "core.pattern_database",
        "core.backtesting",
        "core.statistics",
        "core.pattern_analyzer",
        "patterns.candlestick_patterns",
        "patterns.geometric_patterns",
        "patterns.harmonic_patterns",
    ]

    results = []

    for import_name in imports_to_test:
        try:
            __import__(import_name)
            print(f"✅ {import_name}")
            results.append(True)
        except Exception as e:
            print(f"❌ {import_name}: {e}")
            results.append(False)

    return sum(results), len(results)


def main():
    """Главная функция."""
    print("\n" + "=" * 60)
    print("МИНИМАЛЬНЫЙ ТЕСТ ПРОЕКТА")
    print("=" * 60)

    # Создаем необходимые директории
    os.makedirs("logs", exist_ok=True)

    print("\n1. Тест конфигурации...")
    test1 = test_config()

    print("\n2. Тест MT5...")
    test2 = test_mt5_simple()

    print("\n3. Тест основных модулей...")
    test3 = test_core_modules()

    print("\n4. Тест всех импортов...")
    successful, total = test_import_all()

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ:")
    print("=" * 60)

    print(f"Конфигурация: {'✅' if test1 else '❌'}")
    print(f"MT5: {'✅' if test2 else '❌'}")
    print(f"Основные модули: {'✅' if test3 else '❌'}")
    print(f"Импорты: {successful}/{total} успешно")

    if test1 and test2 and test3 and successful >= total * 0.8:
        print("\n🎉 Проект готов к запуску!")
        print("\nЗапустите: python main.py")
    else:
        print("\n⚠️  Есть проблемы, требующие исправления.")

    return test1 and test2 and test3


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

