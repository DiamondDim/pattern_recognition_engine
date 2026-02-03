"""
Главный модуль Pattern Recognition Engine
"""

import sys
import argparse
import time
from datetime import datetime
import pandas as pd

# Импорты ядра
from core.data_feeder import DataFeeder
from core.pattern_detector import PatternDetector
from core.pattern_database import PatternDatabase

# Импорты утилит
from utils.logger import setup_logger
from utils.visualization import plot_patterns

# Импорт конфигурации
try:
    from config import config
except ImportError:
    # Для обратной совместимости
    from config import config

logger = setup_logger(__name__)


def main():
    """Главная функция приложения"""

    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(
        description='Pattern Recognition Engine for Financial Markets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py --mode detect --symbol EURUSD --timeframe H1
  python main.py --mode backtest --symbol EURUSD --timeframe H1 --bars 1000
  python main.py --mode gui
  python main.py --mode scan --all-symbols
        """
    )

    parser.add_argument('--mode', choices=['detect', 'backtest', 'gui', 'scan', 'test'],
                       default='detect', help='Режим работы')
    parser.add_argument('--symbol', default='EURUSD', help='Торговый символ')
    parser.add_argument('--timeframe', default='H1', help='Таймфрейм')
    parser.add_argument('--bars', type=int, default=1000, help='Количество баров')
    parser.add_argument('--all-symbols', action='store_true', help='Сканировать все символы')
    parser.add_argument('--plot', action='store_true', help='Показать график с паттернами')
    parser.add_argument('--verbose', '-v', action='store_true', help='Подробный вывод')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("PATTERN RECOGNITION ENGINE")
    print(f"Запуск в режиме: {args.mode}")
    print(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")

    try:
        if args.mode == 'detect':
            run_detection_mode(args)
        elif args.mode == 'backtest':
            run_backtest_mode(args)
        elif args.mode == 'gui':
            run_gui_mode()
        elif args.mode == 'scan':
            run_scan_mode(args)
        elif args.mode == 'test':
            run_test_mode()
        else:
            print(f"Режим {args.mode} не поддерживается")

    except KeyboardInterrupt:
        print("\n\nПрервано пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        print(f"\nОшибка выполнения: {e}")
    finally:
        print("\nЗавершение работы Pattern Recognition Engine")


def run_detection_mode(args):
    """Режим детекции паттернов"""
    print(f"=== ДЕТЕКЦИЯ ПАТТЕРНОВ ===")
    print(f"Символ: {args.symbol}")
    print(f"Таймфрейм: {args.timeframe}")
    print(f"Баров для анализа: {args.bars}")

    # Инициализация компонентов
    feeder = DataFeeder()
    detector = PatternDetector()
    database = PatternDatabase()

    try:
        # Получение данных
        print(f"\nПолучение данных для {args.symbol}...")

        # Используем настройки из config
        data = feeder.get_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            bars=args.bars
        )

        if data.empty:
            print(f"Не удалось получить данные для {args.symbol}")
            return

        print(f"Получено {len(data)} баров")
        if len(data) > 0:
            print(f"Период: {data.index[0]} - {data.index[-1]}")

        # Детекция паттернов
        print("\nЗапуск детекции паттернов...")
        start_time = time.time()

        patterns = detector.detect(data, symbol=args.symbol, timeframe=args.timeframe)

        elapsed = time.time() - start_time
        print(f"Детекция завершена за {elapsed:.2f} секунд")

        # Обработка результатов
        if patterns:
            print(f"\nНайдено {len(patterns)} паттернов:")

            for i, pattern in enumerate(patterns, 1):
                print(f"\n{i}. {pattern.get('name', 'Unknown')} ({pattern.get('type', 'unknown')})")
                print(f"   Качество: {pattern.get('quality', 0):.2%}")

                if 'timestamp' in pattern:
                    print(f"   Время: {pattern['timestamp']}")

                if 'price' in pattern:
                    print(f"   Цена: {pattern['price']}")

                if 'direction' in pattern:
                    print(f"   Направление: {pattern['direction']}")

                # Сохранение в базу данных
                try:
                    database.save_pattern(
                        symbol=args.symbol,
                        timeframe=args.timeframe,
                        pattern_name=pattern.get('name', 'Unknown'),
                        pattern_type=pattern.get('type', 'unknown'),
                        quality=pattern.get('quality', 0),
                        metadata=pattern
                    )
                except Exception as e:
                    print(f"   Ошибка сохранения в БД: {e}")

            # Визуализация
            if args.plot and config.VISUALIZATION.ENABLE_PLOTTING:
                print("\nГенерация графика...")
                plot_patterns(data, patterns, args.symbol, args.timeframe)

        else:
            print("Паттерны не найдены")

        # Статистика
        print(f"\n=== СТАТИСТИКА ===")
        stats = detector.get_statistics() if hasattr(detector, 'get_statistics') else {}
        for key, value in stats.items():
            print(f"{key}: {value}")

    except Exception as e:
        logger.error(f"Ошибка в режиме детекции: {e}", exc_info=True)
        print(f"Ошибка: {e}")

    finally:
        if hasattr(feeder, 'clear_cache'):
            feeder.clear_cache()
        print("\nДетекция завершена")


def run_backtest_mode(args):
    """Режим тестирования стратегии"""
    print("=== BACKTESTING MODE ===")
    print(f"В разработке...")
    # TODO: Реализовать backtesting с учетом RFD-инструментов


def run_gui_mode():
    """Режим графического интерфейса"""
    print("=== GUI MODE ===")
    try:
        from gui.main_window import PatternRecognitionApp
        app = PatternRecognitionApp()
        app.run()
    except ImportError as e:
        print(f"GUI модуль не доступен: {e}")
    except Exception as e:
        print(f"Ошибка запуска GUI: {e}")


def run_scan_mode(args):
    """Режим сканирования всех символов"""
    print("=== SCAN ALL SYMBOLS ===")

    # Используем символы из конфига
    symbols_to_scan = config.MT5.SYMBOLS if args.all_symbols else [args.symbol]

    feeder = DataFeeder()
    detector = PatternDetector()
    database = PatternDatabase()

    total_patterns = 0

    for symbol in symbols_to_scan:
        print(f"\nСканирование {symbol}...")

        for timeframe in config.MT5.TIMEFRAMES:
            try:
                data = feeder.get_data(symbol, timeframe, 500)
                if len(data) < 50:
                    print(f"  {timeframe}: недостаточно данных ({len(data)} баров)")
                    continue

                patterns = detector.detect(data, symbol=symbol, timeframe=timeframe)

                if patterns:
                    print(f"  {timeframe}: найдено {len(patterns)} паттернов")
                    total_patterns += len(patterns)

                    for pattern in patterns:
                        database.save_pattern(
                            symbol=symbol,
                            timeframe=timeframe,
                            pattern_name=pattern.get('name', 'Unknown'),
                            pattern_type=pattern.get('type', 'unknown'),
                            quality=pattern.get('quality', 0),
                            metadata=pattern
                        )
                else:
                    print(f"  {timeframe}: паттерны не найдены")

            except Exception as e:
                print(f"  {timeframe}: ошибка - {e}")

    print(f"\nСканирование завершено. Всего найдено {total_patterns} паттернов")


def run_test_mode():
    """Тестовый режим"""
    print("=== TEST MODE ===")

    # Тест подключения к MT5
    from utils.mt5_connector import mt5_connector

    print("Тестирование подключения к MT5...")
    if mt5_connector.initialized:
        print("✓ MT5 подключен успешно")

        # Тест получения данных
        print("\nТест получения данных...")
        data = mt5_connector.get_historical_data('EURUSD', 'H1', 100)
        if not data.empty:
            print(f"✓ Данные получены: {len(data)} баров")
            print(f"  Период: {data.index[0]} - {data.index[-1]}")
        else:
            print("✗ Не удалось получить данные")
    else:
        print("✗ MT5 не подключен")

    print("\nТестирование завершено")


if __name__ == "__main__":
    main()

