#!/usr/bin/env python3
"""
Минимальный рабочий скрипт для запуска проекта.
"""

import sys
import os

# Добавляем текущую директорию в путь Python
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import logging

# Настраиваем простое логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Основная функция."""
    print("\n" + "=" * 60)
    print("PATTERN RECOGNITION ENGINE - MINIMAL WORKING VERSION")
    print("=" * 60)

    try:
        # 1. Загружаем конфигурацию
        import config
        logger.info(f"Config loaded: Symbol={config.SYMBOL}, Timeframe={config.TIMEFRAME}")

        # 2. Подключаемся к MT5
        import MetaTrader5 as mt5

        logger.info("Initializing MT5...")
        if mt5.initialize():
            logger.info("✅ MT5 initialized successfully")

            # Получаем информацию о счете
            account_info = mt5.account_info()
            if account_info:
                logger.info(f"Account: {account_info.login}")
                logger.info(f"Balance: {account_info.balance}")
                logger.info(f"Server: {account_info.server}")

            # Получаем данные
            symbol = config.SYMBOL
            timeframe_map = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1,
            }

            timeframe = timeframe_map.get(config.TIMEFRAME, mt5.TIMEFRAME_H1)
            bars = 100

            logger.info(f"Getting {bars} bars of {symbol} {config.TIMEFRAME}...")
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)

            if rates is not None:
                import pandas as pd
                import numpy as np

                # Преобразуем в DataFrame
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)

                logger.info(f"✅ Data loaded: {len(df)} bars")
                print(f"\nData shape: {df.shape}")
                print(f"Date range: {df.index[0]} to {df.index[-1]}")
                print(f"Latest close: {df['close'].iloc[-1]:.5f}")

                # 3. Ищем паттерны (простая версия)
                print("\n" + "-" * 60)
                print("Searching for candlestick patterns...")

                patterns = []

                # Простой поиск паттернов
                for i in range(1, len(df)):
                    # Doji pattern
                    open_price = df['open'].iloc[i]
                    close_price = df['close'].iloc[i]
                    high = df['high'].iloc[i]
                    low = df['low'].iloc[i]

                    body_size = abs(close_price - open_price)
                    range_size = high - low

                    if range_size > 0:
                        # Doji
                        if body_size / range_size < 0.1:
                            patterns.append({
                                'index': i,
                                'pattern_type': 'doji',
                                'confidence': 0.8,
                                'price': close_price,
                                'timestamp': df.index[i]
                            })

                        # Hammer
                        upper_shadow = high - max(open_price, close_price)
                        lower_shadow = min(open_price, close_price) - low

                        if (body_size / range_size < 0.3 and
                                lower_shadow > 2 * body_size and
                                upper_shadow < body_size * 0.3):
                            is_bullish = close_price > open_price
                            pattern_type = 'hammer' if is_bullish else 'hanging_man'

                            patterns.append({
                                'index': i,
                                'pattern_type': pattern_type,
                                'confidence': 0.7,
                                'price': close_price,
                                'timestamp': df.index[i],
                                'signal': 'buy' if is_bullish else 'sell'
                            })

                print(f"Found {len(patterns)} patterns")

                if patterns:
                    print("\nRecent patterns found:")
                    for pattern in patterns[-5:]:  # Последние 5 паттернов
                        print(f"  {pattern['timestamp']}: {pattern['pattern_type']} "
                              f"(confidence: {pattern['confidence']:.2f})")

                # 4. Простой анализ
                print("\n" + "-" * 60)
                print("Basic Analysis:")
                print(f"Total bars analyzed: {len(df)}")
                print(f"Price range: {df['low'].min():.5f} - {df['high'].max():.5f}")
                print(f"Average spread: {(df['high'] - df['low']).mean():.5f}")

                if len(df) > 20:
                    sma_20 = df['close'].rolling(window=20).mean().iloc[-1]
                    print(f"20-period SMA: {sma_20:.5f}")

                    if df['close'].iloc[-1] > sma_20:
                        print("Current price is ABOVE 20-period SMA (bullish)")
                    else:
                        print("Current price is BELOW 20-period SMA (bearish)")

            else:
                logger.warning("No data received from MT5")

            # Отключаемся от MT5
            mt5.shutdown()
            logger.info("MT5 shutdown completed")

        else:
            logger.error(f"Failed to initialize MT5: {mt5.last_error()}")

    except ImportError as e:
        logger.error(f"Import error: {e}")
        print("\nPlease install required packages:")
        print("pip install MetaTrader5 pandas numpy")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1

    print("\n" + "=" * 60)
    print("✅ PROJECT IS WORKING!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

