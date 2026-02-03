"""
Модуль загрузки данных с интеграцией MT5Connector для RFD-инструментов
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import threading
import queue
import time

try:
    from config import config
except ImportError:
    # Для обратной совместимости
    from config import config

from ..utils.mt5_connector import mt5_connector


class DataFeeder:
    """Класс для загрузки и управления рыночными данными"""

    def __init__(self, custom_config: Dict = None):
        self.config = custom_config or {}
        self.data_cache = {}  # Кэш данных: {symbol: {timeframe: DataFrame}}
        self.update_queue = queue.Queue()
        self.running = False
        self.update_thread = None

        # Используем настройки из конфига по умолчанию
        self.symbols = config.MT5.SYMBOLS
        self.timeframes = config.MT5.TIMEFRAMES

        # Инициализируем MT5
        if config.MT5.ENABLED and not mt5_connector.initialized:
            if config.MT5.AUTO_CONNECT:
                if not mt5_connector._init_mt5():
                    print("Предупреждение: MT5 не инициализирован, будут использоваться тестовые данные")

    def get_data(self, symbol: str, timeframe: str,
                 bars: int = None, use_cache: bool = True) -> pd.DataFrame:
        """
        Получение данных для указанного символа и таймфрейма

        Args:
            symbol: Торговый символ (без префикса RFD)
            timeframe: Таймфрейм
            bars: Количество баров (если None - из конфига)
            use_cache: Использовать кэш

        Returns:
            DataFrame с данными
        """
        if bars is None:
            bars = config.DETECTION.MAX_CANDLES_FOR_PATTERN

        # Проверяем кэш
        cache_key = f"{symbol}_{timeframe}"
        if use_cache and cache_key in self.data_cache:
            cached_data = self.data_cache[cache_key]
            if len(cached_data) >= bars:
                return cached_data.tail(bars).copy()

        # Получаем данные через MT5Connector
        if config.MT5.ENABLED and mt5_connector.initialized:
            data = mt5_connector.get_historical_data(symbol, timeframe, bars)
        else:
            print(f"MT5 не доступен, используем тестовые данные для {symbol}")
            data = self._generate_test_data(symbol, timeframe, bars)

        if data.empty:
            print(f"Не удалось получить данные для {symbol}, используем тестовые данные")
            data = self._generate_test_data(symbol, timeframe, bars)

        # Кэшируем данные
        self.data_cache[cache_key] = data

        return data.copy()

    def get_multiple_symbols(self, symbols: List[str], timeframe: str,
                             bars: int = None) -> Dict[str, pd.DataFrame]:
        """Получение данных для нескольких символов"""
        result = {}
        for symbol in symbols:
            data = self.get_data(symbol, timeframe, bars)
            if not data.empty:
                result[symbol] = data
        return result

    def update_data(self, symbol: str, timeframe: str):
        """Обновление данных для символа"""
        try:
            # Получаем последние 100 баров для обновления
            new_data = mt5_connector.get_historical_data(symbol, timeframe, 100)

            if new_data.empty:
                return

            cache_key = f"{symbol}_{timeframe}"

            if cache_key in self.data_cache:
                # Объединяем с существующими данными
                old_data = self.data_cache[cache_key]
                combined = pd.concat([old_data, new_data])
                combined = combined[~combined.index.duplicated(keep='last')]
                self.data_cache[cache_key] = combined.sort_index()
            else:
                self.data_cache[cache_key] = new_data

            print(f"Данные обновлены для {symbol} ({timeframe}): {len(new_data)} новых баров")

        except Exception as e:
            print(f"Ошибка обновления данных для {symbol}: {e}")

    def get_current_prices(self, symbols: List[str]) -> Dict[str, Dict]:
        """Получение текущих цен для списка символов"""
        prices = {}
        for symbol in symbols:
            price_data = mt5_connector.get_current_price(symbol)
            if price_data:
                prices[symbol] = price_data
        return prices

    def start_real_time_updates(self, update_interval: int = 60):
        """Запуск потокового обновления данных"""
        if self.running:
            return

        self.running = True
        self.update_thread = threading.Thread(
            target=self._update_worker,
            args=(update_interval,),
            daemon=True
        )
        self.update_thread.start()
        print(f"Запущено потоковое обновление данных каждые {update_interval} секунд")

    def stop_real_time_updates(self):
        """Остановка потокового обновления"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        print("Потоковое обновление данных остановлено")

    def _update_worker(self, interval: int):
        """Рабочий поток для обновления данных"""
        while self.running:
            try:
                # Обновляем данные для всех символов и таймфреймов из конфига
                for symbol in self.symbols:
                    for timeframe in self.timeframes:
                        self.update_data(symbol, timeframe)

                # Ждем указанный интервал
                time.sleep(interval)

            except Exception as e:
                print(f"Ошибка в потоке обновления: {e}")
                time.sleep(interval)

    def _generate_test_data(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        """Генерация тестовых данных (используется при отсутствии подключения к MT5)"""
        print(f"Генерация тестовых данных для {symbol} ({timeframe})")

        # Определяем базовую цену в зависимости от символа
        base_prices = {
            'EURUSD': 1.0800,
            'GBPUSD': 1.2600,
            'USDJPY': 150.00,
            'XAUUSD': 1950.00,
            'USDRUB': 90.00,
            'EURGBP': 0.8600
        }

        base_price = base_prices.get(symbol, 1.0000)

        # Определяем частоту в зависимости от таймфрейма
        freq_map = {
            'M1': '1min',
            'M5': '5min',
            'M15': '15min',
            'M30': '30min',
            'H1': '1H',
            'H4': '4H',
            'D1': '1D',
            'W1': '1W',
            'MN1': '1M'
        }

        freq = freq_map.get(timeframe, '1H')

        # Генерируем даты
        end_date = datetime.now()
        dates = pd.date_range(end=end_date, periods=bars, freq=freq)

        # Генерируем случайные цены с трендом
        np.random.seed(42)  # Для воспроизводимости
        returns = np.random.normal(0.0001, 0.005, bars)
        price = base_price * np.exp(np.cumsum(returns))

        # Создаем OHLC данные
        data = pd.DataFrame(index=dates)
        data['close'] = price

        # Генерируем open, high, low
        data['open'] = data['close'].shift(1).fillna(base_price)
        data['high'] = data[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.001, bars))
        data['low'] = data[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.001, bars))

        # Объемы
        data['volume'] = np.random.randint(1000, 100000, bars)

        # Проверяем, что low <= high <= close и low <= open <= high
        data['high'] = data[['high', 'close', 'open']].max(axis=1)
        data['low'] = data[['low', 'close', 'open']].min(axis=1)

        return data

    def clear_cache(self):
        """Очистка кэша данных"""
        self.data_cache.clear()
        print("Кэш данных очищен")

    def __del__(self):
        self.stop_real_time_updates()

