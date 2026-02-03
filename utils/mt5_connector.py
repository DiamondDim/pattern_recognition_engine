"""
Утилиты для работы с MetaTrader 5 с учетом специфики RFD-инструментов
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import time
import traceback

# Исправляем импорт
try:
    from config import config
except ImportError:
    # Создаем простой конфиг для тестирования
    class SimpleConfig:
        class MT5Config:
            PATH = r"C:\Program Files\MetaTrader 5 Alfa-Forex\terminal64.exe"
            LOGIN = 0
            PASSWORD = ''
            SERVER = 'Alfa-Forex-MT5'
            SYMBOL_PREFIX = "RFD."
            TIMEOUT = 10000

        MT5 = MT5Config()

    config = SimpleConfig()
    print("Используется простой конфиг MT5 для тестирования")


class MT5Connector:
    """Класс для работы с MetaTrader 5 с поддержкой RFD-префиксов"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = False
            self.account_info = None
            self.terminal_info = None

            # Используем настройки из конфига
            self.mt5_config = config.MT5
            self._init_mt5()

    def _init_mt5(self):
        """Инициализация подключения к MT5"""
        try:
            print(f"Инициализация MT5 с терминалом: {self.mt5_config.PATH}")

            # Инициализируем MT5 с указанием пути к терминалу
            if not mt5.initialize(
                path=self.mt5_config.PATH,
                login=self.mt5_config.LOGIN,
                password=self.mt5_config.PASSWORD,
                server=self.mt5_config.SERVER,
                timeout=self.mt5_config.TIMEOUT
            ):
                error = mt5.last_error()
                print(f"Ошибка инициализации MT5: {error}")

                # Пробуем инициализировать без пути (если терминал уже запущен)
                print("Попытка подключения к запущенному терминалу...")
                if not mt5.initialize():
                    print(f"Ошибка подключения: {mt5.last_error()}")
                    return False

            self.initialized = True

            # Получаем информацию о счете
            self.account_info = mt5.account_info()
            self.terminal_info = mt5.terminal_info()

            print(f"Успешно подключены к MT5")

            # Безопасный вывод информации о сервере
            if self.terminal_info:
                server_info = 'N/A'
                if hasattr(self.terminal_info, 'community_server'):
                    server_info = self.terminal_info.community_server
                elif hasattr(self.terminal_info, 'server'):
                    server_info = self.terminal_info.server
                elif hasattr(self.terminal_info, 'name'):
                    server_info = self.terminal_info.name

                print(f"  Сервер: {server_info}")
            else:
                print("  Сервер: N/A")

            if self.account_info:
                print(f"  Счет: {self.account_info.login}")
                print(f"  Баланс: {self.account_info.balance}")
                print(f"  Валюта: {self.account_info.currency}")
            else:
                print("  Счет: N/A")
                print("  Баланс: N/A")
                print("  Валюта: N/A")

            print(f"  Префикс инструментов: {self.mt5_config.SYMBOL_PREFIX}")

            return True

        except Exception as e:
            print(f"Критическая ошибка инициализации MT5: {e}")
            traceback.print_exc()
            return False

    def add_symbol_prefix(self, symbol: str) -> str:
        """Добавление префикса RFD к символу"""
        if not symbol.startswith(self.mt5_config.SYMBOL_PREFIX):
            return f"{self.mt5_config.SYMBOL_PREFIX}{symbol}"
        return symbol

    def remove_symbol_prefix(self, symbol: str) -> str:
        """Удаление префикса RFD из символа"""
        if symbol.startswith(self.mt5_config.SYMBOL_PREFIX):
            return symbol[len(self.mt5_config.SYMBOL_PREFIX):]
        return symbol

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Получение информации о торговом инструменте"""
        try:
            full_symbol = self.add_symbol_prefix(symbol)
            info = mt5.symbol_info(full_symbol)

            if info is None:
                print(f"Инструмент {full_symbol} не найден")
                return None

            return {
                'name': self.remove_symbol_prefix(info.name),
                'full_name': info.name,
                'digits': info.digits,
                'point': info.point
            }

        except Exception as e:
            print(f"Ошибка получения информации о символе {symbol}: {e}")
            return None

    def get_historical_data(self, symbol: str, timeframe: str,
                          bars: int = 1000, from_date: datetime = None) -> pd.DataFrame:
        """
        Получение исторических данных с учетом RFD-префиксов

        Args:
            symbol: Название символа (без префикса)
            timeframe: Таймфрейм ('M1', 'H1', 'D1' и т.д.)
            bars: Количество баров
            from_date: Дата, начиная с которой получать данные

        Returns:
            DataFrame с историческими данными
        """
        try:
            if not self.initialized:
                if not self._init_mt5():
                    return pd.DataFrame()

            full_symbol = self.add_symbol_prefix(symbol)

            # Конвертируем timeframe в формат MT5
            tf_mapping = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1,
                'W1': mt5.TIMEFRAME_W1,
                'MN1': mt5.TIMEFRAME_MN1
            }

            if timeframe not in tf_mapping:
                print(f"Неподдерживаемый таймфрейм: {timeframe}")
                return pd.DataFrame()

            mt5_timeframe = tf_mapping[timeframe]

            # Получаем данные
            if from_date:
                rates = mt5.copy_rates_from(full_symbol, mt5_timeframe, from_date, bars)
            else:
                rates = mt5.copy_rates_from_pos(full_symbol, mt5_timeframe, 0, bars)

            if rates is None or len(rates) == 0:
                print(f"Не удалось получить данные для {full_symbol}")
                return pd.DataFrame()

            # Конвертируем в DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            # Переименовываем колонки
            df.columns = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']

            print(f"Получено {len(df)} баров для {symbol} ({timeframe})")
            return df

        except Exception as e:
            print(f"Ошибка получения данных для {symbol}: {e}")
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """Получение текущих цен bid/ask"""
        try:
            full_symbol = self.add_symbol_prefix(symbol)
            tick = mt5.symbol_info_tick(full_symbol)

            if tick is None:
                return None

            return {
                'symbol': symbol,
                'time': pd.to_datetime(tick.time, unit='s'),
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume
            }
        except Exception as e:
            print(f"Ошибка получения текущей цены для {symbol}: {e}")
            return None

    def check_market_hours(self, symbol: str) -> bool:
        """Проверка, открыт ли рынок для торговли"""
        try:
            full_symbol = self.add_symbol_prefix(symbol)
            info = mt5.symbol_info(full_symbol)

            if info is None:
                return False

            # Простая проверка - всегда возвращаем True для тестирования
            return True

        except Exception as e:
            print(f"Ошибка проверки времени торговли: {e}")
            return False

    def shutdown(self):
        """Завершение работы с MT5"""
        if self.initialized:
            mt5.shutdown()
            self.initialized = False
            print("MT5 соединение закрыто")

    def __del__(self):
        self.shutdown()


# Синглтон экземпляр
mt5_connector = MT5Connector()

