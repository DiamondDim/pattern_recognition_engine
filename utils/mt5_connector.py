import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Optional, Dict, Any
import logging

# Настройка логирования
logger = logging.getLogger(__name__)


class MT5Connector:
    """
    Класс для подключения к MetaTrader 5 и получения данных
    """

    def __init__(self):
        """Инициализация коннектора MT5"""
        self.connected = False
        self.login = None
        self.password = None
        self.server = None
        self.symbol_cache = {}
        self._initialize_params()

    def _initialize_params(self):
        """Инициализация параметров из переменных окружения или конфига"""
        try:
            from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
            self.login = MT5_LOGIN
            self.password = MT5_PASSWORD
            self.server = MT5_SERVER
            logger.info("Параметры MT5 загружены из config.py")
        except ImportError:
            import os
            self.login = os.getenv('MT5_LOGIN')
            self.password = os.getenv('MT5_PASSWORD')
            self.server = os.getenv('MT5_SERVER')
            logger.info("Параметры MT5 загружены из переменных окружения")
        except Exception as e:
            logger.warning(f"Не удалось загрузить параметры MT5: {e}")

    def connect(self) -> bool:
        """
        Подключение к терминалу MetaTrader 5

        Returns:
            bool: True если подключение успешно, False в противном случае
        """
        try:
            if not mt5.initialize():
                logger.error(f"Ошибка инициализации MT5: {mt5.last_error()}")
                return False

            self.connected = True
            logger.info("Успешно подключено к MT5")

            # Если есть учетные данные, пытаемся авторизоваться
            if self.login and self.password and self.server:
                authorized = mt5.login(
                    login=int(self.login),
                    password=self.password,
                    server=self.server
                )
                if authorized:
                    logger.info(f"Авторизован как {self.login} на сервере {self.server}")
                else:
                    logger.warning(f"Не удалось авторизоваться: {mt5.last_error()}")

            return True

        except Exception as e:
            logger.error(f"Критическая ошибка подключения к MT5: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Отключение от терминала MT5"""
        try:
            mt5.shutdown()
            self.connected = False
            logger.info("Отключено от MT5")
        except Exception as e:
            logger.error(f"Ошибка при отключении от MT5: {e}")

    def get_full_symbol_name(self, base_symbol: str) -> str:
        """
        Получение полного имени символа с суффиксом

        Args:
            base_symbol (str): Базовое имя символа (например, 'EURUSD')

        Returns:
            str: Полное имя символа (например, 'EURUSDrfd')
        """
        if base_symbol in self.symbol_cache:
            return self.symbol_cache[base_symbol]

        try:
            # Получаем все доступные символы
            all_symbols = mt5.symbols_get()

            # Ищем символы, начинающиеся с базового имени
            matching_symbols = []
            for symbol in all_symbols:
                if symbol.name.startswith(base_symbol):
                    matching_symbols.append(symbol.name)

            if matching_symbols:
                # Предпочитаем символ без суффикса, если есть
                if base_symbol in matching_symbols:
                    full_symbol = base_symbol
                else:
                    full_symbol = matching_symbols[0]

                self.symbol_cache[base_symbol] = full_symbol
                logger.info(f"Найден символ: {base_symbol} -> {full_symbol}")
                return full_symbol
            else:
                logger.warning(f"Символ {base_symbol} не найден среди {len(all_symbols)} доступных")
                return base_symbol

        except Exception as e:
            logger.error(f"Ошибка поиска символа {base_symbol}: {e}")
            return base_symbol

    def get_rates(self, symbol: str, timeframe: str, bars: int = 1000) -> pd.DataFrame:
        """
        Получение исторических данных

        Args:
            symbol (str): Имя символа
            timeframe (str): Таймфрейм ('M1', 'M5', 'H1', etc.)
            bars (int): Количество баров для получения

        Returns:
            pd.DataFrame: DataFrame с историческими данными
        """
        if not self.connected:
            if not self.connect():
                logger.error("Не удалось подключиться к MT5 для получения данных")
                return pd.DataFrame()

        try:
            # Преобразуем таймфрейм в формат MT5
            timeframe_map = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1,
                'W1': mt5.TIMEFRAME_W1,
                'MN1': mt5.TIMEFRAME_MN1,
            }

            mt5_timeframe = timeframe_map.get(timeframe.upper())
            if mt5_timeframe is None:
                logger.error(f"Неподдерживаемый таймфрейм: {timeframe}")
                return pd.DataFrame()

            # Получаем полное имя символа
            full_symbol = self.get_full_symbol_name(symbol)

            # Получаем исторические данные
            rates = mt5.copy_rates_from_pos(full_symbol, mt5_timeframe, 0, bars)

            if rates is None or len(rates) == 0:
                logger.error(f"Не удалось получить данные для {full_symbol}: {mt5.last_error()}")
                return pd.DataFrame()

            # Преобразуем в DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            # Стандартизируем названия колонок
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'tick_volume': 'Volume'
            }, inplace=True)

            logger.info(f"Получено {len(df)} баров для {full_symbol} ({timeframe})")
            return df

        except Exception as e:
            logger.error(f"Ошибка при получении данных для {symbol}: {e}")
            return pd.DataFrame()

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Получение информации о символе

        Args:
            symbol (str): Имя символа

        Returns:
            Optional[Dict]: Словарь с информацией о символе или None
        """
        if not self.connected:
            if not self.connect():
                return None

        try:
            full_symbol = self.get_full_symbol_name(symbol)
            info = mt5.symbol_info(full_symbol)

            if info is None:
                logger.warning(f"Информация о символе {full_symbol} не найдена")
                return None

            # Преобразуем в словарь
            info_dict = {}
            for attr in dir(info):
                if not attr.startswith('_'):
                    try:
                        value = getattr(info, attr)
                        # Пропускаем методы
                        if not callable(value):
                            info_dict[attr] = value
                    except:
                        continue

            return info_dict

        except Exception as e:
            logger.error(f"Ошибка при получении информации о символе {symbol}: {e}")
            return None

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        Получение информации о счете

        Returns:
            Optional[Dict]: Словарь с информацией о счете или None
        """
        if not self.connected:
            if not self.connect():
                return None

        try:
            account_info = mt5.account_info()

            if account_info is None:
                logger.warning("Информация о счете не найдена")
                return None

            # Преобразуем в словарь
            account_dict = {}
            for attr in dir(account_info):
                if not attr.startswith('_'):
                    try:
                        value = getattr(account_info, attr)
                        if not callable(value):
                            account_dict[attr] = value
                    except:
                        continue

            return account_dict

        except Exception as e:
            logger.error(f"Ошибка при получении информации о счете: {e}")
            return None

    def is_connected(self) -> bool:
        """
        Проверка подключения к MT5

        Returns:
            bool: True если подключено, False в противном случае
        """
        return self.connected

    def wait_for_connection(self, timeout_seconds: int = 30) -> bool:
        """
        Ожидание подключения к MT5

        Args:
            timeout_seconds (int): Таймаут в секундах

        Returns:
            bool: True если подключение установлено, False в противном случае
        """
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            if self.connect():
                return True
            time.sleep(1)

        logger.error(f"Таймаут подключения к MT5 ({timeout_seconds} секунд)")
        return False


# Создаем глобальный экземпляр для удобства
mt5_connector = MT5Connector()

