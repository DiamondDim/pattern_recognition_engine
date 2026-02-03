# utils/mt5_connector.py

"""
Модуль подключения к MetaTrader 5
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json

from config import config
from utils.logger import logger

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 не установлен. MT5 функциональность недоступна.")


class MT5Connector:
    """Класс для работы с MetaTrader 5"""

    def __init__(self):
        self.logger = logger.bind(module="mt5_connector")
        self.connected = False

        if not MT5_AVAILABLE:
            self.logger.error("MetaTrader5 библиотека не установлена")

    async def connect(self) -> bool:
        """Подключение к MetaTrader 5"""
        if not MT5_AVAILABLE:
            self.logger.error("MetaTrader5 библиотека недоступна")
            return False

        try:
            # Инициализация MT5
            if not mt5.initialize(
                path=config.MT5.PATH,
                login=config.MT5.LOGIN,
                password=config.MT5.PASSWORD,
                server=config.MT5.SERVER,
                timeout=config.MT5.TIMEOUT
            ):
                self.logger.error(f"Ошибка инициализации MT5: {mt5.last_error()}")
                return False

            self.connected = True
            self.logger.info("Успешное подключение к MetaTrader 5")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка подключения к MT5: {e}")
            return False

    async def disconnect(self):
        """Отключение от MetaTrader 5"""
        if MT5_AVAILABLE and self.connected:
            mt5.shutdown()
            self.connected = False
            self.logger.info("Отключено от MetaTrader 5")

    async def get_historical_data(self,
                                 symbol: str,
                                 timeframe: str,
                                 bars: int = 1000,
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Получение исторических данных

        Args:
            symbol: Торговый символ
            timeframe: Таймфрейм
            bars: Количество баров
            start_date: Начальная дата
            end_date: Конечная дата

        Returns:
            DataFrame с историческими данными или None
        """
        if not self.connected:
            if not await self.connect():
                return None

        try:
            # Конвертация таймфрейма
            mt5_timeframe = self._convert_timeframe(timeframe)
            if mt5_timeframe is None:
                self.logger.error(f"Неподдерживаемый таймфрейм: {timeframe}")
                return None

            # Получение данных
            if start_date and end_date:
                rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
            else:
                rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)

            if rates is None:
                self.logger.error(f"Ошибка получения данных для {symbol}: {mt5.last_error()}")
                return None

            # Конвертация в DataFrame
            df = pd.DataFrame(rates)

            # Конвертация времени
            df['time'] = pd.to_datetime(df['time'], unit='s')

            self.logger.info(f"Получено данных для {symbol}: {len(df)} баров")
            return df

        except Exception as e:
            self.logger.error(f"Ошибка получения данных из MT5: {e}")
            return None

    async def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Получение текущей цены символа

        Args:
            symbol: Торговый символ

        Returns:
            Словарь с текущими ценами или None
        """
        if not self.connected:
            if not await self.connect():
                return None

        try:
            tick = mt5.symbol_info_tick(symbol)

            if tick is None:
                self.logger.error(f"Ошибка получения тика для {symbol}: {mt5.last_error()}")
                return None

            return {
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'time': pd.to_datetime(tick.time, unit='s').isoformat()
            }

        except Exception as e:
            self.logger.error(f"Ошибка получения цены для {symbol}: {e}")
            return None

    async def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Получение информации об аккаунте"""
        if not self.connected:
            if not await self.connect():
                return None

        try:
            account_info = mt5.account_info()

            if account_info is None:
                self.logger.error(f"Ошибка получения информации об аккаунте: {mt5.last_error()}")
                return None

            return {
                'login': account_info.login,
                'name': account_info.name,
                'server': account_info.server,
                'currency': account_info.currency,
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'free_margin': account_info.margin_free,
                'leverage': account_info.leverage,
                'profit': account_info.profit
            }

        except Exception as e:
            self.logger.error(f"Ошибка получения информации об аккаунте: {e}")
            return None

    async def place_order(self,
                         symbol: str,
                         order_type: str,  # 'buy' или 'sell'
                         volume: float,
                         price: Optional[float] = None,
                         sl: Optional[float] = None,
                         tp: Optional[float] = None,
                         comment: str = "PRE Order") -> Optional[Dict[str, Any]]:
        """
        Размещение ордера

        Args:
            symbol: Торговый символ
            order_type: Тип ордера
            volume: Объем
            price: Цена (None для рыночного ордера)
            sl: Стоп-лосс
            tp: Тейк-профит
            comment: Комментарий

        Returns:
            Информация об ордере или None
        """
        if not self.connected:
            if not await self.connect():
                return None

        try:
            # Получаем текущую цену если не указана
            if price is None:
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    self.logger.error(f"Не удалось получить цену для {symbol}")
                    return None

                if order_type.lower() == 'buy':
                    price = tick.ask
                else:
                    price = tick.bid

            # Определяем тип ордера
            if order_type.lower() == 'buy':
                order_type_mt5 = mt5.ORDER_TYPE_BUY
            elif order_type.lower() == 'sell':
                order_type_mt5 = mt5.ORDER_TYPE_SELL
            else:
                self.logger.error(f"Неизвестный тип ордера: {order_type}")
                return None

            # Подготавливаем запрос
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type_mt5,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 10,
                "magic": 234000,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # Отправляем ордер
            result = mt5.order_send(request)

            if result is None:
                self.logger.error(f"Ошибка отправки ордера: {mt5.last_error()}")
                return None

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Ошибка ордера: {result.retcode} - {result.comment}")
                return None

            self.logger.info(f"Ордер размещен: {symbol} {order_type} {volume} @ {price}")

            return {
                'order_id': result.order,
                'deal_id': result.deal,
                'symbol': result.symbol,
                'volume': result.volume,
                'price': result.price,
                'sl': result.sl,
                'tp': result.tp,
                'profit': result.profit,
                'comment': result.comment
            }

        except Exception as e:
            self.logger.error(f"Ошибка размещения ордера: {e}")
            return None

    async def get_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Получение списка открытых ордеров"""
        if not self.connected:
            if not await self.connect():
                return []

        try:
            orders = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()

            if orders is None:
                return []

            order_list = []
            for order in orders:
                order_list.append({
                    'ticket': order.ticket,
                    'symbol': order.symbol,
                    'type': 'buy' if order.type == 0 else 'sell',
                    'volume': order.volume,
                    'open_price': order.price_open,
                    'current_price': order.price_current,
                    'sl': order.sl,
                    'tp': order.tp,
                    'profit': order.profit,
                    'open_time': pd.to_datetime(order.time, unit='s').isoformat()
                })

            return order_list

        except Exception as e:
            self.logger.error(f"Ошибка получения ордеров: {e}")
            return []

    async def close_order(self, ticket: int, volume: Optional[float] = None) -> bool:
        """
        Закрытие ордера

        Args:
            ticket: Номер тикета ордера
            volume: Объем для закрытия (None для полного закрытия)

        Returns:
            Успешность закрытия
        """
        if not self.connected:
            if not await self.connect():
                return False

        try:
            # Получаем информацию об ордере
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                self.logger.error(f"Ордер {ticket} не найден")
                return False

            position = positions[0]

            # Определяем тип закрытия
            if position.type == mt5.POSITION_TYPE_BUY:
                close_type = mt5.ORDER_TYPE_SELL
            else:
                close_type = mt5.ORDER_TYPE_BUY

            # Получаем текущую цену
            tick = mt5.symbol_info_tick(position.symbol)
            if tick is None:
                self.logger.error(f"Не удалось получить цену для {position.symbol}")
                return False

            close_price = tick.bid if position.type == mt5.POSITION_TYPE_BUY else tick.ask

            # Объем для закрытия
            close_volume = volume if volume is not None else position.volume

            # Подготавливаем запрос
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": close_volume,
                "type": close_type,
                "position": position.ticket,
                "price": close_price,
                "deviation": 10,
                "magic": 234000,
                "comment": "PRE Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # Отправляем запрос на закрытие
            result = mt5.order_send(request)

            if result is None:
                self.logger.error(f"Ошибка закрытия ордера: {mt5.last_error()}")
                return False

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Ошибка закрытия ордера: {result.retcode} - {result.comment}")
                return False

            self.logger.info(f"Ордер {ticket} закрыт: {close_volume} @ {close_price}")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка закрытия ордера: {e}")
            return False

    async def export_data_to_file(self,
                                 symbol: str,
                                 timeframe: str,
                                 bars: int = 1000,
                                 filepath: Optional[str] = None) -> bool:
        """
        Экспорт данных в файл

        Args:
            symbol: Торговый символ
            timeframe: Таймфрейм
            bars: Количество баров
            filepath: Путь к файлу

        Returns:
            Успешность экспорта
        """
        if filepath is None:
            filepath = config.MT5.INPUT_FILE_PATH

        try:
            # Получаем данные
            df = await self.get_historical_data(symbol, timeframe, bars)

            if df is None or df.empty:
                self.logger.error("Не удалось получить данные для экспорта")
                return False

            # Сохраняем в файл
            df.to_csv(filepath, index=False)

            self.logger.info(f"Данные экспортированы в {filepath}: {len(df)} баров")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка экспорта данных: {e}")
            return False

    def _convert_timeframe(self, timeframe: str) -> Optional[int]:
        """Конвертация строкового таймфрейма в MT5 константу"""
        if not MT5_AVAILABLE:
            return None

        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M2': mt5.TIMEFRAME_M2,
            'M3': mt5.TIMEFRAME_M3,
            'M4': mt5.TIMEFRAME_M4,
            'M5': mt5.TIMEFRAME_M5,
            'M6': mt5.TIMEFRAME_M6,
            'M10': mt5.TIMEFRAME_M10,
            'M12': mt5.TIMEFRAME_M12,
            'M15': mt5.TIMEFRAME_M15,
            'M20': mt5.TIMEFRAME_M20,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H2': mt5.TIMEFRAME_H2,
            'H3': mt5.TIMEFRAME_H3,
            'H4': mt5.TIMEFRAME_H4,
            'H6': mt5.TIMEFRAME_H6,
            'H8': mt5.TIMEFRAME_H8,
            'H12': mt5.TIMEFRAME_H12,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }

        return timeframe_map.get(timeframe.upper())

