import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging
from .. import config

logger = logging.getLogger(__name__)

# Сопоставление таймфреймов
TIMEFRAME_MT5 = {
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

def initialize_mt5() -> bool:
    """Инициализация подключения к MT5"""
    try:
        if not mt5.initialize(
            path=config.MT5_PATH,
            login=config.MT5_LOGIN,
            password=config.MT5_PASSWORD,
            server=config.MT5_SERVER,
            timeout=config.MT5_TIMEOUT,
            portable=False
        ):
            logger.error(f"Ошибка инициализации MT5: {mt5.last_error()}")
            return False
        
        logger.info(f"Успешно подключены к MT5")
        logger.info(f"  Сервер: {mt5.terminal_info().server}")
        logger.info(f"  Счет: {mt5.account_info().login}")
        logger.info(f"  Баланс: {mt5.account_info().balance}")
        logger.info(f"  Валюта: {mt5.account_info().currency}")
        
        # Проверяем префикс инструментов
        symbol_info = mt5.symbol_info(config.SYMBOL)
        if symbol_info:
            logger.info(f"  Префикс инструментов: {config.SYMBOL_PREFIX}.")
        else:
            logger.warning(f"Символ {config.SYMBOL} не найден в Market Watch")
            
        return True
    except Exception as e:
        logger.error(f"Исключение при инициализации MT5: {e}")
        return False

def shutdown_mt5():
    """Закрытие соединения с MT5"""
    mt5.shutdown()
    logger.info("MT5 соединение закрыто")

def get_symbol_data(symbol: str, timeframe: str, count: int) -> pd.DataFrame:
    """Получение исторических данных для символа"""
    try:
        # Критическое исправление: правильная обработка префиксов
        # Префикс в MT5 обычно ставится ПОСЛЕ названия символа (например, EURUSD.pro)
        # Но в конфигурации указан префикс RFD. который ставится ПЕРЕД
        
        # Сохраняем оригинальный символ для логирования
        original_symbol = symbol
        
        # Если в конфиге есть префикс, пробуем разные варианты
        if config.SYMBOL_PREFIX:
            # Вариант 1: символ с префиксом в начале (как в конфиге)
            prefixed_symbol = f"{config.SYMBOL_PREFIX}{symbol.replace(config.SYMBOL_PREFIX, '')}"
            # Вариант 2: символ без префикса
            clean_symbol = symbol.replace(config.SYMBOL_PREFIX, '')
            
            # Пробуем сначала с префиксом
            selected_symbol = prefixed_symbol
            logger.info(f"Пытаемся получить данные для символа: {selected_symbol} (исходный: {original_symbol})")
            
            # Выбираем символ на рынке
            if not mt5.symbol_select(selected_symbol, True):
                logger.warning(f"Символ {selected_symbol} не найден, пробуем {clean_symbol}")
                selected_symbol = clean_symbol
                if not mt5.symbol_select(selected_symbol, True):
                    logger.error(f"Не удалось найти символ {selected_symbol}")
                    return pd.DataFrame()
        else:
            selected_symbol = symbol
        
        # Получаем данные
        timeframe_mt5 = TIMEFRAME_MT5.get(timeframe, mt5.TIMEFRAME_H1)
        rates = mt5.copy_rates_from_pos(selected_symbol, timeframe_mt5, 0, count)
        
        if rates is None or len(rates) == 0:
            logger.error(f"Не удалось получить данные для {selected_symbol}")
            # Пробуем получить данные другим способом
            current_time = datetime.now()
            from_time = current_time - timedelta(days=30)  # 30 дней назад
            rates = mt5.copy_rates_range(selected_symbol, timeframe_mt5, from_time, current_time)
            
            if rates is None:
                logger.error(f"Не удалось получить данные диапазоном для {selected_symbol}")
                return pd.DataFrame()
        
        # Конвертируем в DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Добавляем имя символа
        df['symbol'] = original_symbol
        
        logger.info(f"Успешно получено {len(df)} баров для {original_symbol} (использован {selected_symbol})")
        return df
        
    except Exception as e:
        logger.error(f"Ошибка при получении данных для {symbol}: {e}")
        return pd.DataFrame()

def get_current_price(symbol: str) -> Optional[Dict]:
    """Получение текущей цены символа"""
    try:
        # Аналогичная логика с префиксами
        if config.SYMBOL_PREFIX:
            prefixed_symbol = f"{config.SYMBOL_PREFIX}{symbol.replace(config.SYMBOL_PREFIX, '')}"
            clean_symbol = symbol.replace(config.SYMBOL_PREFIX, '')
            
            selected_symbol = prefixed_symbol
            tick = mt5.symbol_info_tick(selected_symbol)
            
            if tick is None:
                selected_symbol = clean_symbol
                tick = mt5.symbol_info_tick(selected_symbol)
        else:
            tick = mt5.symbol_info_tick(symbol)
        
        if tick:
            return {
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'time': pd.to_datetime(tick.time, unit='s')
            }
        return None
    except Exception as e:
        logger.error(f"Ошибка получения текущей цены для {symbol}: {e}")
        return None

def get_account_info() -> Dict:
    """Получение информации о счете"""
    try:
        account = mt5.account_info()
        if account:
            return {
                'login': account.login,
                'balance': account.balance,
                'equity': account.equity,
                'margin': account.margin,
                'free_margin': account.margin_free,
                'margin_level': account.margin_level,
                'currency': account.currency,
                'leverage': account.leverage,
                'name': account.name,
                'server': account.server
            }
        return {}
    except Exception as e:
        logger.error(f"Ошибка получения информации о счете: {e}")
        return {}

def place_order(symbol: str, order_type: str, volume: float, 
                stop_loss: float = 0.0, take_profit: float = 0.0,
                comment: str = "") -> Optional[int]:
    """Размещение ордера"""
    try:
        # Определяем тип ордера
        if order_type.lower() == 'buy':
            order_type_mt5 = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
        elif order_type.lower() == 'sell':
            order_type_mt5 = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        else:
            logger.error(f"Неизвестный тип ордера: {order_type}")
            return None
        
        # Подготавливаем запрос
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type_mt5,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 10,
            "magic": config.MAGIC_NUMBER,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Отправляем ордер
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Ошибка размещения ордера: {result.retcode} - {result.comment}")
            return None
        
        logger.info(f"Ордер размещен успешно: #{result.order}")
        return result.order
        
    except Exception as e:
        logger.error(f"Ошибка при размещении ордера: {e}")
        return None

def close_order(order_id: int) -> bool:
    """Закрытие ордера"""
    try:
        order_info = mt5.orders_get(ticket=order_id)
        if not order_info:
            logger.error(f"Ордер #{order_id} не найден")
            return False
        
        order = order_info[0]
        symbol = order.symbol
        volume = order.volume_current
        order_type = order.type
        
        # Определяем противоположный тип для закрытия
        if order_type == mt5.ORDER_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
        
        # Подготавливаем запрос на закрытие
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": close_type,
            "position": order_id,
            "price": price,
            "deviation": 10,
            "magic": config.MAGIC_NUMBER,
            "comment": "Closed by Pattern Recognition Engine",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Отправляем запрос
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Ошибка закрытия ордера: {result.retcode} - {result.comment}")
            return False
        
        logger.info(f"Ордер #{order_id} закрыт успешно")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при закрытии ордера: {e}")
        return False

def get_open_orders() -> List[Dict]:
    """Получение списка открытых ордеров"""
    try:
        orders = mt5.orders_get()
        if orders is None:
            return []
        
        orders_list = []
        for order in orders:
            orders_list.append({
                'ticket': order.ticket,
                'symbol': order.symbol,
                'type': 'buy' if order.type == mt5.ORDER_TYPE_BUY else 'sell',
                'volume': order.volume_current,
                'open_price': order.price_open,
                'current_price': order.price_current,
                'profit': order.profit,
                'comment': order.comment,
                'open_time': pd.to_datetime(order.time_setup, unit='s')
            })
        
        return orders_list
    except Exception as e:
        logger.error(f"Ошибка получения открытых ордеров: {e}")
        return []

def get_order_history(days: int = 7) -> List[Dict]:
    """Получение истории ордеров"""
    try:
        from_time = datetime.now() - timedelta(days=days)
        to_time = datetime.now()
        
        history = mt5.history_deals_get(from_time, to_time)
        if history is None:
            return []
        
        history_list = []
        for deal in history:
            history_list.append({
                'ticket': deal.ticket,
                'order': deal.order,
                'symbol': deal.symbol,
                'type': 'buy' if deal.type == 0 else 'sell',
                'volume': deal.volume,
                'price': deal.price,
                'profit': deal.profit,
                'commission': deal.commission,
                'swap': deal.swap,
                'time': pd.to_datetime(deal.time, unit='s')
            })
        
        return history_list
    except Exception as e:
        logger.error(f"Ошибка получения истории ордеров: {e}")
        return []

