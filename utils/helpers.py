import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
import hashlib
import json
import random
import string

logger = logging.getLogger(__name__)

class Helpers:
    """Вспомогательные функции для торговой системы"""
    
    @staticmethod
    def calculate_position_size(balance: float, risk_per_trade: float, 
                               entry_price: float, stop_loss: float,
                               risk_reward_ratio: float = 2.0) -> Dict[str, Any]:
        """Расчет размера позиции на основе риска"""
        if entry_price <= 0 or stop_loss <= 0:
            return {
                'position_size': 0,
                'risk_amount': 0,
                'reward_amount': 0,
                'stop_loss_pips': 0,
                'take_profit_price': entry_price,
                'risk_per_share': 0
            }
        
        # Расчет риска на одну акцию/лот
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return {
                'position_size': 0,
                'risk_amount': 0,
                'reward_amount': 0,
                'stop_loss_pips': 0,
                'take_profit_price': entry_price,
                'risk_per_share': 0
            }
        
        # Сумма риска на сделку
        risk_amount = balance * risk_per_trade
        
        # Размер позиции
        position_size = risk_amount / risk_per_share
        
        # Расчет тейк-профита
        if entry_price > stop_loss:  # Long позиция
            take_profit_price = entry_price + (risk_per_share * risk_reward_ratio)
        else:  # Short позиция
            take_profit_price = entry_price - (risk_per_share * risk_reward_ratio)
        
        # Расчет потенциальной прибыли
        reward_amount = abs(take_profit_price - entry_price) * position_size
        
        # Расчет стоп-лосса в пипсах (для Forex)
        stop_loss_pips = abs(entry_price - stop_loss) * 10000  # Для пар с 4 знаками после запятой
        
        return {
            'position_size': position_size,
            'risk_amount': risk_amount,
            'reward_amount': reward_amount,
            'stop_loss_pips': stop_loss_pips,
            'take_profit_price': take_profit_price,
            'risk_per_share': risk_per_share,
            'risk_reward_ratio': risk_reward_ratio
        }
    
    @staticmethod
    def calculate_pivot_points(high: float, low: float, close: float, 
                              method: str = 'standard') -> Dict[str, float]:
        """Расчет уровней Pivot Points"""
        if method == 'standard':
            # Стандартный метод
            pivot = (high + low + close) / 3
            
            r1 = (2 * pivot) - low
            s1 = (2 * pivot) - high
            
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            
            r3 = high + 2 * (pivot - low)
            s3 = low - 2 * (high - pivot)
            
        elif method == 'fibonacci':
            # Метод Фибоначчи
            pivot = (high + low + close) / 3
            
            diff = high - low
            
            r1 = pivot + 0.382 * diff
            r2 = pivot + 0.618 * diff
            r3 = pivot + 1.0 * diff
            
            s1 = pivot - 0.382 * diff
            s2 = pivot - 0.618 * diff
            s3 = pivot - 1.0 * diff
        
        elif method == 'woodie':
            # Метод Вуди
            pivot = (high + low + 2 * close) / 4
            
            r1 = (2 * pivot) - low
            s1 = (2 * pivot) - high
            
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            
            r3 = high + 2 * (pivot - low)
            s3 = low - 2 * (high - pivot)
        
        else:  # camarilla
            # Метод Камарилья
            pivot = (high + low + close) / 3
            
            r1 = close + (high - low) * 1.1 / 12
            s1 = close - (high - low) * 1.1 / 12
            
            r2 = close + (high - low) * 1.1 / 6
            s2 = close - (high - low) * 1.1 / 6
            
            r3 = close + (high - low) * 1.1 / 4
            s3 = close - (high - low) * 1.1 / 4
            
            r4 = close + (high - low) * 1.1 / 2
            s4 = close - (high - low) * 1.1 / 2
        
        result = {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }
        
        if method == 'camarilla':
            result.update({'r4': r4, 's4': s4})
        
        return result
    
    @staticmethod
    def detect_support_resistance(prices: pd.Series, window: int = 20, 
                                 threshold: float = 0.02, 
                                 method: str = 'local_extrema') -> Dict[str, List[float]]:
        """Детекция уровней поддержки и сопротивления"""
        if len(prices) < window * 2:
            return {'support': [], 'resistance': []}
        
        support_levels = []
        resistance_levels = []
        
        if method == 'local_extrema':
            # Метод локальных экстремумов
            for i in range(window, len(prices) - window):
                window_prices = prices.iloc[i-window:i+window]
                
                # Проверка на локальный минимум (поддержка)
                if prices.iloc[i] == window_prices.min():
                    support_levels.append(prices.iloc[i])
                
                # Проверка на локальный максимум (сопротивление)
                if prices.iloc[i] == window_prices.max():
                    resistance_levels.append(prices.iloc[i])
        
        elif method == 'pivot_points':
            # Метод точек разворота
            for i in range(window, len(prices) - window, window):
                window_high = prices.iloc[i-window:i+window].max()
                window_low = prices.iloc[i-window:i+window].min()
                window_close = prices.iloc[i+window-1]
                
                pivot = (window_high + window_low + window_close) / 3
                
                # Уровни поддержки и сопротивления на основе Pivot Points
                r1 = (2 * pivot) - window_low
                s1 = (2 * pivot) - window_high
                
                resistance_levels.extend([pivot, r1])
                support_levels.extend([pivot, s1])
        
        # Кластеризация близких уровней
        clustered_support = Helpers._cluster_levels(support_levels, threshold)
        clustered_resistance = Helpers._cluster_levels(resistance_levels, threshold)
        
        # Сортировка
        clustered_support.sort()
        clustered_resistance.sort()
        
        return {
            'support': clustered_support,
            'resistance': clustered_resistance,
            'method': method,
            'window': window,
            'threshold': threshold
        }
    
    @staticmethod
    def _cluster_levels(levels: List[float], threshold: float) -> List[float]:
        """Кластеризация близких ценовых уровней"""
        if not levels:
            return []
        
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]
        
        for price in levels[1:]:
            # Проверяем, близок ли текущий уровень к последнему в кластере
            last_price = current_cluster[-1]
            
            if abs(price - last_price) / last_price <= threshold:
                current_cluster.append(price)
            else:
                # Добавляем среднее значение кластера
                clusters.append(np.mean(current_cluster))
                current_cluster = [price]
        
        if current_cluster:
            clusters.append(np.mean(current_cluster))
        
        return clusters
    
    @staticmethod
    def calculate_fibonacci_levels(high: float, low: float, 
                                  is_uptrend: bool = True) -> Dict[str, float]:
        """Расчет уровней Фибоначчи"""
        if high <= low:
            return {}
        
        diff = high - low
        
        levels = {
            '0.0': low,
            '0.236': low + diff * 0.236,
            '0.382': low + diff * 0.382,
            '0.5': low + diff * 0.5,
            '0.618': low + diff * 0.618,
            '0.786': low + diff * 0.786,
            '1.0': high,
            '1.272': high + diff * 0.272 if is_uptrend else low - diff * 0.272,
            '1.618': high + diff * 0.618 if is_uptrend else low - diff * 0.618,
            '2.618': high + diff * 1.618 if is_uptrend else low - diff * 1.618,
            '4.236': high + diff * 3.236 if is_uptrend else low - diff * 3.236
        }
        
        return levels
    
    @staticmethod
    def generate_trade_id(symbol: str, timestamp: datetime, 
                         signal: str, length: int = 8) -> str:
        """Генерация уникального ID для сделки"""
        # Создаем базовую строку
        base_string = f"{symbol}_{timestamp.strftime('%Y%m%d%H%M%S')}_{signal}"
        
        # Добавляем случайность
        random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
        unique_string = f"{base_string}_{random_str}"
        
        # Создаем хэш
        hash_object = hashlib.md5(unique_string.encode())
        return hash_object.hexdigest()[:length].upper()
    
    @staticmethod
    def validate_symbol(symbol: str) -> Tuple[bool, str]:
        """Валидация имени символа"""
        if not symbol or not isinstance(symbol, str):
            return False, "Символ не может быть пустым"
        
        # Очищаем символ от возможных префиксов/суффиксов
        clean_symbol = symbol.upper().strip()
        
        # Убираем возможные префиксы брокера
        prefixes_to_remove = ['RFDRU.', 'RFD.', '.PRO', '.MICEX', '.FX']
        for prefix in prefixes_to_remove:
            if clean_symbol.startswith(prefix):
                clean_symbol = clean_symbol[len(prefix):]
            elif clean_symbol.endswith(prefix):
                clean_symbol = clean_symbol[:-len(prefix)]
        
        # Проверяем формат (например, EURUSD, GBPUSD, USDJPY и т.д.)
        if len(clean_symbol) < 6 or len(clean_symbol) > 12:
            return False, f"Некорректная длина символа: {len(clean_symbol)}"
        
        # Проверяем, что содержит только буквы
        if not clean_symbol.isalpha():
            return False, "Символ должен содержать только буквы"
        
        # Проверяем типичные пары Forex
        common_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 
                       'AUDUSD', 'USDCAD', 'NZDUSD', 'EURGBP',
                       'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY']
        
        if clean_symbol in common_pairs:
            return True, f"Валидный Forex символ: {clean_symbol}"
        else:
            # Проверяем формат XXXYYY (6 символов)
            if len(clean_symbol) == 6:
                base = clean_symbol[:3]
                quote = clean_symbol[3:]
                
                if base.isalpha() and quote.isalpha():
                    return True, f"Валидный формат символа: {base}/{quote}"
                else:
                    return False, "Некорректный формат символа"
            else:
                return True, f"Символ принят: {clean_symbol}"
    
    @staticmethod
    def calculate_timeframe_seconds(timeframe: str) -> int:
        """Конвертация таймфрейма в секунды"""
        timeframe_map = {
            'M1': 60,
            'M2': 120,
            'M3': 180,
            'M5': 300,
            'M10': 600,
            'M15': 900,
            'M30': 1800,
            'H1': 3600,
            'H2': 7200,
            'H3': 10800,
            'H4': 14400,
            'H6': 21600,
            'H8': 28800,
            'H12': 43200,
            'D1': 86400,
            'W1': 604800,
            'MN1': 2592000
        }
        
        timeframe_upper = timeframe.upper()
        return timeframe_map.get(timeframe_upper, 3600)  # По умолчанию H1
    
    @staticmethod
    def merge_signals(signals: List[Dict], weights: Optional[List[float]] = None, 
                     method: str = 'weighted_vote') -> Dict[str, Any]:
        """Объединение нескольких сигналов"""
        if not signals:
            return {
                'signal': 'hold',
                'confidence': 0.0,
                'signal_count': 0,
                'method': method,
                'details': []
            }
        
        if weights is None:
            weights = [1.0] * len(signals)
        
        # Нормализация весов
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0] * len(signals)
            total_weight = len(signals)
        
        normalized_weights = [w / total_weight for w in weights]
        
        if method == 'weighted_vote':
            # Взвешенное голосование
            vote_buy = 0
            vote_sell = 0
            total_confidence = 0
            
            for signal, weight in zip(signals, normalized_weights):
                signal_type = signal.get('signal', 'hold')
                confidence = signal.get('confidence', 0.5)
                
                if signal_type == 'buy':
                    vote_buy += weight * confidence
                elif signal_type == 'sell':
                    vote_sell += weight * confidence
                
                total_confidence += confidence * weight
            
            # Определение результирующего сигнала
            if vote_buy > vote_sell and vote_buy > 0.6:
                result_signal = 'buy'
                result_confidence = vote_buy
            elif vote_sell > vote_buy and vote_sell > 0.6:
                result_signal = 'sell'
                result_confidence = vote_sell
            else:
                result_signal = 'hold'
                result_confidence = max(vote_buy, vote_sell)
        
        elif method == 'majority':
            # Простое большинство
            buy_count = sum(1 for s in signals if s.get('signal') == 'buy')
            sell_count = sum(1 for s in signals if s.get('signal') == 'sell')
            
            if buy_count > sell_count:
                result_signal = 'buy'
                result_confidence = buy_count / len(signals)
            elif sell_count > buy_count:
                result_signal = 'sell'
                result_confidence = sell_count / len(signals)
            else:
                result_signal = 'hold'
                result_confidence = 0.5
        
        else:  # highest_confidence
            # Сигнал с наибольшей уверенностью
            valid_signals = [s for s in signals if s.get('signal') in ['buy', 'sell']]
            
            if valid_signals:
                best_signal = max(valid_signals, key=lambda x: x.get('confidence', 0))
                result_signal = best_signal.get('signal', 'hold')
                result_confidence = best_signal.get('confidence', 0.5)
            else:
                result_signal = 'hold'
                result_confidence = 0.5
        
        return {
            'signal': result_signal,
            'confidence': result_confidence,
            'signal_count': len(signals),
            'buy_votes': vote_buy if 'vote_buy' in locals() else 0,
            'sell_votes': vote_sell if 'vote_sell' in locals() else 0,
            'method': method,
            'details': signals
        }
    
    @staticmethod
    def calculate_optimal_risk(balance: float, win_rate: float, 
                             avg_win: float, avg_loss: float,
                             max_risk_per_trade: float = 0.02,
                             use_kelly: bool = True) -> float:
        """Расчет оптимального размера риска"""
        if avg_loss == 0 or balance <= 0:
            return max_risk_per_trade / 2  # Консервативное значение по умолчанию
        
        if use_kelly:
            # Упрощенная формула Келли: f* = (p * b - q) / b
            # где p - вероятность выигрыша, q = 1-p, b = средний выигрыш / средний проигрыш
            p = win_rate / 100
            q = 1 - p
            b = avg_win / abs(avg_loss)
            
            if b > 0:
                kelly = (p * b - q) / b
            else:
                kelly = 0
        else:
            # Консервативный подход: половина от обратного к максимальной просадке
            kelly = 0.01  # 1% по умолчанию
        
        # Фракционный Келли (половина) для уменьшения риска
        fractional_kelly = max(0, kelly * 0.5)
        
        # Ограничиваем максимальный риск
        optimal_risk = min(fractional_kelly, max_risk_per_trade)
        
        # Минимальный риск 0.5%
        optimal_risk = max(optimal_risk, 0.005)
        
        # Округляем до двух знаков
        optimal_risk = round(optimal_risk, 4)
        
        return optimal_risk
    
    @staticmethod
    def format_currency(amount: float, currency: str = 'USD') -> str:
        """Форматирование денежной суммы"""
        if currency == 'USD':
            return f"${amount:,.2f}"
        elif currency == 'EUR':
            return f"€{amount:,.2f}"
        elif currency == 'GBP':
            return f"£{amount:,.2f}"
        elif currency == 'JPY':
            return f"¥{amount:,.0f}"
        elif currency == 'RUB':
            return f"{amount:,.0f} ₽"
        elif currency == 'CHF':
            return f"CHF {amount:,.2f}"
        elif currency == 'AUD':
            return f"A${amount:,.2f}"
        elif currency == 'CAD':
            return f"C${amount:,.2f}"
        elif currency == 'CNY':
            return f"¥{amount:,.2f}"
        else:
            return f"{amount:,.2f} {currency}"
    
    @staticmethod
    def calculate_time_until_close(market: str = 'forex') -> Dict[str, Any]:
        """Расчет времени до закрытия рынка"""
        now = datetime.now()
        
        if market == 'forex':
            # Forex закрывается в пятницу 22:00 GMT и открывается в воскресенье 22:00 GMT
            # Определяем день недели (0 - понедельник, 6 - воскресенье)
            weekday = now.weekday()
            hour = now.hour
            
            if weekday == 4:  # Пятница
                # Закрытие в 22:00 GMT
                close_time = now.replace(hour=22, minute=0, second=0, microsecond=0)
                if now < close_time:
                    time_left = close_time - now
                    status = 'open'
                    next_open = now.replace(hour=22, minute=0) + timedelta(days=2)  # Воскресенье 22:00
                else:
                    time_left = timedelta(days=2) - (now - close_time)
                    status = 'closed'
                    next_open = now.replace(hour=22, minute=0) + timedelta(days=2)
            elif weekday == 5:  # Суббота
                status = 'closed'
                next_open = now.replace(hour=22, minute=0) + timedelta(days=1)
                time_left = next_open - now
            elif weekday == 6:  # Воскресенье
                open_time = now.replace(hour=22, minute=0, second=0, microsecond=0)
                if now < open_time:
                    status = 'closed'
                    next_open = open_time
                    time_left = next_open - now
                else:
                    status = 'open'
                    # Следующее закрытие в пятницу
                    days_until_friday = (4 - weekday) % 7
                    if days_until_friday == 0 and now.hour >= 22:
                        days_until_friday = 7
                    close_time = now.replace(hour=22, minute=0) + timedelta(days=days_until_friday)
                    time_left = close_time - now
                    next_open = None
            else:  # Понедельник-Четверг
                status = 'open'
                # Следующее закрытие в пятницу
                days_until_friday = 4 - weekday
                close_time = now.replace(hour=22, minute=0) + timedelta(days=days_until_friday)
                time_left = close_time - now
                next_open = None
            
            return {
                'market': market,
                'status': status,
                'time_left': time_left,
                'time_left_hours': time_left.total_seconds() / 3600,
                'next_open': next_open,
                'current_time': now
            }
        
        else:
            # Для других рынков возвращаем общую информацию
            return {
                'market': market,
                'status': 'unknown',
                'time_left': timedelta(0),
                'time_left_hours': 0,
                'next_open': None,
                'current_time': now
            }
    
    @staticmethod
    def save_to_json(data: Any, filename: str, indent: int = 2):
        """Сохранение данных в JSON файл"""
        try:
            # Создаем директорию если не существует
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Функция для сериализации datetime
            def json_serializer(obj):
                if isinstance(obj, (datetime, pd.Timestamp)):
                    return obj.isoformat()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif pd.isna(obj):
                    return None
                raise TypeError(f"Type {type(obj)} not serializable")
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, default=json_serializer, ensure_ascii=False)
            
            logger.debug(f"Данные сохранены в {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка сохранения в JSON: {e}")
            return False
    
    @staticmethod
    def load_from_json(filename: str) -> Optional[Any]:
        """Загрузка данных из JSON файла"""
        try:
            if not os.path.exists(filename):
                logger.warning(f"Файл не существует: {filename}")
                return None
            
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Функция для десериализации datetime
            def json_deserializer(obj):
                if isinstance(obj, str):
                    # Пробуем распарсить как datetime
                    for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', 
                               '%Y-%m-%d', '%Y/%m/%d %H:%M:%S'):
                        try:
                            return datetime.strptime(obj, fmt)
                        except:
                            continue
                return obj
            
            # Рекурсивно применяем десериализатор
            def recursive_deserialize(obj):
                if isinstance(obj, dict):
                    return {k: recursive_deserialize(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [recursive_deserialize(v) for v in obj]
                else:
                    return json_deserializer(obj)
            
            data = recursive_deserialize(data)
            logger.debug(f"Данные загружены из {filename}")
            return data
            
        except Exception as e:
            logger.error(f"Ошибка загрузки из JSON: {e}")
            return None
    
    @staticmethod
    def calculate_market_volatility(prices: pd.Series, window: int = 20) -> pd.Series:
        """Расчет волатильности рынка"""
        if len(prices) < window:
            return pd.Series([np.nan] * len(prices), index=prices.index)
        
        returns = prices.pct_change().dropna()
        volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Годовая волатильность
        return volatility
    
    @staticmethod
    def detect_market_regime(prices: pd.Series, window: int = 50, 
                            threshold: float = 0.15) -> pd.Series:
        """Определение рыночного режима (тренд/флэт)"""
        if len(prices) < window:
            return pd.Series(['unknown'] * len(prices), index=prices.index)
        
        # Расчет ADX (упрощенный)
        high = prices.rolling(window=window).max()
        low = prices.rolling(window=window).min()
        
        # Истинный диапазон
        tr = pd.concat([
            high - low,
            abs(high - prices.shift()),
            abs(low - prices.shift())
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(window=window).mean()
        
        # Направленное движение
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=window).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=window).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)
        adx = dx.rolling(window=window).mean()
        
        # Определение режима
        regime = pd.Series('unknown', index=prices.index)
        regime[adx > threshold] = 'trending'
        regime[adx <= threshold] = 'ranging'
        
        return regime
    
    @staticmethod
    def create_unique_filename(base_name: str, extension: str = '.csv') -> str:
        """Создание уникального имени файла с временной меткой"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{base_name}_{timestamp}{extension}"

