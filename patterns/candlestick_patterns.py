"""
Candlestick pattern detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class CandlestickPatterns:
    """Candlestick pattern detector."""

    def __init__(self):
        """Initialize candlestick pattern detector."""
        logger.info("CandlestickPatterns initialized")

    def detect_doji(self, data: pd.DataFrame, threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Detect Doji patterns."""
        patterns = []
        for i in range(len(data)):
            if i < 1:
                continue

            open_price = data['open'].iloc[i]
            close_price = data['close'].iloc[i]
            high = data['high'].iloc[i]
            low = data['low'].iloc[i]

            # Doji: открытие и закрытие почти равны
            body_size = abs(close_price - open_price)
            range_size = high - low

            if range_size > 0 and body_size / range_size < threshold:
                pattern = {
                    'index': i,
                    'pattern_type': 'doji',
                    'confidence': 1 - (body_size / range_size),
                    'signal': 'neutral',
                    'description': f"Doji pattern at index {i}"
                }
                patterns.append(pattern)

        return patterns

    def detect_hammer(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect Hammer patterns."""
        patterns = []
        for i in range(len(data)):
            if i < 1:
                continue

            open_price = data['open'].iloc[i]
            close_price = data['close'].iloc[i]
            high = data['high'].iloc[i]
            low = data['low'].iloc[i]

            body_size = abs(close_price - open_price)
            upper_shadow = high - max(open_price, close_price)
            lower_shadow = min(open_price, close_price) - low

            # Hammer: маленькое тело, маленькая верхняя тень, длинная нижняя тень
            total_size = high - low
            if total_size > 0:
                if (body_size / total_size < 0.3 and
                        lower_shadow > 2 * body_size and
                        upper_shadow < body_size * 0.3):
                    is_bullish = close_price > open_price
                    pattern_type = 'hammer' if is_bullish else 'hanging_man'

                    pattern = {
                        'index': i,
                        'pattern_type': pattern_type,
                        'confidence': 0.7,
                        'signal': 'bullish' if is_bullish else 'bearish',
                        'description': f"{pattern_type} pattern at index {i}"
                    }
                    patterns.append(pattern)

        return patterns

    def detect_engulfing(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect Engulfing patterns."""
        patterns = []
        for i in range(1, len(data)):
            prev_open = data['open'].iloc[i - 1]
            prev_close = data['close'].iloc[i - 1]
            curr_open = data['open'].iloc[i]
            curr_close = data['close'].iloc[i]

            prev_body = abs(prev_close - prev_open)
            curr_body = abs(curr_close - curr_open)

            # Engulfing: текущая свеча полностью поглощает предыдущую
            if (curr_body > prev_body and
                    min(curr_open, curr_close) < min(prev_open, prev_close) and
                    max(curr_open, curr_close) > max(prev_open, prev_close)):
                is_bullish = curr_close > curr_open
                pattern_type = 'bullish_engulfing' if is_bullish else 'bearish_engulfing'

                pattern = {
                    'index': i,
                    'pattern_type': pattern_type,
                    'confidence': 0.8,
                    'signal': 'bullish' if is_bullish else 'bearish',
                    'description': f"{pattern_type} pattern at index {i}"
                }
                patterns.append(pattern)

        return patterns

    def detect_all_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect all candlestick patterns."""
        all_patterns = []

        all_patterns.extend(self.detect_doji(data))
        all_patterns.extend(self.detect_hammer(data))
        all_patterns.extend(self.detect_engulfing(data))

        logger.info(f"Detected {len(all_patterns)} candlestick patterns")
        return all_patterns


class DojiPattern(BasePattern):
    """
    Класс для обнаружения паттерна Doji (доджи)
    """

    def detect(self, threshold: float = 0.1, min_wick_ratio: float = 2.0) -> List[Dict[str, Any]]:
        """
        Обнаружение паттерна Doji

        Args:
            threshold (float): Максимальное соотношение тела к диапазону
            min_wick_ratio (float): Минимальное соотношение теней

        Returns:
            list: Список обнаруженных паттернов Doji
        """
        patterns = []

        if self.data is None or self.data.empty:
            logger.warning("Нет данных для обнаружения паттерна Doji")
            return patterns

        try:
            for i in range(1, len(self.data) - 1):
                current = self.data.iloc[i]

                # Проверяем, что есть необходимые колонки
                if not all(col in current for col in ['Open', 'High', 'Low', 'Close']):
                    continue

                # Рассчитываем размеры
                body_size = abs(current['Close'] - current['Open'])
                total_range = current['High'] - current['Low']

                # Избегаем деления на ноль
                if total_range == 0:
                    continue

                body_ratio = body_size / total_range

                # Проверяем условие Doji: маленькое тело
                if body_ratio < threshold:
                    # Рассчитываем тени
                    upper_wick = current['High'] - max(current['Open'], current['Close'])
                    lower_wick = min(current['Open'], current['Close']) - current['Low']

                    # Проверяем наличие обеих теней
                    if upper_wick > 0 and lower_wick > 0:
                        # Рассчитываем соотношение теней
                        wick_ratio = max(upper_wick, lower_wick) / min(upper_wick, lower_wick)

                        # Проверяем соотношение теней
                        if wick_ratio >= min_wick_ratio:
                            # Определяем тип Doji
                            if abs(upper_wick - lower_wick) < total_range * 0.1:
                                doji_type = "perfect_doji"
                            elif upper_wick > lower_wick * 2:
                                doji_type = "dragonfly_doji"
                            elif lower_wick > upper_wick * 2:
                                doji_type = "gravestone_doji"
                            else:
                                doji_type = "doji"

                            # Создаем описание паттерна
                            pattern = {
                                'index': i,
                                'pattern_type': doji_type,
                                'timestamp': self.data.index[i],
                                'open': current['Open'],
                                'high': current['High'],
                                'low': current['Low'],
                                'close': current['Close'],
                                'body_size': body_size,
                                'total_range': total_range,
                                'body_ratio': body_ratio,
                                'upper_wick': upper_wick,
                                'lower_wick': lower_wick,
                                'wick_ratio': wick_ratio,
                                'volume': current.get('Volume', 0),
                                'confidence': 1.0 - body_ratio,  # Чем меньше тело, тем выше уверенность
                                'trend_context': self._get_trend_context(i),
                                'signal': self._get_signal_type(i, doji_type)
                            }

                            patterns.append(pattern)

            logger.info(f"Обнаружено {len(patterns)} паттернов Doji")
            self.patterns = patterns
            return patterns

        except Exception as e:
            logger.error(f"Ошибка при обнаружении паттерна Doji: {e}")
            return []

    def _get_trend_context(self, index: int, lookback: int = 20) -> str:
        """
        Определение трендового контекста

        Args:
            index (int): Индекс текущей свечи
            lookback (int): Количество свечей для анализа

        Returns:
            str: Контекст тренда
        """
        if index < lookback:
            return "insufficient_data"

        try:
            # Получаем данные для анализа
            start_idx = max(0, index - lookback)
            data_slice = self.data.iloc[start_idx:index]

            if len(data_slice) < 5:
                return "insufficient_data"

            # Рассчитываем скользящие средние
            closes = data_slice['Close'].values
            sma_short = closes[-5:].mean() if len(closes) >= 5 else closes.mean()
            sma_long = closes.mean()

            # Определяем тренд
            price_change = (closes[-1] - closes[0]) / closes[0] * 100

            if price_change > 2:
                return "strong_uptrend"
            elif price_change > 0.5:
                return "uptrend"
            elif price_change < -2:
                return "strong_downtrend"
            elif price_change < -0.5:
                return "downtrend"
            else:
                return "sideways"

        except Exception as e:
            logger.error(f"Ошибка при определении трендового контекста: {e}")
            return "unknown"

    def _get_signal_type(self, index: int, doji_type: str) -> str:
        """
        Определение типа сигнала на основе Doji

        Args:
            index (int): Индекс свечи
            doji_type (str): Тип Doji

        Returns:
            str: Тип сигнала (bullish/bearish/neutral)
        """
        if index < 2 or index >= len(self.data) - 1:
            return "neutral"

        try:
            prev_candle = self.data.iloc[index - 1]
            current = self.data.iloc[index]

            # Определяем направление предыдущей свечи
            prev_direction = "bullish" if prev_candle['Close'] > prev_candle['Open'] else "bearish"

            # Сигналы для разных типов Doji
            if doji_type == "dragonfly_doji":
                # Dragonfly Doji обычно бычий сигнал
                return "bullish"
            elif doji_type == "gravestone_doji":
                # Gravestone Doji обычно медвежий сигнал
                return "bearish"
            elif doji_type == "perfect_doji":
                # Perfect Doji требует подтверждения
                return "neutral"
            else:
                # Обычный Doji - нейтральный сигнал
                return "neutral"

        except Exception as e:
            logger.error(f"Ошибка при определении типа сигнала: {e}")
            return "neutral"


class HammerPattern(BasePattern):
    """
    Класс для обнаружения паттерна Hammer (молот)
    """

    def detect(self, min_body_ratio: float = 0.3, max_upper_wick: float = 0.1,
               min_lower_wick_ratio: float = 2.0) -> List[Dict[str, Any]]:
        """
        Обнаружение паттерна Hammer

        Args:
            min_body_ratio (float): Минимальное соотношение тела к диапазону
            max_upper_wick (float): Максимальное соотношение верхней тени к диапазону
            min_lower_wick_ratio (float): Минимальное соотношение нижней тени к телу

        Returns:
            list: Список обнаруженных паттернов Hammer
        """
        patterns = []

        if self.data is None or self.data.empty:
            logger.warning("Нет данных для обнаружения паттерна Hammer")
            return patterns

        try:
            for i in range(1, len(self.data) - 1):
                current = self.data.iloc[i]

                # Проверяем, что есть необходимые колонки
                if not all(col in current for col in ['Open', 'High', 'Low', 'Close']):
                    continue

                # Рассчитываем размеры
                body_size = abs(current['Close'] - current['Open'])
                total_range = current['High'] - current['Low']

                if total_range == 0:
                    continue

                body_ratio = body_size / total_range

                # Проверяем условие маленького тела
                if body_ratio < min_body_ratio:
                    # Рассчитываем тени
                    upper_wick = current['High'] - max(current['Open'], current['Close'])
                    lower_wick = min(current['Open'], current['Close']) - current['Low']

                    # Проверяем верхнюю тень
                    upper_wick_ratio = upper_wick / total_range

                    if upper_wick_ratio <= max_upper_wick and lower_wick > 0:
                        # Проверяем соотношение нижней тени к телу
                        if body_size > 0 and lower_wick >= body_size * min_lower_wick_ratio:
                            # Определяем, является ли паттерн бычьим или медвежьим
                            is_bullish = current['Close'] > current['Open']

                            # Определяем тип паттерна
                            if is_bullish:
                                pattern_type = "hammer"
                            else:
                                pattern_type = "inverted_hammer"

                            # Создаем описание паттерна
                            pattern = {
                                'index': i,
                                'pattern_type': pattern_type,
                                'timestamp': self.data.index[i],
                                'open': current['Open'],
                                'high': current['High'],
                                'low': current['Low'],
                                'close': current['Close'],
                                'body_size': body_size,
                                'total_range': total_range,
                                'body_ratio': body_ratio,
                                'upper_wick': upper_wick,
                                'lower_wick': lower_wick,
                                'lower_wick_ratio': lower_wick / body_size if body_size > 0 else 0,
                                'is_bullish': is_bullish,
                                'volume': current.get('Volume', 0),
                                'volume_ratio': self._get_volume_ratio(i),
                                'confidence': 0.8 - (body_ratio * 0.5),  # Корректировка уверенности
                                'position_in_trend': self._get_position_in_trend(i),
                                'signal': "bullish" if is_bullish else "neutral"
                            }

                            patterns.append(pattern)

            logger.info(f"Обнаружено {len(patterns)} паттернов Hammer")
            self.patterns = patterns
            return patterns

        except Exception as e:
            logger.error(f"Ошибка при обнаружении паттерна Hammer: {e}")
            return []

    def _get_volume_ratio(self, index: int, lookback: int = 10) -> float:
        """
        Расчет соотношения объема

        Args:
            index (int): Индекс свечи
            lookback (int): Количество свечей для расчета среднего объема

        Returns:
            float: Соотношение объема
        """
        if index < lookback or 'Volume' not in self.data.columns:
            return 1.0

        try:
            current_volume = self.data.iloc[index]['Volume']

            # Рассчитываем средний объем за предыдущие свечи
            start_idx = max(0, index - lookback)
            avg_volume = self.data.iloc[start_idx:index]['Volume'].mean()

            if avg_volume > 0:
                return current_volume / avg_volume
            else:
                return 1.0

        except Exception as e:
            logger.error(f"Ошибка при расчете соотношения объема: {e}")
            return 1.0

    def _get_position_in_trend(self, index: int, lookback: int = 10) -> str:
        """
        Определение позиции в тренде

        Args:
            index (int): Индекс свечи
            lookback (int): Количество свечей для анализа

        Returns:
            str: Позиция в тренде
        """
        if index < lookback:
            return "unknown"

        try:
            # Получаем данные для анализа
            start_idx = max(0, index - lookback)
            data_slice = self.data.iloc[start_idx:index + 1]

            if len(data_slice) < 5:
                return "unknown"

            # Определяем максимумы и минимумы
            highs = data_slice['High'].values
            lows = data_slice['Low'].values
            current_high = highs[-1]
            current_low = lows[-1]

            max_high = highs.max()
            min_low = lows.min()

            # Определяем позицию
            if current_high >= max_high * 0.99:
                return "new_high"
            elif current_low <= min_low * 1.01:
                return "new_low"
            else:
                return "within_range"

        except Exception as e:
            logger.error(f"Ошибка при определении позиции в тренде: {e}")
            return "unknown"


class EngulfingPattern(BasePattern):
    """
    Класс для обнаружения паттерна Engulfing (поглощение)
    """

    def detect(self, min_body_ratio: float = 1.5, volume_increase: float = 1.2) -> List[Dict[str, Any]]:
        """
        Обнаружение паттерна Engulfing

        Args:
            min_body_ratio (float): Минимальное соотношение тел свечей
            volume_increase (float): Минимальное увеличение объема

        Returns:
            list: Список обнаруженных паттернов Engulfing
        """
        patterns = []

        if self.data is None or self.data.empty:
            logger.warning("Нет данных для обнаружения паттерна Engulfing")
            return patterns

        try:
            for i in range(2, len(self.data)):
                current = self.data.iloc[i]
                previous = self.data.iloc[i - 1]

                # Проверяем, что есть необходимые колонки
                required_cols = ['Open', 'High', 'Low', 'Close']
                if not all(col in current for col in required_cols) or \
                        not all(col in previous for col in required_cols):
                    continue

                # Определяем направления свечей
                current_bullish = current['Close'] > current['Open']
                previous_bullish = previous['Close'] > previous['Open']

                # Engulfing должен быть противоположного направления
                if current_bullish == previous_bullish:
                    continue

                # Размеры тел
                current_body = abs(current['Close'] - current['Open'])
                previous_body = abs(previous['Close'] - previous['Open'])

                if previous_body == 0:
                    continue

                # Соотношение тел
                body_ratio = current_body / previous_body

                # Проверяем условие поглощения
                if body_ratio >= min_body_ratio:
                    # Проверяем полное поглощение
                    is_full_engulfing = (
                            current['High'] >= previous['High'] and
                            current['Low'] <= previous['Low']
                    )

                    # Проверяем поглощение тела
                    is_body_engulfing = (
                            max(current['Open'], current['Close']) >= max(previous['Open'], previous['Close']) and
                            min(current['Open'], current['Close']) <= min(previous['Open'], previous['Close'])
                    )

                    if is_body_engulfing or is_full_engulfing:
                        # Проверяем объем
                        current_volume = current.get('Volume', 1)
                        previous_volume = previous.get('Volume', 1)
                        volume_ratio = current_volume / previous_volume if previous_volume > 0 else 1

                        if volume_ratio >= volume_increase:
                            # Определяем тип паттерна
                            if current_bullish:
                                pattern_type = "bullish_engulfing"
                                signal = "bullish"
                            else:
                                pattern_type = "bearish_engulfing"
                                signal = "bearish"

                            # Создаем описание паттерна
                            pattern = {
                                'index': i,
                                'pattern_type': pattern_type,
                                'timestamp': self.data.index[i],
                                'current_open': current['Open'],
                                'current_close': current['Close'],
                                'current_high': current['High'],
                                'current_low': current['Low'],
                                'previous_open': previous['Open'],
                                'previous_close': previous['Close'],
                                'previous_high': previous['High'],
                                'previous_low': previous['Low'],
                                'current_body': current_body,
                                'previous_body': previous_body,
                                'body_ratio': body_ratio,
                                'volume_ratio': volume_ratio,
                                'is_full_engulfing': is_full_engulfing,
                                'is_body_engulfing': is_body_engulfing,
                                'signal': signal,
                                'confidence': min(0.9, body_ratio / 3),
                                'price_action': self._analyze_price_action(i),
                                'trend_context': self._get_engulfing_trend_context(i)
                            }

                            patterns.append(pattern)

            logger.info(f"Обнаружено {len(patterns)} паттернов Engulfing")
            self.patterns = patterns
            return patterns

        except Exception as e:
            logger.error(f"Ошибка при обнаружении паттерна Engulfing: {e}")
            return []

    def _analyze_price_action(self, index: int, window: int = 5) -> Dict[str, Any]:
        """
        Анализ ценового действия вокруг паттерна

        Args:
            index (int): Индекс свечи с паттерном
            window (int): Окно анализа

        Returns:
            dict: Результаты анализа ценового действия
        """
        result = {
            'before_trend': 'unknown',
            'after_trend': 'unknown',
            'volatility_before': 0,
            'volatility_after': 0
        }

        if index < window or index + window >= len(self.data):
            return result

        try:
            # Анализ до паттерна
            before_data = self.data.iloc[index - window:index]
            if len(before_data) > 1:
                before_start = before_data.iloc[0]['Close']
                before_end = before_data.iloc[-1]['Close']
                result['before_trend'] = 'up' if before_end > before_start else 'down'
                result['volatility_before'] = before_data['Close'].std()

            # Анализ после паттерна (если данные доступны)
            if index + 1 < len(self.data):
                after_end_idx = min(index + window + 1, len(self.data))
                after_data = self.data.iloc[index + 1:after_end_idx]

                if len(after_data) > 1:
                    after_start = after_data.iloc[0]['Close']
                    after_end = after_data.iloc[-1]['Close']
                    result['after_trend'] = 'up' if after_end > after_start else 'down'
                    result['volatility_after'] = after_data['Close'].std()

        except Exception as e:
            logger.error(f"Ошибка при анализе ценового действия: {e}")

        return result

    def _get_engulfing_trend_context(self, index: int, lookback: int = 20) -> str:
        """
        Определение трендового контекста для паттерна Engulfing

        Args:
            index (int): Индекс свечи
            lookback (int): Количество свечей для анализа

        Returns:
            str: Контекст тренда
        """
        if index < lookback:
            return "insufficient_data"

        try:
            # Получаем данные для анализа
            start_idx = max(0, index - lookback)
            data_slice = self.data.iloc[start_idx:index]

            if len(data_slice) < 5:
                return "insufficient_data"

            # Анализируем тренд
            closes = data_slice['Close'].values
            price_change = (closes[-1] - closes[0]) / closes[0] * 100

            # Рассчитываем скользящие средние
            sma_short = closes[-10:].mean() if len(closes) >= 10 else closes.mean()
            sma_long = closes.mean()

            # Определяем контекст
            if price_change > 3 and sma_short > sma_long:
                return "strong_uptrend"
            elif price_change > 1:
                return "uptrend"
            elif price_change < -3 and sma_short < sma_long:
                return "strong_downtrend"
            elif price_change < -1:
                return "downtrend"
            else:
                return "sideways"

        except Exception as e:
            logger.error(f"Ошибка при определении трендового контекста: {e}")
            return "unknown"


# Создаем экземпляры классов для удобства
doji_pattern = DojiPattern(None)
hammer_pattern = HammerPattern(None)
engulfing_pattern = EngulfingPattern(None)

