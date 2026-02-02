"""
Классы свечных паттернов
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import talib

from .base_pattern import (
    BasePattern, PatternType, PatternDirection,
    PatternPoint, MarketContext
)
from config import DETECTION_CONFIG


class CandlestickPattern(BasePattern):
    """Базовый класс для свечных паттернов"""

    def __init__(self, name: str, abbreviation: str = ""):
        super().__init__(PatternType.CANDLESTICK, name, abbreviation)
        self.required_candles = 3  # По умолчанию для большинства свечных паттернов
        self.body_ratio_threshold = 0.1  # Порог для определения доджи

    def _calculate_candle_metrics(self, opens: np.ndarray, highs: np.ndarray,
                                  lows: np.ndarray, closes: np.ndarray) -> Dict[str, np.ndarray]:
        """Расчет метрик свечей"""
        metrics = {}

        # Тела свечей
        metrics['bodies'] = closes - opens
        metrics['body_sizes'] = np.abs(metrics['bodies'])

        # Высоты свечей
        metrics['heights'] = highs - lows

        # Тени
        metrics['upper_shadows'] = np.where(
            metrics['bodies'] > 0,
            highs - closes,  # Для бычьих свечей
            highs - opens  # Для медвежьих свечей
        )

        metrics['lower_shadows'] = np.where(
            metrics['bodies'] > 0,
            opens - lows,  # Для бычьих свечей
            closes - lows  # Для медвежьих свечей
        )

        # Соотношения
        with np.errstate(divide='ignore', invalid='ignore'):
            metrics['body_to_height_ratio'] = np.where(
                metrics['heights'] > 0,
                metrics['body_sizes'] / metrics['heights'],
                0
            )

            metrics['upper_shadow_ratio'] = np.where(
                metrics['heights'] > 0,
                metrics['upper_shadows'] / metrics['heights'],
                0
            )

            metrics['lower_shadow_ratio'] = np.where(
                metrics['heights'] > 0,
                metrics['lower_shadows'] / metrics['heights'],
                0
            )

        return metrics

    def _is_doji(self, body_size: float, height: float) -> bool:
        """Определение доджи"""
        if height == 0:
            return False
        return body_size / height < self.body_ratio_threshold

    def _is_long_candle(self, body_size: float, avg_body_size: float) -> bool:
        """Определение длинной свечи"""
        return body_size > avg_body_size * 1.5

    def _is_engulfing(self, prev_open: float, prev_close: float,
                      curr_open: float, curr_close: float) -> Tuple[bool, bool]:
        """Определение поглощения"""
        prev_body = abs(prev_close - prev_open)
        curr_body = abs(curr_close - curr_open)

        # Бычье поглощение
        bullish = (prev_close < prev_open and  # Предыдущая медвежья
                   curr_close > curr_open and  # Текущая бычья
                   curr_open < prev_close and  # Открытие ниже закрытия предыдущей
                   curr_close > prev_open)  # Закрытие выше открытия предыдущей

        # Медвежье поглощение
        bearish = (prev_close > prev_open and  # Предыдущая бычья
                   curr_close < curr_open and  # Текущая медвежья
                   curr_open > prev_close and  # Открытие выше закрытия предыдущей
                   curr_close < prev_open)  # Закрытие ниже открытия предыдущей

        return bullish, bearish


class EngulfingPattern(CandlestickPattern):
    """Паттерн Поглощение"""

    def __init__(self, is_bullish: bool = True):
        name = "Bullish Engulfing" if is_bullish else "Bearish Engulfing"
        abbreviation = "BENG" if is_bullish else "SENG"
        super().__init__(name, abbreviation)

        self.is_bullish = is_bullish
        self.required_candles = 2

    def detect(self, data, highs, lows, closes, volumes, timestamps, **kwargs) -> bool:
        opens = kwargs.get('opens', np.array([]))

        if len(opens) < 2 or len(closes) < 2:
            return False

        # Берем последние две свечи
        prev_open = opens[-2]
        prev_close = closes[-2]
        curr_open = opens[-1]
        curr_close = closes[-1]

        # Определяем поглощение
        bullish_engulfing, bearish_engulfing = self._is_engulfing(
            prev_open, prev_close, curr_open, curr_close
        )

        if self.is_bullish and bullish_engulfing:
            self.direction = PatternDirection.BULLISH
            self._save_points([-2, -1], opens, closes, highs, lows, timestamps)
            self._is_detected = True
            return True
        elif not self.is_bullish and bearish_engulfing:
            self.direction = PatternDirection.BEARISH
            self._save_points([-2, -1], opens, closes, highs, lows, timestamps)
            self._is_detected = True
            return True

        return False

    def _save_points(self, indices, opens, closes, highs, lows, timestamps):
        """Сохранение точек паттерна"""
        self.points = []

        for idx_offset in indices:
            idx = len(opens) + idx_offset if idx_offset < 0 else idx_offset
            if 0 <= idx < len(opens):
                self.points.append(
                    PatternPoint(
                        index=idx,
                        timestamp=timestamps[idx],
                        price=closes[idx],
                        point_type='candle_close'
                    )
                )

    def calculate_quality(self) -> float:
        if not self._is_detected or len(self.points) < 2:
            return 0.0

        scores = []

        # 1. Размер поглощающей свечи
        if len(self.points) == 2:
            # Сравниваем тела свечей
            # Для этого нужно иметь доступ к opens и closes
            if hasattr(self, '_opens') and hasattr(self, '_closes'):
                idx1 = self.points[0].index
                idx2 = self.points[1].index

                if idx1 < len(self._opens) and idx2 < len(self._opens):
                    prev_body = abs(self._closes[idx1] - self._opens[idx1])
                    curr_body = abs(self._closes[idx2] - self._opens[idx2])

                    if prev_body > 0:
                        size_ratio = curr_body / prev_body
                        size_score = min(size_ratio / 2, 1.0)  # Чем больше, тем лучше
                        scores.append(size_score)

        # 2. Объем
        if self.metadata.volume_confirmation:
            scores.append(0.8)
        else:
            scores.append(0.5)

        # 3. Положение в тренде
        if self.metadata.market_context == MarketContext.DOWNTREND and self.is_bullish:
            scores.append(0.9)  # Бычье поглощение в нисходящем тренде - хорошо
        elif self.metadata.market_context == MarketContext.UPTREND and not self.is_bullish:
            scores.append(0.9)  # Медвежье поглощение в восходящем тренде - хорошо
        else:
            scores.append(0.6)

        return np.mean(scores) if scores else 0.0

    def calculate_targets(self, current_price: float):
        # Для свечных паттернов используем стандартный расчет
        return super().calculate_targets(current_price)


class DojiPattern(CandlestickPattern):
    """Паттерн Доджи"""

    def __init__(self, doji_type: str = "standard"):
        """
        Args:
            doji_type: "standard", "long_legged", "dragonfly", "gravestone"
        """
        name = f"{doji_type.replace('_', ' ').title()} Doji"
        abbreviation = {
            "standard": "DOJI",
            "long_legged": "LLD",
            "dragonfly": "DFD",
            "gravestone": "GSD"
        }.get(doji_type, "DOJI")

        super().__init__(name, abbreviation)

        self.doji_type = doji_type
        self.required_candles = 1

    def detect(self, data, highs, lows, closes, volumes, timestamps, **kwargs) -> bool:
        opens = kwargs.get('opens', np.array([]))

        if len(opens) < 1:
            return False

        # Берем последнюю свечу
        curr_open = opens[-1]
        curr_close = closes[-1]
        curr_high = highs[-1]
        curr_low = lows[-1]

        # Расчет метрик
        body_size = abs(curr_close - curr_open)
        height = curr_high - curr_low

        if height == 0:
            return False

        # Проверяем доджи
        if not self._is_doji(body_size, height):
            return False

        # Определяем тип доджи
        upper_shadow = curr_high - max(curr_open, curr_close)
        lower_shadow = min(curr_open, curr_close) - curr_low

        # Стандартный доджи
        if self.doji_type == "standard":
            # У стандартного доджи тени примерно равны
            if abs(upper_shadow - lower_shadow) / height < 0.3:
                self.direction = PatternDirection.NEUTRAL
                self._save_points([-1], opens, closes, highs, lows, timestamps)
                self._is_detected = True
                return True

        # Доджи с длинными ногами
        elif self.doji_type == "long_legged":
            # Длинные верхняя и нижняя тени
            if upper_shadow > height * 0.4 and lower_shadow > height * 0.4:
                self.direction = PatternDirection.NEUTRAL
                self._save_points([-1], opens, closes, highs, lows, timestamps)
                self._is_detected = True
                return True

        # Доджи-стрекоза
        elif self.doji_type == "dragonfly":
            # Очень маленькая или отсутствующая верхняя тень, длинная нижняя
            if upper_shadow < height * 0.1 and lower_shadow > height * 0.7:
                self.direction = PatternDirection.BULLISH
                self._save_points([-1], opens, closes, highs, lows, timestamps)
                self._is_detected = True
                return True

        # Доджи-надгробие
        elif self.doji_type == "gravestone":
            # Очень маленькая или отсутствующая нижняя тень, длинная верхняя
            if lower_shadow < height * 0.1 and upper_shadow > height * 0.7:
                self.direction = PatternDirection.BEARISH
                self._save_points([-1], opens, closes, highs, lows, timestamps)
                self._is_detected = True
                return True

        return False

    def _save_points(self, indices, opens, closes, highs, lows, timestamps):
        """Сохранение точек паттерна"""
        self.points = []

        for idx_offset in indices:
            idx = len(opens) + idx_offset if idx_offset < 0 else idx_offset
            if 0 <= idx < len(opens):
                self.points.append(
                    PatternPoint(
                        index=idx,
                        timestamp=timestamps[idx],
                        price=closes[idx],
                        point_type='doji_center'
                    )
                )

    def calculate_quality(self) -> float:
        if not self._is_detected or not self.points:
            return 0.0

        scores = []

        # 1. Четкость доджи
        if hasattr(self, '_opens') and hasattr(self, '_closes') and \
                hasattr(self, '_highs') and hasattr(self, '_lows'):

            idx = self.points[0].index
            if idx < len(self._opens):
                open_price = self._opens[idx]
                close_price = self._closes[idx]
                high = self._highs[idx]
                low = self._lows[idx]

                body_size = abs(close_price - open_price)
                height = high - low

                if height > 0:
                    doji_score = 1 - (body_size / height) / self.body_ratio_threshold
                    scores.append(max(0, doji_score))

        # 2. Длина теней в соответствии с типом
        if self.doji_type in ["dragonfly", "gravestone", "long_legged"]:
            if hasattr(self, '_opens') and hasattr(self, '_closes') and \
                    hasattr(self, '_highs') and hasattr(self, '_lows'):

                idx = self.points[0].index
                if idx < len(self._opens):
                    open_price = self._opens[idx]
                    close_price = self._closes[idx]
                    high = self._highs[idx]
                    low = self._lows[idx]

                    upper_shadow = high - max(open_price, close_price)
                    lower_shadow = min(open_price, close_price) - low
                    height = high - low

                    if height > 0:
                        if self.doji_type == "dragonfly":
                            # Длинная нижняя тень, короткая верхняя
                            lower_ratio = lower_shadow / height
                            upper_ratio = upper_shadow / height
                            shadow_score = min(lower_ratio / 0.7, 1.0) * (1 - min(upper_ratio / 0.1, 1.0))
                            scores.append(shadow_score)

                        elif self.doji_type == "gravestone":
                            # Длинная верхняя тень, короткая нижняя
                            upper_ratio = upper_shadow / height
                            lower_ratio = lower_shadow / height
                            shadow_score = min(upper_ratio / 0.7, 1.0) * (1 - min(lower_ratio / 0.1, 1.0))
                            scores.append(shadow_score)

                        elif self.doji_type == "long_legged":
                            # Длинные обе тени
                            upper_ratio = upper_shadow / height
                            lower_ratio = lower_shadow / height
                            shadow_score = min(upper_ratio / 0.4, 1.0) * min(lower_ratio / 0.4, 1.0)
                            scores.append(shadow_score)

        # 3. Объем
        if self.metadata.volume_confirmation:
            scores.append(0.7)
        else:
            scores.append(0.4)

        return np.mean(scores) if scores else 0.0

    def calculate_targets(self, current_price: float):
        # Доджи обычно не дает четких целей, используем стандартный расчет
        return super().calculate_targets(current_price)


class HammerPattern(CandlestickPattern):
    """Паттерны Молот и Висячий человек"""

    def __init__(self, is_hammer: bool = True):
        name = "Hammer" if is_hammer else "Hanging Man"
        abbreviation = "HAM" if is_hammer else "HGM"
        super().__init__(name, abbreviation)

        self.is_hammer = is_hammer
        self.required_candles = 1

        # Параметры для определения
        self.min_lower_shadow_ratio = 2.0  # Нижняя тень минимум в 2 раза больше тела
        self.max_upper_shadow_ratio = 0.3  # Верхняя тень не более 30% от высоты
        self.max_body_to_height_ratio = 0.3  # Тело не более 30% от высоты

    def detect(self, data, highs, lows, closes, volumes, timestamps, **kwargs) -> bool:
        opens = kwargs.get('opens', np.array([]))

        if len(opens) < 1:
            return False

        # Берем последнюю свечу
        curr_open = opens[-1]
        curr_close = closes[-1]
        curr_high = highs[-1]
        curr_low = lows[-1]

        # Расчет метрик
        body_size = abs(curr_close - curr_open)
        height = curr_high - curr_low

        if height == 0:
            return False

        upper_shadow = curr_high - max(curr_open, curr_close)
        lower_shadow = min(curr_open, curr_close) - curr_low

        # Соотношения
        body_to_height = body_size / height
        lower_to_body = lower_shadow / body_size if body_size > 0 else 0
        upper_to_height = upper_shadow / height

        # Проверяем условия для молота/висячего человека
        is_valid = (
                body_to_height <= self.max_body_to_height_ratio and  # Маленькое тело
                lower_to_body >= self.min_lower_shadow_ratio and  # Длинная нижняя тень
                upper_to_height <= self.max_upper_shadow_ratio  # Короткая или отсутствующая верхняя тень
        )

        if not is_valid:
            return False

        # Определяем направление
        # Молот - бычий паттерн, обычно в конце нисходящего тренда
        # Висячий человек - медвежий паттерн, обычно в конце восходящего тренда
        if self.is_hammer:
            self.direction = PatternDirection.BULLISH
        else:
            self.direction = PatternDirection.BEARISH

        self._save_points([-1], opens, closes, highs, lows, timestamps)
        self._is_detected = True
        return True

    def _save_points(self, indices, opens, closes, highs, lows, timestamps):
        """Сохранение точек паттерна"""
        self.points = []

        for idx_offset in indices:
            idx = len(opens) + idx_offset if idx_offset < 0 else idx_offset
            if 0 <= idx < len(opens):
                # Сохраняем точку закрытия и точку минимума (для молота)
                self.points.append(
                    PatternPoint(
                        index=idx,
                        timestamp=timestamps[idx],
                        price=closes[idx],
                        point_type='candle_close'
                    )
                )

                self.points.append(
                    PatternPoint(
                        index=idx,
                        timestamp=timestamps[idx],
                        price=lows[idx],
                        point_type='hammer_low'
                    )
                )

    def calculate_quality(self) -> float:
        if not self._is_detected or not self.points:
            return 0.0

        scores = []

        # 1. Соотношение теней
        if hasattr(self, '_opens') and hasattr(self, '_closes') and \
                hasattr(self, '_highs') and hasattr(self, '_lows'):

            idx = self.points[0].index
            if idx < len(self._opens):
                open_price = self._opens[idx]
                close_price = self._closes[idx]
                high = self._highs[idx]
                low = self._lows[idx]

                body_size = abs(close_price - open_price)
                height = high - low
                lower_shadow = min(open_price, close_price) - low
                upper_shadow = high - max(open_price, close_price)

                if body_size > 0 and height > 0:
                    # Нижняя тень должна быть длинной
                    lower_ratio = lower_shadow / body_size
                    lower_score = min(lower_ratio / self.min_lower_shadow_ratio, 1.0)
                    scores.append(lower_score)

                    # Верхняя тень должна быть короткой
                    upper_ratio = upper_shadow / height
                    upper_score = 1 - min(upper_ratio / self.max_upper_shadow_ratio, 1.0)
                    scores.append(upper_score)

                    # Тело должно быть маленьким
                    body_ratio = body_size / height
                    body_score = 1 - min(body_ratio / self.max_body_to_height_ratio, 1.0)
                    scores.append(body_score)

        # 2. Положение в тренде
        if self.is_hammer:
            # Молот лучше работает в нисходящем тренде
            if self.metadata.market_context == MarketContext.DOWNTREND:
                scores.append(0.9)
            else:
                scores.append(0.6)
        else:
            # Висячий человек лучше работает в восходящем тренде
            if self.metadata.market_context == MarketContext.UPTREND:
                scores.append(0.9)
            else:
                scores.append(0.6)

        # 3. Объем
        if self.metadata.volume_confirmation:
            scores.append(0.7)
        else:
            scores.append(0.4)

        return np.mean(scores) if scores else 0.0

    def calculate_targets(self, current_price: float):
        # Для молота/висячего человека цели рассчитываем от высоты свечи
        if not self._is_detected or not self.points:
            return super().calculate_targets(current_price)

        if hasattr(self, '_highs') and hasattr(self, '_lows'):
            idx = self.points[0].index
            if idx < len(self._highs):
                high = self._highs[idx]
                low = self._lows[idx]
                height = high - low

                if self.is_hammer:
                    # Бычий молот - цель вверх
                    self.targets.entry_price = current_price
                    self.targets.stop_loss = low * 0.995  # 0.5% ниже минимума
                    self.targets.take_profit = current_price + height

                    self.targets.target1 = current_price + height * 0.5
                    self.targets.target2 = current_price + height
                    self.targets.target3 = current_price + height * 1.5
                else:
                    # Медвежий висячий человек - цель вниз
                    self.targets.entry_price = current_price
                    self.targets.stop_loss = high * 1.005  # 0.5% выше максимума
                    self.targets.take_profit = current_price - height

                    self.targets.target1 = current_price - height * 0.5
                    self.targets.target2 = current_price - height
                    self.targets.target3 = current_price - height * 1.5

                # Риск/прибыль
                if self.targets.entry_price and self.targets.stop_loss and self.targets.take_profit:
                    risk = abs(self.targets.entry_price - self.targets.stop_loss)
                    reward = abs(self.targets.take_profit - self.targets.entry_price)
                    self.targets.profit_risk_ratio = reward / risk if risk > 0 else 0

        return self.targets


class MorningStarPattern(CandlestickPattern):
    """Паттерн Утренняя звезда"""

    def __init__(self):
        super().__init__("Morning Star", "MST")
        self.required_candles = 3

    def detect(self, data, highs, lows, closes, volumes, timestamps, **kwargs) -> bool:
        opens = kwargs.get('opens', np.array([]))

        if len(opens) < 3:
            return False

        # Берем последние три свечи
        candle1_open = opens[-3]
        candle1_close = closes[-3]

        candle2_open = opens[-2]
        candle2_close = closes[-2]
        candle2_high = highs[-2]
        candle2_low = lows[-2]

        candle3_open = opens[-1]
        candle3_close = closes[-1]

        # Условия для Утренней звезды:
        # 1. Первая свеча - длинная медвежья
        # 2. Вторая свеча - маленькое тело (доджи или маленькая свеча), может быть гэпом
        # 3. Третья свеча - длинная бычья, закрывается выше середины тела первой свечи

        # Проверяем первую свечу
        candle1_body = abs(candle1_close - candle1_open)
        candle1_height = highs[-3] - lows[-3]

        if candle1_height == 0:
            return False

        # Первая свеча должна быть медвежьей и длинной
        is_candle1_bearish = candle1_close < candle1_open
        is_candle1_long = candle1_body / candle1_height > 0.6

        if not (is_candle1_bearish and is_candle1_long):
            return False

        # Проверяем вторую свечу
        candle2_body = abs(candle2_close - candle2_open)
        candle2_height = candle2_high - candle2_low

        if candle2_height == 0:
            return False

        # Вторая свеча должна иметь маленькое тело (возможно доджи)
        is_candle2_small = candle2_body / candle2_height < 0.3

        # Вторая свеча может открыться с гэпом вниз
        has_gap_down = candle2_open < candle1_close

        if not (is_candle2_small or has_gap_down):
            return False

        # Проверяем третью свечу
        candle3_body = abs(candle3_close - candle3_open)
        candle3_height = highs[-1] - lows[-1]

        if candle3_height == 0:
            return False

        # Третья свеча должна быть бычьей и длинной
        is_candle3_bullish = candle3_close > candle3_open
        is_candle3_long = candle3_body / candle3_height > 0.6

        if not (is_candle3_bullish and is_candle3_long):
            return False

        # Третья свеча должна закрыться выше середины тела первой свечи
        candle1_mid = (candle1_open + candle1_close) / 2
        if candle3_close <= candle1_mid:
            return False

        # Третья свеча должна закрыться выше открытия первой свечи (идеально)
        if candle3_close > candle1_open:
            quality_boost = True

        # Все условия выполнены
        self.direction = PatternDirection.BULLISH
        self._save_points([-3, -2, -1], opens, closes, highs, lows, timestamps)
        self._is_detected = True
        return True

    def _save_points(self, indices, opens, closes, highs, lows, timestamps):
        """Сохранение точек паттерна"""
        self.points = []

        for idx_offset in indices:
            idx = len(opens) + idx_offset if idx_offset < 0 else idx_offset
            if 0 <= idx < len(opens):
                point_type = 'star_candle'
                if idx_offset == -3:
                    point_type = 'first_candle'
                elif idx_offset == -2:
                    point_type = 'star'
                elif idx_offset == -1:
                    point_type = 'third_candle'

                self.points.append(
                    PatternPoint(
                        index=idx,
                        timestamp=timestamps[idx],
                        price=closes[idx],
                        point_type=point_type
                    )
                )

    def calculate_quality(self) -> float:
        if not self._is_detected or len(self.points) < 3:
            return 0.0

        scores = []

        # 1. Длина свечей
        if hasattr(self, '_opens') and hasattr(self, '_closes') and \
                hasattr(self, '_highs') and hasattr(self, '_lows'):

            # Первая свеча (медвежья)
            idx1 = self.points[0].index
            candle1_body = abs(self._closes[idx1] - self._opens[idx1])
            candle1_height = self._highs[idx1] - self._lows[idx1]

            # Вторая свеча (звезда)
            idx2 = self.points[1].index
            candle2_body = abs(self._closes[idx2] - self._opens[idx2])
            candle2_height = self._highs[idx2] - self._lows[idx2]

            # Третья свеча (бычья)
            idx3 = self.points[2].index
            candle3_body = abs(self._closes[idx3] - self._opens[idx3])
            candle3_height = self._highs[idx3] - self._lows[idx3]

            if candle1_height > 0 and candle2_height > 0 and candle3_height > 0:
                # Первая свеча должна быть длинной
                candle1_ratio = candle1_body / candle1_height
                candle1_score = min(candle1_ratio / 0.6, 1.0)
                scores.append(candle1_score)

                # Вторая свеча должна быть маленькой
                candle2_ratio = candle2_body / candle2_height
                candle2_score = 1 - min(candle2_ratio / 0.3, 1.0)
                scores.append(candle2_score)

                # Третья свеча должна быть длинной
                candle3_ratio = candle3_body / candle3_height
                candle3_score = min(candle3_ratio / 0.6, 1.0)
                scores.append(candle3_score)

        # 2. Положение звезды
        if len(self.points) == 3:
            # Звезда должна быть ниже первой свечи
            if self.points[1].price < self.points[0].price:
                scores.append(0.8)
            else:
                scores.append(0.4)

            # Третья свеча должна закрыться выше середины первой
            candle1_mid = (self._opens[idx1] + self._closes[idx1]) / 2
            if self.points[2].price > candle1_mid:
                scores.append(0.9)
            else:
                scores.append(0.5)

        # 3. Объем
        if self.metadata.volume_confirmation:
            scores.append(0.8)
        else:
            scores.append(0.5)

        # 4. Тренд
        if self.metadata.market_context == MarketContext.DOWNTREND:
            scores.append(0.9)  # Утренняя звезда лучше в нисходящем тренде
        else:
            scores.append(0.6)

        return np.mean(scores) if scores else 0.0

    def calculate_targets(self, current_price: float):
        # Для утренней звезды цель - высота паттерна
        if not self._is_detected or len(self.points) < 3:
            return super().calculate_targets(current_price)

        if hasattr(self, '_highs') and hasattr(self, '_lows'):
            idx1 = self.points[0].index
            idx3 = self.points[2].index

            # Высота от максимума первой свечи до минимума звезды (второй свечи)
            pattern_high = self._highs[idx1]
            pattern_low = self._lows[self.points[1].index]
            pattern_height = pattern_high - pattern_low

            self.targets.entry_price = current_price
            self.targets.stop_loss = pattern_low * 0.99  # 1% ниже минимума звезды
            self.targets.take_profit = current_price + pattern_height

            self.targets.target1 = current_price + pattern_height * 0.5
            self.targets.target2 = current_price + pattern_height
            self.targets.target3 = current_price + pattern_height * 1.5

            # Риск/прибыль
            if self.targets.entry_price and self.targets.stop_loss and self.targets.take_profit:
                risk = abs(self.targets.entry_price - self.targets.stop_loss)
                reward = abs(self.targets.take_profit - self.targets.entry_price)
                self.targets.profit_risk_ratio = reward / risk if risk > 0 else 0

        return self.targets


class EveningStarPattern(MorningStarPattern):
    """Паттерн Вечерняя звезда (обратная Утренней звезде)"""

    def __init__(self):
        super().__init__()
        self.name = "Evening Star"
        self.abbreviation = "EST"

    def detect(self, data, highs, lows, closes, volumes, timestamps, **kwargs) -> bool:
        opens = kwargs.get('opens', np.array([]))

        if len(opens) < 3:
            return False

        # Берем последние три свечи
        candle1_open = opens[-3]
        candle1_close = closes[-3]

        candle2_open = opens[-2]
        candle2_close = closes[-2]
        candle2_high = highs[-2]
        candle2_low = lows[-2]

        candle3_open = opens[-1]
        candle3_close = closes[-1]

        # Условия для Вечерней звезды:
        # 1. Первая свеча - длинная бычья
        # 2. Вторая свеча - маленькое тело (доджи или маленькая свеча), может быть гэпом
        # 3. Третья свеча - длинная медвежья, закрывается ниже середины тела первой свечи

        # Проверяем первую свечу
        candle1_body = abs(candle1_close - candle1_open)
        candle1_height = highs[-3] - lows[-3]

        if candle1_height == 0:
            return False

        # Первая свеча должна быть бычьей и длинной
        is_candle1_bullish = candle1_close > candle1_open
        is_candle1_long = candle1_body / candle1_height > 0.6

        if not (is_candle1_bullish and is_candle1_long):
            return False

        # Проверяем вторую свечу
        candle2_body = abs(candle2_close - candle2_open)
        candle2_height = candle2_high - candle2_low

        if candle2_height == 0:
            return False

        # Вторая свеча должна иметь маленькое тело (возможно доджи)
        is_candle2_small = candle2_body / candle2_height < 0.3

        # Вторая свеча может открыться с гэпом вверх
        has_gap_up = candle2_open > candle1_close

        if not (is_candle2_small or has_gap_up):
            return False

        # Проверяем третью свечу
        candle3_body = abs(candle3_close - candle3_open)
        candle3_height = highs[-1] - lows[-1]

        if candle3_height == 0:
            return False

        # Третья свеча должна быть медвежьей и длинной
        is_candle3_bearish = candle3_close < candle3_open
        is_candle3_long = candle3_body / candle3_height > 0.6

        if not (is_candle3_bearish and is_candle3_long):
            return False

        # Третья свеча должна закрыться ниже середины тела первой свечи
        candle1_mid = (candle1_open + candle1_close) / 2
        if candle3_close >= candle1_mid:
            return False

        # Третья свеча должна закрыться ниже открытия первой свечи (идеально)
        if candle3_close < candle1_open:
            quality_boost = True

        # Все условия выполнены
        self.direction = PatternDirection.BEARISH
        self._save_points([-3, -2, -1], opens, closes, highs, lows, timestamps)
        self._is_detected = True
        return True

    def calculate_targets(self, current_price: float):
        # Для вечерней звезды цель - высота паттерна вниз
        if not self._is_detected or len(self.points) < 3:
            return super().calculate_targets(current_price)

        if hasattr(self, '_highs') and hasattr(self, '_lows'):
            idx1 = self.points[0].index
            idx3 = self.points[2].index

            # Высота от минимума первой свечи до максимума звезды (второй свечи)
            pattern_low = self._lows[idx1]
            pattern_high = self._highs[self.points[1].index]
            pattern_height = pattern_high - pattern_low

            self.targets.entry_price = current_price
            self.targets.stop_loss = pattern_high * 1.01  # 1% выше максимума звезды
            self.targets.take_profit = current_price - pattern_height

            self.targets.target1 = current_price - pattern_height * 0.5
            self.targets.target2 = current_price - pattern_height
            self.targets.target3 = current_price - pattern_height * 1.5

            # Риск/прибыль
            if self.targets.entry_price and self.targets.stop_loss and self.targets.take_profit:
                risk = abs(self.targets.entry_price - self.targets.stop_loss)
                reward = abs(self.targets.take_profit - self.targets.entry_price)
                self.targets.profit_risk_ratio = reward / risk if risk > 0 else 0

        return self.targets

