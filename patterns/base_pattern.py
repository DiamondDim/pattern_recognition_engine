"""
Базовый класс для всех паттернов
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field

@dataclass
class PatternPoint:
    """Точка паттерна"""
    index: int  # Индекс свечи
    price: float  # Цена точки
    time: Optional[str] = None  # Временная метка
    type: Optional[str] = None  # Тип точки (high, low, pivot и т.д.)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PatternResult:
    """Результат обнаружения паттерна"""
    name: str
    direction: str  # bullish, bearish, neutral
    points: List[PatternPoint]
    quality: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0
    targets: Dict[str, float] = field(default_factory=dict)  # Целевые уровни
    metadata: Dict[str, Any] = field(default_factory=dict)

class BasePattern(ABC):
    """Абстрактный базовый класс для всех паттернов"""

    def __init__(self, name: str, min_points: int = 3):
        self.name = name
        self.min_points = min_points
        self.pattern_type = self.__class__.__module__.split('.')[-1]

    @abstractmethod
    def detect(self, data: Dict[str, np.ndarray]) -> List[PatternResult]:
        """
        Детектирование паттерна

        Args:
            data: Входные данные OHLC

        Returns:
            Список обнаруженных паттернов
        """
        pass

    def validate_points(self, points: List[PatternPoint]) -> bool:
        """Валидация точек паттерна"""
        if len(points) < self.min_points:
            return False

        # Проверка уникальности индексов
        indices = [p.index for p in points]
        if len(set(indices)) != len(indices):
            return False

        # Проверка порядка индексов
        if sorted(indices) != indices:
            return False

        return True

    def calculate_quality(self, points: List[PatternPoint],
                         ideal_pattern: List[Tuple[float, float]]) -> float:
        """
        Расчет качества паттерна на основе отклонения от идеальной формы

        Args:
            points: Фактические точки паттерна
            ideal_pattern: Идеальные относительные координаты [(x_rel, y_rel), ...]

        Returns:
            Оценка качества 0.0-1.0
        """
        if len(points) != len(ideal_pattern):
            return 0.0

        # Нормализация координат
        indices = [p.index for p in points]
        prices = [p.price for p in points]

        min_idx = min(indices)
        max_idx = max(indices)
        min_price = min(prices)
        max_price = max(prices)

        if max_idx == min_idx or max_price == min_price:
            return 0.5

        # Преобразование в относительные координаты
        actual_points = []
        for i, point in enumerate(points):
            x_rel = (point.index - min_idx) / (max_idx - min_idx)
            y_rel = (point.price - min_price) / (max_price - min_price)
            actual_points.append((x_rel, y_rel))

        # Расчет среднеквадратичной ошибки
        mse = 0.0
        for actual, ideal in zip(actual_points, ideal_pattern):
            dx = actual[0] - ideal[0]
            dy = actual[1] - ideal[1]
            mse += dx*dx + dy*dy

        mse /= len(actual_points)

        # Преобразование MSE в качество (чем меньше ошибка, тем выше качество)
        quality = 1.0 / (1.0 + 10.0 * mse)
        return min(max(quality, 0.0), 1.0)

    def calculate_confidence(self, points: List[PatternPoint],
                           data: Dict[str, np.ndarray]) -> float:
        """
        Расчет уверенности в паттерне

        Args:
            points: Точки паттерна
            data: Входные данные

        Returns:
            Оценка уверенности 0.0-1.0
        """
        if not points:
            return 0.0

        # 1. Проверка свечей вокруг паттерна
        candle_score = self._evaluate_candles_around_pattern(points, data)

        # 2. Проверка объема
        volume_score = self._evaluate_volume(points, data)

        # 3. Проверка тренда
        trend_score = self._evaluate_trend(points, data)

        # Итоговая уверенность
        confidence = (candle_score + volume_score + trend_score) / 3.0
        return min(max(confidence, 0.0), 1.0)

    def _evaluate_candles_around_pattern(self, points: List[PatternPoint],
                                       data: Dict[str, np.ndarray]) -> float:
        """Оценка свечей вокруг паттерна"""
        if len(points) < 2:
            return 0.5

        closes = data.get('close', [])
        if len(closes) == 0:
            return 0.5

        start_idx = points[0].index
        end_idx = points[-1].index

        # Анализируем свечи внутри паттерна
        pattern_candles = closes[start_idx:end_idx+1]

        if len(pattern_candles) < 2:
            return 0.5

        # Проверяем, есть ли длинные тени (признак борьбы)
        if 'high' in data and 'low' in data:
            highs = data['high'][start_idx:end_idx+1]
            lows = data['low'][start_idx:end_idx+1]

            avg_wick_size = np.mean((highs - closes[start_idx:end_idx+1]) +
                                   (closes[start_idx:end_idx+1] - lows))
            if avg_wick_size > 0:
                # Большие тени - снижаем уверенность
                wick_ratio = avg_wick_size / (np.max(highs) - np.min(lows))
                return max(0.0, 1.0 - wick_ratio * 2)

        return 0.7

    def _evaluate_volume(self, points: List[PatternPoint],
                        data: Dict[str, np.ndarray]) -> float:
        """Оценка объема"""
        if 'volume' not in data:
            return 0.5

        volumes = data['volume']
        if len(volumes) == 0:
            return 0.5

        start_idx = points[0].index
        end_idx = points[-1].index

        pattern_volumes = volumes[start_idx:end_idx+1]

        if len(pattern_volumes) == 0:
            return 0.5

        # Проверяем, выше ли объем на паттерне, чем средний
        if start_idx > 20:
            avg_volume_before = np.mean(volumes[start_idx-20:start_idx])
        else:
            avg_volume_before = np.mean(volumes[:start_idx]) if start_idx > 0 else 0

        if avg_volume_before > 0:
            volume_ratio = np.mean(pattern_volumes) / avg_volume_before
            # Высокий объем на паттерне - хороший знак
            if volume_ratio > 1.5:
                return 0.9
            elif volume_ratio > 1.0:
                return 0.7
            else:
                return 0.4

        return 0.5

    def _evaluate_trend(self, points: List[PatternPoint],
                       data: Dict[str, np.ndarray]) -> float:
        """Оценка тренда"""
        if 'close' not in data:
            return 0.5

        closes = data['close']
        if len(closes) < 10:
            return 0.5

        start_idx = points[0].index

        # Анализируем тренд до паттерна
        lookback = min(20, start_idx)
        if lookback > 5:
            prices_before = closes[start_idx-lookback:start_idx]
            if len(prices_before) > 1:
                # Линейная регрессия
                x = np.arange(len(prices_before))
                slope, _ = np.polyfit(x, prices_before, 1)

                # Для разворотных паттернов хотим видеть сильный предыдущий тренд
                # Для продолжения - слабый тренд или консолидацию
                trend_strength = abs(slope)

                if self.name.lower() in ['head_shoulders', 'double_top', 'double_bottom']:
                    # Разворотные паттерны
                    return min(1.0, trend_strength * 100)
                else:
                    # Паттерны продолжения
                    return max(0.0, 1.0 - trend_strength * 100)

        return 0.5

    def calculate_targets(self, points: List[PatternPoint],
                         pattern_type: str) -> Dict[str, float]:
        """
        Расчет целевых уровней для паттерна

        Args:
            points: Точки паттерна
            pattern_type: Тип паттерна

        Returns:
            Словарь с целевыми уровнями
        """
        if len(points) < 2:
            return {}

        prices = [p.price for p in points]
        indices = [p.index for p in points]

        # Базовая реализация - высота паттерна
        pattern_height = max(prices) - min(prices)

        # Точка входа (последняя точка паттерна)
        entry_price = points[-1].price

        # Определение направления
        direction = self._determine_direction(points)

        targets = {
            'entry_price': entry_price,
            'pattern_height': pattern_height
        }

        # Добавление конкретных целей в зависимости от типа паттерна
        if pattern_type == 'head_shoulders':
            neckline = self._calculate_neckline(points)
            if neckline is not None:
                targets['neckline'] = neckline
                if direction == 'bearish':
                    targets['stop_loss'] = max(prices) * 1.01
                    targets['take_profit'] = entry_price - pattern_height
                else:
                    targets['stop_loss'] = min(prices) * 0.99
                    targets['take_profit'] = entry_price + pattern_height

        elif pattern_type == 'double_top':
            resistance = max(prices)
            targets['resistance'] = resistance
            if direction == 'bearish':
                targets['stop_loss'] = resistance * 1.01
                targets['take_profit'] = entry_price - pattern_height

        elif pattern_type == 'double_bottom':
            support = min(prices)
            targets['support'] = support
            if direction == 'bullish':
                targets['stop_loss'] = support * 0.99
                targets['take_profit'] = entry_price + pattern_height

        elif pattern_type in ['triangle', 'wedge', 'flag']:
            # Для паттернов продолжения
            if direction == 'bullish':
                targets['stop_loss'] = min(prices) * 0.99
                targets['take_profit'] = entry_price + pattern_height
            else:
                targets['stop_loss'] = max(prices) * 1.01
                targets['take_profit'] = entry_price - pattern_height

        return targets

    def _determine_direction(self, points: List[PatternPoint]) -> str:
        """Определение направления паттерна"""
        if len(points) < 2:
            return 'neutral'

        prices = [p.price for p in points]

        # Простой алгоритм определения направления
        first_half = prices[:len(prices)//2]
        second_half = prices[len(prices)//2:]

        avg_first = np.mean(first_half)
        avg_second = np.mean(second_half)

        if avg_second > avg_first * 1.01:
            return 'bullish'
        elif avg_second < avg_first * 0.99:
            return 'bearish'
        else:
            return 'neutral'

    def _calculate_neckline(self, points: List[PatternPoint]) -> Optional[float]:
        """Расчет линии шеи для паттерна Голова и Плечи"""
        if len(points) < 4:
            return None

        # Ищем две точки для линии шеи (обычно точки 1 и 3 в 5-точечном паттерне)
        if len(points) >= 5:
            # Для Head and Shoulders: точки 1 и 4 (0-indexed: 0 и 3)
            neckline_points = [points[0], points[3]]
        elif len(points) >= 4:
            # Для Inverse Head and Shoulders: точки 1 и 3 (0-indexed: 0 и 2)
            neckline_points = [points[0], points[2]]
        else:
            return None

        # Линейная интерполяция
        x1, y1 = neckline_points[0].index, neckline_points[0].price
        x2, y2 = neckline_points[1].index, neckline_points[1].price

        if x2 == x1:
            return None

        # Уравнение линии: y = slope * x + intercept
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # Возвращаем цену на последнем баре паттерна
        last_idx = points[-1].index
        return slope * last_idx + intercept

    def create_result(self,
                     name: str,
                     points: List[PatternPoint],
                     quality: float,
                     confidence: float,
                     targets: Optional[Dict[str, float]] = None) -> PatternResult:
        """
        Создание результата паттерна

        Args:
            name: Название паттерна
            points: Точки паттерна
            quality: Качество паттерна
            confidence: Уверенность
            targets: Целевые уровни

        Returns:
            PatternResult
        """
        if targets is None:
            targets = {}

        direction = self._determine_direction(points)

        # Автогенерация стоп-лосса и тейк-профита если не заданы
        if 'stop_loss' not in targets or 'take_profit' not in targets:
            auto_targets = self.calculate_targets(points, name)
            targets.update(auto_targets)

        return PatternResult(
            name=name,
            direction=direction,
            points=points,
            quality=quality,
            confidence=confidence,
            targets=targets,
            metadata={
                'pattern_type': self.pattern_type,
                'detector': self.__class__.__name__
            }
        )

