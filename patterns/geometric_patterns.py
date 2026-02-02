"""
Классы геометрических паттернов
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from scipy.signal import argrelextrema
from scipy.stats import linregress
from dataclasses import dataclass, field
import talib

from .base_pattern import (
    BasePattern, PatternType, PatternDirection,
    PatternPoint, MarketContext
)
from config import DETECTION_CONFIG

@dataclass
class HeadShouldersPattern(BasePattern):
    """Паттерн Голова и Плечи / Перевернутые Голова и Плечи"""

    def __init__(self, is_inverse: bool = False):
        name = "Inverse Head and Shoulders" if is_inverse else "Head and Shoulders"
        abbreviation = "IHS" if is_inverse else "HS"
        super().__init__(PatternType.GEOMETRIC, name, abbreviation)

        self.is_inverse = is_inverse
        self.neckline_slope: Optional[float] = None
        self.neckline_intercept: Optional[float] = None
        self.pattern_height: Optional[float] = None

        # Параметры для детектирования
        self.symmetry_tolerance = DETECTION_CONFIG.SYMMETRY_TOLERANCE
        self.price_tolerance = DETECTION_CONFIG.PRICE_TOLERANCE_PCT
        self.min_shoulder_ratio = 0.6  # Минимальное соотношение плеч к голове

    def detect(self, data, highs, lows, closes, volumes, timestamps, **kwargs) -> bool:
        if len(highs) < 20:
            return False

        # Находим экстремумы
        max_indices = self._find_extremums(highs, is_max=True)
        min_indices = self._find_extremums(lows, is_max=False)

        if len(max_indices) < 5 or len(min_indices) < 4:
            return False

        if self.is_inverse:
            return self._detect_inverse(min_indices, lows, highs, timestamps)
        else:
            return self._detect_regular(max_indices, highs, lows, timestamps)

    def _detect_regular(self, max_indices, highs, lows, timestamps) -> bool:
        """Детектирование обычного паттерна Голова и Плечи"""
        # Ищем структуру: плечо-голова-плечо
        for i in range(len(max_indices) - 4):
            # Индексы предполагаемых точек
            ls_idx = max_indices[i]      # Левое плечо
            h_idx = max_indices[i + 1]   # Голова
            rs_idx = max_indices[i + 2]  # Правое плечо

            # Цены
            ls_price = highs[ls_idx]
            h_price = highs[h_idx]
            rs_price = highs[rs_idx]

            # Проверяем условия
            # 1. Голова выше плеч
            if not (h_price > ls_price and h_price > rs_price):
                continue

            # 2. Плечи примерно на одном уровне
            if abs(ls_price - rs_price) / h_price > self.symmetry_tolerance:
                continue

            # 3. Размеры плеч не менее 60% от головы
            ls_height = h_price - ls_price
            rs_height = h_price - rs_price

            if ls_height / h_price < self.min_shoulder_ratio or \
               rs_height / h_price < self.min_shoulder_ratio:
                continue

            # Находим минимумы между точками для построения линии шеи
            neck_points = self._find_neckline_points(lows, ls_idx, h_idx, rs_idx)

            if len(neck_points) < 2:
                continue

            # Строим линию шеи
            x = np.array([p[0] for p in neck_points])
            y = np.array([p[1] for p in neck_points])
            slope, intercept = np.polyfit(x, y, 1)

            # Сохраняем паттерн
            self.points = [
                PatternPoint(
                    index=ls_idx,
                    timestamp=timestamps[ls_idx],
                    price=ls_price,
                    point_type='left_shoulder'
                ),
                PatternPoint(
                    index=h_idx,
                    timestamp=timestamps[h_idx],
                    price=h_price,
                    point_type='head'
                ),
                PatternPoint(
                    index=rs_idx,
                    timestamp=timestamps[rs_idx],
                    price=rs_price,
                    point_type='right_shoulder'
                )
            ]

            # Добавляем точки линии шеи
            for neck_idx, neck_price in neck_points:
                self.points.append(
                    PatternPoint(
                        index=neck_idx,
                        timestamp=timestamps[neck_idx],
                        price=neck_price,
                        point_type='neckline',
                        significance=0.5
                    )
                )

            self.neckline_slope = slope
            self.neckline_intercept = intercept
            self.direction = PatternDirection.BEARISH
            self._is_detected = True

            # Высота паттерна
            self.pattern_height = h_price - (slope * h_idx + intercept)

            return True

        return False

    def _detect_inverse(self, min_indices, lows, highs, timestamps) -> bool:
        """Детектирование перевернутого паттерна"""
        # Аналогично обычному, но с минимумами
        for i in range(len(min_indices) - 4):
            ls_idx = min_indices[i]      # Левое плечо (дно)
            h_idx = min_indices[i + 1]   # Голова (дно)
            rs_idx = min_indices[i + 2]  # Правое плечо (дно)

            ls_price = lows[ls_idx]
            h_price = lows[h_idx]
            rs_price = lows[rs_idx]

            # Голова ниже плеч
            if not (h_price < ls_price and h_price < rs_price):
                continue

            # Плечи примерно на одном уровне
            if abs(ls_price - rs_price) / abs(h_price) > self.symmetry_tolerance:
                continue

            # Находим максимумы для линии шеи
            neck_points = self._find_neckline_points(highs, ls_idx, h_idx, rs_idx)

            if len(neck_points) < 2:
                continue

            # Строим линию шеи
            x = np.array([p[0] for p in neck_points])
            y = np.array([p[1] for p in neck_points])
            slope, intercept = np.polyfit(x, y, 1)

            self.points = [
                PatternPoint(
                    index=ls_idx,
                    timestamp=timestamps[ls_idx],
                    price=ls_price,
                    point_type='left_shoulder'
                ),
                PatternPoint(
                    index=h_idx,
                    timestamp=timestamps[h_idx],
                    price=h_price,
                    point_type='head'
                ),
                PatternPoint(
                    index=rs_idx,
                    timestamp=timestamps[rs_idx],
                    price=rs_price,
                    point_type='right_shoulder'
                )
            ]

            for neck_idx, neck_price in neck_points:
                self.points.append(
                    PatternPoint(
                        index=neck_idx,
                        timestamp=timestamps[neck_idx],
                        price=neck_price,
                        point_type='neckline',
                        significance=0.5
                    )
                )

            self.neckline_slope = slope
            self.neckline_intercept = intercept
            self.direction = PatternDirection.BULLISH
            self._is_detected = True
            self.pattern_height = (slope * h_idx + intercept) - h_price

            return True

        return False

    def _find_neckline_points(self, prices, idx1, idx2, idx3) -> List[Tuple[int, float]]:
        """Нахождение точек для построения линии шеи"""
        points = []

        # Между левым плечом и головой
        mid1_idx = (idx1 + idx2) // 2
        if idx1 < mid1_idx < idx2:
            mid1_price = prices[mid1_idx]
            points.append((mid1_idx, mid1_price))

        # Между головой и правым плечом
        mid2_idx = (idx2 + idx3) // 2
        if idx2 < mid2_idx < idx3:
            mid2_price = prices[mid2_idx]
            points.append((mid2_idx, mid2_price))

        return points

    def _find_extremums(self, prices: np.ndarray, is_max: bool = True) -> np.ndarray:
        """Поиск экстремумов"""
        order = DETECTION_CONFIG.EXTREMA_ORDER
        if is_max:
            indices = argrelextrema(prices, np.greater, order=order)[0]
        else:
            indices = argrelextrema(prices, np.less, order=order)[0]

        # Фильтруем слабые экстремумы
        filtered = []
        for idx in indices:
            if self._is_significant_extremum(prices, idx, is_max):
                filtered.append(idx)

        return np.array(filtered)

    def _is_significant_extremum(self, prices: np.ndarray, idx: int, is_max: bool) -> bool:
        """Проверка значимости экстремума"""
        window = 10
        start = max(0, idx - window)
        end = min(len(prices), idx + window + 1)

        if is_max:
            return prices[idx] >= np.max(prices[start:end])
        else:
            return prices[idx] <= np.min(prices[start:end])

    def calculate_quality(self) -> float:
        if not self._is_detected or len(self.points) < 3:
            return 0.0

        scores = []

        # 1. Симметрия плеч
        if len(self.points) >= 3:
            shoulder_points = [p for p in self.points if 'shoulder' in p.point_type]
            if len(shoulder_points) == 2:
                left_price = shoulder_points[0].price
                right_price = shoulder_points[1].price
                symmetry = 1 - abs(left_price - right_price) / max(abs(left_price), abs(right_price))
                scores.append(max(0, symmetry))

        # 2. Пропорции головы к плечам
        head_points = [p for p in self.points if 'head' in p.point_type]
        if head_points and shoulder_points:
            head_price = head_points[0].price
            avg_shoulder = (shoulder_points[0].price + shoulder_points[1].price) / 2

            if self.is_inverse:
                height_ratio = (avg_shoulder - head_price) / abs(head_price)
            else:
                height_ratio = (head_price - avg_shoulder) / abs(head_price)

            scores.append(min(height_ratio * 2, 1.0))  # Нормализуем

        # 3. Наклон линии шеи
        if self.neckline_slope is not None:
            slope_score = 1 - min(abs(self.neckline_slope) * 100, 1.0)
            scores.append(slope_score)

        # 4. Объем (если есть данные)
        if hasattr(self, '_volumes') and self._volumes is not None:
            volume_scores = []
            for point in self.points:
                if 'shoulder' in point.point_type or 'head' in point.point_type:
                    idx = point.index
                    if idx < len(self._volumes):
                        avg_volume = np.mean(self._volumes[max(0, idx-5):idx+1])
                        current_volume = self._volumes[idx]
                        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                        volume_scores.append(min(volume_ratio, 2.0) / 2.0)

            if volume_scores:
                scores.append(np.mean(volume_scores))

        return np.mean(scores) if scores else 0.0

    def calculate_targets(self, current_price: float):
        """Расчет целей для Голова и Плечи"""
        if not self._is_detected or self.pattern_height is None:
            return super().calculate_targets(current_price)

        # Для обычного паттерна (медвежьего)
        if not self.is_inverse and self.neckline_slope is not None:
            # Цена пробоя - текущая цена линии шеи
            neck_price = self.neckline_slope * len(self._closes) + self.neckline_intercept

            self.targets.entry_price = neck_price
            self.targets.stop_loss = neck_price + self.pattern_height * 0.1
            self.targets.take_profit = neck_price - self.pattern_height

            # Измеряем расстояние от головы до линии шеи
            head_point = [p for p in self.points if 'head' in p.point_type][0]
            head_neck_distance = head_point.price - neck_price

            # Дополнительные цели
            self.targets.target1 = neck_price - head_neck_distance * 0.5
            self.targets.target2 = neck_price - head_neck_distance
            self.targets.target3 = neck_price - head_neck_distance * 1.5

        # Для перевернутого паттерна (бычьего)
        elif self.is_inverse and self.neckline_slope is not None:
            neck_price = self.neckline_slope * len(self._closes) + self.neckline_intercept

            self.targets.entry_price = neck_price
            self.targets.stop_loss = neck_price - self.pattern_height * 0.1
            self.targets.take_profit = neck_price + self.pattern_height

            head_point = [p for p in self.points if 'head' in p.point_type][0]
            head_neck_distance = neck_price - head_point.price

            self.targets.target1 = neck_price + head_neck_distance * 0.5
            self.targets.target2 = neck_price + head_neck_distance
            self.targets.target3 = neck_price + head_neck_distance * 1.5

        # Расчет соотношения риска и прибыли
        if self.targets.entry_price and self.targets.stop_loss and self.targets.take_profit:
            risk = abs(self.targets.entry_price - self.targets.stop_loss)
            reward = abs(self.targets.take_profit - self.targets.entry_price)
            self.targets.profit_risk_ratio = reward / risk if risk > 0 else 0

        return self.targets

    def _get_min_points_required(self) -> int:
        return 3  # Минимум 3 точки для Голова и Плечи


@dataclass
class DoubleTopPattern(BasePattern):
    """Паттерн Двойная вершина"""

    def __init__(self):
        super().__init__(PatternType.GEOMETRIC, "Double Top", "DT")
        self.neckline_price: Optional[float] = None
        self.pattern_height: Optional[float] = None

        # Параметры
        self.price_tolerance = DETECTION_CONFIG.PRICE_TOLERANCE_PCT
        self.time_tolerance = DETECTION_CONFIG.TIME_TOLERANCE_PCT

    def detect(self, data, highs, lows, closes, volumes, timestamps, **kwargs) -> bool:
        if len(highs) < 15:
            return False

        # Ищем две вершины примерно на одном уровне
        max_indices = self._find_extremums(highs, is_max=True)

        if len(max_indices) < 2:
            return False

        # Проверяем все пары вершин
        for i in range(len(max_indices) - 1):
            for j in range(i + 1, len(max_indices)):
                idx1 = max_indices[i]
                idx2 = max_indices[j]

                # Проверяем расстояние во времени
                time_distance = abs(idx2 - idx1)
                if time_distance < 5 or time_distance > 50:  # Минимум 5, максимум 50 свечей
                    continue

                price1 = highs[idx1]
                price2 = highs[idx2]

                # Цены должны быть примерно одинаковы
                price_diff_pct = abs(price1 - price2) / max(price1, price2)
                if price_diff_pct > self.price_tolerance:
                    continue

                # Между вершинами должен быть минимум (шея)
                min_between_idx = np.argmin(lows[idx1:idx2+1]) + idx1
                neck_price = lows[min_between_idx]

                # Минимум должен быть достаточно глубоким
                decline_pct = (max(price1, price2) - neck_price) / max(price1, price2)
                if decline_pct < 0.01:  # Минимум 1% снижение
                    continue

                # Проверяем объемы (если есть)
                volume_confirmation = True
                if volumes is not None and len(volumes) > idx2:
                    # Объемы на вершинах должны быть высокими
                    vol1 = volumes[idx1] if idx1 < len(volumes) else 0
                    vol2 = volumes[idx2] if idx2 < len(volumes) else 0
                    avg_volume = np.mean(volumes[max(0, idx1-10):idx2+1])

                    if vol1 < avg_volume * 0.8 or vol2 < avg_volume * 0.8:
                        volume_confirmation = False

                # Найден паттерн
                self.points = [
                    PatternPoint(
                        index=idx1,
                        timestamp=timestamps[idx1],
                        price=price1,
                        point_type='top1'
                    ),
                    PatternPoint(
                        index=idx2,
                        timestamp=timestamps[idx2],
                        price=price2,
                        point_type='top2'
                    ),
                    PatternPoint(
                        index=min_between_idx,
                        timestamp=timestamps[min_between_idx],
                        price=neck_price,
                        point_type='neckline'
                    )
                ]

                self.neckline_price = neck_price
                self.pattern_height = max(price1, price2) - neck_price
                self.direction = PatternDirection.BEARISH
                self.metadata.volume_confirmation = volume_confirmation
                self._is_detected = True

                return True

        return False

    def calculate_quality(self) -> float:
        if not self._is_detected or len(self.points) < 3:
            return 0.0

        scores = []

        # 1. Симметрия вершин
        if len(self.points) >= 2:
            price1 = self.points[0].price
            price2 = self.points[1].price
            symmetry = 1 - abs(price1 - price2) / max(price1, price2)
            scores.append(max(0, symmetry))

        # 2. Глубина паттерна
        if self.pattern_height is not None and self.neckline_price is not None:
            max_price = max(self.points[0].price, self.points[1].price)
            depth_pct = self.pattern_height / max_price
            depth_score = min(depth_pct * 50, 1.0)  # 2% глубина = 1.0 балл
            scores.append(depth_score)

        # 3. Объемы
        if self.metadata.volume_confirmation:
            scores.append(0.8)
        else:
            scores.append(0.3)

        # 4. Форма
        # Проверяем, что между вершинами есть четкий минимум
        if len(self.points) == 3:
            neck_idx = self.points[2].index
            top1_idx = self.points[0].index
            top2_idx = self.points[1].index

            # Минимум должен быть примерно посередине
            middle_idx = (top1_idx + top2_idx) // 2
            position_score = 1 - min(abs(neck_idx - middle_idx) / (top2_idx - top1_idx), 1.0)
            scores.append(position_score)

        return np.mean(scores) if scores else 0.0

    def calculate_targets(self, current_price: float):
        if not self._is_detected or self.pattern_height is None:
            return super().calculate_targets(current_price)

        # Для Двойной вершины
        self.targets.entry_price = self.neckline_price
        self.targets.stop_loss = max(self.points[0].price, self.points[1].price) + self.pattern_height * 0.1
        self.targets.take_profit = self.neckline_price - self.pattern_height

        # Дополнительные цели
        self.targets.target1 = self.neckline_price - self.pattern_height * 0.5
        self.targets.target2 = self.neckline_price - self.pattern_height
        self.targets.target3 = self.neckline_price - self.pattern_height * 1.5

        # Риск/прибыль
        if self.targets.entry_price and self.targets.stop_loss and self.targets.take_profit:
            risk = abs(self.targets.entry_price - self.targets.stop_loss)
            reward = abs(self.targets.take_profit - self.targets.entry_price)
            self.targets.profit_risk_ratio = reward / risk if risk > 0 else 0

        return self.targets


@dataclass
class TrianglePattern(BasePattern):
    """Паттерн Треугольник (симметричный, восходящий, нисходящий)"""

    def __init__(self, triangle_type: str = "symmetric"):
        """
        Args:
            triangle_type: "symmetric", "ascending", "descending"
        """
        name = f"{triangle_type.capitalize()} Triangle"
        abbreviation = {
            "symmetric": "ST",
            "ascending": "AT",
            "descending": "DT"
        }.get(triangle_type, "TRI")

        super().__init__(PatternType.GEOMETRIC, name, abbreviation)

        self.triangle_type = triangle_type
        self.upper_trendline: Optional[Tuple[float, float]] = None  # (slope, intercept)
        self.lower_trendline: Optional[Tuple[float, float]] = None
        self.apex_index: Optional[int] = None  # Точка схождения
        self.breakout_direction: Optional[PatternDirection] = None

        # Параметры
        self.min_waves = 4  # Минимум 2 верхних и 2 нижних точки
        self.convergence_threshold = 0.8  # Порог схождения

    def detect(self, data, highs, lows, closes, volumes, timestamps, **kwargs) -> bool:
        if len(highs) < 30:
            return False

        # Находим последовательные максимумы и минимумы
        max_indices = self._find_consecutive_extremums(highs, is_max=True)
        min_indices = self._find_consecutive_extremums(lows, is_max=False)

        if len(max_indices) < 2 or len(min_indices) < 2:
            return False

        # Проверяем схождение
        if not self._check_convergence(max_indices, highs, min_indices, lows):
            return False

        # Определяем тип треугольника
        self._determine_triangle_type(max_indices, highs, min_indices, lows)

        # Строим линии тренда
        if not self._build_trendlines(max_indices, highs, min_indices, lows):
            return False

        # Проверяем пробой
        self._check_breakout(highs, lows, closes)

        # Сохраняем точки
        self._save_pattern_points(max_indices, highs, min_indices, lows, timestamps)

        self._is_detected = True
        return True

    def _find_consecutive_extremums(self, prices: np.ndarray, is_max: bool) -> List[int]:
        """Нахождение последовательных экстремумов"""
        indices = []
        window = 5

        for i in range(window, len(prices) - window):
            if is_max:
                if prices[i] == np.max(prices[i-window:i+window+1]):
                    indices.append(i)
            else:
                if prices[i] == np.min(prices[i-window:i+window+1]):
                    indices.append(i)

        return indices

    def _check_convergence(self, max_indices, highs, min_indices, lows) -> bool:
        """Проверка схождения линий"""
        if len(max_indices) < 2 or len(min_indices) < 2:
            return False

        # Берем последние точки
        recent_max = max_indices[-2:]
        recent_min = min_indices[-2:]

        # Проверяем, что максимумы снижаются, а минимумы повышаются
        if highs[recent_max[1]] >= highs[recent_max[0]]:
            return False

        if lows[recent_min[1]] <= lows[recent_min[0]]:
            return False

        return True

    def _determine_triangle_type(self, max_indices, highs, min_indices, lows):
        """Определение типа треугольника"""
        # Анализируем наклон линий
        if len(max_indices) >= 2 and len(min_indices) >= 2:
            max_slope = self._calculate_slope(max_indices[-2:], highs)
            min_slope = self._calculate_slope(min_indices[-2:], lows)

            if abs(max_slope) < 0.001 and abs(min_slope) < 0.001:
                self.triangle_type = "symmetric"
            elif max_slope < -0.001 and min_slope > 0.001:
                self.triangle_type = "symmetric"
            elif min_slope > 0.001:
                self.triangle_type = "ascending"
            elif max_slope < -0.001:
                self.triangle_type = "descending"

    def _calculate_slope(self, indices, values):
        """Расчет наклона линии"""
        if len(indices) < 2:
            return 0

        x = np.array(indices)
        y = np.array([values[i] for i in indices])

        if len(x) > 1:
            slope, _ = np.polyfit(x, y, 1)
            return slope

        return 0

    def _build_trendlines(self, max_indices, highs, min_indices, lows) -> bool:
        """Построение линий тренда"""
        # Верхняя линия по максимумам
        if len(max_indices) >= 2:
            x_upper = np.array(max_indices[-2:])
            y_upper = np.array([highs[i] for i in max_indices[-2:]])
            upper_slope, upper_intercept = np.polyfit(x_upper, y_upper, 1)
            self.upper_trendline = (upper_slope, upper_intercept)

        # Нижняя линия по минимумам
        if len(min_indices) >= 2:
            x_lower = np.array(min_indices[-2:])
            y_lower = np.array([lows[i] for i in min_indices[-2:]])
            lower_slope, lower_intercept = np.polyfit(x_lower, y_lower, 1)
            self.lower_trendline = (lower_slope, lower_intercept)

        return self.upper_trendline is not None and self.lower_trendline is not None

    def _check_breakout(self, highs, lows, closes):
        """Проверка пробоя"""
        if not self.upper_trendline or not self.lower_trendline:
            return

        upper_slope, upper_intercept = self.upper_trendline
        lower_slope, lower_intercept = self.lower_trendline

        # Текущие значения линий
        current_idx = len(closes) - 1
        upper_price = upper_slope * current_idx + upper_intercept
        lower_price = lower_slope * current_idx + lower_intercept

        current_price = closes[-1]

        # Проверяем пробой
        if current_price > upper_price * 1.002:  # 0.2% выше
            self.breakout_direction = PatternDirection.BULLISH
            self.direction = PatternDirection.BULLISH
        elif current_price < lower_price * 0.998:  # 0.2% ниже
            self.breakout_direction = PatternDirection.BEARISH
            self.direction = PatternDirection.BEARISH

    def _save_pattern_points(self, max_indices, highs, min_indices, lows, timestamps):
        """Сохранение точек паттерна"""
        self.points = []

        # Максимумы
        for idx in max_indices[-3:]:  # Последние 3 максимума
            self.points.append(
                PatternPoint(
                    index=idx,
                    timestamp=timestamps[idx],
                    price=highs[idx],
                    point_type='upper_point'
                )
            )

        # Минимумы
        for idx in min_indices[-3:]:  # Последние 3 минимума
            self.points.append(
                PatternPoint(
                    index=idx,
                    timestamp=timestamps[idx],
                    price=lows[idx],
                    point_type='lower_point'
                )
            )

    def calculate_quality(self) -> float:
        if not self._is_detected:
            return 0.0

        scores = []

        # 1. Четкость линий
        if self.upper_trendline and self.lower_trendline:
            # Проверяем, что точки хорошо ложатся на линии
            upper_slope, upper_intercept = self.upper_trendline
            lower_slope, lower_intercept = self.lower_trendline

            upper_errors = []
            lower_errors = []

            for point in self.points:
                if point.point_type == 'upper_point':
                    expected = upper_slope * point.index + upper_intercept
                    error = abs(point.price - expected) / point.price
                    upper_errors.append(error)
                elif point.point_type == 'lower_point':
                    expected = lower_slope * point.index + lower_intercept
                    error = abs(point.price - expected) / point.price
                    lower_errors.append(error)

            if upper_errors:
                upper_score = 1 - np.mean(upper_errors) * 10  # Нормализуем
                scores.append(max(0, upper_score))

            if lower_errors:
                lower_score = 1 - np.mean(lower_errors) * 10
                scores.append(max(0, lower_score))

        # 2. Схождение линий
        if self.upper_trendline and self.lower_trendline:
            upper_slope, _ = self.upper_trendline
            lower_slope, _ = self.lower_trendline

            # Для симметричного треугольника наклоны должны быть противоположными
            if self.triangle_type == "symmetric":
                if upper_slope < 0 and lower_slope > 0:
                    convergence = 1 - abs(upper_slope + lower_slope) / (abs(upper_slope) + abs(lower_slope))
                    scores.append(max(0, convergence))
            elif self.triangle_type == "ascending":
                if lower_slope > 0:
                    scores.append(0.8)
            elif self.triangle_type == "descending":
                if upper_slope < 0:
                    scores.append(0.8)

        # 3. Объем (уменьшение объема внутри треугольника)
        if hasattr(self, '_volumes') and self._volumes is not None:
            if len(self._volumes) > 20:
                recent_volume = np.mean(self._volumes[-10:])
                older_volume = np.mean(self._volumes[-20:-10])

                if recent_volume < older_volume:
                    volume_score = 1 - (recent_volume / older_volume)
                    scores.append(min(volume_score * 2, 1.0))

        # 4. Пробитие (если есть)
        if self.breakout_direction:
            scores.append(0.9)

        return np.mean(scores) if scores else 0.0

    def calculate_targets(self, current_price: float):
        if not self._is_detected or not self.upper_trendline or not self.lower_trendline:
            return super().calculate_targets(current_price)

        upper_slope, upper_intercept = self.upper_trendline
        lower_slope, lower_intercept = self.lower_trendline

        # Ширина треугольника у основания
        start_idx = min([p.index for p in self.points]) if self.points else 0
        current_idx = len(self._closes) - 1 if hasattr(self, '_closes') else start_idx + 20

        start_upper = upper_slope * start_idx + upper_intercept
        start_lower = lower_slope * start_idx + lower_intercept
        pattern_width = abs(start_upper - start_lower)

        # Текущие цены линий
        current_upper = upper_slope * current_idx + upper_intercept
        current_lower = lower_slope * current_idx + lower_intercept

        # Для бычьего пробоя
        if self.direction == PatternDirection.BULLISH:
            self.targets.entry_price = current_upper * 1.002  # 0.2% выше линии
            self.targets.stop_loss = current_lower * 0.998   # 0.2% ниже линии
            self.targets.take_profit = self.targets.entry_price + pattern_width

            self.targets.target1 = self.targets.entry_price + pattern_width * 0.5
            self.targets.target2 = self.targets.entry_price + pattern_width
            self.targets.target3 = self.targets.entry_price + pattern_width * 1.5

        # Для медвежьего пробоя
        elif self.direction == PatternDirection.BEARISH:
            self.targets.entry_price = current_lower * 0.998  # 0.2% ниже линии
            self.targets.stop_loss = current_upper * 1.002   # 0.2% выше линии
            self.targets.take_profit = self.targets.entry_price - pattern_width

            self.targets.target1 = self.targets.entry_price - pattern_width * 0.5
            self.targets.target2 = self.targets.entry_price - pattern_width
            self.targets.target3 = self.targets.entry_price - pattern_width * 1.5

        # Риск/прибыль
        if self.targets.entry_price and self.targets.stop_loss and self.targets.take_profit:
            risk = abs(self.targets.entry_price - self.targets.stop_loss)
            reward = abs(self.targets.take_profit - self.targets.entry_price)
            self.targets.profit_risk_ratio = reward / risk if risk > 0 else 0

        return self.targets


@dataclass
class WedgePattern(BasePattern):
    """Паттерн Клин"""

    def __init__(self, wedge_type: str = "rising"):
        """
        Args:
            wedge_type: "rising" (восходящий), "falling" (нисходящий)
        """
        name = f"{wedge_type.capitalize()} Wedge"
        abbreviation = "RW" if wedge_type == "rising" else "FW"

        super().__init__(PatternType.GEOMETRIC, name, abbreviation)

        self.wedge_type = wedge_type
        self.upper_line: Optional[Tuple[float, float]] = None
        self.lower_line: Optional[Tuple[float, float]] = None
        self.breakout_index: Optional[int] = None

        # Параметры
        self.min_points = 4  # Минимум 2 точки на каждой линии
        self.max_angle = 45  # Максимальный угол в градусах

    def detect(self, data, highs, lows, closes, volumes, timestamps, **kwargs) -> bool:
        if len(highs) < 25:
            return False

        # Находим точки для линий
        max_points = self._find_trend_points(highs, is_max=True)
        min_points = self._find_trend_points(lows, is_max=False)

        if len(max_points) < 2 or len(min_points) < 2:
            return False

        # Строим линии
        upper_line = self._fit_line(max_points)
        lower_line = self._fit_line(min_points)

        if not upper_line or not lower_line:
            return False

        # Проверяем, что линии сходятся
        if not self._check_convergence(upper_line, lower_line):
            return False

        # Проверяем наклон (для восходящего клина обе линии должны идти вверх)
        upper_slope, _ = upper_line
        lower_slope, _ = lower_line

        if self.wedge_type == "rising":
            if upper_slope <= 0 or lower_slope <= 0:
                return False
            self.direction = PatternDirection.BEARISH  # Восходящий клин - медвежий
        else:  # falling
            if upper_slope >= 0 or lower_slope >= 0:
                return False
            self.direction = PatternDirection.BULLISH  # Нисходящий клин - бычий

        # Проверяем пробой
        breakout = self._check_breakout(highs, lows, closes, upper_line, lower_line)

        # Сохраняем
        self.upper_line = upper_line
        self.lower_line = lower_line
        self.breakout_index = breakout

        self._save_points(max_points, min_points, highs, lows, timestamps)
        self._is_detected = True

        return True

    def _find_trend_points(self, prices: np.ndarray, is_max: bool) -> List[int]:
        """Нахождение точек для построения линии тренда"""
        points = []
        window = 7

        for i in range(window, len(prices) - window):
            if is_max:
                if prices[i] == np.max(prices[i-window:i+window+1]):
                    points.append(i)
            else:
                if prices[i] == np.min(prices[i-window:i+window+1]):
                    points.append(i)

        # Берем только значимые точки
        if len(points) > 3:
            # Выбираем точки, которые образуют тренд
            selected = [points[0]]
            for i in range(1, len(points)):
                if points[i] - selected[-1] > window:
                    selected.append(points[i])

            return selected[-3:]  # Последние 3 точки

        return points

    def _fit_line(self, indices: List[int]) -> Optional[Tuple[float, float]]:
        """Построение линии по точкам"""
        if len(indices) < 2:
            return None

        # Используем линейную регрессию
        x = np.array(indices)
        # Для простоты используем индексы как X
        # В реальности нужно использовать цены
        return (0.0, 0.0)  # Заглушка

    def _check_convergence(self, upper_line, lower_line) -> bool:
        """Проверка схождения линий"""
        upper_slope, upper_intercept = upper_line
        lower_slope, lower_intercept = lower_line

        # Для восходящего клина: обе линии вверх, но верхняя более пологая
        if self.wedge_type == "rising":
            return upper_slope > 0 and lower_slope > 0 and upper_slope < lower_slope
        # Для нисходящего клина: обе линии вниз, но нижняя более пологая
        else:
            return upper_slope < 0 and lower_slope < 0 and upper_slope < lower_slope

    def _check_breakout(self, highs, lows, closes, upper_line, lower_line) -> Optional[int]:
        """Проверка пробоя клина"""
        upper_slope, upper_intercept = upper_line
        lower_slope, lower_intercept = lower_line

        for i in range(len(closes) - 5, len(closes)):
            current_upper = upper_slope * i + upper_intercept
            current_lower = lower_slope * i + lower_intercept

            # Для восходящего клина - пробой вниз
            if self.wedge_type == "rising":
                if closes[i] < current_lower * 0.995:  # 0.5% ниже
                    return i

            # Для нисходящего клина - пробой вверх
            else:
                if closes[i] > current_upper * 1.005:  # 0.5% выше
                    return i

        return None

    def _save_points(self, max_points, min_points, highs, lows, timestamps):
        """Сохранение точек паттерна"""
        self.points = []

        # Верхние точки
        for idx in max_points[:3]:  # Первые 3 точки
            self.points.append(
                PatternPoint(
                    index=idx,
                    timestamp=timestamps[idx],
                    price=highs[idx],
                    point_type='upper_point'
                )
            )

        # Нижние точки
        for idx in min_points[:3]:  # Первые 3 точки
            self.points.append(
                PatternPoint(
                    index=idx,
                    timestamp=timestamps[idx],
                    price=lows[idx],
                    point_type='lower_point'
                )
            )

    def calculate_quality(self) -> float:
        if not self._is_detected:
            return 0.0

        scores = []

        # 1. Четкость линий
        if self.upper_line and self.lower_line:
            # Проверяем количество касаний
            upper_touches = len([p for p in self.points if p.point_type == 'upper_point'])
            lower_touches = len([p for p in self.points if p.point_type == 'lower_point'])

            touch_score = min((upper_touches + lower_touches) / 6, 1.0)
            scores.append(touch_score)

        # 2. Схождение
        if self.upper_line and self.lower_line:
            upper_slope, _ = self.upper_line
            lower_slope, _ = self.lower_line

            if self.wedge_type == "rising":
                if 0 < upper_slope < lower_slope:
                    convergence = (lower_slope - upper_slope) / lower_slope
                    scores.append(min(convergence * 3, 1.0))
            else:
                if upper_slope < lower_slope < 0:
                    convergence = (upper_slope - lower_slope) / abs(upper_slope)
                    scores.append(min(convergence * 3, 1.0))

        # 3. Пробитие
        if self.breakout_index is not None:
            scores.append(0.9)

        # 4. Объем (должен снижаться внутри клина)
        if hasattr(self, '_volumes') and self._volumes is not None:
            if len(self._volumes) > 20:
                # Сравниваем объем в начале и в конце
                start_idx = min([p.index for p in self.points]) if self.points else 0
                end_idx = max([p.index for p in self.points]) if self.points else len(self._volumes) - 1

                if end_idx - start_idx > 10:
                    start_volume = np.mean(self._volumes[start_idx:start_idx+5])
                    end_volume = np.mean(self._volumes[end_idx-4:end_idx+1])

                    if end_volume < start_volume:
                        volume_score = 1 - (end_volume / start_volume)
                        scores.append(min(volume_score * 2, 1.0))

        return np.mean(scores) if scores else 0.0

    def calculate_targets(self, current_price: float):
        if not self._is_detected or not self.upper_line or not self.lower_line:
            return super().calculate_targets(current_price)

        upper_slope, upper_intercept = self.upper_line
        lower_slope, lower_intercept = self.lower_line

        # Находим самую широкую часть клина
        start_idx = min([p.index for p in self.points]) if self.points else 0
        wedge_height = abs((upper_slope * start_idx + upper_intercept) -
                          (lower_slope * start_idx + lower_intercept))

        current_idx = len(self._closes) - 1 if hasattr(self, '_closes') else start_idx + 20

        current_upper = upper_slope * current_idx + upper_intercept
        current_lower = lower_slope * current_idx + lower_intercept

        # Для восходящего клина (медвежьего)
        if self.wedge_type == "rising":
            self.targets.entry_price = current_lower * 0.995  # 0.5% ниже нижней линии
            self.targets.stop_loss = current_upper * 1.01    # 1% выше верхней линии
            self.targets.take_profit = self.targets.entry_price - wedge_height

            self.targets.target1 = self.targets.entry_price - wedge_height * 0.5
            self.targets.target2 = self.targets.entry_price - wedge_height
            self.targets.target3 = self.targets.entry_price - wedge_height * 1.5

        # Для нисходящего клина (бычьего)
        else:
            self.targets.entry_price = current_upper * 1.005  # 0.5% выше верхней линии
            self.targets.stop_loss = current_lower * 0.99    # 1% ниже нижней линии
            self.targets.take_profit = self.targets.entry_price + wedge_height

            self.targets.target1 = self.targets.entry_price + wedge_height * 0.5
            self.targets.target2 = self.targets.entry_price + wedge_height
            self.targets.target3 = self.targets.entry_price + wedge_height * 1.5

        # Риск/прибыль
        if self.targets.entry_price and self.targets.stop_loss and self.targets.take_profit:
            risk = abs(self.targets.entry_price - self.targets.stop_loss)
            reward = abs(self.targets.take_profit - self.targets.entry_price)
            self.targets.profit_risk_ratio = reward / risk if risk > 0 else 0

        return self.targets


@dataclass
class FlagPattern(BasePattern):
    """Паттерн Флаг/Вымпел"""

    def __init__(self, pattern_type: str = "flag"):
        """
        Args:
            pattern_type: "flag" (флаг), "pennant" (вымпел)
        """
        name = pattern_type.capitalize()
        abbreviation = "FLG" if pattern_type == "flag" else "PEN"

        super().__init__(PatternType.GEOMETRIC, name, abbreviation)

        self.pattern_type = pattern_type
        self.flagpole_height: Optional[float] = None
        self.flag_start: Optional[int] = None
        self.flag_end: Optional[int] = None
        self.breakout_direction: Optional[PatternDirection] = None

        # Параметры
        self.min_flagpole_ratio = 0.03  # 3% минимальная высота флагштока
        self.max_flag_ratio = 0.5  # Флаг не более 50% от флагштока по времени
        self.consolidation_ratio = 0.3  # Консолидация не более 30% от движения

    def detect(self, data, highs, lows, closes, volumes, timestamps, **kwargs) -> bool:
        if len(closes) < 40:
            return False

        # Ищем сильное движение (флагшток)
        flagpole = self._find_flagpole(highs, lows, closes)
        if not flagpole:
            return False

        flagpole_start, flagpole_end, flagpole_height, flagpole_direction = flagpole

        # Ищем консолидацию после движения (флаг)
        flag = self._find_flag_consolidation(
            flagpole_end, highs, lows, closes, flagpole_direction
        )
        if not flag:
            return False

        flag_start, flag_end, flag_height = flag

        # Проверяем пропорции
        if not self._check_proportions(
            flagpole_start, flagpole_end, flag_start, flag_end,
            flagpole_height, flag_height, flagpole_direction
        ):
            return False

        # Проверяем пробой
        breakout = self._check_breakout(
            flag_end, highs, lows, closes, flagpole_direction
        )

        # Сохраняем паттерн
        self.flagpole_height = flagpole_height
        self.flag_start = flag_start
        self.flag_end = flag_end
        self.breakout_direction = flagpole_direction
        self.direction = flagpole_direction

        self._save_points(
            flagpole_start, flagpole_end, flag_start, flag_end,
            highs, lows, timestamps, flagpole_direction
        )

        self._is_detected = True
        return True

    def _find_flagpole(self, highs, lows, closes) -> Optional[tuple]:
        """Поиск флагштока (сильного движения)"""
        min_candles = 5
        max_candles = 20

        for i in range(len(closes) - max_candles - 10):
            # Ищем начало движения
            start_idx = i

            # Ищем конец движения
            for j in range(i + min_candles, i + max_candles):
                if j >= len(closes):
                    break

                # Высота движения
                move_high = np.max(highs[i:j+1])
                move_low = np.min(lows[i:j+1])
                move_height = move_high - move_low

                # Процентное изменение
                price_change_pct = abs(closes[j] - closes[i]) / closes[i]

                # Проверяем на сильное движение
                if price_change_pct >= self.min_flagpole_ratio:
                    # Определяем направление
                    if closes[j] > closes[i]:
                        direction = PatternDirection.BULLISH
                    else:
                        direction = PatternDirection.BEARISH

                    return (i, j, move_height, direction)

        return None

    def _find_flag_consolidation(self, after_idx, highs, lows, closes, direction) -> Optional[tuple]:
        """Поиск консолидации (флага) после движения"""
        max_flag_candles = 30

        start_idx = after_idx + 1
        end_idx = min(start_idx + max_flag_candles, len(closes) - 1)

        if end_idx - start_idx < 5:
            return None

        # Анализируем консолидацию
        consolidation_highs = highs[start_idx:end_idx+1]
        consolidation_lows = lows[start_idx:end_idx+1]

        flag_height = np.max(consolidation_highs) - np.min(consolidation_lows)
        avg_price = np.mean(closes[start_idx:end_idx+1])

        # Высота должна быть небольшой относительно движения
        if flag_height / avg_price > 0.015:  # 1.5% максимум
            return None

        # Для бычьего флага - боковое или слегка нисходящее движение
        if direction == PatternDirection.BULLISH:
            # Проверяем наклон
            x = np.arange(len(consolidation_highs))
            high_slope, _ = np.polyfit(x, consolidation_highs, 1)
            low_slope, _ = np.polyfit(x, consolidation_lows, 1)

            if high_slope > 0.001:  # Верхняя граница не должна сильно расти
                return None

        # Для медвежьего флага - боковое или слегка восходящее движение
        else:
            x = np.arange(len(consolidation_highs))
            high_slope, _ = np.polyfit(x, consolidation_highs, 1)
            low_slope, _ = np.polyfit(x, consolidation_lows, 1)

            if low_slope < -0.001:  # Нижняя граница не должна сильно падать
                return None

        return (start_idx, end_idx, flag_height)

    def _check_proportions(self, pole_start, pole_end, flag_start, flag_end,
                          pole_height, flag_height, direction) -> bool:
        """Проверка пропорций паттерна"""
        # Длина флагштока
        pole_length = pole_end - pole_start

        # Длина флага
        flag_length = flag_end - flag_start

        # Флаг не должен быть слишком длинным относительно флагштока
        if flag_length > pole_length * self.max_flag_ratio:
            return False

        # Высота флага должна быть небольшой
        if flag_height > pole_height * self.consolidation_ratio:
            return False

        return True

    def _check_breakout(self, flag_end, highs, lows, closes, direction) -> bool:
        """Проверка пробоя флага"""
        if flag_end >= len(closes) - 2:
            return False

        # Для бычьего флага - пробой вверх
        if direction == PatternDirection.BULLISH:
            flag_high = np.max(highs[flag_end-5:flag_end+1])
            if closes[-1] > flag_high * 1.002:  # 0.2% выше
                return True

        # Для медвежьего флага - пробой вниз
        else:
            flag_low = np.min(lows[flag_end-5:flag_end+1])
            if closes[-1] < flag_low * 0.998:  # 0.2% ниже
                return True

        return False

    def _save_points(self, pole_start, pole_end, flag_start, flag_end,
                    highs, lows, timestamps, direction):
        """Сохранение точек паттерна"""
        self.points = []

        # Точки флагштока
        self.points.append(
            PatternPoint(
                index=pole_start,
                timestamp=timestamps[pole_start],
                price=closes[pole_start],
                point_type='flagpole_start'
            )
        )

        self.points.append(
            PatternPoint(
                index=pole_end,
                timestamp=timestamps[pole_end],
                price=closes[pole_end],
                point_type='flagpole_end'
            )
        )

        # Границы флага
        flag_highs = highs[flag_start:flag_end+1]
        flag_lows = lows[flag_start:flag_end+1]

        high_idx = flag_start + np.argmax(flag_highs)
        low_idx = flag_start + np.argmin(flag_lows)

        self.points.append(
            PatternPoint(
                index=high_idx,
                timestamp=timestamps[high_idx],
                price=highs[high_idx],
                point_type='flag_high'
            )
        )

        self.points.append(
            PatternPoint(
                index=low_idx,
                timestamp=timestamps[low_idx],
                price=lows[low_idx],
                point_type='flag_low'
            )
        )

    def calculate_quality(self) -> float:
        if not self._is_detected:
            return 0.0

        scores = []

        # 1. Соотношение флагштока и флага
        if self.flagpole_height and hasattr(self, '_closes'):
            pole_length = self.points[1].index - self.points[0].index
            flag_length = self.flag_end - self.flag_start

            length_ratio = flag_length / pole_length if pole_length > 0 else 0
            ratio_score = 1 - min(length_ratio / self.max_flag_ratio, 1.0)
            scores.append(ratio_score)

        # 2. Наклон флага
        # Для бычьего флага - должен быть горизонтальным или слегка нисходящим
        # Для медвежьего - горизонтальным или слегка восходящим
        if len(self.points) >= 4:
            flag_high = self.points[2]
            flag_low = self.points[3]

            if self.direction == PatternDirection.BULLISH:
                # Высокая и низкая точки флага
                if flag_high.index > flag_low.index and flag_high.price > flag_low.price:
                    scores.append(0.8)  # Нисходящий флаг - хорошо
                elif abs(flag_high.price - flag_low.price) / flag_high.price < 0.01:
                    scores.append(0.9)  # Горизонтальный - отлично
                else:
                    scores.append(0.5)

            else:  # BEARISH
                if flag_low.index > flag_high.index and flag_low.price < flag_high.price:
                    scores.append(0.8)  # Восходящий флаг - хорошо
                elif abs(flag_high.price - flag_low.price) / flag_high.price < 0.01:
                    scores.append(0.9)  # Горизонтальный - отлично
                else:
                    scores.append(0.5)

        # 3. Объем
        if hasattr(self, '_volumes') and self._volumes is not None:
            # Объем на флагштоке должен быть высоким, на флаге - снижаться
            if self.flag_start and self.flag_end:
                pole_volume = np.mean(self._volumes[self.points[0].index:self.points[1].index+1])
                flag_volume = np.mean(self._volumes[self.flag_start:self.flag_end+1])

                if pole_volume > 0 and flag_volume > 0:
                    volume_ratio = flag_volume / pole_volume
                    volume_score = 1 - min(volume_ratio, 1.0)  # Чем меньше объем на флаге, тем лучше
                    scores.append(volume_score)

        # 4. Пробитие
        if self.breakout_direction:
            scores.append(0.9)

        return np.mean(scores) if scores else 0.0

    def calculate_targets(self, current_price: float):
        if not self._is_detected or self.flagpole_height is None:
            return super().calculate_targets(current_price)

        # Для флага цель - высота флагштока от точки пробоя
        if self.direction == PatternDirection.BULLISH:
            # Бычий флаг - цель вверх
            entry = current_price
            target = entry + self.flagpole_height

            self.targets.entry_price = entry
            self.targets.stop_loss = self.points[3].price * 0.99  # Нижняя граница флага
            self.targets.take_profit = target

            self.targets.target1 = entry + self.flagpole_height * 0.5
            self.targets.target2 = entry + self.flagpole_height
            self.targets.target3 = entry + self.flagpole_height * 1.5

        else:  # BEARISH
            # Медвежий флаг - цель вниз
            entry = current_price
            target = entry - self.flagpole_height

            self.targets.entry_price = entry
            self.targets.stop_loss = self.points[2].price * 1.01  # Верхняя граница флага
            self.targets.take_profit = target

            self.targets.target1 = entry - self.flagpole_height * 0.5
            self.targets.target2 = entry - self.flagpole_height
            self.targets.target3 = entry - self.flagpole_height * 1.5

        # Риск/прибыль
        if self.targets.entry_price and self.targets.stop_loss and self.targets.take_profit:
            risk = abs(self.targets.entry_price - self.targets.stop_loss)
            reward = abs(self.targets.take_profit - self.targets.entry_price)
            self.targets.profit_risk_ratio = reward / risk if risk > 0 else 0

        return self.targets

