"""
Модуль геометрических паттернов
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from scipy.spatial import KDTree
from scipy.signal import find_peaks

from .base_pattern import BasePattern, PatternPoint, PatternResult
from config import config

@dataclass
class PricePoint:
    """Точка цены с индексом"""
    index: int
    price: float
    type: str  # 'high' или 'low'

class GeometricPatterns(BasePattern):
    """Класс для детектирования геометрических паттернов"""

    def __init__(self):
        super().__init__(name="geometric_patterns", min_points=4)

        # Паттерны которые мы будем искать
        self.patterns = {
            'head_shoulders': self._detect_head_shoulders,
            'inverse_head_shoulders': self._detect_inverse_head_shoulders,
            'double_top': self._detect_double_top,
            'double_bottom': self._detect_double_bottom,
            'triangle': self._detect_triangle,
            'wedge': self._detect_wedge,
            'rectangle': self._detect_rectangle,
            'flag': self._detect_flag,
            'pennant': self._detect_pennant
        }

        # Параметры для поиска экстремумов
        self.peak_prominence = 0.005  # Минимальная значимость экстремума
        self.peak_distance = 5  # Минимальное расстояние между экстремумами

    def detect(self, data: Dict[str, np.ndarray]) -> List[PatternResult]:
        """
        Детектирование геометрических паттернов

        Args:
            data: Входные данные OHLC

        Returns:
            Список обнаруженных паттернов
        """
        results = []

        # Проверка входных данных
        required = ['high', 'low']
        if not all(key in data for key in required):
            return results

        # Находим экстремумы
        highs, lows = self._find_extremes(data)

        if len(highs) < 2 or len(lows) < 2:
            return results

        # Создаем список всех экстремумов
        all_points = self._create_all_points(highs, lows)

        # Ищем каждый паттерн
        for pattern_name, detector in self.patterns.items():
            if config.DETECTION.ENABLE_GEOMETRIC:
                pattern_results = detector(all_points, highs, lows, data)
                results.extend(pattern_results)

        # Фильтрация результатов
        filtered_results = self._filter_results(results)

        return filtered_results

    def _find_extremes(self, data: Dict[str, np.ndarray]) -> Tuple[List[PricePoint], List[PricePoint]]:
        """Поиск экстремумов (пиков и впадин)"""
        highs = []
        lows = []

        try:
            # Используем scipy для поиска пиков
            high_prices = data['high']
            low_prices = data['low']

            # Находим локальные максимумы
            high_indices, high_properties = find_peaks(
                high_prices,
                prominence=self.peak_prominence * np.mean(high_prices),
                distance=self.peak_distance
            )

            # Находим локальные минимумы (инвертируем цены)
            low_indices, low_properties = find_peaks(
                -low_prices,
                prominence=self.peak_prominence * np.mean(low_prices),
                distance=self.peak_distance
            )

            # Создаем точки максимумов
            for idx in high_indices:
                if idx < len(high_prices):
                    point = PricePoint(
                        index=idx,
                        price=float(high_prices[idx]),
                        type='high'
                    )
                    highs.append(point)

            # Создаем точки минимумов
            for idx in low_indices:
                if idx < len(low_prices):
                    point = PricePoint(
                        index=idx,
                        price=float(low_prices[idx]),
                        type='low'
                    )
                    lows.append(point)

            # Сортируем по индексу
            highs.sort(key=lambda x: x.index)
            lows.sort(key=lambda x: x.index)

        except Exception as e:
            # Резервный метод если scipy не работает
            highs, lows = self._find_extremes_simple(data)

        return highs, lows

    def _find_extremes_simple(self, data: Dict[str, np.ndarray]) -> Tuple[List[PricePoint], List[PricePoint]]:
        """Простой поиск экстремумов (резервный метод)"""
        highs = []
        lows = []

        high_prices = data['high']
        low_prices = data['low']

        window = 5

        for i in range(window, len(high_prices) - window):
            # Проверяем максимум
            if high_prices[i] == np.max(high_prices[i-window:i+window+1]):
                point = PricePoint(
                    index=i,
                    price=float(high_prices[i]),
                    type='high'
                )
                highs.append(point)

            # Проверяем минимум
            if low_prices[i] == np.min(low_prices[i-window:i+window+1]):
                point = PricePoint(
                    index=i,
                    price=float(low_prices[i]),
                    type='low'
                )
                lows.append(point)

        # Удаляем слишком близкие экстремумы
        highs = self._filter_close_points(highs, min_distance=window)
        lows = self._filter_close_points(lows, min_distance=window)

        return highs, lows

    def _filter_close_points(self, points: List[PricePoint], min_distance: int = 5) -> List[PricePoint]:
        """Фильтрация слишком близких точек"""
        if not points:
            return points

        filtered = [points[0]]

        for i in range(1, len(points)):
            if points[i].index - filtered[-1].index >= min_distance:
                filtered.append(points[i])

        return filtered

    def _create_all_points(self, highs: List[PricePoint], lows: List[PricePoint]) -> List[PricePoint]:
        """Создание общего списка всех экстремумов"""
        all_points = highs + lows
        all_points.sort(key=lambda x: x.index)
        return all_points

    def _detect_head_shoulders(self, all_points: List[PricePoint],
                              highs: List[PricePoint],
                              lows: List[PricePoint],
                              data: Dict[str, np.ndarray]) -> List[PatternResult]:
        """Детектирование паттерна Голова и Плечи"""
        results = []

        # Нужно как минимум 5 точек: левое плечо, голова, правое плечо + 2 точки для линии шеи
        if len(highs) < 3 or len(lows) < 2:
            return results

        # Ищем последовательность: high-low-high-low-high
        for i in range(len(all_points) - 4):
            points = all_points[i:i+5]

            # Проверяем паттерн: H-L-H-L-H (типы точек)
            pattern_types = [p.type for p in points]
            expected_pattern = ['high', 'low', 'high', 'low', 'high']

            if pattern_types != expected_pattern:
                continue

            # Получаем цены
            left_shoulder = points[0].price
            neckline1 = points[1].price  # Первая точка линии шеи
            head = points[2].price
            neckline2 = points[3].price  # Вторая точка линии шеи
            right_shoulder = points[4].price

            # Голова должна быть выше плеч
            if not (head > left_shoulder and head > right_shoulder):
                continue

            # Плечи должны быть примерно на одном уровне
            shoulder_diff = abs(left_shoulder - right_shoulder) / head
            if shoulder_diff > config.DETECTION.SYMMETRY_TOLERANCE:
                continue

            # Линия шеи должна быть примерно горизонтальной
            neckline_slope = abs(neckline2 - neckline1) / (points[3].index - points[1].index)
            if neckline_slope > 0.001:  # Слишком наклонная линия шеи
                continue

            # Качество паттерна
            quality = self._calculate_head_shoulders_quality(
                left_shoulder, head, right_shoulder, neckline1, neckline2
            )

            if quality < config.DETECTION.MIN_PATTERN_QUALITY:
                continue

            # Создаем точки паттерна
            pattern_points = [
                PatternPoint(index=points[0].index, price=left_shoulder, type='left_shoulder'),
                PatternPoint(index=points[1].index, price=neckline1, type='neckline1'),
                PatternPoint(index=points[2].index, price=head, type='head'),
                PatternPoint(index=points[3].index, price=neckline2, type='neckline2'),
                PatternPoint(index=points[4].index, price=right_shoulder, type='right_shoulder')
            ]

            # Расчет целей
            neckline_price = (neckline1 + neckline2) / 2
            pattern_height = head - neckline_price
            entry_price = points[4].price  # Цена на правом плече

            targets = {
                'entry_price': entry_price,
                'neckline': neckline_price,
                'pattern_height': pattern_height,
                'stop_loss': head * 1.01,  # Стоп выше головы
                'take_profit': entry_price - pattern_height  # Цель равна высоте паттерна
            }

            # Создаем результат
            result = PatternResult(
                name='head_shoulders',
                direction='bearish',
                points=pattern_points,
                quality=quality,
                confidence=self.calculate_confidence(pattern_points, data),
                targets=targets,
                metadata={
                    'pattern_type': 'reversal',
                    'shoulder_diff': shoulder_diff,
                    'neckline_slope': neckline_slope
                }
            )

            results.append(result)

        return results

    def _detect_inverse_head_shoulders(self, all_points: List[PricePoint],
                                      highs: List[PricePoint],
                                      lows: List[PricePoint],
                                      data: Dict[str, np.ndarray]) -> List[PatternResult]:
        """Детектирование паттерна Перевернутая Голова и Плечи"""
        results = []

        # Нужно как минимум 5 точек: левое плечо, голова, правое плечо + 2 точки для линии шеи
        if len(lows) < 3 or len(highs) < 2:
            return results

        # Ищем последовательность: low-high-low-high-low
        for i in range(len(all_points) - 4):
            points = all_points[i:i+5]

            # Проверяем паттерн: L-H-L-H-L (типы точек)
            pattern_types = [p.type for p in points]
            expected_pattern = ['low', 'high', 'low', 'high', 'low']

            if pattern_types != expected_pattern:
                continue

            # Получаем цены
            left_shoulder = points[0].price
            neckline1 = points[1].price  # Первая точка линии шеи
            head = points[2].price
            neckline2 = points[3].price  # Вторая точка линии шеи
            right_shoulder = points[4].price

            # Голова должна быть ниже плеч
            if not (head < left_shoulder and head < right_shoulder):
                continue

            # Плечи должны быть примерно на одном уровне
            shoulder_diff = abs(left_shoulder - right_shoulder) / head
            if shoulder_diff > config.DETECTION.SYMMETRY_TOLERANCE:
                continue

            # Линия шеи должна быть примерно горизонтальной
            neckline_slope = abs(neckline2 - neckline1) / (points[3].index - points[1].index)
            if neckline_slope > 0.001:  # Слишком наклонная линия шеи
                continue

            # Качество паттерна
            quality = self._calculate_head_shoulders_quality(
                left_shoulder, head, right_shoulder, neckline1, neckline2, inverse=True
            )

            if quality < config.DETECTION.MIN_PATTERN_QUALITY:
                continue

            # Создаем точки паттерна
            pattern_points = [
                PatternPoint(index=points[0].index, price=left_shoulder, type='left_shoulder'),
                PatternPoint(index=points[1].index, price=neckline1, type='neckline1'),
                PatternPoint(index=points[2].index, price=head, type='head'),
                PatternPoint(index=points[3].index, price=neckline2, type='neckline2'),
                PatternPoint(index=points[4].index, price=right_shoulder, type='right_shoulder')
            ]

            # Расчет целей
            neckline_price = (neckline1 + neckline2) / 2
            pattern_height = neckline_price - head
            entry_price = points[4].price  # Цена на правом плече

            targets = {
                'entry_price': entry_price,
                'neckline': neckline_price,
                'pattern_height': pattern_height,
                'stop_loss': head * 0.99,  # Стоп ниже головы
                'take_profit': entry_price + pattern_height  # Цель равна высоте паттерна
            }

            # Создаем результат
            result = PatternResult(
                name='inverse_head_shoulders',
                direction='bullish',
                points=pattern_points,
                quality=quality,
                confidence=self.calculate_confidence(pattern_points, data),
                targets=targets,
                metadata={
                    'pattern_type': 'reversal',
                    'shoulder_diff': shoulder_diff,
                    'neckline_slope': neckline_slope
                }
            )

            results.append(result)

        return results

    def _calculate_head_shoulders_quality(self,
                                         left_shoulder: float,
                                         head: float,
                                         right_shoulder: float,
                                         neckline1: float,
                                         neckline2: float,
                                         inverse: bool = False) -> float:
        """Расчет качества паттерна Голова и Плечи"""
        # 1. Симметрия плеч
        shoulder_diff = abs(left_shoulder - right_shoulder)
        avg_shoulder = (left_shoulder + right_shoulder) / 2

        if inverse:
            # Для перевернутого паттерна
            head_to_shoulder = avg_shoulder - head
        else:
            # Для обычного паттерна
            head_to_shoulder = head - avg_shoulder

        if head_to_shoulder <= 0:
            return 0.0

        symmetry_score = 1.0 - (shoulder_diff / head_to_shoulder)
        symmetry_score = max(0.0, min(1.0, symmetry_score))

        # 2. Глубина головы
        if inverse:
            depth_score = (head_to_shoulder) / avg_shoulder
        else:
            depth_score = (head_to_shoulder) / avg_shoulder

        depth_score = min(1.0, depth_score * 10)  # Нормализация

        # 3. Горизонтальность линии шеи
        neckline_diff = abs(neckline2 - neckline1)
        avg_neckline = (neckline1 + neckline2) / 2

        if avg_neckline > 0:
            neckline_score = 1.0 - (neckline_diff / avg_neckline)
        else:
            neckline_score = 1.0

        neckline_score = max(0.0, min(1.0, neckline_score))

        # Итоговое качество
        quality = (symmetry_score * 0.4 + depth_score * 0.3 + neckline_score * 0.3)
        return quality

    def _detect_double_top(self, all_points: List[PricePoint],
                          highs: List[PricePoint],
                          lows: List[PricePoint],
                          data: Dict[str, np.ndarray]) -> List[PatternResult]:
        """Детектирование Двойной вершины"""
        results = []

        if len(highs) < 2 or len(lows) < 1:
            return results

        # Ищем последовательность: high-low-high
        for i in range(len(all_points) - 2):
            points = all_points[i:i+3]

            # Проверяем паттерн: H-L-H (типы точек)
            pattern_types = [p.type for p in points]
            expected_pattern = ['high', 'low', 'high']

            if pattern_types != expected_pattern:
                continue

            # Получаем цены
            top1 = points[0].price
            bottom = points[1].price  # Дно между вершинами
            top2 = points[2].price

            # Вершины должны быть примерно на одном уровне
            top_diff = abs(top1 - top2) / ((top1 + top2) / 2)
            if top_diff > config.DETECTION.PRICE_TOLERANCE_PCT:
                continue

            # Дно должно быть значительно ниже вершин
            if not (bottom < top1 * 0.98 and bottom < top2 * 0.98):
                continue

            # Качество паттерна
            quality = self._calculate_double_pattern_quality(top1, top2, bottom, is_top=True)

            if quality < config.DETECTION.MIN_PATTERN_QUALITY:
                continue

            # Создаем точки паттерна
            pattern_points = [
                PatternPoint(index=points[0].index, price=top1, type='top1'),
                PatternPoint(index=points[1].index, price=bottom, type='bottom'),
                PatternPoint(index=points[2].index, price=top2, type='top2')
            ]

            # Расчет целей
            entry_price = points[2].price
            pattern_height = ((top1 + top2) / 2) - bottom

            targets = {
                'entry_price': entry_price,
                'neckline': bottom,
                'pattern_height': pattern_height,
                'stop_loss': max(top1, top2) * 1.01,
                'take_profit': entry_price - pattern_height
            }

            # Создаем результат
            result = PatternResult(
                name='double_top',
                direction='bearish',
                points=pattern_points,
                quality=quality,
                confidence=self.calculate_confidence(pattern_points, data),
                targets=targets,
                metadata={
                    'pattern_type': 'reversal',
                    'top_diff_pct': top_diff,
                    'depth_pct': (top1 - bottom) / top1
                }
            )

            results.append(result)

        return results

    def _detect_double_bottom(self, all_points: List[PricePoint],
                             highs: List[PricePoint],
                             lows: List[PricePoint],
                             data: Dict[str, np.ndarray]) -> List[PatternResult]:
        """Детектирование Двойного дна"""
        results = []

        if len(lows) < 2 or len(highs) < 1:
            return results

        # Ищем последовательность: low-high-low
        for i in range(len(all_points) - 2):
            points = all_points[i:i+3]

            # Проверяем паттерн: L-H-L (типы точек)
            pattern_types = [p.type for p in points]
            expected_pattern = ['low', 'high', 'low']

            if pattern_types != expected_pattern:
                continue

            # Получаем цены
            bottom1 = points[0].price
            top = points[1].price  # Вершина между днами
            bottom2 = points[2].price

            # Дна должны быть примерно на одном уровне
            bottom_diff = abs(bottom1 - bottom2) / ((bottom1 + bottom2) / 2)
            if bottom_diff > config.DETECTION.PRICE_TOLERANCE_PCT:
                continue

            # Вершина должна быть значительно выше дна
            if not (top > bottom1 * 1.02 and top > bottom2 * 1.02):
                continue

            # Качество паттерна
            quality = self._calculate_double_pattern_quality(bottom1, bottom2, top, is_top=False)

            if quality < config.DETECTION.MIN_PATTERN_QUALITY:
                continue

            # Создаем точки паттерна
            pattern_points = [
                PatternPoint(index=points[0].index, price=bottom1, type='bottom1'),
                PatternPoint(index=points[1].index, price=top, type='top'),
                PatternPoint(index=points[2].index, price=bottom2, type='bottom2')
            ]

            # Расчет целей
            entry_price = points[2].price
            pattern_height = top - ((bottom1 + bottom2) / 2)

            targets = {
                'entry_price': entry_price,
                'neckline': top,
                'pattern_height': pattern_height,
                'stop_loss': min(bottom1, bottom2) * 0.99,
                'take_profit': entry_price + pattern_height
            }

            # Создаем результат
            result = PatternResult(
                name='double_bottom',
                direction='bullish',
                points=pattern_points,
                quality=quality,
                confidence=self.calculate_confidence(pattern_points, data),
                targets=targets,
                metadata={
                    'pattern_type': 'reversal',
                    'bottom_diff_pct': bottom_diff,
                    'height_pct': (top - bottom1) / bottom1
                }
            )

            results.append(result)

        return results

    def _calculate_double_pattern_quality(self,
                                         level1: float,
                                         level2: float,
                                         opposite: float,
                                         is_top: bool = True) -> float:
        """Расчет качества двойных паттернов"""
        # 1. Сходство уровней
        avg_level = (level1 + level2) / 2
        level_diff = abs(level1 - level2)
        similarity_score = 1.0 - (level_diff / avg_level)
        similarity_score = max(0.0, min(1.0, similarity_score))

        # 2. Глубина/высота паттерна
        if is_top:
            # Для двойной вершины
            depth = avg_level - opposite
            depth_score = min(1.0, depth / avg_level * 5)  # Нормализация
        else:
            # Для двойного дна
            height = opposite - avg_level
            depth_score = min(1.0, height / avg_level * 5)  # Нормализация

        # 3. Временной интервал между уровнями
        # (этот параметр должен быть передан, но для простоты опустим)
        time_score = 0.7

        # Итоговое качество
        quality = (similarity_score * 0.5 + depth_score * 0.3 + time_score * 0.2)
        return quality

    def _detect_triangle(self, all_points: List[PricePoint],
                        highs: List[PricePoint],
                        lows: List[PricePoint],
                        data: Dict[str, np.ndarray]) -> List[PatternResult]:
        """Детектирование Треугольника"""
        results = []

        # Нужно как минимум 4 точки: 2 максимума и 2 минимума
        if len(highs) < 2 or len(lows) < 2:
            return results

        # Ищем сходящиеся линии тренда
        for i in range(len(highs) - 1):
            for j in range(len(lows) - 1):
                # Берем две последовательные вершины
                high1 = highs[i]
                high2 = highs[i+1]

                # Берем две последовательные впадины
                low1 = lows[j]
                low2 = lows[j+1]

                # Проверяем, что точки идут в правильном порядке
                if not (high1.index < low1.index < high2.index < low2.index):
                    continue

                # Вычисляем линии тренда
                # Линия сопротивления через high1 и high2
                high_slope = (high2.price - high1.price) / (high2.index - high1.index)

                # Линия поддержки через low1 и low2
                low_slope = (low2.price - low1.price) / (low2.index - low1.index)

                # Для треугольника линии должны сходиться
                # (одна с положительным наклоном, другая с отрицательным, или обе сходятся)
                if abs(high_slope - low_slope) < 0.0001:
                    continue  # Параллельные линии - не треугольник

                # Определяем тип треугольника
                triangle_type = self._determine_triangle_type(high_slope, low_slope)

                if triangle_type == 'unknown':
                    continue

                # Качество паттерна
                quality = self._calculate_triangle_quality(
                    high1, high2, low1, low2, high_slope, low_slope
                )

                if quality < config.DETECTION.MIN_PATTERN_QUALITY:
                    continue

                # Создаем точки паттерна
                pattern_points = [
                    PatternPoint(index=high1.index, price=high1.price, type='high1'),
                    PatternPoint(index=low1.index, price=low1.price, type='low1'),
                    PatternPoint(index=high2.index, price=high2.price, type='high2'),
                    PatternPoint(index=low2.index, price=low2.price, type='low2')
                ]

                # Расчет целей
                # Для треугольника цель - высота основания, спроецированная от точки пробоя
                base_height = max(high1.price, high2.price) - min(low1.price, low2.price)

                # Определяем направление (по наклону линий)
                direction = self._determine_triangle_direction(triangle_type)
                entry_price = low2.price if direction == 'bullish' else high2.price

                targets = {
                    'entry_price': entry_price,
                    'base_height': base_height,
                    'stop_loss': low2.price * 0.99 if direction == 'bullish' else high2.price * 1.01,
                    'take_profit': entry_price + base_height if direction == 'bullish' else entry_price - base_height
                }

                # Создаем результат
                result = PatternResult(
                    name=f'triangle_{triangle_type}',
                    direction=direction,
                    points=pattern_points,
                    quality=quality,
                    confidence=self.calculate_confidence(pattern_points, data),
                    targets=targets,
                    metadata={
                        'pattern_type': 'continuation',
                        'triangle_type': triangle_type,
                        'high_slope': high_slope,
                        'low_slope': low_slope
                    }
                )

                results.append(result)

        return results

    def _determine_triangle_type(self, high_slope: float, low_slope: float) -> str:
        """Определение типа треугольника"""
        if high_slope < 0 and low_slope > 0:
            return 'symmetrical'  # Симметричный
        elif high_slope < 0 and abs(low_slope) < 0.0001:
            return 'descending'  # Нисходящий
        elif abs(high_slope) < 0.0001 and low_slope > 0:
            return 'ascending'  # Восходящий
        else:
            return 'unknown'

    def _determine_triangle_direction(self, triangle_type: str) -> str:
        """Определение направления пробоя треугольника"""
        if triangle_type == 'ascending':
            return 'bullish'  # Восходящий треугольник обычно пробивается вверх
        elif triangle_type == 'descending':
            return 'bearish'  # Нисходящий треугольник обычно пробивается вниз
        else:
            # Симметричный треугольник - направление не определено
            return 'neutral'

    def _calculate_triangle_quality(self,
                                   high1: PricePoint,
                                   high2: PricePoint,
                                   low1: PricePoint,
                                   low2: PricePoint,
                                   high_slope: float,
                                   low_slope: float) -> float:
        """Расчет качества треугольника"""
        # 1. Сходимость линий
        convergence_score = 1.0 - min(1.0, abs(high_slope - low_slope) * 1000)

        # 2. Количество касаний (чем больше, тем лучше)
        # Здесь у нас всего 2 касания каждой линии, поэтому фиксированная оценка
        touch_score = 0.6

        # 3. Объем (недоступен в этой функции)
        volume_score = 0.5

        # Итоговое качество
        quality = (convergence_score * 0.4 + touch_score * 0.4 + volume_score * 0.2)
        return quality

    def _detect_wedge(self, all_points: List[PricePoint],
                     highs: List[PricePoint],
                     lows: List[PricePoint],
                     data: Dict[str, np.ndarray]) -> List[PatternResult]:
        """Детектирование Клина"""
        # Реализация похожа на треугольник, но линии имеют одинаковый наклон
        results = []

        if len(highs) < 2 or len(lows) < 2:
            return results

        # Поиск клина (сходящиеся линии с одинаковым направлением)
        for i in range(len(highs) - 1):
            for j in range(len(lows) - 1):
                high1 = highs[i]
                high2 = highs[i+1]
                low1 = lows[j]
                low2 = lows[j+1]

                if not (high1.index < low1.index < high2.index < low2.index):
                    continue

                # Вычисляем наклоны
                high_slope = (high2.price - high1.price) / (high2.index - high1.index)
                low_slope = (low2.price - low1.price) / (low2.index - low1.index)

                # Для клина линии должны сходиться и иметь одинаковый знак наклона
                if high_slope * low_slope <= 0:
                    continue  # Разные направления - не клин

                # Линии должны сходиться (расстояние между ними уменьшается)
                if abs(high_slope) <= abs(low_slope):
                    continue

                # Определяем тип клина
                if high_slope < 0 and low_slope < 0:
                    wedge_type = 'falling'  # Падающий клин
                    direction = 'bullish'   # Обычно пробивается вверх
                elif high_slope > 0 and low_slope > 0:
                    wedge_type = 'rising'   # Восходящий клин
                    direction = 'bearish'   # Обычно пробивается вниз
                else:
                    continue

                # Качество
                quality = self._calculate_wedge_quality(high_slope, low_slope)

                if quality < config.DETECTION.MIN_PATTERN_QUALITY:
                    continue

                # Создаем точки
                pattern_points = [
                    PatternPoint(index=high1.index, price=high1.price, type='high1'),
                    PatternPoint(index=low1.index, price=low1.price, type='low1'),
                    PatternPoint(index=high2.index, price=high2.price, type='high2'),
                    PatternPoint(index=low2.index, price=low2.price, type='low2')
                ]

                # Цели
                base_height = max(high1.price, high2.price) - min(low1.price, low2.price)
                entry_price = low2.price if direction == 'bullish' else high2.price

                targets = {
                    'entry_price': entry_price,
                    'base_height': base_height,
                    'stop_loss': low2.price * 0.99 if direction == 'bullish' else high2.price * 1.01,
                    'take_profit': entry_price + base_height if direction == 'bullish' else entry_price - base_height
                }

                result = PatternResult(
                    name=f'wedge_{wedge_type}',
                    direction=direction,
                    points=pattern_points,
                    quality=quality,
                    confidence=self.calculate_confidence(pattern_points, data),
                    targets=targets,
                    metadata={
                        'pattern_type': 'reversal' if wedge_type == 'falling' else 'continuation',
                        'wedge_type': wedge_type,
                        'high_slope': high_slope,
                        'low_slope': low_slope
                    }
                )

                results.append(result)

        return results

    def _calculate_wedge_quality(self, high_slope: float, low_slope: float) -> float:
        """Расчет качества клина"""
        # Чем больше разница в наклонах (при одинаковом знаке), тем лучше
        slope_diff = abs(abs(high_slope) - abs(low_slope))
        quality = min(1.0, slope_diff * 1000)  # Нормализация

        return quality

    def _detect_rectangle(self, all_points: List[PricePoint],
                         highs: List[PricePoint],
                         lows: List[PricePoint],
                         data: Dict[str, np.ndarray]) -> List[PatternResult]:
        """Детектирование Прямоугольника (канала)"""
        results = []

        # Нужно как минимум 2 максимума и 2 минимума на примерно одинаковых уровнях
        if len(highs) < 2 or len(lows) < 2:
            return results

        # Ищем горизонтальные уровни поддержки и сопротивления
        support_levels = self._find_horizontal_levels(lows, tolerance=config.DETECTION.PRICE_TOLERANCE_PCT)
        resistance_levels = self._find_horizontal_levels(highs, tolerance=config.DETECTION.PRICE_TOLERANCE_PCT)

        for support in support_levels:
            for resistance in resistance_levels:
                # Проверяем что уровни не слишком близко
                channel_height = resistance - support
                if channel_height <= 0 or channel_height / resistance < 0.01:
                    continue

                # Ищем точки касания этих уровней
                support_touches = [p for p in lows if abs(p.price - support) / support <= config.DETECTION.PRICE_TOLERANCE_PCT]
                resistance_touches = [p for p in highs if abs(p.price - resistance) / resistance <= config.DETECTION.PRICE_TOLERANCE_PCT]

                if len(support_touches) < 2 or len(resistance_touches) < 2:
                    continue

                # Сортируем по времени
                support_touches.sort(key=lambda x: x.index)
                resistance_touches.sort(key=lambda x: x.index)

                # Берем первые две точки каждого уровня
                sup1, sup2 = support_touches[:2]
                res1, res2 = resistance_touches[:2]

                # Проверяем чередование
                if not (min(sup1.index, res1.index) < max(sup1.index, res1.index) <
                        min(sup2.index, res2.index) < max(sup2.index, res2.index)):
                    continue

                # Качество
                quality = self._calculate_rectangle_quality(sup1, sup2, res1, res2, support, resistance)

                if quality < config.DETECTION.MIN_PATTERN_QUALITY:
                    continue

                # Создаем точки
                pattern_points = [
                    PatternPoint(index=sup1.index, price=sup1.price, type='support1'),
                    PatternPoint(index=res1.index, price=res1.price, type='resistance1'),
                    PatternPoint(index=sup2.index, price=sup2.price, type='support2'),
                    PatternPoint(index=res2.index, price=res2.price, type='resistance2')
                ]

                # Направление (определяем по последнему движению)
                last_move = 'up' if res2.index > sup2.index else 'down'
                direction = 'bullish' if last_move == 'up' else 'bearish'
                entry_price = sup2.price if direction == 'bullish' else res2.price

                targets = {
                    'entry_price': entry_price,
                    'channel_height': channel_height,
                    'support': support,
                    'resistance': resistance,
                    'stop_loss': support * 0.99 if direction == 'bullish' else resistance * 1.01,
                    'take_profit': entry_price + channel_height if direction == 'bullish' else entry_price - channel_height
                }

                result = PatternResult(
                    name='rectangle',
                    direction=direction,
                    points=pattern_points,
                    quality=quality,
                    confidence=self.calculate_confidence(pattern_points, data),
                    targets=targets,
                    metadata={
                        'pattern_type': 'continuation',
                        'channel_height_pct': channel_height / resistance,
                        'support_touches': len(support_touches),
                        'resistance_touches': len(resistance_touches)
                    }
                )

                results.append(result)

        return results

    def _detect_flag(self, all_points: List[PricePoint],
                    highs: List[PricePoint],
                    lows: List[PricePoint],
                    data: Dict[str, np.ndarray]) -> List[PatternResult]:
        """Детектирование Флага"""
        # Флаг - это небольшой прямоугольник/клин после сильного движения
        results = []

        # Сначала ищем сильное движение (флагшток)
        if 'close' not in data:
            return results

        closes = data['close']
        if len(closes) < 20:
            return results

        # Ищем сильные движения
        for i in range(10, len(closes) - 10):
            # Проверяем движение за последние 5-10 свечей
            lookback = 5
            price_change = abs(closes[i] - closes[i-lookback]) / closes[i-lookback]

            if price_change < 0.03:  # Меньше 3% - не достаточно сильное движение
                continue

            # Определяем направление движения
            direction = 'bullish' if closes[i] > closes[i-lookback] else 'bearish'

            # Ищем флаг после движения
            flag_start = i
            flag_data = {
                'high': data['high'][flag_start:flag_start+10],
                'low': data['low'][flag_start:flag_start+10]
            }

            # Ищем прямоугольник или клин в этих данных
            flag_patterns = []

            # Проверяем прямоугольник
            rectangle_results = self._detect_rectangle_in_range(flag_data)
            if rectangle_results:
                flag_patterns.extend(rectangle_results)

            # Проверяем клин
            wedge_results = self._detect_wedge_in_range(flag_data)
            if wedge_results:
                flag_patterns.extend(wedge_results)

            if not flag_patterns:
                continue

            # Берем лучший паттерн
            best_pattern = max(flag_patterns, key=lambda x: x.quality)

            # Адаптируем точки под общую нумерацию
            for point in best_pattern.points:
                point.index += flag_start

            # Обновляем цели
            flag_height = best_pattern.targets.get('base_height', 0)
            pole_height = price_change * closes[i-lookback]

            # Цель флага - высота флагштока
            if direction == 'bullish':
                entry_price = best_pattern.points[-1].price
                targets = {
                    'entry_price': entry_price,
                    'pole_height': pole_height,
                    'flag_height': flag_height,
                    'stop_loss': min(p.price for p in best_pattern.points) * 0.99,
                    'take_profit': entry_price + pole_height
                }
            else:
                entry_price = best_pattern.points[-1].price
                targets = {
                    'entry_price': entry_price,
                    'pole_height': pole_height,
                    'flag_height': flag_height,
                    'stop_loss': max(p.price for p in best_pattern.points) * 1.01,
                    'take_profit': entry_price - pole_height
                }

            best_pattern.targets.update(targets)
            best_pattern.direction = direction

            results.append(best_pattern)

        return results

    def _detect_pennant(self, all_points: List[PricePoint],
                       highs: List[PricePoint],
                       lows: List[PricePoint],
                       data: Dict[str, np.ndarray]) -> List[PatternResult]:
        """Детектирование Вымпела"""
        # Вымпел - это маленький симметричный треугольник после сильного движения
        # Реализация похожа на флаг
        results = []

        if 'close' not in data:
            return results

        closes = data['close']
        if len(closes) < 20:
            return results

        # Ищем сильные движения
        for i in range(10, len(closes) - 10):
            lookback = 5
            price_change = abs(closes[i] - closes[i-lookback]) / closes[i-lookback]

            if price_change < 0.03:
                continue

            direction = 'bullish' if closes[i] > closes[i-lookback] else 'bearish'

            # Ищем треугольник после движения
            pennant_start = i
            pennant_data = {
                'high': data['high'][pennant_start:pennant_start+8],
                'low': data['low'][pennant_start:pennant_start+8]
            }

            # Ищем симметричный треугольник
            triangle_results = self._detect_triangle_in_range(pennant_data)
            if not triangle_results:
                continue

            # Берем лучший треугольник
            best_pattern = max(triangle_results, key=lambda x: x.quality)

            # Проверяем что это симметричный треугольник
            if 'symmetrical' not in best_pattern.name:
                continue

            # Адаптируем точки
            for point in best_pattern.points:
                point.index += pennant_start

            # Обновляем цели
            pole_height = price_change * closes[i-lookback]

            if direction == 'bullish':
                entry_price = best_pattern.points[-1].price
                targets = {
                    'entry_price': entry_price,
                    'pole_height': pole_height,
                    'stop_loss': min(p.price for p in best_pattern.points) * 0.99,
                    'take_profit': entry_price + pole_height
                }
            else:
                entry_price = best_pattern.points[-1].price
                targets = {
                    'entry_price': entry_price,
                    'pole_height': pole_height,
                    'stop_loss': max(p.price for p in best_pattern.points) * 1.01,
                    'take_profit': entry_price - pole_height
                }

            best_pattern.targets.update(targets)
            best_pattern.direction = direction
            best_pattern.name = 'pennant'

            results.append(best_pattern)

        return results

    def _detect_rectangle_in_range(self, data: Dict[str, np.ndarray]) -> List[PatternResult]:
        """Детектирование прямоугольника в заданном диапазоне данных"""
        # Упрощенная версия для внутреннего использования
        # Здесь должна быть реализация, но для краткости пропустим
        return []

    def _detect_wedge_in_range(self, data: Dict[str, np.ndarray]) -> List[PatternResult]:
        """Детектирование клина в заданном диапазоне данных"""
        # Упрощенная версия
        return []

    def _detect_triangle_in_range(self, data: Dict[str, np.ndarray]) -> List[PatternResult]:
        """Детектирование треугольника в заданном диапазоне данных"""
        # Упрощенная версия
        return []

    def _find_horizontal_levels(self, points: List[PricePoint], tolerance: float = 0.002) -> List[float]:
        """Поиск горизонтальных уровней (кластеризация цен)"""
        if not points:
            return []

        prices = [p.price for p in points]

        # Простая кластеризация
        clusters = []

        for price in prices:
            found_cluster = False

            for cluster in clusters:
                avg_price = np.mean(cluster)
                if abs(price - avg_price) / avg_price <= tolerance:
                    cluster.append(price)
                    found_cluster = True
                    break

            if not found_cluster:
                clusters.append([price])

        # Возвращаем средние цены кластеров с минимум 2 точками
        levels = []
        for cluster in clusters:
            if len(cluster) >= 2:
                levels.append(np.mean(cluster))

        return levels

    def _calculate_rectangle_quality(self,
                                    sup1: PricePoint,
                                    sup2: PricePoint,
                                    res1: PricePoint,
                                    res2: PricePoint,
                                    support: float,
                                    resistance: float) -> float:
        """Расчет качества прямоугольника"""
        # 1. Точность касаний
        sup1_error = abs(sup1.price - support) / support
        sup2_error = abs(sup2.price - support) / support
        res1_error = abs(res1.price - resistance) / resistance
        res2_error = abs(res2.price - resistance) / resistance

        avg_error = (sup1_error + sup2_error + res1_error + res2_error) / 4
        accuracy_score = 1.0 - min(1.0, avg_error * 100)

        # 2. Параллельность (горизонтальные линии)
        # Для горизонтальных линий это всегда отлично
        parallel_score = 1.0

        # 3. Количество касаний (у нас 2)
        touch_score = 0.6

        # 4. Длительность (разница во времени)
        time_diff = max(sup2.index, res2.index) - min(sup1.index, res1.index)
        duration_score = min(1.0, time_diff / 50)  # Нормализация

        quality = (accuracy_score * 0.3 + parallel_score * 0.2 +
                  touch_score * 0.3 + duration_score * 0.2)

        return quality

    def _filter_results(self, results: List[PatternResult]) -> List[PatternResult]:
        """Фильтрация результатов"""
        if not results:
            return []

        # 1. Фильтрация по качеству
        filtered = [r for r in results if r.quality >= config.DETECTION.MIN_PATTERN_QUALITY]

        # 2. Фильтрация по уверенности
        filtered = [r for r in filtered if r.confidence >= config.DETECTION.CONFIDENCE_THRESHOLD]

        # 3. Удаление пересекающихся паттернов
        filtered.sort(key=lambda x: x.quality * x.confidence, reverse=True)

        used_indices = set()
        unique_results = []

        for result in filtered:
            # Получаем все индексы свечей паттерна
            pattern_indices = {p.index for p in result.points}

            # Проверяем пересечение с уже выбранными паттернами
            if not pattern_indices.intersection(used_indices):
                unique_results.append(result)
                used_indices.update(pattern_indices)

        return unique_results

