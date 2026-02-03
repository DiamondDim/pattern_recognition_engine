"""
Модуль гармонических паттернов
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from scipy.spatial import KDTree
from scipy.signal import find_peaks

from .base_pattern import BasePattern, PatternPoint, PatternResult
from config import config

@dataclass
class ExtremePoint:
    """Экстремальная точка для гармонических паттернов"""
    index: int
    price: float
    type: str  # 'X', 'A', 'B', 'C', 'D'
    is_high: bool

class HarmonicPatterns(BasePattern):
    """Класс для детектирования гармонических паттернов"""

    def __init__(self):
        super().__init__(name="harmonic_patterns", min_points=4)

        # Определения гармонических паттернов (идеальные соотношения Фибоначчи)
        self.pattern_definitions = {
            'gartley': {
                'AB': 0.618,    # XA * 0.618
                'BC': 0.618,    # AB * 0.618
                'CD': 1.618,    # BC * 1.618
                'AD': 0.786     # XA * 0.786
            },
            'butterfly': {
                'AB': 0.786,    # XA * 0.786
                'BC': 0.618,    # AB * 0.618
                'CD': 2.618,    # BC * 2.618
                'AD': 1.272     # XA * 1.272
            },
            'bat': {
                'AB': 0.382,    # XA * 0.382
                'BC': 0.618,    # AB * 0.618
                'CD': 2.618,    # BC * 2.618
                'AD': 0.886     # XA * 0.886
            },
            'crab': {
                'AB': 0.382,    # XA * 0.382
                'BC': 0.618,    # AB * 0.618
                'CD': 3.618,    # BC * 3.618
                'AD': 1.618     # XA * 1.618
            },
            'shark': {
                'AB': 0.382,    # XA * 0.382
                'BC': 1.130,    # AB * 1.130
                'CD': 1.618,    # BC * 1.618
                'AD': 0.886     # XA * 0.886
            },
            'cypher': {
                'AB': 0.382,    # XA * 0.382
                'BC': 1.130,    # AB * 1.130
                'CD': 1.414,    # BC * 1.414
                'AD': 0.786     # XA * 0.786
            },
            'five_o': {
                'AB': 0.618,    # XA * 0.618
                'BC': 1.618,    # AB * 1.618
                'CD': 1.618,    # BC * 1.618
                'AD': 0.618     # XA * 0.618
            }
        }

        # Допустимое отклонение от идеальных соотношений
        self.fib_tolerance = config.DETECTION.FIBONACCI_TOLERANCE

    def detect(self, data: Dict[str, np.ndarray]) -> List[PatternResult]:
        """
        Детектирование гармонических паттернов

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
        extremes = self._find_all_extremes(data)

        if len(extremes) < config.DETECTION.MIN_HARMONIC_POINTS:
            return results

        # Ищем гармонические паттерны
        for pattern_name, fib_ratios in self.pattern_definitions.items():
            if config.DETECTION.ENABLE_HARMONIC:
                pattern_results = self._detect_specific_pattern(
                    extremes, pattern_name, fib_ratios, data
                )
                results.extend(pattern_results)

        # Фильтрация результатов
        filtered_results = self._filter_results(results)

        return filtered_results

    def _find_all_extremes(self, data: Dict[str, np.ndarray]) -> List[ExtremePoint]:
        """Поиск всех экстремумов (пиков и впадин)"""
        extremes = []

        try:
            # Используем scipy для поиска пиков
            high_prices = data['high']
            low_prices = data['low']

            # Находим локальные максимумы
            high_indices, _ = find_peaks(
                high_prices,
                prominence=self.peak_prominence * np.mean(high_prices),
                distance=config.DETECTION.MIN_CANDLES_FOR_PATTERN // 2
            )

            # Находим локальные минимумы
            low_indices, _ = find_peaks(
                -low_prices,
                prominence=self.peak_prominence * np.mean(low_prices),
                distance=config.DETECTION.MIN_CANDLES_FOR_PATTERN // 2
            )

            # Создаем точки максимумов
            for idx in high_indices:
                if idx < len(high_prices):
                    point = ExtremePoint(
                        index=idx,
                        price=float(high_prices[idx]),
                        type='',
                        is_high=True
                    )
                    extremes.append(point)

            # Создаем точки минимумов
            for idx in low_indices:
                if idx < len(low_prices):
                    point = ExtremePoint(
                        index=idx,
                        price=float(low_prices[idx]),
                        type='',
                        is_high=False
                    )
                    extremes.append(point)

            # Сортируем по индексу
            extremes.sort(key=lambda x: x.index)

            # Чередование максимумов и минимумов
            filtered_extremes = []
            for i, point in enumerate(extremes):
                if i == 0:
                    filtered_extremes.append(point)
                else:
                    # Проверяем что тип чередуется
                    if point.is_high != filtered_extremes[-1].is_high:
                        filtered_extremes.append(point)

            extremes = filtered_extremes

        except Exception as e:
            # Резервный метод
            extremes = self._find_extremes_simple(data)

        return extremes

    def _find_extremes_simple(self, data: Dict[str, np.ndarray]) -> List[ExtremePoint]:
        """Простой поиск экстремумов (резервный метод)"""
        extremes = []

        high_prices = data['high']
        low_prices = data['low']

        window = config.DETECTION.MIN_CANDLES_FOR_PATTERN // 2

        for i in range(window, len(high_prices) - window):
            # Проверяем максимум
            if high_prices[i] == np.max(high_prices[i-window:i+window+1]):
                point = ExtremePoint(
                    index=i,
                    price=float(high_prices[i]),
                    type='',
                    is_high=True
                )
                extremes.append(point)

            # Проверяем минимум
            if low_prices[i] == np.min(low_prices[i-window:i+window+1]):
                point = ExtremePoint(
                    index=i,
                    price=float(low_prices[i]),
                    type='',
                    is_high=False
                )
                extremes.append(point)

        # Сортируем и чередуем
        extremes.sort(key=lambda x: x.index)

        filtered = []
        for point in extremes:
            if not filtered or point.is_high != filtered[-1].is_high:
                filtered.append(point)

        return filtered

    def _detect_specific_pattern(self,
                                extremes: List[ExtremePoint],
                                pattern_name: str,
                                fib_ratios: Dict[str, float],
                                data: Dict[str, np.ndarray]) -> List[PatternResult]:
        """Детектирование конкретного гармонического паттерна"""
        results = []

        # Нужно минимум 5 точек для гармонического паттерна (X, A, B, C, D)
        if len(extremes) < 5:
            return results

        # Проверяем все возможные комбинации из 5 точек
        for i in range(len(extremes) - 4):
            points = extremes[i:i+5]

            # Проверяем чередование максимумов и минимумов
            if not self._check_alternation(points):
                continue

            # Присваиваем типа точкам (X, A, B, C, D)
            typed_points = self._assign_point_types(points)

            # Проверяем гармонические соотношения
            is_valid, actual_ratios = self._check_fibonacci_ratios(typed_points, fib_ratios)

            if not is_valid:
                continue

            # Рассчитываем качество паттерна
            quality = self._calculate_harmonic_quality(actual_ratios, fib_ratios)

            if quality < config.DETECTION.MIN_PATTERN_QUALITY:
                continue

            # Определяем направление
            direction = self._determine_harmonic_direction(typed_points)

            # Создаем точки паттерна в формате PatternPoint
            pattern_points = []
            for j, point in enumerate(typed_points):
                pattern_point = PatternPoint(
                    index=point.index,
                    price=point.price,
                    type=point.type,
                    metadata={'is_high': point.is_high}
                )
                pattern_points.append(pattern_point)

            # Рассчитываем цели
            targets = self._calculate_harmonic_targets(typed_points, pattern_name, direction)

            # Создаем результат
            result = PatternResult(
                name=pattern_name,
                direction=direction,
                points=pattern_points,
                quality=quality,
                confidence=self.calculate_confidence(pattern_points, data),
                targets=targets,
                metadata={
                    'pattern_type': 'harmonic',
                    'fibonacci_ratios': actual_ratios,
                    'ideal_ratios': fib_ratios
                }
            )

            results.append(result)

        return results

    def _check_alternation(self, points: List[ExtremePoint]) -> bool:
        """Проверка чередования максимумов и минимумов"""
        for i in range(1, len(points)):
            if points[i].is_high == points[i-1].is_high:
                return False
        return True

    def _assign_point_types(self, points: List[ExtremePoint]) -> List[ExtremePoint]:
        """Присвоение типа точкам (X, A, B, C, D)"""
        typed_points = points.copy()

        # Первая точка всегда X
        typed_points[0].type = 'X'

        # Определяем направление движения от X к A
        if typed_points[1].price > typed_points[0].price:
            # Движение вверх: X - низ, A - высоко
            typed_points[1].type = 'A'
        else:
            # Движение вниз: X - высоко, A - низко
            typed_points[1].type = 'A'

        # Остальные точки
        typed_points[2].type = 'B'
        typed_points[3].type = 'C'
        typed_points[4].type = 'D'

        return typed_points

    def _check_fibonacci_ratios(self,
                               points: List[ExtremePoint],
                               ideal_ratios: Dict[str, float]) -> Tuple[bool, Dict[str, float]]:
        """Проверка соотношений Фибоначчи"""
        # Извлекаем цены
        X = points[0].price
        A = points[1].price
        B = points[2].price
        C = points[3].price
        D = points[4].price

        # Вычисляем длины волн
        XA = abs(A - X)
        AB = abs(B - A)
        BC = abs(C - B)
        # CD рассчитывается ниже

        if XA == 0:
            return False, {}

        # Вычисляем фактические соотношения
        actual_ratios = {}

        # AB как процент от XA
        actual_ratios['AB'] = AB / XA if XA != 0 else 0

        # BC как процент от AB
        actual_ratios['BC'] = BC / AB if AB != 0 else 0

        # CD как процент от BC (пока неизвестно D, используем проекцию)
        # Сначала вычисляем идеальное D на основе паттерна
        if 'AD' in ideal_ratios:
            # AD как процент от XA
            ideal_AD = ideal_ratios['AD']

            # Определяем направление
            if A > X:  # Бычье движение XA
                D_projected = X + ideal_AD * XA if ideal_AD > 0 else X - abs(ideal_AD) * XA
            else:  # Медвежье движение XA
                D_projected = X - ideal_AD * XA if ideal_AD > 0 else X + abs(ideal_AD) * XA

            CD = abs(D_projected - C)
            actual_ratios['CD'] = CD / BC if BC != 0 else 0

        # Проверяем соответствие идеальным соотношениям
        is_valid = True
        for ratio_key, ideal_value in ideal_ratios.items():
            if ratio_key in actual_ratios:
                actual_value = actual_ratios[ratio_key]
                if actual_value == 0:
                    is_valid = False
                    break

                # Проверяем отклонение
                deviation = abs(actual_value - ideal_value) / ideal_value
                if deviation > self.fib_tolerance:
                    is_valid = False
                    break

        return is_valid, actual_ratios

    def _calculate_harmonic_quality(self,
                                   actual_ratios: Dict[str, float],
                                   ideal_ratios: Dict[str, float]) -> float:
        """Расчет качества гармонического паттерна"""
        if not actual_ratios:
            return 0.0

        total_error = 0.0
        count = 0

        for ratio_key, ideal_value in ideal_ratios.items():
            if ratio_key in actual_ratios:
                actual_value = actual_ratios[ratio_key]
                if actual_value > 0:
                    error = abs(actual_value - ideal_value) / ideal_value
                    total_error += error
                    count += 1

        if count == 0:
            return 0.0

        avg_error = total_error / count
        quality = 1.0 - min(1.0, avg_error / self.fib_tolerance)

        return quality

    def _determine_harmonic_direction(self, points: List[ExtremePoint]) -> str:
        """Определение направления гармонического паттерна"""
        # Паттерн считается бычьим, если D ниже C для покупки
        # или медвежьим, если D выше C для продажи

        # Но в гармонических паттернах направление определяется по XA движению
        X = points[0].price
        A = points[1].price
        D = points[4].price

        if A > X:  # Бычье движение XA
            # Для бычьих паттернов (Gartley, Butterfly, Crab)
            # D должна быть ниже C для покупки
            return 'bullish'
        else:  # Медвежье движение XA
            # Для медвежьих паттернов
            # D должна быть выше C для продажи
            return 'bearish'

    def _calculate_harmonic_targets(self,
                                   points: List[ExtremePoint],
                                   pattern_name: str,
                                   direction: str) -> Dict[str, float]:
        """Расчет целевых уровней для гармонического паттерна"""
        X = points[0].price
        A = points[1].price
        B = points[2].price
        C = points[3].price
        D = points[4].price

        # Вычисляем соотношения
        XA = abs(A - X)
        AB = abs(B - A)
        BC = abs(C - B)

        # Определяем точку входа (D)
        entry_price = D

        # Стоп-лосс зависит от паттерна
        if pattern_name in ['gartley', 'bat']:
            stop_loss = X * 1.01 if direction == 'bearish' else X * 0.99
        elif pattern_name == 'butterfly':
            stop_loss = A * 1.01 if direction == 'bearish' else A * 0.99
        elif pattern_name == 'crab':
            stop_loss = (X + (A - X) * 1.618) * 1.01 if direction == 'bearish' else (X - (X - A) * 1.618) * 0.99
        else:
            stop_loss = C * 1.01 if direction == 'bearish' else C * 0.99

        # Тейк-профит (обычно 61.8% от CD)
        if direction == 'bullish':
            take_profit = entry_price + abs(D - C) * 0.618
        else:
            take_profit = entry_price - abs(D - C) * 0.618

        return {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'XA_length': XA,
            'AB_length': AB,
            'BC_length': BC,
            'pattern_type': pattern_name
        }

    def _filter_results(self, results: List[PatternResult]) -> List[PatternResult]:
        """Фильтрация результатов"""
        if not results:
            return []

        # 1. Фильтрация по качеству
        filtered = [r for r in results if r.quality >= config.DETECTION.MIN_PATTERN_QUALITY]

        # 2. Фильтрация по уверенности
        filtered = [r for r in filtered if r.confidence >= config.DETECTION.CONFIDENCE_THRESHOLD]

        # 3. Удаление дубликатов (похожие паттерны на одних и тех же точках)
        filtered.sort(key=lambda x: x.quality * x.confidence, reverse=True)

        used_points = set()
        unique_results = []

        for result in filtered:
            # Создаем ключ на основе индексов точек
            point_key = tuple(sorted(p.index for p in result.points))

            if point_key not in used_points:
                unique_results.append(result)
                used_points.add(point_key)

        return unique_results

    def _calculate_fibonacci_levels(self,
                                   X: float,
                                   A: float,
                                   B: float,
                                   C: float,
                                   pattern_type: str) -> Tuple[float, Dict[str, float]]:
        """
        Расчет уровней Фибоначчи для гармонического паттерна

        Args:
            X, A, B, C: Цены точек
            pattern_type: Тип паттерна

        Returns:
            (D, fibonacci_levels)
        """
        # Вычисляем длины волн
        XA = abs(A - X)
        AB = abs(B - A)
        BC = abs(C - B)

        # Определяем идеальные соотношения для паттерна
        if pattern_type in self.pattern_definitions:
            ratios = self.pattern_definitions[pattern_type]
        else:
            ratios = self.pattern_definitions['gartley']  # По умолчанию

        # Вычисляем проекцию D
        AB_ratio = AB / XA if XA != 0 else 0
        BC_ratio = BC / AB if AB != 0 else 0

        # Определяем направление
        if A > X:  # Бычье движение XA
            direction = 1
        else:  # Медвежье движение XA
            direction = -1

        # Рассчитываем D на основе паттерна
        if 'AD' in ratios:
            AD_ratio = ratios['AD']

            if direction == 1:  # Бычий
                D = X + AD_ratio * XA
            else:  # Медвежий
                D = X - AD_ratio * XA
        else:
            # Альтернативный расчет
            CD_ratio = ratios.get('CD', 1.618)
            D = C + direction * BC * CD_ratio

        # Вычисляем все соотношения
        fibonacci_levels = {
            'AB': AB_ratio,
            'BC': BC_ratio,
            'CD': abs(D - C) / BC if BC != 0 else 0,
            'AD': abs(D - X) / XA if XA != 0 else 0
        }

        return D, fibonacci_levels

    @property
    def peak_prominence(self) -> float:
        """Минимальная значимость экстремума"""
        return 0.003  # 0.3%

