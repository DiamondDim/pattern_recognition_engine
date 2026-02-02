"""
Гармонические паттерны
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from scipy.optimize import minimize

from .base_pattern import (
    BasePattern, PatternType, PatternDirection,
    PatternPoint, MarketContext
)
from config import DETECTION_CONFIG


@dataclass
class HarmonicPattern(BasePattern):
    """Базовый класс для гармонических паттернов"""

    def __init__(self, name: str, abbreviation: str = ""):
        super().__init__(PatternType.HARMONIC, name, abbreviation)

        # Фибо уровни для гармонических паттернов
        self.fibonacci_levels = {
            '0.0': 0.0,
            '0.236': 0.236,
            '0.382': 0.382,
            '0.5': 0.5,
            '0.618': 0.618,
            '0.786': 0.786,
            '0.886': 0.886,
            '1.0': 1.0,
            '1.13': 1.13,
            '1.27': 1.27,
            '1.414': 1.414,
            '1.618': 1.618,
            '2.0': 2.0,
            '2.24': 2.24,
            '2.618': 2.618,
            '3.14': 3.14
        }

        # Допуски для фибо уровней
        self.fib_tolerance = 0.05  # 5%

        # Точки гармонического паттерна (X, A, B, C, D)
        self.point_x: Optional[PatternPoint] = None
        self.point_a: Optional[PatternPoint] = None
        self.point_b: Optional[PatternPoint] = None
        self.point_c: Optional[PatternPoint] = None
        self.point_d: Optional[PatternPoint] = None

        # Измерения паттерна
        self.xa_price_diff: Optional[float] = None  # Разница цен XA
        self.ab_retracement: Optional[float] = None  # Откат AB
        self.bc_retracement: Optional[float] = None  # Откат BC
        self.cd_extension: Optional[float] = None  # Расширение CD
        self.xd_extension: Optional[float] = None  # Расширение XD

    def _calculate_fib_retracement(self, start: float, end: float,
                                   level: float) -> float:
        """Расчет уровня Фибоначчи для отката"""
        diff = end - start
        return start + diff * level

    def _calculate_fib_extension(self, start: float, end: float,
                                 level: float) -> float:
        """Расчет уровня Фибоначчи для расширения"""
        diff = end - start
        return end + diff * level

    def _is_fib_level_match(self, actual: float, expected: float,
                            tolerance: float = None) -> bool:
        """Проверка соответствия фибо уровню"""
        if tolerance is None:
            tolerance = self.fib_tolerance

        if expected == 0:
            return False

        deviation = abs((actual - expected) / expected)
        return deviation <= tolerance

    def _find_swing_points(self, highs: np.ndarray, lows: np.ndarray,
                           min_distance: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Нахождение точек разворота (swing points)"""
        from scipy.signal import argrelextrema

        # Находим локальные максимумы и минимумы
        max_indices = argrelextrema(highs, np.greater, order=min_distance)[0]
        min_indices = argrelextrema(lows, np.less, order=min_distance)[0]

        # Объединяем и сортируем все точки
        all_points = []

        for idx in max_indices:
            all_points.append({
                'index': idx,
                'price': highs[idx],
                'type': 'high'
            })

        for idx in min_indices:
            all_points.append({
                'index': idx,
                'price': lows[idx],
                'type': 'low'
            })

        # Сортируем по индексу
        all_points.sort(key=lambda x: x['index'])

        # Отфильтровываем слабые точки
        filtered_points = []
        for i in range(1, len(all_points) - 1):
            prev = all_points[i - 1]
            curr = all_points[i]
            next_ = all_points[i + 1]

            # Для максимума: текущая цена должна быть выше соседних
            if curr['type'] == 'high':
                if curr['price'] > prev['price'] and curr['price'] > next_['price']:
                    filtered_points.append(curr)
            # Для минимума: текущая цена должна быть ниже соседних
            else:
                if curr['price'] < prev['price'] and curr['price'] < next_['price']:
                    filtered_points.append(curr)

        # Разделяем обратно на максимумы и минимумы
        filtered_highs = [p for p in filtered_points if p['type'] == 'high']
        filtered_lows = [p for p in filtered_points if p['type'] == 'low']

        return filtered_highs, filtered_lows

    def _validate_pattern_structure(self, points: List[Dict]) -> bool:
        """Валидация структуры гармонического паттерна"""
        if len(points) < 5:
            return False

        # Проверяем чередование максимумов и минимумов
        for i in range(len(points) - 1):
            if points[i]['type'] == points[i + 1]['type']:
                return False

        return True


@dataclass
class ABCDPattern(HarmonicPattern):
    """Паттерн ABCD"""

    def __init__(self):
        super().__init__("ABCD Pattern", "ABCD")
        self.pattern_variants = ['bullish', 'bearish']
        self.required_points = 4  # A, B, C, D

    def detect(self, data, highs, lows, closes, volumes, timestamps, **kwargs) -> bool:
        # Находим точки разворота
        swing_highs, swing_lows = self._find_swing_points(highs, lows, min_distance=3)

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return False

        # Объединяем все точки и сортируем
        all_points = swing_highs + swing_lows
        all_points.sort(key=lambda x: x['index'])

        # Ищем структуру ABCD
        for i in range(len(all_points) - 3):
            point_a = all_points[i]
            point_b = all_points[i + 1]
            point_c = all_points[i + 2]
            point_d = all_points[i + 3]

            # Проверяем чередование
            if not (point_a['type'] != point_b['type'] and
                    point_b['type'] != point_c['type'] and
                    point_c['type'] != point_d['type']):
                continue

            # Определяем направление
            if point_a['type'] == 'low':  # Бычий ABCD
                is_bullish = (point_b['price'] > point_a['price'] and
                              point_c['price'] < point_b['price'] and
                              point_d['price'] > point_c['price'])

                if is_bullish:
                    # Проверяем фибо соотношения
                    ab_diff = point_b['price'] - point_a['price']
                    bc_diff = point_b['price'] - point_c['price']
                    cd_diff = point_d['price'] - point_c['price']

                    # AB и CD должны быть примерно равны
                    if ab_diff > 0 and cd_diff > 0:
                        cd_ab_ratio = cd_diff / ab_diff

                        if self._is_fib_level_match(cd_ab_ratio, 1.0, 0.1):  # CD ≈ AB
                            self._save_abcd_points(point_a, point_b, point_c,
                                                   point_d, timestamps, True)
                            self.direction = PatternDirection.BULLISH
                            self._is_detected = True
                            return True

            else:  # Медвежий ABCD
                is_bearish = (point_b['price'] < point_a['price'] and
                              point_c['price'] > point_b['price'] and
                              point_d['price'] < point_c['price'])

                if is_bearish:
                    ab_diff = point_a['price'] - point_b['price']
                    bc_diff = point_c['price'] - point_b['price']
                    cd_diff = point_c['price'] - point_d['price']

                    if ab_diff > 0 and cd_diff > 0:
                        cd_ab_ratio = cd_diff / ab_diff

                        if self._is_fib_level_match(cd_ab_ratio, 1.0, 0.1):
                            self._save_abcd_points(point_a, point_b, point_c,
                                                   point_d, timestamps, False)
                            self.direction = PatternDirection.BEARISH
                            self._is_detected = True
                            return True

        return False

    def _save_abcd_points(self, point_a, point_b, point_c, point_d,
                          timestamps, is_bullish: bool):
        """Сохранение точек паттерна ABCD"""
        self.points = []

        # Точка A
        self.points.append(
            PatternPoint(
                index=point_a['index'],
                timestamp=timestamps[point_a['index']],
                price=point_a['price'],
                point_type='A'
            )
        )

        # Точка B
        self.points.append(
            PatternPoint(
                index=point_b['index'],
                timestamp=timestamps[point_b['index']],
                price=point_b['price'],
                point_type='B'
            )
        )

        # Точка C
        self.points.append(
            PatternPoint(
                index=point_c['index'],
                timestamp=timestamps[point_c['index']],
                price=point_c['price'],
                point_type='C'
            )
        )

        # Точка D
        self.points.append(
            PatternPoint(
                index=point_d['index'],
                timestamp=timestamps[point_d['index']],
                price=point_d['price'],
                point_type='D'
            )
        )

        # Сохраняем ссылки
        self.point_a = self.points[0]
        self.point_b = self.points[1]
        self.point_c = self.points[2]
        self.point_d = self.points[3]

        # Рассчитываем измерения
        if is_bullish:
            self.xa_price_diff = self.point_b.price - self.point_a.price
            self.ab_retracement = (self.point_b.price - self.point_c.price) / self.xa_price_diff
            self.bc_retracement = (self.point_c.price - self.point_a.price) / (self.point_b.price - self.point_a.price)
            self.cd_extension = (self.point_d.price - self.point_c.price) / (self.point_b.price - self.point_c.price)
        else:
            self.xa_price_diff = self.point_a.price - self.point_b.price
            self.ab_retracement = (self.point_c.price - self.point_b.price) / self.xa_price_diff
            self.bc_retracement = (self.point_a.price - self.point_c.price) / (self.point_a.price - self.point_b.price)
            self.cd_extension = (self.point_c.price - self.point_d.price) / (self.point_c.price - self.point_b.price)

    def calculate_quality(self) -> float:
        if not self._is_detected:
            return 0.0

        scores = []

        # 1. Соотношение AB = CD
        if self.cd_extension is not None:
            cd_score = 1 - min(abs(self.cd_extension - 1.0) / 0.2, 1.0)
            scores.append(cd_score)

        # 2. Откат BC (должен быть 0.618)
        if self.bc_retracement is not None:
            bc_score = 1 - min(abs(self.bc_retracement - 0.618) / 0.2, 1.0)
            scores.append(bc_score)

        # 3. Общая симметрия
        if len(self.points) == 4:
            # Время между AB и CD должно быть примерно одинаковым
            time_ab = (self.point_b.timestamp - self.point_a.timestamp).total_seconds()
            time_cd = (self.point_d.timestamp - self.point_c.timestamp).total_seconds()

            if time_ab > 0 and time_cd > 0:
                time_ratio = min(time_ab, time_cd) / max(time_ab, time_cd)
                scores.append(time_ratio)

        return np.mean(scores) if scores else 0.0

    def calculate_targets(self, current_price: float):
        if not self._is_detected or len(self.points) < 4:
            return super().calculate_targets(current_price)

        # Для ABCD паттерна
        if self.direction == PatternDirection.BULLISH:
            # Бычий ABCD: цель - расширение от D
            entry_price = current_price
            stop_loss = self.point_d.price * 0.99  # 1% ниже точки D
            take_profit = entry_price + (self.point_d.price - self.point_c.price) * 1.618

            self.targets.entry_price = entry_price
            self.targets.stop_loss = stop_loss
            self.targets.take_profit = take_profit

            # Дополнительные цели
            self.targets.target1 = entry_price + (self.point_d.price - self.point_c.price)
            self.targets.target2 = take_profit
            self.targets.target3 = entry_price + (self.point_d.price - self.point_c.price) * 2.618

        else:  # BEARISH
            # Медвежий ABCD: цель - расширение от D вниз
            entry_price = current_price
            stop_loss = self.point_d.price * 1.01  # 1% выше точки D
            take_profit = entry_price - (self.point_c.price - self.point_d.price) * 1.618

            self.targets.entry_price = entry_price
            self.targets.stop_loss = stop_loss
            self.targets.take_profit = take_profit

            self.targets.target1 = entry_price - (self.point_c.price - self.point_d.price)
            self.targets.target2 = take_profit
            self.targets.target3 = entry_price - (self.point_c.price - self.point_d.price) * 2.618

        # Риск/прибыль
        if self.targets.entry_price and self.targets.stop_loss and self.targets.take_profit:
            risk = abs(self.targets.entry_price - self.targets.stop_loss)
            reward = abs(self.targets.take_profit - self.targets.entry_price)
            self.targets.profit_risk_ratio = reward / risk if risk > 0 else 0

        return self.targets


@dataclass
class GartleyPattern(HarmonicPattern):
    """Паттерн Гартли"""

    def __init__(self):
        super().__init__("Gartley Pattern", "GART")
        self.pattern_variants = ['bullish', 'bearish']
        self.required_points = 5  # X, A, B, C, D

    def detect(self, data, highs, lows, closes, volumes, timestamps, **kwargs) -> bool:
        # Находим точки разворота
        swing_highs, swing_lows = self._find_swing_points(highs, lows, min_distance=5)

        if len(swing_highs) < 3 or len(swing_lows) < 2:
            return False

        # Объединяем все точки
        all_points = swing_highs + swing_lows
        all_points.sort(key=lambda x: x['index'])

        # Ищем структуру XABCD
        for i in range(len(all_points) - 4):
            point_x = all_points[i]
            point_a = all_points[i + 1]
            point_b = all_points[i + 2]
            point_c = all_points[i + 3]
            point_d = all_points[i + 4]

            # Проверяем чередование
            types = [p['type'] for p in [point_x, point_a, point_b, point_c, point_d]]
            for j in range(len(types) - 1):
                if types[j] == types[j + 1]:
                    break
            else:
                # Все типы чередуются
                if self._check_gartley_ratios(point_x, point_a, point_b, point_c, point_d):
                    self._save_gartley_points(point_x, point_a, point_b,
                                              point_c, point_d, timestamps)
                    self._is_detected = True

                    # Определяем направление
                    if point_x['type'] == 'low':  # Бычий Гартли
                        self.direction = PatternDirection.BULLISH
                    else:  # Медвежий Гартли
                        self.direction = PatternDirection.BEARISH

                    return True

        return False

    def _check_gartley_ratios(self, x, a, b, c, d) -> bool:
        """Проверка фибо соотношений для паттерна Гартли"""
        # Для бычьего Гартли: X - низ, A - высоко, B - низко, C - высоко, D - низко
        # Для медвежьего Гартли: X - высоко, A - низко, B - высоко, C - низко, D - высоко

        is_bullish = x['type'] == 'low'

        if is_bullish:
            # AB откат: должен быть 0.618 от XA
            xa_diff = a['price'] - x['price']
            ab_diff = a['price'] - b['price']

            if xa_diff == 0:
                return False

            ab_retracement = ab_diff / xa_diff

            # BC откат: должен быть 0.382-0.886 от AB
            bc_diff = c['price'] - b['price']
            if ab_diff == 0:
                return False
            bc_retracement = bc_diff / ab_diff

            # CD расширение: должно быть 1.27 от BC
            cd_diff = d['price'] - c['price']
            if bc_diff == 0:
                return False
            cd_extension = cd_diff / bc_diff

            # Проверяем соотношения
            ab_ok = self._is_fib_level_match(ab_retracement, 0.618, 0.1)
            bc_ok = (self._is_fib_level_match(bc_retracement, 0.382, 0.1) or
                     self._is_fib_level_match(bc_retracement, 0.886, 0.1))
            cd_ok = self._is_fib_level_match(cd_extension, 1.27, 0.1)

            return ab_ok and bc_ok and cd_ok

        else:  # Медвежий Гартли
            xa_diff = x['price'] - a['price']
            ab_diff = b['price'] - a['price']

            if xa_diff == 0:
                return False

            ab_retracement = ab_diff / xa_diff

            bc_diff = b['price'] - c['price']
            if ab_diff == 0:
                return False
            bc_retracement = bc_diff / ab_diff

            cd_diff = c['price'] - d['price']
            if bc_diff == 0:
                return False
            cd_extension = cd_diff / bc_diff

            ab_ok = self._is_fib_level_match(ab_retracement, 0.618, 0.1)
            bc_ok = (self._is_fib_level_match(bc_retracement, 0.382, 0.1) or
                     self._is_fib_level_match(bc_retracement, 0.886, 0.1))
            cd_ok = self._is_fib_level_match(cd_extension, 1.27, 0.1)

            return ab_ok and bc_ok and cd_ok

    def _save_gartley_points(self, x, a, b, c, d, timestamps):
        """Сохранение точек паттерна Гартли"""
        self.points = []

        # Точка X
        self.points.append(
            PatternPoint(
                index=x['index'],
                timestamp=timestamps[x['index']],
                price=x['price'],
                point_type='X'
            )
        )

        # Точка A
        self.points.append(
            PatternPoint(
                index=a['index'],
                timestamp=timestamps[a['index']],
                price=a['price'],
                point_type='A'
            )
        )

        # Точка B
        self.points.append(
            PatternPoint(
                index=b['index'],
                timestamp=timestamps[b['index']],
                price=b['price'],
                point_type='B'
            )
        )

        # Точка C
        self.points.append(
            PatternPoint(
                index=c['index'],
                timestamp=timestamps[c['index']],
                price=c['price'],
                point_type='C'
            )
        )

        # Точка D
        self.points.append(
            PatternPoint(
                index=d['index'],
                timestamp=timestamps[d['index']],
                price=d['price'],
                point_type='D'
            )
        )

        # Сохраняем ссылки
        self.point_x = self.points[0]
        self.point_a = self.points[1]
        self.point_b = self.points[2]
        self.point_c = self.points[3]
        self.point_d = self.points[4]

        # Рассчитываем измерения
        if self.direction == PatternDirection.BULLISH:
            self.xa_price_diff = self.point_a.price - self.point_x.price
            self.ab_retracement = (self.point_a.price - self.point_b.price) / self.xa_price_diff
            self.bc_retracement = (self.point_c.price - self.point_b.price) / (self.point_a.price - self.point_b.price)
            self.cd_extension = (self.point_d.price - self.point_c.price) / (self.point_c.price - self.point_b.price)
        else:
            self.xa_price_diff = self.point_x.price - self.point_a.price
            self.ab_retracement = (self.point_b.price - self.point_a.price) / self.xa_price_diff
            self.bc_retracement = (self.point_b.price - self.point_c.price) / (self.point_b.price - self.point_a.price)
            self.cd_extension = (self.point_c.price - self.point_d.price) / (self.point_b.price - self.point_c.price)

    def calculate_quality(self) -> float:
        if not self._is_detected:
            return 0.0

        scores = []

        # 1. Соотношение AB (0.618)
        if self.ab_retracement is not None:
            ab_score = 1 - min(abs(self.ab_retracement - 0.618) / 0.1, 1.0)
            scores.append(ab_score)

        # 2. Соотношение BC (0.382 или 0.886)
        if self.bc_retracement is not None:
            bc_score_382 = 1 - min(abs(self.bc_retracement - 0.382) / 0.1, 1.0)
            bc_score_886 = 1 - min(abs(self.bc_retracement - 0.886) / 0.1, 1.0)
            bc_score = max(bc_score_382, bc_score_886)
            scores.append(bc_score)

        # 3. Соотношение CD (1.27)
        if self.cd_extension is not None:
            cd_score = 1 - min(abs(self.cd_extension - 1.27) / 0.1, 1.0)
            scores.append(cd_score)

        # 4. Общая симметрия
        if len(self.points) == 5:
            # Проверяем временные соотношения
            time_xa = (self.point_a.timestamp - self.point_x.timestamp).total_seconds()
            time_ac = (self.point_c.timestamp - self.point_a.timestamp).total_seconds()
            time_cd = (self.point_d.timestamp - self.point_c.timestamp).total_seconds()

            if time_xa > 0 and time_cd > 0:
                # XA и CD должны быть примерно равны
                time_ratio = min(time_xa, time_cd) / max(time_xa, time_cd)
                scores.append(time_ratio)

        return np.mean(scores) if scores else 0.0

    def calculate_targets(self, current_price: float):
        if not self._is_detected or len(self.points) < 5:
            return super().calculate_targets(current_price)

        # Для паттерна Гартли
        if self.direction == PatternDirection.BULLISH:
            # Бычий Гартли: точка D - зона покупки
            entry_price = current_price
            stop_loss = self.point_d.price * 0.99  # 1% ниже точки D

            # Цели: 0.618 и 1.0 от XA
            target1 = self.point_d.price + (self.point_a.price - self.point_x.price) * 0.618
            target2 = self.point_d.price + (self.point_a.price - self.point_x.price)

            self.targets.entry_price = entry_price
            self.targets.stop_loss = stop_loss
            self.targets.take_profit = target2

            self.targets.target1 = target1
            self.targets.target2 = target2
            self.targets.target3 = self.point_d.price + (self.point_a.price - self.point_x.price) * 1.618

        else:  # BEARISH
            # Медвежий Гартли: точка D - зона продажи
            entry_price = current_price
            stop_loss = self.point_d.price * 1.01  # 1% выше точки D

            target1 = self.point_d.price - (self.point_x.price - self.point_a.price) * 0.618
            target2 = self.point_d.price - (self.point_x.price - self.point_a.price)

            self.targets.entry_price = entry_price
            self.targets.stop_loss = stop_loss
            self.targets.take_profit = target2

            self.targets.target1 = target1
            self.targets.target2 = target2
            self.targets.target3 = self.point_d.price - (self.point_x.price - self.point_a.price) * 1.618

        # Риск/прибыль
        if self.targets.entry_price and self.targets.stop_loss and self.targets.take_profit:
            risk = abs(self.targets.entry_price - self.targets.stop_loss)
            reward = abs(self.targets.take_profit - self.targets.entry_price)
            self.targets.profit_risk_ratio = reward / risk if risk > 0 else 0

        return self.targets


@dataclass
class ButterflyPattern(HarmonicPattern):
    """Паттерн Бабочка"""

    def __init__(self):
        super().__init__("Butterfly Pattern", "BUTT")
        self.pattern_variants = ['bullish', 'bearish']
        self.required_points = 5  # X, A, B, C, D

    def detect(self, data, highs, lows, closes, volumes, timestamps, **kwargs) -> bool:
        # Находим точки разворота
        swing_highs, swing_lows = self._find_swing_points(highs, lows, min_distance=7)

        if len(swing_highs) < 3 or len(swing_lows) < 2:
            return False

        all_points = swing_highs + swing_lows
        all_points.sort(key=lambda x: x['index'])

        # Ищем структуру XABCD
        for i in range(len(all_points) - 4):
            point_x = all_points[i]
            point_a = all_points[i + 1]
            point_b = all_points[i + 2]
            point_c = all_points[i + 3]
            point_d = all_points[i + 4]

            # Проверяем чередование
            types = [p['type'] for p in [point_x, point_a, point_b, point_c, point_d]]
            for j in range(len(types) - 1):
                if types[j] == types[j + 1]:
                    break
            else:
                if self._check_butterfly_ratios(point_x, point_a, point_b, point_c, point_d):
                    self._save_butterfly_points(point_x, point_a, point_b,
                                                point_c, point_d, timestamps)
                    self._is_detected = True

                    if point_x['type'] == 'low':
                        self.direction = PatternDirection.BULLISH
                    else:
                        self.direction = PatternDirection.BEARISH

                    return True

        return False

    def _check_butterfly_ratios(self, x, a, b, c, d) -> bool:
        """Проверка фибо соотношений для паттерна Бабочка"""
        is_bullish = x['type'] == 'low'

        if is_bullish:
            # AB откат: должен быть 0.786 от XA
            xa_diff = a['price'] - x['price']
            ab_diff = a['price'] - b['price']

            if xa_diff == 0:
                return False

            ab_retracement = ab_diff / xa_diff

            # BC откат: должен быть 0.382-0.886 от AB
            bc_diff = c['price'] - b['price']
            if ab_diff == 0:
                return False
            bc_retracement = bc_diff / ab_diff

            # CD расширение: должно быть 1.618-2.24 от BC
            cd_diff = d['price'] - c['price']
            if bc_diff == 0:
                return False
            cd_extension = cd_diff / bc_diff

            # Проверяем соотношения
            ab_ok = self._is_fib_level_match(ab_retracement, 0.786, 0.1)
            bc_ok = (self._is_fib_level_match(bc_retracement, 0.382, 0.1) or
                     self._is_fib_level_match(bc_retracement, 0.886, 0.1))
            cd_ok = (self._is_fib_level_match(cd_extension, 1.618, 0.15) or
                     self._is_fib_level_match(cd_extension, 2.24, 0.15))

            return ab_ok and bc_ok and cd_ok

        else:  # Медвежий
            xa_diff = x['price'] - a['price']
            ab_diff = b['price'] - a['price']

            if xa_diff == 0:
                return False

            ab_retracement = ab_diff / xa_diff

            bc_diff = b['price'] - c['price']
            if ab_diff == 0:
                return False
            bc_retracement = bc_diff / ab_diff

            cd_diff = c['price'] - d['price']
            if bc_diff == 0:
                return False
            cd_extension = cd_diff / bc_diff

            ab_ok = self._is_fib_level_match(ab_retracement, 0.786, 0.1)
            bc_ok = (self._is_fib_level_match(bc_retracement, 0.382, 0.1) or
                     self._is_fib_level_match(bc_retracement, 0.886, 0.1))
            cd_ok = (self._is_fib_level_match(cd_extension, 1.618, 0.15) or
                     self._is_fib_level_match(cd_extension, 2.24, 0.15))

            return ab_ok and bc_ok and cd_ok

    def _save_butterfly_points(self, x, a, b, c, d, timestamps):
        """Сохранение точек паттерна Бабочка"""
        self.points = []

        self.points.append(
            PatternPoint(
                index=x['index'],
                timestamp=timestamps[x['index']],
                price=x['price'],
                point_type='X'
            )
        )

        self.points.append(
            PatternPoint(
                index=a['index'],
                timestamp=timestamps[a['index']],
                price=a['price'],
                point_type='A'
            )
        )

        self.points.append(
            PatternPoint(
                index=b['index'],
                timestamp=timestamps[b['index']],
                price=b['price'],
                point_type='B'
            )
        )

        self.points.append(
            PatternPoint(
                index=c['index'],
                timestamp=timestamps[c['index']],
                price=c['price'],
                point_type='C'
            )
        )

        self.points.append(
            PatternPoint(
                index=d['index'],
                timestamp=timestamps[d['index']],
                price=d['price'],
                point_type='D'
            )
        )

        self.point_x = self.points[0]
        self.point_a = self.points[1]
        self.point_b = self.points[2]
        self.point_c = self.points[3]
        self.point_d = self.points[4]

        if self.direction == PatternDirection.BULLISH:
            self.xa_price_diff = self.point_a.price - self.point_x.price
            self.ab_retracement = (self.point_a.price - self.point_b.price) / self.xa_price_diff
            self.bc_retracement = (self.point_c.price - self.point_b.price) / (self.point_a.price - self.point_b.price)
            self.cd_extension = (self.point_d.price - self.point_c.price) / (self.point_c.price - self.point_b.price)
        else:
            self.xa_price_diff = self.point_x.price - self.point_a.price
            self.ab_retracement = (self.point_b.price - self.point_a.price) / self.xa_price_diff
            self.bc_retracement = (self.point_b.price - self.point_c.price) / (self.point_b.price - self.point_a.price)
            self.cd_extension = (self.point_c.price - self.point_d.price) / (self.point_b.price - self.point_c.price)

    def calculate_quality(self) -> float:
        if not self._is_detected:
            return 0.0

        scores = []

        # 1. Соотношение AB (0.786)
        if self.ab_retracement is not None:
            ab_score = 1 - min(abs(self.ab_retracement - 0.786) / 0.1, 1.0)
            scores.append(ab_score)

        # 2. Соотношение CD (1.618 или 2.24)
        if self.cd_extension is not None:
            cd_score_1618 = 1 - min(abs(self.cd_extension - 1.618) / 0.15, 1.0)
            cd_score_224 = 1 - min(abs(self.cd_extension - 2.24) / 0.15, 1.0)
            cd_score = max(cd_score_1618, cd_score_224)
            scores.append(cd_score)

        # 3. Расширение XD (1.27)
        if self.point_x and self.point_d:
            xd_diff = abs(self.point_d.price - self.point_x.price)
            xa_diff = abs(self.point_a.price - self.point_x.price)

            if xa_diff > 0:
                xd_extension = xd_diff / xa_diff
                xd_score = 1 - min(abs(xd_extension - 1.27) / 0.1, 1.0)
                scores.append(xd_score)

        return np.mean(scores) if scores else 0.0

    def calculate_targets(self, current_price: float):
        if not self._is_detected or len(self.points) < 5:
            return super().calculate_targets(current_price)

        # Для паттерна Бабочка
        if self.direction == PatternDirection.BULLISH:
            # Бычья Бабочка: точка D - зона покупки
            entry_price = current_price
            stop_loss = self.point_d.price * 0.99

            # Цели: 0.382 и 0.618 от XA
            target1 = self.point_d.price + (self.point_a.price - self.point_x.price) * 0.382
            target2 = self.point_d.price + (self.point_a.price - self.point_x.price) * 0.618

            self.targets.entry_price = entry_price
            self.targets.stop_loss = stop_loss
            self.targets.take_profit = target2

            self.targets.target1 = target1
            self.targets.target2 = target2
            self.targets.target3 = self.point_d.price + (self.point_a.price - self.point_x.price)

        else:  # BEARISH
            # Медвежья Бабочка: точка D - зона продажи
            entry_price = current_price
            stop_loss = self.point_d.price * 1.01

            target1 = self.point_d.price - (self.point_x.price - self.point_a.price) * 0.382
            target2 = self.point_d.price - (self.point_x.price - self.point_a.price) * 0.618

            self.targets.entry_price = entry_price
            self.targets.stop_loss = stop_loss
            self.targets.take_profit = target2

            self.targets.target1 = target1
            self.targets.target2 = target2
            self.targets.target3 = self.point_d.price - (self.point_x.price - self.point_a.price)

        # Риск/прибыль
        if self.targets.entry_price and self.targets.stop_loss and self.targets.take_profit:
            risk = abs(self.targets.entry_price - self.targets.stop_loss)
            reward = abs(self.targets.take_profit - self.targets.entry_price)
            self.targets.profit_risk_ratio = reward / risk if risk > 0 else 0

        return self.targets


@dataclass
class BatPattern(HarmonicPattern):
    """Паттерн Bat (Летучая мышь)"""

    def __init__(self):
        super().__init__("Bat Pattern", "BAT")
        self.pattern_variants = ['bullish', 'bearish']
        self.required_points = 5  # X, A, B, C, D

    def detect(self, data, highs, lows, closes, volumes, timestamps, **kwargs) -> bool:
        swing_highs, swing_lows = self._find_swing_points(highs, lows, min_distance=6)

        if len(swing_highs) < 3 or len(swing_lows) < 2:
            return False

        all_points = swing_highs + swing_lows
        all_points.sort(key=lambda x: x['index'])

        for i in range(len(all_points) - 4):
            point_x = all_points[i]
            point_a = all_points[i + 1]
            point_b = all_points[i + 2]
            point_c = all_points[i + 3]
            point_d = all_points[i + 4]

            types = [p['type'] for p in [point_x, point_a, point_b, point_c, point_d]]
            for j in range(len(types) - 1):
                if types[j] == types[j + 1]:
                    break
            else:
                if self._check_bat_ratios(point_x, point_a, point_b, point_c, point_d):
                    self._save_bat_points(point_x, point_a, point_b,
                                          point_c, point_d, timestamps)
                    self._is_detected = True

                    if point_x['type'] == 'low':
                        self.direction = PatternDirection.BULLISH
                    else:
                        self.direction = PatternDirection.BEARISH

                    return True

        return False

    def _check_bat_ratios(self, x, a, b, c, d) -> bool:
        """Проверка фибо соотношений для паттерна Bat"""
        is_bullish = x['type'] == 'low'

        if is_bullish:
            # AB откат: должен быть 0.382-0.5 от XA
            xa_diff = a['price'] - x['price']
            ab_diff = a['price'] - b['price']

            if xa_diff == 0:
                return False

            ab_retracement = ab_diff / xa_diff

            # BC откат: должен быть 0.382-0.886 от AB
            bc_diff = c['price'] - b['price']
            if ab_diff == 0:
                return False
            bc_retracement = bc_diff / ab_diff

            # CD расширение: должно быть 1.618-2.618 от BC
            cd_diff = d['price'] - c['price']
            if bc_diff == 0:
                return False
            cd_extension = cd_diff / bc_diff

            # XD расширение: должно быть 0.886 от XA
            xd_diff = d['price'] - x['price']
            xd_extension = xd_diff / xa_diff

            ab_ok = (self._is_fib_level_match(ab_retracement, 0.382, 0.05) or
                     self._is_fib_level_match(ab_retracement, 0.5, 0.05))
            bc_ok = (self._is_fib_level_match(bc_retracement, 0.382, 0.1) or
                     self._is_fib_level_match(bc_retracement, 0.886, 0.1))
            cd_ok = (self._is_fib_level_match(cd_extension, 1.618, 0.15) or
                     self._is_fib_level_match(cd_extension, 2.618, 0.15))
            xd_ok = self._is_fib_level_match(xd_extension, 0.886, 0.05)

            return ab_ok and bc_ok and cd_ok and xd_ok

        else:  # Медвежий
            xa_diff = x['price'] - a['price']
            ab_diff = b['price'] - a['price']

            if xa_diff == 0:
                return False

            ab_retracement = ab_diff / xa_diff

            bc_diff = b['price'] - c['price']
            if ab_diff == 0:
                return False
            bc_retracement = bc_diff / ab_diff

            cd_diff = c['price'] - d['price']
            if bc_diff == 0:
                return False
            cd_extension = cd_diff / bc_diff

            xd_diff = x['price'] - d['price']
            xd_extension = xd_diff / xa_diff

            ab_ok = (self._is_fib_level_match(ab_retracement, 0.382, 0.05) or
                     self._is_fib_level_match(ab_retracement, 0.5, 0.05))
            bc_ok = (self._is_fib_level_match(bc_retracement, 0.382, 0.1) or
                     self._is_fib_level_match(bc_retracement, 0.886, 0.1))
            cd_ok = (self._is_fib_level_match(cd_extension, 1.618, 0.15) or
                     self._is_fib_level_match(cd_extension, 2.618, 0.15))
            xd_ok = self._is_fib_level_match(xd_extension, 0.886, 0.05)

            return ab_ok and bc_ok and cd_ok and xd_ok

    def _save_bat_points(self, x, a, b, c, d, timestamps):
        """Сохранение точек паттерна Bat"""
        self.points = []

        self.points.append(
            PatternPoint(
                index=x['index'],
                timestamp=timestamps[x['index']],
                price=x['price'],
                point_type='X'
            )
        )

        self.points.append(
            PatternPoint(
                index=a['index'],
                timestamp=timestamps[a['index']],
                price=a['price'],
                point_type='A'
            )
        )

        self.points.append(
            PatternPoint(
                index=b['index'],
                timestamp=timestamps[b['index']],
                price=b['price'],
                point_type='B'
            )
        )

        self.points.append(
            PatternPoint(
                index=c['index'],
                timestamp=timestamps[c['index']],
                price=c['price'],
                point_type='C'
            )
        )

        self.points.append(
            PatternPoint(
                index=d['index'],
                timestamp=timestamps[d['index']],
                price=d['price'],
                point_type='D'
            )
        )

        self.point_x = self.points[0]
        self.point_a = self.points[1]
        self.point_b = self.points[2]
        self.point_c = self.points[3]
        self.point_d = self.points[4]

        if self.direction == PatternDirection.BULLISH:
            self.xa_price_diff = self.point_a.price - self.point_x.price
            self.ab_retracement = (self.point_a.price - self.point_b.price) / self.xa_price_diff
            self.bc_retracement = (self.point_c.price - self.point_b.price) / (self.point_a.price - self.point_b.price)
            self.cd_extension = (self.point_d.price - self.point_c.price) / (self.point_c.price - self.point_b.price)
            self.xd_extension = (self.point_d.price - self.point_x.price) / self.xa_price_diff
        else:
            self.xa_price_diff = self.point_x.price - self.point_a.price
            self.ab_retracement = (self.point_b.price - self.point_a.price) / self.xa_price_diff
            self.bc_retracement = (self.point_b.price - self.point_c.price) / (self.point_b.price - self.point_a.price)
            self.cd_extension = (self.point_c.price - self.point_d.price) / (self.point_b.price - self.point_c.price)
            self.xd_extension = (self.point_x.price - self.point_d.price) / self.xa_price_diff

    def calculate_quality(self) -> float:
        if not self._is_detected:
            return 0.0

        scores = []

        # 1. Соотношение XD (0.886)
        if self.xd_extension is not None:
            xd_score = 1 - min(abs(self.xd_extension - 0.886) / 0.05, 1.0)
            scores.append(xd_score)

        # 2. Соотношение AB (0.382 или 0.5)
        if self.ab_retracement is not None:
            ab_score_382 = 1 - min(abs(self.ab_retracement - 0.382) / 0.05, 1.0)
            ab_score_5 = 1 - min(abs(self.ab_retracement - 0.5) / 0.05, 1.0)
            ab_score = max(ab_score_382, ab_score_5)
            scores.append(ab_score)

        # 3. Соотношение CD (1.618 или 2.618)
        if self.cd_extension is not None:
            cd_score_1618 = 1 - min(abs(self.cd_extension - 1.618) / 0.15, 1.0)
            cd_score_2618 = 1 - min(abs(self.cd_extension - 2.618) / 0.15, 1.0)
            cd_score = max(cd_score_1618, cd_score_2618)
            scores.append(cd_score)

        return np.mean(scores) if scores else 0.0

    def calculate_targets(self, current_price: float):
        if not self._is_detected or len(self.points) < 5:
            return super().calculate_targets(current_price)

        # Для паттерна Bat
        if self.direction == PatternDirection.BULLISH:
            entry_price = current_price
            stop_loss = self.point_d.price * 0.99

            # Цели: 0.382, 0.618, 0.786 от XA
            target1 = self.point_d.price + (self.point_a.price - self.point_x.price) * 0.382
            target2 = self.point_d.price + (self.point_a.price - self.point_x.price) * 0.618
            target3 = self.point_d.price + (self.point_a.price - self.point_x.price) * 0.786

            self.targets.entry_price = entry_price
            self.targets.stop_loss = stop_loss
            self.targets.take_profit = target2

            self.targets.target1 = target1
            self.targets.target2 = target2
            self.targets.target3 = target3

        else:  # BEARISH
            entry_price = current_price
            stop_loss = self.point_d.price * 1.01

            target1 = self.point_d.price - (self.point_x.price - self.point_a.price) * 0.382
            target2 = self.point_d.price - (self.point_x.price - self.point_a.price) * 0.618
            target3 = self.point_d.price - (self.point_x.price - self.point_a.price) * 0.786

            self.targets.entry_price = entry_price
            self.targets.stop_loss = stop_loss
            self.targets.take_profit = target2

            self.targets.target1 = target1
            self.targets.target2 = target2
            self.targets.target3 = target3

        # Риск/прибыль
        if self.targets.entry_price and self.targets.stop_loss and self.targets.take_profit:
            risk = abs(self.targets.entry_price - self.targets.stop_loss)
            reward = abs(self.targets.take_profit - self.targets.entry_price)
            self.targets.profit_risk_ratio = reward / risk if risk > 0 else 0

        return self.targets


@dataclass
class CrabPattern(HarmonicPattern):
    """Паттерн Crab (Краб)"""

    def __init__(self):
        super().__init__("Crab Pattern", "CRAB")
        self.pattern_variants = ['bullish', 'bearish']
        self.required_points = 5  # X, A, B, C, D

    def detect(self, data, highs, lows, closes, volumes, timestamps, **kwargs) -> bool:
        swing_highs, swing_lows = self._find_swing_points(highs, lows, min_distance=8)

        if len(swing_highs) < 3 or len(swing_lows) < 2:
            return False

        all_points = swing_highs + swing_lows
        all_points.sort(key=lambda x: x['index'])

        for i in range(len(all_points) - 4):
            point_x = all_points[i]
            point_a = all_points[i + 1]
            point_b = all_points[i + 2]
            point_c = all_points[i + 3]
            point_d = all_points[i + 4]

            types = [p['type'] for p in [point_x, point_a, point_b, point_c, point_d]]
            for j in range(len(types) - 1):
                if types[j] == types[j + 1]:
                    break
            else:
                if self._check_crab_ratios(point_x, point_a, point_b, point_c, point_d):
                    self._save_crab_points(point_x, point_a, point_b,
                                           point_c, point_d, timestamps)
                    self._is_detected = True

                    if point_x['type'] == 'low':
                        self.direction = PatternDirection.BULLISH
                    else:
                        self.direction = PatternDirection.BEARISH

                    return True

        return False

    def _check_crab_ratios(self, x, a, b, c, d) -> bool:
        """Проверка фибо соотношений для паттерна Crab"""
        is_bullish = x['type'] == 'low'

        if is_bullish:
            # AB откат: должен быть 0.382-0.618 от XA
            xa_diff = a['price'] - x['price']
            ab_diff = a['price'] - b['price']

            if xa_diff == 0:
                return False

            ab_retracement = ab_diff / xa_diff

            # BC откат: должен быть 0.382-0.886 от AB
            bc_diff = c['price'] - b['price']
            if ab_diff == 0:
                return False
            bc_retracement = bc_diff / ab_diff

            # CD расширение: должно быть 2.618-3.618 от BC
            cd_diff = d['price'] - c['price']
            if bc_diff == 0:
                return False
            cd_extension = cd_diff / bc_diff

            # XD расширение: должно быть 1.618 от XA
            xd_diff = d['price'] - x['price']
            xd_extension = xd_diff / xa_diff

            ab_ok = (self._is_fib_level_match(ab_retracement, 0.382, 0.1) or
                     self._is_fib_level_match(ab_retracement, 0.618, 0.1))
            bc_ok = (self._is_fib_level_match(bc_retracement, 0.382, 0.1) or
                     self._is_fib_level_match(bc_retracement, 0.886, 0.1))
            cd_ok = (self._is_fib_level_match(cd_extension, 2.618, 0.2) or
                     self._is_fib_level_match(cd_extension, 3.618, 0.2))
            xd_ok = self._is_fib_level_match(xd_extension, 1.618, 0.1)

            return ab_ok and bc_ok and cd_ok and xd_ok

        else:  # Медвежий
            xa_diff = x['price'] - a['price']
            ab_diff = b['price'] - a['price']

            if xa_diff == 0:
                return False

            ab_retracement = ab_diff / xa_diff

            bc_diff = b['price'] - c['price']
            if ab_diff == 0:
                return False
            bc_retracement = bc_diff / ab_diff

            cd_diff = c['price'] - d['price']
            if bc_diff == 0:
                return False
            cd_extension = cd_diff / bc_diff

            xd_diff = x['price'] - d['price']
            xd_extension = xd_diff / xa_diff

            ab_ok = (self._is_fib_level_match(ab_retracement, 0.382, 0.1) or
                     self._is_fib_level_match(ab_retracement, 0.618, 0.1))
            bc_ok = (self._is_fib_level_match(bc_retracement, 0.382, 0.1) or
                     self._is_fib_level_match(bc_retracement, 0.886, 0.1))
            cd_ok = (self._is_fib_level_match(cd_extension, 2.618, 0.2) or
                     self._is_fib_level_match(cd_extension, 3.618, 0.2))
            xd_ok = self._is_fib_level_match(xd_extension, 1.618, 0.1)

            return ab_ok and bc_ok and cd_ok and xd_ok

    def _save_crab_points(self, x, a, b, c, d, timestamps):
        """Сохранение точек паттерна Crab"""
        self.points = []

        self.points.append(
            PatternPoint(
                index=x['index'],
                timestamp=timestamps[x['index']],
                price=x['price'],
                point_type='X'
            )
        )

        self.points.append(
            PatternPoint(
                index=a['index'],
                timestamp=timestamps[a['index']],
                price=a['price'],
                point_type='A'
            )
        )

        self.points.append(
            PatternPoint(
                index=b['index'],
                timestamp=timestamps[b['index']],
                price=b['price'],
                point_type='B'
            )
        )

        self.points.append(
            PatternPoint(
                index=c['index'],
                timestamp=timestamps[c['index']],
                price=c['price'],
                point_type='C'
            )
        )

        self.points.append(
            PatternPoint(
                index=d['index'],
                timestamp=timestamps[d['index']],
                price=d['price'],
                point_type='D'
            )
        )

        self.point_x = self.points[0]
        self.point_a = self.points[1]
        self.point_b = self.points[2]
        self.point_c = self.points[3]
        self.point_d = self.points[4]

        if self.direction == PatternDirection.BULLISH:
            self.xa_price_diff = self.point_a.price - self.point_x.price
            self.ab_retracement = (self.point_a.price - self.point_b.price) / self.xa_price_diff
            self.bc_retracement = (self.point_c.price - self.point_b.price) / (self.point_a.price - self.point_b.price)
            self.cd_extension = (self.point_d.price - self.point_c.price) / (self.point_c.price - self.point_b.price)
            self.xd_extension = (self.point_d.price - self.point_x.price) / self.xa_price_diff
        else:
            self.xa_price_diff = self.point_x.price - self.point_a.price
            self.ab_retracement = (self.point_b.price - self.point_a.price) / self.xa_price_diff
            self.bc_retracement = (self.point_b.price - self.point_c.price) / (self.point_b.price - self.point_a.price)
            self.cd_extension = (self.point_c.price - self.point_d.price) / (self.point_b.price - self.point_c.price)
            self.xd_extension = (self.point_x.price - self.point_d.price) / self.xa_price_diff

    def calculate_quality(self) -> float:
        if not self._is_detected:
            return 0.0

        scores = []

        # 1. Соотношение XD (1.618)
        if self.xd_extension is not None:
            xd_score = 1 - min(abs(self.xd_extension - 1.618) / 0.1, 1.0)
            scores.append(xd_score)

        # 2. Соотношение CD (2.618 или 3.618)
        if self.cd_extension is not None:
            cd_score_2618 = 1 - min(abs(self.cd_extension - 2.618) / 0.2, 1.0)
            cd_score_3618 = 1 - min(abs(self.cd_extension - 3.618) / 0.2, 1.0)
            cd_score = max(cd_score_2618, cd_score_3618)
            scores.append(cd_score)

        # 3. Общая симметрия
        if len(self.points) == 5:
            # Время между XA и CD должно быть примерно равным
            time_xa = (self.point_a.timestamp - self.point_x.timestamp).total_seconds()
            time_cd = (self.point_d.timestamp - self.point_c.timestamp).total_seconds()

            if time_xa > 0 and time_cd > 0:
                time_ratio = min(time_xa, time_cd) / max(time_xa, time_cd)
                scores.append(time_ratio)

        return np.mean(scores) if scores else 0.0

    def calculate_targets(self, current_price: float):
        if not self._is_detected or len(self.points) < 5:
            return super().calculate_targets(current_price)

        # Для паттерна Crab
        if self.direction == PatternDirection.BULLISH:
            entry_price = current_price
            stop_loss = self.point_d.price * 0.99

            # Цели: 0.382, 0.618, 1.0 от XA
            target1 = self.point_d.price + (self.point_a.price - self.point_x.price) * 0.382
            target2 = self.point_d.price + (self.point_a.price - self.point_x.price) * 0.618
            target3 = self.point_d.price + (self.point_a.price - self.point_x.price)

            self.targets.entry_price = entry_price
            self.targets.stop_loss = stop_loss
            self.targets.take_profit = target2

            self.targets.target1 = target1
            self.targets.target2 = target2
            self.targets.target3 = target3

        else:  # BEARISH
            entry_price = current_price
            stop_loss = self.point_d.price * 1.01

            target1 = self.point_d.price - (self.point_x.price - self.point_a.price) * 0.382
            target2 = self.point_d.price - (self.point_x.price - self.point_a.price) * 0.618
            target3 = self.point_d.price - (self.point_x.price - self.point_a.price)

            self.targets.entry_price = entry_price
            self.targets.stop_loss = stop_loss
            self.targets.take_profit = target2

            self.targets.target1 = target1
            self.targets.target2 = target2
            self.targets.target3 = target3

        # Риск/прибыль
        if self.targets.entry_price and self.targets.stop_loss and self.targets.take_profit:
            risk = abs(self.targets.entry_price - self.targets.stop_loss)
            reward = abs(self.targets.take_profit - self.targets.entry_price)
            self.targets.profit_risk_ratio = reward / risk if risk > 0 else 0

        return self.targets

