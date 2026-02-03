"""
Модуль свечных паттернов
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import talib

from .base_pattern import BasePattern, PatternPoint, PatternResult
from config import config

@dataclass
class Candle:
    """Свеча"""
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None

class CandlestickPatterns(BasePattern):
    """Класс для детектирования свечных паттернов"""

    def __init__(self):
        super().__init__(name="candlestick_patterns", min_points=1)

        # Паттерны которые мы будем искать
        self.patterns = {
            'doji': self._detect_doji,
            'hammer': self._detect_hammer,
            'shooting_star': self._detect_shooting_star,
            'engulfing': self._detect_engulfing,
            'harami': self._detect_harami,
            'morning_star': self._detect_morning_star,
            'evening_star': self._detect_evening_star,
            'three_white_soldiers': self._detect_three_white_soldiers,
            'three_black_crows': self._detect_three_black_crows,
            'piercing_line': self._detect_piercing_line,
            'dark_cloud_cover': self._detect_dark_cloud_cover
        }

    def detect(self, data: Dict[str, np.ndarray]) -> List[PatternResult]:
        """
        Детектирование всех свечных паттернов

        Args:
            data: Входные данные OHLC

        Returns:
            Список обнаруженных паттернов
        """
        results = []

        # Проверка входных данных
        required = ['open', 'high', 'low', 'close']
        if not all(key in data for key in required):
            return results

        # Создаем массив свечей
        candles = self._create_candles(data)

        # Ищем каждый паттерн
        for pattern_name, detector in self.patterns.items():
            if config.DETECTION.ENABLE_CANDLESTICK:
                pattern_results = detector(candles, data)
                results.extend(pattern_results)

        # Фильтрация результатов
        filtered_results = self._filter_results(results)

        return filtered_results

    def _create_candles(self, data: Dict[str, np.ndarray]) -> List[Candle]:
        """Создание списка свечей из данных"""
        candles = []

        opens = data['open']
        highs = data['high']
        lows = data['low']
        closes = data['close']
        volumes = data.get('volume', np.zeros_like(opens))

        n = min(len(opens), len(highs), len(lows), len(closes))

        for i in range(n):
            candle = Candle(
                open=opens[i],
                high=highs[i],
                low=lows[i],
                close=closes[i],
                volume=volumes[i] if i < len(volumes) else None
            )
            candles.append(candle)

        return candles

    def _detect_doji(self, candles: List[Candle], data: Dict[str, np.ndarray]) -> List[PatternResult]:
        """Детектирование Дожи"""
        results = []

        for i in range(len(candles)):
            if i < 1:
                continue

            candle = candles[i]
            body_size = abs(candle.close - candle.open)
            total_range = candle.high - candle.low

            if total_range == 0:
                continue

            # Дожи - маленькое тело и длинные тени
            body_to_range_ratio = body_size / total_range

            if body_to_range_ratio < 0.1:  # Тело меньше 10% от общего диапазона
                # Проверяем тренд
                prev_candle = candles[i-1]

                # Определение направления
                if prev_candle.close > prev_candle.open:
                    direction = 'bearish'  # Бычий тренд, затем Дожи - возможен разворот
                else:
                    direction = 'bullish'  # Медвежий тренд, затем Дожи - возможен разворот

                # Создаем точку паттерна
                point = PatternPoint(
                    index=i,
                    price=candle.close,
                    type='doji',
                    metadata={
                        'body_size': float(body_size),
                        'total_range': float(total_range),
                        'body_ratio': float(body_to_range_ratio)
                    }
                )

                # Расчет качества
                quality = 1.0 - body_to_range_ratio  # Чем меньше тело, тем лучше

                # Расчет уверенности
                volume_factor = 1.0
                if candle.volume and i > 0 and candles[i-1].volume:
                    volume_ratio = candle.volume / candles[i-1].volume
                    volume_factor = min(1.0, volume_ratio / 2.0)  # Высокий объем увеличивает уверенность

                confidence = 0.6 * volume_factor

                # Создаем результат
                result = self.create_result(
                    name='doji',
                    points=[point],
                    quality=quality,
                    confidence=confidence,
                    targets={
                        'entry_price': candle.close,
                        'stop_loss': candle.high if direction == 'bearish' else candle.low,
                        'take_profit': candle.close + (candle.high - candle.low) if direction == 'bullish'
                                     else candle.close - (candle.high - candle.low)
                    }
                )

                results.append(result)

        return results

    def _detect_hammer(self, candles: List[Candle], data: Dict[str, np.ndarray]) -> List[PatternResult]:
        """Детектирование Молота"""
        results = []

        for i in range(len(candles)):
            if i < 1:
                continue

            candle = candles[i]
            body_size = abs(candle.close - candle.open)
            total_range = candle.high - candle.low

            if total_range == 0:
                continue

            upper_shadow = candle.high - max(candle.open, candle.close)
            lower_shadow = min(candle.open, candle.close) - candle.low
            body_middle = (candle.open + candle.close) / 2

            # Критерии Молота:
            # 1. Длинная нижняя тень (минимум в 2 раза больше тела)
            # 2. Маленькая или отсутствующая верхняя тень
            # 3. Тело в верхней трети свечи
            # 4. Предыдущий тренд нисходящий

            if (lower_shadow >= 2 * body_size and  # Длинная нижняя тень
                upper_shadow <= body_size * 0.3 and  # Короткая верхняя тень
                body_middle > candle.low + total_range * 0.6):  # Тело в верхней части

                # Проверяем предыдущий тренд
                if i >= 2:
                    prev_trend = self._check_downtrend(candles, i, lookback=5)

                    if prev_trend:
                        # Создаем точку паттерна
                        point = PatternPoint(
                            index=i,
                            price=candle.close,
                            type='hammer',
                            metadata={
                                'body_size': float(body_size),
                                'lower_shadow': float(lower_shadow),
                                'upper_shadow': float(upper_shadow)
                            }
                        )

                        # Расчет качества
                        quality = min(1.0, lower_shadow / (body_size + 0.0001))

                        # Расчет уверенности
                        confidence = 0.7

                        # Создаем результат
                        result = self.create_result(
                            name='hammer',
                            points=[point],
                            quality=quality,
                            confidence=confidence,
                            targets={
                                'entry_price': candle.close,
                                'stop_loss': candle.low * 0.995,
                                'take_profit': candle.close + (candle.high - candle.low) * 2
                            }
                        )

                        results.append(result)

        return results

    def _detect_shooting_star(self, candles: List[Candle], data: Dict[str, np.ndarray]) -> List[PatternResult]:
        """Детектирование Падающей звезды"""
        results = []

        for i in range(len(candles)):
            if i < 1:
                continue

            candle = candles[i]
            body_size = abs(candle.close - candle.open)
            total_range = candle.high - candle.low

            if total_range == 0:
                continue

            upper_shadow = candle.high - max(candle.open, candle.close)
            lower_shadow = min(candle.open, candle.close) - candle.low
            body_middle = (candle.open + candle.close) / 2

            # Критерии Падающей звезды:
            # 1. Длинная верхняя тень (минимум в 2 раза больше тела)
            # 2. Маленькая или отсутствующая нижняя тень
            # 3. Тело в нижней трети свечи
            # 4. Предыдущий тренд восходящий

            if (upper_shadow >= 2 * body_size and  # Длинная верхняя тень
                lower_shadow <= body_size * 0.3 and  # Короткая нижняя тень
                body_middle < candle.low + total_range * 0.4):  # Тело в нижней части

                # Проверяем предыдущий тренд
                if i >= 2:
                    prev_trend = self._check_uptrend(candles, i, lookback=5)

                    if prev_trend:
                        # Создаем точку паттерна
                        point = PatternPoint(
                            index=i,
                            price=candle.close,
                            type='shooting_star',
                            metadata={
                                'body_size': float(body_size),
                                'lower_shadow': float(lower_shadow),
                                'upper_shadow': float(upper_shadow)
                            }
                        )

                        # Расчет качества
                        quality = min(1.0, upper_shadow / (body_size + 0.0001))

                        # Расчет уверенности
                        confidence = 0.7

                        # Создаем результат
                        result = self.create_result(
                            name='shooting_star',
                            points=[point],
                            quality=quality,
                            confidence=confidence,
                            targets={
                                'entry_price': candle.close,
                                'stop_loss': candle.high * 1.005,
                                'take_profit': candle.close - (candle.high - candle.low) * 2
                            }
                        )

                        results.append(result)

        return results

    def _detect_engulfing(self, candles: List[Candle], data: Dict[str, np.ndarray]) -> List[PatternResult]:
        """Детектирование поглощающей свечи"""
        results = []

        for i in range(len(candles)):
            if i < 1:
                continue

            current = candles[i]
            previous = candles[i-1]

            current_body = abs(current.close - current.open)
            previous_body = abs(previous.close - previous.open)

            # Бычья поглощающая свеча
            if (current.close > current.open and  # Текущая свеча бычья
                previous.close < previous.open and  # Предыдущая свеча медвежья
                current.open < previous.close and  # Открытие текущей ниже закрытия предыдущей
                current.close > previous.open):    # Закрытие текущей выше открытия предыдущей

                # Создаем точки паттерна
                point1 = PatternPoint(index=i-1, price=previous.close, type='engulfed')
                point2 = PatternPoint(index=i, price=current.close, type='engulfing')

                # Расчет качества
                engulf_ratio = current_body / previous_body
                quality = min(1.0, engulf_ratio / 2.0)

                # Создаем результат
                result = self.create_result(
                    name='bullish_engulfing',
                    points=[point1, point2],
                    quality=quality,
                    confidence=0.7,
                    targets={
                        'entry_price': current.close,
                        'stop_loss': current.low * 0.995,
                        'take_profit': current.close + current_body * 2
                    }
                )

                results.append(result)

            # Медвежья поглощающая свеча
            elif (current.close < current.open and  # Текущая свеча медвежья
                  previous.close > previous.open and  # Предыдущая свеча бычья
                  current.open > previous.close and  # Открытие текущей выше закрытия предыдущей
                  current.close < previous.open):    # Закрытие текущей ниже открытия предыдущей

                # Создаем точки паттерна
                point1 = PatternPoint(index=i-1, price=previous.close, type='engulfed')
                point2 = PatternPoint(index=i, price=current.close, type='engulfing')

                # Расчет качества
                engulf_ratio = current_body / previous_body
                quality = min(1.0, engulf_ratio / 2.0)

                # Создаем результат
                result = self.create_result(
                    name='bearish_engulfing',
                    points=[point1, point2],
                    quality=quality,
                    confidence=0.7,
                    targets={
                        'entry_price': current.close,
                        'stop_loss': current.high * 1.005,
                        'take_profit': current.close - current_body * 2
                    }
                )

                results.append(result)

        return results

    def _detect_harami(self, candles: List[Candle], data: Dict[str, np.ndarray]) -> List[PatternResult]:
        """Детектирование Харами"""
        results = []

        for i in range(len(candles)):
            if i < 1:
                continue

            current = candles[i]
            previous = candles[i-1]

            current_high = current.high
            current_low = current.low
            previous_high = previous.high
            previous_low = previous.low

            # Бычья Харами
            if (previous.close < previous.open and  # Предыдущая свеча медвежья
                current.high < previous.open and    # Максимум текущей ниже открытия предыдущей
                current.low > previous.close and    # Минимум текущей выше закрытия предыдущей
                current.close > current.open):      # Текущая свеча бычья

                # Создаем точки паттерна
                point1 = PatternPoint(index=i-1, price=previous.close, type='parent')
                point2 = PatternPoint(index=i, price=current.close, type='child')

                # Расчет качества
                quality = 0.7

                # Создаем результат
                result = self.create_result(
                    name='bullish_harami',
                    points=[point1, point2],
                    quality=quality,
                    confidence=0.6,
                    targets={
                        'entry_price': current.close,
                        'stop_loss': current.low * 0.995,
                        'take_profit': current.close + (previous.open - previous.close)
                    }
                )

                results.append(result)

            # Медвежья Харами
            elif (previous.close > previous.open and  # Предыдущая свеча бычья
                  current.high < previous.close and   # Максимум текущей ниже закрытия предыдущей
                  current.low > previous.open and     # Минимум текущей выше открытия предыдущей
                  current.close < current.open):      # Текущая свеча медвежья

                # Создаем точки паттерна
                point1 = PatternPoint(index=i-1, price=previous.close, type='parent')
                point2 = PatternPoint(index=i, price=current.close, type='child')

                # Расчет качества
                quality = 0.7

                # Создаем результат
                result = self.create_result(
                    name='bearish_harami',
                    points=[point1, point2],
                    quality=quality,
                    confidence=0.6,
                    targets={
                        'entry_price': current.close,
                        'stop_loss': current.high * 1.005,
                        'take_profit': current.close - (previous.close - previous.open)
                    }
                )

                results.append(result)

        return results

    def _detect_morning_star(self, candles: List[Candle], data: Dict[str, np.ndarray]) -> List[PatternResult]:
        """Детектирование Утренней звезды"""
        results = []

        for i in range(len(candles)):
            if i < 2:
                continue

            first = candles[i-2]   # Медвежья свеча
            second = candles[i-1]  # Маленькая свеча (дожи или с маленьким телом)
            third = candles[i]     # Бычья свеча

            # Проверяем условия
            first_is_bearish = first.close < first.open
            third_is_bullish = third.close > third.open

            if not (first_is_bearish and third_is_bullish):
                continue

            # Вторая свеча - маленькое тело
            second_body = abs(second.close - second.open)
            second_range = second.high - second.low

            if second_range == 0 or second_body / second_range > 0.3:
                continue

            # Гэпы
            gap_down = second.high < first.close
            gap_up = third.low > second.high

            if gap_down and gap_up:
                # Создаем точки паттерна
                point1 = PatternPoint(index=i-2, price=first.close, type='first')
                point2 = PatternPoint(index=i-1, price=second.close, type='star')
                point3 = PatternPoint(index=i, price=third.close, type='third')

                # Расчет качества
                quality = 0.8

                # Создаем результат
                result = self.create_result(
                    name='morning_star',
                    points=[point1, point2, point3],
                    quality=quality,
                    confidence=0.7,
                    targets={
                        'entry_price': third.close,
                        'stop_loss': third.low * 0.995,
                        'take_profit': third.close + abs(first.close - third.close)
                    }
                )

                results.append(result)

        return results

    def _detect_evening_star(self, candles: List[Candle], data: Dict[str, np.ndarray]) -> List[PatternResult]:
        """Детектирование Вечерней звезды"""
        results = []

        for i in range(len(candles)):
            if i < 2:
                continue

            first = candles[i-2]   # Бычья свеча
            second = candles[i-1]  # Маленькая свеча (дожи или с маленьким телом)
            third = candles[i]     # Медвежья свеча

            # Проверяем условия
            first_is_bullish = first.close > first.open
            third_is_bearish = third.close < third.open

            if not (first_is_bullish and third_is_bearish):
                continue

            # Вторая свеча - маленькое тело
            second_body = abs(second.close - second.open)
            second_range = second.high - second.low

            if second_range == 0 or second_body / second_range > 0.3:
                continue

            # Гэпы
            gap_up = second.low > first.close
            gap_down = third.high < second.low

            if gap_up and gap_down:
                # Создаем точки паттерна
                point1 = PatternPoint(index=i-2, price=first.close, type='first')
                point2 = PatternPoint(index=i-1, price=second.close, type='star')
                point3 = PatternPoint(index=i, price=third.close, type='third')

                # Расчет качества
                quality = 0.8

                # Создаем результат
                result = self.create_result(
                    name='evening_star',
                    points=[point1, point2, point3],
                    quality=quality,
                    confidence=0.7,
                    targets={
                        'entry_price': third.close,
                        'stop_loss': third.high * 1.005,
                        'take_profit': third.close - abs(first.close - third.close)
                    }
                )

                results.append(result)

        return results

    def _detect_three_white_soldiers(self, candles: List[Candle], data: Dict[str, np.ndarray]) -> List[PatternResult]:
        """Детектирование Трех белых солдат"""
        results = []

        for i in range(len(candles)):
            if i < 3:
                continue

            # Проверяем три последовательные бычьи свечи
            candles_to_check = candles[i-3:i]

            all_bullish = all(c.close > c.open for c in candles_to_check)
            if not all_bullish:
                continue

            # Проверяем, что каждая следующая свеча закрывается выше предыдущей
            closes = [c.close for c in candles_to_check]
            if not all(closes[j] > closes[j-1] for j in range(1, len(closes))):
                continue

            # Проверяем, что тела свечей примерно одинакового размера
            bodies = [abs(c.close - c.open) for c in candles_to_check]
            avg_body = np.mean(bodies)
            if any(abs(b - avg_body) > avg_body * 0.5 for b in bodies):
                continue

            # Создаем точки паттерна
            points = [
                PatternPoint(index=i-3+j, price=c.close, type=f'soldier_{j+1}')
                for j, c in enumerate(candles_to_check)
            ]

            # Расчет качества
            quality = 0.8

            # Создаем результат
            result = self.create_result(
                name='three_white_soldiers',
                points=points,
                quality=quality,
                confidence=0.75,
                targets={
                    'entry_price': candles_to_check[-1].close,
                    'stop_loss': candles_to_check[0].low * 0.995,
                    'take_profit': candles_to_check[-1].close + avg_body * 3
                }
            )

            results.append(result)

        return results

    def _detect_three_black_crows(self, candles: List[Candle], data: Dict[str, np.ndarray]) -> List[PatternResult]:
        """Детектирование Трех черных ворон"""
        results = []

        for i in range(len(candles)):
            if i < 3:
                continue

            # Проверяем три последовательные медвежьи свечи
            candles_to_check = candles[i-3:i]

            all_bearish = all(c.close < c.open for c in candles_to_check)
            if not all_bearish:
                continue

            # Проверяем, что каждая следующая свеча закрывается ниже предыдущей
            closes = [c.close for c in candles_to_check]
            if not all(closes[j] < closes[j-1] for j in range(1, len(closes))):
                continue

            # Проверяем, что тела свечей примерно одинакового размера
            bodies = [abs(c.close - c.open) for c in candles_to_check]
            avg_body = np.mean(bodies)
            if any(abs(b - avg_body) > avg_body * 0.5 for b in bodies):
                continue

            # Создаем точки паттерна
            points = [
                PatternPoint(index=i-3+j, price=c.close, type=f'crow_{j+1}')
                for j, c in enumerate(candles_to_check)
            ]

            # Расчет качества
            quality = 0.8

            # Создаем результат
            result = self.create_result(
                name='three_black_crows',
                points=points,
                quality=quality,
                confidence=0.75,
                targets={
                    'entry_price': candles_to_check[-1].close,
                    'stop_loss': candles_to_check[0].high * 1.005,
                    'take_profit': candles_to_check[-1].close - avg_body * 3
                }
            )

            results.append(result)

        return results

    def _detect_piercing_line(self, candles: List[Candle], data: Dict[str, np.ndarray]) -> List[PatternResult]:
        """Детектирование Пронизывающей линии"""
        results = []

        for i in range(len(candles)):
            if i < 1:
                continue

            current = candles[i]
            previous = candles[i-1]

            # Предыдущая свеча должна быть медвежьей
            if not (previous.close < previous.open):
                continue

            # Текущая свеча должна быть бычьей
            if not (current.close > current.open):
                continue

            # Текущая свеча должна открыться ниже закрытия предыдущей
            if current.open >= previous.close:
                continue

            # Текущая свеча должна закрыться выше середины тела предыдущей свечи
            previous_mid = (previous.open + previous.close) / 2
            if current.close <= previous_mid:
                continue

            # Но ниже открытия предыдущей свечи
            if current.close >= previous.open:
                continue

            # Создаем точки паттерна
            point1 = PatternPoint(index=i-1, price=previous.close, type='bearish')
            point2 = PatternPoint(index=i, price=current.close, type='piercing')

            # Расчет качества
            penetration = (current.close - previous_mid) / (previous.open - previous_mid)
            quality = min(1.0, penetration)

            # Создаем результат
            result = self.create_result(
                name='piercing_line',
                points=[point1, point2],
                quality=quality,
                confidence=0.7,
                targets={
                    'entry_price': current.close,
                    'stop_loss': current.low * 0.995,
                    'take_profit': current.close + abs(previous.close - current.close) * 2
                }
            )

            results.append(result)

        return results

    def _detect_dark_cloud_cover(self, candles: List[Candle], data: Dict[str, np.ndarray]) -> List[PatternResult]:
        """Детектирование Тучки"""
        results = []

        for i in range(len(candles)):
            if i < 1:
                continue

            current = candles[i]
            previous = candles[i-1]

            # Предыдущая свеча должна быть бычьей
            if not (previous.close > previous.open):
                continue

            # Текущая свеча должна быть медвежьей
            if not (current.close < current.open):
                continue

            # Текущая свеча должна открыться выше закрытия предыдущей
            if current.open <= previous.close:
                continue

            # Текущая свеча должна закрыться ниже середины тела предыдущей свечи
            previous_mid = (previous.open + previous.close) / 2
            if current.close >= previous_mid:
                continue

            # Но выше открытия предыдущей свечи
            if current.close <= previous.open:
                continue

            # Создаем точки паттерна
            point1 = PatternPoint(index=i-1, price=previous.close, type='bullish')
            point2 = PatternPoint(index=i, price=current.close, type='dark_cloud')

            # Расчет качества
            penetration = (previous_mid - current.close) / (previous_mid - previous.open)
            quality = min(1.0, penetration)

            # Создаем результат
            result = self.create_result(
                name='dark_cloud_cover',
                points=[point1, point2],
                quality=quality,
                confidence=0.7,
                targets={
                    'entry_price': current.close,
                    'stop_loss': current.high * 1.005,
                    'take_profit': current.close - abs(previous.close - current.close) * 2
                }
            )

            results.append(result)

        return results

    def _check_downtrend(self, candles: List[Candle], current_idx: int, lookback: int = 5) -> bool:
        """Проверка нисходящего тренда"""
        if current_idx < lookback:
            return False

        prices = [c.close for c in candles[current_idx-lookback:current_idx]]

        # Простая проверка тренда
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)

        return slope < 0

    def _check_uptrend(self, candles: List[Candle], current_idx: int, lookback: int = 5) -> bool:
        """Проверка восходящего тренда"""
        if current_idx < lookback:
            return False

        prices = [c.close for c in candles[current_idx-lookback:current_idx]]

        # Простая проверка тренда
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)

        return slope > 0

    def _filter_results(self, results: List[PatternResult]) -> List[PatternResult]:
        """Фильтрация результатов"""
        if not results:
            return []

        # Фильтруем по минимальному качеству
        filtered = [r for r in results if r.quality >= config.DETECTION.MIN_PATTERN_QUALITY]

        # Фильтруем по минимальной уверенности
        filtered = [r for r in filtered if r.confidence >= config.DETECTION.CONFIDENCE_THRESHOLD]

        # Убираем дубликаты (паттерны на одних и тех же свечах)
        seen_indices = set()
        unique_results = []

        for result in filtered:
            # Создаем ключ на основе индексов свечей
            indices = tuple(sorted(p.index for p in result.points))
            if indices not in seen_indices:
                seen_indices.add(indices)
                unique_results.append(result)

        return unique_results

