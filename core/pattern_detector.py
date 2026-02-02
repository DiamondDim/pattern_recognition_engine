"""
Основной детектор паттернов
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from scipy.signal import argrelextrema, find_peaks
from dataclasses import dataclass, field
import talib
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

from patterns import (
    create_pattern, get_available_patterns, PatternType, PatternDirection
)
from config import DETECTION_CONFIG, ANALYSIS_CONFIG
from utils.logger import logger


@dataclass
class DetectionResult:
    """Результат детектирования паттернов"""

    patterns: List[Dict[str, Any]] = field(default_factory=list)
    indicators: Dict[str, np.ndarray] = field(default_factory=dict)
    extremums: Dict[str, np.ndarray] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.patterns)

    @property
    def bullish_count(self) -> int:
        return len([p for p in self.patterns if p.get('direction') == 'bullish'])

    @property
    def bearish_count(self) -> int:
        return len([p for p in self.patterns if p.get('direction') == 'bearish'])


class PatternDetector:
    """Класс для детектирования всех типов паттернов"""

    def __init__(self, config: DETECTION_CONFIG = None):
        self.config = config or DETECTION_CONFIG
        self.logger = logger.bind(name="PatternDetector")

        # Инициализация детекторов
        self._init_pattern_detectors()

        # Кэш для оптимизации
        self._cache = {}
        self.cache_enabled = True

        # Статистика
        self.detection_stats = {
            'total_processed': 0,
            'patterns_found': 0,
            'by_type': {},
            'by_direction': {'bullish': 0, 'bearish': 0, 'neutral': 0},
            'avg_quality': 0.0,
            'avg_confidence': 0.0
        }

    def _init_pattern_detectors(self):
        """Инициализация детекторов паттернов"""
        self.pattern_detectors = {}

        # Создаем детекторы для всех включенных паттернов
        all_patterns = (
                self.config.ENABLED_GEOMETRIC +
                self.config.ENABLED_CANDLESTICK +
                self.config.ENABLED_HARMONIC
        )

        for pattern_name in all_patterns:
            try:
                pattern = create_pattern(pattern_name)
                self.pattern_detectors[pattern_name] = pattern

                # Инициализируем статистику
                self.detection_stats['by_type'][pattern_name] = {
                    'count': 0,
                    'avg_quality': 0.0,
                    'success_rate': 0.0
                }

            except Exception as e:
                self.logger.warning(f"Не удалось создать детектор для {pattern_name}: {e}")

        self.logger.info(f"Инициализировано {len(self.pattern_detectors)} детекторов паттернов")

    def find_extremums(self,
                       highs: np.ndarray,
                       lows: np.ndarray,
                       method: str = 'scipy') -> Dict[str, np.ndarray]:
        """
        Нахождение экстремумов

        Args:
            highs: Массив максимумов
            lows: Массив минимумов
            method: Метод поиска ('scipy', 'prominence', 'simple')

        Returns:
            Dict с индексами и ценами экстремумов
        """
        cache_key = f"extremums_{len(highs)}_{len(lows)}_{method}"

        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        result = {
            'maxima': np.array([], dtype=int),
            'minima': np.array([], dtype=int),
            'maxima_prices': np.array([]),
            'minima_prices': np.array([])
        }

        if len(highs) < 10 or len(lows) < 10:
            return result

        try:
            if method == 'scipy':
                # Используем scipy.signal.argrelextrema
                order = self.config.EXTREMA_ORDER

                max_indices = argrelextrema(highs, np.greater, order=order)[0]
                min_indices = argrelextrema(lows, np.less, order=order)[0]

            elif method == 'prominence':
                # Используем scipy.signal.find_peaks с параметром prominence
                prominence_val = np.std(highs) * 0.5  # Автоматический расчет значимости

                max_indices, _ = find_peaks(highs, prominence=prominence_val)
                min_indices, _ = find_peaks(-lows, prominence=prominence_val)

            elif method == 'simple':
                # Простой алгоритм поиска локальных экстремумов
                max_indices = self._simple_extremum_search(highs, is_max=True)
                min_indices = self._simple_extremum_search(lows, is_max=False)

            else:
                raise ValueError(f"Неизвестный метод: {method}")

            # Фильтрация экстремумов по значимости
            if len(max_indices) > 0:
                significant_max = self._filter_significant_extremums(
                    highs, max_indices, is_max=True
                )
                result['maxima'] = significant_max
                result['maxima_prices'] = highs[significant_max]

            if len(min_indices) > 0:
                significant_min = self._filter_significant_extremums(
                    lows, min_indices, is_max=False
                )
                result['minima'] = significant_min
                result['minima_prices'] = lows[significant_min]

            # Сохраняем в кэш
            if self.cache_enabled:
                self._cache[cache_key] = result

            self.logger.debug(f"Найдено {len(result['maxima'])} максимумов и {len(result['minima'])} минимумов")

        except Exception as e:
            self.logger.error(f"Ошибка при поиске экстремумов: {e}")

        return result

    def _simple_extremum_search(self, prices: np.ndarray, is_max: bool = True) -> np.ndarray:
        """Простой алгоритм поиска экстремумов"""
        window = self.config.EXTREMA_ORDER
        indices = []

        for i in range(window, len(prices) - window):
            if is_max:
                if prices[i] == np.max(prices[i - window:i + window + 1]):
                    indices.append(i)
            else:
                if prices[i] == np.min(prices[i - window:i + window + 1]):
                    indices.append(i)

        return np.array(indices)

    def _filter_significant_extremums(self,
                                      prices: np.ndarray,
                                      indices: np.ndarray,
                                      is_max: bool = True) -> np.ndarray:
        """Фильтрация значимых экстремумов"""
        if len(indices) < 3:
            return indices

        filtered = []
        min_prominence = np.std(prices) * self.config.MIN_PROMINENCE_PCT

        for idx in indices:
            # Проверяем значимость по prominence
            if self._calculate_prominence(prices, idx, is_max) >= min_prominence:
                filtered.append(idx)

        return np.array(filtered)

    def _calculate_prominence(self,
                              prices: np.ndarray,
                              idx: int,
                              is_max: bool = True) -> float:
        """Расчет значимости (prominence) экстремума"""
        if idx < 0 or idx >= len(prices):
            return 0.0

        # Для максимума: разница между значением и наивысшей точкой контура
        if is_max:
            # Ищем более низкие точки слева и справа
            left_min = prices[idx]
            right_min = prices[idx]

            # Влево
            for i in range(idx - 1, max(0, idx - 50), -1):
                if prices[i] < left_min:
                    left_min = prices[i]
                elif prices[i] > prices[idx]:
                    break

            # Вправо
            for i in range(idx + 1, min(len(prices), idx + 50)):
                if prices[i] < right_min:
                    right_min = prices[i]
                elif prices[i] > prices[idx]:
                    break

            prominence = prices[idx] - max(left_min, right_min)

        # Для минимума: разница между наинизшей точкой контура и значением
        else:
            # Ищем более высокие точки слева и справа
            left_max = prices[idx]
            right_max = prices[idx]

            # Влево
            for i in range(idx - 1, max(0, idx - 50), -1):
                if prices[i] > left_max:
                    left_max = prices[i]
                elif prices[i] < prices[idx]:
                    break

            # Вправо
            for i in range(idx + 1, min(len(prices), idx + 50)):
                if prices[i] > right_max:
                    right_max = prices[i]
                elif prices[i] < prices[idx]:
                    break

            prominence = min(left_max, right_max) - prices[idx]

        return prominence

    def calculate_indicators(self,
                             opens: np.ndarray,
                             highs: np.ndarray,
                             lows: np.ndarray,
                             closes: np.ndarray,
                             volumes: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Расчет технических индикаторов

        Returns:
            Dict с индикаторами
        """
        indicators = {}

        try:
            # RSI
            if len(closes) > 14:
                indicators['rsi'] = talib.RSI(closes, timeperiod=14)

            # MACD
            if len(closes) > 26:
                macd, macd_signal, macd_hist = talib.MACD(closes)
                indicators['macd'] = macd
                indicators['macd_signal'] = macd_signal
                indicators['macd_hist'] = macd_hist

            # Bollinger Bands
            if len(closes) > 20:
                upper, middle, lower = talib.BBANDS(closes)
                indicators['bb_upper'] = upper
                indicators['bb_middle'] = middle
                indicators['bb_lower'] = lower

            # Stochastic
            if len(highs) > 14 and len(lows) > 14:
                slowk, slowd = talib.STOCH(highs, lows, closes)
                indicators['stoch_k'] = slowk
                indicators['stoch_d'] = slowd

            # ATR (Average True Range)
            if len(highs) > 14 and len(lows) > 14 and len(closes) > 14:
                atr = talib.ATR(highs, lows, closes, timeperiod=14)
                indicators['atr'] = atr

            # ADX (Average Directional Index)
            if len(highs) > 14 and len(lows) > 14 and len(closes) > 14:
                adx = talib.ADX(highs, lows, closes, timeperiod=14)
                indicators['adx'] = adx

            # Volume indicators
            if volumes is not None and len(volumes) > 0:
                if len(volumes) > 20:
                    indicators['volume_sma'] = talib.SMA(volumes, timeperiod=20)

                # OBV (On Balance Volume)
                if len(closes) > 1:
                    obv = talib.OBV(closes, volumes)
                    indicators['obv'] = obv

            # Скользящие средние
            if len(closes) > 50:
                indicators['sma_20'] = talib.SMA(closes, timeperiod=20)
                indicators['sma_50'] = talib.SMA(closes, timeperiod=50)
                indicators['sma_200'] = talib.SMA(closes, timeperiod=200)

            if len(closes) > 20:
                indicators['ema_20'] = talib.EMA(closes, timeperiod=20)

            self.logger.debug(f"Рассчитано {len(indicators)} индикаторов")

        except Exception as e:
            self.logger.error(f"Ошибка расчета индикаторов: {e}")

        return indicators

    def detect_all_patterns(self,
                            symbol: str,
                            timeframe: str,
                            data: Dict[str, np.ndarray],
                            indicators: Optional[Dict[str, np.ndarray]] = None) -> DetectionResult:
        """
        Детектирование всех паттернов на данных

        Args:
            symbol: Символ инструмента
            timeframe: Таймфрейм
            data: Словарь с данными OHLCV
            indicators: Предварительно рассчитанные индикаторы

        Returns:
            DetectionResult с найденными паттернами
        """
        result = DetectionResult()

        # Извлекаем данные
        opens = data.get('open', np.array([]))
        highs = data.get('high', np.array([]))
        lows = data.get('low', np.array([]))
        closes = data.get('close', np.array([]))
        volumes = data.get('volume', np.array([]))
        timestamps = data.get('timestamp', np.array([]))

        # Проверяем достаточность данных
        if len(closes) < self.config.MIN_CANDLES_FOR_PATTERN:
            self.logger.warning(f"Недостаточно данных: {len(closes)} < {self.config.MIN_CANDLES_FOR_PATTERN}")
            return result

        # Рассчитываем индикаторы, если не предоставлены
        if indicators is None:
            indicators = self.calculate_indicators(opens, highs, lows, closes, volumes)

        result.indicators = indicators

        # Находим экстремумы
        extremums = self.find_extremums(highs, lows)
        result.extremums = extremums

        # Анализируем контекст рынка
        market_context = self._analyze_market_context(
            closes, highs, lows, volumes, indicators
        )

        # Детектируем каждый тип паттерна
        detected_patterns = []

        # Геометрические паттерны
        geometric_patterns = self._detect_geometric_patterns(
            symbol, timeframe, opens, highs, lows, closes, volumes,
            timestamps, extremums, market_context, indicators
        )
        detected_patterns.extend(geometric_patterns)

        # Свечные паттерны
        candlestick_patterns = self._detect_candlestick_patterns(
            symbol, timeframe, opens, highs, lows, closes, volumes,
            timestamps, market_context, indicators
        )
        detected_patterns.extend(candlestick_patterns)

        # Гармонические паттерны (будут реализованы позже)
        harmonic_patterns = self._detect_harmonic_patterns(
            symbol, timeframe, opens, highs, lows, closes, volumes,
            timestamps, extremums, market_context, indicators
        )
        detected_patterns.extend(harmonic_patterns)

        # Фильтруем по качеству
        filtered_patterns = []
        for pattern in detected_patterns:
            quality = pattern.get('metadata', {}).get('quality_score', 0)
            confidence = pattern.get('metadata', {}).get('confidence', 0)

            if (quality >= self.config.MIN_PATTERN_QUALITY and
                    confidence >= self.config.CONFIDENCE_THRESHOLD):
                filtered_patterns.append(pattern)

        # Обновляем статистику
        self._update_statistics(filtered_patterns)

        result.patterns = filtered_patterns

        # Добавляем общую статистику
        result.statistics = {
            'total_patterns': len(filtered_patterns),
            'bullish_patterns': len([p for p in filtered_patterns if p.get('direction') == 'bullish']),
            'bearish_patterns': len([p for p in filtered_patterns if p.get('direction') == 'bearish']),
            'avg_quality': np.mean(
                [p.get('metadata', {}).get('quality_score', 0) for p in filtered_patterns]) if filtered_patterns else 0,
            'market_context': market_context,
            'detection_time': datetime.now().isoformat()
        }

        self.logger.info(
            f"Найдено {len(filtered_patterns)} паттернов для {symbol} {timeframe} "
            f"(бычьих: {result.statistics['bullish_patterns']}, "
            f"медвежьих: {result.statistics['bearish_patterns']})"
        )

        return result

    def _analyze_market_context(self,
                                closes: np.ndarray,
                                highs: np.ndarray,
                                lows: np.ndarray,
                                volumes: np.ndarray,
                                indicators: Dict[str, np.ndarray]) -> str:
        """Анализ контекста рынка"""
        if len(closes) < 50:
            return "neutral"

        # Определяем тренд по скользящим средним
        trend = "neutral"

        if 'sma_50' in indicators and 'sma_200' in indicators:
            sma_50 = indicators['sma_50'][-1]
            sma_200 = indicators['sma_200'][-1]

            if not np.isnan(sma_50) and not np.isnan(sma_200):
                if sma_50 > sma_200 * 1.02:  # 2% выше
                    trend = "uptrend"
                elif sma_50 < sma_200 * 0.98:  # 2% ниже
                    trend = "downtrend"

        # Проверяем волатильность
        volatility = np.std(closes[-20:]) / np.mean(closes[-20:]) if len(closes) >= 20 else 0

        # Проверяем ADX для силы тренда
        if 'adx' in indicators:
            adx = indicators['adx'][-1]
            if not np.isnan(adx):
                if adx > 25:
                    trend_strength = "strong"
                elif adx > 20:
                    trend_strength = "moderate"
                else:
                    trend_strength = "weak"

        # Определяем, есть ли боковое движение
        recent_high = np.max(highs[-20:])
        recent_low = np.min(lows[-20:])
        range_pct = (recent_high - recent_low) / recent_low if recent_low > 0 else 0

        if range_pct < 0.02 and trend == "neutral":  # Диапазон менее 2%
            return "sideways"

        return trend

    def _detect_geometric_patterns(self,
                                   symbol: str,
                                   timeframe: str,
                                   opens: np.ndarray,
                                   highs: np.ndarray,
                                   lows: np.ndarray,
                                   closes: np.ndarray,
                                   volumes: np.ndarray,
                                   timestamps: np.ndarray,
                                   extremums: Dict[str, np.ndarray],
                                   market_context: str,
                                   indicators: Dict[str, np.ndarray]) -> List[Dict]:
        """Детектирование геометрических паттернов"""
        patterns = []

        # Создаем копии данных для каждого детектора
        data = {
            'opens': opens,
            'highs': highs,
            'lows': lows,
            'closes': closes,
            'volumes': volumes,
            'timestamps': timestamps
        }

        # Детектируем каждый геометрический паттерн
        for pattern_name in self.config.ENABLED_GEOMETRIC:
            if pattern_name not in self.pattern_detectors:
                continue

            try:
                detector = self.pattern_detectors[pattern_name]

                # Устанавливаем данные
                detector._opens = opens
                detector._highs = highs
                detector._lows = lows
                detector._closes = closes
                detector._volumes = volumes

                # Устанавливаем метаданные
                detector.metadata.symbol = symbol
                detector.metadata.timeframe = timeframe
                detector.metadata.market_context = market_context

                # Устанавливаем значения индикаторов
                if 'rsi' in indicators and len(indicators['rsi']) > 0:
                    detector.metadata.rsi_value = indicators['rsi'][-1]

                if 'macd' in indicators and len(indicators['macd']) > 0:
                    detector.metadata.macd_value = indicators['macd'][-1]

                if 'adx' in indicators and len(indicators['adx']) > 0:
                    detector.metadata.adx_value = indicators['adx'][-1]

                if 'atr' in indicators and len(indicators['atr']) > 0:
                    detector.metadata.atr_value = indicators['atr'][-1]

                # Устанавливаем волатильность и объем
                if len(closes) >= 20:
                    detector.metadata.volatility_pct = np.std(closes[-20:]) / np.mean(closes[-20:])

                if len(volumes) >= 20:
                    detector.metadata.average_volume = np.mean(volumes[-20:])

                # Детектирование
                is_detected = detector.detect(
                    data=np.column_stack((opens, highs, lows, closes, volumes)),
                    highs=highs,
                    lows=lows,
                    closes=closes,
                    volumes=volumes,
                    timestamps=timestamps
                )

                if is_detected:
                    # Рассчитываем качество
                    quality = detector.calculate_quality()
                    detector.metadata.quality_score = quality

                    # Рассчитываем цели
                    current_price = closes[-1]
                    targets = detector.calculate_targets(current_price)

                    # Анализируем силу
                    strength = detector.analyze_strength()

                    # Проверяем объем (если есть)
                    if volumes is not None and len(volumes) > 0:
                        # Проверяем, увеличился ли объем на ключевых точках
                        volume_increase = self._check_volume_confirmation(
                            detector.points, volumes
                        )
                        detector.metadata.volume_confirmation = volume_increase

                    # Конвертируем в словарь
                    pattern_dict = detector.to_dict()
                    pattern_dict['detection_time'] = datetime.now().isoformat()

                    patterns.append(pattern_dict)

                    # Логируем обнаружение
                    self.logger.info(
                        f"Обнаружен геометрический паттерн: {pattern_name} | "
                        f"Качество: {quality:.2f} | "
                        f"Направление: {detector.direction.value}"
                    )

                    # Сбрасываем детектор для следующего использования
                    # Для этого нужно создать новый экземпляр
                    self.pattern_detectors[pattern_name] = create_pattern(pattern_name)

            except Exception as e:
                self.logger.error(f"Ошибка при детектировании паттерна {pattern_name}: {e}")
                # Пересоздаем детектор при ошибке
                self.pattern_detectors[pattern_name] = create_pattern(pattern_name)
                continue

        return patterns

    def _detect_candlestick_patterns(self,
                                     symbol: str,
                                     timeframe: str,
                                     opens: np.ndarray,
                                     highs: np.ndarray,
                                     lows: np.ndarray,
                                     closes: np.ndarray,
                                     volumes: np.ndarray,
                                     timestamps: np.ndarray,
                                     market_context: str,
                                     indicators: Dict[str, np.ndarray]) -> List[Dict]:
        """Детектирование свечных паттернов"""
        patterns = []

        # Используем TA-Lib для некоторых свечных паттернов
        talib_patterns = self._detect_talib_patterns(opens, highs, lows, closes)

        for pattern_name, pattern_result in talib_patterns.items():
            if pattern_result != 0:
                # Создаем паттерн
                try:
                    detector = create_pattern(pattern_name)

                    # Устанавливаем данные
                    detector._opens = opens
                    detector._highs = highs
                    detector._lows = lows
                    detector._closes = closes
                    detector._volumes = volumes

                    # Устанавливаем метаданные
                    detector.metadata.symbol = symbol
                    detector.metadata.timeframe = timeframe
                    detector.metadata.market_context = market_context

                    # Определяем направление
                    if pattern_result > 0:
                        detector.direction = PatternDirection.BULLISH
                    else:
                        detector.direction = PatternDirection.BEARISH

                    # Сохраняем точки (последняя свеча)
                    if len(timestamps) > 0:
                        detector.points.append(
                            PatternPoint(
                                index=len(closes) - 1,
                                timestamp=timestamps[-1],
                                price=closes[-1],
                                point_type='candle_close'
                            )
                        )

                    # Рассчитываем качество
                    quality = detector.calculate_quality()
                    detector.metadata.quality_score = quality
                    detector.metadata.confidence = quality  # Для свечных паттернов качество = уверенность

                    # Рассчитываем цели
                    current_price = closes[-1]
                    targets = detector.calculate_targets(current_price)

                    # Конвертируем в словарь
                    pattern_dict = detector.to_dict()
                    pattern_dict['detection_time'] = datetime.now().isoformat()

                    patterns.append(pattern_dict)

                    self.logger.debug(
                        f"Обнаружен свечной паттерн TA-Lib: {pattern_name} | "
                        f"Направление: {detector.direction.value}"
                    )

                except Exception as e:
                    self.logger.error(f"Ошибка обработки TA-Lib паттерна {pattern_name}: {e}")
                    continue

        # Дополнительные свечные паттерны (наши кастомные)
        for pattern_name in self.config.ENABLED_CANDLESTICK:
            if pattern_name in talib_patterns:
                continue  # Уже обработали через TA-Lib

            if pattern_name not in self.pattern_detectors:
                continue

            try:
                detector = self.pattern_detectors[pattern_name]

                # Устанавливаем данные
                detector._opens = opens
                detector._highs = highs
                detector._lows = lows
                detector._closes = closes
                detector._volumes = volumes

                # Устанавливаем метаданные
                detector.metadata.symbol = symbol
                detector.metadata.timeframe = timeframe
                detector.metadata.market_context = market_context

                # Детектирование
                is_detected = detector.detect(
                    data=np.column_stack((opens, highs, lows, closes, volumes)),
                    highs=highs,
                    lows=lows,
                    closes=closes,
                    volumes=volumes,
                    timestamps=timestamps,
                    opens=opens  # Дополнительный параметр для свечных паттернов
                )

                if is_detected:
                    # Рассчитываем качество
                    quality = detector.calculate_quality()
                    detector.metadata.quality_score = quality
                    detector.metadata.confidence = quality

                    # Рассчитываем цели
                    current_price = closes[-1]
                    targets = detector.calculate_targets(current_price)

                    # Конвертируем в словарь
                    pattern_dict = detector.to_dict()
                    pattern_dict['detection_time'] = datetime.now().isoformat()

                    patterns.append(pattern_dict)

                    self.logger.debug(
                        f"Обнаружен свечной паттерн: {pattern_name} | "
                        f"Качество: {quality:.2f} | "
                        f"Направление: {detector.direction.value}"
                    )

                    # Сбрасываем детектор
                    self.pattern_detectors[pattern_name] = create_pattern(pattern_name)

            except Exception as e:
                self.logger.error(f"Ошибка при детектировании свечного паттерна {pattern_name}: {e}")
                self.pattern_detectors[pattern_name] = create_pattern(pattern_name)
                continue

        return patterns

    def _detect_talib_patterns(self,
                               opens: np.ndarray,
                               highs: np.ndarray,
                               lows: np.ndarray,
                               closes: np.ndarray) -> Dict[str, int]:
        """Детектирование свечных паттернов с помощью TA-Lib"""
        patterns = {}

        try:
            # Bullish patterns
            patterns['engulfing_bullish'] = talib.CDLENGULFING(opens, highs, lows, closes)[-1]
            patterns['hammer'] = talib.CDLHAMMER(opens, highs, lows, closes)[-1]
            patterns['morning_star'] = talib.CDLMORNINGSTAR(opens, highs, lows, closes)[-1]
            patterns['piercing_pattern'] = talib.CDLPIERCING(opens, highs, lows, closes)[-1]
            patterns['three_white_soldiers'] = talib.CDL3WHITESOLDIERS(opens, highs, lows, closes)[-1]

            # Bearish patterns
            patterns['engulfing_bearish'] = 0  # Обрабатывается через engulfing_bullish
            patterns['hanging_man'] = talib.CDLHANGINGMAN(opens, highs, lows, closes)[-1]
            patterns['evening_star'] = talib.CDLEVENINGSTAR(opens, highs, lows, closes)[-1]
            patterns['dark_cloud_cover'] = talib.CDLDARKCLOUDCOVER(opens, highs, lows, closes)[-1]
            patterns['three_black_crows'] = talib.CDL3BLACKCROWS(opens, highs, lows, closes)[-1]

            # Doji patterns
            patterns['doji'] = talib.CDLDOJI(opens, highs, lows, closes)[-1]
            patterns['long_legged_doji'] = talib.CDLLONGLEGGEDDOJI(opens, highs, lows, closes)[-1]
            patterns['dragonfly_doji'] = talib.CDLDRAGONFLYDOJI(opens, highs, lows, closes)[-1]
            patterns['gravestone_doji'] = talib.CDLGRAVESTONEDOJI(opens, highs, lows, closes)[-1]

        except Exception as e:
            self.logger.error(f"Ошибка TA-Lib: {e}")

        return patterns

    def _detect_harmonic_patterns(self,
                                  symbol: str,
                                  timeframe: str,
                                  opens: np.ndarray,
                                  highs: np.ndarray,
                                  lows: np.ndarray,
                                  closes: np.ndarray,
                                  volumes: np.ndarray,
                                  timestamps: np.ndarray,
                                  extremums: Dict[str, np.ndarray],
                                  market_context: str,
                                  indicators: Dict[str, np.ndarray]) -> List[Dict]:
        """Детектирование гармонических паттернов (заглушка)"""
        # TODO: Реализовать гармонические паттерны
        return []

    def _check_volume_confirmation(self,
                                   points: List[PatternPoint],
                                   volumes: np.ndarray) -> bool:
        """Проверка подтверждения объемом"""
        if not points or len(volumes) == 0:
            return False

        # Проверяем объем на ключевых точках
        volume_increases = []

        for point in points:
            idx = point.index

            if 0 <= idx < len(volumes):
                # Сравниваем с средним объемом предыдущих периодов
                start_idx = max(0, idx - 10)
                avg_volume = np.mean(volumes[start_idx:idx])

                if avg_volume > 0:
                    volume_ratio = volumes[idx] / avg_volume
                    volume_increases.append(volume_ratio > 1.2)  # 20% выше среднего

        if not volume_increases:
            return False

        # Если на большинстве точек объем увеличился
        return sum(volume_increases) / len(volume_increases) >= 0.5

    def _update_statistics(self, patterns: List[Dict]):
        """Обновление статистики детектирования"""
        self.detection_stats['total_processed'] += 1
        self.detection_stats['patterns_found'] += len(patterns)

        if patterns:
            qualities = [p.get('metadata', {}).get('quality_score', 0) for p in patterns]
            confidences = [p.get('metadata', {}).get('confidence', 0) for p in patterns]

            self.detection_stats['avg_quality'] = np.mean(qualities)
            self.detection_stats['avg_confidence'] = np.mean(confidences)

            # Статистика по типам и направлениям
            for pattern in patterns:
                pattern_name = pattern.get('name', 'unknown')
                direction = pattern.get('direction', 'neutral')

                # По типам
                if pattern_name not in self.detection_stats['by_type']:
                    self.detection_stats['by_type'][pattern_name] = {
                        'count': 0,
                        'avg_quality': 0.0,
                        'success_rate': 0.0
                    }

                stats = self.detection_stats['by_type'][pattern_name]
                stats['count'] += 1
                stats['avg_quality'] = (
                        (stats['avg_quality'] * (stats['count'] - 1) +
                         pattern.get('metadata', {}).get('quality_score', 0)) / stats['count']
                )

                # По направлениям
                if direction in self.detection_stats['by_direction']:
                    self.detection_stats['by_direction'][direction] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики детектирования"""
        return self.detection_stats.copy()

    def reset_statistics(self):
        """Сброс статистики"""
        self.detection_stats = {
            'total_processed': 0,
            'patterns_found': 0,
            'by_type': {},
            'by_direction': {'bullish': 0, 'bearish': 0, 'neutral': 0},
            'avg_quality': 0.0,
            'avg_confidence': 0.0
        }

        # Сброс статистики по типам
        for pattern_name in self.pattern_detectors.keys():
            self.detection_stats['by_type'][pattern_name] = {
                'count': 0,
                'avg_quality': 0.0,
                'success_rate': 0.0
            }

    def clear_cache(self):
        """Очистка кэша"""
        self._cache.clear()

