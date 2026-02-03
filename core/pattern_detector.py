"""
Модуль детекции паттернов
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
from enum import Enum
import hashlib

from config import DETECTION_CONFIG
from utils.logger import logger
from utils.helpers import generate_id, validate_ohlc_data, find_support_resistance
from patterns.base_pattern import BasePattern
from patterns.candlestick_patterns import CandlestickPatterns
from patterns.geometric_patterns import GeometricPatterns
from patterns.harmonic_patterns import HarmonicPatterns


class PatternType(Enum):
    """Типы паттернов"""
    CANDLESTICK = 'candlestick'
    GEOMETRIC = 'geometric'
    HARMONIC = 'harmonic'
    TECHNICAL = 'technical'


@dataclass
class DetectionResult:
    """Результат детекции паттерна"""
    pattern_type: PatternType
    pattern_name: str
    pattern_data: Dict[str, Any]
    confidence: float
    position: Tuple[int, int]  # start_idx, end_idx
    metadata: Dict[str, Any]


class PatternDetector:
    """Детектор паттернов"""

    def __init__(self, config: DETECTION_CONFIG = None):
        self.config = config or DETECTION_CONFIG
        self.logger = logger.bind(module="PatternDetector")

        # Инициализация детекторов
        self.candlestick_detector = CandlestickPatterns(config)
        self.geometric_detector = GeometricPatterns(config)
        self.harmonic_detector = HarmonicPatterns(config)

        # Кэш обнаруженных паттернов
        self.detected_patterns = []
        self.pattern_cache = {}

    def detect_all_patterns(self,
                           ohlc_data: Dict[str, np.ndarray],
                           indicators: Optional[Dict[str, np.ndarray]] = None) -> List[DetectionResult]:
        """
        Детекция всех типов паттернов

        Args:
            ohlc_data: Данные OHLC
            indicators: Технические индикаторы

        Returns:
            Список обнаруженных паттернов
        """
        try:
            # Валидация данных
            if not validate_ohlc_data(ohlc_data):
                self.logger.error("Невалидные данные OHLC")
                return []

            self.logger.info(f"Начало детекции паттернов на {len(ohlc_data['close'])} барах")

            results = []

            # Детекция свечных паттернов
            candlestick_patterns = self._detect_candlestick_patterns(ohlc_data)
            results.extend(candlestick_patterns)

            # Детекция геометрических паттернов
            geometric_patterns = self._detect_geometric_patterns(ohlc_data)
            results.extend(geometric_patterns)

            # Детекция гармонических паттернов
            harmonic_patterns = self._detect_harmonic_patterns(ohlc_data)
            results.extend(harmonic_patterns)

            # Детекция технических паттернов (на основе индикаторов)
            if indicators:
                technical_patterns = self._detect_technical_patterns(ohlc_data, indicators)
                results.extend(technical_patterns)

            # Сортировка по уверенности
            results.sort(key=lambda x: x.confidence, reverse=True)

            # Фильтрация пересекающихся паттернов
            filtered_results = self._filter_overlapping_patterns(results)

            self.logger.info(f"Обнаружено паттернов: {len(filtered_results)}")
            self.detected_patterns = filtered_results

            return filtered_results

        except Exception as e:
            self.logger.error(f"Ошибка детекции паттернов: {e}")
            return []

    def _detect_candlestick_patterns(self, ohlc_data: Dict[str, np.ndarray]) -> List[DetectionResult]:
        """Детекция свечных паттернов"""
        try:
            patterns = self.candlestick_detector.detect_all(ohlc_data)
            results = []

            for pattern in patterns:
                detection_result = DetectionResult(
                    pattern_type=PatternType.CANDLESTICK,
                    pattern_name=pattern.get('name', 'unknown'),
                    pattern_data=pattern,
                    confidence=pattern.get('confidence', 0.5),
                    position=(pattern.get('start_index', 0), pattern.get('end_index', 0)),
                    metadata={
                        'symbol': pattern.get('symbol', 'UNKNOWN'),
                        'timeframe': pattern.get('timeframe', 'UNKNOWN'),
                        'timestamp': datetime.now().isoformat()
                    }
                )
                results.append(detection_result)

            self.logger.debug(f"Обнаружено свечных паттернов: {len(results)}")
            return results

        except Exception as e:
            self.logger.error(f"Ошибка детекции свечных паттернов: {e}")
            return []

    def _detect_geometric_patterns(self, ohlc_data: Dict[str, np.ndarray]) -> List[DetectionResult]:
        """Детекция геометрических паттернов"""
        try:
            patterns = self.geometric_detector.detect_all(ohlc_data)
            results = []

            for pattern in patterns:
                detection_result = DetectionResult(
                    pattern_type=PatternType.GEOMETRIC,
                    pattern_name=pattern.get('name', 'unknown'),
                    pattern_data=pattern,
                    confidence=pattern.get('confidence', 0.5),
                    position=(pattern.get('start_index', 0), pattern.get('end_index', 0)),
                    metadata={
                        'symbol': pattern.get('symbol', 'UNKNOWN'),
                        'timeframe': pattern.get('timeframe', 'UNKNOWN'),
                        'timestamp': datetime.now().isoformat()
                    }
                )
                results.append(detection_result)

            self.logger.debug(f"Обнаружено геометрических паттернов: {len(results)}")
            return results

        except Exception as e:
            self.logger.error(f"Ошибка детекции геометрических паттернов: {e}")
            return []

    def _detect_harmonic_patterns(self, ohlc_data: Dict[str, np.ndarray]) -> List[DetectionResult]:
        """Детекция гармонических паттернов"""
        try:
            patterns = self.harmonic_detector.detect_all(ohlc_data)
            results = []

            for pattern in patterns:
                detection_result = DetectionResult(
                    pattern_type=PatternType.HARMONIC,
                    pattern_name=pattern.get('name', 'unknown'),
                    pattern_data=pattern,
                    confidence=pattern.get('confidence', 0.5),
                    position=(pattern.get('start_index', 0), pattern.get('end_index', 0)),
                    metadata={
                        'symbol': pattern.get('symbol', 'UNKNOWN'),
                        'timeframe': pattern.get('timeframe', 'UNKNOWN'),
                        'timestamp': datetime.now().isoformat()
                    }
                )
                results.append(detection_result)

            self.logger.debug(f"Обнаружено гармонических паттернов: {len(results)}")
            return results

        except Exception as e:
            self.logger.error(f"Ошибка детекции гармонических паттернов: {e}")
            return []

    def _detect_technical_patterns(self,
                                  ohlc_data: Dict[str, np.ndarray],
                                  indicators: Dict[str, np.ndarray]) -> List[DetectionResult]:
        """Детекция технических паттернов (дивергенции, конвергенции)"""
        try:
            results = []

            # Детекция дивергенций RSI
            if 'rsi' in indicators:
                rsi_divergences = self._detect_rsi_divergence(ohlc_data, indicators['rsi'])
                results.extend(rsi_divergences)

            # Детекция дивергенций MACD
            if 'macd' in indicators and 'macd_signal' in indicators:
                macd_divergences = self._detect_macd_divergence(ohlc_data, indicators['macd'])
                results.extend(macd_divergences)

            # Детекция паттернов объема
            if 'volume' in ohlc_data:
                volume_patterns = self._detect_volume_patterns(ohlc_data)
                results.extend(volume_patterns)

            self.logger.debug(f"Обнаружено технических паттернов: {len(results)}")
            return results

        except Exception as e:
            self.logger.error(f"Ошибка детекции технических паттернов: {e}")
            return []

    def _detect_rsi_divergence(self,
                              ohlc_data: Dict[str, np.ndarray],
                              rsi: np.ndarray) -> List[DetectionResult]:
        """Детекция дивергенции RSI"""
        results = []
        closes = ohlc_data.get('close', np.array([]))
        highs = ohlc_data.get('high', np.array([]))
        lows = ohlc_data.get('low', np.array([]))

        if len(closes) < 20 or len(rsi) < 20:
            return results

        window = 10
        for i in range(window, len(closes) - window):
            # Поиск локальных максимумов/минимумов цены
            local_price_max = np.max(highs[i-window:i+window+1])
            local_price_min = np.min(lows[i-window:i+window+1])

            local_rsi_max = np.max(rsi[i-window:i+window+1])
            local_rsi_min = np.min(rsi[i-window:i+window+1])

            # Проверка на дивергенцию
            price_idx_max = np.argmax(highs[i-window:i+window+1]) + i - window
            price_idx_min = np.argmin(lows[i-window:i+window+1]) + i - window

            rsi_idx_max = np.argmax(rsi[i-window:i+window+1]) + i - window
            rsi_idx_min = np.argmin(rsi[i-window:i+window+1]) + i - window

            # Медвежья дивергенция (цена делает новый максимум, RSI - нет)
            if (abs(price_idx_max - i) < 5 and abs(rsi_idx_max - i) < 5 and
                highs[i] == local_price_max and rsi[i] < local_rsi_max * 0.95):

                pattern_data = {
                    'name': 'RSI_Bearish_Divergence',
                    'type': 'divergence',
                    'direction': 'bearish',
                    'price_high': highs[i],
                    'rsi_high': rsi[i],
                    'index': i,
                    'confidence': 0.7,
                    'start_index': max(0, i - window),
                    'end_index': min(len(closes) - 1, i + window)
                }

                result = DetectionResult(
                    pattern_type=PatternType.TECHNICAL,
                    pattern_name='RSI_Bearish_Divergence',
                    pattern_data=pattern_data,
                    confidence=0.7,
                    position=(max(0, i - window), min(len(closes) - 1, i + window)),
                    metadata={'divergence_type': 'bearish', 'indicator': 'RSI'}
                )
                results.append(result)

            # Бычья дивергенция (цена делает новый минимум, RSI - нет)
            elif (abs(price_idx_min - i) < 5 and abs(rsi_idx_min - i) < 5 and
                  lows[i] == local_price_min and rsi[i] > local_rsi_min * 1.05):

                pattern_data = {
                    'name': 'RSI_Bullish_Divergence',
                    'type': 'divergence',
                    'direction': 'bullish',
                    'price_low': lows[i],
                    'rsi_low': rsi[i],
                    'index': i,
                    'confidence': 0.7,
                    'start_index': max(0, i - window),
                    'end_index': min(len(closes) - 1, i + window)
                }

                result = DetectionResult(
                    pattern_type=PatternType.TECHNICAL,
                    pattern_name='RSI_Bullish_Divergence',
                    pattern_data=pattern_data,
                    confidence=0.7,
                    position=(max(0, i - window), min(len(closes) - 1, i + window)),
                    metadata={'divergence_type': 'bullish', 'indicator': 'RSI'}
                )
                results.append(result)

        return results

    def _detect_macd_divergence(self,
                               ohlc_data: Dict[str, np.ndarray],
                               macd: np.ndarray) -> List[DetectionResult]:
        """Детекция дивергенции MACD"""
        results = []
        closes = ohlc_data.get('close', np.array([]))
        highs = ohlc_data.get('high', np.array([]))
        lows = ohlc_data.get('low', np.array([]))

        if len(closes) < 30 or len(macd) < 30:
            return results

        window = 15
        for i in range(window, len(closes) - window):
            local_price_max = np.max(highs[i-window:i+window+1])
            local_price_min = np.min(lows[i-window:i+window+1])

            local_macd_max = np.max(macd[i-window:i+window+1])
            local_macd_min = np.min(macd[i-window:i+window+1])

            price_idx_max = np.argmax(highs[i-window:i+window+1]) + i - window
            price_idx_min = np.argmin(lows[i-window:i+window+1]) + i - window

            macd_idx_max = np.argmax(macd[i-window:i+window+1]) + i - window
            macd_idx_min = np.argmin(macd[i-window:i+window+1]) + i - window

            # Медвежья дивергенция MACD
            if (abs(price_idx_max - i) < 5 and abs(macd_idx_max - i) < 5 and
                highs[i] == local_price_max and macd[i] < local_macd_max * 0.8):

                pattern_data = {
                    'name': 'MACD_Bearish_Divergence',
                    'type': 'divergence',
                    'direction': 'bearish',
                    'price_high': highs[i],
                    'macd_value': macd[i],
                    'index': i,
                    'confidence': 0.65,
                    'start_index': max(0, i - window),
                    'end_index': min(len(closes) - 1, i + window)
                }

                result = DetectionResult(
                    pattern_type=PatternType.TECHNICAL,
                    pattern_name='MACD_Bearish_Divergence',
                    pattern_data=pattern_data,
                    confidence=0.65,
                    position=(max(0, i - window), min(len(closes) - 1, i + window)),
                    metadata={'divergence_type': 'bearish', 'indicator': 'MACD'}
                )
                results.append(result)

            # Бычья дивергенция MACD
            elif (abs(price_idx_min - i) < 5 and abs(macd_idx_min - i) < 5 and
                  lows[i] == local_price_min and macd[i] > local_macd_min * 1.2):

                pattern_data = {
                    'name': 'MACD_Bullish_Divergence',
                    'type': 'divergence',
                    'direction': 'bullish',
                    'price_low': lows[i],
                    'macd_value': macd[i],
                    'index': i,
                    'confidence': 0.65,
                    'start_index': max(0, i - window),
                    'end_index': min(len(closes) - 1, i + window)
                }

                result = DetectionResult(
                    pattern_type=PatternType.TECHNICAL,
                    pattern_name='MACD_Bullish_Divergence',
                    pattern_data=pattern_data,
                    confidence=0.65,
                    position=(max(0, i - window), min(len(closes) - 1, i + window)),
                    metadata={'divergence_type': 'bullish', 'indicator': 'MACD'}
                )
                results.append(result)

        return results

    def _detect_volume_patterns(self, ohlc_data: Dict[str, np.ndarray]) -> List[DetectionResult]:
        """Детекция паттернов объема"""
        results = []
        closes = ohlc_data.get('close', np.array([]))
        opens = ohlc_data.get('open', np.array([]))
        volumes = ohlc_data.get('volume', np.array([]))

        if len(closes) < 10 or len(volumes) < 10:
            return results

        # Поиск объемных кластеров
        volume_ma = pd.Series(volumes).rolling(window=20).mean()
        volume_std = pd.Series(volumes).rolling(window=20).std()

        for i in range(20, len(closes)):
            if volumes[i] > volume_ma[i] + 2 * volume_std[i]:
                # Высокий объем на закрытии
                if closes[i] > opens[i]:  # Бычья свеча
                    pattern_data = {
                        'name': 'Volume_Spike_Bullish',
                        'type': 'volume',
                        'direction': 'bullish',
                        'volume': volumes[i],
                        'volume_avg': volume_ma[i],
                        'index': i,
                        'confidence': 0.6,
                        'start_index': max(0, i - 5),
                        'end_index': min(len(closes) - 1, i + 5)
                    }

                    result = DetectionResult(
                        pattern_type=PatternType.TECHNICAL,
                        pattern_name='Volume_Spike_Bullish',
                        pattern_data=pattern_data,
                        confidence=0.6,
                        position=(max(0, i - 5), min(len(closes) - 1, i + 5)),
                        metadata={'pattern_type': 'volume_spike', 'direction': 'bullish'}
                    )
                    results.append(result)

                elif closes[i] < opens[i]:  # Медвежья свеча
                    pattern_data = {
                        'name': 'Volume_Spike_Bearish',
                        'type': 'volume',
                        'direction': 'bearish',
                        'volume': volumes[i],
                        'volume_avg': volume_ma[i],
                        'index': i,
                        'confidence': 0.6,
                        'start_index': max(0, i - 5),
                        'end_index': min(len(closes) - 1, i + 5)
                    }

                    result = DetectionResult(
                        pattern_type=PatternType.TECHNICAL,
                        pattern_name='Volume_Spike_Bearish',
                        pattern_data=pattern_data,
                        confidence=0.6,
                        position=(max(0, i - 5), min(len(closes) - 1, i + 5)),
                        metadata={'pattern_type': 'volume_spike', 'direction': 'bearish'}
                    )
                    results.append(result)

        return results

    def _filter_overlapping_patterns(self, patterns: List[DetectionResult]) -> List[DetectionResult]:
        """Фильтрация пересекающихся паттернов"""
        if not patterns:
            return []

        # Сортируем по конечному индексу
        patterns.sort(key=lambda x: x.position[1])

        filtered = []
        last_end = -1

        for pattern in patterns:
            start, end = pattern.position

            # Если паттерн не пересекается с предыдущим
            if start > last_end:
                filtered.append(pattern)
                last_end = end
            else:
                # Выбираем паттерн с более высокой уверенностью
                if filtered and filtered[-1].confidence < pattern.confidence:
                    filtered[-1] = pattern
                    last_end = max(last_end, end)

        return filtered

    def get_pattern_by_type(self, pattern_type: PatternType) -> List[DetectionResult]:
        """
        Получение паттернов по типу

        Args:
            pattern_type: Тип паттерна

        Returns:
            Список паттернов заданного типа
        """
        return [p for p in self.detected_patterns if p.pattern_type == pattern_type]

    def get_pattern_by_name(self, pattern_name: str) -> List[DetectionResult]:
        """
        Получение паттернов по имени

        Args:
            pattern_name: Имя паттерна

        Returns:
            Список паттернов с заданным именем
        """
        return [p for p in self.detected_patterns if p.pattern_name == pattern_name]

    def clear_cache(self):
        """Очистка кэша паттернов"""
        self.detected_patterns.clear()
        self.pattern_cache.clear()
        self.logger.info("Кэш паттернов очищен")

    def save_detection_results(self, filepath: str) -> bool:
        """
        Сохранение результатов детекции

        Args:
            filepath: Путь для сохранения

        Returns:
            Успешность сохранения
        """
        try:
            import json

            results_data = []
            for result in self.detected_patterns:
                result_dict = {
                    'pattern_type': result.pattern_type.value,
                    'pattern_name': result.pattern_name,
                    'pattern_data': result.pattern_data,
                    'confidence': result.confidence,
                    'position': result.position,
                    'metadata': result.metadata
                }
                results_data.append(result_dict)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, default=str)

            self.logger.info(f"Результаты детекции сохранены: {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка сохранения результатов: {e}")
            return False

    def load_detection_results(self, filepath: str) -> bool:
        """
        Загрузка результатов детекции

        Args:
            filepath: Путь к файлу

        Returns:
            Успешность загрузки
        """
        try:
            import json

            with open(filepath, 'r', encoding='utf-8') as f:
                results_data = json.load(f)

            self.detected_patterns.clear()
            for result_dict in results_data:
                result = DetectionResult(
                    pattern_type=PatternType(result_dict['pattern_type']),
                    pattern_name=result_dict['pattern_name'],
                    pattern_data=result_dict['pattern_data'],
                    confidence=result_dict['confidence'],
                    position=tuple(result_dict['position']),
                    metadata=result_dict['metadata']
                )
                self.detected_patterns.append(result)

            self.logger.info(f"Результаты детекции загружены: {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка загрузки результатов: {e}")
            return False

