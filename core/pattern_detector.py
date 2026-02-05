"""
Модуль детектирования паттернов
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from datetime import datetime

# Исправляем импорт для обратной совместимости
try:
    from config import config, DETECTION_CONFIG, DETECTION
except ImportError:
    try:
        # Пробуем разные варианты
        from config import DETECTION_CONFIG as DETECTION
    except ImportError:
        # Создаем fallback конфиг
        DETECTION = type('DetectionConfig', (), {
            'MIN_CANDLES_FOR_PATTERN': 5,
            'MAX_CANDLES_FOR_PATTERN': 100,
            'MAX_PATTERNS_PER_SYMBOL': 50,
            'MIN_PATTERN_QUALITY': 0.6,
            'ENABLE_CANDLESTICK': True,
            'ENABLE_GEOMETRIC': True,
            'ENABLE_HARMONIC': True
        })()


class PatternDetector:
    """Класс для обнаружения торговых паттернов"""

    def __init__(self, custom_config: Dict = None):
        self.config = custom_config or {}
        self.detected_patterns = []

        # Используем настройки из конфига
        self.detection_config = config.DETECTION

    def detect(self, data: pd.DataFrame, symbol: str = None, timeframe: str = None) -> List[Dict]:
        """
        Обнаружение паттернов в данных

        Args:
            data: DataFrame с данными OHLC
            symbol: Название символа (опционально)
            timeframe: Таймфрейм (опционально)

        Returns:
            Список обнаруженных паттернов
        """
        if len(data) < self.detection_config.MIN_CANDLES_FOR_PATTERN:
            print(f"Недостаточно данных для анализа: {len(data)} баров")
            return []

        print(f"Запуск детектирования паттернов на {len(data)} барах")

        patterns = []

        # Детекция свечных паттернов
        if self.detection_config.ENABLE_CANDLESTICK:
            candlestick_patterns = self._detect_candlestick_patterns(data)
            patterns.extend(candlestick_patterns)

        # Детекция геометрических паттернов
        if self.detection_config.ENABLE_GEOMETRIC:
            geometric_patterns = self._detect_geometric_patterns(data)
            patterns.extend(geometric_patterns)

        # Детекция гармонических паттернов
        if self.detection_config.ENABLE_HARMONIC:
            harmonic_patterns = self._detect_harmonic_patterns(data)
            patterns.extend(harmonic_patterns)

        # Фильтрация по качеству
        filtered_patterns = [
            p for p in patterns
            if p.get('quality', 0) >= self.detection_config.MIN_PATTERN_QUALITY
        ]

        # Сортировка по качеству
        filtered_patterns.sort(key=lambda x: x.get('quality', 0), reverse=True)

        # Ограничение количества
        if len(filtered_patterns) > self.detection_config.MAX_PATTERNS_PER_SYMBOL:
            filtered_patterns = filtered_patterns[:self.detection_config.MAX_PATTERNS_PER_SYMBOL]

        # Добавляем метаданные
        for pattern in filtered_patterns:
            pattern['symbol'] = symbol
            pattern['timeframe'] = timeframe
            pattern['detection_time'] = datetime.now().isoformat()

        self.detected_patterns = filtered_patterns
        return filtered_patterns

    def _detect_candlestick_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """Детекция свечных паттернов"""
        patterns = []

        # Простая реализация для демонстрации
        # В реальном проекте здесь должна быть сложная логика

        # Ищем дожи
        for i in range(1, len(data) - 1):
            row = data.iloc[i]
            prev_row = data.iloc[i-1]

            # Доджи (очень маленькое тело)
            body_size = abs(row['close'] - row['open'])
            range_size = row['high'] - row['low']

            if range_size > 0 and body_size / range_size < 0.1:
                patterns.append({
                    'name': 'Doji',
                    'type': 'candlestick',
                    'timestamp': data.index[i],
                    'price': row['close'],
                    'quality': 0.7,
                    'direction': 'neutral'
                })

            # Молот (Hammer)
            lower_shadow = row['close'] - row['low'] if row['close'] > row['open'] else row['open'] - row['low']
            upper_shadow = row['high'] - row['close'] if row['close'] > row['open'] else row['high'] - row['open']
            body = abs(row['close'] - row['open'])

            if lower_shadow > 2 * body and upper_shadow < body * 0.1:
                direction = 'bullish' if row['close'] > row['open'] else 'bearish'
                patterns.append({
                    'name': 'Hammer' if direction == 'bullish' else 'Hanging Man',
                    'type': 'candlestick',
                    'timestamp': data.index[i],
                    'price': row['close'],
                    'quality': 0.6,
                    'direction': direction
                })

        return patterns

    def _detect_geometric_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """Детекция геометрических паттернов"""
        patterns = []

        # Простая реализация для демонстрации
        # Ищем поддержку и сопротивление

        # Вычисляем скользящие средние
        data['SMA20'] = data['close'].rolling(window=20).mean()
        data['SMA50'] = data['close'].rolling(window=50).mean()

        # Ищем пересечения SMA
        for i in range(1, len(data)):
            if pd.isna(data['SMA20'].iloc[i]) or pd.isna(data['SMA50'].iloc[i]):
                continue

            # Золотое пересечение
            if (data['SMA20'].iloc[i-1] <= data['SMA50'].iloc[i-1] and
                data['SMA20'].iloc[i] > data['SMA50'].iloc[i]):
                patterns.append({
                    'name': 'Golden Cross',
                    'type': 'geometric',
                    'timestamp': data.index[i],
                    'price': data['close'].iloc[i],
                    'quality': 0.65,
                    'direction': 'bullish'
                })

            # Мертвое пересечение
            if (data['SMA20'].iloc[i-1] >= data['SMA50'].iloc[i-1] and
                data['SMA20'].iloc[i] < data['SMA50'].iloc[i]):
                patterns.append({
                    'name': 'Death Cross',
                    'type': 'geometric',
                    'timestamp': data.index[i],
                    'price': data['close'].iloc[i],
                    'quality': 0.65,
                    'direction': 'bearish'
                })

        return patterns

    def _detect_harmonic_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """Детекция гармонических паттернов"""
        patterns = []

        # Простая реализация для демонспекции
        # В реальном проекте здесь должна быть сложная логика фибоначчи

        # Ищем экстремумы
        data['high_extreme'] = data['high'].rolling(window=5, center=True).max() == data['high']
        data['low_extreme'] = data['low'].rolling(window=5, center=True).min() == data['low']

        high_points = data[data['high_extreme']].index.tolist()
        low_points = data[data['low_extreme']].index.tolist()

        # Простая проверка на паттерн "Голова и плечи"
        if len(high_points) >= 3:
            # Берем последние 3 максимума
            recent_highs = high_points[-3:]
            high_values = [data.loc[idx, 'high'] for idx in recent_highs]

            # Проверяем паттерн "Голова и плечи"
            if (high_values[0] < high_values[1] and
                high_values[2] < high_values[1] and
                abs(high_values[0] - high_values[2]) / high_values[1] < self.detection_config.FIBONACCI_TOLERANCE):

                patterns.append({
                    'name': 'Head and Shoulders',
                    'type': 'harmonic',
                    'timestamp': recent_highs[2],
                    'price': data.loc[recent_highs[2], 'high'],
                    'quality': 0.7,
                    'direction': 'bearish'
                })

        return patterns

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики по обнаруженным паттернам"""
        if not self.detected_patterns:
            return {}

        stats = {
            'total_patterns': len(self.detected_patterns),
            'pattern_types': {},
            'avg_quality': 0,
            'symbols': {},
            'timeframes': {}
        }

        # Считаем статистику по типам
        for pattern in self.detected_patterns:
            pattern_type = pattern.get('type', 'unknown')
            stats['pattern_types'][pattern_type] = stats['pattern_types'].get(pattern_type, 0) + 1

            stats['avg_quality'] += pattern.get('quality', 0)

            symbol = pattern.get('symbol', 'unknown')
            stats['symbols'][symbol] = stats['symbols'].get(symbol, 0) + 1

            timeframe = pattern.get('timeframe', 'unknown')
            stats['timeframes'][timeframe] = stats['timeframes'].get(timeframe, 0) + 1

        if self.detected_patterns:
            stats['avg_quality'] /= len(self.detected_patterns)

        return stats


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
import json

# Настройка логирования
logger = logging.getLogger(__name__)

# Импортируем модули паттернов
try:
    from patterns.candlestick_patterns import DojiPattern, HammerPattern, EngulfingPattern
    from patterns.geometric_patterns import TrianglePattern, ChannelPattern
    from patterns.harmonic_patterns import HarmonicPattern
    from patterns.base_pattern import BasePattern
except ImportError as e:
    logger.warning(f"Не удалось импортировать модули паттернов: {e}")


    # Создаем заглушки для тестирования
    class BasePattern:
        def __init__(self, data):
            self.data = data
            self.patterns = []


    class DojiPattern(BasePattern):
        def detect(self, **kwargs):
            return []


    class HammerPattern(BasePattern):
        def detect(self, **kwargs):
            return []


    class EngulfingPattern(BasePattern):
        def detect(self, **kwargs):
            return []


    class TrianglePattern(BasePattern):
        def detect(self, **kwargs):
            return []


    class ChannelPattern(BasePattern):
        def detect(self, **kwargs):
            return []


    class HarmonicPattern(BasePattern):
        def detect(self, **kwargs):
            return []


class PatternDetector:
    """
    Класс для обнаружения паттернов на финансовых данных
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация детектора паттернов

        Args:
            config (dict): Конфигурация детектора
        """
        self.config = config or {}
        self.detected_patterns = []
        self.pattern_stats = {}
        self._initialize_detectors()

    def _initialize_detectors(self):
        """Инициализация детекторов паттернов"""
        self.detectors = {
            'candlestick': {
                'doji': DojiPattern,
                'hammer': HammerPattern,
                'engulfing': EngulfingPattern
            },
            'geometric': {
                'triangle': TrianglePattern,
                'channel': ChannelPattern
            },
            'harmonic': {
                'harmonic': HarmonicPattern
            }
        }

    def detect_all_patterns(self, data: pd.DataFrame,
                            pattern_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Обнаружение всех типов паттернов

        Args:
            data (pd.DataFrame): Финансовые данные
            pattern_types (list): Список типов паттернов для обнаружения

        Returns:
            list: Список обнаруженных паттернов
        """
        if data.empty:
            logger.warning("Получены пустые данные для обнаружения паттернов")
            return []

        if pattern_types is None:
            pattern_types = ['candlestick', 'geometric', 'harmonic']

        self.detected_patterns = []
        total_patterns_found = 0

        logger.info(f"Начало обнаружения паттернов. Типы: {pattern_types}")

        # Обнаружение свечных паттернов
        if 'candlestick' in pattern_types:
            candle_patterns = self.detect_candlestick_patterns(data)
            self.detected_patterns.extend(candle_patterns)
            total_patterns_found += len(candle_patterns)
            logger.info(f"Обнаружено {len(candle_patterns)} свечных паттернов")

        # Обнаружение геометрических паттернов
        if 'geometric' in pattern_types:
            geo_patterns = self.detect_geometric_patterns(data)
            self.detected_patterns.extend(geo_patterns)
            total_patterns_found += len(geo_patterns)
            logger.info(f"Обнаружено {len(geo_patterns)} геометрических паттернов")

        # Обнаружение гармонических паттернов
        if 'harmonic' in pattern_types:
            harmonic_patterns = self.detect_harmonic_patterns(data)
            self.detected_patterns.extend(harmonic_patterns)
            total_patterns_found += len(harmonic_patterns)
            logger.info(f"Обнаружено {len(harmonic_patterns)} гармонических паттернов")

        # Добавляем метаданные
        self._add_pattern_metadata(data)

        # Рассчитываем статистику
        self._calculate_pattern_statistics()

        logger.info(f"Всего обнаружено {total_patterns_found} паттернов")
        return self.detected_patterns

    def detect_candlestick_patterns(self, data: pd.DataFrame,
                                    lookback_period: int = 50) -> List[Dict[str, Any]]:
        """
        Обнаружение свечных паттернов

        Args:
            data (pd.DataFrame): Финансовые данные
            lookback_period (int): Период для анализа

        Returns:
            list: Список свечных паттернов
        """
        patterns = []

        if len(data) < lookback_period:
            logger.warning(f"Недостаточно данных для анализа свечных паттернов: {len(data)} < {lookback_period}")
            return patterns

        try:
            # Doji Pattern
            doji_detector = self.detectors['candlestick']['doji'](data)
            doji_patterns = doji_detector.detect(threshold=0.1, min_wick_ratio=2.0)

            for pattern in doji_patterns:
                pattern['pattern_family'] = 'candlestick'
                pattern['pattern_type'] = 'doji'
                pattern['detection_time'] = datetime.now()
                patterns.append(pattern)

            # Hammer Pattern
            hammer_detector = self.detectors['candlestick']['hammer'](data)
            hammer_patterns = hammer_detector.detect(min_body_ratio=0.3, max_upper_wick=0.1)

            for pattern in hammer_patterns:
                pattern['pattern_family'] = 'candlestick'
                pattern['pattern_type'] = 'hammer'
                pattern['detection_time'] = datetime.now()
                patterns.append(pattern)

            # Engulfing Pattern
            engulfing_detector = self.detectors['candlestick']['engulfing'](data)
            engulfing_patterns = engulfing_detector.detect(min_body_ratio=1.5)

            for pattern in engulfing_patterns:
                pattern['pattern_family'] = 'candlestick'
                pattern['pattern_type'] = 'engulfing'
                pattern['detection_time'] = datetime.now()
                patterns.append(pattern)

            logger.debug(f"Обнаружено {len(patterns)} свечных паттернов")

        except Exception as e:
            logger.error(f"Ошибка при обнаружении свечных паттернов: {e}")

        return patterns

    def detect_geometric_patterns(self, data: pd.DataFrame,
                                  min_points: int = 5,
                                  sensitivity: float = 0.8) -> List[Dict[str, Any]]:
        """
        Обнаружение геометрических паттернов

        Args:
            data (pd.DataFrame): Финансовые данные
            min_points (int): Минимальное количество точек для паттерна
            sensitivity (float): Чувствительность обнаружения

        Returns:
            list: Список геометрических паттернов
        """
        patterns = []

        if len(data) < min_points * 2:
            logger.warning(f"Недостаточно данных для геометрических паттернов: {len(data)} < {min_points * 2}")
            return patterns

        try:
            # Triangle Pattern
            triangle_detector = self.detectors['geometric']['triangle'](data)
            triangle_patterns = triangle_detector.detect(
                min_points=min_points,
                sensitivity=sensitivity
            )

            for pattern in triangle_patterns:
                pattern['pattern_family'] = 'geometric'
                pattern['pattern_type'] = 'triangle'
                pattern['detection_time'] = datetime.now()
                pattern['confidence'] = sensitivity
                patterns.append(pattern)

            # Channel Pattern
            channel_detector = self.detectors['geometric']['channel'](data)
            channel_patterns = channel_detector.detect(
                min_points=min_points,
                sensitivity=sensitivity
            )

            for pattern in channel_patterns:
                pattern['pattern_family'] = 'geometric'
                pattern['pattern_type'] = 'channel'
                pattern['detection_time'] = datetime.now()
                pattern['confidence'] = sensitivity
                patterns.append(pattern)

            logger.debug(f"Обнаружено {len(patterns)} геометрических паттернов")

        except Exception as e:
            logger.error(f"Ошибка при обнаружении геометрических паттернов: {e}")

        return patterns

    def detect_harmonic_patterns(self, data: pd.DataFrame,
                                 max_patterns: int = 10,
                                 tolerance: float = 0.05) -> List[Dict[str, Any]]:
        """
        Обнаружение гармонических паттернов

        Args:
            data (pd.DataFrame): Финансовые данные
            max_patterns (int): Максимальное количество паттернов для поиска
            tolerance (float): Допуск для соотношений Фибоначчи

        Returns:
            list: Список гармонических паттернов
        """
        patterns = []

        if len(data) < 100:
            logger.warning(f"Недостаточно данных для гармонических паттернов: {len(data)} < 100")
            return patterns

        try:
            harmonic_detector = self.detectors['harmonic']['harmonic'](data)
            harmonic_patterns = harmonic_detector.detect(
                max_patterns=max_patterns,
                fib_tolerance=tolerance
            )

            for pattern in harmonic_patterns:
                pattern['pattern_family'] = 'harmonic'
                pattern['detection_time'] = datetime.now()
                pattern['tolerance'] = tolerance

                # Определяем тип гармонического паттерна
                if 'pattern_type' not in pattern:
                    pattern['pattern_type'] = 'harmonic'

                patterns.append(pattern)

            logger.debug(f"Обнаружено {len(patterns)} гармонических паттернов")

        except Exception as e:
            logger.error(f"Ошибка при обнаружении гармонических паттернов: {e}")

        return patterns

    def _add_pattern_metadata(self, data: pd.DataFrame):
        """
        Добавление метаданных к обнаруженным паттернам

        Args:
            data (pd.DataFrame): Исходные данные
        """
        for pattern in self.detected_patterns:
            # Добавляем информацию о данных
            if 'index' in pattern and pattern['index'] < len(data):
                idx = pattern['index']
                pattern['price'] = data.iloc[idx]['Close'] if 'Close' in data.columns else 0
                pattern['volume'] = data.iloc[idx]['Volume'] if 'Volume' in data.columns else 0

            # Добавляем уникальный ID
            if 'id' not in pattern:
                pattern['id'] = f"{pattern.get('pattern_type', 'unknown')}_{pattern.get('index', 0)}"

            # Добавляем временную метку, если её нет
            if 'detection_time' not in pattern:
                pattern['detection_time'] = datetime.now()

    def _calculate_pattern_statistics(self):
        """Расчет статистики по обнаруженным паттернам"""
        if not self.detected_patterns:
            self.pattern_stats = {}
            return

        try:
            # Группировка по типам паттернов
            type_counts = {}
            family_counts = {}
            confidence_sum = {}
            confidence_count = {}

            for pattern in self.detected_patterns:
                pattern_type = pattern.get('pattern_type', 'unknown')
                pattern_family = pattern.get('pattern_family', 'unknown')
                confidence = pattern.get('confidence', 0)

                # Подсчет по типам
                type_counts[pattern_type] = type_counts.get(pattern_type, 0) + 1
                family_counts[pattern_family] = family_counts.get(pattern_family, 0) + 1

                # Статистика уверенности
                if pattern_type not in confidence_sum:
                    confidence_sum[pattern_type] = 0
                    confidence_count[pattern_type] = 0

                confidence_sum[pattern_type] += confidence
                confidence_count[pattern_type] += 1

            # Расчет средней уверенности
            avg_confidence = {}
            for pattern_type in confidence_sum:
                if confidence_count[pattern_type] > 0:
                    avg_confidence[pattern_type] = confidence_sum[pattern_type] / confidence_count[pattern_type]

            # Общая статистика
            self.pattern_stats = {
                'total_patterns': len(self.detected_patterns),
                'type_distribution': type_counts,
                'family_distribution': family_counts,
                'avg_confidence_by_type': avg_confidence,
                'detection_time': datetime.now(),
                'most_common_type': max(type_counts, key=type_counts.get) if type_counts else 'none',
                'most_common_family': max(family_counts, key=family_counts.get) if family_counts else 'none'
            }

            logger.info(f"Статистика паттернов: {json.dumps(self.pattern_stats, default=str, indent=2)}")

        except Exception as e:
            logger.error(f"Ошибка при расчете статистики паттернов: {e}")
            self.pattern_stats = {'error': str(e)}

    def get_pattern_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики по обнаруженным паттернам

        Returns:
            dict: Статистика паттернов
        """
        return self.pattern_stats.copy()

    def save_patterns_to_file(self, filename: str):
        """
        Сохранение паттернов в файл

        Args:
            filename (str): Имя файла для сохранения
        """
        try:
            with open(filename, 'w') as f:
                json.dump(self.detected_patterns, f, default=str, indent=2)
            logger.info(f"Паттерны сохранены в файл: {filename}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении паттернов в файл {filename}: {e}")

    def load_patterns_from_file(self, filename: str) -> List[Dict[str, Any]]:
        """
        Загрузка паттернов из файла

        Args:
            filename (str): Имя файла для загрузки

        Returns:
            list: Загруженные паттерны
        """
        try:
            with open(filename, 'r') as f:
                patterns = json.load(f)

            # Преобразуем строковые даты обратно в datetime
            for pattern in patterns:
                if 'detection_time' in pattern and isinstance(pattern['detection_time'], str):
                    try:
                        pattern['detection_time'] = datetime.fromisoformat(
                            pattern['detection_time'].replace('Z', '+00:00')
                        )
                    except:
                        pattern['detection_time'] = datetime.now()

            self.detected_patterns = patterns
            logger.info(f"Загружено {len(patterns)} паттернов из файла: {filename}")
            return patterns

        except Exception as e:
            logger.error(f"Ошибка при загрузке паттернов из файла {filename}: {e}")
            return []

    def filter_patterns_by_confidence(self, min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """
        Фильтрация паттернов по уверенности

        Args:
            min_confidence (float): Минимальная уверенность

        Returns:
            list: Отфильтрованные паттерны
        """
        filtered_patterns = [
            p for p in self.detected_patterns
            if p.get('confidence', 0) >= min_confidence
        ]

        logger.info(f"Отфильтровано {len(filtered_patterns)} паттернов с уверенностью >= {min_confidence}")
        return filtered_patterns

    def get_patterns_by_type(self, pattern_type: str) -> List[Dict[str, Any]]:
        """
        Получение паттернов определенного типа

        Args:
            pattern_type (str): Тип паттерна

        Returns:
            list: Паттерны указанного типа
        """
        patterns = [
            p for p in self.detected_patterns
            if p.get('pattern_type', '').lower() == pattern_type.lower()
        ]

        logger.info(f"Найдено {len(patterns)} паттернов типа '{pattern_type}'")
        return patterns

    def clear_patterns(self):
        """Очистка списка обнаруженных паттернов"""
        self.detected_patterns = []
        self.pattern_stats = {}
        logger.info("Список паттернов очищен")


# Создаем глобальный экземпляр для удобства
pattern_detector = PatternDetector()

