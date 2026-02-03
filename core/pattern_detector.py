"""
Модуль детектирования паттернов
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from config import config
from utils.logger import logger
from patterns.candlestick_patterns import CandlestickPatterns
from patterns.geometric_patterns import GeometricPatterns
from patterns.harmonic_patterns import HarmonicPatterns


@dataclass
class DetectionResult:
    """Результат детектирования паттернов"""

    symbol: str
    timeframe: str
    timestamp: str
    patterns: List[Dict[str, Any]] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0

    @property
    def total_patterns(self) -> int:
        """Общее количество найденных паттернов"""
        return len(self.patterns)

    @property
    def pattern_types(self) -> Dict[str, int]:
        """Количество паттернов по типам"""
        types = {}
        for pattern in self.patterns:
            pattern_type = pattern.get('type', 'unknown')
            types[pattern_type] = types.get(pattern_type, 0) + 1
        return types

    @property
    def pattern_directions(self) -> Dict[str, int]:
        """Количество паттернов по направлениям"""
        directions = {}
        for pattern in self.patterns:
            direction = pattern.get('direction', 'neutral')
            directions[direction] = directions.get(direction, 0) + 1
        return directions


class PatternDetector:
    """Класс для детектирования паттернов"""

    def __init__(self):
        self.logger = logger.bind(module="pattern_detector")

        # Инициализация детекторов
        self.candlestick_detector = CandlestickPatterns()
        self.geometric_detector = GeometricPatterns()
        self.harmonic_detector = HarmonicPatterns()

        # Пул потоков для параллельной обработки
        self.thread_pool = ThreadPoolExecutor(
            max_workers=config.MAX_WORKERS
        )

        # Кэш для оптимизации
        self.detection_cache = {}
        self.cache_size = config.CACHE_SIZE if hasattr(config, 'CACHE_SIZE') else 1000

        # Статистика
        self.detection_stats = {
            'total_detections': 0,
            'total_patterns_found': 0,
            'avg_processing_time': 0.0,
            'pattern_types_found': {},
            'symbols_processed': set()
        }

    async def detect_all_patterns(self,
                                  symbol: str,
                                  timeframe: str,
                                  data: Dict[str, np.ndarray]) -> DetectionResult:
        """
        Детектирование всех типов паттернов

        Args:
            symbol: Торговый символ
            timeframe: Таймфрейм
            data: Входные данные OHLC

        Returns:
            Результат детектирования
        """
        import time
        start_time = time.time()

        self.logger.info(f"Детектирование паттернов для {symbol} {timeframe}")

        # Проверка входных данных
        if not self._validate_data(data):
            self.logger.error("Неверные входные данные")
            return DetectionResult(symbol=symbol, timeframe=timeframe, timestamp=self._get_timestamp())

        # Ключ кэша
        cache_key = self._generate_cache_key(symbol, timeframe, data)

        # Проверка кэша
        if config.USE_CACHE and cache_key in self.detection_cache:
            self.logger.debug(f"Результат загружен из кэша: {cache_key}")
            cached_result = self.detection_cache[cache_key]
            cached_result.timestamp = self._get_timestamp()
            return cached_result

        # Запуск параллельного детектирования
        tasks = []

        if config.DETECTION.ENABLE_CANDLESTICK:
            tasks.append(self._detect_candlestick_patterns(symbol, timeframe, data))

        if config.DETECTION.ENABLE_GEOMETRIC:
            tasks.append(self._detect_geometric_patterns(symbol, timeframe, data))

        if config.DETECTION.ENABLE_HARMONIC:
            tasks.append(self._detect_harmonic_patterns(symbol, timeframe, data))

        # Выполнение всех задач параллельно
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Сбор результатов
        all_patterns = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Ошибка детектирования: {result}")
                continue
            if result:
                all_patterns.extend(result)

        # Фильтрация по качеству
        filtered_patterns = self._filter_patterns_by_quality(all_patterns)

        # Ограничение количества
        final_patterns = filtered_patterns[:config.DETECTION.MAX_PATTERNS_PER_SYMBOL]

        # Расчет статистики
        processing_time = time.time() - start_time
        statistics = self._calculate_statistics(final_patterns, processing_time)

        # Создание результата
        result = DetectionResult(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=self._get_timestamp(),
            patterns=final_patterns,
            statistics=statistics,
            processing_time=processing_time
        )

        # Кэширование результата
        if config.USE_CACHE:
            self._add_to_cache(cache_key, result)

        # Обновление статистики
        self._update_detection_stats(symbol, final_patterns, processing_time)

        self.logger.info(f"Найдено паттернов: {len(final_patterns)} за {processing_time:.2f} сек")

        return result

    async def _detect_candlestick_patterns(self,
                                           symbol: str,
                                           timeframe: str,
                                           data: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Детектирование свечных паттернов"""
        try:
            # Используем ThreadPoolExecutor для блокирующих операций
            loop = asyncio.get_event_loop()
            patterns = await loop.run_in_executor(
                self.thread_pool,
                self.candlestick_detector.detect_all,
                data
            )

            # Форматирование результатов
            formatted_patterns = []
            for pattern in patterns:
                formatted = self._format_pattern(
                    pattern=pattern,
                    symbol=symbol,
                    timeframe=timeframe,
                    pattern_type='candlestick',
                    data=data
                )
                if formatted:
                    formatted_patterns.append(formatted)

            self.logger.debug(f"Свечных паттернов найдено: {len(formatted_patterns)}")
            return formatted_patterns

        except Exception as e:
            self.logger.error(f"Ошибка детектирования свечных паттернов: {e}")
            return []

    async def _detect_geometric_patterns(self,
                                         symbol: str,
                                         timeframe: str,
                                         data: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Детектирование геометрических паттернов"""
        try:
            loop = asyncio.get_event_loop()
            patterns = await loop.run_in_executor(
                self.thread_pool,
                self.geometric_detector.detect_all,
                data
            )

            formatted_patterns = []
            for pattern in patterns:
                formatted = self._format_pattern(
                    pattern=pattern,
                    symbol=symbol,
                    timeframe=timeframe,
                    pattern_type='geometric',
                    data=data
                )
                if formatted:
                    formatted_patterns.append(formatted)

            self.logger.debug(f"Геометрических паттернов найдено: {len(formatted_patterns)}")
            return formatted_patterns

        except Exception as e:
            self.logger.error(f"Ошибка детектирования геометрических паттернов: {e}")
            return []

    async def _detect_harmonic_patterns(self,
                                        symbol: str,
                                        timeframe: str,
                                        data: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Детектирование гармонических паттернов"""
        try:
            loop = asyncio.get_event_loop()
            patterns = await loop.run_in_executor(
                self.thread_pool,
                self.harmonic_detector.detect_all,
                data
            )

            formatted_patterns = []
            for pattern in patterns:
                formatted = self._format_pattern(
                    pattern=pattern,
                    symbol=symbol,
                    timeframe=timeframe,
                    pattern_type='harmonic',
                    data=data
                )
                if formatted:
                    formatted_patterns.append(formatted)

            self.logger.debug(f"Гармонических паттернов найдено: {len(formatted_patterns)}")
            return formatted_patterns

        except Exception as e:
            self.logger.error(f"Ошибка детектирования гармонических паттернов: {e}")
            return []

    def _format_pattern(self,
                        pattern: Dict[str, Any],
                        symbol: str,
                        timeframe: str,
                        pattern_type: str,
                        data: Dict[str, np.ndarray]) -> Optional[Dict[str, Any]]:
        """
        Форматирование паттерна в стандартный формат

        Args:
            pattern: Исходный паттерн
            symbol: Символ
            timeframe: Таймфрейм
            pattern_type: Тип паттерна
            data: Входные данные

        Returns:
            Отформатированный паттерн или None
        """
        try:
            # Базовые поля
            formatted = {
                'id': self._generate_pattern_id(pattern, symbol, timeframe),
                'name': pattern.get('name', 'unknown'),
                'type': pattern_type,
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': pattern.get('direction', 'neutral'),
                'detected_at': self._get_timestamp(),
                'quality_score': float(pattern.get('quality', 0.5)),
                'confidence_score': float(pattern.get('confidence', pattern.get('quality', 0.5))),
                'points': pattern.get('points', []),
                'targets': pattern.get('targets', {}),
                'metadata': {
                    'original_pattern': pattern,
                    'pattern_type': pattern_type,
                    'detector': self.__class__.__name__
                }
            }

            # Добавляем индексы и цены
            if 'points' in pattern:
                points = pattern['points']
                if points and isinstance(points, list):
                    # Находим последнюю точку паттерна
                    last_point = max(points, key=lambda x: x.get('index', 0))
                    formatted['last_point_index'] = last_point.get('index')
                    formatted['last_point_price'] = last_point.get('price')

                    # Расчет возраста паттерна
                    current_index = len(data.get('close', [])) - 1
                    if 'last_point_index' in formatted:
                        formatted['age'] = current_index - formatted['last_point_index']

            # Проверка минимального качества
            if formatted['quality_score'] < config.DETECTION.MIN_PATTERN_QUALITY:
                return None

            # Проверка минимальной уверенности
            if formatted['confidence_score'] < config.DETECTION.CONFIDENCE_THRESHOLD:
                return None

            return formatted

        except Exception as e:
            self.logger.error(f"Ошибка форматирования паттерна: {e}")
            return None

    def _filter_patterns_by_quality(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Фильтрация паттернов по качеству"""
        if not patterns:
            return []

        # Сортировка по качеству (по убыванию)
        sorted_patterns = sorted(
            patterns,
            key=lambda x: x.get('quality_score', 0) * x.get('confidence_score', 0),
            reverse=True
        )

        # Удаление дубликатов (похожие паттерны)
        unique_patterns = self._remove_duplicate_patterns(sorted_patterns)

        return unique_patterns

    def _remove_duplicate_patterns(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Удаление дубликатов и похожих паттернов"""
        if not patterns:
            return []

        unique_patterns = []
        seen_signatures = set()

        for pattern in patterns:
            # Создание сигнатуры паттерна
            signature = self._create_pattern_signature(pattern)

            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_patterns.append(pattern)
            else:
                self.logger.debug(f"Дубликат паттерна пропущен: {pattern.get('name')}")

        return unique_patterns

    def _create_pattern_signature(self, pattern: Dict[str, Any]) -> str:
        """Создание уникальной сигнатуры паттерна"""
        name = pattern.get('name', '')
        pattern_type = pattern.get('type', '')
        symbol = pattern.get('symbol', '')
        timeframe = pattern.get('timeframe', '')

        # Используем точки паттерна для создания сигнатуры
        points = pattern.get('points', [])
        point_signature = ''

        for point in points[:4]:  # Используем первые 4 точки
            if isinstance(point, dict):
                idx = point.get('index', 0)
                price = point.get('price', 0)
                point_signature += f"{idx}:{price:.5f},"

        return f"{symbol}_{timeframe}_{pattern_type}_{name}_{point_signature}"

    def _calculate_statistics(self,
                              patterns: List[Dict[str, Any]],
                              processing_time: float) -> Dict[str, Any]:
        """Расчет статистики детектирования"""
        if not patterns:
            return {
                'total_patterns': 0,
                'avg_quality': 0.0,
                'avg_confidence': 0.0,
                'pattern_distribution': {},
                'processing_time': processing_time
            }

        # Качество и уверенность
        quality_scores = [p.get('quality_score', 0) for p in patterns]
        confidence_scores = [p.get('confidence_score', 0) for p in patterns]

        # Распределение по типам
        type_distribution = {}
        direction_distribution = {}

        for pattern in patterns:
            pattern_type = pattern.get('type', 'unknown')
            direction = pattern.get('direction', 'neutral')

            type_distribution[pattern_type] = type_distribution.get(pattern_type, 0) + 1
            direction_distribution[direction] = direction_distribution.get(direction, 0) + 1

        return {
            'total_patterns': len(patterns),
            'avg_quality': float(np.mean(quality_scores)) if quality_scores else 0.0,
            'avg_confidence': float(np.mean(confidence_scores)) if confidence_scores else 0.0,
            'max_quality': float(np.max(quality_scores)) if quality_scores else 0.0,
            'min_quality': float(np.min(quality_scores)) if quality_scores else 0.0,
            'pattern_types': type_distribution,
            'pattern_directions': direction_distribution,
            'processing_time': processing_time,
            'timestamp': self._get_timestamp()
        }

    def _update_detection_stats(self,
                                symbol: str,
                                patterns: List[Dict[str, Any]],
                                processing_time: float):
        """Обновление статистики детектирования"""
        self.detection_stats['total_detections'] += 1
        self.detection_stats['total_patterns_found'] += len(patterns)
        self.detection_stats['symbols_processed'].add(symbol)

        # Обновление среднего времени обработки
        current_avg = self.detection_stats['avg_processing_time']
        total_detections = self.detection_stats['total_detections']

        new_avg = (current_avg * (total_detections - 1) + processing_time) / total_detections
        self.detection_stats['avg_processing_time'] = new_avg

        # Обновление типов паттернов
        for pattern in patterns:
            pattern_type = pattern.get('type', 'unknown')
            self.detection_stats['pattern_types_found'][pattern_type] = \
                self.detection_stats['pattern_types_found'].get(pattern_type, 0) + 1

    def get_detection_statistics(self) -> Dict[str, Any]:
        """Получение статистики детектирования"""
        return {
            **self.detection_stats,
            'total_symbols_processed': len(self.detection_stats['symbols_processed']),
            'cache_size': len(self.detection_cache),
            'cache_hits': getattr(self, 'cache_hits', 0),
            'cache_misses': getattr(self, 'cache_misses', 0)
        }

    def clear_cache(self):
        """Очистка кэша детектирования"""
        self.detection_cache.clear()
        self.logger.info("Кэш детектирования очищен")

    def _validate_data(self, data: Dict[str, np.ndarray]) -> bool:
        """Валидация входных данных"""
        required_fields = ['open', 'high', 'low', 'close']

        for field in required_fields:
            if field not in data:
                self.logger.error(f"Отсутствует обязательное поле: {field}")
                return False

            if not isinstance(data[field], np.ndarray):
                self.logger.error(f"Поле {field} должно быть numpy.ndarray")
                return False

            if len(data[field]) == 0:
                self.logger.error(f"Поле {field} пустое")
                return False

        # Проверка согласованности длин
        lengths = [len(data[field]) for field in required_fields]
        if len(set(lengths)) > 1:
            self.logger.error("Длины массивов данных не совпадают")
            return False

        # Проверка минимального количества свечей
        if lengths[0] < config.DETECTION.MIN_CANDLES_FOR_PATTERN:
            self.logger.error(f"Слишком мало данных: {lengths[0]} < {config.DETECTION.MIN_CANDLES_FOR_PATTERN}")
            return False

        return True

    def _generate_cache_key(self,
                            symbol: str,
                            timeframe: str,
                            data: Dict[str, np.ndarray]) -> str:
        """Генерация ключа кэша"""
        import hashlib

        # Используем последние N свечей для ключа
        n_samples = min(100, len(data['close']))

        key_data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'close_prices': data['close'][-n_samples:].tobytes(),
            'high_prices': data['high'][-n_samples:].tobytes(),
            'low_prices': data['low'][-n_samples:].tobytes()
        }

        # Создаем хэш
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _add_to_cache(self, key: str, result: DetectionResult):
        """Добавление результата в кэш"""
        if len(self.detection_cache) >= self.cache_size:
            # Удаляем самый старый элемент
            oldest_key = next(iter(self.detection_cache))
            del self.detection_cache[oldest_key]

        # Сохраняем в кэш
        self.detection_cache[key] = result

        # Счетчик попаданий в кэш
        if not hasattr(self, 'cache_hits'):
            self.cache_hits = 0
            self.cache_misses = 0

        self.cache_misses += 1

    def _get_timestamp(self) -> str:
        """Получение текущей временной метки"""
        from datetime import datetime
        return datetime.now().isoformat()

    def _generate_pattern_id(self,
                             pattern: Dict[str, Any],
                             symbol: str,
                             timeframe: str) -> str:
        """Генерация уникального ID паттерна"""
        import hashlib

        pattern_data = {
            'name': pattern.get('name', ''),
            'symbol': symbol,
            'timeframe': timeframe,
            'points': pattern.get('points', []),
            'timestamp': self._get_timestamp()
        }

        pattern_string = json.dumps(pattern_data, sort_keys=True, default=str)
        return hashlib.md5(pattern_string.encode()).hexdigest()[:12]

    async def shutdown(self):
        """Корректное завершение работы детектора"""
        self.thread_pool.shutdown(wait=True)
        self.clear_cache()
        self.logger.info("PatternDetector завершил работу")

