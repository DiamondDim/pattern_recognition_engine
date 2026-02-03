"""
Модуль детектирования паттернов
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from datetime import datetime

try:
    from config import config
except ImportError:
    # Для обратной совместимости
    from config import config


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

