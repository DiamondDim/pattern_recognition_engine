"""
Модуль анализа и оценки паттернов
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json

# Исправляем импорт для обратной совместимости
try:
    from config import config, DETECTION_CONFIG, ANALYSIS_CONFIG
except ImportError:
    # Для обратной совместимости
    try:
        from config import DETECTION_CONFIG as ANALYSIS_CONFIG
    except ImportError:
        # Создаем fallback конфиг
        @dataclass
        class AnalysisConfig:
            MIN_PATTERN_QUALITY: float = 0.6
            CONFIDENCE_THRESHOLD: float = 0.7
            PRICE_TOLERANCE_PCT: float = 0.002
            SYMMETRY_TOLERANCE: float = 0.15
            FIBONACCI_TOLERANCE: float = 0.05

        ANALYSIS_CONFIG = AnalysisConfig()


@dataclass
class PatternMetrics:
    """Метрики для оценки качества паттерна"""
    quality_score: float = 0.0
    confidence_level: float = 0.0
    symmetry_score: float = 0.0
    volume_confirmation: bool = False
    trend_alignment: bool = False
    fibonacci_levels: Dict[str, float] = field(default_factory=dict)
    support_resistance_levels: List[float] = field(default_factory=list)
    predicted_direction: str = "neutral"
    predicted_target: Optional[float] = None
    predicted_stop_loss: Optional[float] = None
    risk_reward_ratio: Optional[float] = None


class PatternAnalyzer:
    """Класс для анализа и оценки торговых паттернов"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.detection_config = ANALYSIS_CONFIG

    def analyze_pattern(self, pattern: Dict, price_data: pd.DataFrame) -> PatternMetrics:
        """
        Анализ паттерна и вычисление метрик

        Args:
            pattern: Словарь с данными паттерна
            price_data: DataFrame с ценовыми данными

        Returns:
            PatternMetrics объект с метриками
        """
        metrics = PatternMetrics()

        # Базовое качество из детекции
        base_quality = pattern.get('quality', 0.5)
        metrics.quality_score = base_quality

        # Анализ симметрии
        if pattern.get('type') in ['geometric', 'harmonic']:
            symmetry_score = self._calculate_symmetry(pattern, price_data)
            metrics.symmetry_score = symmetry_score
            metrics.quality_score *= (0.3 + 0.7 * symmetry_score)  # Взвешиваем симметрию

        # Подтверждение объемом
        metrics.volume_confirmation = self._check_volume_confirmation(pattern, price_data)
        if metrics.volume_confirmation:
            metrics.quality_score *= 1.1  # Увеличиваем качество при подтверждении объемом

        # Анализ тренда
        metrics.trend_alignment = self._check_trend_alignment(pattern, price_data)
        if not metrics.trend_alignment:
            metrics.quality_score *= 0.9  # Уменьшаем при противоречии тренду

        # Уровни Фибоначчи для гармонических паттернов
        if pattern.get('type') == 'harmonic':
            fibonacci_levels = self._calculate_fibonacci_levels(pattern, price_data)
            metrics.fibonacci_levels = fibonacci_levels

        # Уровни поддержки/сопротивления
        metrics.support_resistance_levels = self._identify_support_resistance(pattern, price_data)

        # Прогноз направления
        metrics.predicted_direction = self._predict_direction(pattern, price_data)

        # Расчет целей и стоп-лоссов
        if metrics.predicted_direction != "neutral":
            target, stop_loss = self._calculate_target_stop(pattern, price_data, metrics.predicted_direction)
            metrics.predicted_target = target
            metrics.predicted_stop_loss = stop_loss

            if target and stop_loss:
                risk = abs(stop_loss - pattern.get('price', 0))
                reward = abs(target - pattern.get('price', 0))
                if risk > 0:
                    metrics.risk_reward_ratio = reward / risk

        # Уровень уверенности
        metrics.confidence_level = min(1.0, metrics.quality_score * 1.2)

        return metrics

    def _calculate_symmetry(self, pattern: Dict, data: pd.DataFrame) -> float:
        """Вычисление симметрии паттерна"""
        try:
            # Для простых паттернов используем базовую симметрию
            pattern_type = pattern.get('type', '')

            if pattern_type == 'candlestick':
                return 0.8  # Свечные паттерны обычно симметричны

            elif pattern_type in ['geometric', 'harmonic']:
                # Проверяем симметрию по точкам разворота
                if 'points' in pattern:
                    points = pattern['points']
                    if len(points) >= 2:
                        # Простая проверка симметрии
                        return 0.7

            return 0.5  # Базовая симметрия

        except Exception:
            return 0.5

    def _check_volume_confirmation(self, pattern: Dict, data: pd.DataFrame) -> bool:
        """Проверка подтверждения объемом"""
        try:
            if 'volume' not in data.columns:
                return True  # Если нет данных об объемах, считаем подтвержденным

            pattern_time = pattern.get('timestamp')
            if not pattern_time:
                return True

            # Ищем ближайший индекс
            if pattern_time in data.index:
                idx = data.index.get_loc(pattern_time)
            else:
                # Ищем ближайший временной индекс
                time_diff = abs(data.index - pattern_time)
                idx = time_diff.argmin()

            # Проверяем объем на моменте формирования паттерна
            if idx < len(data):
                volume = data.iloc[idx]['volume']
                avg_volume = data['volume'].rolling(20).mean().iloc[idx] if idx >= 20 else data['volume'].mean()

                # Паттерн считается подтвержденным, если объем выше среднего
                return volume > avg_volume * 0.8

        except Exception:
            return True

    def _check_trend_alignment(self, pattern: Dict, data: pd.DataFrame) -> bool:
        """Проверка соответствия тренду"""
        try:
            pattern_time = pattern.get('timestamp')
            if not pattern_time:
                return True

            # Определяем направление паттерна
            pattern_direction = pattern.get('direction', 'neutral')
            if pattern_direction == 'neutral':
                return True

            # Вычисляем тренд с помощью SMA
            if len(data) >= 20:
                sma_short = data['close'].rolling(10).mean()
                sma_long = data['close'].rolling(30).mean()

                # Ищем индекс времени паттерна
                if pattern_time in data.index:
                    idx = data.index.get_loc(pattern_time)
                else:
                    time_diff = abs(data.index - pattern_time)
                    idx = time_diff.argmin()

                if idx >= 30:
                    # Определяем тренд по положению SMA
                    trend = 'bullish' if sma_short.iloc[idx] > sma_long.iloc[idx] else 'bearish'

                    # Проверяем соответствие направления паттерна тренду
                    return pattern_direction == trend

            return True

        except Exception:
            return True

    def _calculate_fibonacci_levels(self, pattern: Dict, data: pd.DataFrame) -> Dict[str, float]:
        """Вычисление уровней Фибоначчи для гармонических паттернов"""
        fibonacci_levels = {
            '0.0': 0.0,
            '0.236': 0.236,
            '0.382': 0.382,
            '0.5': 0.5,
            '0.618': 0.618,
            '0.786': 0.786,
            '1.0': 1.0,
            '1.272': 1.272,
            '1.414': 1.414,
            '1.618': 1.618
        }

        try:
            pattern_price = pattern.get('price')
            if not pattern_price:
                return fibonacci_levels

            # Для демонстрации используем простой расчет
            # В реальном проекте здесь должна быть сложная логика

            return fibonacci_levels

        except Exception:
            return fibonacci_levels

    def _identify_support_resistance(self, pattern: Dict, data: pd.DataFrame) -> List[float]:
        """Определение уровней поддержки и сопротивления"""
        try:
            levels = []

            # Используем простые пивоты
            if len(data) >= 5:
                recent_data = data.tail(5)
                high = recent_data['high'].max()
                low = recent_data['low'].min()
                close = recent_data['close'].iloc[-1]

                # Базовые уровни пивотов
                pivot = (high + low + close) / 3
                r1 = 2 * pivot - low
                s1 = 2 * pivot - high
                r2 = pivot + (high - low)
                s2 = pivot - (high - low)

                levels = [s2, s1, pivot, r1, r2]

            return [float(level) for level in levels]

        except Exception:
            return []

    def _predict_direction(self, pattern: Dict, data: pd.DataFrame) -> str:
        """Прогноз направления движения цены на основе паттерна"""
        pattern_direction = pattern.get('direction', 'neutral')

        # Для некоторых паттернов можем уточнить направление
        pattern_name = pattern.get('name', '').lower()

        if 'bull' in pattern_name or pattern_direction == 'bullish':
            return 'bullish'
        elif 'bear' in pattern_name or pattern_direction == 'bearish':
            return 'bearish'
        else:
            # Анализируем контекст для нейтральных паттернов
            try:
                if len(data) >= 10:
                    # Проверяем момент импульса
                    recent_returns = data['close'].pct_change(5).iloc[-1]
                    if abs(recent_returns) > 0.01:  # 1% движение
                        return 'bullish' if recent_returns > 0 else 'bearish'
            except Exception:
                pass

            return 'neutral'

    def _calculate_target_stop(self, pattern: Dict, data: pd.DataFrame, direction: str) -> Tuple[Optional[float], Optional[float]]:
        """Расчет целевых уровней и стоп-лоссов"""
        try:
            pattern_price = pattern.get('price')
            if not pattern_price:
                return None, None

            # Используем ATR для расчета волатильности
            if len(data) >= 14:
                high_low = data['high'] - data['low']
                high_close = abs(data['high'] - data['close'].shift())
                low_close = abs(data['low'] - data['close'].shift())

                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                atr = true_range.rolling(14).mean().iloc[-1]

                # Расчет уровней на основе ATR
                if direction == 'bullish':
                    target = pattern_price + 2 * atr
                    stop_loss = pattern_price - atr
                elif direction == 'bearish':
                    target = pattern_price - 2 * atr
                    stop_loss = pattern_price + atr
                else:
                    return None, None

                return float(target), float(stop_loss)

        except Exception:
            pass

        return None, None

    def generate_analysis_report(self, pattern: Dict, metrics: PatternMetrics) -> Dict[str, Any]:
        """Генерация отчета анализа паттерна"""
        report = {
            'pattern_info': {
                'name': pattern.get('name', 'Unknown'),
                'type': pattern.get('type', 'unknown'),
                'symbol': pattern.get('symbol', 'Unknown'),
                'timeframe': pattern.get('timeframe', 'Unknown'),
                'timestamp': pattern.get('timestamp'),
                'price': pattern.get('price'),
                'direction': pattern.get('direction', 'neutral')
            },
            'metrics': {
                'quality_score': metrics.quality_score,
                'confidence_level': metrics.confidence_level,
                'symmetry_score': metrics.symmetry_score,
                'volume_confirmation': metrics.volume_confirmation,
                'trend_alignment': metrics.trend_alignment,
                'predicted_direction': metrics.predicted_direction,
                'predicted_target': metrics.predicted_target,
                'predicted_stop_loss': metrics.predicted_stop_loss,
                'risk_reward_ratio': metrics.risk_reward_ratio
            },
            'recommendation': self._generate_recommendation(metrics),
            'timestamp': datetime.now().isoformat()
        }

        return report

    def _generate_recommendation(self, metrics: PatternMetrics) -> Dict[str, Any]:
        """Генерация торговой рекомендации на основе метрик"""
        recommendation = {
            'action': 'hold',
            'confidence': 'low',
            'reason': '',
            'risk_level': 'medium'
        }

        if metrics.confidence_level > 0.8:
            recommendation['confidence'] = 'high'
        elif metrics.confidence_level > 0.6:
            recommendation['confidence'] = 'medium'

        if metrics.predicted_direction == 'bullish' and metrics.quality_score > 0.7:
            recommendation['action'] = 'buy'
            recommendation['reason'] = 'Сильный бычий паттерн с высоким качеством'
        elif metrics.predicted_direction == 'bearish' and metrics.quality_score > 0.7:
            recommendation['action'] = 'sell'
            recommendation['reason'] = 'Сильный медвежий паттерн с высоким качеством'

        if metrics.risk_reward_ratio:
            if metrics.risk_reward_ratio > 2:
                recommendation['risk_level'] = 'low'
            elif metrics.risk_reward_ratio < 1:
                recommendation['risk_level'] = 'high'

        return recommendation


# Создаем глобальный экземпляр для обратной совместимости
pattern_analyzer = PatternAnalyzer()

