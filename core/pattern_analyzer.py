# core/pattern_analyzer.py

"""
Модуль анализа паттернов
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings
from collections import defaultdict
from scipy import stats

from config import ANALYSIS_CONFIG
from utils.logger import logger
from utils.helpers import calculate_returns, calculate_volatility, detect_trend


class PatternAnalyzer:
    """Анализатор паттернов"""

    def __init__(self, config: ANALYSIS_CONFIG = None):
        self.config = config or ANALYSIS_CONFIG
        self.logger = logger.bind(module="PatternAnalyzer")
        self.pattern_history = defaultdict(list)
        self.statistics_cache = {}

    def analyze_pattern(self,
                       pattern: Dict[str, Any],
                       market_data: Dict[str, np.ndarray],
                       historical_patterns: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Полный анализ паттерна

        Args:
            pattern: Паттерн для анализа
            market_data: Рыночные данные
            historical_patterns: Исторические паттерны для сравнения

        Returns:
            Результаты анализа
        """
        try:
            analysis = {
                'metadata': pattern.get('metadata', {}),
                'basic_analysis': self._analyze_basic_properties(pattern),
                'market_context': self._analyze_market_context(pattern, market_data),
                'strength_analysis': self._analyze_pattern_strength(pattern),
                'risk_analysis': self._analyze_risk(pattern),
                'statistical_significance': self._analyze_statistical_significance(pattern, historical_patterns),
                'trading_signals': self._generate_trading_signals(pattern, market_data),
                'quality_score': 0.0,
                'confidence_level': 0.0
            }

            # Расчет общего скора качества
            quality_score = self._calculate_quality_score(analysis)
            analysis['quality_score'] = quality_score

            # Уровень уверенности
            confidence = self._calculate_confidence_level(analysis)
            analysis['confidence_level'] = confidence

            # Генерация рекомендаций
            analysis['recommendations'] = self._generate_recommendations(analysis)

            return analysis

        except Exception as e:
            self.logger.error(f"Ошибка анализа паттерна: {e}")
            return {}

    def _analyze_basic_properties(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ базовых свойств паттерна"""
        points = pattern.get('points', [])
        metadata = pattern.get('metadata', {})
        targets = pattern.get('targets', {})

        if len(points) < 2:
            return {}

        # Координаты точек
        point_prices = [p.get('price', 0) for p in points]
        point_indices = [p.get('index', 0) for p in points]

        # Базовые статистики
        price_range = max(point_prices) - min(point_prices) if point_prices else 0
        avg_price = np.mean(point_prices) if point_prices else 0
        price_std = np.std(point_prices) if len(point_prices) > 1 else 0

        # Временные характеристики
        time_range = max(point_indices) - min(point_indices) if len(point_indices) > 1 else 0
        timeframe = metadata.get('timeframe', 'UNKNOWN')

        # Соотношения
        if price_range > 0:
            volatility_ratio = price_std / price_range
        else:
            volatility_ratio = 0

        # Целевые уровни
        entry_price = targets.get('entry_price', 0)
        stop_loss = targets.get('stop_loss', 0)
        take_profit = targets.get('take_profit', 0)

        if entry_price > 0 and stop_loss > 0:
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price) if take_profit > 0 else 0
            risk_reward = reward / risk if risk > 0 else 0
            risk_percentage = risk / entry_price * 100 if entry_price > 0 else 0
        else:
            risk_reward = 0
            risk_percentage = 0

        return {
            'point_count': len(points),
            'price_range': price_range,
            'avg_price': avg_price,
            'price_std': price_std,
            'time_range': time_range,
            'timeframe': timeframe,
            'volatility_ratio': volatility_ratio,
            'risk_reward_ratio': risk_reward,
            'risk_percentage': risk_percentage,
            'pattern_type': pattern.get('name', 'Unknown'),
            'direction': pattern.get('direction', 'neutral')
        }

    def _analyze_market_context(self,
                               pattern: Dict[str, Any],
                               market_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Анализ рыночного контекста"""
        closes = market_data.get('close', np.array([]))
        volumes = market_data.get('volume', np.array([]))
        pattern_points = pattern.get('points', [])

        if len(closes) == 0 or len(pattern_points) == 0:
            return {}

        # Определение индекса паттерна
        pattern_indices = [p.get('index', 0) for p in pattern_points]
        pattern_end_idx = max(pattern_indices) if pattern_indices else 0

        # Анализ тренда
        trend_window = self.config.TREND_WINDOW
        lookback_start = max(0, pattern_end_idx - trend_window)

        if lookback_start < len(closes):
            lookback_prices = closes[lookback_start:pattern_end_idx]
            trend_strength, trend_direction = detect_trend(lookback_prices)
        else:
            trend_strength, trend_direction = 0.0, 'neutral'

        # Анализ волатильности
        if len(closes) > 20:
            volatility_window = min(20, len(closes))
            recent_prices = closes[max(0, pattern_end_idx - volatility_window):pattern_end_idx]
            volatility = np.std(recent_prices) / np.mean(recent_prices) * 100 if len(recent_prices) > 1 else 0

            # Нормализованная волатильность
            if len(closes) > 100:
                historical_prices = closes[max(0, len(closes) - 100):]
                hist_volatility = np.std(historical_prices) / np.mean(historical_prices) * 100
                volatility_ratio = volatility / hist_volatility if hist_volatility > 0 else 1.0
            else:
                volatility_ratio = 1.0
        else:
            volatility = 0
            volatility_ratio = 1.0

        # Анализ объема
        volume_analysis = {}
        if len(volumes) > 0 and pattern_end_idx < len(volumes):
            pattern_volumes = []
            for idx in pattern_indices:
                if idx < len(volumes):
                    pattern_volumes.append(volumes[idx])

            if pattern_volumes:
                avg_pattern_volume = np.mean(pattern_volumes)

                # Сравнение с средним объемом
                if len(volumes) > 20:
                    avg_recent_volume = np.mean(volumes[max(0, pattern_end_idx - 20):pattern_end_idx])
                    volume_ratio = avg_pattern_volume / avg_recent_volume if avg_recent_volume > 0 else 1.0
                else:
                    volume_ratio = 1.0

                volume_analysis = {
                    'avg_pattern_volume': avg_pattern_volume,
                    'volume_ratio': volume_ratio,
                    'volume_trend': 'above_average' if volume_ratio > 1.2 else
                                   'below_average' if volume_ratio < 0.8 else 'normal'
                }

        # Анализ поддержки/сопротивления
        support_resistance = self._analyze_support_resistance(closes, pattern_indices)

        return {
            'trend_strength': trend_strength,
            'trend_direction': trend_direction,
            'volatility': volatility,
            'volatility_ratio': volatility_ratio,
            'volume_analysis': volume_analysis,
            'support_resistance': support_resistance,
            'pattern_position': self._analyze_pattern_position(closes, pattern_indices)
        }

    def _analyze_pattern_position(self,
                                 closes: np.ndarray,
                                 pattern_indices: List[int]) -> Dict[str, Any]:
        """Анализ позиции паттерна относительно ценового диапазона"""
        if len(closes) == 0 or len(pattern_indices) == 0:
            return {}

        pattern_end_idx = max(pattern_indices)
        lookback_window = 100

        start_idx = max(0, pattern_end_idx - lookback_window)
        recent_prices = closes[start_idx:pattern_end_idx]

        if len(recent_prices) == 0:
            return {}

        # Ценовые уровни
        current_price = closes[pattern_end_idx] if pattern_end_idx < len(closes) else recent_prices[-1]
        recent_high = np.max(recent_prices)
        recent_low = np.min(recent_prices)
        price_range = recent_high - recent_low

        if price_range > 0:
            position_ratio = (current_price - recent_low) / price_range
        else:
            position_ratio = 0.5

        # Определение позиции
        if position_ratio > 0.7:
            position = 'near_high'
        elif position_ratio < 0.3:
            position = 'near_low'
        else:
            position = 'middle'

        # Расстояние до экстремумов
        distance_to_high = abs(recent_high - current_price) / current_price * 100 if current_price > 0 else 0
        distance_to_low = abs(current_price - recent_low) / current_price * 100 if current_price > 0 else 0

        return {
            'current_price': current_price,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'position_ratio': position_ratio,
            'position': position,
            'distance_to_high_pct': distance_to_high,
            'distance_to_low_pct': distance_to_low,
            'price_range': price_range
        }

    def _analyze_support_resistance(self,
                                   closes: np.ndarray,
                                   pattern_indices: List[int]) -> Dict[str, Any]:
        """Анализ уровней поддержки и сопротивления"""
        if len(closes) == 0:
            return {}

        pattern_end_idx = max(pattern_indices) if pattern_indices else 0
        lookback_window = 50

        start_idx = max(0, pattern_end_idx - lookback_window)
        recent_prices = closes[start_idx:pattern_end_idx]

        if len(recent_prices) < 10:
            return {}

        # Поиск локальных экстремумов
        window = 5
        supports = []
        resistances = []

        for i in range(window, len(recent_prices) - window):
            local_min = np.min(recent_prices[i - window:i + window + 1])
            local_max = np.max(recent_prices[i - window:i + window + 1])

            if abs(recent_prices[i] - local_min) < 0.001 * local_min:
                supports.append(recent_prices[i])
            if abs(recent_prices[i] - local_max) < 0.001 * local_max:
                resistances.append(recent_prices[i])

        # Текущая цена
        current_price = recent_prices[-1]

        # Ближайшие уровни
        nearest_support = None
        nearest_resistance = None

        if supports:
            # Уровни ниже текущей цены
            below_supports = [s for s in supports if s < current_price]
            if below_supports:
                nearest_support = max(below_supports)

        if resistances:
            # Уровни выше текущей цены
            above_resistances = [r for r in resistances if r > current_price]
            if above_resistances:
                nearest_resistance = min(above_resistances)

        return {
            'supports': supports,
            'resistances': resistances,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'support_distance_pct': abs(current_price - nearest_support) / current_price * 100
            if nearest_support else None,
            'resistance_distance_pct': abs(nearest_resistance - current_price) / current_price * 100
            if nearest_resistance else None
        }

    def _analyze_pattern_strength(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ силы паттерна"""
        points = pattern.get('points', [])
        metadata = pattern.get('metadata', {})

        if len(points) < 2:
            return {'total_score': 0.0}

        scores = {}

        # 1. Качество формы (геометрическая точность)
        form_score = self._calculate_form_score(points)
        scores['form_score'] = form_score

        # 2. Четкость точек
        clarity_score = self._calculate_clarity_score(points)
        scores['clarity_score'] = clarity_score

        # 3. Соотношения Фибоначчи (для гармонических паттернов)
        fibonacci_score = self._calculate_fibonacci_score(points)
        scores['fibonacci_score'] = fibonacci_score

        # 4. Объемное подтверждение
        volume_score = metadata.get('volume_confirmation', 0.5)
        scores['volume_score'] = volume_score

        # 5. Временные соотношения
        time_score = self._calculate_time_score(points)
        scores['time_score'] = time_score

        # Общий счет
        weights = self.config.STRENGTH_WEIGHTS
        total_score = (
            form_score * weights['form'] +
            clarity_score * weights['clarity'] +
            fibonacci_score * weights['fibonacci'] +
            volume_score * weights['volume'] +
            time_score * weights['time']
        )

        scores['total_score'] = total_score
        scores['strength_level'] = self._get_strength_level(total_score)

        return scores

    def _calculate_form_score(self, points: List[Dict[str, Any]]) -> float:
        """Расчет скора качества формы"""
        if len(points) < 3:
            return 0.0

        # Извлекаем координаты
        indices = [p.get('index', 0) for p in points]
        prices = [p.get('price', 0) for p in points]

        # Нормализация
        norm_indices = [(i - min(indices)) / (max(indices) - min(indices)) if len(indices) > 1 else 0
                       for i in indices]
        norm_prices = [(p - min(prices)) / (max(prices) - min(prices)) if len(prices) > 1 else 0
                      for p in prices]

        # Расчет углов
        angle_scores = []
        for i in range(len(points) - 2):
            p1 = np.array([norm_indices[i], norm_prices[i]])
            p2 = np.array([norm_indices[i+1], norm_prices[i+1]])
            p3 = np.array([norm_indices[i+2], norm_prices[i+2]])

            v1 = p2 - p1
            v2 = p3 - p2

            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)

                # Идеальный угол для паттернов (например, 90-120 градусов)
                ideal_angle = np.pi / 2  # 90 градусов
                angle_diff = abs(angle - ideal_angle) / np.pi
                angle_score = max(0, 1 - angle_diff)
                angle_scores.append(angle_score)

        form_score = np.mean(angle_scores) if angle_scores else 0.5
        return float(form_score)

    def _calculate_clarity_score(self, points: List[Dict[str, Any]]) -> float:
        """Расчет скора четкости точек"""
        if len(points) < 2:
            return 0.0

        prices = [p.get('price', 0) for p in points]

        # Проверка на выбросы
        price_mean = np.mean(prices)
        price_std = np.std(prices)

        if price_std == 0:
            return 1.0

        # Z-score для каждой точки
        z_scores = [(p - price_mean) / price_std for p in prices]
        clarity_scores = [max(0, 1 - abs(z) / 3) for z in z_scores]  # Чем ближе к 0, тем лучше

        clarity_score = np.mean(clarity_scores)
        return float(clarity_score)

    def _calculate_fibonacci_score(self, points: List[Dict[str, Any]]) -> float:
        """Расчет скора соотношений Фибоначчи"""
        if len(points) < 4:
            return 0.5  # Нейтральный скор для негармонических паттернов

        # Извлекаем ключевые точки для гармонических паттернов
        key_prices = []
        for point in points:
            point_type = point.get('point_type', '')
            if any(keyword in point_type.lower() for keyword in ['x', 'a', 'b', 'c', 'd', '0', '1', '2', '3']):
                key_prices.append(point.get('price', 0))

        if len(key_prices) < 4:
            return 0.5

        # Расчет соотношений
        fibonacci_levels = [0.382, 0.5, 0.618, 0.786, 0.886, 1.0, 1.272, 1.618]

        # Пример для паттерна ABCD
        if len(key_prices) >= 4:
            XA = abs(key_prices[1] - key_prices[0])
            AB = abs(key_prices[2] - key_prices[1])
            BC = abs(key_prices[3] - key_prices[2])

            if XA > 0 and AB > 0 and BC > 0:
                ratios = [
                    AB / XA,  # AB/XA
                    BC / AB   # BC/AB
                ]

                # Проверка близости к уровням Фибоначчи
                scores = []
                for ratio in ratios:
                    closest_fibo = min(fibonacci_levels, key=lambda x: abs(x - ratio))
                    fibo_score = max(0, 1 - abs(ratio - closest_fibo) / 0.1)  # 10% допуск
                    scores.append(fibo_score)

                fibonacci_score = np.mean(scores)
                return float(fibonacci_score)

        return 0.5

    def _calculate_time_score(self, points: List[Dict[str, Any]]) -> float:
        """Расчет скора временных соотношений"""
        if len(points) < 3:
            return 0.5

        indices = [p.get('index', 0) for p in points]
        time_diffs = [indices[i+1] - indices[i] for i in range(len(indices)-1)]

        if len(time_diffs) < 2:
            return 0.5

        # Проверка на равномерность
        time_std = np.std(time_diffs)
        time_mean = np.mean(time_diffs)

        if time_mean == 0:
            return 0.5

        cv = time_std / time_mean  # Коэффициент вариации
        time_score = max(0, 1 - cv)  # Чем меньше вариация, тем лучше

        return float(time_score)

    def _get_strength_level(self, score: float) -> str:
        """Определение уровня силы"""
        if score >= 0.8:
            return 'strong'
        elif score >= 0.6:
            return 'moderate'
        elif score >= 0.4:
            return 'weak'
        else:
            return 'very_weak'

    def _analyze_risk(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ рисков"""
        targets = pattern.get('targets', {})
        metadata = pattern.get('metadata', {})

        entry_price = targets.get('entry_price', 0)
        stop_loss = targets.get('stop_loss', 0)
        take_profit = targets.get('take_profit', 0)

        if entry_price <= 0 or stop_loss <= 0:
            return {'risk_level': 'unknown'}

        # Расчет риска
        risk_amount = abs(entry_price - stop_loss)
        reward_amount = abs(take_profit - entry_price) if take_profit > 0 else 0

        risk_reward = reward_amount / risk_amount if risk_amount > 0 else 0
        risk_percentage = risk_amount / entry_price * 100

        # Уровень риска
        if risk_percentage > 5:
            risk_level = 'high'
        elif risk_percentage > 2:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        # Probability of Success (на основе исторической статистики)
        hist_success = metadata.get('historical_success_rate', 0.5)
        expected_value = hist_success * reward_amount - (1 - hist_success) * risk_amount

        return {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_amount': risk_amount,
            'reward_amount': reward_amount,
            'risk_reward_ratio': risk_reward,
            'risk_percentage': risk_percentage,
            'risk_level': risk_level,
            'historical_success_rate': hist_success,
            'expected_value': expected_value,
            'risk_adjusted_return': expected_value / risk_amount if risk_amount > 0 else 0
        }

    def _analyze_statistical_significance(self,
                                         pattern: Dict[str, Any],
                                         historical_patterns: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Анализ статистической значимости"""
        if not historical_patterns:
            return {'significance': 'unknown', 'confidence': 0.5}

        # Фильтрация аналогичных паттернов
        similar_patterns = self._find_similar_patterns(pattern, historical_patterns)

        if len(similar_patterns) < self.config.MIN_SAMPLE_SIZE:
            return {
                'significance': 'insufficient_data',
                'sample_size': len(similar_patterns),
                'confidence': 0.5
            }

        # Статистика успешности
        successful_patterns = [p for p in similar_patterns
                             if p.get('metadata', {}).get('outcome', '') == 'success']
        success_rate = len(successful_patterns) / len(similar_patterns)

        # Доверительный интервал
        if len(similar_patterns) > 0:
            # Приближенный доверительный интервал для пропорции
            z_score = 1.96  # 95% доверительный уровень
            margin_of_error = z_score * np.sqrt(success_rate * (1 - success_rate) / len(similar_patterns))
            confidence_interval = (max(0, success_rate - margin_of_error),
                                 min(1, success_rate + margin_of_error))
        else:
            confidence_interval = (0, 0)

        # Статистическая значимость
        if len(similar_patterns) >= 30 and success_rate > 0.5:
            # Z-тест для проверки гипотезы
            p_value = self._calculate_p_value(success_rate, len(similar_patterns))
            significant = p_value < 0.05
        else:
            p_value = None
            significant = False

        return {
            'sample_size': len(similar_patterns),
            'success_rate': success_rate,
            'confidence_interval': confidence_interval,
            'p_value': p_value,
            'significant': significant,
            'significance_level': 'high' if significant else 'low'
        }

    def _find_similar_patterns(self,
                              pattern: Dict[str, Any],
                              historical_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Поиск аналогичных паттернов"""
        similar_patterns = []

        pattern_type = pattern.get('name', '')
        pattern_direction = pattern.get('direction', '')

        for hist_pattern in historical_patterns:
            # Проверка типа и направления
            if (hist_pattern.get('name', '') == pattern_type and
                hist_pattern.get('direction', '') == pattern_direction):

                # Проверка схожести по размеру
                pattern_points = pattern.get('points', [])
                hist_points = hist_pattern.get('points', [])

                if len(pattern_points) > 0 and len(hist_points) > 0:
                    pattern_prices = [p.get('price', 0) for p in pattern_points]
                    hist_prices = [p.get('price', 0) for p in hist_points]

                    pattern_range = max(pattern_prices) - min(pattern_prices)
                    hist_range = max(hist_prices) - min(hist_prices)

                    if pattern_range > 0 and hist_range > 0:
                        size_ratio = min(pattern_range, hist_range) / max(pattern_range, hist_range)
                        if size_ratio > 0.7:  # 70% схожесть по размеру
                            similar_patterns.append(hist_pattern)

        return similar_patterns

    def _calculate_p_value(self, success_rate: float, sample_size: int) -> float:
        """Расчет p-value для проверки гипотезы"""
        # Нулевая гипотеза: success_rate = 0.5 (случайное угадывание)
        p0 = 0.5

        # Стандартная ошибка
        se = np.sqrt(p0 * (1 - p0) / sample_size)

        # Z-статистика
        z = (success_rate - p0) / se

        # p-value для двустороннего теста
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        return float(p_value)

    def _generate_trading_signals(self,
                                 pattern: Dict[str, Any],
                                 market_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Генерация торговых сигналов"""
        direction = pattern.get('direction', 'neutral')
        targets = pattern.get('targets', {})
        metadata = pattern.get('metadata', {})

        entry_price = targets.get('entry_price', 0)
        stop_loss = targets.get('stop_loss', 0)
        take_profit = targets.get('take_profit', 0)

        # Текущая цена
        closes = market_data.get('close', np.array([]))
        current_price = closes[-1] if len(closes) > 0 else 0

        # Сигнал входа
        entry_signal = None
        if entry_price > 0:
            if direction == 'bullish':
                if current_price <= entry_price * 1.01:  # 1% допуск
                    entry_signal = 'buy'
                elif current_price > entry_price * 1.01:
                    entry_signal = 'wait_for_pullback'
            elif direction == 'bearish':
                if current_price >= entry_price * 0.99:  # 1% допуск
                    entry_signal = 'sell'
                elif current_price < entry_price * 0.99:
                    entry_signal = 'wait_for_rally'

        # Сила сигнала
        quality_score = metadata.get('quality_score', 0)
        if quality_score > 0.7:
            signal_strength = 'strong'
        elif quality_score > 0.5:
            signal_strength = 'moderate'
        else:
            signal_strength = 'weak'

        # Временные рекомендации
        timeframe = metadata.get('timeframe', 'UNKNOWN')
        if 'M' in timeframe:
            time_horizon = 'intraday'
        elif 'H' in timeframe:
            time_horizon = 'short_term'
        elif 'D' in timeframe:
            time_horizon = 'medium_term'
        else:
            time_horizon = 'long_term'

        return {
            'direction': direction,
            'entry_signal': entry_signal,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'current_price': current_price,
            'distance_to_entry_pct': abs(current_price - entry_price) / entry_price * 100
            if entry_price > 0 else 0,
            'signal_strength': signal_strength,
            'time_horizon': time_horizon,
            'recommended_action': self._get_recommended_action(entry_signal, signal_strength)
        }

    def _get_recommended_action(self, entry_signal: Optional[str], signal_strength: str) -> str:
        """Получение рекомендуемого действия"""
        if entry_signal == 'buy':
            if signal_strength == 'strong':
                return 'enter_long_aggressive'
            elif signal_strength == 'moderate':
                return 'enter_long_cautious'
            else:
                return 'enter_long_small'
        elif entry_signal == 'sell':
            if signal_strength == 'strong':
                return 'enter_short_aggressive'
            elif signal_strength == 'moderate':
                return 'enter_short_cautious'
            else:
                return 'enter_short_small'
        elif entry_signal == 'wait_for_pullback':
            return 'wait_for_better_entry'
        elif entry_signal == 'wait_for_rally':
            return 'wait_for_better_entry'
        else:
            return 'no_action'

    def _calculate_quality_score(self, analysis: Dict[str, Any]) -> float:
        """Расчет общего скора качества"""
        weights = self.config.QUALITY_WEIGHTS

        # Базовые свойства
        basic = analysis.get('basic_analysis', {})
        basic_score = (
            (basic.get('volatility_ratio', 0) if basic.get('volatility_ratio', 0) < 0.5 else 0.5) +
            (min(basic.get('risk_reward_ratio', 0), 3) / 3)  # Нормализация до 0-1
        ) / 2 if basic else 0.5

        # Сила паттерна
        strength = analysis.get('strength_analysis', {})
        strength_score = strength.get('total_score', 0.5)

        # Рыночный контекст
        market = analysis.get('market_context', {})
        trend_score = market.get('trend_strength', 0.5)

        # Проверка соответствия направления паттерна и тренда
        direction = analysis.get('basic_analysis', {}).get('direction', 'neutral')
        trend_direction = market.get('trend_direction', 'neutral')

        trend_alignment = 1.0 if direction == trend_direction else 0.3

        market_score = (trend_score + trend_alignment) / 2

        # Статистическая значимость
        stats = analysis.get('statistical_significance', {})
        if stats.get('significance_level') == 'high':
            stats_score = 0.8
        elif stats.get('significance_level') == 'low':
            stats_score = 0.4
        else:
            stats_score = 0.6

        # Общий скор
        quality_score = (
            basic_score * weights['basic'] +
            strength_score * weights['strength'] +
            market_score * weights['market'] +
            stats_score * weights['statistical']
        )

        return float(min(1.0, max(0.0, quality_score)))

    def _calculate_confidence_level(self, analysis: Dict[str, Any]) -> float:
        """Расчет уровня уверенности"""
        quality_score = analysis.get('quality_score', 0.5)
        stats = analysis.get('statistical_significance', {})

        sample_size = stats.get('sample_size', 0)
        if sample_size >= 50:
            sample_factor = 1.0
        elif sample_size >= 20:
            sample_factor = 0.8
        elif sample_size >= 10:
            sample_factor = 0.6
        else:
            sample_factor = 0.4

        confidence = quality_score * sample_factor

        # Учет силы сигнала
        signals = analysis.get('trading_signals', {})
        signal_strength = signals.get('signal_strength', 'weak')

        if signal_strength == 'strong':
            strength_factor = 1.0
        elif signal_strength == 'moderate':
            strength_factor = 0.8
        else:
            strength_factor = 0.6

        confidence *= strength_factor

        return float(min(1.0, max(0.0, confidence)))

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация рекомендаций"""
        quality_score = analysis.get('quality_score', 0)
        confidence = analysis.get('confidence_level', 0)
        signals = analysis.get('trading_signals', {})
        risk = analysis.get('risk_analysis', {})

        # Основная рекомендация
        if quality_score >= 0.7 and confidence >= 0.6:
            primary_action = 'strong_buy' if signals.get('direction') == 'bullish' else 'strong_sell'
            position_size = 'full'
        elif quality_score >= 0.5 and confidence >= 0.4:
            primary_action = 'moderate_buy' if signals.get('direction') == 'bullish' else 'moderate_sell'
            position_size = 'half'
        else:
            primary_action = 'avoid'
            position_size = 'none'

        # Управление рисками
        risk_level = risk.get('risk_level', 'medium')
        if risk_level == 'high':
            risk_management = 'reduce_position_size'
        elif risk_level == 'medium':
            risk_management = 'normal_position'
        else:
            risk_management = 'can_increase_position'

        # Временные рамки
        time_horizon = signals.get('time_horizon', 'medium_term')

        return {
            'primary_action': primary_action,
            'position_size': position_size,
            'risk_management': risk_management,
            'time_horizon': time_horizon,
            'confidence_level': confidence,
            'quality_tier': self._get_quality_tier(quality_score),
            'entry_conditions': self._get_entry_conditions(signals),
            'exit_strategy': self._get_exit_strategy(signals, risk)
        }

    def _get_quality_tier(self, quality_score: float) -> str:
        """Определение уровня качества"""
        if quality_score >= 0.8:
            return 'A'
        elif quality_score >= 0.7:
            return 'B'
        elif quality_score >= 0.6:
            return 'C'
        elif quality_score >= 0.5:
            return 'D'
        else:
            return 'F'

    def _get_entry_conditions(self, signals: Dict[str, Any]) -> List[str]:
        """Получение условий входа"""
        conditions = []

        entry_signal = signals.get('entry_signal')
        if entry_signal in ['buy', 'sell']:
            conditions.append(f"Вход по цене: {signals.get('entry_price', 0):.4f}")

        distance = signals.get('distance_to_entry_pct', 0)
        if distance > 2:
            conditions.append(f"Текущая цена на {distance:.1f}% от целевой - ждать коррекции")
        elif distance <= 1:
            conditions.append("Цена в зоне входа")

        if signals.get('signal_strength') == 'strong':
            conditions.append("Сильный сигнал - можно входить агрессивно")

        return conditions

    def _get_exit_strategy(self,
                          signals: Dict[str, Any],
                          risk: Dict[str, Any]) -> Dict[str, Any]:
        """Получение стратегии выхода"""
        entry_price = signals.get('entry_price', 0)
        stop_loss = signals.get('stop_loss', 0)
        take_profit = signals.get('take_profit', 0)

        if entry_price <= 0 or stop_loss <= 0:
            return {}

        # Уровни тейк-профита
        tp_levels = []
        if take_profit > 0:
            tp1 = take_profit
            tp_levels.append({
                'level': 1,
                'price': tp1,
                'reward_ratio': abs(tp1 - entry_price) / abs(entry_price - stop_loss)
            })

            # Дополнительные уровни
            if abs(tp1 - entry_price) > 2 * abs(entry_price - stop_loss):
                tp_levels.append({
                    'level': 2,
                    'price': entry_price + (tp1 - entry_price) * 0.5,
                    'reward_ratio': 0.5
                })

        # Динамический стоп-лосс
        dynamic_sl = {
            'break_even': entry_price,
            'trailing_start': entry_price + abs(tp1 - entry_price) * 0.3 if tp_levels else entry_price * 1.02,
            'trailing_step': abs(entry_price - stop_loss) * 0.5
        }

        return {
            'stop_loss': stop_loss,
            'take_profit_levels': tp_levels,
            'dynamic_stop_loss': dynamic_sl,
            'risk_reward_ratio': risk.get('risk_reward_ratio', 0),
            'exit_conditions': [
                "При достижении тейк-профита",
                "При срабатывании стоп-лосса",
                "При изменении рыночных условий"
            ]
        }

