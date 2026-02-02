"""
Анализатор паттернов для поиска исторических аналогов и статистического анализа
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from scipy.spatial.distance import euclidean
from sklearn.neighbors import NearestNeighbors
import pandas as pd

from patterns import BasePattern, PatternType, PatternDirection
from config import ANALYSIS_CONFIG
from utils.logger import logger


@dataclass
class PatternFeatureVector:
    """Вектор признаков паттерна для сравнения"""

    pattern_id: str
    pattern_type: str
    direction: str

    # Геометрические признаки
    normalized_height: float  # Высота паттерна (нормализованная)
    normalized_width: float  # Ширина паттерна (нормализованная)
    symmetry_score: float  # Оценка симметрии
    complexity_score: float  # Оценка сложности

    # Контекстные признаки
    trend_strength: float  # Сила тренда перед паттерном
    volatility: float  # Волатильность
    volume_ratio: float  # Отношение объема

    # Временные признаки
    timeframe_multiplier: float  # Множитель таймфрейма
    hour_of_day: float  # Час дня (нормализованный)
    day_of_week: float  # День недели (нормализованный)

    # Индикаторы
    rsi_value: float  # Значение RSI
    macd_value: float  # Значение MACD
    adx_value: float  # Значение ADX

    # Результат
    success: Optional[bool] = None
    profit_loss_ratio: Optional[float] = None
    holding_period: Optional[float] = None

    @property
    def feature_array(self) -> np.ndarray:
        """Вектор признаков как numpy array"""
        features = [
            self.normalized_height,
            self.normalized_width,
            self.symmetry_score,
            self.complexity_score,
            self.trend_strength,
            self.volatility,
            self.volume_ratio,
            self.timeframe_multiplier,
            self.hour_of_day,
            self.day_of_week,
            self.rsi_value,
            self.macd_value,
            self.adx_value,
        ]
        return np.array(features)

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type,
            'direction': self.direction,
            'features': self.feature_array.tolist(),
            'success': self.success,
            'profit_loss_ratio': self.profit_loss_ratio,
            'holding_period': self.holding_period
        }


class PatternAnalyzer:
    """Анализатор паттернов для поиска аналогов и статистики"""

    def __init__(self, config: ANALYSIS_CONFIG = None):
        self.config = config or ANALYSIS_CONFIG
        self.logger = logger.bind(name="PatternAnalyzer")

        # База векторов признаков
        self.feature_vectors: List[PatternFeatureVector] = []

        # Модель поиска ближайших соседей
        self.nn_model: Optional[NearestNeighbors] = None

        # Статистика
        self.analysis_stats = {
            'total_patterns_analyzed': 0,
            'similar_patterns_found': 0,
            'avg_similarity_score': 0.0,
            'prediction_accuracy': 0.0
        }

    def extract_features(self, pattern: Dict[str, Any]) -> PatternFeatureVector:
        """
        Извлечение признаков из паттерна

        Args:
            pattern: Словарь с данными паттерна

        Returns:
            Вектор признаков
        """
        try:
            # Базовые признаки
            pattern_type = pattern.get('type', 'unknown')
            direction = pattern.get('direction', 'neutral')

            # Геометрические признаки
            points = pattern.get('points', [])
            prices = [p['price'] for p in points] if points else [0]

            # Высота и ширина паттерна
            pattern_height = max(prices) - min(prices) if len(prices) > 1 else 0
            pattern_width = len(points)

            # Нормализация
            avg_price = np.mean(prices) if prices else 1
            normalized_height = pattern_height / avg_price if avg_price > 0 else 0
            normalized_width = pattern_width / 100  # Нормализуем к 100 свечам

            # Симметрия и сложность
            symmetry_score = pattern.get('strength_analysis', {}).get('geometric_quality', 0.5)
            complexity_score = pattern.get('complexity_level', 1) / 3  # Нормализуем к 0-1

            # Контекстные признаки
            metadata = pattern.get('metadata', {})
            trend_strength = metadata.get('adx_value', 0) / 100  # ADX 0-100 -> 0-1
            volatility = metadata.get('volatility_pct', 0)
            volume_ratio = metadata.get('average_volume', 1) / 1000000  # Нормализуем

            # Временные признаки
            detected_time = datetime.fromisoformat(pattern.get('detection_time', datetime.now().isoformat()))
            hour_of_day = detected_time.hour / 24  # Нормализуем к 0-1
            day_of_week = detected_time.weekday() / 7  # Нормализуем к 0-1

            # Множитель таймфрейма
            timeframe = metadata.get('timeframe', 'H1')
            timeframe_map = {
                'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
                'H1': 60, 'H4': 240, 'D1': 1440, 'W1': 10080, 'MN': 43200
            }
            timeframe_multiplier = np.log(timeframe_map.get(timeframe, 60)) / np.log(43200)

            # Индикаторы
            rsi_value = metadata.get('rsi_value', 50) / 100  # RSI 0-100 -> 0-1
            macd_value = metadata.get('macd_value', 0) / 100  # Нормализуем
            adx_value = metadata.get('adx_value', 0) / 100  # ADX 0-100 -> 0-1

            # Создаем вектор признаков
            feature_vector = PatternFeatureVector(
                pattern_id=pattern.get('id', 'unknown'),
                pattern_type=pattern_type,
                direction=direction,
                normalized_height=normalized_height,
                normalized_width=normalized_width,
                symmetry_score=symmetry_score,
                complexity_score=complexity_score,
                trend_strength=trend_strength,
                volatility=volatility,
                volume_ratio=volume_ratio,
                timeframe_multiplier=timeframe_multiplier,
                hour_of_day=hour_of_day,
                day_of_week=day_of_week,
                rsi_value=rsi_value,
                macd_value=macd_value,
                adx_value=adx_value
            )

            return feature_vector

        except Exception as e:
            self.logger.error(f"Ошибка извлечения признаков: {e}")

            # Возвращаем вектор по умолчанию при ошибке
            return PatternFeatureVector(
                pattern_id='error',
                pattern_type='unknown',
                direction='neutral',
                normalized_height=0,
                normalized_width=0,
                symmetry_score=0.5,
                complexity_score=0.5,
                trend_strength=0.5,
                volatility=0,
                volume_ratio=1,
                timeframe_multiplier=0.5,
                hour_of_day=0.5,
                day_of_week=0.5,
                rsi_value=0.5,
                macd_value=0,
                adx_value=0.5
            )

    def find_similar_patterns(self,
                              pattern: Dict[str, Any],
                              historical_patterns: List[Dict[str, Any]],
                              n_neighbors: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """
        Поиск исторически похожих паттернов

        Args:
            pattern: Текущий паттерн для сравнения
            historical_patterns: Исторические паттерны для поиска
            n_neighbors: Количество ближайших соседей

        Returns:
            List кортежей (паттерн, оценка схожести)
        """
        if not historical_patterns:
            return []

        try:
            # Извлекаем признаки текущего паттерна
            current_features = self.extract_features(pattern)

            # Извлекаем признаки исторических паттернов
            historical_features = []
            historical_patterns_filtered = []

            for hist_pattern in historical_patterns:
                # Фильтруем по типу и направлению
                if (hist_pattern.get('type') == pattern.get('type') and
                        hist_pattern.get('direction') == pattern.get('direction')):
                    features = self.extract_features(hist_pattern)
                    historical_features.append(features.feature_array)
                    historical_patterns_filtered.append(hist_pattern)

            if not historical_features:
                return []

            # Преобразуем в numpy array
            X = np.array(historical_features)

            # Используем KNN для поиска ближайших соседей
            if len(historical_features) >= n_neighbors:
                nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(historical_features)),
                                      metric='euclidean')
                nn.fit(X)

                # Находим ближайшие к текущему паттерну
                distances, indices = nn.kneighbors([current_features.feature_array])

                # Формируем результат
                similar_patterns = []
                for i, idx in enumerate(indices[0]):
                    hist_pattern = historical_patterns_filtered[idx]
                    similarity = 1 / (1 + distances[0][i])  # Преобразуем расстояние в схожесть
                    similar_patterns.append((hist_pattern, similarity))

                return similar_patterns

            else:
                # Если мало исторических данных, используем прямое сравнение
                similarities = []
                for hist_pattern, features in zip(historical_patterns_filtered, historical_features):
                    # Евклидово расстояние
                    distance = euclidean(current_features.feature_array, features)
                    similarity = 1 / (1 + distance)
                    similarities.append((hist_pattern, similarity))

                # Сортируем по убыванию схожести
                similarities.sort(key=lambda x: x[1], reverse=True)
                return similarities[:n_neighbors]

        except Exception as e:
            self.logger.error(f"Ошибка поиска похожих паттернов: {e}")
            return []

    def calculate_success_rate(self, similar_patterns: List[Tuple[Dict[str, Any], float]]) -> float:
        """
        Расчет успешности на основе похожих паттернов

        Args:
            similar_patterns: Список похожих паттернов с оценками схожести

        Returns:
            Процент успешности (0-1)
        """
        if not similar_patterns:
            return 0.5  # Возвращаем 50% при отсутствии данных

        try:
            successes = []
            weights = []

            for pattern, similarity in similar_patterns:
                # Проверяем, есть ли информация об успешности
                stats = pattern.get('statistics', {})
                success_rate = stats.get('historical_success_rate', 0)

                # Если есть конкретный результат
                if 'outcome' in pattern:
                    outcome = pattern['outcome']
                    if outcome in ['success', 'profit', 'win']:
                        successes.append(1.0)
                    elif outcome in ['failure', 'loss']:
                        successes.append(0.0)
                    else:
                        successes.append(success_rate)
                else:
                    successes.append(success_rate)

                # Вес на основе схожести
                weights.append(similarity)

            if not weights:
                return 0.5

            # Взвешенное среднее
            total_weight = sum(weights)
            if total_weight > 0:
                weighted_success = sum(s * w for s, w in zip(successes, weights)) / total_weight
                return float(weighted_success)
            else:
                return 0.5

        except Exception as e:
            self.logger.error(f"Ошибка расчета успешности: {e}")
            return 0.5

    def calculate_average_profit(self, similar_patterns: List[Tuple[Dict[str, Any], float]]) -> float:
        """
        Расчет средней прибыли на основе похожих паттернов

        Args:
            similar_patterns: Список похожих паттернов

        Returns:
            Средняя прибыль в пунктах
        """
        if not similar_patterns:
            return 0.0

        try:
            profits = []
            weights = []

            for pattern, similarity in similar_patterns:
                # Получаем цели паттерна
                targets = pattern.get('targets', {})
                entry = targets.get('entry_price')
                take_profit = targets.get('take_profit')

                if entry and take_profit:
                    # Рассчитываем прибыль в пунктах
                    if pattern.get('direction') == 'bullish':
                        profit = take_profit - entry
                    else:
                        profit = entry - take_profit

                    profits.append(profit)
                    weights.append(similarity)

            if not profits:
                return 0.0

            # Взвешенное среднее
            total_weight = sum(weights)
            if total_weight > 0:
                weighted_profit = sum(p * w for p, w in zip(profits, weights)) / total_weight
                return float(weighted_profit)
            else:
                return np.mean(profits) if profits else 0.0

        except Exception as e:
            self.logger.error(f"Ошибка расчета средней прибыли: {e}")
            return 0.0

    def analyze_pattern_quality(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Полный анализ качества паттерна

        Args:
            pattern: Паттерн для анализа

        Returns:
            Словарь с результатами анализа
        """
        analysis = {
            'geometric_quality': pattern.get('metadata', {}).get('quality_score', 0),
            'context_score': 0.0,
            'confirmation_score': 0.0,
            'risk_reward_score': 0.0,
            'historical_success_score': 0.0,
            'overall_score': 0.0
        }

        try:
            # 1. Оценка контекста
            metadata = pattern.get('metadata', {})
            market_context = metadata.get('market_context', 'neutral')
            direction = pattern.get('direction', 'neutral')

            context_score = 0.5  # Базовая оценка

            if market_context == 'uptrend' and direction == 'bullish':
                context_score = 0.9
            elif market_context == 'downtrend' and direction == 'bearish':
                context_score = 0.9
            elif market_context == 'sideways':
                context_score = 0.7
            elif market_context == 'volatile':
                context_score = 0.6

            analysis['context_score'] = context_score

            # 2. Оценка подтверждений
            confirmation_score = 0.0
            confirmations = 0
            total_possible = 3

            if metadata.get('volume_confirmation', False):
                confirmation_score += 0.3
                confirmations += 1

            if metadata.get('trend_confirmation', False):
                confirmation_score += 0.3
                confirmations += 1

            if metadata.get('indicator_confirmation', False):
                confirmation_score += 0.4
                confirmations += 1

            analysis['confirmation_score'] = confirmation_score

            # 3. Оценка риска/прибыли
            targets = pattern.get('targets', {})
            risk_reward = targets.get('profit_risk_ratio', 0)

            if risk_reward >= 3:
                risk_reward_score = 1.0
            elif risk_reward >= 2:
                risk_reward_score = 0.8
            elif risk_reward >= 1.5:
                risk_reward_score = 0.6
            elif risk_reward >= 1:
                risk_reward_score = 0.4
            else:
                risk_reward_score = 0.2

            analysis['risk_reward_score'] = risk_reward_score

            # 4. Историческая успешность
            stats = pattern.get('statistics', {})
            historical_success = stats.get('historical_success_rate', 0.5)
            analysis['historical_success_score'] = historical_success

            # 5. Общая оценка (взвешенная)
            weights = {
                'geometric_quality': 0.25,
                'context_score': 0.20,
                'confirmation_score': 0.15,
                'risk_reward_score': 0.20,
                'historical_success_score': 0.20
            }

            overall_score = (
                    analysis['geometric_quality'] * weights['geometric_quality'] +
                    analysis['context_score'] * weights['context_score'] +
                    analysis['confirmation_score'] * weights['confirmation_score'] +
                    analysis['risk_reward_score'] * weights['risk_reward_score'] +
                    analysis['historical_success_score'] * weights['historical_success_score']
            )

            analysis['overall_score'] = overall_score

            # 6. Рекомендация
            if overall_score >= 0.8:
                analysis['recommendation'] = 'STRONG_BUY' if direction == 'bullish' else 'STRONG_SELL'
                analysis['confidence'] = 'HIGH'
            elif overall_score >= 0.6:
                analysis['recommendation'] = 'BUY' if direction == 'bullish' else 'SELL'
                analysis['confidence'] = 'MEDIUM'
            elif overall_score >= 0.4:
                analysis['recommendation'] = 'WEAK_BUY' if direction == 'bullish' else 'WEAK_SELL'
                analysis['confidence'] = 'LOW'
            else:
                analysis['recommendation'] = 'HOLD'
                analysis['confidence'] = 'VERY_LOW'

            # Обновляем статистику
            self.analysis_stats['total_patterns_analyzed'] += 1

        except Exception as e:
            self.logger.error(f"Ошибка анализа качества паттерна: {e}")
            analysis['error'] = str(e)

        return analysis

    def predict_pattern_outcome(self,
                                pattern: Dict[str, Any],
                                historical_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Прогнозирование исхода паттерна

        Args:
            pattern: Паттерн для прогнозирования
            historical_patterns: Исторические данные

        Returns:
            Словарь с прогнозом
        """
        prediction = {
            'probability_success': 0.5,
            'expected_profit': 0.0,
            'expected_risk': 0.0,
            'expected_holding_period': 0.0,
            'confidence': 0.0
        }

        try:
            # Находим похожие паттерны
            similar_patterns = self.find_similar_patterns(
                pattern, historical_patterns, n_neighbors=20
            )

            if not similar_patterns:
                return prediction

            # Рассчитываем вероятность успеха
            success_rate = self.calculate_success_rate(similar_patterns)

            # Рассчитываем среднюю прибыль
            avg_profit = self.calculate_average_profit(similar_patterns)

            # Рассчитываем риск (на основе стоп-лосса похожих паттернов)
            avg_risk = self._calculate_average_risk(similar_patterns)

            # Рассчитываем средний период удержания
            avg_holding = self._calculate_average_holding_period(similar_patterns)

            # Уверенность прогноза (на основе схожести и количества)
            avg_similarity = np.mean([s for _, s in similar_patterns])
            count = len(similar_patterns)

            confidence = min(avg_similarity * (count / 20), 1.0)  # Нормализуем

            # Заполняем результат
            prediction['probability_success'] = success_rate
            prediction['expected_profit'] = avg_profit
            prediction['expected_risk'] = avg_risk
            prediction['expected_holding_period'] = avg_holding
            prediction['confidence'] = confidence

            # Дополнительная информация
            prediction['similar_patterns_count'] = count
            prediction['average_similarity'] = avg_similarity

            # Обновляем статистику
            self.analysis_stats['similar_patterns_found'] += count
            self.analysis_stats['avg_similarity_score'] = (
                    (self.analysis_stats['avg_similarity_score'] *
                     (self.analysis_stats['total_patterns_analyzed'] - 1) +
                     avg_similarity) / self.analysis_stats['total_patterns_analyzed']
            )

        except Exception as e:
            self.logger.error(f"Ошибка прогнозирования исхода: {e}")
            prediction['error'] = str(e)

        return prediction

    def _calculate_average_risk(self, similar_patterns: List[Tuple[Dict[str, Any], float]]) -> float:
        """Расчет среднего риска"""
        risks = []
        weights = []

        for pattern, similarity in similar_patterns:
            targets = pattern.get('targets', {})
            entry = targets.get('entry_price')
            stop_loss = targets.get('stop_loss')

            if entry and stop_loss:
                if pattern.get('direction') == 'bullish':
                    risk = entry - stop_loss
                else:
                    risk = stop_loss - entry

                risks.append(risk)
                weights.append(similarity)

        if not risks:
            return 0.0

        total_weight = sum(weights)
        if total_weight > 0:
            return sum(r * w for r, w in zip(risks, weights)) / total_weight
        else:
            return np.mean(risks) if risks else 0.0

    def _calculate_average_holding_period(self, similar_patterns: List[Tuple[Dict[str, Any], float]]) -> float:
        """Расчет среднего периода удержания"""
        periods = []
        weights = []

        for pattern, similarity in similar_patterns:
            stats = pattern.get('statistics', {})
            holding_period = stats.get('avg_holding_period', 0)

            if holding_period > 0:
                periods.append(holding_period)
                weights.append(similarity)

        if not periods:
            return 0.0

        total_weight = sum(weights)
        if total_weight > 0:
            return sum(p * w for p, w in zip(periods, weights)) / total_weight
        else:
            return np.mean(periods) if periods else 0.0

    def build_prediction_model(self, historical_patterns: List[Dict[str, Any]]):
        """
        Построение модели для прогнозирования

        Args:
            historical_patterns: Исторические паттерны для обучения
        """
        try:
            if not historical_patterns:
                self.logger.warning("Нет данных для построения модели")
                return

            # Извлекаем признаки и результаты
            X = []
            y = []

            for pattern in historical_patterns:
                # Извлекаем признаки
                features = self.extract_features(pattern)

                # Извлекаем результат (если есть)
                if 'outcome' in pattern:
                    outcome = pattern['outcome']
                    if outcome in ['success', 'profit', 'win']:
                        y.append(1)
                    elif outcome in ['failure', 'loss']:
                        y.append(0)
                    else:
                        continue  # Пропускаем если результат неясен
                else:
                    continue  # Пропускаем если нет результата

                X.append(features.feature_array)

            if len(X) < 10:
                self.logger.warning(f"Недостаточно данных для обучения: {len(X)} образцов")
                return

            # Строим модель KNN
            self.nn_model = NearestNeighbors(n_neighbors=min(10, len(X)), metric='euclidean')
            self.nn_model.fit(np.array(X))

            # Сохраняем данные для предсказаний
            self.training_patterns = historical_patterns
            self.training_features = X
            self.training_outcomes = y

            self.logger.info(f"Модель построена на {len(X)} образцах")

        except Exception as e:
            self.logger.error(f"Ошибка построения модели: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики анализа"""
        return self.analysis_stats.copy()

    def reset_statistics(self):
        """Сброс статистики"""
        self.analysis_stats = {
            'total_patterns_analyzed': 0,
            'similar_patterns_found': 0,
            'avg_similarity_score': 0.0,
            'prediction_accuracy': 0.0
        }

