"""
Модуль анализа паттернов
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio

from config import config
from utils.logger import logger

@dataclass
class AnalysisResult:
    """Результат анализа паттернов"""

    patterns: List[Dict[str, Any]] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0

class PatternAnalyzer:
    """Класс для анализа паттернов"""

    def __init__(self):
        self.logger = logger.bind(module="pattern_analyzer")

        # Статистика анализа
        self.analysis_stats = {
            'total_analyses': 0,
            'patterns_analyzed': 0,
            'trading_signals_generated': 0,
            'avg_processing_time': 0.0
        }

    async def analyze_patterns(self,
                              patterns: List[Dict[str, Any]],
                              data: Dict[str, np.ndarray]) -> AnalysisResult:
        """
        Анализ найденных паттернов

        Args:
            patterns: Список паттернов для анализа
            data: Входные данные OHLC

        Returns:
            Результат анализа
        """
        import time
        start_time = time.time()

        self.logger.info(f"Анализ {len(patterns)} паттернов")

        if not patterns:
            self.logger.warning("Нет паттернов для анализа")
            return AnalysisResult()

        # Анализ каждого паттерна
        analyzed_patterns = []
        for pattern in patterns:
            analyzed = await self._analyze_single_pattern(pattern, data)
            if analyzed:
                analyzed_patterns.append(analyzed)

        # Генерация рекомендаций
        recommendations = await self._generate_recommendations(analyzed_patterns, data)

        # Оценка рисков
        risk_assessment = await self._assess_risks(analyzed_patterns, data)

        # Расчет статистики
        statistics = self._calculate_analysis_statistics(analyzed_patterns)

        # Время обработки
        processing_time = time.time() - start_time

        # Обновление статистики
        self._update_analysis_stats(len(analyzed_patterns), processing_time)

        self.logger.info(f"Анализ завершен: {len(analyzed_patterns)} паттернов за {processing_time:.2f} сек")

        return AnalysisResult(
            patterns=analyzed_patterns,
            statistics=statistics,
            recommendations=recommendations,
            risk_assessment=risk_assessment,
            processing_time=processing_time
        )

    async def _analyze_single_pattern(self,
                                     pattern: Dict[str, Any],
                                     data: Dict[str, np.ndarray]) -> Optional[Dict[str, Any]]:
        """
        Анализ одного паттерна

        Args:
            pattern: Паттерн для анализа
            data: Входные данные

        Returns:
            Проанализированный паттерн или None
        """
        try:
            # Базовый анализ
            analyzed = pattern.copy()

            # Добавление метаданных анализа
            analyzed['analysis'] = {
                'analyzed_at': self._get_timestamp(),
                'analyst': self.__class__.__name__
            }

            # Анализ качества
            quality_metrics = await self._analyze_pattern_quality(pattern, data)
            analyzed['analysis']['quality_metrics'] = quality_metrics

            # Расчет потенциальной прибыли
            profit_potential = await self._calculate_profit_potential(pattern, data)
            analyzed['analysis']['profit_potential'] = profit_potential

            # Оценка риска
            risk_metrics = await self._calculate_risk_metrics(pattern, data)
            analyzed['analysis']['risk_metrics'] = risk_metrics

            # Генерация торгового сигнала
            trading_signal = await self._generate_trading_signal(pattern, data)
            analyzed['analysis']['trading_signal'] = trading_signal

            # Временной анализ
            time_analysis = await self._analyze_timing(pattern, data)
            analyzed['analysis']['time_analysis'] = time_analysis

            # Расчет общего скора
            overall_score = self._calculate_overall_score(quality_metrics, profit_potential, risk_metrics)
            analyzed['analysis']['overall_score'] = overall_score

            # Проверка, стоит ли рассматривать паттерн для торговли
            analyzed['tradable'] = self._is_pattern_tradable(analyzed)

            return analyzed

        except Exception as e:
            self.logger.error(f"Ошибка анализа паттерна {pattern.get('name')}: {e}")
            return None

    async def _analyze_pattern_quality(self,
                                      pattern: Dict[str, Any],
                                      data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Анализ качества паттерна"""
        try:
            # Базовое качество из детектирования
            base_quality = pattern.get('quality_score', 0.5)
            base_confidence = pattern.get('confidence_score', 0.5)

            # Анализ симметрии (для геометрических паттернов)
            symmetry_score = await self._calculate_symmetry_score(pattern, data)

            # Анализ пропорций (для гармонических паттернов)
            proportion_score = await self._calculate_proportion_score(pattern, data)

            # Анализ свечей (для свечных паттернов)
            candle_score = await self._calculate_candle_score(pattern, data)

            # Анализ объема
            volume_score = await self._calculate_volume_score(pattern, data)

            # Итоговый скоринг качества
            quality_factors = {
                'base_quality': base_quality,
                'symmetry': symmetry_score,
                'proportions': proportion_score,
                'candles': candle_score,
                'volume': volume_score
            }

            # Веса факторов
            weights = {
                'base_quality': 0.3,
                'symmetry': 0.2,
                'proportions': 0.2,
                'candles': 0.2,
                'volume': 0.1
            }

            # Расчет итогового качества
            total_quality = 0.0
            for factor, weight in weights.items():
                total_quality += quality_factors.get(factor, 0.5) * weight

            return {
                'scores': quality_factors,
                'weights': weights,
                'total_quality': float(total_quality),
                'adjusted_quality': float((total_quality + base_quality) / 2)
            }

        except Exception as e:
            self.logger.error(f"Ошибка анализа качества: {e}")
            return {'total_quality': pattern.get('quality_score', 0.5)}

    async def _calculate_symmetry_score(self,
                                       pattern: Dict[str, Any],
                                       data: Dict[str, np.ndarray]) -> float:
        """Расчет скора симметрии паттерна"""
        try:
            points = pattern.get('points', [])
            if len(points) < 4:
                return 0.5

            # Для паттернов с четным количеством точек
            if pattern.get('type') == 'geometric':
                # Проверка симметрии ценовых уровней
                price_points = [p.get('price', 0) for p in points if 'price' in p]
                if len(price_points) >= 4:
                    # Расчет коэффициента симметрии
                    left_side = abs(price_points[1] - price_points[0])
                    right_side = abs(price_points[3] - price_points[2])

                    if left_side > 0 and right_side > 0:
                        ratio = min(left_side, right_side) / max(left_side, right_side)
                        return float(ratio)

            return 0.5

        except Exception as e:
            self.logger.debug(f"Ошибка расчета симметрии: {e}")
            return 0.5

    async def _calculate_proportion_score(self,
                                        pattern: Dict[str, Any],
                                        data: Dict[str, np.ndarray]) -> float:
        """Расчет скора пропорций паттерна"""
        try:
            if pattern.get('type') != 'harmonic':
                return 0.5

            fib_levels = pattern.get('metadata', {}).get('original_pattern', {}).get('fibonacci_levels', {})
            if not fib_levels:
                return 0.5

            # Идеальные пропорции для гармонических паттернов
            ideal_proportions = {
                'gartley': {'AB': 0.618, 'BC': 0.618, 'CD': 1.618},
                'butterfly': {'AB': 0.786, 'BC': 0.618, 'CD': 2.618},
                'bat': {'AB': 0.382, 'BC': 0.618, 'CD': 2.618},
                'crab': {'AB': 0.382, 'BC': 0.618, 'CD': 3.618},
                'shark': {'AB': 0.382, 'BC': 1.130, 'CD': 1.618}
            }

            pattern_name = pattern.get('name', '').lower()
            if pattern_name not in ideal_proportions:
                return 0.5

            ideal = ideal_proportions[pattern_name]
            total_error = 0.0

            for level, ideal_value in ideal.items():
                actual_value = fib_levels.get(level, 0.0)
                error = abs(actual_value - ideal_value) / ideal_value if ideal_value > 0 else 1.0
                total_error += error

            avg_error = total_error / len(ideal)
            score = max(0.0, 1.0 - avg_error)

            return float(score)

        except Exception as e:
            self.logger.debug(f"Ошибка расчета пропорций: {e}")
            return 0.5

    async def _calculate_candle_score(self,
                                     pattern: Dict[str, Any],
                                     data: Dict[str, np.ndarray]) -> float:
        """Расчет скора свечей для паттерна"""
        try:
            if pattern.get('type') != 'candlestick':
                return 0.5

            # Анализ свечей вокруг паттерна
            points = pattern.get('points', [])
            if not points:
                return 0.5

            # Находим свечи паттерна
            pattern_indices = [p.get('index', 0) for p in points]
            if not pattern_indices:
                return 0.5

            start_idx = min(pattern_indices)
            end_idx = max(pattern_indices)

            # Анализируем свечи до и после паттерна
            window_size = 5
            pre_pattern_idx = max(0, start_idx - window_size)
            post_pattern_idx = min(len(data['close']) - 1, end_idx + window_size)

            # Анализ тренда до паттерна
            pre_close_prices = data['close'][pre_pattern_idx:start_idx]
            if len(pre_close_prices) > 1:
                pre_trend = np.polyfit(range(len(pre_close_prices)), pre_close_prices, 1)[0]
            else:
                pre_trend = 0

            # Анализ тренда после паттерна
            post_close_prices = data['close'][end_idx:post_pattern_idx]
            if len(post_close_prices) > 1:
                post_trend = np.polyfit(range(len(post_close_prices)), post_close_prices, 1)[0]
            else:
                post_trend = 0

            # Оценка эффективности паттерна
            if pattern.get('direction') == 'bullish':
                # Для бычьего паттерна хотим видеть рост после него
                effectiveness = 1.0 if post_trend > 0 else 0.5
            else:
                # Для медвежьего паттерна хотим видеть падение после него
                effectiveness = 1.0 if post_trend < 0 else 0.5

            return float(effectiveness)

        except Exception as e:
            self.logger.debug(f"Ошибка анализа свечей: {e}")
            return 0.5

    async def _calculate_volume_score(self,
                                     pattern: Dict[str, Any],
                                     data: Dict[str, np.ndarray]) -> float:
        """Расчет скора объема"""
        try:
            if 'volume' not in data:
                return 0.5

            points = pattern.get('points', [])
            if not points:
                return 0.5

            # Индексы паттерна
            pattern_indices = [p.get('index', 0) for p in points]
            pattern_volumes = [data['volume'][i] for i in pattern_indices if i < len(data['volume'])]

            if not pattern_volumes:
                return 0.5

            # Средний объем паттерна
            avg_pattern_volume = np.mean(pattern_volumes)

            # Средний объем до паттерна
            start_idx = min(pattern_indices)
            lookback = min(20, start_idx)
            pre_volumes = data['volume'][start_idx - lookback:start_idx]

            if len(pre_volumes) > 0:
                avg_pre_volume = np.mean(pre_volumes)
                volume_ratio = avg_pattern_volume / avg_pre_volume if avg_pre_volume > 0 else 1.0

                # Высокий объем на паттерне - хороший знак
                if volume_ratio > 1.5:
                    return 0.8
                elif volume_ratio > 1.0:
                    return 0.6
                else:
                    return 0.4
            else:
                return 0.5

        except Exception as e:
            self.logger.debug(f"Ошибка анализа объема: {e}")
            return 0.5

    async def _calculate_profit_potential(self,
                                         pattern: Dict[str, Any],
                                         data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Расчет потенциальной прибыли"""
        try:
            targets = pattern.get('targets', {})
            entry_price = targets.get('entry_price')
            stop_loss = targets.get('stop_loss')
            take_profit = targets.get('take_profit')

            if not all([entry_price, stop_loss, take_profit]):
                return {
                    'risk_reward_ratio': 0.0,
                    'potential_profit_pips': 0.0,
                    'potential_loss_pips': 0.0,
                    'profit_probability': 0.5
                }

            # Расчет в пипсах (предполагаем 5 знаков для большинства валютных пар)
            if pattern.get('direction') == 'bullish':
                profit_pips = (take_profit - entry_price) * 10000
                loss_pips = (entry_price - stop_loss) * 10000
            else:
                profit_pips = (entry_price - take_profit) * 10000
                loss_pips = (stop_loss - entry_price) * 10000

            # Соотношение риск/прибыль
            risk_reward_ratio = profit_pips / loss_pips if loss_pips > 0 else 0.0

            # Вероятность прибыли (на основе качества паттерна)
            quality = pattern.get('quality_score', 0.5)
            confidence = pattern.get('confidence_score', 0.5)
            profit_probability = (quality + confidence) / 2

            return {
                'risk_reward_ratio': float(risk_reward_ratio),
                'potential_profit_pips': float(profit_pips),
                'potential_loss_pips': float(loss_pips),
                'profit_probability': float(profit_probability),
                'entry_price': float(entry_price),
                'stop_loss': float(stop_loss),
                'take_profit': float(take_profit)
            }

        except Exception as e:
            self.logger.error(f"Ошибка расчета прибыли: {e}")
            return {
                'risk_reward_ratio': 0.0,
                'potential_profit_pips': 0.0,
                'potential_loss_pips': 0.0,
                'profit_probability': 0.5
            }

    async def _calculate_risk_metrics(self,
                                     pattern: Dict[str, Any],
                                     data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Расчет метрик риска"""
        try:
            # Волатильность
            closes = data.get('close', [])
            if len(closes) >= 20:
                returns = np.diff(closes[-20:]) / closes[-21:-1]
                volatility = np.std(returns) * np.sqrt(252)  # Годовая волатильность
            else:
                volatility = 0.1  # По умолчанию

            # Размер позиции (на основе волатильности)
            if volatility > 0:
                position_size = min(1.0, 0.1 / volatility)  # Ограничиваем размер позиции
            else:
                position_size = 0.1

            # Максимальный допустимый риск
            max_risk = config.BACKTESTING.RISK_PER_TRADE

            # Оценка drawdown риска
            quality = pattern.get('quality_score', 0.5)
            drawdown_risk = (1.0 - quality) * 0.1  # Риск просадки

            return {
                'volatility': float(volatility),
                'position_size': float(position_size),
                'max_risk_per_trade': float(max_risk),
                'drawdown_risk': float(drawdown_risk),
                'market_conditions': self._assess_market_conditions(data),
                'risk_level': 'low' if quality > 0.7 else 'medium' if quality > 0.5 else 'high'
            }

        except Exception as e:
            self.logger.error(f"Ошибка расчета рисков: {e}")
            return {
                'volatility': 0.1,
                'position_size': 0.1,
                'max_risk_per_trade': 0.02,
                'drawdown_risk': 0.05,
                'market_conditions': 'unknown',
                'risk_level': 'medium'
            }

    def _assess_market_conditions(self, data: Dict[str, np.ndarray]) -> str:
        """Оценка рыночных условий"""
        try:
            closes = data.get('close', [])
            if len(closes) < 20:
                return 'neutral'

            # Расчет тренда
            trend_coef = np.polyfit(range(20), closes[-20:], 1)[0]

            # Расчет волатильности
            returns = np.diff(closes[-20:]) / closes[-21:-1]
            volatility = np.std(returns)

            # Классификация
            if abs(trend_coef) > 0.001 and volatility < 0.005:
                return 'trending_low_vol'
            elif abs(trend_coef) > 0.001 and volatility >= 0.005:
                return 'trending_high_vol'
            elif abs(trend_coef) <= 0.001 and volatility < 0.005:
                return 'ranging_low_vol'
            else:
                return 'ranging_high_vol'

        except Exception:
            return 'unknown'

    async def _generate_trading_signal(self,
                                      pattern: Dict[str, Any],
                                      data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Генерация торгового сигнала"""
        try:
            quality = pattern.get('quality_score', 0.5)
            profit_potential = pattern.get('analysis', {}).get('profit_potential', {})
            risk_reward = profit_potential.get('risk_reward_ratio', 0.0)

            # Определение силы сигнала
            if quality >= 0.8 and risk_reward >= 2.0:
                signal_strength = 'strong'
                action = 'enter'
            elif quality >= 0.6 and risk_reward >= 1.5:
                signal_strength = 'moderate'
                action = 'enter'
            elif quality >= 0.5 and risk_reward >= 1.0:
                signal_strength = 'weak'
                action = 'monitor'
            else:
                signal_strength = 'very_weak'
                action = 'avoid'

            # Целевые уровни
            targets = pattern.get('targets', {})

            return {
                'action': action,
                'strength': signal_strength,
                'direction': pattern.get('direction', 'neutral'),
                'entry_price': targets.get('entry_price'),
                'stop_loss': targets.get('stop_loss'),
                'take_profit': targets.get('take_profit'),
                'confidence': pattern.get('confidence_score', 0.5),
                'valid_until': self._calculate_signal_expiry(pattern),
                'reasoning': f"Pattern quality: {quality:.2f}, Risk/Reward: {risk_reward:.2f}"
            }

        except Exception as e:
            self.logger.error(f"Ошибка генерации сигнала: {e}")
            return {
                'action': 'avoid',
                'strength': 'very_weak',
                'direction': 'neutral',
                'confidence': 0.0
            }

    async def _analyze_timing(self,
                             pattern: Dict[str, Any],
                             data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Анализ временных аспектов паттерна"""
        try:
            points = pattern.get('points', [])
            if not points:
                return {'age': 0, 'time_score': 0.5}

            # Возраст паттерна
            last_point = max(points, key=lambda x: x.get('index', 0))
            current_index = len(data.get('close', [])) - 1
            age = current_index - last_point.get('index', current_index)

            # Оптимальный возраст для входа (0-5 свечей после паттерна)
            if age <= 5:
                time_score = 0.8
            elif age <= 10:
                time_score = 0.6
            elif age <= 20:
                time_score = 0.4
            else:
                time_score = 0.2

            return {
                'age': age,
                'time_score': time_score,
                'optimal_entry_window': age <= 5,
                'expired': age > 20
            }

        except Exception as e:
            self.logger.debug(f"Ошибка временного анализа: {e}")
            return {'age': 0, 'time_score': 0.5}

    def _calculate_overall_score(self,
                                quality_metrics: Dict[str, Any],
                                profit_potential: Dict[str, Any],
                                risk_metrics: Dict[str, Any]) -> float:
        """Расчет общего скора паттерна"""
        try:
            weights = {
                'quality': 0.4,
                'profit': 0.3,
                'risk': 0.3
            }

            quality_score = quality_metrics.get('adjusted_quality', 0.5)
            profit_score = min(1.0, profit_potential.get('risk_reward_ratio', 0.0) / 3.0)
            risk_score = 1.0 - risk_metrics.get('drawdown_risk', 0.1) * 10

            overall = (
                quality_score * weights['quality'] +
                profit_score * weights['profit'] +
                risk_score * weights['risk']
            )

            return float(max(0.0, min(1.0, overall)))

        except Exception:
            return 0.5

    def _is_pattern_tradable(self, pattern: Dict[str, Any]) -> bool:
        """Проверка, пригоден ли паттерн для торговли"""
        try:
            analysis = pattern.get('analysis', {})

            # Минимальные требования
            if analysis.get('overall_score', 0) < 0.6:
                return False

            signal = analysis.get('trading_signal', {})
            if signal.get('action') != 'enter':
                return False

            if signal.get('strength') == 'very_weak':
                return False

            # Проверка временных параметров
            time_analysis = analysis.get('time_analysis', {})
            if time_analysis.get('expired', False):
                return False

            return True

        except Exception:
            return False

    async def _generate_recommendations(self,
                                      patterns: List[Dict[str, Any]],
                                      data: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Генерация торговых рекомендаций"""
        recommendations = []

        # Сортируем паттерны по общему скору
        tradable_patterns = [p for p in patterns if p.get('tradable', False)]
        sorted_patterns = sorted(
            tradable_patterns,
            key=lambda x: x.get('analysis', {}).get('overall_score', 0),
            reverse=True
        )

        # Берем топ-3 паттерна
        for pattern in sorted_patterns[:3]:
            analysis = pattern.get('analysis', {})
            signal = analysis.get('trading_signal', {})

            recommendation = {
                'pattern_id': pattern.get('id'),
                'pattern_name': pattern.get('name'),
                'symbol': pattern.get('symbol'),
                'timeframe': pattern.get('timeframe'),
                'action': signal.get('action', 'avoid'),
                'direction': pattern.get('direction', 'neutral'),
                'strength': signal.get('strength', 'very_weak'),
                'overall_score': analysis.get('overall_score', 0),
                'risk_reward_ratio': analysis.get('profit_potential', {}).get('risk_reward_ratio', 0),
                'entry_price': signal.get('entry_price'),
                'stop_loss': signal.get('stop_loss'),
                'take_profit': signal.get('take_profit'),
                'confidence': pattern.get('confidence_score', 0.5),
                'valid_until': signal.get('valid_until'),
                'reasoning': signal.get('reasoning', ''),
                'timestamp': self._get_timestamp()
            }

            recommendations.append(recommendation)

        return recommendations

    async def _assess_risks(self,
                           patterns: List[Dict[str, Any]],
                           data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Оценка общих рисков"""
        try:
            if not patterns:
                return {'overall_risk': 'low', 'market_risk': 'unknown'}

            # Оценка рыночного риска
            market_conditions = self._assess_market_conditions(data)

            # Количество сильных сигналов
            strong_signals = sum(
                1 for p in patterns
                if p.get('analysis', {}).get('trading_signal', {}).get('strength') == 'strong'
            )

            # Среднее качество паттернов
            avg_quality = np.mean([p.get('quality_score', 0) for p in patterns])

            # Оценка общего риска
            if market_conditions in ['trending_high_vol', 'ranging_high_vol']:
                market_risk = 'high'
            elif market_conditions in ['trending_low_vol', 'ranging_low_vol']:
                market_risk = 'medium'
            else:
                market_risk = 'unknown'

            # Определение общего уровня риска
            if avg_quality > 0.7 and strong_signals >= 2:
                overall_risk = 'low'
            elif avg_quality > 0.5:
                overall_risk = 'medium'
            else:
                overall_risk = 'high'

            return {
                'overall_risk': overall_risk,
                'market_risk': market_risk,
                'market_conditions': market_conditions,
                'avg_pattern_quality': float(avg_quality),
                'strong_signals_count': strong_signals,
                'total_patterns': len(patterns),
                'recommendation': 'trade' if overall_risk == 'low' else 'caution' if overall_risk == 'medium' else 'avoid'
            }

        except Exception as e:
            self.logger.error(f"Ошибка оценки рисков: {e}")
            return {'overall_risk': 'unknown', 'market_risk': 'unknown'}

    def _calculate_analysis_statistics(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Расчет статистики анализа"""
        if not patterns:
            return {
                'total_analyzed': 0,
                'tradable_patterns': 0,
                'avg_overall_score': 0.0,
                'signal_distribution': {}
            }

        tradable_count = sum(1 for p in patterns if p.get('tradable', False))
        overall_scores = [p.get('analysis', {}).get('overall_score', 0) for p in patterns]

        # Распределение сигналов
        signal_distribution = {}
        for p in patterns:
            signal = p.get('analysis', {}).get('trading_signal', {}).get('action', 'avoid')
            signal_distribution[signal] = signal_distribution.get(signal, 0) + 1

        return {
            'total_analyzed': len(patterns),
            'tradable_patterns': tradable_count,
            'tradable_percentage': tradable_count / len(patterns) if patterns else 0,
            'avg_overall_score': float(np.mean(overall_scores)) if overall_scores else 0.0,
            'max_overall_score': float(np.max(overall_scores)) if overall_scores else 0.0,
            'min_overall_score': float(np.min(overall_scores)) if overall_scores else 0.0,
            'signal_distribution': signal_distribution,
            'timestamp': self._get_timestamp()
        }

    def _update_analysis_stats(self, patterns_count: int, processing_time: float):
        """Обновление статистики анализа"""
        self.analysis_stats['total_analyses'] += 1
        self.analysis_stats['patterns_analyzed'] += patterns_count

        # Обновление среднего времени обработки
        current_avg = self.analysis_stats['avg_processing_time']
        total_analyses = self.analysis_stats['total_analyses']

        new_avg = (current_avg * (total_analyses - 1) + processing_time) / total_analyses
        self.analysis_stats['avg_processing_time'] = new_avg

        # Обновление счетчика сигналов
        self.analysis_stats['trading_signals_generated'] += patterns_count

    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Получение статистики анализа"""
        return self.analysis_stats.copy()

    def _get_timestamp(self) -> str:
        """Получение текущей временной метки"""
        return datetime.now().isoformat()

    def _calculate_signal_expiry(self, pattern: Dict[str, Any]) -> str:
        """Расчет срока действия сигнала"""
        expiry_hours = {
            'strong': 24,
            'moderate': 12,
            'weak': 6,
            'very_weak': 0
        }

        signal_strength = pattern.get('analysis', {}).get('trading_signal', {}).get('strength', 'very_weak')
        hours = expiry_hours.get(signal_strength, 0)

        if hours > 0:
            expiry = datetime.now() + timedelta(hours=hours)
            return expiry.isoformat()
        else:
            return datetime.now().isoformat()

