import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PatternAnalyzer:
    """
    Класс для анализа эффективности торговых паттернов
    """

    def __init__(self, lookforward_bars: int = 20):
        """
        Инициализация анализатора паттернов

        Args:
            lookforward_bars (int): Количество баров для анализа после паттерна
        """
        self.lookforward_bars = lookforward_bars
        self.analysis_results = {}
        self.pattern_performance = {}

    def analyze_patterns(self, patterns: List[Dict[str, Any]],
                         price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализ эффективности паттернов

        Args:
            patterns (list): Список паттернов
            price_data (pd.DataFrame): Ценовые данные

        Returns:
            dict: Результаты анализа
        """
        if not patterns or price_data.empty:
            logger.warning("Нет данных для анализа паттернов")
            return {}

        logger.info(f"Анализ {len(patterns)} паттернов")

        try:
            # Основной анализ
            analysis = {
                'summary': self._analyze_summary(patterns),
                'performance_by_type': self._analyze_performance_by_type(patterns, price_data),
                'time_analysis': self._analyze_time_patterns(patterns),
                'market_conditions': self._analyze_market_conditions(patterns, price_data),
                'risk_metrics': self._analyze_pattern_risk(patterns, price_data),
                'recommendations': self._generate_recommendations(patterns, price_data),
                'analysis_time': datetime.now()
            }

            # Сохраняем результаты
            self.analysis_results = analysis
            self._calculate_pattern_performance(patterns, price_data)

            logger.info(f"Анализ завершен: {len(patterns)} паттернов проанализировано")

            return analysis

        except Exception as e:
            logger.error(f"Ошибка анализа паттернов: {e}")
            return {'error': str(e)}

    def _analyze_summary(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Сводный анализ паттернов

        Args:
            patterns (list): Список паттернов

        Returns:
            dict: Сводная статистика
        """
        if not patterns:
            return {}

        try:
            total_patterns = len(patterns)

            # Распределение по типам
            pattern_types = {}
            pattern_families = {}

            # Статистика уверенности
            confidences = []

            # Временные метрики
            detection_times = []
            current_time = datetime.now()

            for pattern in patterns:
                # Типы паттернов
                p_type = pattern.get('pattern_type', 'unknown')
                p_family = pattern.get('pattern_family', 'unknown')

                pattern_types[p_type] = pattern_types.get(p_type, 0) + 1
                pattern_families[p_family] = pattern_families.get(p_family, 0) + 1

                # Уверенность
                confidence = pattern.get('confidence', 0)
                if confidence is not None:
                    confidences.append(float(confidence))

                # Время обнаружения
                detection_time = pattern.get('detection_time')
                if detection_time:
                    if isinstance(detection_time, str):
                        try:
                            detection_time = datetime.fromisoformat(detection_time.replace('Z', '+00:00'))
                        except:
                            detection_time = current_time
                    detection_times.append(detection_time)

            # Расчет метрик
            avg_confidence = np.mean(confidences) if confidences else 0

            # Недавние паттерны (последние 24 часа)
            if detection_times:
                recent_count = 0
                for dt in detection_times:
                    if isinstance(dt, datetime):
                        time_diff = (current_time - dt).total_seconds()
                        if time_diff <= 86400:  # 24 часа в секундах
                            recent_count += 1
            else:
                recent_count = 0

            # Самый частый тип
            most_common_type = 'none'
            most_common_family = 'none'

            if pattern_types:
                most_common_type = max(pattern_types, key=pattern_types.get)
            if pattern_families:
                most_common_family = max(pattern_families, key=pattern_families.get)

            summary = {
                'total_patterns': total_patterns,
                'pattern_type_distribution': pattern_types,
                'pattern_family_distribution': pattern_families,
                'avg_confidence': float(avg_confidence),
                'min_confidence': float(min(confidences)) if confidences else 0,
                'max_confidence': float(max(confidences)) if confidences else 0,
                'recent_patterns_24h': recent_count,
                'most_common_type': most_common_type,
                'most_common_family': most_common_family,
                'patterns_per_day': total_patterns / max(1, len(detection_times) / 30) if detection_times else 0
            }

            return summary

        except Exception as e:
            logger.error(f"Ошибка сводного анализа: {e}")
            return {}

    def _analyze_performance_by_type(self, patterns: List[Dict[str, Any]],
                                     price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализ производительности по типам паттернов

        Args:
            patterns (list): Список паттернов
            price_data (pd.DataFrame): Ценовые данные

        Returns:
            dict: Производительность по типам
        """
        performance = {}

        if not patterns or price_data.empty:
            return performance

        try:
            # Группировка по типам
            patterns_by_type = {}
            for pattern in patterns:
                p_type = pattern.get('pattern_type', 'unknown')
                if p_type not in patterns_by_type:
                    patterns_by_type[p_type] = []
                patterns_by_type[p_type].append(pattern)

            # Анализ каждого типа
            for p_type, type_patterns in patterns_by_type.items():
                type_stats = {
                    'count': len(type_patterns),
                    'avg_confidence': 0,
                    'success_rate': 0,
                    'avg_price_change': 0,
                    'performance_metrics': {}
                }

                # Статистика уверенности
                confidences = [p.get('confidence', 0) for p in type_patterns]
                if confidences:
                    type_stats['avg_confidence'] = float(np.mean(confidences))
                    type_stats['min_confidence'] = float(min(confidences))
                    type_stats['max_confidence'] = float(max(confidences))

                # Анализ эффективности
                success_count = 0
                price_changes = []

                for pattern in type_patterns:
                    # Оцениваем успешность паттерна
                    success = self._evaluate_pattern_success(pattern, price_data)
                    if success:
                        success_count += 1

                    # Изменение цены после паттерна
                    price_change = self._calculate_price_change(pattern, price_data)
                    if price_change is not None:
                        price_changes.append(price_change)

                # Расчет метрик
                if type_patterns:
                    type_stats['success_rate'] = (success_count / len(type_patterns)) * 100

                if price_changes:
                    type_stats['avg_price_change'] = float(np.mean(price_changes))
                    type_stats['price_change_std'] = float(np.std(price_changes)) if len(price_changes) > 1 else 0
                    positive_changes = sum(1 for pc in price_changes if pc > 0)
                    type_stats['positive_change_rate'] = (positive_changes / len(price_changes)) * 100

                # Дополнительные метрики
                type_stats['performance_metrics'] = {
                    'expected_value': self._calculate_expected_value(type_patterns, price_data),
                    'risk_adjusted_return': self._calculate_risk_adjusted_return(type_patterns, price_data)
                }

                performance[p_type] = type_stats

            return performance

        except Exception as e:
            logger.error(f"Ошибка анализа производительности: {e}")
            return {}

    def _evaluate_pattern_success(self, pattern: Dict[str, Any],
                                  price_data: pd.DataFrame) -> bool:
        """
        Оценка успешности паттерна

        Args:
            pattern (dict): Данные паттерна
            price_data (pd.DataFrame): Ценовые данные

        Returns:
            bool: True если паттерн успешный
        """
        try:
            pattern_index = pattern.get('index', 0)
            if pattern_index >= len(price_data) - 5:
                return False

            # Определяем направление паттерна
            pattern_type = str(pattern.get('pattern_type', '')).lower()
            signal = str(pattern.get('signal', '')).lower()

            is_bullish = any(
                word in pattern_type for word in ['bullish', 'hammer', 'engulfing', 'gartley', 'ascending']) or \
                         signal in ['buy', 'bullish']
            is_bearish = any(word in pattern_type for word in ['bearish', 'shooting', 'doji', 'bat', 'descending']) or \
                         signal in ['sell', 'bearish']

            # Если не можем определить направление, считаем нейтральным
            if not is_bullish and not is_bearish:
                return False

            # Анализ движения цены после паттерна
            lookforward = min(self.lookforward_bars, len(price_data) - pattern_index - 1)
            if lookforward <= 0:
                return False

            future_data = price_data.iloc[pattern_index + 1:pattern_index + lookforward + 1]

            entry_price = price_data.iloc[pattern_index]['Close']
            max_price = future_data['High'].max()
            min_price = future_data['Low'].min()

            # Определяем успешность
            if is_bullish:
                # Для бычьего паттерна - цена должна вырасти
                price_change = (max_price - entry_price) / entry_price
                success = price_change > 0.005  # 0.5% рост
            else:  # is_bearish
                # Для медвежьего паттерна - цена должна упасть
                price_change = (entry_price - min_price) / entry_price
                success = price_change > 0.005  # 0.5% падение

            return success

        except Exception as e:
            logger.error(f"Ошибка оценки успешности паттерна: {e}")
            return False

    def _calculate_price_change(self, pattern: Dict[str, Any],
                                price_data: pd.DataFrame) -> Optional[float]:
        """
        Расчет изменения цены после паттерна

        Args:
            pattern (dict): Данные паттерна
            price_data (pd.DataFrame): Ценовые данные

        Returns:
            float: Изменение цены в процентах или None
        """
        try:
            pattern_index = pattern.get('index', 0)
            if pattern_index >= len(price_data) - 5:
                return None

            lookforward = min(self.lookforward_bars, len(price_data) - pattern_index - 1)
            if lookforward <= 0:
                return None

            entry_price = price_data.iloc[pattern_index]['Close']
            exit_price = price_data.iloc[pattern_index + lookforward]['Close']

            price_change = ((exit_price - entry_price) / entry_price) * 100
            return float(price_change)

        except Exception as e:
            logger.error(f"Ошибка расчета изменения цены: {e}")
            return None

    def _calculate_expected_value(self, patterns: List[Dict[str, Any]],
                                  price_data: pd.DataFrame) -> float:
        """
        Расчет математического ожидания для типа паттерна

        Args:
            patterns (list): Паттерны одного типа
            price_data (pd.DataFrame): Ценовые данные

        Returns:
            float: Математическое ожидание
        """
        if not patterns:
            return 0.0

        try:
            price_changes = []
            for pattern in patterns:
                change = self._calculate_price_change(pattern, price_data)
                if change is not None:
                    price_changes.append(change)

            if price_changes:
                return float(np.mean(price_changes))
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Ошибка расчета мат. ожидания: {e}")
            return 0.0

    def _calculate_risk_adjusted_return(self, patterns: List[Dict[str, Any]],
                                        price_data: pd.DataFrame) -> float:
        """
        Расчет риск-скорректированной доходности

        Args:
            patterns (list): Паттерны одного типа
            price_data (pd.DataFrame): Ценовые данные

        Returns:
            float: Риск-скорректированная доходность
        """
        if not patterns:
            return 0.0

        try:
            price_changes = []
            for pattern in patterns:
                change = self._calculate_price_change(pattern, price_data)
                if change is not None:
                    price_changes.append(change)

            if len(price_changes) > 1:
                avg_return = np.mean(price_changes)
                volatility = np.std(price_changes)

                if volatility > 0:
                    return float(avg_return / volatility)
                else:
                    return float(avg_return)
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Ошибка расчета риск-скорректированной доходности: {e}")
            return 0.0

    def _analyze_time_patterns(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Анализ временных закономерностей

        Args:
            patterns (list): Список паттернов

        Returns:
            dict: Временные закономерности
        """
        time_analysis = {
            'by_hour': {},
            'by_day': {},
            'by_month': {},
            'time_between_patterns': {}
        }

        if not patterns:
            return time_analysis

        try:
            detection_times = []

            for pattern in patterns:
                detection_time = pattern.get('detection_time')
                if detection_time:
                    if isinstance(detection_time, str):
                        try:
                            detection_time = datetime.fromisoformat(detection_time.replace('Z', '+00:00'))
                        except:
                            continue

                    detection_times.append(detection_time)

            if not detection_times:
                return time_analysis

            # Анализ по часам, дням, месяцам
            for dt in detection_times:
                hour = dt.hour
                day = dt.strftime('%A')
                month = dt.month

                time_analysis['by_hour'][hour] = time_analysis['by_hour'].get(hour, 0) + 1
                time_analysis['by_day'][day] = time_analysis['by_day'].get(day, 0) + 1
                time_analysis['by_month'][month] = time_analysis['by_month'].get(month, 0) + 1

            # Самые частые периоды
            if time_analysis['by_hour']:
                peak_hour = max(time_analysis['by_hour'], key=time_analysis['by_hour'].get)
                time_analysis['peak_hour'] = peak_hour

            if time_analysis['by_day']:
                peak_day = max(time_analysis['by_day'], key=time_analysis['by_day'].get)
                time_analysis['peak_day'] = peak_day

            # Время между паттернами
            if len(detection_times) > 1:
                detection_times.sort()
                time_diffs = []

                for i in range(1, len(detection_times)):
                    diff = (detection_times[i] - detection_times[i - 1]).total_seconds() / 3600  # в часах
                    time_diffs.append(diff)

                if time_diffs:
                    time_analysis['time_between_patterns'] = {
                        'avg_hours': float(np.mean(time_diffs)),
                        'median_hours': float(np.median(time_diffs)),
                        'min_hours': float(min(time_diffs)),
                        'max_hours': float(max(time_diffs))
                    }

            return time_analysis

        except Exception as e:
            logger.error(f"Ошибка анализа временных закономерностей: {e}")
            return time_analysis

    def _analyze_market_conditions(self, patterns: List[Dict[str, Any]],
                                   price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализ рыночных условий при возникновении паттернов

        Args:
            patterns (list): Список паттернов
            price_data (pd.DataFrame): Ценовые данные

        Returns:
            dict: Условия рынка
        """
        market_conditions = {
            'volatility_levels': {'high': 0, 'medium': 0, 'low': 0},
            'trend_context': {'uptrend': 0, 'downtrend': 0, 'range': 0},
            'volume_conditions': {'high': 0, 'normal': 0, 'low': 0}
        }

        if not patterns or price_data.empty:
            return market_conditions

        try:
            for pattern in patterns:
                idx = pattern.get('index', 0)
                if idx < 20 or idx >= len(price_data) - 5:
                    continue

                # Анализ волатильности
                recent_prices = price_data.iloc[max(0, idx - 20):idx]['Close']
                if len(recent_prices) > 1:
                    returns = recent_prices.pct_change().dropna()
                    if len(returns) > 0:
                        volatility = returns.std() * 100  # В процентах

                        if volatility > 2.0:
                            market_conditions['volatility_levels']['high'] += 1
                        elif volatility > 1.0:
                            market_conditions['volatility_levels']['medium'] += 1
                        else:
                            market_conditions['volatility_levels']['low'] += 1

                # Анализ тренда
                if idx >= 50:
                    sma_short = price_data.iloc[max(0, idx - 10):idx]['Close'].mean()
                    sma_long = price_data.iloc[max(0, idx - 50):idx]['Close'].mean()

                    if sma_short > sma_long * 1.02:  # 2% выше
                        market_conditions['trend_context']['uptrend'] += 1
                    elif sma_short < sma_long * 0.98:  # 2% ниже
                        market_conditions['trend_context']['downtrend'] += 1
                    else:
                        market_conditions['trend_context']['range'] += 1
                else:
                    market_conditions['trend_context']['range'] += 1

                # Анализ объема
                if 'Volume' in price_data.columns and idx >= 20:
                    current_volume = price_data.iloc[idx]['Volume']
                    avg_volume = price_data.iloc[max(0, idx - 20):idx]['Volume'].mean()

                    if avg_volume > 0:
                        volume_ratio = current_volume / avg_volume

                        if volume_ratio > 1.5:
                            market_conditions['volume_conditions']['high'] += 1
                        elif volume_ratio < 0.5:
                            market_conditions['volume_conditions']['low'] += 1
                        else:
                            market_conditions['volume_conditions']['normal'] += 1

            # Преобразуем в проценты
            total_patterns = len(patterns)
            if total_patterns > 0:
                for category in market_conditions:
                    for level in market_conditions[category]:
                        market_conditions[category][level] = (market_conditions[category][level] / total_patterns) * 100

            return market_conditions

        except Exception as e:
            logger.error(f"Ошибка анализа рыночных условий: {e}")
            return market_conditions

    def _analyze_pattern_risk(self, patterns: List[Dict[str, Any]],
                              price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализ риска паттернов

        Args:
            patterns (list): Список паттернов
            price_data (pd.DataFrame): Ценовые данные

        Returns:
            dict: Метрики риска
        """
        risk_metrics = {
            'pattern_stability': {},
            'false_signal_rate': 0,
            'risk_reward_stats': {}
        }

        if not patterns or price_data.empty:
            return risk_metrics

        try:
            # Анализ стабильности паттернов
            pattern_changes = []
            false_signals = 0

            for pattern in patterns:
                idx = pattern.get('index', 0)

                # Изменение цены после паттерна
                price_change = self._calculate_price_change(pattern, price_data)
                if price_change is not None:
                    pattern_changes.append(abs(price_change))

                # Ложные сигналы (противоположное движение)
                signal = str(pattern.get('signal', '')).lower()
                if signal in ['buy', 'bullish']:
                    if price_change is not None and price_change < -1:  # >1% падение после бычьего сигнала
                        false_signals += 1
                elif signal in ['sell', 'bearish']:
                    if price_change is not None and price_change > 1:  # >1% рост после медвежьего сигнала
                        false_signals += 1

            # Статистика стабильности
            if pattern_changes:
                risk_metrics['pattern_stability'] = {
                    'avg_price_change': float(np.mean(pattern_changes)),
                    'price_change_std': float(np.std(pattern_changes)) if len(pattern_changes) > 1 else 0,
                    'max_price_change': float(max(pattern_changes)) if pattern_changes else 0,
                    'min_price_change': float(min(pattern_changes)) if pattern_changes else 0
                }

            # Статистика ложных сигналов
            if patterns:
                risk_metrics['false_signal_rate'] = (false_signals / len(patterns)) * 100

            # Статистика риск/вознаграждение
            risk_rewards = []
            for pattern in patterns:
                rr_ratio = pattern.get('risk_reward_ratio')
                if rr_ratio:
                    try:
                        risk_rewards.append(float(rr_ratio))
                    except (ValueError, TypeError):
                        continue

            if risk_rewards:
                positive_count = sum(1 for rr in risk_rewards if rr > 1)
                risk_metrics['risk_reward_stats'] = {
                    'avg': float(np.mean(risk_rewards)),
                    'median': float(np.median(risk_rewards)),
                    'min': float(min(risk_rewards)),
                    'max': float(max(risk_rewards)),
                    'positive_ratio': (positive_count / len(risk_rewards)) * 100
                }

            return risk_metrics

        except Exception as e:
            logger.error(f"Ошибка анализа риска: {e}")
            return risk_metrics

    def _generate_recommendations(self, patterns: List[Dict[str, Any]],
                                  price_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Генерация рекомендаций на основе анализа

        Args:
            patterns (list): Список паттернов
            price_data (pd.DataFrame): Ценовые данные

        Returns:
            list: Рекомендации
        """
        recommendations = []

        if not patterns:
            recommendations.append({
                'type': 'warning',
                'message': 'No patterns detected for analysis',
                'priority': 'low'
            })
            return recommendations

        try:
            # Анализируем производительность по типам
            performance_by_type = self._analyze_performance_by_type(patterns, price_data)

            # Рекомендации по типам паттернов
            best_patterns = []
            worst_patterns = []

            for p_type, stats in performance_by_type.items():
                success_rate = stats.get('success_rate', 0)

                if success_rate > 60:
                    best_patterns.append((p_type, success_rate))
                elif success_rate < 40:
                    worst_patterns.append((p_type, success_rate))

            # Рекомендация 1: Лучшие паттерны
            if best_patterns:
                best_patterns.sort(key=lambda x: x[1], reverse=True)
                recommendations.append({
                    'type': 'recommendation',
                    'message': f"Best performing patterns: {', '.join([p[0] for p in best_patterns[:3]])}",
                    'priority': 'high',
                    'details': {
                        'success_rates': {p[0]: p[1] for p in best_patterns[:3]},
                        'action': 'Focus on these pattern types for trading'
                    }
                })

            # Рекомендация 2: Худшие паттерны
            if worst_patterns:
                recommendations.append({
                    'type': 'warning',
                    'message': f"Low performing patterns: {', '.join([p[0] for p in worst_patterns[:3]])}",
                    'priority': 'medium',
                    'details': {
                        'success_rates': {p[0]: p[1] for p in worst_patterns[:3]},
                        'action': 'Avoid or use with caution these pattern types'
                    }
                })

            # Рекомендация 3: Временные паттерны
            time_analysis = self._analyze_time_patterns(patterns)
            peak_hour = time_analysis.get('peak_hour')
            peak_day = time_analysis.get('peak_day')

            if peak_hour is not None:
                recommendations.append({
                    'type': 'info',
                    'message': f"Patterns most frequent at hour {peak_hour}:00",
                    'priority': 'low',
                    'details': {
                        'hour': peak_hour,
                        'action': 'Pay extra attention during this time period'
                    }
                })

            # Рекомендация 4: Рыночные условия
            market_conditions = self._analyze_market_conditions(patterns, price_data)

            # Анализ волатильности
            volatility_dist = market_conditions.get('volatility_levels', {})
            if volatility_dist.get('high', 0) > 50:
                recommendations.append({
                    'type': 'warning',
                    'message': 'Most patterns occur in high volatility conditions',
                    'priority': 'medium',
                    'details': {
                        'volatility_distribution': volatility_dist,
                        'action': 'Consider reducing position size or using wider stops'
                    }
                })

            # Рекомендация 5: Общие метрики
            total_patterns = len(patterns)
            if total_patterns > 50:
                recommendations.append({
                    'type': 'info',
                    'message': f'Sufficient sample size: {total_patterns} patterns analyzed',
                    'priority': 'low',
                    'details': {
                        'sample_size': total_patterns,
                        'action': 'Results are statistically significant'
                    }
                })
            elif total_patterns < 20:
                recommendations.append({
                    'type': 'warning',
                    'message': f'Small sample size: only {total_patterns} patterns analyzed',
                    'priority': 'high',
                    'details': {
                        'sample_size': total_patterns,
                        'action': 'Collect more data before making trading decisions'
                    }
                })

            # Рекомендация 6: На основе риска
            risk_metrics = self._analyze_pattern_risk(patterns, price_data)
            false_signal_rate = risk_metrics.get('false_signal_rate', 0)

            if false_signal_rate > 30:
                recommendations.append({
                    'type': 'warning',
                    'message': f'High false signal rate: {false_signal_rate:.1f}%',
                    'priority': 'high',
                    'details': {
                        'false_signal_rate': false_signal_rate,
                        'action': 'Use additional confirmation before entering trades'
                    }
                })

            return recommendations

        except Exception as e:
            logger.error(f"Ошибка генерации рекомендаций: {e}")
            return [{
                'type': 'error',
                'message': f'Error generating recommendations: {str(e)}',
                'priority': 'high'
            }]

    def _calculate_pattern_performance(self, patterns: List[Dict[str, Any]],
                                       price_data: pd.DataFrame):
        """
        Расчет производительности паттернов для внутреннего использования

        Args:
            patterns (list): Список паттернов
            price_data (pd.DataFrame): Ценовые данные
        """
        self.pattern_performance = {}

        for pattern in patterns:
            p_type = pattern.get('pattern_type', 'unknown')

            if p_type not in self.pattern_performance:
                self.pattern_performance[p_type] = {
                    'count': 0,
                    'success_count': 0,
                    'price_changes': []
                }

            self.pattern_performance[p_type]['count'] += 1

            # Оценка успешности
            success = self._evaluate_pattern_success(pattern, price_data)
            if success:
                self.pattern_performance[p_type]['success_count'] += 1

            # Изменение цены
            price_change = self._calculate_price_change(pattern, price_data)
            if price_change is not None:
                self.pattern_performance[p_type]['price_changes'].append(price_change)

    def get_best_pattern_types(self, min_samples: int = 5) -> List[Tuple[str, float]]:
        """
        Получение лучших типов паттернов

        Args:
            min_samples (int): Минимальное количество образцов

        Returns:
            list: Лучшие типы паттернов с успешностью
        """
        best_patterns = []

        for p_type, stats in self.pattern_performance.items():
            if stats['count'] >= min_samples and stats['count'] > 0:
                success_rate = (stats['success_count'] / stats['count']) * 100
                best_patterns.append((p_type, success_rate))

        # Сортировка по успешности
        best_patterns.sort(key=lambda x: x[1], reverse=True)

        return best_patterns

    def save_analysis_report(self, filename: str):
        """
        Сохранение отчета анализа

        Args:
            filename (str): Имя файла
        """
        try:
            import json
            from datetime import datetime

            report = {
                'analysis_results': self.analysis_results,
                'pattern_performance': self.pattern_performance,
                'generated_at': datetime.now().isoformat()
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, default=str, indent=2)

            logger.info(f"Отчет анализа сохранен в {filename}")

        except Exception as e:
            logger.error(f"Ошибка сохранения отчета анализа: {e}")


# Создаем глобальный экземпляр для удобства
pattern_analyzer = PatternAnalyzer()

