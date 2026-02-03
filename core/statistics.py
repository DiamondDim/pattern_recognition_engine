# core/statistics.py

"""
Модуль статистических расчетов для паттернов
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report

from config import STATISTICS_CONFIG
from utils.logger import logger
from utils.helpers import calculate_returns, calculate_volatility


class PatternStatistics:
    """Статистический анализатор паттернов"""

    def __init__(self, config: STATISTICS_CONFIG = None):
        self.config = config or STATISTICS_CONFIG
        self.logger = logger.bind(module="PatternStatistics")

    def calculate_pattern_statistics(self,
                                   patterns: List[Dict[str, Any]],
                                   price_data: Dict[str, np.ndarray],
                                   forward_period: int = 20) -> Dict[str, Any]:
        """
        Расчет статистики паттернов

        Args:
            patterns: Список паттернов
            price_data: Ценовые данные
            forward_period: Период для оценки будущей доходности

        Returns:
            Статистика паттернов
        """
        try:
            if not patterns:
                self.logger.warning("Нет паттернов для статистики")
                return {}

            closes = price_data.get('close', np.array([]))
            if len(closes) == 0:
                self.logger.error("Нет ценовых данных")
                return {}

            # Группировка паттернов по типам
            pattern_groups = self._group_patterns_by_type(patterns)

            statistics = {
                'overall': self._calculate_overall_statistics(patterns, closes, forward_period),
                'by_type': {},
                'by_direction': {},
                'performance_metrics': {},
                'significance_tests': {}
            }

            # Статистика по типам
            for pattern_type, type_patterns in pattern_groups.items():
                statistics['by_type'][pattern_type] = self._calculate_type_statistics(
                    type_patterns, closes, forward_period)

            # Статистика по направлениям
            statistics['by_direction'] = self._calculate_direction_statistics(
                patterns, closes, forward_period)

            # Метрики производительности
            statistics['performance_metrics'] = self._calculate_performance_metrics(
                patterns, closes, forward_period)

            # Статистические тесты
            statistics['significance_tests'] = self._perform_significance_tests(
                patterns, closes, forward_period)

            return statistics

        except Exception as e:
            self.logger.error(f"Ошибка расчета статистики: {e}")
            return {}

    def _group_patterns_by_type(self, patterns: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Группировка паттернов по типам"""
        groups = {}
        for pattern in patterns:
            pattern_type = pattern.get('pattern_type', 'unknown')
            if pattern_type not in groups:
                groups[pattern_type] = []
            groups[pattern_type].append(pattern)

        return groups

    def _calculate_overall_statistics(self,
                                    patterns: List[Dict[str, Any]],
                                    closes: np.ndarray,
                                    forward_period: int) -> Dict[str, Any]:
        """Расчет общей статистики"""
        total_patterns = len(patterns)

        if total_patterns == 0:
            return {}

        # Доходности паттернов
        forward_returns = []
        directions = []
        confidences = []
        durations = []

        for pattern in patterns:
            end_idx = pattern.get('end_index', 0)
            direction = pattern.get('direction', 'neutral')
            confidence = pattern.get('confidence', 0.5)

            # Расчет будущей доходности
            if end_idx + forward_period < len(closes):
                future_return = (closes[end_idx + forward_period] / closes[end_idx]) - 1
                forward_returns.append(future_return)

                # Учет направления
                if direction == 'bearish':
                    forward_returns[-1] = -forward_returns[-1]

            directions.append(direction)
            confidences.append(confidence)

            # Длительность паттерна
            start_idx = pattern.get('start_index', 0)
            durations.append(end_idx - start_idx)

        if not forward_returns:
            return {
                'total_patterns': total_patterns,
                'avg_confidence': np.mean(confidences) if confidences else 0,
                'avg_duration': np.mean(durations) if durations else 0,
                'error': 'Недостаточно данных для расчета доходности'
            }

        forward_returns = np.array(forward_returns)

        # Базовые статистики
        stats = {
            'total_patterns': total_patterns,
            'patterns_with_returns': len(forward_returns),
            'avg_forward_return': np.mean(forward_returns),
            'median_forward_return': np.median(forward_returns),
            'std_forward_return': np.std(forward_returns),
            'min_forward_return': np.min(forward_returns),
            'max_forward_return': np.max(forward_returns),
            'positive_returns_count': np.sum(forward_returns > 0),
            'negative_returns_count': np.sum(forward_returns < 0),
            'positive_return_rate': np.mean(forward_returns > 0),
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'avg_duration': np.mean(durations) if durations else 0,
            'direction_distribution': {
                'bullish': directions.count('bullish'),
                'bearish': directions.count('bearish'),
                'neutral': directions.count('neutral')
            }
        }

        # Дополнительные метрики
        if len(forward_returns) > 1:
            stats['sharpe_ratio'] = self._calculate_sharpe_ratio(forward_returns)
            stats['sortino_ratio'] = self._calculate_sortino_ratio(forward_returns)
            stats['win_loss_ratio'] = self._calculate_win_loss_ratio(forward_returns)
            stats['profit_factor'] = self._calculate_profit_factor(forward_returns)
            stats['max_drawdown'] = self._calculate_max_drawdown(forward_returns)

        return stats

    def _calculate_type_statistics(self,
                                  patterns: List[Dict[str, Any]],
                                  closes: np.ndarray,
                                  forward_period: int) -> Dict[str, Any]:
        """Расчет статистики по типу паттерна"""
        if not patterns:
            return {}

        return self._calculate_overall_statistics(patterns, closes, forward_period)

    def _calculate_direction_statistics(self,
                                       patterns: List[Dict[str, Any]],
                                       closes: np.ndarray,
                                       forward_period: int) -> Dict[str, Any]:
        """Расчет статистики по направлениям"""
        bullish_patterns = [p for p in patterns if p.get('direction') == 'bullish']
        bearish_patterns = [p for p in patterns if p.get('direction') == 'bearish']
        neutral_patterns = [p for p in patterns if p.get('direction') == 'neutral']

        return {
            'bullish': self._calculate_overall_statistics(bullish_patterns, closes, forward_period),
            'bearish': self._calculate_overall_statistics(bearish_patterns, closes, forward_period),
            'neutral': self._calculate_overall_statistics(neutral_patterns, closes, forward_period)
        }

    def _calculate_performance_metrics(self,
                                     patterns: List[Dict[str, Any]],
                                     closes: np.ndarray,
                                     forward_period: int) -> Dict[str, Any]:
        """Расчет метрик производительности"""
        if not patterns:
            return {}

        # Собираем предсказания и фактические результаты
        predictions = []
        actuals = []
        confidences = []

        for pattern in patterns:
            end_idx = pattern.get('end_index', 0)
            direction = pattern.get('direction', 'neutral')
            confidence = pattern.get('confidence', 0.5)

            if end_idx + forward_period >= len(closes):
                continue

            # Фактическая доходность
            future_return = (closes[end_idx + forward_period] / closes[end_idx]) - 1

            # Кодирование предсказаний
            if direction == 'bullish':
                pred = 1  # Рост
            elif direction == 'bearish':
                pred = -1  # Падение
                future_return = -future_return  # Инвертируем для медвежьих паттернов
            else:
                pred = 0  # Нейтрально

            predictions.append(pred)
            actuals.append(1 if future_return > 0 else -1 if future_return < 0 else 0)
            confidences.append(confidence)

        if not predictions:
            return {'error': 'Недостаточно данных для метрик'}

        predictions = np.array(predictions)
        actuals = np.array(actuals)
        confidences = np.array(confidences)

        # Метрики классификации
        metrics = {}

        # Точность (accuracy)
        accuracy = np.mean(predictions == actuals)
        metrics['accuracy'] = accuracy

        # Точность для бычьих паттернов
        bull_mask = predictions == 1
        if np.any(bull_mask):
            bull_accuracy = np.mean(actuals[bull_mask] == 1)
            metrics['bullish_accuracy'] = bull_accuracy
            metrics['bullish_precision'] = np.sum((predictions == 1) & (actuals == 1)) / np.sum(predictions == 1)
            metrics['bullish_recall'] = np.sum((predictions == 1) & (actuals == 1)) / np.sum(actuals == 1)

        # Точность для медвежьих паттернов
        bear_mask = predictions == -1
        if np.any(bear_mask):
            bear_accuracy = np.mean(actuals[bear_mask] == -1)
            metrics['bearish_accuracy'] = bear_accuracy
            metrics['bearish_precision'] = np.sum((predictions == -1) & (actuals == -1)) / np.sum(predictions == -1)
            metrics['bearish_recall'] = np.sum((predictions == -1) & (actuals == -1)) / np.sum(actuals == -1)

        # Матрица ошибок
        try:
            cm = confusion_matrix(actuals, predictions, labels=[-1, 0, 1])
            metrics['confusion_matrix'] = cm.tolist()
        except:
            metrics['confusion_matrix'] = []

        # Качество предсказаний в зависимости от уверенности
        if len(confidences) > 0:
            # Разделяем по квантилям уверенности
            quantiles = np.quantile(confidences, [0.25, 0.5, 0.75])
            confidence_levels = ['low', 'medium', 'high', 'very_high']

            for i in range(4):
                if i == 0:
                    mask = confidences <= quantiles[0]
                elif i == 1:
                    mask = (confidences > quantiles[0]) & (confidences <= quantiles[1])
                elif i == 2:
                    mask = (confidences > quantiles[1]) & (confidences <= quantiles[2])
                else:
                    mask = confidences > quantiles[2]

                if np.any(mask):
                    level_accuracy = np.mean(actuals[mask] == predictions[mask])
                    metrics[f'accuracy_{confidence_levels[i]}_confidence'] = level_accuracy

        # ROC-кривая (упрощенная)
        thresholds = np.linspace(0, 1, 20)
        tpr_list = []
        fpr_list = []

        for threshold in thresholds:
            high_conf_mask = confidences >= threshold
            if np.any(high_conf_mask):
                tpr = np.mean(actuals[high_conf_mask] == predictions[high_conf_mask])
                fpr = np.mean(actuals[high_conf_mask] != predictions[high_conf_mask])
                tpr_list.append(tpr)
                fpr_list.append(fpr)

        if tpr_list and fpr_list:
            metrics['roc_curve'] = {
                'tpr': tpr_list,
                'fpr': fpr_list,
                'auc': np.trapz(tpr_list, fpr_list) if len(tpr_list) > 1 else 0
            }

        return metrics

    def _perform_significance_tests(self,
                                  patterns: List[Dict[str, Any]],
                                  closes: np.ndarray,
                                  forward_period: int) -> Dict[str, Any]:
        """Выполнение статистических тестов"""
        tests = {}

        # Собираем доходности паттернов
        pattern_returns = []
        for pattern in patterns:
            end_idx = pattern.get('end_index', 0)
            direction = pattern.get('direction', 'neutral')

            if end_idx + forward_period < len(closes):
                future_return = (closes[end_idx + forward_period] / closes[end_idx]) - 1

                if direction == 'bearish':
                    future_return = -future_return

                pattern_returns.append(future_return)

        if len(pattern_returns) < 10:
            tests['error'] = 'Недостаточно данных для тестов'
            return tests

        pattern_returns = np.array(pattern_returns)

        # 1. Тест на нормальность (Shapiro-Wilk)
        try:
            stat, p_value = stats.shapiro(pattern_returns)
            tests['normality_test'] = {
                'test': 'Shapiro-Wilk',
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > 0.05
            }
        except:
            tests['normality_test'] = {'error': 'Не удалось выполнить тест'}

        # 2. Тест на среднее (t-тест)
        try:
            # Нулевая гипотеза: средняя доходность = 0
            stat, p_value = stats.ttest_1samp(pattern_returns, 0)
            tests['mean_test'] = {
                'test': 'One-sample t-test',
                'statistic': stat,
                'p_value': p_value,
                'mean_significantly_different': p_value < 0.05,
                'sample_mean': np.mean(pattern_returns)
            }
        except:
            tests['mean_test'] = {'error': 'Не удалось выполнить тест'}

        # 3. Сравнение бычьих и медвежьих паттернов
        bullish_returns = []
        bearish_returns = []

        for pattern in patterns:
            end_idx = pattern.get('end_index', 0)
            direction = pattern.get('direction', '')

            if end_idx + forward_period >= len(closes):
                continue

            future_return = (closes[end_idx + forward_period] / closes[end_idx]) - 1

            if direction == 'bullish':
                bullish_returns.append(future_return)
            elif direction == 'bearish':
                bearish_returns.append(-future_return)

        if len(bullish_returns) >= 5 and len(bearish_returns) >= 5:
            try:
                stat, p_value = stats.ttest_ind(bullish_returns, bearish_returns, equal_var=False)
                tests['bullish_vs_bearish'] = {
                    'test': 'Welch\'s t-test',
                    'statistic': stat,
                    'p_value': p_value,
                    'significant_difference': p_value < 0.05,
                    'bullish_mean': np.mean(bullish_returns),
                    'bearish_mean': np.mean(bearish_returns),
                    'bullish_std': np.std(bullish_returns),
                    'bearish_std': np.std(bearish_returns)
                }
            except:
                tests['bullish_vs_bearish'] = {'error': 'Не удалось выполнить тест'}

        # 4. Тест на автокорреляцию
        if len(pattern_returns) >= 20:
            try:
                # Автокорреляция первого порядка
                autocorr = np.corrcoef(pattern_returns[:-1], pattern_returns[1:])[0, 1]
                tests['autocorrelation'] = {
                    'lag1_autocorrelation': autocorr,
                    'is_significant': abs(autocorr) > 2 / np.sqrt(len(pattern_returns))
                }
            except:
                tests['autocorrelation'] = {'error': 'Не удалось рассчитать автокорреляцию'}

        return tests

    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Расчет коэффициента Шарпа"""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - risk_free_rate / 252  # Ежедневная безрисковая ставка
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return float(sharpe)

    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Расчет коэффициента Сортино"""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - risk_free_rate / 252
        negative_returns = excess_returns[excess_returns < 0]

        if len(negative_returns) == 0:
            downside_std = 0.0
        else:
            downside_std = np.std(negative_returns)

        if downside_std == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0

        sortino = np.mean(excess_returns) / downside_std * np.sqrt(252)
        return float(sortino)

    def _calculate_win_loss_ratio(self, returns: np.ndarray) -> float:
        """Расчет соотношения выигрышных/проигрышных сделок"""
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]

        if len(losing_trades) == 0:
            return float('inf') if len(winning_trades) > 0 else 0.0

        avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0
        avg_loss = abs(np.mean(losing_trades)) if len(losing_trades) > 0 else 0

        if avg_loss == 0:
            return float('inf') if avg_win > 0 else 0.0

        return avg_win / avg_loss

    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Расчет профит-фактора"""
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Расчет максимальной просадки"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max

        if len(drawdown) == 0:
            return 0.0

        max_dd = np.min(drawdown)
        return float(abs(max_dd))

    def calculate_correlation_matrix(self,
                                   pattern_types: List[str],
                                   pattern_returns: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Расчет матрицы корреляций между типами паттернов

        Args:
            pattern_types: Список типов паттернов
            pattern_returns: Доходности по типам паттернов

        Returns:
            Матрица корреляций
        """
        try:
            # Создаем DataFrame с доходностями
            returns_data = {}
            valid_types = []

            for p_type in pattern_types:
                if p_type in pattern_returns and len(pattern_returns[p_type]) > 10:
                    returns_data[p_type] = pattern_returns[p_type]
                    valid_types.append(p_type)

            if len(valid_types) < 2:
                return {'error': 'Недостаточно данных для корреляции'}

            # Создаем матрицу корреляций
            n_types = len(valid_types)
            correlation_matrix = np.zeros((n_types, n_types))
            p_value_matrix = np.zeros((n_types, n_types))

            for i in range(n_types):
                for j in range(n_types):
                    if i == j:
                        correlation_matrix[i, j] = 1.0
                        p_value_matrix[i, j] = 0.0
                    else:
                        # Выравниваем длины
                        returns_i = returns_data[valid_types[i]]
                        returns_j = returns_data[valid_types[j]]

                        min_len = min(len(returns_i), len(returns_j))
                        if min_len < 10:
                            correlation_matrix[i, j] = 0.0
                            p_value_matrix[i, j] = 1.0
                        else:
                            corr, p_value = stats.pearsonr(returns_i[:min_len], returns_j[:min_len])
                            correlation_matrix[i, j] = corr
                            p_value_matrix[i, j] = p_value

            # Анализ корреляций
            significant_correlations = []
            for i in range(n_types):
                for j in range(i + 1, n_types):
                    if abs(correlation_matrix[i, j]) > 0.5 and p_value_matrix[i, j] < 0.05:
                        significant_correlations.append({
                            'type1': valid_types[i],
                            'type2': valid_types[j],
                            'correlation': float(correlation_matrix[i, j]),
                            'p_value': float(p_value_matrix[i, j])
                        })

            return {
                'correlation_matrix': correlation_matrix.tolist(),
                'p_value_matrix': p_value_matrix.tolist(),
                'pattern_types': valid_types,
                'significant_correlations': significant_correlations,
                'avg_correlation': float(np.mean(np.abs(correlation_matrix[np.triu_indices(n_types, k=1)])))
            }

        except Exception as e:
            self.logger.error(f"Ошибка расчета корреляций: {e}")
            return {}

    def calculate_confidence_impact(self,
                                  patterns: List[Dict[str, Any]],
                                  closes: np.ndarray,
                                  forward_period: int = 20) -> Dict[str, Any]:
        """
        Анализ влияния уверенности на точность предсказаний

        Args:
            patterns: Список паттернов
            closes: Ценовые данные
            forward_period: Период оценки

        Returns:
            Анализ влияния уверенности
        """
        try:
            if not patterns:
                return {}

            # Собираем данные
            confidences = []
            correct_predictions = []
            returns = []

            for pattern in patterns:
                end_idx = pattern.get('end_index', 0)
                direction = pattern.get('direction', 'neutral')
                confidence = pattern.get('confidence', 0.5)

                if end_idx + forward_period >= len(closes):
                    continue

                future_return = (closes[end_idx + forward_period] / closes[end_idx]) - 1

                # Определяем правильность предсказания
                if direction == 'bullish':
                    correct = future_return > 0
                elif direction == 'bearish':
                    correct = future_return < 0
                    future_return = -future_return
                else:
                    correct = abs(future_return) < 0.01  # Нейтральный - небольшие движения

                confidences.append(confidence)
                correct_predictions.append(correct)
                returns.append(future_return)

            if len(confidences) < 10:
                return {'error': 'Недостаточно данных'}

            confidences = np.array(confidences)
            correct_predictions = np.array(correct_predictions)
            returns = np.array(returns)

            # Разделение по квантилям уверенности
            quantiles = np.quantile(confidences, [0.25, 0.5, 0.75])
            bins = [
                (0, quantiles[0]),                    # Низкая уверенность
                (quantiles[0], quantiles[1]),         # Средняя уверенность
                (quantiles[1], quantiles[2]),         # Высокая уверенность
                (quantiles[2], 1)                     # Очень высокая уверенность
            ]

            results = {}
            for i, (low, high) in enumerate(bins):
                mask = (confidences >= low) & (confidences < high)
                if i == 3:  # Для последнего бина включаем верхнюю границу
                    mask = (confidences >= low) & (confidences <= high)

                if np.sum(mask) >= 3:
                    accuracy = np.mean(correct_predictions[mask])
                    avg_return = np.mean(returns[mask])
                    std_return = np.std(returns[mask]) if len(returns[mask]) > 1 else 0
                    count = np.sum(mask)

                    results[f'bin_{i+1}'] = {
                        'confidence_range': (float(low), float(high)),
                        'accuracy': float(accuracy),
                        'avg_return': float(avg_return),
                        'std_return': float(std_return),
                        'count': int(count),
                        'sharpe_ratio': float(avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0
                    }

            # Корреляция уверенности и точности
            if len(confidences) >= 10:
                corr_coef, p_value = stats.pearsonr(confidences, correct_predictions.astype(float))
                results['correlation_analysis'] = {
                    'pearson_correlation': float(corr_coef),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }

                # Линейная регрессия
                slope, intercept, r_value, p_value, std_err = stats.linregress(confidences, correct_predictions)
                results['regression_analysis'] = {
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'r_squared': float(r_value ** 2),
                    'p_value': float(p_value)
                }

            return results

        except Exception as e:
            self.logger.error(f"Ошибка анализа влияния уверенности: {e}")
            return {}

    def generate_statistical_report(self,
                                  statistics: Dict[str, Any],
                                  output_format: str = 'dict') -> Union[Dict[str, Any], str]:
        """
        Генерация статистического отчета

        Args:
            statistics: Статистические данные
            output_format: Формат вывода ('dict', 'json', 'text')

        Returns:
            Статистический отчет
        """
        try:
            if not statistics:
                return "Нет данных для отчета"

            report = {
                'summary': self._generate_summary(statistics),
                'detailed_analysis': self._generate_detailed_analysis(statistics),
                'recommendations': self._generate_recommendations(statistics),
                'timestamp': datetime.now().isoformat()
            }

            if output_format == 'json':
                import json
                return json.dumps(report, indent=2, default=str)
            elif output_format == 'text':
                return self._format_text_report(report)
            else:
                return report

        except Exception as e:
            self.logger.error(f"Ошибка генерации отчета: {e}")
            return "Ошибка генерации отчета"

    def _generate_summary(self, statistics: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация сводки"""
        overall = statistics.get('overall', {})
        performance = statistics.get('performance_metrics', {})

        return {
            'total_patterns_analyzed': overall.get('total_patterns', 0),
            'average_forward_return': overall.get('avg_forward_return', 0),
            'positive_return_rate': overall.get('positive_return_rate', 0),
            'overall_accuracy': performance.get('accuracy', 0),
            'sharpe_ratio': overall.get('sharpe_ratio', 0),
            'profit_factor': overall.get('profit_factor', 0),
            'max_drawdown': overall.get('max_drawdown', 0)
        }

    def _generate_detailed_analysis(self, statistics: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация детального анализа"""
        analysis = {}

        # Анализ по типам паттернов
        by_type = statistics.get('by_type', {})
        best_type = None
        best_accuracy = 0

        for pattern_type, stats in by_type.items():
            accuracy = stats.get('positive_return_rate', 0)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_type = pattern_type

        if best_type:
            analysis['best_performing_type'] = {
                'type': best_type,
                'accuracy': best_accuracy,
                'details': by_type[best_type]
            }

        # Анализ статистической значимости
        significance = statistics.get('significance_tests', {})
        mean_test = significance.get('mean_test', {})
        if mean_test.get('mean_significantly_different', False):
            analysis['statistical_significance'] = {
                'message': 'Доходность паттернов статистически значимо отличается от нуля',
                'p_value': mean_test.get('p_value', 1),
                'confidence_level': 1 - mean_test.get('p_value', 1)
            }
        else:
            analysis['statistical_significance'] = {
                'message': 'Доходность паттернов не статистически значима',
                'p_value': mean_test.get('p_value', 1)
            }

        # Анализ надежности
        confidence_impact = statistics.get('performance_metrics', {}).get('accuracy_very_high_confidence', 0)
        if confidence_impact > 0.7:
            analysis['reliability'] = 'Высокая (точность растет с уверенностью)'
        elif confidence_impact > 0.6:
            analysis['reliability'] = 'Средняя'
        else:
            analysis['reliability'] = 'Низкая'

        return analysis

    def _generate_recommendations(self, statistics: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций"""
        recommendations = []

        overall = statistics.get('overall', {})
        performance = statistics.get('performance_metrics', {})

        # Рекомендации на основе точности
        accuracy = performance.get('accuracy', 0)
        if accuracy > 0.6:
            recommendations.append("Паттерны показывают хорошую точность - можно использовать в торговле")
        elif accuracy > 0.55:
            recommendations.append("Паттерны показывают умеренную точность - использовать с осторожностью")
        else:
            recommendations.append("Паттерны не показывают статистически значимой точности - требуется дополнительная проверка")

        # Рекомендации на основе доходности
        avg_return = overall.get('avg_forward_return', 0)
        if avg_return > 0.02:  # 2%
            recommendations.append("Высокая средняя доходность - потенциал для прибыльной торговли")
        elif avg_return > 0:
            recommendations.append("Положительная доходность - потенциал есть")
        else:
            recommendations.append("Отрицательная доходность - требуется оптимизация стратегии")

        # Рекомендации на основе статзначимости
        significance = statistics.get('significance_tests', {}).get('mean_test', {})
        if not significance.get('mean_significantly_different', False):
            recommendations.append("Результаты не статистически значимы - требуется больше данных или оптимизация")

        # Рекомендации по управлению рисками
        max_dd = overall.get('max_drawdown', 0)
        if max_dd > 0.2:  # 20%
            recommendations.append("Высокая максимальная просадка - требуется улучшение управления рисками")
        elif max_dd > 0.1:
            recommendations.append("Умеренная просадка - стандартное управление рисками")

        return recommendations

    def _format_text_report(self, report: Dict[str, Any]) -> str:
        """Форматирование текстового отчета"""
        lines = []

        # Заголовок
        lines.append("=" * 80)
        lines.append("СТАТИСТИЧЕСКИЙ ОТЧЕТ ПО АНАЛИЗУ ПАТТЕРНОВ")
        lines.append("=" * 80)

        # Сводка
        summary = report.get('summary', {})
        lines.append("\nСВОДКА:")
        lines.append("-" * 40)
        lines.append(f"Всего проанализировано паттернов: {summary.get('total_patterns_analyzed', 0)}")
        lines.append(f"Средняя доходность: {summary.get('average_forward_return', 0):.2%}")
        lines.append(f"Процент успешных паттернов: {summary.get('positive_return_rate', 0):.2%}")
        lines.append(f"Общая точность: {summary.get('overall_accuracy', 0):.2%}")
        lines.append(f"Коэффициент Шарпа: {summary.get('sharpe_ratio', 0):.2f}")
        lines.append(f"Профит-фактор: {summary.get('profit_factor', 0):.2f}")
        lines.append(f"Максимальная просадка: {summary.get('max_drawdown', 0):.2%}")

        # Детальный анализ
        analysis = report.get('detailed_analysis', {})
        lines.append("\nДЕТАЛЬНЫЙ АНАЛИЗ:")
        lines.append("-" * 40)

        best_type = analysis.get('best_performing_type', {})
        if best_type:
            lines.append(f"Лучший тип паттернов: {best_type.get('type', 'N/A')}")
            lines.append(f"Точность лучшего типа: {best_type.get('accuracy', 0):.2%}")

        significance = analysis.get('statistical_significance', {})
        lines.append(f"\nСтатистическая значимость: {significance.get('message', 'N/A')}")
        lines.append(f"Уровень надежности: {analysis.get('reliability', 'N/A')}")

        # Рекомендации
        recommendations = report.get('recommendations', [])
        lines.append("\nРЕКОМЕНДАЦИИ:")
        lines.append("-" * 40)
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}")

        # Временная метка
        lines.append(f"\nОтчет сгенерирован: {report.get('timestamp', 'N/A')}")
        lines.append("=" * 80)

        return "\n".join(lines)

