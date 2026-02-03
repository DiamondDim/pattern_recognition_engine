"""
Модуль статистического анализа
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import scipy.stats as stats

from config import config
from utils.logger import logger


@dataclass
class StatisticalReport:
    """Статистический отчет"""

    basic_stats: Dict[str, Any] = field(default_factory=dict)
    distribution_stats: Dict[str, Any] = field(default_factory=dict)
    correlation_stats: Dict[str, Any] = field(default_factory=dict)
    time_series_stats: Dict[str, Any] = field(default_factory=dict)
    pattern_stats: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class StatisticsModule:
    """Модуль статистического анализа"""

    def __init__(self):
        self.logger = logger.bind(module="statistics")

    async def analyze_data(self, data: Dict[str, np.ndarray]) -> StatisticalReport:
        """
        Статистический анализ данных

        Args:
            data: Входные данные

        Returns:
            Статистический отчет
        """
        self.logger.info("Запуск статистического анализа")

        try:
            # Базовые статистики
            basic_stats = await self._calculate_basic_statistics(data)

            # Анализ распределения
            distribution_stats = await self._analyze_distribution(data)

            # Корреляционный анализ
            correlation_stats = await self._calculate_correlations(data)

            # Анализ временных рядов
            time_series_stats = await self._analyze_time_series(data)

            # Статистика паттернов (если есть)
            pattern_stats = await self._analyze_patterns(data)

            return StatisticalReport(
                basic_stats=basic_stats,
                distribution_stats=distribution_stats,
                correlation_stats=correlation_stats,
                time_series_stats=time_series_stats,
                pattern_stats=pattern_stats
            )

        except Exception as e:
            self.logger.error(f"Ошибка статистического анализа: {e}")
            return StatisticalReport()

    async def _calculate_basic_statistics(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Расчет базовых статистик"""
        stats_dict = {}

        for key, values in data.items():
            if isinstance(values, np.ndarray) and values.dtype in [np.float64, np.float32, np.int64, np.int32]:
                # Пропускаем временные метки и индексы
                if key in ['time', 'timestamp', 'index']:
                    continue

                # Базовые статистики
                stats_dict[key] = {
                    'count': int(len(values)),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'q25': float(np.percentile(values, 25)),
                    'q75': float(np.percentile(values, 75)),
                    'iqr': float(np.percentile(values, 75) - np.percentile(values, 25)),
                    'skewness': float(stats.skew(values)),
                    'kurtosis': float(stats.kurtosis(values))
                }

        return stats_dict

    async def _analyze_distribution(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Анализ распределения данных"""
        distribution = {}

        # Анализируем только ценовые данные
        price_keys = ['open', 'high', 'low', 'close']

        for key in price_keys:
            if key in data:
                values = data[key]

                # Проверка нормальности распределения
                if len(values) >= 8:  # Минимальный размер для теста Шапиро-Уилка
                    shapiro_stat, shapiro_p = stats.shapiro(values[:5000])  # Ограничиваем для производительности
                    norm_test = {
                        'shapiro_wilk': {
                            'statistic': float(shapiro_stat),
                            'p_value': float(shapiro_p),
                            'is_normal': shapiro_p > 0.05
                        }
                    }
                else:
                    norm_test = {'error': 'insufficient_data'}

                # Гистограмма (упрощенная)
                hist, bin_edges = np.histogram(values, bins='auto')

                distribution[key] = {
                    'normality_test': norm_test,
                    'histogram': {
                        'counts': hist.tolist(),
                        'bin_edges': bin_edges.tolist()
                    },
                    'density_estimate': {
                        'bandwidth': self._calculate_bandwidth(values),
                        'modes': self._find_modes(values)
                    }
                }

        return distribution

    async def _calculate_correlations(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Расчет корреляций"""
        correlations = {}

        # Ценовые данные для корреляции
        price_keys = ['open', 'high', 'low', 'close']
        price_data = {}

        for key in price_keys:
            if key in data:
                price_data[key] = data[key]

        if len(price_data) < 2:
            return {'error': 'insufficient_data'}

        # Создаем DataFrame для удобства
        df = pd.DataFrame(price_data)

        # Матрица корреляций
        corr_matrix = df.corr().to_dict()

        # Проверка значимости корреляций
        significant_correlations = []
        for i, col1 in enumerate(df.columns):
            for j, col2 in enumerate(df.columns):
                if i < j:  # Верхний треугольник
                    corr_value = corr_matrix[col1][col2]
                    n = len(df)

                    # t-тест для корреляции
                    if abs(corr_value) < 1:  # Избегаем деления на ноль
                        t_stat = corr_value * np.sqrt((n - 2) / (1 - corr_value ** 2))
                        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

                        if p_value < 0.05:  # Значимая корреляция
                            significant_correlations.append({
                                'pair': f"{col1}-{col2}",
                                'correlation': float(corr_value),
                                'p_value': float(p_value),
                                'is_significant': True
                            })

        correlations['matrix'] = corr_matrix
        correlations['significant_pairs'] = significant_correlations
        correlations['summary'] = {
            'total_pairs': len(significant_correlations),
            'avg_correlation': float(df.corr().abs().mean().mean()),
            'max_correlation': float(df.corr().abs().max().max())
        }

        return correlations

    async def _analyze_time_series(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Анализ временных рядов"""
        time_series = {}

        if 'close' not in data:
            return {'error': 'no_close_data'}

        closes = data['close']

        if len(closes) < 10:
            return {'error': 'insufficient_data'}

        # Стационарность (тест Дики-Фуллера)
        from statsmodels.tsa.stattools import adfuller

        try:
            adf_result = adfuller(closes)
            stationarity = {
                'adf_statistic': float(adf_result[0]),
                'p_value': float(adf_result[1]),
                'critical_values': {k: float(v) for k, v in adf_result[4].items()},
                'is_stationary': adf_result[1] < 0.05
            }
        except Exception as e:
            stationarity = {'error': str(e)}

        # Автокорреляция
        try:
            from statsmodels.tsa.stattools import acf
            max_lag = min(40, len(closes) // 4)
            autocorr = acf(closes, nlags=max_lag, fft=True)

            # Находим значимые автокорреляции
            significant_lags = []
            for lag, value in enumerate(autocorr[1:], start=1):  # Пропускаем lag 0
                if abs(value) > 1.96 / np.sqrt(len(closes)):  # 95% доверительный интервал
                    significant_lags.append({
                        'lag': lag,
                        'autocorrelation': float(value),
                        'is_significant': True
                    })
        except Exception as e:
            autocorr = {'error': str(e)}
            significant_lags = []

        # Тренд
        try:
            x = np.arange(len(closes))
            slope, intercept = np.polyfit(x, closes, 1)
            trend_line = slope * x + intercept

            # Качество тренда (R²)
            residuals = closes - trend_line
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((closes - np.mean(closes)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        except Exception as e:
            slope = intercept = r_squared = 0
            trend_line = closes

        time_series = {
            'stationarity': stationarity,
            'autocorrelation': {
                'values': autocorr.tolist() if isinstance(autocorr, np.ndarray) else autocorr,
                'significant_lags': significant_lags,
                'max_lag': max_lag
            },
            'trend_analysis': {
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(r_squared),
                'trend_strength': 'strong' if abs(slope) > 0.001 else 'weak' if abs(slope) > 0.0001 else 'none',
                'direction': 'up' if slope > 0 else 'down' if slope < 0 else 'flat'
            },
            'volatility': {
                'daily': float(np.std(np.diff(closes) / closes[:-1]) if len(closes) > 1 else 0),
                'annualized': float(np.std(np.diff(closes) / closes[:-1]) * np.sqrt(252) if len(closes) > 1 else 0),
                'avg_true_range': self._calculate_atr(data) if all(k in data for k in ['high', 'low', 'close']) else 0
            }
        }

        return time_series

    async def _analyze_patterns(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Анализ паттернов в данных"""
        pattern_stats = {}

        # Простая детекция паттернов (здесь можно расширить)
        try:
            # Обнаружение экстремумов
            extremes = self._find_local_extremes(data['close'])

            # Анализ расстояний между экстремумами
            if len(extremes) >= 2:
                distances = []
                directions = []

                for i in range(1, len(extremes)):
                    dist = extremes[i]['index'] - extremes[i - 1]['index']
                    price_change = extremes[i]['price'] - extremes[i - 1]['price']

                    distances.append(dist)
                    directions.append('up' if price_change > 0 else 'down')

                pattern_stats['extremes'] = {
                    'count': len(extremes),
                    'avg_distance': float(np.mean(distances)) if distances else 0,
                    'std_distance': float(np.std(distances)) if len(distances) > 1 else 0,
                    'direction_changes': sum(
                        1 for i in range(1, len(directions)) if directions[i] != directions[i - 1]),
                    'avg_price_change': float(np.mean([abs(extremes[i]['price'] - extremes[i - 1]['price'])
                                                       for i in range(1, len(extremes))])) if len(extremes) > 1 else 0
                }
        except Exception as e:
            pattern_stats['extremes'] = {'error': str(e)}

        # Анализ кластеризации цен
        try:
            if 'close' in data:
                closes = data['close']
                # Используем простую кластеризацию на основе процентилей
                percentiles = np.percentile(closes, [25, 50, 75])

                pattern_stats['clustering'] = {
                    'price_levels': {
                        'support_1': float(percentiles[0]),
                        'median': float(percentiles[1]),
                        'resistance_1': float(percentiles[2])
                    },
                    'distribution': {
                        'below_support': float(np.sum(closes < percentiles[0]) / len(closes)),
                        'between_support_resistance': float(np.sum((closes >= percentiles[0]) &
                                                                   (closes <= percentiles[2])) / len(closes)),
                        'above_resistance': float(np.sum(closes > percentiles[2]) / len(closes))
                    }
                }
        except Exception as e:
            pattern_stats['clustering'] = {'error': str(e)}

        return pattern_stats

    def _find_local_extremes(self, prices: np.ndarray, window: int = 5) -> List[Dict[str, Any]]:
        """Поиск локальных экстремумов"""
        extremes = []

        for i in range(window, len(prices) - window):
            local_window = prices[i - window:i + window + 1]

            if prices[i] == np.max(local_window):
                extremes.append({
                    'index': i,
                    'price': float(prices[i]),
                    'type': 'high'
                })
            elif prices[i] == np.min(local_window):
                extremes.append({
                    'index': i,
                    'price': float(prices[i]),
                    'type': 'low'
                })

        return extremes

    def _calculate_atr(self, data: Dict[str, np.ndarray], period: int = 14) -> float:
        """Расчет Average True Range"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']

            if len(high) < period:
                return 0.0

            # Расчет True Range
            tr = np.zeros(len(high))
            for i in range(1, len(high)):
                hl = high[i] - low[i]
                hc = abs(high[i] - close[i - 1])
                lc = abs(low[i] - close[i - 1])
                tr[i] = max(hl, hc, lc)

            # Скользящее среднее TR
            atr = np.mean(tr[-period:])
            return float(atr)

        except Exception:
            return 0.0

    def _calculate_bandwidth(self, values: np.ndarray) -> float:
        """Расчет оптимальной ширины окна для оценки плотности"""
        # Правило Сильвермана
        iqr = np.percentile(values, 75) - np.percentile(values, 25)
        sigma = min(np.std(values), iqr / 1.34)
        n = len(values)

        if sigma == 0:
            return 1.06 * np.std(values) * n ** (-0.2) if np.std(values) > 0 else 1.0
        else:
            return 1.06 * sigma * n ** (-0.2)

    def _find_modes(self, values: np.ndarray, bins: int = 50) -> List[float]:
        """Поиск мод в распределении"""
        try:
            hist, bin_edges = np.histogram(values, bins=bins)

            # Находим локальные максимумы гистограммы
            modes = []
            for i in range(1, len(hist) - 1):
                if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
                    # Интерполируем положение моды
                    mode_pos = bin_edges[i] + (bin_edges[i + 1] - bin_edges[i]) / 2
                    modes.append(float(mode_pos))

            return modes
        except Exception:
            return []

    async def calculate_performance_metrics(self,
                                            trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Расчет метрик производительности на основе сделок"""
        if not trades:
            return {
                'total_trades': 0,
                'error': 'no_trades'
            }

        try:
            # Извлекаем P&L
            pnls = [trade.get('pnl', 0) for trade in trades]
            pnls_pips = [trade.get('pnl_pips', 0) for trade in trades]

            # Базовые метрики
            winning_trades = [p for p in pnls if p > 0]
            losing_trades = [p for p in pnls if p <= 0]

            total_trades = len(trades)
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

            # Прибыль/убыток
            total_profit = sum(winning_trades)
            total_loss = abs(sum(losing_trades))
            net_profit = total_profit - total_loss

            # Коэффициент прибыли
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

            # Средние значения
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = np.mean(losing_trades) if losing_trades else 0

            # Соотношение средних
            avg_win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

            # Максимальные значения
            largest_win = max(winning_trades) if winning_trades else 0
            largest_loss = min(losing_trades) if losing_trades else 0

            # Просадка (упрощенная)
            equity_curve = [0]
            for pnl in pnls:
                equity_curve.append(equity_curve[-1] + pnl)

            max_drawdown = self._calculate_max_drawdown(equity_curve)

            # Коэффициент Шарпа (упрощенный)
            sharpe_ratio = self._calculate_sharpe_ratio(pnls)

            # Коэффициент Сортино
            sortino_ratio = self._calculate_sortino_ratio(pnls)

            # Ожидаемая прибыль
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))

            return {
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': float(win_rate),
                'total_profit': float(total_profit),
                'total_loss': float(total_loss),
                'net_profit': float(net_profit),
                'profit_factor': float(profit_factor) if profit_factor != float('inf') else float('inf'),
                'avg_win': float(avg_win),
                'avg_loss': float(avg_loss),
                'avg_win_loss_ratio': float(avg_win_loss_ratio) if avg_win_loss_ratio != float('inf') else float('inf'),
                'largest_win': float(largest_win),
                'largest_loss': float(largest_loss),
                'max_drawdown': float(max_drawdown),
                'sharpe_ratio': float(sharpe_ratio),
                'sortino_ratio': float(sortino_ratio),
                'expectancy': float(expectancy),
                'avg_trade': float(np.mean(pnls)) if pnls else 0,
                'std_trade': float(np.std(pnls)) if len(pnls) > 1 else 0,
                'avg_pips_per_trade': float(np.mean(pnls_pips)) if pnls_pips else 0,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Ошибка расчета метрик производительности: {e}")
            return {'error': str(e)}

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Расчет максимальной просадки"""
        peak = equity_curve[0]
        max_dd = 0.0

        for value in equity_curve:
            if value > peak:
                peak = value

            dd = (peak - value) / peak if peak != 0 else 0
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Расчет коэффициента Шарпа"""
        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate / 252

        if np.std(excess_returns) == 0:
            return 0.0

        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return float(sharpe)

    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Расчет коэффициента Сортино"""
        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate / 252

        # Только отрицательные возвраты
        negative_returns = excess_returns[excess_returns < 0]

        if len(negative_returns) == 0 or np.std(negative_returns) == 0:
            return 0.0

        sortino = np.mean(excess_returns) / np.std(negative_returns) * np.sqrt(252)
        return float(sortino)