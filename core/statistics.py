"""
Модуль для статистических расчетов
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
import warnings

# Исправляем импорт для обратной совместимости
try:
    from config import config, ML_CONFIG, STATISTICS_CONFIG
except ImportError:
    # Для обратной совместимости
    try:
        from config import ML_CONFIG as STATISTICS_CONFIG
    except ImportError:
        # Создаем fallback конфиг
        STATISTICS_CONFIG = type('Config', (), {
            'FEATURE_WINDOW': 20,
            'USE_TECHNICAL_INDICATORS': True
        })()


def calculate_statistics(returns: pd.Series) -> Dict[str, float]:
    """
    Расчет базовых статистических показателей

    Args:
        returns: Серия с доходностями

    Returns:
        Словарь со статистическими показателями
    """
    if len(returns) == 0:
        return {}

    stats_dict = {}

    # Базовая статистика
    stats_dict['mean_return'] = float(returns.mean())
    stats_dict['std_return'] = float(returns.std())
    stats_dict['variance'] = float(returns.var())

    # Коэффициенты
    if stats_dict['std_return'] != 0:
        stats_dict['sharpe_ratio'] = float(stats_dict['mean_return'] / stats_dict['std_return'] * np.sqrt(252))
        stats_dict['sortino_ratio'] = float(calculate_sortino_ratio(returns))

    # Максимальная просадка
    stats_dict['max_drawdown'] = float(calculate_max_drawdown(returns))

    # Критерии оценки
    stats_dict['skewness'] = float(returns.skew())
    stats_dict['kurtosis'] = float(returns.kurtosis())

    # Процент положительных сделок
    positive_trades = (returns > 0).sum()
    stats_dict['win_rate'] = float(positive_trades / len(returns) if len(returns) > 0 else 0)

    # Profit Factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    stats_dict['profit_factor'] = float(gross_profit / gross_loss if gross_loss > 0 else 0)

    return stats_dict


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Расчет максимальной просадки

    Args:
        returns: Серия с доходностями

    Returns:
        Максимальная просадка в процентах
    """
    if len(returns) == 0:
        return 0.0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max

    return float(abs(drawdown.min()))


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Расчет коэффициента Сортино

    Args:
        returns: Серия с доходностями
        risk_free_rate: Безрисковая ставка (годовая)

    Returns:
        Коэффициент Сортино
    """
    if len(returns) == 0:
        return 0.0

    # Годовая доходность
    annual_return = (1 + returns.mean()) ** 252 - 1

    # Нижнее полуотклонение
    negative_returns = returns[returns < 0]
    if len(negative_returns) == 0:
        return 0.0

    downside_deviation = negative_returns.std() * np.sqrt(252)

    if downside_deviation == 0:
        return 0.0

    return float((annual_return - risk_free_rate) / downside_deviation)


def calculate_calmar_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Расчет коэффициента Калмара

    Args:
        returns: Серия с доходностями
        risk_free_rate: Безрисковая ставка (годовая)

    Returns:
        Коэффициент Калмара
    """
    if len(returns) == 0:
        return 0.0

    max_dd = calculate_max_drawdown(returns)
    if max_dd == 0:
        return 0.0

    annual_return = (1 + returns.mean()) ** 252 - 1

    return float((annual_return - risk_free_rate) / max_dd)


class StatisticsCalculator:
    """Класс для расширенного статистического анализа"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.statistics_config = STATISTICS_CONFIG

    def calculate_trade_statistics(self, trades: List[Dict]) -> Dict[str, float]:
        """
        Расчет статистики по сделкам

        Args:
            trades: Список сделок

        Returns:
            Статистика по сделкам
        """
        if not trades:
            return {}

        # Извлекаем результаты сделок
        results = [trade.get('result', 0) for trade in trades if 'result' in trade]

        if not results:
            return {}

        returns = pd.Series(results)

        # Базовая статистика
        stats = calculate_statistics(returns)

        # Дополнительная статистика по сделкам
        stats['total_trades'] = len(trades)
        stats['profitable_trades'] = sum(1 for r in results if r > 0)
        stats['losing_trades'] = sum(1 for r in results if r < 0)

        # Средняя прибыльная/убыточная сделка
        profitable_results = [r for r in results if r > 0]
        losing_results = [abs(r) for r in results if r < 0]

        if profitable_results:
            stats['avg_profitable_trade'] = float(np.mean(profitable_results))
            stats['max_profitable_trade'] = float(np.max(profitable_results))

        if losing_results:
            stats['avg_losing_trade'] = float(np.mean(losing_results))
            stats['max_losing_trade'] = float(np.max(losing_results))

        # Серии прибылей/убытков
        stats['max_consecutive_wins'] = self._calculate_max_consecutive_wins(results)
        stats['max_consecutive_losses'] = self._calculate_max_consecutive_losses(results)

        # Recovery Factor
        stats['recovery_factor'] = self._calculate_recovery_factor(results)

        return stats

    def _calculate_max_consecutive_wins(self, results: List[float]) -> int:
        """Расчет максимальной серии прибыльных сделок"""
        max_wins = 0
        current_wins = 0

        for result in results:
            if result > 0:
                current_wins += 1
                max_wins = max(max_wins, current_wins)
            else:
                current_wins = 0

        return max_wins

    def _calculate_max_consecutive_losses(self, results: List[float]) -> int:
        """Расчет максимальной серии убыточных сделок"""
        max_losses = 0
        current_losses = 0

        for result in results:
            if result < 0:
                current_losses += 1
                max_losses = max(max_losses, current_losses)
            else:
                current_losses = 0

        return max_losses

    def _calculate_recovery_factor(self, results: List[float]) -> float:
        """
        Расчет фактора восстановления

        Recovery Factor = Net Profit / Maximum Drawdown
        """
        if not results:
            return 0.0

        cumulative = np.cumsum(results)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative

        if len(drawdowns) == 0 or max(drawdowns) == 0:
            return 0.0

        net_profit = cumulative[-1] if len(cumulative) > 0 else 0

        return float(net_profit / max(drawdowns))

    def calculate_monte_carlo_simulation(self, returns: pd.Series,
                                         simulations: int = 1000,
                                         periods: int = 252) -> Dict[str, float]:
        """
        Монте-Карло симуляция доходности

        Args:
            returns: Исторические доходности
            simulations: Количество симуляций
            periods: Количество периодов в симуляции

        Returns:
            Результаты симуляции
        """
        if len(returns) < 2:
            return {}

        mean_return = returns.mean()
        std_return = returns.std()

        simulated_results = []

        for _ in range(simulations):
            # Генерируем случайные доходности
            simulated_returns = np.random.normal(mean_return, std_return, periods)

            # Рассчитываем конечное значение
            final_value = np.prod(1 + simulated_returns)
            simulated_results.append(final_value)

        # Статистика по симуляциям
        results_array = np.array(simulated_results)

        return {
            'mean_simulated_return': float(np.mean(results_array)),
            'median_simulated_return': float(np.median(results_array)),
            'std_simulated_return': float(np.std(results_array)),
            'var_95': float(np.percentile(results_array, 5)),
            'var_99': float(np.percentile(results_array, 1)),
            'cvar_95': float(results_array[results_array <= np.percentile(results_array, 5)].mean()),
            'cvar_99': float(results_array[results_array <= np.percentile(results_array, 1)].mean())
        }

    def calculate_correlation_matrix(self, assets_returns: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Расчет матрицы корреляций

        Args:
            assets_returns: Словарь с доходностями активов

        Returns:
            DataFrame с матрицей корреляций
        """
        if not assets_returns:
            return pd.DataFrame()

        # Объединяем все серии в DataFrame
        returns_df = pd.DataFrame(assets_returns)

        # Удаляем NaN значения
        returns_df = returns_df.dropna()

        if len(returns_df) < 2:
            return pd.DataFrame()

        # Рассчитываем корреляционную матрицу
        correlation_matrix = returns_df.corr()

        return correlation_matrix

    def calculate_rolling_statistics(self, returns: pd.Series, window: int = 20) -> pd.DataFrame:
        """
        Расчет скользящей статистики

        Args:
            returns: Серия с доходностями
            window: Размер окна

        Returns:
            DataFrame со скользящей статистикой
        """
        if len(returns) < window:
            return pd.DataFrame()

        rolling_stats = pd.DataFrame(index=returns.index)

        # Скользящая средняя
        rolling_stats['rolling_mean'] = returns.rolling(window=window).mean()

        # Скользящее стандартное отклонение
        rolling_stats['rolling_std'] = returns.rolling(window=window).std()

        # Скользящий коэффициент Шарпа
        rolling_stats['rolling_sharpe'] = (rolling_stats['rolling_mean'] /
                                           rolling_stats['rolling_std'] * np.sqrt(252))

        # Скользящая максимальная просадка
        rolling_stats['rolling_max_drawdown'] = returns.rolling(window=window).apply(
            lambda x: calculate_max_drawdown(pd.Series(x))
        )

        return rolling_stats.dropna()

