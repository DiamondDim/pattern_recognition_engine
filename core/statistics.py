import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class StatisticsCalculator:
    """
    Класс для расчета статистики торговых результатов
    """

    def __init__(self):
        """Инициализация калькулятора статистики"""
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate_all_statistics(self, trades: List[Dict[str, Any]],
                                 equity_curve: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Расчет всей статистики

        Args:
            trades (list): Список сделок
            equity_curve (list): Кривая капитала

        Returns:
            dict: Полная статистика
        """
        statistics = {
            'basic_metrics': {},
            'risk_metrics': {},
            'performance_metrics': {},
            'time_metrics': {},
            'distribution_metrics': {},
            'calculated_at': datetime.now()
        }

        if not trades:
            statistics['error'] = 'No trades provided'
            return statistics

        # Базовые метрики
        statistics['basic_metrics'] = self.calculate_basic_metrics(trades)

        # Метрики риска
        statistics['risk_metrics'] = self.calculate_risk_metrics(equity_curve)

        # Метрики производительности
        statistics['performance_metrics'] = self.calculate_performance_metrics(trades, equity_curve)

        # Временные метрики
        statistics['time_metrics'] = self.calculate_time_metrics(trades)

        # Метрики распределения
        statistics['distribution_metrics'] = self.calculate_distribution_metrics(trades)

        # Сводная статистика
        statistics['summary'] = self._create_summary(statistics)

        return statistics

    def calculate_basic_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Расчет базовых метрик

        Args:
            trades (list): Список сделок

        Returns:
            dict: Базовые метрики
        """
        if not trades:
            return {}

        try:
            # Общие показатели
            total_trades = len(trades)
            total_pnl = sum(t.get('pnl', 0) for t in trades)
            total_commission = sum(t.get('commission', 0) for t in trades)
            net_pnl = total_pnl - total_commission

            # Прибыльные и убыточные сделки
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in trades if t.get('pnl', 0) <= 0]

            win_count = len(winning_trades)
            loss_count = len(losing_trades)

            # Винрейт
            win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

            # Средние показатели
            avg_win = np.mean([t.get('pnl', 0) for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.get('pnl', 0) for t in losing_trades]) if losing_trades else 0
            avg_win_percent = np.mean([t.get('pnl_percent', 0) for t in winning_trades]) if winning_trades else 0
            avg_loss_percent = np.mean([t.get('pnl_percent', 0) for t in losing_trades]) if losing_trades else 0

            # Фактор прибыли
            total_win = sum(t.get('pnl', 0) for t in winning_trades)
            total_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))

            profit_factor = total_win / total_loss if total_loss > 0 else float('inf')

            # Максимальные прибыль/убыток
            max_win = max([t.get('pnl', 0) for t in winning_trades]) if winning_trades else 0
            max_loss = min([t.get('pnl', 0) for t in losing_trades]) if losing_trades else 0

            # Средняя сделка
            avg_trade = net_pnl / total_trades if total_trades > 0 else 0

            metrics = {
                'total_trades': total_trades,
                'winning_trades': win_count,
                'losing_trades': loss_count,
                'win_rate_percent': win_rate,
                'total_pnl': total_pnl,
                'net_pnl': net_pnl,
                'total_commission': total_commission,
                'avg_win': float(avg_win),
                'avg_loss': float(avg_loss),
                'avg_win_percent': float(avg_win_percent),
                'avg_loss_percent': float(avg_loss_percent),
                'profit_factor': float(profit_factor),
                'max_win': float(max_win),
                'max_loss': float(max_loss),
                'avg_trade': float(avg_trade),
                'expectancy': self._calculate_expectancy(win_rate, avg_win, avg_loss)
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Ошибка расчета базовых метрик: {e}")
            return {}

    def _calculate_expectancy(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Расчет математического ожидания

        Args:
            win_rate (float): Винрейт в процентах
            avg_win (float): Средняя прибыль
            avg_loss (float): Средний убыток

        Returns:
            float: Математическое ожидание
        """
        win_probability = win_rate / 100
        loss_probability = 1 - win_probability

        expectancy = (win_probability * avg_win) - (loss_probability * abs(avg_loss))
        return float(expectancy)

    def calculate_risk_metrics(self, equity_curve: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Расчет метрик риска

        Args:
            equity_curve (list): Кривая капитала

        Returns:
            dict: Метрики риска
        """
        if not equity_curve or len(equity_curve) < 2:
            return {}

        try:
            # Извлекаем капитал из кривой
            equities = [e.get('equity', 0) for e in equity_curve]
            equity_series = pd.Series(equities)

            # Максимальная просадка
            cumulative_max = equity_series.expanding().max()
            drawdowns = cumulative_max - equity_series
            drawdowns_percent = (drawdowns / cumulative_max) * 100

            max_drawdown = float(drawdowns.max())
            max_drawdown_percent = float(drawdowns_percent.max())
            avg_drawdown = float(drawdowns.mean())
            avg_drawdown_percent = float(drawdowns_percent.mean())

            # Волатильность
            returns = equity_series.pct_change().dropna()

            if len(returns) > 1:
                # Годовая волатильность (предполагаем 252 торговых дня)
                volatility = returns.std() * np.sqrt(252) * 100
                daily_volatility = returns.std() * 100
            else:
                volatility = daily_volatility = 0

            # Value at Risk (VaR)
            var_95 = np.percentile(returns, 5) * 100 if len(returns) > 0 else 0
            var_99 = np.percentile(returns, 1) * 100 if len(returns) > 0 else 0

            # Conditional VaR (CVaR)
            if len(returns) > 0:
                cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
                cvar_99 = returns[returns <= np.percentile(returns, 1)].mean() * 100
            else:
                cvar_95 = cvar_99 = 0

            # Коэффициент восстановления
            if max_drawdown > 0:
                final_equity = equity_series.iloc[-1]
                initial_equity = equity_series.iloc[0]
                recovery_factor = (final_equity - initial_equity) / max_drawdown
            else:
                recovery_factor = float('inf')

            # Статистика по просадкам
            drawdown_stats = self._analyze_drawdowns(drawdowns_percent)

            metrics = {
                'max_drawdown': max_drawdown,
                'max_drawdown_percent': max_drawdown_percent,
                'avg_drawdown': avg_drawdown,
                'avg_drawdown_percent': avg_drawdown_percent,
                'annual_volatility_percent': float(volatility),
                'daily_volatility_percent': float(daily_volatility),
                'var_95_percent': float(var_95),
                'var_99_percent': float(var_99),
                'cvar_95_percent': float(cvar_95),
                'cvar_99_percent': float(cvar_99),
                'recovery_factor': float(recovery_factor),
                'drawdown_stats': drawdown_stats
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Ошибка расчета метрик риска: {e}")
            return {}

    def _analyze_drawdowns(self, drawdowns_percent: pd.Series) -> Dict[str, Any]:
        """
        Анализ просадок

        Args:
            drawdowns_percent (pd.Series): Просадки в процентах

        Returns:
            dict: Статистика просадок
        """
        try:
            # Находим периоды просадок
            in_drawdown = drawdowns_percent > 0

            if not in_drawdown.any():
                return {
                    'drawdown_count': 0,
                    'avg_drawdown_duration': 0,
                    'max_drawdown_duration': 0
                }

            # Подсчитываем количество просадок
            drawdown_count = 0
            drawdown_durations = []
            current_duration = 0

            for i in range(1, len(in_drawdown)):
                if in_drawdown.iloc[i]:
                    current_duration += 1
                    if not in_drawdown.iloc[i - 1]:
                        drawdown_count += 1
                elif in_drawdown.iloc[i - 1] and current_duration > 0:
                    drawdown_durations.append(current_duration)
                    current_duration = 0

            # Добавляем последнюю просадку
            if current_duration > 0:
                drawdown_durations.append(current_duration)

            # Статистика по длительности
            if drawdown_durations:
                avg_duration = np.mean(drawdown_durations)
                max_duration = np.max(drawdown_durations)
            else:
                avg_duration = max_duration = 0

            return {
                'drawdown_count': int(drawdown_count),
                'avg_drawdown_duration': float(avg_duration),
                'max_drawdown_duration': int(max_duration)
            }

        except Exception as e:
            self.logger.error(f"Ошибка анализа просадок: {e}")
            return {'error': str(e)}

    def calculate_performance_metrics(self, trades: List[Dict[str, Any]],
                                      equity_curve: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Расчет метрик производительности

        Args:
            trades (list): Список сделок
            equity_curve (list): Кривая капитала

        Returns:
            dict: Метрики производительности
        """
        if not equity_curve or len(equity_curve) < 2:
            return {}

        try:
            # Извлекаем капитал
            equities = [e.get('equity', 0) for e in equity_curve]
            initial_equity = equities[0]
            final_equity = equities[-1]

            # Общая доходность
            total_return = (final_equity - initial_equity) / initial_equity * 100

            # Годовая доходность
            if len(equity_curve) >= 2:
                # Оцениваем длительность в годах
                first_date = equity_curve[0].get('timestamp')
                last_date = equity_curve[-1].get('timestamp')

                if isinstance(first_date, datetime) and isinstance(last_date, datetime):
                    days = (last_date - first_date).days
                    years = days / 365.25 if days > 0 else 1
                else:
                    years = len(equity_curve) / 252  # Предполагаем торговые дни

                if years > 0:
                    annual_return = ((final_equity / initial_equity) ** (1 / years) - 1) * 100
                else:
                    annual_return = total_return
            else:
                annual_return = total_return

            # Коэффициент Шарпа
            equity_series = pd.Series(equities)
            returns = equity_series.pct_change().dropna()

            if len(returns) > 1 and returns.std() > 0:
                # Безрисковая ставка (предполагаем 0 для простоты)
                risk_free_rate = 0
                excess_returns = returns - risk_free_rate / 252

                sharpe_ratio = np.mean(excess_returns) / returns.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0

            # Коэффициент Сортино
            if len(returns) > 1:
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0 and downside_returns.std() > 0:
                    sortino_ratio = np.mean(returns) / downside_returns.std() * np.sqrt(252)
                else:
                    sortino_ratio = sharpe_ratio
            else:
                sortino_ratio = 0

            # Коэффициент Кальмара
            risk_metrics = self.calculate_risk_metrics(equity_curve)
            max_drawdown_percent = risk_metrics.get('max_drawdown_percent', 0)

            if max_drawdown_percent > 0:
                calmar_ratio = annual_return / max_drawdown_percent
            else:
                calmar_ratio = float('inf')

            # Коэффициент восстановления
            recovery_factor = risk_metrics.get('recovery_factor', 0)

            # Коэффициент выживаемости
            if len(trades) > 0:
                survival_rate = self._calculate_survival_rate(trades)
            else:
                survival_rate = 100

            metrics = {
                'total_return_percent': float(total_return),
                'annual_return_percent': float(annual_return),
                'sharpe_ratio': float(sharpe_ratio),
                'sortino_ratio': float(sortino_ratio),
                'calmar_ratio': float(calmar_ratio),
                'recovery_factor': float(recovery_factor),
                'survival_rate_percent': float(survival_rate),
                'final_equity': float(final_equity),
                'initial_equity': float(initial_equity)
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Ошибка расчета метрик производительности: {e}")
            return {}

    def _calculate_survival_rate(self, trades: List[Dict[str, Any]]) -> float:
        """
        Расчет коэффициента выживаемости

        Args:
            trades (list): Список сделок

        Returns:
            float: Коэффициент выживаемости в процентах
        """
        if not trades:
            return 100.0

        try:
            # Подсчитываем серии убытков
            loss_series = []
            current_series = 0

            for trade in trades:
                if trade.get('pnl', 0) <= 0:
                    current_series += 1
                else:
                    if current_series > 0:
                        loss_series.append(current_series)
                        current_series = 0

            # Добавляем последнюю серию
            if current_series > 0:
                loss_series.append(current_series)

            # Рассчитываем вероятность выживания
            if loss_series:
                max_consecutive_losses = max(loss_series)
                # Простая эвристика: чем больше максимальная серия убытков,
                # тем ниже вероятность выживания
                survival_rate = max(0, 100 - (max_consecutive_losses * 5))
            else:
                survival_rate = 100.0

            return survival_rate

        except Exception as e:
            self.logger.error(f"Ошибка расчета коэффициента выживаемости: {e}")
            return 100.0

    def calculate_time_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Расчет временных метрик

        Args:
            trades (list): Список сделок

        Returns:
            dict: Временные метрики
        """
        if not trades:
            return {}

        try:
            # Время удержания позиций
            holding_periods = [t.get('holding_bars', 0) for t in trades]
            avg_holding = np.mean(holding_periods) if holding_periods else 0
            median_holding = np.median(holding_periods) if holding_periods else 0

            # Время между сделками
            if len(trades) > 1:
                # Сортируем сделки по времени выхода
                sorted_trades = sorted(trades, key=lambda x: x.get('exit_time', datetime.min))

                time_diffs = []
                for i in range(1, len(sorted_trades)):
                    time1 = sorted_trades[i - 1].get('exit_time')
                    time2 = sorted_trades[i].get('exit_time')

                    if isinstance(time1, datetime) and isinstance(time2, datetime):
                        diff = (time2 - time1).total_seconds() / 3600  # в часах
                        time_diffs.append(diff)

                if time_diffs:
                    avg_time_between = np.mean(time_diffs)
                    median_time_between = np.median(time_diffs)
                else:
                    avg_time_between = median_time_between = 0
            else:
                avg_time_between = median_time_between = 0

            # Распределение по времени суток
            hour_distribution = {}
            for trade in trades:
                exit_time = trade.get('exit_time')
                if isinstance(exit_time, datetime):
                    hour = exit_time.hour
                    hour_distribution[hour] = hour_distribution.get(hour, 0) + 1

            # Сделок в день
            if len(trades) > 1:
                first_trade_time = min(t.get('entry_time', datetime.now()) for t in trades)
                last_trade_time = max(t.get('exit_time', datetime.now()) for t in trades)

                if isinstance(first_trade_time, datetime) and isinstance(last_trade_time, datetime):
                    days = (last_trade_time - first_trade_time).days
                    trades_per_day = len(trades) / max(1, days)
                else:
                    trades_per_day = 0
            else:
                trades_per_day = 1

            metrics = {
                'avg_holding_bars': float(avg_holding),
                'median_holding_bars': float(median_holding),
                'avg_time_between_hours': float(avg_time_between),
                'median_time_between_hours': float(median_time_between),
                'trades_per_day': float(trades_per_day),
                'hour_distribution': hour_distribution
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Ошибка расчета временных метрик: {e}")
            return {}

    def calculate_distribution_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Расчет метрик распределения

        Args:
            trades (list): Список сделок

        Returns:
            dict: Метрики распределения
        """
        if not trades:
            return {}

        try:
            # PnL распределение
            pnls = [t.get('pnl', 0) for t in trades]
            pnl_percents = [t.get('pnl_percent', 0) for t in trades]

            # Статистики распределения
            skewness = stats.skew(pnls) if len(pnls) > 1 else 0
            kurtosis = stats.kurtosis(pnls) if len(pnls) > 1 else 0

            # Квантили
            quantiles = {
                'q1': float(np.percentile(pnls, 25)) if pnls else 0,
                'median': float(np.percentile(pnls, 50)) if pnls else 0,
                'q3': float(np.percentile(pnls, 75)) if pnls else 0,
                'p10': float(np.percentile(pnls, 10)) if pnls else 0,
                'p90': float(np.percentile(pnls, 90)) if pnls else 0
            }

            # Стандартное отклонение
            std_dev = float(np.std(pnls)) if len(pnls) > 1 else 0
            std_dev_percent = float(np.std(pnl_percents)) if len(pnl_percents) > 1 else 0

            # Серии побед/поражений
            outcomes = [1 if t.get('pnl', 0) > 0 else 0 for t in trades]
            max_win_streak = self._max_consecutive(outcomes, 1)
            max_loss_streak = self._max_consecutive(outcomes, 0)

            # Распределение по размеру сделки
            position_sizes = [t.get('position_size', 0) for t in trades]
            avg_position_size = float(np.mean(position_sizes)) if position_sizes else 0

            metrics = {
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'pnl_distribution': quantiles,
                'std_dev': std_dev,
                'std_dev_percent': std_dev_percent,
                'max_win_streak': max_win_streak,
                'max_loss_streak': max_loss_streak,
                'avg_position_size': avg_position_size,
                'position_size_std': float(np.std(position_sizes)) if len(position_sizes) > 1 else 0
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Ошибка расчета метрик распределения: {e}")
            return {}

    def _max_consecutive(self, arr: List[int], value: int) -> int:
        """
        Поиск максимальной последовательности значений

        Args:
            arr (list): Массив значений
            value (int): Значение для поиска

        Returns:
            int: Максимальная длина последовательности
        """
        max_count = count = 0

        for item in arr:
            if item == value:
                count += 1
                max_count = max(max_count, count)
            else:
                count = 0

        return max_count

    def _create_summary(self, statistics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Создание сводной статистики

        Args:
            statistics (dict): Полная статистика

        Returns:
            dict: Сводная статистика
        """
        basic = statistics.get('basic_metrics', {})
        performance = statistics.get('performance_metrics', {})
        risk = statistics.get('risk_metrics', {})

        summary = {
            'total_trades': basic.get('total_trades', 0),
            'win_rate_percent': basic.get('win_rate_percent', 0),
            'total_return_percent': performance.get('total_return_percent', 0),
            'annual_return_percent': performance.get('annual_return_percent', 0),
            'sharpe_ratio': performance.get('sharpe_ratio', 0),
            'max_drawdown_percent': risk.get('max_drawdown_percent', 0),
            'profit_factor': basic.get('profit_factor', 0),
            'expectancy': basic.get('expectancy', 0),
            'calculated_at': statistics.get('calculated_at')
        }

        # Оценка стратегии
        grade = self._grade_strategy(summary)
        summary['strategy_grade'] = grade

        return summary

    def _grade_strategy(self, summary: Dict[str, Any]) -> str:
        """
        Оценка стратегии по метрикам

        Args:
            summary (dict): Сводная статистика

        Returns:
            str: Оценка (A-F)
        """
        try:
            score = 0

            # Винрейт
            win_rate = summary.get('win_rate_percent', 0)
            if win_rate > 60:
                score += 3
            elif win_rate > 50:
                score += 2
            elif win_rate > 40:
                score += 1

            # Общая доходность
            total_return = summary.get('total_return_percent', 0)
            if total_return > 50:
                score += 3
            elif total_return > 20:
                score += 2
            elif total_return > 0:
                score += 1
            elif total_return < -20:
                score -= 2

            # Коэффициент Шарпа
            sharpe = summary.get('sharpe_ratio', 0)
            if sharpe > 2:
                score += 3
            elif sharpe > 1:
                score += 2
            elif sharpe > 0.5:
                score += 1

            # Максимальная просадка
            max_dd = summary.get('max_drawdown_percent', 0)
            if max_dd < 10:
                score += 3
            elif max_dd < 20:
                score += 2
            elif max_dd < 30:
                score += 1
            elif max_dd > 50:
                score -= 2

            # Фактор прибыли
            profit_factor = summary.get('profit_factor', 0)
            if profit_factor > 3:
                score += 3
            elif profit_factor > 2:
                score += 2
            elif profit_factor > 1.5:
                score += 1

            # Оценка на основе общего балла
            if score >= 12:
                return 'A'
            elif score >= 9:
                return 'B'
            elif score >= 6:
                return 'C'
            elif score >= 3:
                return 'D'
            else:
                return 'F'

        except Exception as e:
            self.logger.error(f"Ошибка оценки стратегии: {e}")
            return 'N/A'

    def generate_report(self, statistics: Dict[str, Any],
                        filename: Optional[str] = None) -> str:
        """
        Генерация текстового отчета

        Args:
            statistics (dict): Статистика
            filename (str): Имя файла для сохранения (опционально)

        Returns:
            str: Текстовый отчет
        """
        try:
            basic = statistics.get('basic_metrics', {})
            performance = statistics.get('performance_metrics', {})
            risk = statistics.get('risk_metrics', {})
            time_metrics = statistics.get('time_metrics', {})
            summary = statistics.get('summary', {})

            report_lines = [
                "=" * 80,
                "TRADING STATISTICS REPORT",
                "=" * 80,
                f"Generated: {statistics.get('calculated_at', datetime.now())}",
                "",
                "SUMMARY",
                "-" * 40,
                f"Strategy Grade: {summary.get('strategy_grade', 'N/A')}",
                f"Total Trades: {basic.get('total_trades', 0)}",
                f"Win Rate: {basic.get('win_rate_percent', 0):.1f}%",
                f"Total Return: {performance.get('total_return_percent', 0):.2f}%",
                f"Annual Return: {performance.get('annual_return_percent', 0):.2f}%",
                f"Max Drawdown: {risk.get('max_drawdown_percent', 0):.2f}%",
                f"Sharpe Ratio: {performance.get('sharpe_ratio', 0):.3f}",
                f"Profit Factor: {basic.get('profit_factor', 0):.2f}",
                "",
                "BASIC METRICS",
                "-" * 40,
                f"Winning Trades: {basic.get('winning_trades', 0)}",
                f"Losing Trades: {basic.get('losing_trades', 0)}",
                f"Total PnL: ${basic.get('total_pnl', 0):.2f}",
                f"Net PnL: ${basic.get('net_pnl', 0):.2f}",
                f"Total Commission: ${basic.get('total_commission', 0):.2f}",
                f"Average Win: ${basic.get('avg_win', 0):.2f} ({basic.get('avg_win_percent', 0):.2f}%)",
                f"Average Loss: ${basic.get('avg_loss', 0):.2f} ({basic.get('avg_loss_percent', 0):.2f}%)",
                f"Max Win: ${basic.get('max_win', 0):.2f}",
                f"Max Loss: ${basic.get('max_loss', 0):.2f}",
                f"Expectancy: ${basic.get('expectancy', 0):.2f}",
                "",
                "PERFORMANCE METRICS",
                "-" * 40,
                f"Sortino Ratio: {performance.get('sortino_ratio', 0):.3f}",
                f"Calmar Ratio: {performance.get('calmar_ratio', 0):.3f}",
                f"Recovery Factor: {performance.get('recovery_factor', 0):.3f}",
                f"Survival Rate: {performance.get('survival_rate_percent', 0):.1f}%",
                f"Final Equity: ${performance.get('final_equity', 0):.2f}",
                "",
                "RISK METRICS",
                "-" * 40,
                f"Average Drawdown: {risk.get('avg_drawdown_percent', 0):.2f}%",
                f"Annual Volatility: {risk.get('annual_volatility_percent', 0):.2f}%",
                f"Daily Volatility: {risk.get('daily_volatility_percent', 0):.2f}%",
                f"VaR (95%): {risk.get('var_95_percent', 0):.2f}%",
                f"CVaR (95%): {risk.get('cvar_95_percent', 0):.2f}%",
                f"Drawdown Count: {risk.get('drawdown_stats', {}).get('drawdown_count', 0)}",
                f"Avg Drawdown Duration: {risk.get('drawdown_stats', {}).get('avg_drawdown_duration', 0):.1f} bars",
                "",
                "TIME METRICS",
                "-" * 40,
                f"Avg Holding Period: {time_metrics.get('avg_holding_bars', 0):.1f} bars",
                f"Median Holding Period: {time_metrics.get('median_holding_bars', 0):.1f} bars",
                f"Avg Time Between Trades: {time_metrics.get('avg_time_between_hours', 0):.1f} hours",
                f"Trades Per Day: {time_metrics.get('trades_per_day', 0):.2f}",
                "",
                "=" * 80
            ]

            report = "\n".join(report_lines)

            # Сохраняем в файл, если указано
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(report)
                self.logger.info(f"Отчет сохранен в {filename}")

            return report

        except Exception as e:
            self.logger.error(f"Ошибка генерации отчета: {e}")
            return f"Error generating report: {e}"


# Создаем глобальный экземпляр для удобства
statistics_calculator = StatisticsCalculator()

