import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class StatisticsCalculator:
    """Калькулятор статистических показателей для торговли"""
    
    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """Расчет доходности"""
        if len(prices) < 2:
            return pd.Series([], dtype=float)
        
        returns = prices.pct_change().dropna()
        return returns
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, annualization_factor: int = 252) -> float:
        """Расчет волатильности (стандартное отклонение доходности)"""
        if len(returns) < 2:
            return 0.0
        
        daily_volatility = returns.std()
        annual_volatility = daily_volatility * np.sqrt(annualization_factor)
        return annual_volatility
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02,
                              annualization_factor: int = 252) -> float:
        """Расчет коэффициента Шарпа"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / annualization_factor)
        
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(annualization_factor)
        return sharpe
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02,
                               annualization_factor: int = 252) -> float:
        """Расчет коэффициента Сортино"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / annualization_factor)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return 0.0
        
        downside_std = downside_returns.std()
        if downside_std == 0:
            return 0.0
        
        sortino = (excess_returns.mean() / downside_std) * np.sqrt(annualization_factor)
        return sortino
    
    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """Расчет максимальной просадки"""
        if len(prices) < 2:
            return 0.0, None, None
        
        # Рассчитываем кумулятивный максимум
        cumulative_max = prices.expanding().max()
        
        # Рассчитываем просадку от максимума
        drawdown = (prices - cumulative_max) / cumulative_max
        
        max_drawdown = drawdown.min()
        max_drawdown_index = drawdown.idxmin()
        
        # Находим пик перед просадкой
        if max_drawdown_index is not None:
            peak_index = prices.loc[:max_drawdown_index].idxmax()
        else:
            peak_index = None
        
        return abs(max_drawdown), peak_index, max_drawdown_index
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, prices: pd.Series,
                              annualization_factor: int = 252) -> float:
        """Расчет коэффициента Калмара"""
        if len(returns) < 2:
            return 0.0
        
        max_dd, _, _ = StatisticsCalculator.calculate_max_drawdown(prices)
        
        if max_dd == 0:
            return 0.0
        
        annual_return = returns.mean() * annualization_factor
        calmar = annual_return / max_dd
        return calmar
    
    @staticmethod
    def calculate_win_rate(trades: List[Dict]) -> float:
        """Расчет процента прибыльных сделок"""
        if not trades:
            return 0.0
        
        winning_trades = [t for t in trades if t.get('profit', 0) > 0]
        win_rate = len(winning_trades) / len(trades) * 100
        return win_rate
    
    @staticmethod
    def calculate_profit_factor(trades: List[Dict]) -> float:
        """Расчет профит-фактора"""
        if not trades:
            return 0.0
        
        gross_profit = sum(t.get('profit', 0) for t in trades if t.get('profit', 0) > 0)
        gross_loss = abs(sum(t.get('profit', 0) for t in trades if t.get('profit', 0) < 0))
        
        if gross_loss == 0:
            return float('inf')
        
        profit_factor = gross_profit / gross_loss
        return profit_factor
    
    @staticmethod
    def calculate_average_trade(trades: List[Dict]) -> Dict[str, float]:
        """Расчет средней сделки"""
        if not trades:
            return {'avg_profit': 0, 'avg_win': 0, 'avg_loss': 0, 'avg_win_pct': 0, 'avg_loss_pct': 0}
        
        profits = [t.get('profit', 0) for t in trades]
        profits_pct = [t.get('profit_pct', 0) for t in trades]
        
        winning_trades = [p for p in profits if p > 0]
        winning_trades_pct = [p for p, pct in zip(profits_pct, profits) if p > 0]
        losing_trades = [p for p in profits if p < 0]
        losing_trades_pct = [p for p, pct in zip(profits_pct, profits) if p < 0]
        
        return {
            'avg_profit': np.mean(profits) if profits else 0,
            'avg_win': np.mean(winning_trades) if winning_trades else 0,
            'avg_loss': np.mean(losing_trades) if losing_trades else 0,
            'avg_win_pct': np.mean(winning_trades_pct) if winning_trades_pct else 0,
            'avg_loss_pct': np.mean(losing_trades_pct) if losing_trades_pct else 0
        }
    
    @staticmethod
    def calculate_recovery_factor(trades: List[Dict], max_drawdown: float) -> float:
        """Расчет фактора восстановления"""
        if max_drawdown == 0:
            return 0.0
        
        total_profit = sum(t.get('profit', 0) for t in trades)
        recovery_factor = total_profit / abs(max_drawdown) if max_drawdown != 0 else 0
        return recovery_factor
    
    @staticmethod
    def calculate_expected_value(trades: List[Dict]) -> float:
        """Расчет математического ожидания"""
        if not trades:
            return 0.0
        
        win_rate = StatisticsCalculator.calculate_win_rate(trades) / 100
        avg_trade_stats = StatisticsCalculator.calculate_average_trade(trades)
        
        avg_win = avg_trade_stats['avg_win']
        avg_loss = abs(avg_trade_stats['avg_loss'])
        
        expected_value = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        return expected_value
    
    @staticmethod
    def calculate_kelly_criterion(trades: List[Dict]) -> float:
        """Расчет критерия Келли"""
        if not trades:
            return 0.0
        
        win_rate = StatisticsCalculator.calculate_win_rate(trades) / 100
        avg_trade_stats = StatisticsCalculator.calculate_average_trade(trades)
        
        avg_win = avg_trade_stats['avg_win']
        avg_loss = abs(avg_trade_stats['avg_loss'])
        
        if avg_loss == 0:
            return 0.0
        
        # Формула Келли: f* = (bp - q) / b
        # где b = avg_win / avg_loss, p = win_rate, q = 1 - p
        b = avg_win / avg_loss if avg_loss != 0 else 0
        kelly = (b * win_rate - (1 - win_rate)) / b if b != 0 else 0
        
        # Ограничиваем максимальное значение Келли
        return min(max(kelly, 0), 0.25)  # Максимум 25% капитала на сделку
    
    @staticmethod
    def calculate_correlation(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
        """Расчет корреляции Пирсона и p-value"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0, 1.0
        
        try:
            correlation = x.corr(y)
            if np.isnan(correlation):
                return 0.0, 1.0
            
            # Расчет p-value для корреляции
            n = len(x)
            if n <= 2:
                return correlation, 1.0
            
            t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            
            return correlation, p_value
        except:
            return 0.0, 1.0
    
    @staticmethod
    def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> Tuple[float, float]:
        """Расчет бета-коэффициента и его p-value"""
        if len(asset_returns) != len(market_returns) or len(asset_returns) < 2:
            return 0.0, 1.0
        
        try:
            # Обеспечиваем одинаковую длину
            common_idx = asset_returns.index.intersection(market_returns.index)
            if len(common_idx) < 2:
                return 0.0, 1.0
            
            asset_aligned = asset_returns.loc[common_idx]
            market_aligned = market_returns.loc[common_idx]
            
            # Расчет ковариации и дисперсии
            covariance = asset_aligned.cov(market_aligned)
            market_variance = market_aligned.var()
            
            if market_variance == 0:
                return 0.0, 1.0
            
            beta = covariance / market_variance
            
            # Расчет p-value для бета
            n = len(asset_aligned)
            if n <= 2:
                return beta, 1.0
            
            # Стандартная ошибка бета
            residuals = asset_aligned - beta * market_aligned
            residuals_std = residuals.std()
            
            if residuals_std == 0 or market_variance == 0:
                return beta, 1.0
            
            se_beta = residuals_std / (np.sqrt(market_variance) * np.sqrt(n))
            t_stat = beta / se_beta
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            
            return beta, p_value
        except Exception as e:
            logger.error(f"Ошибка расчета бета: {e}")
            return 0.0, 1.0
    
    @staticmethod
    def calculate_alpha(asset_returns: pd.Series, market_returns: pd.Series,
                       risk_free_rate: float = 0.02, annualization_factor: int = 252) -> Tuple[float, float]:
        """Расчет альфа-коэффициента Дженсена и его p-value"""
        if len(asset_returns) != len(market_returns) or len(asset_returns) < 2:
            return 0.0, 1.0
        
        try:
            # Выравниваем данные
            common_idx = asset_returns.index.intersection(market_returns.index)
            if len(common_idx) < 2:
                return 0.0, 1.0
            
            asset_aligned = asset_returns.loc[common_idx]
            market_aligned = market_returns.loc[common_idx]
            
            # Рассчитываем бета
            beta, beta_pvalue = StatisticsCalculator.calculate_beta(asset_aligned, market_aligned)
            
            # Ежедневная безрисковая ставка
            daily_rf = risk_free_rate / annualization_factor
            
            # Альфа Дженсена
            alpha = (asset_aligned.mean() - daily_rf) - beta * (market_aligned.mean() - daily_rf)
            annual_alpha = alpha * annualization_factor
            
            # Расчет p-value для альфа
            n = len(asset_aligned)
            if n <= 2:
                return annual_alpha, 1.0
            
            # Стандартная ошибка альфа
            residuals = asset_aligned - (alpha + beta * market_aligned)
            residuals_std = residuals.std()
            
            if residuals_std == 0:
                return annual_alpha, 1.0
            
            se_alpha = residuals_std * np.sqrt(1/n + market_aligned.mean()**2 / (market_aligned.var() * n))
            t_stat_alpha = alpha / se_alpha
            p_value_alpha = 2 * (1 - stats.t.cdf(abs(t_stat_alpha), n - 2))
            
            return annual_alpha, p_value_alpha
        except Exception as e:
            logger.error(f"Ошибка расчета альфа: {e}")
            return 0.0, 1.0
    
    @staticmethod
    def calculate_t_test(sample1: pd.Series, sample2: pd.Series = None) -> Dict[str, Any]:
        """Проверка статистической значимости с помощью t-теста"""
        if sample2 is None:
            # Одновыборочный t-тест (проверка на отличие от нуля)
            if len(sample1) < 2:
                return {
                    't_statistic': 0,
                    'p_value': 1.0,
                    'significant_95': False,
                    'significant_99': False,
                    'mean': 0,
                    'std': 0
                }
            
            try:
                t_stat, p_value = stats.ttest_1samp(sample1, 0)
                mean = sample1.mean()
                std = sample1.std()
            except:
                t_stat, p_value = 0, 1.0
                mean, std = 0, 0
        else:
            # Двухвыборочный t-тест
            if len(sample1) < 2 or len(sample2) < 2:
                return {
                    't_statistic': 0,
                    'p_value': 1.0,
                    'significant_95': False,
                    'significant_99': False,
                    'mean1': 0,
                    'mean2': 0,
                    'std1': 0,
                    'std2': 0
                }
            
            try:
                t_stat, p_value = stats.ttest_ind(sample1, sample2)
                mean1, mean2 = sample1.mean(), sample2.mean()
                std1, std2 = sample1.std(), sample2.std()
            except:
                t_stat, p_value = 0, 1.0
                mean1, mean2, std1, std2 = 0, 0, 0, 0
        
        result = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_95': p_value < 0.05,
            'significant_99': p_value < 0.01
        }
        
        if sample2 is None:
            result.update({'mean': mean, 'std': std})
        else:
            result.update({
                'mean1': mean1,
                'mean2': mean2,
                'std1': std1,
                'std2': std2,
                'mean_diff': mean1 - mean2
            })
        
        return result
    
    @staticmethod
    def calculate_r_squared(actual: pd.Series, predicted: pd.Series) -> float:
        """Расчет коэффициента детерминации R²"""
        if len(actual) != len(predicted) or len(actual) < 2:
            return 0.0
        
        try:
            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            
            if ss_tot == 0:
                return 0.0
            
            r_squared = 1 - (ss_res / ss_tot)
            return max(min(r_squared, 1.0), 0.0)
        except:
            return 0.0
    
    @staticmethod
    def calculate_information_ratio(portfolio_returns: pd.Series, 
                                   benchmark_returns: pd.Series,
                                   annualization_factor: int = 252) -> float:
        """Расчет информационного коэффициента"""
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
            return 0.0
        
        try:
            active_returns = portfolio_returns - benchmark_returns
            
            if active_returns.std() == 0:
                return 0.0
            
            ir = (active_returns.mean() / active_returns.std()) * np.sqrt(annualization_factor)
            return ir
        except:
            return 0.0
    
    @staticmethod
    def calculate_treynor_ratio(portfolio_returns: pd.Series, 
                               market_returns: pd.Series,
                               risk_free_rate: float = 0.02,
                               annualization_factor: int = 252) -> float:
        """Расчет коэффициента Трейнора"""
        if len(portfolio_returns) != len(market_returns) or len(portfolio_returns) < 2:
            return 0.0
        
        try:
            beta, _ = StatisticsCalculator.calculate_beta(portfolio_returns, market_returns)
            
            if beta == 0:
                return 0.0
            
            excess_return = portfolio_returns.mean() - (risk_free_rate / annualization_factor)
            treynor = (excess_return / beta) * annualization_factor
            return treynor
        except:
            return 0.0
    
    @staticmethod
    def calculate_skewness_kurtosis(returns: pd.Series) -> Dict[str, float]:
        """Расчет асимметрии и эксцесса"""
        if len(returns) < 2:
            return {'skewness': 0, 'kurtosis': 0}
        
        try:
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            return {'skewness': skewness, 'kurtosis': kurtosis}
        except:
            return {'skewness': 0, 'kurtosis': 0}
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Расчет Value at Risk (VaR) историческим методом"""
        if len(returns) < 10:
            return 0.0
        
        try:
            var = np.percentile(returns, (1 - confidence_level) * 100)
            return var
        except:
            return 0.0
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Расчет Conditional Value at Risk (CVaR)"""
        if len(returns) < 10:
            return 0.0
        
        try:
            var = StatisticsCalculator.calculate_var(returns, confidence_level)
            cvar = returns[returns <= var].mean()
            return cvar if not np.isnan(cvar) else 0.0
        except:
            return 0.0
    
    @staticmethod
    def calculate_all_statistics(trades: List[Dict], equity_curve: pd.DataFrame = None) -> Dict[str, Any]:
        """Расчет всех статистических показателей"""
        if not trades:
            return {
                'error': 'Нет данных о сделках',
                'total_trades': 0,
                'win_rate': 0,
                'total_profit': 0
            }
        
        # Базовые метрики
        total_trades = len(trades)
        win_rate = StatisticsCalculator.calculate_win_rate(trades)
        profit_factor = StatisticsCalculator.calculate_profit_factor(trades)
        avg_trade = StatisticsCalculator.calculate_average_trade(trades)
        expected_value = StatisticsCalculator.calculate_expected_value(trades)
        kelly = StatisticsCalculator.calculate_kelly_criterion(trades)
        
        # Расчеты на основе кривой эквити
        if equity_curve is not None and not equity_curve.empty and 'equity' in equity_curve.columns:
            equity_series = equity_curve['equity']
            returns = StatisticsCalculator.calculate_returns(equity_series)
            
            volatility = StatisticsCalculator.calculate_volatility(returns)
            sharpe = StatisticsCalculator.calculate_sharpe_ratio(returns)
            sortino = StatisticsCalculator.calculate_sortino_ratio(returns)
            max_dd, peak_idx, dd_idx = StatisticsCalculator.calculate_max_drawdown(equity_series)
            calmar = StatisticsCalculator.calculate_calmar_ratio(returns, equity_series)
            recovery = StatisticsCalculator.calculate_recovery_factor(trades, max_dd)
            
            # Дополнительные метрики
            skew_kurt = StatisticsCalculator.calculate_skewness_kurtosis(returns)
            var_95 = StatisticsCalculator.calculate_var(returns, 0.95)
            cvar_95 = StatisticsCalculator.calculate_cvar(returns, 0.95)
        else:
            volatility = sharpe = sortino = max_dd = calmar = recovery = 0.0
            peak_idx = dd_idx = None
            skew_kurt = {'skewness': 0, 'kurtosis': 0}
            var_95 = cvar_95 = 0.0
        
        # Сбор всех результатов
        results = {
            'performance_metrics': {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_profit': sum(t.get('profit', 0) for t in trades),
                'total_profit_pct': (sum(t.get('profit', 0) for t in trades) / 
                                   (trades[0].get('open_price', 1) * trades[0].get('volume', 1)) * 100 
                                   if trades else 0),
                'expected_value': expected_value,
                'kelly_criterion': kelly
            },
            'risk_metrics': {
                'volatility_annual': volatility,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'max_drawdown': max_dd,
                'max_drawdown_pct': max_dd * 100 if max_dd else 0,
                'calmar_ratio': calmar,
                'recovery_factor': recovery,
                'var_95': var_95,
                'cvar_95': cvar_95
            },
            'trade_metrics': avg_trade,
            'distribution_metrics': skew_kurt,
            'drawdown_info': {
                'peak_index': peak_idx,
                'drawdown_index': dd_idx,
                'drawdown_duration': (dd_idx - peak_idx).days if peak_idx and dd_idx else 0
            }
        }
        
        return results

