"""
Statistics module for calculating performance metrics.
"""

import numpy as np
import pandas as pd
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class Statistics:
    """Statistics calculator for trading performance."""

    @staticmethod
    def calculate_sharpe_ratio(returns: List[float],
                               risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio.

        Args:
            returns: List of returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Sharpe ratio
        """
        if not returns:
            return 0.0

        returns_array = np.array(returns)

        # Daily risk-free rate (assuming 252 trading days)
        daily_rf = risk_free_rate / 252

        excess_returns = returns_array - daily_rf

        if len(excess_returns) < 2:
            return 0.0

        if np.std(excess_returns) == 0:
            return 0.0

        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

        return float(sharpe)

    @staticmethod
    def calculate_sortino_ratio(returns: List[float],
                                risk_free_rate: float = 0.02,
                                target_return: float = 0.0) -> float:
        """Calculate Sortino ratio.

        Args:
            returns: List of returns
            risk_free_rate: Annual risk-free rate
            target_return: Target return

        Returns:
            Sortino ratio
        """
        if not returns:
            return 0.0

        returns_array = np.array(returns)

        # Daily risk-free rate
        daily_rf = risk_free_rate / 252

        excess_returns = returns_array - daily_rf - target_return

        # Consider only negative deviations
        negative_returns = excess_returns[excess_returns < 0]

        if len(negative_returns) < 2:
            return 0.0

        if np.std(negative_returns) == 0:
            return 0.0

        sortino = np.mean(excess_returns) / np.std(negative_returns) * np.sqrt(252)

        return float(sortino)

    @staticmethod
    def calculate_max_drawdown(equity_curve: List[float]) -> float:
        """Calculate maximum drawdown.

        Args:
            equity_curve: List of equity values

        Returns:
            Maximum drawdown as percentage
        """
        if not equity_curve:
            return 0.0

        equity = np.array(equity_curve)
        peak = equity[0]
        max_dd = 0.0

        for value in equity:
            if value > peak:
                peak = value

            dd = (peak - value) / peak * 100 if peak > 0 else 0

            if dd > max_dd:
                max_dd = dd

        return float(max_dd)

    @staticmethod
    def calculate_calmar_ratio(returns: List[float],
                               equity_curve: List[float]) -> float:
        """Calculate Calmar ratio.

        Args:
            returns: List of returns
            equity_curve: List of equity values

        Returns:
            Calmar ratio
        """
        if not returns or not equity_curve:
            return 0.0

        annual_return = np.mean(returns) * 252
        max_dd = Statistics.calculate_max_drawdown(equity_curve)

        if max_dd == 0:
            return 0.0

        return float(annual_return / max_dd)

    @staticmethod
    def calculate_var(returns: List[float],
                      confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk (VaR).

        Args:
            returns: List of returns
            confidence_level: Confidence level (0.95 for 95%)

        Returns:
            VaR as percentage
        """
        if not returns:
            return 0.0

        returns_array = np.array(returns)

        # Sort returns
        sorted_returns = np.sort(returns_array)

        # Calculate index for VaR
        index = int((1 - confidence_level) * len(sorted_returns))

        if index >= len(sorted_returns):
            index = len(sorted_returns) - 1

        var = sorted_returns[index] * 100

        return float(var)

    @staticmethod
    def calculate_cvar(returns: List[float],
                       confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (CVaR).

        Args:
            returns: List of returns
            confidence_level: Confidence level

        Returns:
            CVaR as percentage
        """
        if not returns:
            return 0.0

        returns_array = np.array(returns)

        # Sort returns
        sorted_returns = np.sort(returns_array)

        # Calculate index for VaR
        index = int((1 - confidence_level) * len(sorted_returns))

        if index >= len(sorted_returns):
            index = len(sorted_returns) - 1

        # Calculate average of returns below VaR
        if index > 0:
            cvar = np.mean(sorted_returns[:index]) * 100
        else:
            cvar = sorted_returns[0] * 100

        return float(cvar)

    @staticmethod
    def calculate_beta(portfolio_returns: List[float],
                       market_returns: List[float]) -> float:
        """Calculate beta coefficient.

        Args:
            portfolio_returns: Portfolio returns
            market_returns: Market returns

        Returns:
            Beta coefficient
        """
        if (not portfolio_returns or not market_returns or
                len(portfolio_returns) != len(market_returns)):
            return 0.0

        portfolio_array = np.array(portfolio_returns)
        market_array = np.array(market_returns)

        # Calculate covariance and variance
        covariance = np.cov(portfolio_array, market_array)[0, 1]
        variance = np.var(market_array)

        if variance == 0:
            return 0.0

        beta = covariance / variance

        return float(beta)

    @staticmethod
    def calculate_alpha(portfolio_returns: List[float],
                        market_returns: List[float],
                        risk_free_rate: float = 0.02) -> float:
        """Calculate alpha (Jensen's alpha).

        Args:
            portfolio_returns: Portfolio returns
            market_returns: Market returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Alpha as percentage
        """
        if (not portfolio_returns or not market_returns or
                len(portfolio_returns) != len(market_returns)):
            return 0.0

        portfolio_array = np.array(portfolio_returns)
        market_array = np.array(market_returns)

        # Daily risk-free rate
        daily_rf = risk_free_rate / 252

        beta = Statistics.calculate_beta(portfolio_returns, market_returns)

        # Calculate alpha
        alpha = (np.mean(portfolio_array) - daily_rf -
                 beta * (np.mean(market_array) - daily_rf)) * 252 * 100

        return float(alpha)

    @staticmethod
    def calculate_information_ratio(portfolio_returns: List[float],
                                    benchmark_returns: List[float]) -> float:
        """Calculate information ratio.

        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns

        Returns:
            Information ratio
        """
        if (not portfolio_returns or not benchmark_returns or
                len(portfolio_returns) != len(benchmark_returns)):
            return 0.0

        portfolio_array = np.array(portfolio_returns)
        benchmark_array = np.array(benchmark_returns)

        # Calculate active returns
        active_returns = portfolio_array - benchmark_array

        if len(active_returns) < 2:
            return 0.0

        if np.std(active_returns) == 0:
            return 0.0

        ir = np.mean(active_returns) / np.std(active_returns) * np.sqrt(252)

        return float(ir)

    @staticmethod
    def calculate_treynor_ratio(portfolio_returns: List[float],
                                market_returns: List[float],
                                risk_free_rate: float = 0.02) -> float:
        """Calculate Treynor ratio.

        Args:
            portfolio_returns: Portfolio returns
            market_returns: Market returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Treynor ratio
        """
        if not portfolio_returns:
            return 0.0

        portfolio_array = np.array(portfolio_returns)

        # Daily risk-free rate
        daily_rf = risk_free_rate / 252

        beta = Statistics.calculate_beta(portfolio_returns, market_returns)

        if beta == 0:
            return 0.0

        treynor = (np.mean(portfolio_array) - daily_rf) / beta * 252

        return float(treynor)

    @staticmethod
    def calculate_skewness(returns: List[float]) -> float:
        """Calculate skewness of returns.

        Args:
            returns: List of returns

        Returns:
            Skewness
        """
        if not returns or len(returns) < 3:
            return 0.0

        returns_array = np.array(returns)

        if np.std(returns_array) == 0:
            return 0.0

        # Calculate skewness
        mean = np.mean(returns_array)
        std = np.std(returns_array)

        if std == 0:
            return 0.0

        skewness = np.mean(((returns_array - mean) / std) ** 3)

        return float(skewness)

    @staticmethod
    def calculate_kurtosis(returns: List[float]) -> float:
        """Calculate kurtosis of returns.

        Args:
            returns: List of returns

        Returns:
            Kurtosis
        """
        if not returns or len(returns) < 4:
            return 0.0

        returns_array = np.array(returns)

        if np.std(returns_array) == 0:
            return 0.0

        # Calculate kurtosis
        mean = np.mean(returns_array)
        std = np.std(returns_array)

        if std == 0:
            return 0.0

        kurtosis = np.mean(((returns_array - mean) / std) ** 4) - 3

        return float(kurtosis)

    @staticmethod
    def get_summary_statistics(returns: List[float],
                               equity_curve: List[float]) -> dict:
        """Calculate all summary statistics.

        Args:
            returns: List of returns
            equity_curve: List of equity values

        Returns:
            Dictionary with all statistics
        """
        stats = {
            'mean_return': float(np.mean(returns)) * 100 if returns else 0.0,
            'std_return': float(np.std(returns)) * 100 if returns else 0.0,
            'sharpe_ratio': Statistics.calculate_sharpe_ratio(returns),
            'sortino_ratio': Statistics.calculate_sortino_ratio(returns),
            'max_drawdown': Statistics.calculate_max_drawdown(equity_curve),
            'calmar_ratio': Statistics.calculate_calmar_ratio(returns, equity_curve),
            'var_95': Statistics.calculate_var(returns, 0.95),
            'cvar_95': Statistics.calculate_cvar(returns, 0.95),
            'skewness': Statistics.calculate_skewness(returns),
            'kurtosis': Statistics.calculate_kurtosis(returns)
        }

        return stats

