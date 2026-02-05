"""
Backtesting engine for evaluating pattern-based trading strategies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
import warnings

from .statistics import Statistics
import config

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class BacktestingEngine:
    """Main backtesting engine for evaluating trading strategies."""

    def __init__(self, initial_capital: float = None,
                 commission: float = None,
                 slippage: float = None):
        """Initialize backtesting engine.

        Args:
            initial_capital: Starting capital
            commission: Commission per trade (percentage)
            slippage: Slippage percentage
        """
        self.initial_capital = initial_capital if initial_capital else config.INITIAL_CAPITAL
        self.commission = commission if commission else 0.0005  # 0.05%
        self.slippage = slippage if slippage else 0.0001  # 0.01%
        self.statistics = Statistics()

    def run(self,
            data: pd.DataFrame,
            signals: List[Dict[str, Any]],
            stop_loss_pct: float = None,
            take_profit_pct: float = None,
            position_size_pct: float = None) -> Dict[str, Any]:
        """Run backtesting on provided data and signals.

        Args:
            data: Market data DataFrame
            signals: List of trading signals
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            position_size_pct: Position size percentage

        Returns:
            Dictionary with backtesting results
        """
        if stop_loss_pct is None:
            stop_loss_pct = config.STOP_LOSS_PCT
        if take_profit_pct is None:
            take_profit_pct = config.TAKE_PROFIT_PCT
        if position_size_pct is None:
            position_size_pct = config.POSITION_SIZE_PCT

        logger.info(f"Starting backtesting with {len(signals)} signals")
        logger.info(f"Initial capital: ${self.initial_capital:.2f}")

        # Validate inputs
        if data.empty:
            logger.error("Empty data provided for backtesting")
            return self._create_empty_report()

        if not signals:
            logger.warning("No signals provided for backtesting")
            return self._create_empty_report()

        # Initialize tracking variables
        capital = self.initial_capital
        position = 0.0  # Current position size
        entry_price = 0.0
        entry_index = 0
        trades = []
        equity_curve = [capital]

        # Sort signals by index
        signals = sorted(signals, key=lambda x: x.get('index', 0))

        # Process each signal
        for i, signal in enumerate(signals):
            idx = signal.get('index', 0)
            if idx >= len(data):
                continue

            try:
                current_price = data.iloc[idx]['close']
                signal_type = signal.get('type', '').lower()
                strength = signal.get('strength', 0.5)

                # Validate signal type
                if signal_type not in ['buy', 'sell']:
                    logger.warning(f"Invalid signal type: {signal_type}")
                    continue

                # Close existing position if signal is opposite
                if position != 0:
                    should_close = False

                    # Close long position on sell signal
                    if position > 0 and signal_type == 'sell':
                        should_close = True
                    # Close short position on buy signal
                    elif position < 0 and signal_type == 'buy':
                        should_close = True

                    if should_close:
                        # Calculate exit price with slippage
                        if position > 0:  # Long position
                            exit_price = current_price * (1 - self.slippage)
                        else:  # Short position
                            exit_price = current_price * (1 + self.slippage)

                        # Calculate P&L
                        pnl = position * (exit_price - entry_price)
                        commission_amount = abs(position) * self.commission
                        pnl -= commission_amount
                        capital += pnl

                        # Record trade
                        trade = {
                            'entry_index': entry_index,
                            'exit_index': idx,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'position': position,
                            'pnl': pnl,
                            'commission': commission_amount,
                            'type': 'long' if position > 0 else 'short',
                            'status': 'closed',
                            'close_reason': 'signal_reverse'
                        }
                        trades.append(trade)

                        logger.debug(f"Closed position: P&L = ${pnl:.2f}")

                        position = 0.0
                        entry_price = 0.0

                # Open new position if no position
                if position == 0:
                    # Calculate position size based on capital and risk
                    position_size = capital * position_size_pct * strength

                    if position_size <= 0:
                        continue

                    # Calculate number of units (simplified)
                    position = position_size / current_price

                    if signal_type == 'sell':
                        position = -position

                    entry_price = current_price
                    entry_index = idx

                    # Calculate stop loss and take profit
                    if position > 0:  # Long position
                        stop_loss = entry_price * (1 - stop_loss_pct)
                        take_profit = entry_price * (1 + take_profit_pct)
                    else:  # Short position
                        stop_loss = entry_price * (1 + stop_loss_pct)
                        take_profit = entry_price * (1 - take_profit_pct)

                    # Record opening trade
                    trade = {
                        'entry_index': idx,
                        'entry_price': entry_price,
                        'position': position,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'type': 'long' if position > 0 else 'short',
                        'status': 'open',
                        'size': abs(position_size)
                    }
                    trades.append(trade)

                    logger.debug(f"Opened {trade['type']} position at ${entry_price:.5f}")

                # Update equity curve
                current_equity = capital
                if position != 0:
                    unrealized_pnl = position * (current_price - entry_price)
                    current_equity += unrealized_pnl

                equity_curve.append(current_equity)

            except Exception as e:
                logger.error(f"Error processing signal {i}: {e}")
                continue

        # Close any remaining position at the end
        if position != 0 and not data.empty:
            try:
                exit_price = data.iloc[-1]['close']
                if position > 0:  # Long position
                    exit_price *= (1 - self.slippage)
                else:  # Short position
                    exit_price *= (1 + self.slippage)

                pnl = position * (exit_price - entry_price)
                commission_amount = abs(position) * self.commission
                pnl -= commission_amount
                capital += pnl

                # Record final trade
                trade = {
                    'entry_index': entry_index,
                    'exit_index': len(data) - 1,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position': position,
                    'pnl': pnl,
                    'commission': commission_amount,
                    'type': 'long' if position > 0 else 'short',
                    'status': 'closed',
                    'close_reason': 'end_of_data'
                }
                trades.append(trade)

                logger.debug(f"Closed final position: P&L = ${pnl:.2f}")

            except Exception as e:
                logger.error(f"Error closing final position: {e}")

        # Calculate performance metrics
        report = self._generate_report(trades, equity_curve, capital)

        logger.info(f"Backtesting completed: {report['total_trades']} trades, "
                    f"{report['total_return_percent']:.2f}% return")

        return report

    def _create_empty_report(self) -> Dict[str, Any]:
        """Create an empty report when no backtesting can be performed."""
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.initial_capital,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'total_return_percent': 0.0,
            'max_drawdown_percent': 0.0,
            'win_rate_percent': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'trades': [],
            'equity_curve': [self.initial_capital],
            'status': 'no_trades'
        }

    def _generate_report(self, trades: List[Dict[str, Any]],
                         equity_curve: List[float],
                         final_capital: float) -> Dict[str, Any]:
        """Generate comprehensive backtesting report.

        Args:
            trades: List of completed trades
            equity_curve: List of equity values over time
            final_capital: Final capital after all trades

        Returns:
            Dictionary with detailed report
        """
        # Filter only closed trades
        closed_trades = [t for t in trades if t.get('status') == 'closed']
        total_trades = len(closed_trades)

        if total_trades == 0:
            return self._create_empty_report()

        # Calculate basic metrics
        winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in closed_trades if t.get('pnl', 0) <= 0]

        total_pnl = sum(t.get('pnl', 0) for t in closed_trades)
        total_return_pct = (total_pnl / self.initial_capital) * 100

        # Calculate win rate
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0

        # Calculate profit factor
        total_profit = sum(t.get('pnl', 0) for t in winning_trades)
        total_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # Calculate max drawdown
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max * 100
        max_drawdown_pct = drawdowns.min() if not drawdowns.empty else 0

        # Calculate risk-adjusted metrics
        returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i - 1] != 0:
                daily_return = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
                returns.append(daily_return)

        sharpe_ratio = self.statistics.calculate_sharpe_ratio(returns) if returns else 0
        sortino_ratio = self.statistics.calculate_sortino_ratio(returns) if returns else 0

        # Prepare detailed report
        report = {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'total_pnl': total_pnl,
            'total_return_percent': total_return_pct,
            'max_drawdown_percent': max_drawdown_pct,
            'win_rate_percent': win_rate,
            'profit_factor': profit_factor if profit_factor != float('inf') else 0,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'avg_win': np.mean([t.get('pnl', 0) for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t.get('pnl', 0) for t in losing_trades]) if losing_trades else 0,
            'largest_win': max([t.get('pnl', 0) for t in winning_trades], default=0),
            'largest_loss': min([t.get('pnl', 0) for t in losing_trades], default=0),
            'trades': closed_trades,
            'equity_curve': equity_curve,
            'returns': returns,
            'status': 'success'
        }

        return report

    def save_report(self, report: Dict[str, Any],
                    filename: str = 'backtesting_report.json'):
        """Save backtesting report to JSON file.

        Args:
            report: Backtesting report dictionary
            filename: Output filename
        """
        try:
            # Convert numpy types to Python types for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                else:
                    return obj

            serializable_report = convert_to_serializable(report)

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_report, f, indent=2, ensure_ascii=False)

            logger.info(f"Backtesting report saved to {filename}")

        except Exception as e:
            logger.error(f"Failed to save report to {filename}: {e}")

    def get_summary(self, report: Dict[str, Any]) -> str:
        """Get a text summary of the backtesting report.

        Args:
            report: Backtesting report dictionary

        Returns:
            Formatted text summary
        """
        summary_lines = [
            "=" * 60,
            "BACKTESTING RESULTS SUMMARY",
            "=" * 60,
            f"Initial Capital: ${report['initial_capital']:,.2f}",
            f"Final Capital: ${report['final_capital']:,.2f}",
            f"Total P&L: ${report['total_pnl']:,.2f}",
            f"Total Return: {report['total_return_percent']:,.2f}%",
            "",
            f"Total Trades: {report['total_trades']}",
            f"Winning Trades: {report['winning_trades']}",
            f"Losing Trades: {report['losing_trades']}",
            f"Win Rate: {report['win_rate_percent']:.1f}%",
            "",
            f"Max Drawdown: {report['max_drawdown_percent']:.2f}%",
            f"Profit Factor: {report['profit_factor']:.2f}",
            f"Sharpe Ratio: {report['sharpe_ratio']:.3f}",
            f"Sortino Ratio: {report['sortino_ratio']:.3f}",
            "",
            f"Average Win: ${report['avg_win']:,.2f}",
            f"Average Loss: ${report['avg_loss']:,.2f}",
            f"Largest Win: ${report['largest_win']:,.2f}",
            f"Largest Loss: ${report['largest_loss']:,.2f}",
            "=" * 60
        ]

        return "\n".join(summary_lines)

    def optimize_parameters(self,
                            data: pd.DataFrame,
                            signals: List[Dict[str, Any]],
                            param_grid: Dict[str, List[float]]) -> Dict[str, Any]:
        """Optimize backtesting parameters using grid search.

        Args:
            data: Market data
            signals: Trading signals
            param_grid: Parameter grid for optimization

        Returns:
            Dictionary with optimization results
        """
        logger.info("Starting parameter optimization")

        best_result = None
        best_score = -float('inf')
        results = []

        # Generate parameter combinations
        from itertools import product

        param_names = list(param_grid.keys())
        param_values = list(product(*param_grid.values()))

        for i, values in enumerate(param_values):
            params = dict(zip(param_names, values))

            logger.debug(f"Testing parameters: {params}")

            try:
                # Run backtest with current parameters
                report = self.run(
                    data=data,
                    signals=signals,
                    stop_loss_pct=params.get('stop_loss_pct', config.STOP_LOSS_PCT),
                    take_profit_pct=params.get('take_profit_pct', config.TAKE_PROFIT_PCT),
                    position_size_pct=params.get('position_size_pct', config.POSITION_SIZE_PCT)
                )

                # Calculate score (combination of multiple metrics)
                score = (
                        report['total_return_percent'] * 0.4 +
                        report['sharpe_ratio'] * 0.3 +
                        (100 - report['max_drawdown_percent']) * 0.2 +
                        report['win_rate_percent'] * 0.1
                )

                result = {
                    'params': params,
                    'report': report,
                    'score': score
                }

                results.append(result)

                if score > best_score:
                    best_score = score
                    best_result = result

                logger.debug(f"Score: {score:.2f}, Return: {report['total_return_percent']:.2f}%")

            except Exception as e:
                logger.error(f"Error testing parameters {params}: {e}")
                continue

        # Sort results by score
        results.sort(key=lambda x: x['score'], reverse=True)

        optimization_result = {
            'best_result': best_result,
            'all_results': results[:10],  # Top 10 results
            'param_grid': param_grid,
            'total_tests': len(results),
            'best_score': best_score
        }

        logger.info(f"Optimization completed. Best score: {best_score:.2f}")

        return optimization_result

