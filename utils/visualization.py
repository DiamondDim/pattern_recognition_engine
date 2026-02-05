"""
Visualization module for pattern recognition results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, Polygon
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import List, Dict, Any, Optional
import os
import logging

# Убираем проблемный импорт config
# Вместо этого определяем конфигурацию прямо здесь или импортируем безопасно
try:
    import config
    VISUALIZATION_CONFIG = {
        'style': 'seaborn-v0_8-darkgrid',
        'colors': {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'background': '#f0f0f0'
        },
        'figure_size': (12, 8),
        'dpi': 100,
        'save_format': 'png'
    }
except ImportError:
    VISUALIZATION_CONFIG = {
        'style': 'seaborn-v0_8-darkgrid',
        'colors': {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'background': '#f0f0f0'
        },
        'figure_size': (12, 8),
        'dpi': 100,
        'save_format': 'png'
    }

logger = logging.getLogger(__name__)


class Visualization:
    """Main visualization class for pattern recognition results."""

    def __init__(self, style: str = None):
        """Initialize visualization with style settings.

        Args:
            style: Matplotlib style
        """
        self.style = style or VISUALIZATION_CONFIG.get('style', 'seaborn-v0_8-darkgrid')
        self.colors = VISUALIZATION_CONFIG.get('colors', {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'background': '#f0f0f0'
        })

        # Set matplotlib style
        try:
            plt.style.use(self.style)
        except:
            plt.style.use('default')

    def plot_patterns(self, data: pd.DataFrame,
                     patterns: List[Dict[str, Any]] = None,
                     title: str = "Pattern Recognition Results",
                     save_path: Optional[str] = None) -> plt.Figure:
        """Plot price data with detected patterns.

        Args:
            data: OHLC data with datetime index
            patterns: List of detected patterns
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        if data.empty:
            logger.warning("Empty data provided for plotting")
            return None

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                       gridspec_kw={'height_ratios': [3, 1]})

        # Plot OHLC data
        ax1.plot(data.index, data['close'], label='Close Price',
                color=self.colors['primary'], linewidth=1.5, alpha=0.7)

        # Plot volume if available
        if 'volume' in data.columns:
            ax2.bar(data.index, data['volume'], color=self.colors['secondary'],
                   alpha=0.5, label='Volume')

        # Plot patterns if provided
        if patterns:
            for pattern in patterns:
                self._plot_single_pattern(ax1, data, pattern)

        # Formatting
        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_ylabel('Volume', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Format x-axis
        fig.autofmt_xdate()

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        return fig

    def _plot_single_pattern(self, ax: plt.Axes,
                           data: pd.DataFrame,
                           pattern: Dict[str, Any]):
        """Plot a single pattern on the axis.

        Args:
            ax: Matplotlib axis
            data: OHLC data
            pattern: Pattern dictionary
        """
        pattern_type = pattern.get('pattern_type', '').lower()
        index = pattern.get('index', 0)
        confidence = pattern.get('confidence', 0)

        if index >= len(data):
            return

        price = data.iloc[index]['close']

        # Choose color based on pattern type
        if 'bullish' in pattern_type:
            color = 'green'
            marker = '^'
        elif 'bearish' in pattern_type:
            color = 'red'
            marker = 'v'
        else:
            color = 'blue'
            marker = 'o'

        # Plot pattern marker
        ax.plot(data.index[index], price, marker=marker,
               markersize=10, color=color, alpha=0.7,
               label=f"{pattern_type} ({confidence:.2f})")

    def plot_results(self, report: Dict[str, Any],
                    title: str = "Backtesting Results",
                    save_path: Optional[str] = None) -> plt.Figure:
        """Plot backtesting results.

        Args:
            report: Backtesting report dictionary
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        if not report:
            logger.warning("Empty report provided for plotting")
            return None

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))

        # 1. Equity curve
        ax1 = plt.subplot(2, 2, 1)
        if 'equity_curve' in report and report['equity_curve']:
            equity = report['equity_curve']
            ax1.plot(range(len(equity)), equity,
                    color=self.colors['primary'], linewidth=2)
            ax1.set_title('Equity Curve', fontsize=14)
            ax1.set_xlabel('Trade #')
            ax1.set_ylabel('Equity ($)')
            ax1.grid(True, alpha=0.3)

        # 2. Drawdown
        ax2 = plt.subplot(2, 2, 2)
        if 'equity_curve' in report and report['equity_curve']:
            equity = pd.Series(report['equity_curve'])
            rolling_max = equity.expanding().max()
            drawdown = (equity - rolling_max) / rolling_max * 100
            ax2.fill_between(range(len(drawdown)), drawdown, 0,
                           color='red', alpha=0.3)
            ax2.set_title('Drawdown', fontsize=14)
            ax2.set_xlabel('Trade #')
            ax2.set_ylabel('Drawdown (%)')
            ax2.grid(True, alpha=0.3)

        # 3. P&L distribution
        ax3 = plt.subplot(2, 2, 3)
        if 'trades' in report and report['trades']:
            trades = report['trades']
            pnls = [t.get('pnl', 0) for t in trades if isinstance(t, dict)]
            if pnls:
                ax3.hist(pnls, bins=20, color=self.colors['secondary'],
                        alpha=0.7, edgecolor='black')
                ax3.axvline(x=0, color='red', linestyle='--', linewidth=1)
                ax3.set_title('P&L Distribution', fontsize=14)
                ax3.set_xlabel('P&L ($)')
                ax3.set_ylabel('Frequency')
                ax3.grid(True, alpha=0.3)

        # 4. Performance metrics
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')

        metrics_text = []
        if 'total_trades' in report:
            metrics_text.append(f"Total Trades: {report['total_trades']}")
        if 'winning_trades' in report:
            metrics_text.append(f"Winning Trades: {report['winning_trades']}")
        if 'losing_trades' in report:
            metrics_text.append(f"Losing Trades: {report['losing_trades']}")
        if 'win_rate_percent' in report:
            metrics_text.append(f"Win Rate: {report['win_rate_percent']:.1f}%")
        if 'total_pnl' in report:
            metrics_text.append(f"Total P&L: ${report['total_pnl']:.2f}")
        if 'total_return_percent' in report:
            metrics_text.append(f"Total Return: {report['total_return_percent']:.2f}%")
        if 'max_drawdown_percent' in report:
            metrics_text.append(f"Max Drawdown: {report['max_drawdown_percent']:.2f}%")
        if 'profit_factor' in report:
            metrics_text.append(f"Profit Factor: {report['profit_factor']:.2f}")

        if metrics_text:
            ax4.text(0.1, 0.5, '\n'.join(metrics_text),
                    fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Results plot saved to {save_path}")

        return fig

    def plot_equity_curve(self, equity_curve: List[float],
                         title: str = "Equity Curve",
                         save_path: Optional[str] = None) -> plt.Figure:
        """Plot equity curve.

        Args:
            equity_curve: List of equity values
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(equity_curve, color=self.colors['primary'], linewidth=2)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Trade #')
        ax.set_ylabel('Equity ($)')
        ax.grid(True, alpha=0.3)

        # Add final equity annotation
        if equity_curve:
            final_equity = equity_curve[-1]
            initial_equity = equity_curve[0] if equity_curve else 0
            total_return = ((final_equity - initial_equity) / initial_equity * 100) if initial_equity else 0

            ax.annotate(f'Final: ${final_equity:.2f}\nReturn: {total_return:.2f}%',
                       xy=(0.98, 0.02), xycoords='axes fraction',
                       ha='right', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')

        return fig

    def create_interactive_chart(self, data: pd.DataFrame,
                               patterns: List[Dict[str, Any]] = None,
                               title: str = "Interactive Chart") -> go.Figure:
        """Create interactive Plotly chart.

        Args:
            data: OHLC data
            patterns: List of detected patterns
            title: Chart title

        Returns:
            Plotly figure
        """
        fig = make_subplots(rows=2, cols=1,
                           shared_xaxes=True,
                           vertical_spacing=0.03,
                           row_heights=[0.7, 0.3])

        # Candlestick chart
        fig.add_trace(go.Candlestick(x=data.index,
                                    open=data['open'],
                                    high=data['high'],
                                    low=data['low'],
                                    close=data['close'],
                                    name='OHLC'),
                     row=1, col=1)

        # Volume chart
        if 'volume' in data.columns:
            fig.add_trace(go.Bar(x=data.index,
                                y=data['volume'],
                                name='Volume',
                                marker_color=self.colors['secondary']),
                         row=2, col=1)

        # Add pattern markers
        if patterns:
            for pattern in patterns:
                idx = pattern.get('index', 0)
                if idx < len(data):
                    fig.add_trace(go.Scatter(
                        x=[data.index[idx]],
                        y=[data.iloc[idx]['close']],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up' if 'bullish' in pattern.get('pattern_type', '').lower() else 'triangle-down',
                            size=15,
                            color='green' if 'bullish' in pattern.get('pattern_type', '').lower() else 'red'
                        ),
                        name=pattern.get('pattern_type', 'Pattern'),
                        text=f"{pattern.get('pattern_type')} ({pattern.get('confidence', 0):.2f})"
                    ), row=1, col=1)

        # Update layout
        fig.update_layout(title=title,
                         yaxis_title='Price',
                         xaxis_rangeslider_visible=False,
                         template='plotly_white')

        return fig

    def plot_correlation_matrix(self, data: pd.DataFrame,
                               title: str = "Correlation Matrix",
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot correlation matrix.

        Args:
            data: DataFrame with numeric columns
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        # Calculate correlation matrix
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            logger.warning("No numeric columns for correlation matrix")
            return None

        corr_matrix = numeric_data.corr()

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot heatmap
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

        # Add colorbar
        plt.colorbar(im, ax=ax)

        # Set ticks and labels
        ax.set_xticks(np.arange(len(corr_matrix.columns)))
        ax.set_yticks(np.arange(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_matrix.columns)

        # Add correlation values
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                text = ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                             ha="center", va="center",
                             color="black" if abs(corr_matrix.iloc[i, j]) < 0.7 else "white")

        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')

        return fig

    def plot_confusion_matrix(self, cm: np.ndarray,
                            classes: List[str],
                            title: str = "Confusion Matrix",
                            save_path: Optional[str] = None) -> plt.Figure:
        """Plot confusion matrix.

        Args:
            cm: Confusion matrix array
            classes: List of class names
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        # Set ticks and labels
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes,
               yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')

        return fig


def save_plot(fig: plt.Figure, filepath: str, dpi: int = 100) -> bool:
    """Save plot to file.

    Args:
        fig: Matplotlib figure
        filepath: Output file path
        dpi: Resolution in dots per inch

    Returns:
        True if successful, False otherwise
    """
    try:
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        logger.info(f"Plot saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save plot to {filepath}: {e}")
        return False

