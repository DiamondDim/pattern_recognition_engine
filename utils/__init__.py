"""
Utility modules for Pattern Recognition Engine.
"""

from .logger import setup_logging, get_logger, LoggingMixin, logger
from .mt5_connector import MT5Connector, get_mt5_connector
from .helpers import (
    validate_dataframe, calculate_returns,
    normalize_data, create_lagged_features,
    split_train_test, calculate_metrics,
    resample_data, detect_outliers_iqr,
    calculate_correlation_matrix,
    save_dataframe, load_dataframe
)
from .visualization import (
    plot_patterns, plot_results, plot_equity_curve,
    plot_correlation_matrix, plot_confusion_matrix,
    create_interactive_chart, save_plot
)

__all__ = [
    # Logging
    'setup_logging',
    'get_logger',
    'LoggingMixin',
    'logger',

    # MT5
    'MT5Connector',
    'get_mt5_connector',

    # Helpers
    'validate_dataframe',
    'calculate_returns',
    'normalize_data',
    'create_lagged_features',
    'split_train_test',
    'calculate_metrics',
    'resample_data',
    'detect_outliers_iqr',
    'calculate_correlation_matrix',
    'save_dataframe',
    'load_dataframe',

    # Visualization
    'plot_patterns',
    'plot_results',
    'plot_equity_curve',
    'plot_correlation_matrix',
    'plot_confusion_matrix',
    'create_interactive_chart',
    'save_plot'
]

