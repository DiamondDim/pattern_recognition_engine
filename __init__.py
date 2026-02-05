"""
Pattern Recognition Engine
A comprehensive system for detecting and analyzing trading patterns.
"""

__version__ = "1.0.0"
__author__ = "Pattern Recognition Engine Team"
__email__ = "support@pattern-engine.com"

# Import key components for easier access
from .config import (
    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER,
    SYMBOL, TIMEFRAME, BARS_TO_LOAD,
    PATTERN_TYPES, MIN_CONFIDENCE,
    INITIAL_CAPITAL, get_timeframe_mt5, get_config_summary
)

# Define package-level exports
__all__ = [
    # Core modules
    'core',
    'utils',
    'patterns',

    # Configuration
    'config',

    # Version info
    '__version__',
    '__author__',
    '__email__'
]

