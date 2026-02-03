"""
Модуль паттернов для Pattern Recognition Engine
"""

from .base_pattern import BasePattern
from .candlestick_patterns import CandlestickPatterns
from .geometric_patterns import GeometricPatterns
from .harmonic_patterns import HarmonicPatterns

__version__ = "1.0.0"
__author__ = "Pattern Recognition Team"

__all__ = [
    "BasePattern",
    "CandlestickPatterns",
    "GeometricPatterns",
    "HarmonicPatterns"
]

