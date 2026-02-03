"""
Core modules for Pattern Recognition Engine
"""

__version__ = "1.0.0"
__author__ = "Pattern Recognition Team"

from .pattern_detector import PatternDetector
from .pattern_analyzer import PatternAnalyzer
from .pattern_database import PatternDatabase
from .data_feeder import DataFeeder
from .ml_models import PatternMLModel
from .backtesting import PatternBacktester

__all__ = [
    "PatternDetector",
    "PatternAnalyzer",
    "PatternDatabase",
    "DataFeeder",
    "PatternMLModel",
    "PatternBacktester",
]

