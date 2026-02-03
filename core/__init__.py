"""
Пакет ядра Pattern Recognition Engine
"""

from .data_feeder import DataFeeder
from .pattern_detector import PatternDetector
from .pattern_analyzer import PatternAnalyzer, pattern_analyzer
from .pattern_database import PatternDatabase
from .backtesting import BacktestEngine
from .statistics import calculate_statistics, StatisticsCalculator
from .ml_models import MLModel, PatternClassifier
from .pattern_analyzer import PatternAnalyzer  # Дублируем импорт для обратной совместимости

__all__ = [
    'DataFeeder',
    'PatternDetector',
    'PatternAnalyzer',
    'pattern_analyzer',
    'PatternDatabase',
    'BacktestEngine',
    'calculate_statistics',
    'StatisticsCalculator',
    'MLModel',
    'PatternClassifier'
]

