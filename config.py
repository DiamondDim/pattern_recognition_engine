"""
Конфигурация Pattern Recognition Engine
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import json

class ConnectionMode(str, Enum):
    FILE = "file"
    SOCKET = "socket"
    REST = "rest"

class PatternCategory(str, Enum):
    GEOMETRIC = "geometric"
    CANDLESTICK = "candlestick"
    HARMONIC = "harmonic"
    CUSTOM = "custom"

# Пути проекта
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
LOG_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"

# Создание директорий
for directory in [DATA_DIR, INPUT_DIR, OUTPUT_DIR, LOG_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

@dataclass
class DatabaseConfig:
    """Конфигурация базы данных"""
    TYPE: str = "sqlite"  # sqlite, postgresql, mysql
    HOST: str = "localhost"
    PORT: int = 5432
    NAME: str = "patterns.db"
    USER: str = ""
    PASSWORD: str = ""

    @property
    def connection_string(self) -> str:
        if self.TYPE == "sqlite":
            return f"sqlite:///{BASE_DIR / 'data' / self.NAME}"
        elif self.TYPE == "postgresql":
            return f"postgresql://{self.USER}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.NAME}"
        else:
            return ""

@dataclass
class DetectionConfig:
    """Конфигурация детектирования паттернов"""

    # Основные параметры
    MIN_CANDLES_FOR_PATTERN: int = 5
    MAX_CANDLES_FOR_PATTERN: int = 100
    MIN_PATTERN_QUALITY: float = 0.6
    CONFIDENCE_THRESHOLD: float = 0.7

    # Допуски
    PRICE_TOLERANCE_PCT: float = 0.002  # 0.2%
    TIME_TOLERANCE_PCT: float = 0.15    # 15%
    SYMMETRY_TOLERANCE: float = 0.15    # 15%

    # Экстремумы
    EXTREMA_ORDER: int = 5  # Порядок для поиска экстремумов
    MIN_PROMINENCE_PCT: float = 0.001  # Минимальная значимость экстремума

    # Таймфреймы
    TIMEFRAMES: List[str] = field(default_factory=lambda: [
        'M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN'
    ])

    # Типы паттернов для поиска
    ENABLED_GEOMETRIC: List[str] = field(default_factory=lambda: [
        'head_shoulders', 'inverse_head_shoulders',
        'double_top', 'double_bottom',
        'triangle_symmetric', 'triangle_ascending', 'triangle_descending',
        'wedge_rising', 'wedge_falling',
        'flag_bullish', 'flag_bearish',
        'pennant'
    ])

    ENABLED_CANDLESTICK: List[str] = field(default_factory=lambda: [
        'engulfing_bullish', 'engulfing_bearish',
        'hammer', 'hanging_man',
        'doji', 'long_legged_doji', 'dragonfly_doji', 'gravestone_doji',
        'morning_star', 'evening_star',
        'piercing_pattern', 'dark_cloud_cover',
        'three_white_soldiers', 'three_black_crows'
    ])

    ENABLED_HARMONIC: List[str] = field(default_factory=lambda: [
        'abcd', 'gartley', 'butterfly', 'bat', 'crab'
    ])

@dataclass
class AnalysisConfig:
    """Конфигурация анализа"""

    # Исторический анализ
    HISTORY_DEPTH_CANDLES: int = 5000
    MIN_HISTORICAL_MATCHES: int = 5
    SIMILARITY_THRESHOLD: float = 0.8

    # Статистика
    CALCULATE_STATISTICS: bool = True
    SAVE_PATTERN_HISTORY: bool = True

    # ML/AI
    USE_ML_MODELS: bool = False
    ML_MODEL_PATH: str = str(MODELS_DIR / "pattern_predictor.pkl")
    TRAIN_MODEL_PERIOD: int = 100  # дней

@dataclass
class RiskConfig:
    """Конфигурация управления рисками"""

    DEFAULT_RISK_REWARD_RATIO: float = 2.0
    MIN_RISK_REWARD_RATIO: float = 1.5
    MAX_POSITION_SIZE_PCT: float = 2.0  # % от капитала
    STOP_LOSS_PCT: float = 1.0  # % от цены
    TAKE_PROFIT_PCT: float = 2.0  # % от цены

    # Уровни подтверждения
    REQUIRE_VOLUME_CONFIRMATION: bool = True
    REQUIRE_TREND_CONFIRMATION: bool = True
    REQUIRE_MULTIPLE_TIMEFRAME: bool = False

@dataclass
class VisualizationConfig:
    """Конфигурация визуализации"""

    ENABLE_PLOTTING: bool = True
    SAVE_PLOTS: bool = True
    PLOT_FORMAT: str = "png"  # png, svg, pdf
    PLOT_DPI: int = 150

    # Цвета
    COLORS: Dict[str, str] = field(default_factory=lambda: {
        'bullish': '#26A69A',
        'bearish': '#EF5350',
        'neutral': '#78909C',
        'support': '#42A5F5',
        'resistance': '#FF7043',
        'entry': '#7E57C2',
        'stop_loss': '#FF4081',
        'take_profit': '#66BB6A'
    })

    # Размеры
    FIGURE_SIZE: tuple = (16, 10)
    FONT_SIZE: int = 10

@dataclass
class MT5Config:
    """Конфигурация подключения к MetaTrader 5"""

    CONNECTION_MODE: ConnectionMode = ConnectionMode.FILE

    # Файловый режим
    INPUT_FILE_PATH: str = str(INPUT_DIR / "mt5_data.csv")
    OUTPUT_FILE_PATH: str = str(OUTPUT_DIR / "patterns.json")
    UPDATE_INTERVAL_SEC: int = 5

    # Socket режим
    SOCKET_HOST: str = "localhost"
    SOCKET_PORT: int = 5555
    SOCKET_PROTOCOL: str = "tcp"  # tcp, websocket

    # Данные для экспорта
    SYMBOLS: List[str] = field(default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"])
    EXPORT_HISTORY_DAYS: int = 30
    EXPORT_TIMEFRAMES: List[str] = field(default_factory=lambda: ["M5", "M15", "H1", "H4"])

@dataclass
class LoggingConfig:
    """Конфигурация логирования"""

    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = str(LOG_DIR / "pre_engine.log")
    LOG_FORMAT: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    ROTATION: str = "10 MB"
    RETENTION: str = "30 days"
    COMPRESSION: str = "zip"

@dataclass
class AppConfig:
    """Основная конфигурация приложения"""

    APP_NAME: str = "Pattern Recognition Engine"
    VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Конфигурации
    database: DatabaseConfig = DatabaseConfig()
    detection: DetectionConfig = DetectionConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    risk: RiskConfig = RiskConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    mt5: MT5Config = MT5Config()
    logging: LoggingConfig = LoggingConfig()

    def save(self, filepath: str = None):
        """Сохранение конфигурации в файл"""
        if filepath is None:
            filepath = BASE_DIR / "config_backup.json"

        config_dict = {
            'app': {
                'name': self.APP_NAME,
                'version': self.VERSION,
                'debug': self.DEBUG
            },
            'database': self.database.__dict__,
            'detection': self.detection.__dict__,
            'analysis': self.analysis.__dict__,
            'risk': self.risk.__dict__,
            'visualization': self.visualization.__dict__,
            'mt5': self.mt5.__dict__,
            'logging': self.logging.__dict__
        }

        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def load(self, filepath: str):
        """Загрузка конфигурации из файла"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        # Обновляем конфигурации
        for key, value in config_dict.items():
            if hasattr(self, key):
                config_obj = getattr(self, key)
                for k, v in value.items():
                    if hasattr(config_obj, k):
                        setattr(config_obj, k, v)

# Создаем глобальный конфиг
CONFIG = AppConfig()

# Экспортируем конфигурации для удобства
DB_CONFIG = CONFIG.database
DETECTION_CONFIG = CONFIG.detection
ANALYSIS_CONFIG = CONFIG.analysis
RISK_CONFIG = CONFIG.risk
VISUALIZATION_CONFIG = CONFIG.visualization
MT5_CONFIG = CONFIG.mt5
LOGGING_CONFIG = CONFIG.logging

