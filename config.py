"""
Конфигурация Pattern Recognition Engine
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
import yaml

# Базовые пути проекта
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"

# Создание директорий если их нет
for directory in [DATA_DIR, INPUT_DIR, OUTPUT_DIR, MODELS_DIR, LOGS_DIR, CONFIG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

@dataclass
class DatabaseConfig:
    """Конфигурация базы данных"""
    TYPE: str = "sqlite"  # sqlite, postgresql, mysql
    NAME: str = str(PROJECT_ROOT / "patterns.db")
    HOST: str = "localhost"
    PORT: int = 5432
    USERNAME: Optional[str] = None
    PASSWORD: Optional[str] = None
    POOL_SIZE: int = 10
    MAX_OVERFLOW: int = 20

    @property
    def url(self) -> str:
        """Получение URL для подключения к БД"""
        if self.TYPE == "sqlite":
            return f"sqlite:///{self.NAME}"
        elif self.TYPE == "postgresql":
            auth = f"{self.USERNAME}:{self.PASSWORD}@" if self.USERNAME and self.PASSWORD else ""
            return f"postgresql://{auth}{self.HOST}:{self.PORT}/{self.NAME}"
        elif self.TYPE == "mysql":
            auth = f"{self.USERNAME}:{self.PASSWORD}@" if self.USERNAME and self.PASSWORD else ""
            return f"mysql://{auth}{self.HOST}:{self.PORT}/{self.NAME}"
        return ""

@dataclass
class MT5Config:
    """Конфигурация подключения к MetaTrader 5"""
    ENABLED: bool = True
    AUTO_CONNECT: bool = False
    PATH: str = "C:/Program Files/MetaTrader 5/terminal64.exe"
    LOGIN: Optional[int] = None
    PASSWORD: Optional[str] = None
    SERVER: Optional[str] = None
    TIMEOUT: int = 10000
    SYMBOLS: List[str] = field(default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY"])
    TIMEFRAMES: List[str] = field(default_factory=lambda: ["M15", "H1", "H4", "D1"])
    INPUT_FILE_PATH: str = str(INPUT_DIR / "mt5_data.csv")
    OUTPUT_FILE_PATH: str = str(OUTPUT_DIR / "patterns.json")
    UPDATE_INTERVAL: int = 60  # секунды

@dataclass
class DetectionConfig:
    """Конфигурация детектирования паттернов"""
    ENABLE_GEOMETRIC: bool = True
    ENABLE_CANDLESTICK: bool = True
    ENABLE_HARMONIC: bool = True

    # Параметры качества
    MIN_PATTERN_QUALITY: float = 0.6
    CONFIDENCE_THRESHOLD: float = 0.7
    PRICE_TOLERANCE_PCT: float = 0.002  # 0.2%
    SYMMETRY_TOLERANCE: float = 0.15  # 15%

    # Лимиты
    MIN_CANDLES_FOR_PATTERN: int = 5
    MAX_CANDLES_FOR_PATTERN: int = 100
    MAX_PATTERNS_PER_SYMBOL: int = 50

    # Гармонические паттерны
    FIBONACCI_TOLERANCE: float = 0.05  # 5%
    MIN_HARMONIC_POINTS: int = 4

@dataclass
class MLConfig:
    """Конфигурация ML моделей"""
    ENABLED: bool = True
    MODEL_TYPE: str = "random_forest"  # random_forest, gradient_boosting, svm, neural_network
    TRAIN_INTERVAL: str = "daily"  # daily, weekly, monthly
    RETRAIN_THRESHOLD: float = 0.1  # 10% дрейф данных
    ENSEMBLE_WEIGHTING: str = "performance"  # performance, equal, adaptive

    # Параметры моделей
    RANDOM_FOREST_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    })

    NEURAL_NETWORK_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_layer_sizes": (64, 32),
        "activation": "relu",
        "solver": "adam",
        "max_iter": 1000,
        "random_state": 42
    })

    # Feature engineering
    FEATURE_WINDOW: int = 20
    USE_TECHNICAL_INDICATORS: bool = True
    USE_PATTERN_FEATURES: bool = True

@dataclass
class BacktestingConfig:
    """Конфигурация бэктестинга"""
    INITIAL_BALANCE: float = 10000.0
    RISK_PER_TRADE: float = 0.02  # 2% от баланса
    COMMISSION: float = 0.0001  # 0.01%
    SLIPPAGE: float = 0.0001  # 0.01%
    MAX_HOLDING_PERIOD: int = 100  # свечей
    MIN_TRADES_FOR_VALIDATION: int = 20

    # Критерии оптимизации
    OPTIMIZATION_METRIC: str = "sharpe_ratio"  # sharpe_ratio, profit_factor, max_drawdown
    WALK_FORWARD_WINDOW: int = 500
    WALK_FORWARD_STEP: int = 100

    # Monte Carlo
    MONTE_CARLO_SIMULATIONS: int = 1000
    CONFIDENCE_LEVEL: float = 0.95

@dataclass
class VisualizationConfig:
    """Конфигурация визуализации"""
    ENABLE_PLOTTING: bool = True
    SAVE_PLOTS: bool = True
    PLOT_FORMAT: str = "png"  # png, svg, pdf, html
    PLOT_DPI: int = 150
    INTERACTIVE_PLOTS: bool = False

    # Цветовая схема
    COLORS: Dict[str, str] = field(default_factory=lambda: {
        "bullish": "#26A69A",
        "bearish": "#EF5350",
        "neutral": "#78909C",
        "support": "#42A5F5",
        "resistance": "#FF7043",
        "volume_up": "#4CAF50",
        "volume_down": "#F44336",
        "background": "#FFFFFF",
        "grid": "#E0E0E0"
    })

    # Размеры графиков
    FIGURE_WIDTH: int = 15
    FIGURE_HEIGHT: int = 8
    SUBPLOT_HEIGHT_RATIO: List[int] = field(default_factory=lambda: [3, 1])

@dataclass
class LoggingConfig:
    """Конфигурация логирования"""
    LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    FILE: str = str(LOGS_DIR / "pre_engine.log")
    MAX_SIZE_MB: int = 100
    BACKUP_COUNT: int = 5
    FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

    # Мониторинг
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    HEALTH_CHECK_INTERVAL: int = 60  # секунды

@dataclass
class APIConfig:
    """Конфигурация API"""
    ENABLED: bool = False
    HOST: str = "127.0.0.1"
    PORT: int = 8080
    WORKERS: int = 4
    API_KEY_REQUIRED: bool = False
    RATE_LIMIT_PER_MINUTE: int = 60
    CORS_ORIGINS: List[str] = field(default_factory=lambda: ["http://localhost:3000"])

    # WebSocket
    WEBSOCKET_ENABLED: bool = False
    WEBSOCKET_PORT: int = 8765

@dataclass
class CONFIG:
    """Основная конфигурация приложения"""

    # Режимы работы
    MODE: str = "file"  # file, socket, websocket, api
    ENVIRONMENT: str = "development"  # development, testing, production

    # Конфигурации подсистем
    DATABASE: DatabaseConfig = DatabaseConfig()
    MT5: MT5Config = MT5Config()
    DETECTION: DetectionConfig = DetectionConfig()
    ML: MLConfig = MLConfig()
    BACKTESTING: BacktestingConfig = BacktestingConfig()
    VISUALIZATION: VisualizationConfig = VisualizationConfig()
    LOGGING: LoggingConfig = LoggingConfig()
    API: APIConfig = APIConfig()

    # Производительность
    MAX_WORKERS: int = 4
    USE_CACHE: bool = True
    CACHE_SIZE: int = 1000
    USE_NUMBA: bool = True

    # Пути
    PATHS: Dict[str, str] = field(default_factory=lambda: {
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(DATA_DIR),
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "models_dir": str(MODELS_DIR),
        "logs_dir": str(LOGS_DIR),
        "config_dir": str(CONFIG_DIR)
    })

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация конфигурации в словарь"""
        return asdict(self)

    def save(self, filepath: Optional[str] = None):
        """Сохранение конфигурации в файл"""
        if filepath is None:
            filepath = CONFIG_DIR / f"config_{self.ENVIRONMENT}.yaml"

        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def load(cls, filepath: str):
        """Загрузка конфигурации из файла"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        # Создаем экземпляр конфигурации
        config = cls()

        # Обновляем значения из файла
        for key, value in config_dict.items():
            if hasattr(config, key):
                current_value = getattr(config, key)
                if isinstance(current_value, dict) and isinstance(value, dict):
                    # Рекурсивное обновление словарей
                    for sub_key, sub_value in value.items():
                        if sub_key in current_value:
                            current_value[sub_key] = sub_value
                else:
                    setattr(config, key, value)

        return config

    def validate(self) -> List[str]:
        """Валидация конфигурации"""
        errors = []

        # Проверка путей
        for path_name, path_value in self.PATHS.items():
            if path_name.endswith("_dir"):
                path_obj = Path(path_value)
                if not path_obj.exists():
                    try:
                        path_obj.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        errors.append(f"Не удалось создать директорию {path_name}: {e}")

        # Проверка режимов
        if self.MODE not in ["file", "socket", "websocket", "api"]:
            errors.append(f"Недопустимый режим работы: {self.MODE}")

        # Проверка окружения
        if self.ENVIRONMENT not in ["development", "testing", "production"]:
            errors.append(f"Недопустимое окружение: {self.ENVIRONMENT}")

        # Проверка ML конфигурации
        if self.ML.ENABLED and self.ML.MODEL_TYPE not in ["random_forest", "gradient_boosting", "svm", "neural_network"]:
            errors.append(f"Недопустимый тип ML модели: {self.ML.MODEL_TYPE}")

        # Проверка портов
        if self.API.ENABLED:
            if self.API.PORT < 1024 or self.API.PORT > 65535:
                errors.append(f"Недопустимый порт API: {self.API.PORT}")

        if self.LOGGING.ENABLE_METRICS:
            if self.LOGGING.METRICS_PORT < 1024 or self.LOGGING.METRICS_PORT > 65535:
                errors.append(f"Недопустимый порт метрик: {self.LOGGING.METRICS_PORT}")

        return errors

# Глобальный экземпляр конфигурации
config = CONFIG()

# Создание дефолтных конфигурационных файлов для разных окружений
def create_default_configs():
    """Создание дефолтных конфигурационных файлов"""
    environments = ["development", "testing", "production"]

    for env in environments:
        config = CONFIG()
        config.ENVIRONMENT = env

        # Настройки для разных окружений
        if env == "production":
            config.LOGGING.LEVEL = "WARNING"
            config.API.API_KEY_REQUIRED = True
            config.DATABASE.TYPE = "postgresql"
            config.MT5.AUTO_CONNECT = True

        elif env == "testing":
            config.LOGGING.LEVEL = "DEBUG"
            config.DETECTION.MIN_PATTERN_QUALITY = 0.5
            config.BACKTESTING.INITIAL_BALANCE = 1000.0

        # Сохраняем конфигурацию
        config_file = CONFIG_DIR / f"config_{env}.yaml"
        config.save(str(config_file))
        print(f"Создан конфиг для окружения {env}: {config_file}")

# Автоматическое создание конфигов при первом импорте
if not list(CONFIG_DIR.glob("config_*.yaml")):
    create_default_configs()

