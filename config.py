"""
Конфигурация Pattern Recognition Engine
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
import yaml
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

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
    """Конфигурация подключения к MetaTrader 5 с учетом RFD инструментов Alfa-Forex"""
    ENABLED: bool = True
    AUTO_CONNECT: bool = False
    # Путь к терминалу MT5 от Alfa-Forex
    PATH: str = r"C:\Program Files\MetaTrader 5 Alfa-Forex\terminal64.exe"

    # Данные для подключения (из .env для безопасности)
    LOGIN: Optional[int] = field(default_factory=lambda: int(os.getenv('MT5_LOGIN', '0')))
    PASSWORD: Optional[str] = field(default_factory=lambda: os.getenv('MT5_PASSWORD', ''))
    SERVER: Optional[str] = field(default_factory=lambda: os.getenv('MT5_SERVER', 'Alfa-Forex-MT5'))

    # Префикс для RFD инструментов
    SYMBOL_PREFIX: str = "RFD."

    # Основные символы (без префикса, он будет добавляться автоматически)
    SYMBOLS: List[str] = field(default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "USDRUB", "EURGBP"])
    TIMEFRAMES: List[str] = field(default_factory=lambda: ["M1", "M5", "M15", "M30", "H1", "H4", "D1"])

    TIMEOUT: int = 10000
    INPUT_FILE_PATH: str = str(INPUT_DIR / "mt5_data.csv")
    OUTPUT_FILE_PATH: str = str(OUTPUT_DIR / "patterns.json")
    UPDATE_INTERVAL: int = 60  # секунды

    def add_symbol_prefix(self, symbol: str) -> str:
        """Добавление префикса RFD к символу"""
        if not symbol.startswith(self.SYMBOL_PREFIX):
            return f"{self.SYMBOL_PREFIX}{symbol}"
        return symbol

    def remove_symbol_prefix(self, symbol: str) -> str:
        """Удаление префикса RFD из символа"""
        if symbol.startswith(self.SYMBOL_PREFIX):
            return symbol[len(self.SYMBOL_PREFIX):]
        return symbol

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
class RiskManagementConfig:
    """Конфигурация риск-менеджмента"""
    MAX_POSITION_SIZE_PERCENT: float = 2.0  # Макс 2% от депозита на сделку
    MAX_DAILY_LOSS_PERCENT: float = 5.0     # Макс 5% дневного убытка
    STOP_LOSS_ATR_MULTIPLIER: float = 1.5   # SL = ATR * 1.5
    TAKE_PROFIT_RATIO: float = 2.0          # TP/SL = 2:1
    MAX_OPEN_POSITIONS: int = 3            # Макс одновременно открытых позиций
    ENABLE_TRAILING_STOP: bool = True      # Включить трейлинг-стоп
    TRAILING_STOP_ACTIVATION: float = 1.0  # Активация трейлинг-стопа при движении в плюс 1%
    TRAILING_STOP_DISTANCE: float = 0.5    # Дистанция трейлинг-стопа 0.5%

@dataclass
class CONFIG:
    """Основная конфигурация приложения"""
    # Режимы работы
    MODE: str = "file"  # file, socket, websocket, api
    ENVIRONMENT: str = "development"  # development, testing, production

    # Конфигурации подсистем - используем default_factory для каждого dataclass
    DATABASE: DatabaseConfig = field(default_factory=DatabaseConfig)
    MT5: MT5Config = field(default_factory=MT5Config)
    DETECTION: DetectionConfig = field(default_factory=DetectionConfig)
    ML: MLConfig = field(default_factory=MLConfig)
    BACKTESTING: BacktestingConfig = field(default_factory=BacktestingConfig)
    VISUALIZATION: VisualizationConfig = field(default_factory=VisualizationConfig)
    LOGGING: LoggingConfig = field(default_factory=LoggingConfig)
    API: APIConfig = field(default_factory=APIConfig)
    RISK_MANAGEMENT: RiskManagementConfig = field(default_factory=RiskManagementConfig)

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

    def __post_init__(self):
        """Пост-инициализация для создания директорий"""
        # Создаем директории при инициализации
        for path_name, path_value in self.PATHS.items():
            if path_name.endswith("_dir"):
                path_obj = Path(path_value)
                path_obj.mkdir(parents=True, exist_ok=True)

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

        # Создаем экземпляр конфигурации с помощью default_factory
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

        # Проверка MT5 конфигурации
        if self.MT5.ENABLED:
            # Проверка пути к терминалу
            mt5_path = Path(self.MT5.PATH)
            if not mt5_path.exists():
                errors.append(f"Путь к терминалу MT5 не существует: {self.MT5.PATH}")
                errors.append("Убедитесь, что MetaTrader 5 Alfa-Forex установлен по указанному пути")

            # Проверка авторизационных данных
            if self.MT5.LOGIN == 0:
                errors.append("MT5_LOGIN не установлен. Добавьте в .env файл: MT5_LOGIN=ваш_логин")

            if not self.MT5.PASSWORD:
                errors.append("MT5_PASSWORD не установлен. Добавьте в .env файл: MT5_PASSWORD=ваш_пароль")

            if not self.MT5.SERVER:
                errors.append("MT5_SERVER не установлен. Добавьте в .env файл: MT5_SERVER=ваш_сервер")

        # Проверка риск-менеджмента
        if self.RISK_MANAGEMENT.MAX_POSITION_SIZE_PERCENT <= 0 or self.RISK_MANAGEMENT.MAX_POSITION_SIZE_PERCENT > 100:
            errors.append(f"Недопустимый MAX_POSITION_SIZE_PERCENT: {self.RISK_MANAGEMENT.MAX_POSITION_SIZE_PERCENT}%")

        if self.RISK_MANAGEMENT.MAX_DAILY_LOSS_PERCENT <= 0 or self.RISK_MANAGEMENT.MAX_DAILY_LOSS_PERCENT > 100:
            errors.append(f"Недопустимый MAX_DAILY_LOSS_PERCENT: {self.RISK_MANAGEMENT.MAX_DAILY_LOSS_PERCENT}%")

        return errors

# Глобальный экземпляр конфигурации
config = CONFIG()

# Обратная совместимость для старых импортов
DATABASE_CONFIG = config.DATABASE
MT5_CONFIG = config.MT5
DETECTION_CONFIG = config.DETECTION
ML_CONFIG = config.ML
BACKTESTING_CONFIG = config.BACKTESTING
VISUALIZATION_CONFIG = config.VISUALIZATION
LOGGING_CONFIG = config.LOGGING
API_CONFIG = config.API
RISK_MANAGEMENT_CONFIG = config.RISK_MANAGEMENT

# Для удобства - основные переменные
SYMBOLS = config.MT5.SYMBOLS
TIMEFRAMES = config.MT5.TIMEFRAMES
DATABASE_PATH = config.DATABASE.NAME
LOG_LEVEL = config.LOGGING.LEVEL

# Создание дефолтных конфигурационных файлов для разных окружений
def create_default_configs():
    """Создание дефолтных конфигурационных файлов"""
    environments = ["development", "testing", "production"]

    for env in environments:
        # Создаем новый экземпляр CONFIG для каждого окружения
        current_config = CONFIG()
        current_config.ENVIRONMENT = env

        # Настройки для разных окружений
        if env == "production":
            current_config.LOGGING.LEVEL = "WARNING"
            current_config.API.API_KEY_REQUIRED = True
            current_config.DATABASE.TYPE = "postgresql"
            current_config.MT5.AUTO_CONNECT = True

        elif env == "testing":
            current_config.LOGGING.LEVEL = "DEBUG"
            current_config.DETECTION.MIN_PATTERN_QUALITY = 0.5
            current_config.BACKTESTING.INITIAL_BALANCE = 1000.0

        # Сохраняем конфигурацию
        config_file = CONFIG_DIR / f"config_{env}.yaml"
        current_config.save(str(config_file))
        print(f"Создан конфиг для окружения {env}: {config_file}")

# Автоматическое создание конфигов при первом импорте
if not list(CONFIG_DIR.glob("config_*.yaml")):
    create_default_configs()

# Автоматическая валидация конфигурации при импорте
if __name__ != "__main__":
    validation_errors = config.validate()
    if validation_errors:
        print("=" * 60)
        print("КОНФИГУРАЦИОННЫЕ ОШИБКИ:")
        for error in validation_errors:
            print(f"  • {error}")
        print("=" * 60)

        # Для критических ошибок останавливаем запуск
        critical_errors = [e for e in validation_errors if "MT5" in e or "не установлен" in e]
        if critical_errors:
            print("\nКРИТИЧЕСКИЕ ОШИБКИ! Приложение не может быть запущено.")
            print("Создайте .env файл в корне проекта с содержимым:")
            print("=" * 40)
            print("MT5_LOGIN=ваш_логин")
            print("MT5_PASSWORD=ваш_пароль")
            print("MT5_SERVER=ваш_сервер")
            print("=" * 40)
            sys.exit(1)

# Глобальный экземпляр конфигурации
config = CONFIG()

# Обратная совместимость для старых импортов
DATABASE_CONFIG = config.DATABASE
MT5_CONFIG = config.MT5
DETECTION_CONFIG = config.DETECTION
ML_CONFIG = config.ML
BACKTESTING_CONFIG = config.BACKTESTING
BACKTEST_CONFIG = config.BACKTESTING  # Добавили эту строку!
VISUALIZATION_CONFIG = config.VISUALIZATION
LOGGING_CONFIG = config.LOGGING
API_CONFIG = config.API
RISK_MANAGEMENT_CONFIG = config.RISK_MANAGEMENT

# Для удобства - основные переменные
SYMBOLS = config.MT5.SYMBOLS
TIMEFRAMES = config.MT5.TIMEFRAMES
DATABASE_PATH = config.DATABASE.NAME
LOG_LEVEL = config.LOGGING.LEVEL

# Дополнительные переменные для обратной совместимости
ANALYSIS_CONFIG = config.DETECTION  # Алиас для ANALYSIS_CONFIG
PATTERN_DETECTOR_CONFIG = config.DETECTION  # Алиас для PATTERN_DETECTOR_CONFIG
STATISTICS_CONFIG = config.ML  # Алиас для STATISTICS_CONFIG
ML_MODEL_CONFIG = config.ML  # Алиас для ML_MODEL_CONFIG

# Экспорт всех необходимых переменных
__all__ = [
    'config',
    'CONFIG',
    'DatabaseConfig',
    'MT5Config',
    'DetectionConfig',
    'MLConfig',
    'BacktestingConfig',
    'VisualizationConfig',
    'LoggingConfig',
    'APIConfig',
    'RiskManagementConfig',
    'DATABASE_CONFIG',
    'MT5_CONFIG',
    'DETECTION_CONFIG',
    'ML_CONFIG',
    'BACKTESTING_CONFIG',
    'BACKTEST_CONFIG',  # Добавили эту строку!
    'VISUALIZATION_CONFIG',
    'LOGGING_CONFIG',
    'API_CONFIG',
    'RISK_MANAGEMENT_CONFIG',
    'ANALYSIS_CONFIG',
    'PATTERN_DETECTOR_CONFIG',
    'STATISTICS_CONFIG',
    'ML_MODEL_CONFIG',
    'SYMBOLS',
    'TIMEFRAMES',
    'DATABASE_PATH',
    'LOG_LEVEL',
    'PROJECT_ROOT',
    'DATA_DIR',
    'INPUT_DIR',
    'OUTPUT_DIR',
    'MODELS_DIR',
    'LOGS_DIR',
    'CONFIG_DIR'
]

