"""
Модуль для настройки логирования
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

try:
    from config import config
    LOGGING_CONFIG = config.LOGGING
except ImportError:
    # Для обратной совместимости
    from config import LOGGING_CONFIG


def setup_logger(name: str, log_file: Optional[str] = None, level: Optional[str] = None) -> logging.Logger:
    """
    Настройка логгера

    Args:
        name: Имя логгера
        log_file: Путь к файлу лога
        level: Уровень логирования

    Returns:
        Настроенный логгер
    """
    if level is None:
        level = LOGGING_CONFIG.LEVEL
    if log_file is None:
        log_file = LOGGING_CONFIG.FILE

    # Создаем логгер
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Проверяем, не добавлены ли уже обработчики
    if logger.handlers:
        return logger

    # Форматтер
    formatter = logging.Formatter(
        LOGGING_CONFIG.FORMAT,
        datefmt=LOGGING_CONFIG.DATE_FORMAT
    )

    # Обработчик для консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Обработчик для файла
    if log_file:
        # Создаем директорию для логов если ее нет
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=LOGGING_CONFIG.MAX_SIZE_MB * 1024 * 1024,
            backupCount=LOGGING_CONFIG.BACKUP_COUNT
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Глобальный логгер по умолчанию
logger = setup_logger("pattern_recognition_engine")

