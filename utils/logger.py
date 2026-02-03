"""
Модуль логирования
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import structlog
from datetime import datetime

from config import config

def setup_logger(level: str = None, log_file: str = None) -> structlog.BoundLogger:
    """
    Настройка системы логирования

    Args:
        level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Путь к файлу логов

    Returns:
        Настроенный логгер
    """
    if level is None:
        level = config.LOGGING.LEVEL

    if log_file is None:
        log_file = config.LOGGING.FILE

    # Создаем директорию для логов если её нет
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Настраиваем structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Настраиваем стандартное логирование
    handler = logging.StreamHandler(sys.stdout)

    # Добавляем файловый handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')

    # Форматирование
    formatter = logging.Formatter(
        fmt=config.LOGGING.FORMAT,
        datefmt=config.LOGGING.DATE_FORMAT
    )

    handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Получаем root логгер
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))

    # Удаляем существующие handlers
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)

    # Добавляем наши handlers
    logger.addHandler(handler)
    logger.addHandler(file_handler)

    # Создаем bound logger
    bound_logger = structlog.get_logger()

    return bound_logger

# Глобальный логгер
logger = setup_logger()

