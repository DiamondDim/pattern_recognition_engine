"""
Настройка логирования для проекта
"""

import sys
from pathlib import Path
from loguru import logger
from datetime import datetime

from config import LOGGING_CONFIG, LOG_DIR


def setup_logger(name: str = "PRE", level: str = None) -> logger:
    """
    Настройка логгера для проекта

    Args:
        name: Имя логгера
        level: Уровень логирования

    Returns:
        Настроенный логгер
    """

    # Удаляем стандартный обработчик
    logger.remove()

    # Уровень логирования
    log_level = level or LOGGING_CONFIG.LOG_LEVEL

    # Формат для консоли с цветами
    console_format = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan> - "
        "<level>{message}</level>"
    )

    # Формат для файла без цветов
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name}:{function}:{line} - "
        "{message}"
    )

    # Консольный вывод
    logger.add(
        sys.stderr,
        format=console_format,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )

    # Файловый вывод
    logger.add(
        LOGGING_CONFIG.LOG_FILE,
        format=file_format,
        level=log_level,
        rotation=LOGGING_CONFIG.ROTATION,
        retention=LOGGING_CONFIG.RETENTION,
        compression=LOGGING_CONFIG.COMPRESSION,
        backtrace=True,
        diagnose=True
    )

    # Отдельный лог для ошибок
    error_log_file = LOG_DIR / f"errors_{datetime.now().strftime('%Y%m')}.log"
    logger.add(
        error_log_file,
        format=file_format,
        level="ERROR",
        rotation="50 MB",
        retention="90 days",
        backtrace=True,
        diagnose=True
    )

    # Отдельный лог для паттернов
    patterns_log_file = LOG_DIR / "patterns_detected.log"
    logger.add(
        patterns_log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        level="INFO",
        filter=lambda record: "pattern" in record["message"].lower(),
        rotation="10 MB",
        retention="30 days"
    )

    return logger.bind(name=name)


class PatternLogger:
    """Специализированный логгер для паттернов"""

    def __init__(self, main_logger):
        self.logger = main_logger

    def pattern_detected(self, pattern_name: str, symbol: str,
                         timeframe: str, confidence: float):
        """Логирование обнаруженного паттерна"""
        self.logger.info(
            f"Паттерн обнаружен: {pattern_name} | "
            f"Символ: {symbol} | "
            f"ТФ: {timeframe} | "
            f"Уверенность: {confidence:.2%}"
        )

    def pattern_analysis(self, pattern_name: str, historical_matches: int,
                         success_rate: float, risk_reward: float):
        """Логирование анализа паттерна"""
        self.logger.info(
            f"Анализ паттерна: {pattern_name} | "
            f"Исторические аналоги: {historical_matches} | "
            f"Успешность: {success_rate:.2%} | "
            f"Риск/Прибыль: {risk_reward:.2f}"
        )

    def trading_signal(self, symbol: str, pattern: str, direction: str,
                       entry: float, stop_loss: float, take_profit: float):
        """Логирование торгового сигнала"""
        self.logger.warning(
            f"ТОРГОВЫЙ СИГНАЛ | "
            f"Символ: {symbol} | "
            f"Паттерн: {pattern} | "
            f"Направление: {direction} | "
            f"Вход: {entry} | "
            f"Стоп: {stop_loss} | "
            f"Тейк: {take_profit}"
        )


# Создаем глобальный логгер
logger = setup_logger()
pattern_logger = PatternLogger(logger)

