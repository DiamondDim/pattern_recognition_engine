"""
Вспомогательные функции для проекта
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import json
import hashlib
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

from config import CONFIG, DATA_DIR
from utils.logger import logger


def generate_id(prefix: str = "pattern") -> str:
    """
    Генерация уникального ID

    Args:
        prefix: Префикс для ID

    Returns:
        Уникальный ID
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    random_str = hashlib.md5(timestamp.encode()).hexdigest()[:8]
    return f"{prefix}_{timestamp}_{random_str}"


def validate_ohlc_data(data: Dict[str, np.ndarray]) -> bool:
    """
    Валидация данных OHLC

    Args:
        data: Словарь с данными

    Returns:
        True если данные валидны
    """
    required_keys = ['open', 'high', 'low', 'close']

    # Проверяем наличие всех ключей
    for key in required_keys:
        if key not in data:
            logger.error(f"Отсутствует ключ: {key}")
            return False

    # Проверяем длины массивов
    lengths = {key: len(data[key]) for key in required_keys}
    if len(set(lengths.values())) > 1:
        logger.error(f"Несовпадение длин массивов: {lengths}")
        return False

    # Проверяем корректность значений
    for key in required_keys:
        array = data[key]

        # Проверяем на NaN и Inf
        if np.any(np.isnan(array)) or np.any(np.isinf(array)):
            logger.error(f"Некорректные значения в {key}")
            return False

        # Для high, low, close проверяем, что high >= low и high >= close >= low
        if key == 'high':
            highs = array
            lows = data['low']
            closes = data['close']

            if np.any(highs < lows):
                logger.error("Найдены high < low")
                return False

            if np.any(closes > highs) or np.any(closes < lows):
                logger.error("Найдены close вне диапазона high-low")
                return False

    return True


def normalize_prices(prices: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Нормализация цен

    Args:
        prices: Массив цен
        method: Метод нормализации ('minmax', 'zscore', 'log')

    Returns:
        Нормализованный массив
    """
    if len(prices) == 0:
        return prices

    if method == 'minmax':
        min_val = np.min(prices)
        max_val = np.max(prices)

        if max_val - min_val == 0:
            return np.zeros_like(prices)

        return (prices - min_val) / (max_val - min_val)

    elif method == 'zscore':
        mean = np.mean(prices)
        std = np.std(prices)

        if std == 0:
            return np.zeros_like(prices)

        return (prices - mean) / std

    elif method == 'log':
        # Логарифмическое преобразование
        log_prices = np.log(prices + 1e-10)  # Добавляем небольшое значение для избежания log(0)
        return (log_prices - np.mean(log_prices)) / np.std(log_prices)

    else:
        logger.warning(f"Неизвестный метод нормализации: {method}, используем minmax")
        return normalize_prices(prices, method='minmax')


def calculate_returns(prices: np.ndarray, period: int = 1) -> np.ndarray:
    """
    Расчет доходностей

    Args:
        prices: Массив цен
        period: Период для расчета

    Returns:
        Массив доходностей
    """
    if len(prices) <= period:
        return np.array([])

    returns = (prices[period:] / prices[:-period]) - 1
    return returns


def calculate_volatility(prices: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Расчет волатильности (стандартное отклонение доходностей)

    Args:
        prices: Массив цен
        window: Окно для расчета

    Returns:
        Массив волатильности
    """
    if len(prices) <= window:
        return np.array([])

    returns = calculate_returns(prices, period=1)

    if len(returns) < window:
        return np.array([])

    volatility = np.zeros(len(prices))
    volatility[:window] = np.nan

    for i in range(window, len(prices)):
        start = max(0, i - window)
        vol = np.std(returns[start:i])
        volatility[i] = vol

    return volatility


def detect_trend(prices: np.ndarray, method: str = 'linear') -> Tuple[float, str]:
    """
    Определение тренда

    Args:
        prices: Массив цен
        method: Метод определения ('linear', 'ma', 'adx')

    Returns:
        Tuple(сила тренда, направление)
    """
    if len(prices) < 10:
        return 0.0, 'neutral'

    if method == 'linear':
        # Линейная регрессия
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)

        # Наклон в процентах от средней цены
        avg_price = np.mean(prices)
        slope_pct = (slope / avg_price) * 100 if avg_price != 0 else 0

        # Определяем направление
        if slope_pct > 0.1:  # 0.1% на свечу
            direction = 'uptrend'
            strength = min(abs(slope_pct) / 1.0, 1.0)  # Нормализуем к 0-1
        elif slope_pct < -0.1:
            direction = 'downtrend'
            strength = min(abs(slope_pct) / 1.0, 1.0)
        else:
            direction = 'sideways'
            strength = 0.0

        return strength, direction

    elif method == 'ma':
        # По скользящим средним
        if len(prices) < 50:
            return 0.0, 'neutral'

        ma_short = np.mean(prices[-20:])
        ma_long = np.mean(prices[-50:])

        ma_ratio = ma_short / ma_long - 1

        if ma_ratio > 0.01:  # 1% выше
            direction = 'uptrend'
            strength = min(ma_ratio / 0.05, 1.0)  # Нормализуем к 5%
        elif ma_ratio < -0.01:
            direction = 'downtrend'
            strength = min(abs(ma_ratio) / 0.05, 1.0)
        else:
            direction = 'sideways'
            strength = 0.0

        return strength, direction

    else:
        logger.warning(f"Неизвестный метод определения тренда: {method}")
        return 0.0, 'neutral'


def find_support_resistance(prices: np.ndarray,
                            window: int = 20,
                            tolerance: float = 0.01) -> Tuple[List[float], List[float]]:
    """
    Поиск уровней поддержки и сопротивления

    Args:
        prices: Массив цен
        window: Окно для поиска
        tolerance: Допуск в процентах

    Returns:
        Tuple(список поддержек, список сопротивлений)
    """
    if len(prices) < window * 2:
        return [], []

    supports = []
    resistances = []

    # Ищем локальные минимумы и максимумы
    for i in range(window, len(prices) - window):
        local_min = np.min(prices[i - window:i + window + 1])
        local_max = np.max(prices[i - window:i + window + 1])

        # Если цена в центре окна - локальный минимум
        if abs(prices[i] - local_min) / local_min < 0.001:  # 0.1%
            supports.append(prices[i])

        # Если цена в центре окна - локальный максимум
        if abs(prices[i] - local_max) / local_max < 0.001:  # 0.1%
            resistances.append(prices[i])

    # Объединяем близкие уровни
    supports = _merge_close_levels(supports, tolerance)
    resistances = _merge_close_levels(resistances, tolerance)

    return supports, resistances


def _merge_close_levels(levels: List[float], tolerance: float) -> List[float]:
    """Объединение близких уровней"""
    if not levels:
        return []

    levels = sorted(levels)
    merged = []

    current = levels[0]
    current_count = 1

    for level in levels[1:]:
        if abs(level - current) / current <= tolerance:
            # Объединяем близкие уровни (взвешенное среднее)
            current = (current * current_count + level) / (current_count + 1)
            current_count += 1
        else:
            merged.append(current)
            current = level
            current_count = 1

    merged.append(current)
    return merged


def calculate_fibonacci_levels(high: float, low: float) -> Dict[str, float]:
    """
    Расчет уровней Фибоначчи

    Args:
        high: Высокая цена
        low: Низкая цена

    Returns:
        Словарь уровней Фибоначчи
    """
    diff = high - low

    levels = {
        '0.0': low,
        '0.236': low + diff * 0.236,
        '0.382': low + diff * 0.382,
        '0.5': low + diff * 0.5,
        '0.618': low + diff * 0.618,
        '0.786': low + diff * 0.786,
        '1.0': high,
        '1.272': high + diff * 0.272,
        '1.618': high + diff * 0.618
    }

    return levels


def calculate_pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
    """
    Расчет пивот-поинтов

    Args:
        high: Высокая цена предыдущего периода
        low: Низкая цена предыдущего периода
        close: Цена закрытия предыдущего периода

    Returns:
        Словарь пивот-поинтов
    """
    pivot = (high + low + close) / 3

    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)

    return {
        'pivot': pivot,
        'r1': r1, 'r2': r2, 'r3': r3,
        's1': s1, 's2': s2, 's3': s3
    }


def smooth_data(data: np.ndarray, method: str = 'sma', window: int = 3) -> np.ndarray:
    """
    Сглаживание данных

    Args:
        data: Массив данных
        method: Метод сглаживания ('sma', 'ema', 'median')
        window: Окно сглаживания

    Returns:
        Сглаженный массив
    """
    if len(data) < window:
        return data.copy()

    if method == 'sma':
        # Простая скользящая средняя
        smoothed = np.zeros_like(data)
        for i in range(len(data)):
            start = max(0, i - window + 1)
            smoothed[i] = np.mean(data[start:i + 1])

        return smoothed

    elif method == 'ema':
        # Экспоненциальная скользящая средняя
        alpha = 2 / (window + 1)
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]

        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]

        return smoothed

    elif method == 'median':
        # Медианный фильтр
        smoothed = np.zeros_like(data)
        half_window = window // 2

        for i in range(len(data)):
            start = max(0, i - half_window)
            end = min(len(data), i + half_window + 1)
            smoothed[i] = np.median(data[start:end])

        return smoothed

    else:
        logger.warning(f"Неизвестный метод сглаживания: {method}")
        return data.copy()


def detect_outliers(data: np.ndarray, method: str = 'iqr', threshold: float = 1.5) -> np.ndarray:
    """
    Обнаружение выбросов

    Args:
        data: Массив данных
        method: Метод обнаружения ('iqr', 'zscore', 'mad')
        threshold: Порог

    Returns:
        Массив индексов выбросов
    """
    if len(data) < 10:
        return np.array([], dtype=int)

    if method == 'iqr':
        # Метод межквартильного размаха
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        outliers = np.where((data < lower_bound) | (data > upper_bound))[0]

    elif method == 'zscore':
        # Z-score метод
        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return np.array([], dtype=int)

        z_scores = np.abs((data - mean) / std)
        outliers = np.where(z_scores > threshold)[0]

    elif method == 'mad':
        # Median Absolute Deviation
        median = np.median(data)
        mad = np.median(np.abs(data - median))

        if mad == 0:
            return np.array([], dtype=int)

        modified_z_scores = 0.6745 * np.abs(data - median) / mad
        outliers = np.where(modified_z_scores > threshold)[0]

    else:
        logger.warning(f"Неизвестный метод обнаружения выбросов: {method}")
        outliers = np.array([], dtype=int)

    return outliers


def calculate_correlation(series1: np.ndarray, series2: np.ndarray) -> float:
    """
    Расчет корреляции между двумя рядами

    Args:
        series1: Первый ряд
        series2: Второй ряд

    Returns:
        Коэффициент корреляции
    """
    if len(series1) != len(series2):
        logger.error(f"Разные длины рядов: {len(series1)} != {len(series2)}")
        return 0.0

    if len(series1) < 2:
        return 0.0

    # Удаляем NaN значения
    mask = ~(np.isnan(series1) | np.isnan(series2))
    series1_clean = series1[mask]
    series2_clean = series2[mask]

    if len(series1_clean) < 2:
        return 0.0

    # Расчет корреляции Пирсона
    correlation = np.corrcoef(series1_clean, series2_clean)[0, 1]

    return correlation if not np.isnan(correlation) else 0.0


def resample_data(data: Dict[str, np.ndarray],
                  original_freq: str,
                  target_freq: str) -> Dict[str, np.ndarray]:
    """
    Ресемплинг данных OHLC

    Args:
        data: Данные OHLC
        original_freq: Исходная частота (например, '1min')
        target_freq: Целевая частота (например, '5min')

    Returns:
        Ресемплированные данные
    """
    # TODO: Реализовать ресемплинг данных
    # Для этого нужно конвертировать данные в pandas DataFrame,
    # выполнить ресемплинг и конвертировать обратно в numpy arrays

    logger.warning("Ресемплинг данных пока не реализован")
    return data.copy()


def save_data_to_file(data: Dict[str, Any],
                      filename: str,
                      format: str = 'json') -> bool:
    """
    Сохранение данных в файл

    Args:
        data: Данные для сохранения
        filename: Имя файла
        format: Формат файла ('json', 'csv', 'parquet')

    Returns:
        True если успешно сохранено
    """
    try:
        filepath = DATA_DIR / filename

        if format == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)

        elif format == 'csv':
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                df = data

            df.to_csv(filepath, index=False)

        elif format == 'parquet':
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                df = data

            df.to_parquet(filepath)

        else:
            logger.error(f"Неизвестный формат: {format}")
            return False

        logger.debug(f"Данные сохранены в {filepath}")
        return True

    except Exception as e:
        logger.error(f"Ошибка сохранения данных: {e}")
        return False


def load_data_from_file(filename: str,
                        format: str = 'json') -> Optional[Dict[str, Any]]:
    """
    Загрузка данных из файла

    Args:
        filename: Имя файла
        format: Формат файла

    Returns:
        Загруженные данные или None при ошибке
    """
    try:
        filepath = DATA_DIR / filename

        if not filepath.exists():
            logger.error(f"Файл не найден: {filepath}")
            return None

        if format == 'json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

        elif format == 'csv':
            df = pd.read_csv(filepath)
            data = df.to_dict('list')

        elif format == 'parquet':
            df = pd.read_parquet(filepath)
            data = df.to_dict('list')

        else:
            logger.error(f"Неизвестный формат: {format}")
            return None

        logger.debug(f"Данные загружены из {filepath}")
        return data

    except Exception as e:
        logger.error(f"Ошибка загрузки данных: {e}")
        return None


def calculate_statistics(data: np.ndarray) -> Dict[str, float]:
    """
    Расчет статистики данных

    Args:
        data: Массив данных

    Returns:
        Словарь статистики
    """
    if len(data) == 0:
        return {}

    # Удаляем NaN значения
    clean_data = data[~np.isnan(data)]

    if len(clean_data) == 0:
        return {}

    stats = {
        'count': len(clean_data),
        'mean': float(np.mean(clean_data)),
        'median': float(np.median(clean_data)),
        'std': float(np.std(clean_data)),
        'min': float(np.min(clean_data)),
        'max': float(np.max(clean_data)),
        'q1': float(np.percentile(clean_data, 25)),
        'q3': float(np.percentile(clean_data, 75)),
        'skewness': float(pd.Series(clean_data).skew()),
        'kurtosis': float(pd.Series(clean_data).kurtosis())
    }

    return stats


def format_price(price: float, symbol: str = '') -> str:
    """
    Форматирование цены

    Args:
        price: Цена
        symbol: Символ инструмента

    Returns:
        Отформатированная строка
    """
    # Определяем количество знаков после запятой на основе символа
    if any(forex in symbol.upper() for forex in ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']):
        decimals = 5  # Форекс
    elif 'XAU' in symbol.upper() or 'GOLD' in symbol.upper():
        decimals = 2  # Золото
    elif 'BTC' in symbol.upper() or 'ETH' in symbol.upper():
        decimals = 2  # Криптовалюты
    else:
        decimals = 2  # По умолчанию

    return f"{price:.{decimals}f}"


def format_percentage(value: float) -> str:
    """Форматирование процента"""
    return f"{value:.2%}"


def get_current_time() -> str:
    """Получение текущего времени в формате строки"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def time_ago(timestamp: datetime) -> str:
    """
    Форматирование времени в формате "сколько времени назад"

    Args:
        timestamp: Временная метка

    Returns:
        Строка в формате "X минут назад"
    """
    now = datetime.now()
    diff = now - timestamp

    if diff.days > 365:
        years = diff.days // 365
        return f"{years} год(а) назад"
    elif diff.days > 30:
        months = diff.days // 30
        return f"{months} месяц(ев) назад"
    elif diff.days > 0:
        return f"{diff.days} день(дней) назад"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} час(а) назад"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes} минут(ы) назад"
    else:
        return "только что"

