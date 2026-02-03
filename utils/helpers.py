"""
Вспомогательные утилиты для Pattern Recognition Engine
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import json


def validate_data(data: pd.DataFrame, required_columns: List[str] = None) -> Tuple[bool, str]:
    """
    Валидация данных DataFrame

    Args:
        data: DataFrame для валидации
        required_columns: Список обязательных колонок

    Returns:
        Кортеж (успех, сообщение об ошибке)
    """
    if data is None or data.empty:
        return False, "DataFrame пустой или None"

    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False, f"Отсутствуют колонки: {missing_columns}"

    # Проверка на NaN/Inf значения
    if data.isnull().any().any():
        nan_count = data.isnull().sum().sum()
        return False, f"Найдено {nan_count} NaN значений"

    # Проверка корректности цен (high >= low, high >= close, low <= close)
    if 'high' in data.columns and 'low' in data.columns:
        invalid_high_low = data[data['high'] < data['low']]
        if not invalid_high_low.empty:
            return False, f"Найдено {len(invalid_high_low)} строк где high < low"

    return True, "Данные валидны"


def calculate_metrics(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Расчет метрик данных

    Args:
        data: DataFrame с данными

    Returns:
        Словарь с метриками
    """
    if data is None or data.empty:
        return {
            'total_rows': 0,
            'total_columns': 0,
            'date_range': None,
            'price_stats': {},
            'volume_stats': {}
        }

    metrics = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'date_range': None,
        'price_stats': {},
        'volume_stats': {}
    }

    if len(data) > 0:
        metrics['date_range'] = {
            'start': data.index[0].isoformat() if hasattr(data.index[0], 'isoformat') else str(data.index[0]),
            'end': data.index[-1].isoformat() if hasattr(data.index[-1], 'isoformat') else str(data.index[-1])
        }

    # Статистика по ценам
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in data.columns:
            metrics['price_stats'][col] = {
                'min': float(data[col].min()),
                'max': float(data[col].max()),
                'mean': float(data[col].mean()),
                'std': float(data[col].std())
            }

    # Статистика по объемам
    if 'volume' in data.columns:
        metrics['volume_stats'] = {
            'min': int(data['volume'].min()),
            'max': int(data['volume'].max()),
            'mean': float(data['volume'].mean()),
            'std': float(data['volume'].std())
        }

    # Дополнительные метрики
    if 'close' in data.columns and len(data) > 1:
        returns = data['close'].pct_change().dropna()
        if len(returns) > 0:
            metrics['returns'] = {
                'mean': float(returns.mean()),
                'std': float(returns.std()),
                'min': float(returns.min()),
                'max': float(returns.max())
            }

    return metrics


def format_datetime(dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Форматирование datetime в строку

    Args:
        dt: datetime объект
        fmt: Формат строки

    Returns:
        Отформатированная строка
    """
    if dt is None:
        return ""
    return dt.strftime(fmt)


def parse_datetime(dt_str: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> Optional[datetime]:
    """
    Парсинг строки в datetime

    Args:
        dt_str: Строка с датой
        fmt: Формат строки

    Returns:
        datetime объект или None
    """
    try:
        return datetime.strptime(dt_str, fmt)
    except (ValueError, TypeError):
        return None


def generate_hash(data: Any) -> str:
    """
    Генерация хеша для данных

    Args:
        data: Любые данные

    Returns:
        SHA256 хеш в виде строки
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data_str = data.to_json()
    elif isinstance(data, dict):
        data_str = json.dumps(data, sort_keys=True)
    else:
        data_str = str(data)

    return hashlib.sha256(data_str.encode()).hexdigest()

