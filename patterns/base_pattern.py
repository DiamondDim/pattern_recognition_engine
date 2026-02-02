"""
Базовый класс для всех паттернов
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field
from uuid import uuid4

class PatternType(str, Enum):
    """Типы паттернов"""
    GEOMETRIC = "geometric"
    CANDLESTICK = "candlestick"
    HARMONIC = "harmonic"
    CUSTOM = "custom"

class PatternDirection(str, Enum):
    """Направление паттерна"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

class MarketContext(str, Enum):
    """Контекст рынка"""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"

class PatternPoint(BaseModel):
    """Точка паттерна"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    index: int
    timestamp: datetime
    price: float
    point_type: str  # high, low, neckline, shoulder, head, etc.
    significance: float = 1.0  # Важность точки (0-1)

    class Config:
        arbitrary_types_allowed = True

class PatternMetadata(BaseModel):
    """Метаданные паттерна"""
    pattern_id: str = Field(default_factory=lambda: str(uuid4()))
    symbol: str
    timeframe: str
    detected_time: datetime = Field(default_factory=datetime.now)
    market_context: MarketContext = MarketContext.SIDEWAYS

    # Оценки
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reliability_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # Подтверждения
    volume_confirmation: bool = False
    trend_confirmation: bool = False
    indicator_confirmation: bool = False
    multi_timeframe_confirmation: bool = False

    # Индикаторы в момент обнаружения
    rsi_value: Optional[float] = None
    macd_value: Optional[float] = None
    adx_value: Optional[float] = None
    atr_value: Optional[float] = None

    # Волатильность
    volatility_pct: float = 0.0
    average_volume: float = 0.0

    class Config:
        arbitrary_types_allowed = True

class PatternTargets(BaseModel):
    """Целевые уровни паттерна"""
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    profit_risk_ratio: Optional[float] = None

    # Дополнительные уровни
    breakout_level: Optional[float] = None  # Уровень пробоя
    target1: Optional[float] = None  # Первая цель
    target2: Optional[float] = None  # Вторая цель
    target3: Optional[float] = None  # Третья цель

    # Расстояния
    pattern_height: Optional[float] = None
    risk_amount: Optional[float] = None
    reward_amount: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True

class PatternStatistics(BaseModel):
    """Статистика паттерна"""
    historical_matches: int = 0
    historical_success_rate: float = 0.0
    avg_holding_period: float = 0.0  # Средний период удержания
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    sharpe_ratio: Optional[float] = None
    win_loss_ratio: Optional[float] = None
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    profit_factor: Optional[float] = None

    # Распределение результатов
    outcomes_distribution: Dict[str, float] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

class BasePattern(ABC):
    """Абстрактный базовый класс для паттернов"""

    def __init__(self,
                 pattern_type: PatternType,
                 name: str,
                 abbreviation: str = ""):

        self.pattern_type = pattern_type
        self.name = name
        self.abbreviation = abbreviation or name[:3].upper()
        self.direction = PatternDirection.NEUTRAL

        # Структура паттерна
        self.points: List[PatternPoint] = []
        self.metadata = PatternMetadata(symbol="", timeframe="")
        self.targets = PatternTargets()
        self.statistics = PatternStatistics()

        # Временные данные
        self._data: Optional[np.ndarray] = None
        self._highs: Optional[np.ndarray] = None
        self._lows: Optional[np.ndarray] = None
        self._closes: Optional[np.ndarray] = None
        self._volumes: Optional[np.ndarray] = None

        # Флаги состояния
        self._is_detected = False
        self._is_confirmed = False
        self._is_completed = False

        # Характеристики паттерна
        self.complexity_level = 1  # Сложность (1-3)
        self.frequency_score = 0.0  # Частота появления
        self.success_rate = 0.0  # Историческая успешность

    @abstractmethod
    def detect(self,
               data: np.ndarray,
               highs: np.ndarray,
               lows: np.ndarray,
               closes: np.ndarray,
               volumes: np.ndarray,
               timestamps: np.ndarray,
               **kwargs) -> bool:
        """
        Детектирование паттерна

        Returns:
            bool: True если паттерн обнаружен
        """
        pass

    @abstractmethod
    def calculate_quality(self) -> float:
        """
        Расчет качества паттерна (0-1)

        Returns:
            float: Оценка качества
        """
        pass

    def calculate_targets(self, current_price: float) -> PatternTargets:
        """
        Расчет целевых уровней

        Args:
            current_price: Текущая цена

        Returns:
            PatternTargets: Объект с целевыми уровнями
        """
        if not self.points or len(self.points) < 2:
            return self.targets

        try:
            # Базовая реализация расчета целей
            prices = [p.price for p in self.points]
            pattern_high = max(prices)
            pattern_low = min(prices)
            pattern_height = pattern_high - pattern_low

            self.targets.pattern_height = pattern_height

            if self.direction == PatternDirection.BULLISH:
                self._calculate_bullish_targets(current_price, pattern_height)
            elif self.direction == PatternDirection.BEARISH:
                self._calculate_bearish_targets(current_price, pattern_height)

            # Расчет риска и прибыли
            if (self.targets.entry_price and self.targets.stop_loss and
                self.targets.take_profit):

                risk = abs(self.targets.entry_price - self.targets.stop_loss)
                reward = abs(self.targets.take_profit - self.targets.entry_price)

                self.targets.risk_amount = risk
                self.targets.reward_amount = reward
                self.targets.profit_risk_ratio = reward / risk if risk > 0 else 0

            return self.targets

        except Exception as e:
            print(f"Ошибка расчета целей: {e}")
            return self.targets

    def _calculate_bullish_targets(self, current_price: float, pattern_height: float):
        """Расчет целей для бычьего паттерна"""
        # Базовая логика - переопределить в дочерних классах
        self.targets.entry_price = current_price
        self.targets.stop_loss = current_price - pattern_height * 0.5
        self.targets.take_profit = current_price + pattern_height

        # Дополнительные цели
        self.targets.target1 = current_price + pattern_height * 0.5
        self.targets.target2 = current_price + pattern_height
        self.targets.target3 = current_price + pattern_height * 1.5

    def _calculate_bearish_targets(self, current_price: float, pattern_height: float):
        """Расчет целей для медвежьего паттерна"""
        # Базовая логика - переопределить в дочерних классах
        self.targets.entry_price = current_price
        self.targets.stop_loss = current_price + pattern_height * 0.5
        self.targets.take_profit = current_price - pattern_height

        # Дополнительные цели
        self.targets.target1 = current_price - pattern_height * 0.5
        self.targets.target2 = current_price - pattern_height
        self.targets.target3 = current_price - pattern_height * 1.5

    def analyze_strength(self) -> Dict[str, float]:
        """
        Анализ силы паттерна

        Returns:
            Dict с оценками силы по различным параметрам
        """
        strength_scores = {
            'geometric_quality': self.calculate_quality(),
            'volume_confirmation': 1.0 if self.metadata.volume_confirmation else 0.0,
            'trend_alignment': self._calculate_trend_alignment(),
            'indicator_support': self._calculate_indicator_support(),
            'timeframe_confirmation': self._calculate_timeframe_confirmation(),
            'risk_reward_ratio': min(self.targets.profit_risk_ratio or 0, 3.0) / 3.0
        }

        # Итоговый балл
        weights = {
            'geometric_quality': 0.3,
            'volume_confirmation': 0.15,
            'trend_alignment': 0.2,
            'indicator_support': 0.15,
            'timeframe_confirmation': 0.1,
            'risk_reward_ratio': 0.1
        }

        total_score = sum(score * weights.get(param, 0)
                         for param, score in strength_scores.items())

        strength_scores['total_score'] = total_score
        self.metadata.reliability_score = total_score

        return strength_scores

    def _calculate_trend_alignment(self) -> float:
        """Расчет соответствия тренду"""
        # Базовая реализация
        if self.metadata.market_context == MarketContext.UPTREND:
            return 1.0 if self.direction == PatternDirection.BULLISH else 0.3
        elif self.metadata.market_context == MarketContext.DOWNTREND:
            return 1.0 if self.direction == PatternDirection.BEARISH else 0.3
        else:
            return 0.7  # Боковик

    def _calculate_indicator_support(self) -> float:
        """Расчет поддержки индикаторов"""
        score = 0.0

        # RSI
        if self.metadata.rsi_value is not None:
            if self.direction == PatternDirection.BULLISH:
                score += 0.5 if self.metadata.rsi_value < 40 else 0.2
            else:
                score += 0.5 if self.metadata.rsi_value > 60 else 0.2

        # MACD
        if self.metadata.macd_value is not None:
            if self.direction == PatternDirection.BULLISH:
                score += 0.5 if self.metadata.macd_value > 0 else 0.2
            else:
                score += 0.5 if self.metadata.macd_value < 0 else 0.2

        return min(score, 1.0)

    def _calculate_timeframe_confirmation(self) -> float:
        """Расчет подтверждения на других таймфреймах"""
        return 1.0 if self.metadata.multi_timeframe_confirmation else 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация паттерна в словарь"""
        return {
            'id': self.metadata.pattern_id,
            'name': self.name,
            'type': self.pattern_type.value,
            'direction': self.direction.value,
            'abbreviation': self.abbreviation,
            'points': [point.dict() for point in self.points],
            'metadata': self.metadata.dict(),
            'targets': self.targets.dict(),
            'statistics': self.statistics.dict(),
            'strength_analysis': self.analyze_strength(),
            'is_detected': self._is_detected,
            'is_confirmed': self._is_confirmed,
            'is_completed': self._is_completed,
            'complexity_level': self.complexity_level,
            'frequency_score': self.frequency_score,
            'success_rate': self.success_rate
        }

    def to_json(self) -> str:
        """Конвертация паттерна в JSON"""
        import json
        return json.dumps(self.to_dict(), indent=2, default=str)

    def validate_pattern(self) -> Tuple[bool, List[str]]:
        """
        Валидация паттерна

        Returns:
            Tuple[bool, List[str]]: (валиден, список ошибок)
        """
        errors = []

        # Проверка минимального количества точек
        min_points = self._get_min_points_required()
        if len(self.points) < min_points:
            errors.append(f"Недостаточно точек: {len(self.points)} < {min_points}")

        # Проверка качества
        quality = self.calculate_quality()
        if quality < 0.6:
            errors.append(f"Низкое качество: {quality:.2f}")

        # Проверка целевых уровней
        if not self.targets.entry_price:
            errors.append("Не рассчитана цена входа")

        if not self.targets.stop_loss:
            errors.append("Не рассчитан стоп-лосс")

        if not self.targets.take_profit:
            errors.append("Не рассчитан тейк-профит")

        return len(errors) == 0, errors

    def _get_min_points_required(self) -> int:
        """Минимальное необходимое количество точек для паттерна"""
        return 2

    def __str__(self) -> str:
        return (f"{self.name} ({self.direction.value}) | "
                f"Качество: {self.calculate_quality():.2f} | "
                f"Надежность: {self.metadata.reliability_score:.2f} | "
                f"Риск/Прибыль: {self.targets.profit_risk_ratio or 0:.2f}")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"

