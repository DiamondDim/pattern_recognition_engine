"""
Инициализация модуля паттернов
"""

from .base_pattern import (
    BasePattern, PatternType, PatternDirection,
    PatternPoint, PatternMetadata, PatternTargets,
    PatternStatistics, MarketContext
)

from .geometric_patterns import (
    HeadShouldersPattern,
    DoubleTopPattern,
    DoubleBottomPattern,
    TrianglePattern,
    WedgePattern,
    FlagPattern
)

from .candlestick_patterns import (
    EngulfingPattern,
    DojiPattern,
    HammerPattern,
    MorningStarPattern,
    EveningStarPattern
)

# Реестр всех доступных паттернов
PATTERN_REGISTRY = {
    # Геометрические паттерны
    'head_shoulders': HeadShouldersPattern,
    'inverse_head_shoulders': lambda: HeadShouldersPattern(is_inverse=True),
    'double_top': DoubleTopPattern,
    'double_bottom': lambda: DoubleTopPattern(),  # Нужно создать отдельный класс
    'triangle_symmetric': lambda: TrianglePattern(triangle_type="symmetric"),
    'triangle_ascending': lambda: TrianglePattern(triangle_type="ascending"),
    'triangle_descending': lambda: TrianglePattern(triangle_type="descending"),
    'wedge_rising': lambda: WedgePattern(wedge_type="rising"),
    'wedge_falling': lambda: WedgePattern(wedge_type="falling"),
    'flag_bullish': lambda: FlagPattern(pattern_type="flag"),
    'flag_bearish': lambda: FlagPattern(pattern_type="flag"),  # Направление определяется в detect
    'pennant': lambda: FlagPattern(pattern_type="pennant"),

    # Свечные паттерны
    'engulfing_bullish': lambda: EngulfingPattern(is_bullish=True),
    'engulfing_bearish': lambda: EngulfingPattern(is_bullish=False),
    'hammer': lambda: HammerPattern(is_hammer=True),
    'hanging_man': lambda: HammerPattern(is_hammer=False),
    'doji': lambda: DojiPattern(doji_type="standard"),
    'long_legged_doji': lambda: DojiPattern(doji_type="long_legged"),
    'dragonfly_doji': lambda: DojiPattern(doji_type="dragonfly"),
    'gravestone_doji': lambda: DojiPattern(doji_type="gravestone"),
    'morning_star': MorningStarPattern,
    'evening_star': EveningStarPattern,
}


def create_pattern(pattern_name: str, **kwargs) -> BasePattern:
    """
    Фабрика для создания паттернов по имени

    Args:
        pattern_name: Имя паттерна из реестра
        **kwargs: Дополнительные параметры для конструктора

    Returns:
        Экземпляр паттерна
    """
    if pattern_name not in PATTERN_REGISTRY:
        raise ValueError(f"Неизвестный паттерн: {pattern_name}")

    pattern_class = PATTERN_REGISTRY[pattern_name]

    if callable(pattern_class):
        return pattern_class(**kwargs)
    else:
        return pattern_class(**kwargs)


def get_available_patterns() -> List[str]:
    """Получение списка доступных паттернов"""
    return list(PATTERN_REGISTRY.keys())


def get_patterns_by_type(pattern_type: PatternType) -> List[str]:
    """Получение паттернов по типу"""
    patterns = []

    for name, pattern_class in PATTERN_REGISTRY.items():
        # Создаем экземпляр для проверки типа
        try:
            pattern = create_pattern(name)
            if pattern.pattern_type == pattern_type:
                patterns.append(name)
        except:
            continue

    return patterns


__all__ = [
    'BasePattern',
    'PatternType',
    'PatternDirection',
    'PatternPoint',
    'PatternMetadata',
    'PatternTargets',
    'PatternStatistics',
    'MarketContext',
    'HeadShouldersPattern',
    'DoubleTopPattern',
    'TrianglePattern',
    'WedgePattern',
    'FlagPattern',
    'EngulfingPattern',
    'DojiPattern',
    'HammerPattern',
    'MorningStarPattern',
    'EveningStarPattern',
    'create_pattern',
    'get_available_patterns',
    'get_patterns_by_type',
    'PATTERN_REGISTRY'
]

