import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from .base_pattern import Pattern
import logging

logger = logging.getLogger(__name__)

class DojiPattern(Pattern):
    """Паттерн Доджи"""
    
    def detect(self, data: pd.DataFrame) -> List['DojiPattern']:
        patterns = []
        
        if len(data) < 1:
            return patterns
        
        for i in range(len(data)):
            current = data.iloc[i]
            
            open_price = current['open']
            close_price = current['close']
            high_price = current['high']
            low_price = current['low']
            
            # Определение Доджи: тело свечи очень маленькое
            body_size = abs(close_price - open_price)
            total_range = high_price - low_price
            
            if total_range > 0:
                body_ratio = body_size / total_range
                
                # Доджи имеет тело меньше 10% от всего диапазона
                if body_ratio < 0.1:
                    # Определяем тип Доджи
                    upper_shadow = high_price - max(open_price, close_price)
                    lower_shadow = min(open_price, close_price) - low_price
                    
                    if upper_shadow > lower_shadow * 3 and lower_shadow < total_range * 0.1:
                        pattern_type = "gravestone_doji"
                        confidence = 0.8
                    elif lower_shadow > upper_shadow * 3 and upper_shadow < total_range * 0.1:
                        pattern_type = "dragonfly_doji"
                        confidence = 0.8
                    elif upper_shadow > 0 and lower_shadow > 0 and abs(upper_shadow - lower_shadow) < total_range * 0.1:
                        pattern_type = "long_legged_doji"
                        confidence = 0.7
                    else:
                        pattern_type = "doji"
                        confidence = 0.6
                    
                    # Проверяем контекст для повышения уверенности
                    if i > 0:
                        prev_close = data.iloc[i-1]['close']
                        trend = "up" if close_price > prev_close else "down"
                        
                        if pattern_type == "dragonfly_doji" and trend == "down":
                            confidence += 0.1
                        elif pattern_type == "gravestone_doji" and trend == "up":
                            confidence += 0.1
                    
                    pattern = DojiPattern(
                        symbol=data['symbol'].iloc[i] if 'symbol' in data.columns else "Unknown",
                        timestamp=data.index[i],
                        pattern_type=pattern_type,
                        confidence=min(confidence, 1.0),
                        data={
                            'open': open_price,
                            'high': high_price,
                            'low': low_price,
                            'close': close_price,
                            'body_ratio': body_ratio,
                            'upper_shadow': upper_shadow,
                            'lower_shadow': lower_shadow
                        }
                    )
                    patterns.append(pattern)
        
        logger.debug(f"Найдено {len(patterns)} паттернов Доджи")
        return patterns
    
    def get_signal(self) -> str:
        """Доджи обычно сигнализирует о неопределенности, но конкретные типы дают сигналы"""
        if self.pattern_type == "dragonfly_doji":
            # Дракония муха - бычий разворот
            return "buy"
        elif self.pattern_type == "gravestone_doji":
            # Надгробие - медвежий разворот
            return "sell"
        elif self.pattern_type == "long_legged_doji":
            # Длинноногий доджи - сильная неопределенность
            return "hold"
        else:
            # Обычный доджи
            return "hold"

class HammerPattern(Pattern):
    """Паттерн Молот и Повешенный"""
    
    def detect(self, data: pd.DataFrame) -> List['HammerPattern']:
        patterns = []
        
        if len(data) < 3:
            return patterns
        
        for i in range(2, len(data)):
            current = data.iloc[i]
            prev1 = data.iloc[i-1]
            prev2 = data.iloc[i-2]
            
            open_price = current['open']
            close_price = current['close']
            high_price = current['high']
            low_price = current['low']
            
            body_size = abs(close_price - open_price)
            total_range = high_price - low_price
            
            if total_range > 0:
                # Определение молота/повешенного
                upper_shadow = high_price - max(open_price, close_price)
                lower_shadow = min(open_price, close_price) - low_price
                
                # Условия для молота/повешенного:
                # 1. Длинная нижняя тень (минимум 2/3 от всего диапазона)
                # 2. Маленькая верхняя тень
                # 3. Тело маленькое
                is_long_lower_shadow = lower_shadow >= total_range * 0.66
                is_small_upper_shadow = upper_shadow <= total_range * 0.1
                is_small_body = body_size <= total_range * 0.33
                
                if is_long_lower_shadow and is_small_upper_shadow and is_small_body:
                    # Определяем контекст тренда
                    prev_trend_up = prev2['close'] < prev1['close'] < close_price
                    prev_trend_down = prev2['close'] > prev1['close'] > close_price
                    
                    # Определяем бычье или медвежье тело
                    is_bullish_body = close_price > open_price
                    
                    if prev_trend_down and is_bullish_body:
                        # Молот (бычий разворот внизу нисходящего тренда)
                        pattern_type = "hammer"
                        confidence = 0.8
                        if lower_shadow >= total_range * 0.75:
                            confidence = 0.9
                    elif prev_trend_up and not is_bullish_body:
                        # Повешенный (медвежий разворот вверху восходящего тренда)
                        pattern_type = "hanging_man"
                        confidence = 0.8
                        if lower_shadow >= total_range * 0.75:
                            confidence = 0.9
                    else:
                        # Не в контексте тренда - менее надежно
                        pattern_type = "hammer" if is_bullish_body else "hanging_man"
                        confidence = 0.5
                    
                    pattern = HammerPattern(
                        symbol=data['symbol'].iloc[i] if 'symbol' in data.columns else "Unknown",
                        timestamp=data.index[i],
                        pattern_type=pattern_type,
                        confidence=confidence,
                        data={
                            'open': open_price,
                            'high': high_price,
                            'low': low_price,
                            'close': close_price,
                            'body_size': body_size,
                            'total_range': total_range,
                            'upper_shadow': upper_shadow,
                            'lower_shadow': lower_shadow,
                            'is_bullish': is_bullish_body
                        }
                    )
                    patterns.append(pattern)
        
        logger.debug(f"Найдено {len(patterns)} паттернов Молот/Повешенный")
        return patterns
    
    def get_signal(self) -> str:
        """Сигнал зависит от типа паттерна"""
        if self.pattern_type == "hammer":
            return "buy"
        elif self.pattern_type == "hanging_man":
            return "sell"
        return "hold"

class EngulfingPattern(Pattern):
    """Паттерн Поглощение"""
    
    def detect(self, data: pd.DataFrame) -> List['EngulfingPattern']:
        patterns = []
        
        if len(data) < 3:
            return patterns
        
        for i in range(1, len(data)):
            current = data.iloc[i]
            previous = data.iloc[i-1]
            
            curr_open = current['open']
            curr_close = current['close']
            prev_open = previous['open']
            prev_close = previous['close']
            
            # Определяем направление свечей
            prev_bullish = prev_close > prev_open
            curr_bullish = curr_close > curr_open
            
            # Размеры тел
            prev_body = abs(prev_close - prev_open)
            curr_body = abs(curr_close - curr_open)
            
            # Условия поглощения
            is_engulfing = False
            pattern_type = ""
            
            if prev_bullish and not curr_bullish:
                # Медвежье поглощение: бычья свеча, затем медвежья большего размера
                is_engulfing = (curr_open > prev_close and curr_close < prev_open)
                pattern_type = "bearish_engulfing"
            elif not prev_bullish and curr_bullish:
                # Бычье поглощение: медвежья свеча, затем бычья большего размера
                is_engulfing = (curr_open < prev_close and curr_close > prev_open)
                pattern_type = "bullish_engulfing"
            
            # Проверяем размер (текущая свеча должна быть значительно больше)
            if is_engulfing and curr_body > prev_body * 1.5:
                # Проверяем контекст тренда
                if i >= 2:
                    prev_trend = data.iloc[i-2]['close']
                    if pattern_type == "bullish_engulfing" and prev_close < prev_trend:
                        confidence = 0.9  # Разворот внизу нисходящего тренда
                    elif pattern_type == "bearish_engulfing" and prev_close > prev_trend:
                        confidence = 0.9  # Разворот вверху восходящего тренда
                    else:
                        confidence = 0.7
                else:
                    confidence = 0.7
                
                # Увеличиваем уверенность если свеча очень большая
                if curr_body > prev_body * 2:
                    confidence = min(confidence + 0.1, 1.0)
                
                pattern = EngulfingPattern(
                    symbol=data['symbol'].iloc[i] if 'symbol' in data.columns else "Unknown",
                    timestamp=data.index[i],
                    pattern_type=pattern_type,
                    confidence=confidence,
                    data={
                        'prev_open': prev_open,
                        'prev_close': prev_close,
                        'curr_open': curr_open,
                        'curr_close': curr_close,
                        'prev_body': prev_body,
                        'curr_body': curr_body,
                        'size_ratio': curr_body / prev_body if prev_body > 0 else 0
                    }
                )
                patterns.append(pattern)
        
        logger.debug(f"Найдено {len(patterns)} паттернов Поглощение")
        return patterns
    
    def get_signal(self) -> str:
        """Сигнал зависит от типа поглощения"""
        if self.pattern_type == "bullish_engulfing":
            return "buy"
        elif self.pattern_type == "bearish_engulfing":
            return "sell"
        return "hold"

class MorningStarPattern(Pattern):
    """Паттерн Утренняя звезда и Вечерняя звезда"""
    
    def detect(self, data: pd.DataFrame) -> List['MorningStarPattern']:
        patterns = []
        
        if len(data) < 3:
            return patterns
        
        for i in range(2, len(data)):
            first = data.iloc[i-2]
            second = data.iloc[i-1]
            third = data.iloc[i]
            
            # Цены свечей
            first_open = first['open']
            first_close = first['close']
            second_open = second['open']
            second_close = second['close']
            third_open = third['open']
            third_close = third['close']
            
            # Определяем направление свечей
            first_bullish = first_close > first_open
            second_bullish = second_close > second_open
            third_bullish = third_close > third_open
            
            # Размеры тел
            first_body = abs(first_close - first_open)
            second_body = abs(second_close - second_open)
            third_body = abs(third_close - third_open)
            
            # Диапазоны свечей
            first_range = first['high'] - first['low']
            second_range = second['high'] - second['low']
            third_range = third['high'] - third['low']
            
            # Условия для Утренней звезды (бычий разворот)
            is_morning_star = (
                not first_bullish and  # Первая - медвежья
                second_body <= second_range * 0.3 and  # Вторая - маленькая свеча (доджи или спиннинг топ)
                third_bullish and  # Третья - бычья
                third_close > first_body * 0.5 + first_open  # Третья закрывается выше середины первой
            )
            
            # Условия для Вечерней звезды (медвежий разворот)
            is_evening_star = (
                first_bullish and  # Первая - бычья
                second_body <= second_range * 0.3 and  # Вторая - маленькая свеча
                not third_bullish and  # Третья - медвежья
                third_close < first_close - first_body * 0.5  # Третья закрывается ниже середины первой
            )
            
            # Проверяем наличие гэпов для увеличения уверенности
            if is_morning_star or is_evening_star:
                # Проверяем гэпы между свечами
                gap_down = second['high'] < first['low']  # Гэп вниз между первой и второй
                gap_up = third['low'] > second['high']    # Гэп вверх между второй и третьей
                
                pattern_type = "morning_star" if is_morning_star else "evening_star"
                
                # Базовая уверенность
                confidence = 0.7
                
                # Увеличиваем уверенность при наличии гэпов
                if (is_morning_star and gap_down) or (is_evening_star and gap_up):
                    confidence = 0.9
                
                # Увеличиваем уверенность если третья свеча длинная
                if third_body > first_body * 0.8:
                    confidence = min(confidence + 0.1, 1.0)
                
                pattern = MorningStarPattern(
                    symbol=data['symbol'].iloc[i] if 'symbol' in data.columns else "Unknown",
                    timestamp=data.index[i],
                    pattern_type=pattern_type,
                    confidence=confidence,
                    data={
                        'first_candle': {
                            'open': first_open, 'close': first_close,
                            'high': first['high'], 'low': first['low'],
                            'is_bullish': first_bullish
                        },
                        'second_candle': {
                            'open': second_open, 'close': second_close,
                            'high': second['high'], 'low': second['low'],
                            'is_bullish': second_bullish,
                            'body_ratio': second_body / second_range if second_range > 0 else 0
                        },
                        'third_candle': {
                            'open': third_open, 'close': third_close,
                            'high': third['high'], 'low': third['low'],
                            'is_bullish': third_bullish
                        },
                        'has_gap_down': gap_down,
                        'has_gap_up': gap_up
                    }
                )
                patterns.append(pattern)
        
        logger.debug(f"Найдено {len(patterns)} паттернов Утренняя/Вечерняя звезда")
        return patterns
    
    def get_signal(self) -> str:
        """Сигнал зависит от типа звезды"""
        if self.pattern_type == "morning_star":
            return "buy"
        elif self.pattern_type == "evening_star":
            return "sell"
        return "hold"

