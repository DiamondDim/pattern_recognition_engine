import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

from patterns.base_pattern import BasePattern

logger = logging.getLogger(__name__)

class HarmonicPattern(BasePattern):
    """
    Класс для обнаружения гармонических паттернов (Фибоначчи-паттернов)
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """Инициализация гармонического паттерна"""
        super().__init__(data)
        self.fibonacci_levels = {
            'retracement': [0.236, 0.382, 0.5, 0.618, 0.786],
            'extension': [0.618, 1.0, 1.272, 1.414, 1.618, 2.0, 2.618]
        }
        self.pattern_definitions = self._initialize_pattern_definitions()
        
    def _initialize_pattern_definitions(self) -> Dict[str, Dict[str, float]]:
        """
        Инициализация определений гармонических паттернов
        
        Returns:
            dict: Определения паттернов
        """
        return {
            'gartley': {
                'XA': None,  # Любое движение
                'AB': 0.618,  # 61.8% от XA
                'BC': 0.382,  # 38.2% от AB
                'CD': 1.272,  # 127.2% от BC
                'AD': 0.786   # 78.6% от XA
            },
            'butterfly': {
                'XA': None,
                'AB': 0.786,
                'BC': 0.382,
                'CD': 1.618,
                'AD': 1.272
            },
            'bat': {
                'XA': None,
                'AB': 0.382,
                'BC': 0.382,
                'CD': 2.618,
                'AD': 0.886
            },
            'crab': {
                'XA': None,
                'AB': 0.382,
                'BC': 0.382,
                'CD': 3.618,
                'AD': 1.618
            },
            'shark': {
                'XA': None,
                'AB': 1.13,
                'BC': 1.618,
                'CD': 1.27,
                'AD': 0.886
            }
        }
        
    def detect(self, max_patterns: int = 10, fib_tolerance: float = 0.05) -> List[Dict[str, Any]]:
        """
        Обнаружение гармонических паттернов
        
        Args:
            max_patterns (int): Максимальное количество паттернов для поиска
            fib_tolerance (float): Допуск для соотношений Фибоначчи
            
        Returns:
            list: Список обнаруженных гармонических паттернов
        """
        patterns = []
        
        if self.data is None or self.data.empty:
            logger.warning("Нет данных для обнаружения гармонических паттернов")
            return patterns
            
        if len(self.data) < 100:
            logger.warning(f"Недостаточно данных для гармонических паттернов: {len(self.data)} < 100")
            return patterns
            
        try:
            # Находим точки разворота (пивоты)
            pivot_points = self._find_pivot_points()
            
            if len(pivot_points) < 5:
                logger.warning(f"Недостаточно точек разворота: {len(pivot_points)} < 5")
                return patterns
            
            logger.info(f"Найдено {len(pivot_points)} точек разворота для анализа гармонических паттернов")
            
            # Анализируем возможные паттерны
            for i in range(len(pivot_points) - 4):
                # Получаем последовательность из 5 точек (XABCD)
                points = pivot_points[i:i+5]
                
                # Проверяем, что точки образуют зигзаг
                if not self._is_zigzag(points):
                    continue
                
                # Идентифицируем паттерн
                pattern_info = self._identify_pattern(points, fib_tolerance)
                
                if pattern_info:
                    pattern_info['points'] = points
                    pattern_info['pattern_family'] = 'harmonic'
                    pattern_info['detection_time'] = pd.Timestamp.now()
                    
                    # Добавляем дополнительную информацию
                    pattern_info.update(self._analyze_pattern_quality(points, pattern_info['pattern_type']))
                    
                    patterns.append(pattern_info)
                    
                    if len(patterns) >= max_patterns:
                        break
            
            logger.info(f"Обнаружено {len(patterns)} гармонических паттернов")
            self.patterns = patterns
            return patterns
            
        except Exception as e:
            logger.error(f"Ошибка при обнаружении гармонических паттернов: {e}")
            return []
            
    def _find_pivot_points(self, window: int = 5) -> List[Dict[str, Any]]:
        """
        Поиск точек разворота (пивот-поинтов)
        
        Args:
            window (int): Размер окна для поиска экстремумов
            
        Returns:
            list: Список точек разворота
        """
        points = []
        
        try:
            highs = self.data['High'].values
            lows = self.data['Low'].values
            
            for i in range(window, len(self.data) - window):
                # Проверяем максимум
                if highs[i] == max(highs[i-window:i+window+1]):
                    points.append({
                        'index': i,
                        'price': highs[i],
                        'type': 'high',
                        'timestamp': self.data.index[i]
                    })
                
                # Проверяем минимум
                if lows[i] == min(lows[i-window:i+window+1]):
                    points.append({
                        'index': i,
                        'price': lows[i],
                        'type': 'low',
                        'timestamp': self.data.index[i]
                    })
            
            # Сортируем по индексу
            points.sort(key=lambda x: x['index'])
            
            # Удаляем соседние точки одинакового типа
            filtered_points = []
            for i in range(len(points)):
                if i == 0 or points[i]['type'] != points[i-1]['type']:
                    filtered_points.append(points[i])
            
            logger.debug(f"Найдено {len(filtered_points)} точек разворота")
            return filtered_points
            
        except Exception as e:
            logger.error(f"Ошибка при поиске точек разворота: {e}")
            return []
            
    def _is_zigzag(self, points: List[Dict[str, Any]]) -> bool:
        """
        Проверка, образуют ли точки зигзаг (чередование максимумов и минимумов)
        
        Args:
            points (list): Список точек
            
        Returns:
            bool: True если точки образуют зигзаг
        """
        if len(points) < 2:
            return False
            
        # Проверяем чередование типов
        for i in range(1, len(points)):
            if points[i]['type'] == points[i-1]['type']:
                return False
        
        # Проверяем, что цены образуют волну
        if len(points) >= 3:
            # Для паттерна XABCD нужны конкретные движения
            # X->A, A->B, B->C, C->D
            # Проверяем направления движений
            directions = []
            for i in range(1, len(points)):
                if points[i]['price'] > points[i-1]['price']:
                    directions.append('up')
                else:
                    directions.append('down')
            
            # Для гармонических паттернов ожидаем определенную последовательность
            # Например, для бычьего паттерна: X->A вниз, A->B вверх, B->C вниз, C->D вверх
            # Проверяем эту последовательность
            if len(directions) == 4:
                # Проверяем бычий паттерн
                bullish_pattern = directions == ['down', 'up', 'down', 'up']
                # Проверяем медвежий паттерн
                bearish_pattern = directions == ['up', 'down', 'up', 'down']
                
                return bullish_pattern or bearish_pattern
        
        return True
        
    def _identify_pattern(self, points: List[Dict[str, Any]], 
                         tolerance: float = 0.05) -> Optional[Dict[str, Any]]:
        """
        Идентификация гармонического паттерна
        
        Args:
            points (list): Точки XABCD
            tolerance (float): Допуск для соотношений
            
        Returns:
            dict: Информация о паттерне или None
        """
        if len(points) != 5:
            return None
            
        try:
            # Извлекаем цены
            X = points[0]['price']
            A = points[1]['price']
            B = points[2]['price']
            C = points[3]['price']
            D = points[4]['price']
            
            # Определяем направления движений
            XA_move = A - X
            AB_move = B - A
            BC_move = C - B
            XB_move = B - X
            XC_move = C - X
            
            # Проверяем каждое определение паттерна
            for pattern_name, pattern_ratios in self.pattern_definitions.items():
                is_valid = True
                actual_ratios = {}
                
                # Проверяем соотношение AB/XA
                if pattern_ratios['AB'] is not None:
                    AB_ratio = abs(AB_move / XA_move) if XA_move != 0 else 0
                    expected_ratio = pattern_ratios['AB']
                    
                    if not self._is_within_tolerance(AB_ratio, expected_ratio, tolerance):
                        is_valid = False
                    
                    actual_ratios['AB/XA'] = AB_ratio
                
                # Проверяем соотношение BC/AB
                if pattern_ratios['BC'] is not None and AB_move != 0:
                    BC_ratio = abs(BC_move / AB_move)
                    expected_ratio = pattern_ratios['BC']
                    
                    if not self._is_within_tolerance(BC_ratio, expected_ratio, tolerance):
                        is_valid = False
                    
                    actual_ratios['BC/AB'] = BC_ratio
                
                # Проверяем соотношение CD/BC (расчетное)
                if pattern_ratios['CD'] is not None and BC_move != 0:
                    # Предполагаемая точка D на основе паттерна
                    if AB_move > 0:  # A->B вверх
                        projected_D = C + abs(BC_move) * pattern_ratios['CD']
                    else:  # A->B вниз
                        projected_D = C - abs(BC_move) * pattern_ratios['CD']
                    
                    CD_ratio = abs(D - C) / abs(BC_move) if BC_move != 0 else 0
                    expected_ratio = pattern_ratios['CD']
                    
                    if not self._is_within_tolerance(CD_ratio, expected_ratio, tolerance * 1.5):
                        is_valid = False
                    
                    actual_ratios['CD/BC'] = CD_ratio
                    actual_ratios['projected_D'] = projected_D
                
                # Проверяем соотношение AD/XA (расчетное)
                if pattern_ratios['AD'] is not None and XA_move != 0:
                    AD_ratio = abs(D - A) / abs(XA_move)
                    expected_ratio = pattern_ratios['AD']
                    
                    if not self._is_within_tolerance(AD_ratio, expected_ratio, tolerance):
                        is_valid = False
                    
                    actual_ratios['AD/XA'] = AD_ratio
                
                # Если паттерн прошел все проверки
                if is_valid:
                    # Определяем направление паттерна
                    direction = 'bullish' if D > C else 'bearish'
                    
                    # Определяем потенциальную зону разворота
                    if direction == 'bullish':
                        potential_reversal_zone = {
                            'lower': C,
                            'upper': D,
                            'target': D + abs(D - C) * 0.618  # Фибо расширение
                        }
                    else:
                        potential_reversal_zone = {
                            'lower': D,
                            'upper': C,
                            'target': D - abs(C - D) * 0.618
                        }
                    
                    # Рассчитываем уверенность на основе точности соотношений
                    confidence = self._calculate_pattern_confidence(actual_ratios, pattern_ratios, tolerance)
                    
                    pattern_info = {
                        'pattern_type': pattern_name,
                        'direction': direction,
                        'points_indices': [p['index'] for p in points],
                        'prices': {'X': X, 'A': A, 'B': B, 'C': C, 'D': D},
                        'actual_ratios': actual_ratios,
                        'expected_ratios': pattern_ratios,
                        'tolerance': tolerance,
                        'confidence': confidence,
                        'potential_reversal_zone': potential_reversal_zone,
                        'signal': 'buy' if direction == 'bullish' else 'sell',
                        'timestamp': points[4]['timestamp']
                    }
                    
                    return pattern_info
            
            return None
            
        except Exception as e:
            logger.error(f"Ошибка идентификации паттерна: {e}")
            return None
            
    def _is_within_tolerance(self, actual: float, expected: float, 
                            tolerance: float) -> bool:
        """
        Проверка нахождения значения в пределах допуска
        
        Args:
            actual (float): Фактическое значение
            expected (float): Ожидаемое значение
            tolerance (float): Допуск
            
        Returns:
            bool: True если в пределах допуска
        """
        if expected == 0:
            return actual == 0
        
        ratio = actual / expected
        return abs(1 - ratio) <= tolerance
        
    def _calculate_pattern_confidence(self, actual_ratios: Dict[str, float],
                                     expected_ratios: Dict[str, float],
                                     tolerance: float) -> float:
        """
        Расчет уверенности в паттерне
        
        Args:
            actual_ratios (dict): Фактические соотношения
            expected_ratios (dict): Ожидаемые соотношения
            tolerance (float): Допуск
            
        Returns:
            float: Уверенность (0-1)
        """
        if not actual_ratios:
            return 0.5
            
        total_error = 0
        count = 0
        
        for key, expected in expected_ratios.items():
            if expected is not None and key in actual_ratios:
                actual = actual_ratios[key]
                if expected != 0:
                    error = abs(actual / expected - 1)
                    total_error += error
                    count += 1
        
        if count == 0:
            return 0.5
            
        avg_error = total_error / count
        confidence = 1.0 - min(1.0, avg_error / tolerance)
        
        return max(0.1, min(1.0, confidence))
        
    def _analyze_pattern_quality(self, points: List[Dict[str, Any]], 
                                pattern_type: str) -> Dict[str, Any]:
        """
        Анализ качества паттерна
        
        Args:
            points (list): Точки паттерна
            pattern_type (str): Тип паттерна
            
        Returns:
            dict: Информация о качестве
        """
        quality = {
            'volume_analysis': {},
            'time_analysis': {},
            'market_context': {}
        }
        
        try:
            # Анализ объема
            if 'Volume' in self.data.columns:
                volume_data = []
                for point in points:
                    idx = point['index']
                    if idx < len(self.data):
                        volume_data.append(self.data.iloc[idx]['Volume'])
                
                if volume_data:
                    quality['volume_analysis'] = {
                        'avg_volume': np.mean(volume_data),
                        'volume_trend': 'increasing' if volume_data[-1] > volume_data[0] else 'decreasing',
                        'volume_at_D': volume_data[-1] if volume_data else 0
                    }
            
            # Анализ времени
            if len(points) >= 2:
                times = [point['timestamp'] for point in points]
                time_diffs = []
                
                for i in range(1, len(times)):
                    if isinstance(times[i], pd.Timestamp) and isinstance(times[i-1], pd.Timestamp):
                        diff = (times[i] - times[i-1]).total_seconds() / 3600  # в часах
                        time_diffs.append(diff)
                
                if time_diffs:
                    quality['time_analysis'] = {
                        'avg_time_between_points_hours': np.mean(time_diffs),
                        'total_pattern_duration_hours': sum(time_diffs)
                    }
            
            # Контекст рынка
            last_point_idx = points[-1]['index']
            if last_point_idx > 20:
                # Анализ тренда перед паттерном
                prev_data = self.data.iloc[last_point_idx-20:last_point_idx]
                if len(prev_data) > 1:
                    price_change = (prev_data.iloc[-1]['Close'] - prev_data.iloc[0]['Close']) / prev_data.iloc[0]['Close'] * 100
                    
                    if price_change > 3:
                        quality['market_context']['trend'] = 'strong_uptrend'
                    elif price_change > 1:
                        quality['market_context']['trend'] = 'uptrend'
                    elif price_change < -3:
                        quality['market_context']['trend'] = 'strong_downtrend'
                    elif price_change < -1:
                        quality['market_context']['trend'] = 'downtrend'
                    else:
                        quality['market_context']['trend'] = 'sideways'
            
            # Дополнительные проверки для конкретных паттернов
            if pattern_type == 'gartley':
                quality['pattern_specific'] = {
                    'is_ideal': self._check_gartley_ideal_conditions(points)
                }
            elif pattern_type == 'butterfly':
                quality['pattern_specific'] = {
                    'is_ideal': self._check_butterfly_ideal_conditions(points)
                }
            
        except Exception as e:
            logger.error(f"Ошибка анализа качества паттерна: {e}")
            
        return quality
        
    def _check_gartley_ideal_conditions(self, points: List[Dict[str, Any]]) -> bool:
        """
        Проверка идеальных условий для паттерна Гартли
        
        Args:
            points (list): Точки паттерна
            
        Returns:
            bool: True если условия идеальные
        """
        # Для идеального Гартли:
        # 1. Точка B должна быть на 61.8% от XA
        # 2. Точка C должна быть на 38.2% от AB
        # 3. Точка D должна быть на 78.6% от XA
        
        try:
            X = points[0]['price']
            A = points[1]['price']
            B = points[2]['price']
            C = points[3]['price']
            D = points[4]['price']
            
            XA_move = A - X
            AB_move = B - A
            BC_move = C - B
            
            AB_ratio = abs(AB_move / XA_move) if XA_move != 0 else 0
            BC_ratio = abs(BC_move / AB_move) if AB_move != 0 else 0
            AD_ratio = abs(D - A) / abs(XA_move) if XA_move != 0 else 0
            
            # Проверяем близость к идеальным соотношениям
            ideal_AB = 0.618
            ideal_BC = 0.382
            ideal_AD = 0.786
            
            tolerance = 0.03  # Более строгий допуск для "идеального"
            
            return (
                self._is_within_tolerance(AB_ratio, ideal_AB, tolerance) and
                self._is_within_tolerance(BC_ratio, ideal_BC, tolerance) and
                self._is_within_tolerance(AD_ratio, ideal_AD, tolerance)
            )
            
        except Exception as e:
            logger.error(f"Ошибка проверки условий Гартли: {e}")
            return False
            
    def _check_butterfly_ideal_conditions(self, points: List[Dict[str, Any]]) -> bool:
        """
        Проверка идеальных условий для паттерна Баттерфляй
        
        Args:
            points (list): Точки паттерна
            
        Returns:
            bool: True если условия идеальные
        """
        try:
            X = points[0]['price']
            A = points[1]['price']
            B = points[2]['price']
            C = points[3]['price']
            D = points[4]['price']
            
            XA_move = A - X
            AB_move = B - A
            BC_move = C - B
            
            AB_ratio = abs(AB_move / XA_move) if XA_move != 0 else 0
            BC_ratio = abs(BC_move / AB_move) if AB_move != 0 else 0
            AD_ratio = abs(D - A) / abs(XA_move) if XA_move != 0 else 0
            
            # Идеальные соотношения для Баттерфляй
            ideal_AB = 0.786
            ideal_BC = 0.382
            ideal_AD = 1.272
            
            tolerance = 0.03
            
            return (
                self._is_within_tolerance(AB_ratio, ideal_AB, tolerance) and
                self._is_within_tolerance(BC_ratio, ideal_BC, tolerance) and
                self._is_within_tolerance(AD_ratio, ideal_AD, tolerance)
            )
            
        except Exception as e:
            logger.error(f"Ошибка проверки условий Баттерфляй: {e}")
            return False
            
    def get_fibonacci_levels(self, start_price: float, end_price: float, 
                            level_type: str = 'retracement') -> Dict[str, float]:
        """
        Расчет уровней Фибоначчи
        
        Args:
            start_price (float): Начальная цена
            end_price (float): Конечная цена
            level_type (str): Тип уровней ('retracement' или 'extension')
            
        Returns:
            dict: Уровни Фибоначчи
        """
        if level_type not in self.fibonacci_levels:
            level_type = 'retracement'
            
        move = end_price - start_price
        levels = {}
        
        for level in self.fibonacci_levels[level_type]:
            if level_type == 'retracement':
                price_level = end_price - move * level
                levels[f'FIB_{int(level*1000)}'] = price_level
            else:
                price_level = end_price + move * level
                levels[f'EXT_{int(level*1000)}'] = price_level
                
        return levels
        
    def validate_pattern(self, pattern_info: Dict[str, Any], 
                        min_confidence: float = 0.7) -> bool:
        """
        Валидация гармонического паттерна
        
        Args:
            pattern_info (dict): Информация о паттерне
            min_confidence (float): Минимальная уверенность
            
        Returns:
            bool: True если паттерн валиден
        """
        if pattern_info.get('confidence', 0) < min_confidence:
            return False
            
        # Дополнительные проверки
        points = pattern_info.get('points', [])
        if len(points) != 5:
            return False
            
        # Проверяем объем (если есть данные)
        volume_analysis = pattern_info.get('volume_analysis', {})
        volume_at_D = volume_analysis.get('volume_at_D', 0)
        
        # Для гармонических паттернов часто важен объем в точке D
        if volume_at_D > 0:
            # Проверяем, что объем в точке D выше среднего
            avg_volume = volume_analysis.get('avg_volume', volume_at_D)
            if volume_at_D < avg_volume * 0.5:
                return False
        
        return True


# Создаем экземпляр для удобства
harmonic_pattern = HarmonicPattern(None)

