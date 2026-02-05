import pandas as pd
import numpy as np
from scipy import stats, optimize
from typing import List, Dict, Any, Optional, Tuple
import logging

from patterns.base_pattern import BasePattern

logger = logging.getLogger(__name__)


class TrianglePattern(BasePattern):
    """
    Класс для обнаружения геометрических треугольных паттернов
    """

    def detect(self, min_points: int = 5, sensitivity: float = 0.8) -> List[Dict[str, Any]]:
        """
        Обнаружение треугольных паттернов

        Args:
            min_points (int): Минимальное количество точек для паттерна
            sensitivity (float): Чувствительность обнаружения (0-1)

        Returns:
            list: Список обнаруженных треугольных паттернов
        """
        patterns = []

        if self.data is None or self.data.empty:
            logger.warning("Нет данных для обнаружения треугольных паттернов")
            return patterns

        if len(self.data) < min_points * 2:
            logger.warning(f"Недостаточно данных для треугольных паттернов: {len(self.data)} < {min_points * 2}")
            return patterns

        try:
            # Получаем цены
            highs = self.data['High'].values
            lows = self.data['Low'].values

            # Ищем паттерны в скользящем окне
            window_size = min_points * 2

            for i in range(len(self.data) - window_size + 1):
                # Извлекаем окно данных
                window_highs = highs[i:i + window_size]
                window_lows = lows[i:i + window_size]

                # Ищем линии сопротивления и поддержки
                resistance_line = self._find_resistance_line(window_highs)
                support_line = self._find_support_line(window_lows)

                if resistance_line and support_line:
                    # Проверяем сходимость линий
                    is_converging = self._check_lines_convergence(resistance_line, support_line)

                    if is_converging:
                        # Определяем тип треугольника
                        triangle_type = self._determine_triangle_type(resistance_line, support_line)

                        # Рассчитываем точку прорыва
                        breakout_info = self._calculate_breakout_point(
                            resistance_line, support_line, window_size
                        )

                        # Рассчитываем уверенность
                        confidence = self._calculate_triangle_confidence(
                            resistance_line, support_line, window_highs, window_lows
                        )

                        # Учитываем чувствительность
                        if confidence >= sensitivity:
                            # Создаем описание паттерна
                            pattern = {
                                'index': i + window_size - 1,
                                'pattern_type': triangle_type,
                                'timestamp': self.data.index[i + window_size - 1],
                                'start_index': i,
                                'end_index': i + window_size - 1,
                                'resistance_slope': resistance_line['slope'],
                                'support_slope': support_line['slope'],
                                'resistance_intercept': resistance_line['intercept'],
                                'support_intercept': support_line['intercept'],
                                'triangle_type': triangle_type,
                                'breakout_direction': breakout_info['direction'],
                                'breakout_price': breakout_info['price'],
                                'breakout_distance': breakout_info['distance'],
                                'volume_profile': self._analyze_triangle_volume(i, window_size),
                                'confidence': confidence,
                                'signal': self._get_triangle_signal(triangle_type, breakout_info['direction'])
                            }

                            patterns.append(pattern)

            logger.info(f"Обнаружено {len(patterns)} треугольных паттернов")
            self.patterns = patterns
            return patterns

        except Exception as e:
            logger.error(f"Ошибка при обнаружении треугольных паттернов: {e}")
            return []

    def _find_resistance_line(self, highs: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Поиск линии сопротивления

        Args:
            highs (np.ndarray): Массив максимумов

        Returns:
            dict: Параметры линии сопротивления или None
        """
        try:
            if len(highs) < 3:
                return None

            # Используем линейную регрессию для поиска линии
            x = np.arange(len(highs))

            # Ищем локальные максимумы
            local_maxima = []
            for i in range(1, len(highs) - 1):
                if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                    local_maxima.append((i, highs[i]))

            if len(local_maxima) < 2:
                return None

            # Аппроксимируем линию через локальные максимумы
            x_points = [p[0] for p in local_maxima]
            y_points = [p[1] for p in local_maxima]

            # Линейная регрессия
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_points, y_points)

            # Проверяем качество аппроксимации
            if r_value ** 2 < 0.7:  # Низкий R²
                return None

            return {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'type': 'resistance'
            }

        except Exception as e:
            logger.error(f"Ошибка при поиске линии сопротивления: {e}")
            return None

    def _find_support_line(self, lows: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Поиск линии поддержки

        Args:
            lows (np.ndarray): Массив минимумов

        Returns:
            dict: Параметры линии поддержки или None
        """
        try:
            if len(lows) < 3:
                return None

            # Ищем локальные минимумы
            local_minima = []
            for i in range(1, len(lows) - 1):
                if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                    local_minima.append((i, lows[i]))

            if len(local_minima) < 2:
                return None

            # Аппроксимируем линию через локальные минимумы
            x_points = [p[0] for p in local_minima]
            y_points = [p[1] for p in local_minima]

            # Линейная регрессия
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_points, y_points)

            # Проверяем качество аппроксимации
            if r_value ** 2 < 0.7:
                return None

            return {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'type': 'support'
            }

        except Exception as e:
            logger.error(f"Ошибка при поиске линии поддержки: {e}")
            return None

    def _check_lines_convergence(self, resistance_line: Dict[str, float],
                                 support_line: Dict[str, float]) -> bool:
        """
        Проверка сходимости линий

        Args:
            resistance_line (dict): Линия сопротивления
            support_line (dict): Линия поддержки

        Returns:
            bool: True если линии сходятся
        """
        try:
            # Рассчитываем углы наклона в градусах
            resistance_angle = np.degrees(np.arctan(resistance_line['slope']))
            support_angle = np.degrees(np.arctan(support_line['slope']))

            # Линии сходятся, если их углы различаются достаточно
            angle_diff = abs(resistance_angle - support_angle)

            # Для треугольника линии должны сходиться
            # (один угол положительный, другой отрицательный, или оба не очень параллельны)
            return angle_diff > 5 and angle_diff < 45

        except Exception as e:
            logger.error(f"Ошибка при проверке сходимости линий: {e}")
            return False

    def _determine_triangle_type(self, resistance_line: Dict[str, float],
                                 support_line: Dict[str, float]) -> str:
        """
        Определение типа треугольника

        Args:
            resistance_line (dict): Линия сопротивления
            support_line (dict): Линия поддержки

        Returns:
            str: Тип треугольника
        """
        try:
            resistance_slope = resistance_line['slope']
            support_slope = support_line['slope']

            # Симметричный треугольник
            if abs(resistance_slope + support_slope) < 0.001:
                return "symmetrical_triangle"

            # Восходящий треугольник (горизонтальное сопротивление)
            elif abs(resistance_slope) < 0.001 and support_slope > 0:
                return "ascending_triangle"

            # Нисходящий треугольник (горизонтальная поддержка)
            elif abs(support_slope) < 0.001 and resistance_slope < 0:
                return "descending_triangle"

            # Расширяющийся треугольник (дивергирующие линии)
            elif resistance_slope > 0 and support_slope < 0:
                return "expanding_triangle"

            # Сходящийся треугольник
            else:
                return "converging_triangle"

        except Exception as e:
            logger.error(f"Ошибка при определении типа треугольника: {e}")
            return "unknown_triangle"

    def _calculate_breakout_point(self, resistance_line: Dict[str, float],
                                  support_line: Dict[str, float],
                                  window_size: int) -> Dict[str, Any]:
        """
        Расчет точки прорыва треугольника

        Args:
            resistance_line (dict): Линия сопротивления
            support_line (dict): Линия поддержки
            window_size (int): Размер окна

        Returns:
            dict: Информация о точке прорыва
        """
        try:
            # Рассчитываем точку пересечения линий
            x_intersect = (support_line['intercept'] - resistance_line['intercept']) / \
                          (resistance_line['slope'] - support_line['slope'])

            # Проверяем, находится ли точка пересечения в пределах окна
            if 0 <= x_intersect <= window_size:
                y_intersect = resistance_line['slope'] * x_intersect + resistance_line['intercept']

                # Определяем направление прорыва на основе последнего движения
                # (упрощенная логика)
                if resistance_line['slope'] < support_line['slope']:
                    direction = "up"
                    breakout_price = resistance_line['slope'] * window_size + resistance_line['intercept']
                else:
                    direction = "down"
                    breakout_price = support_line['slope'] * window_size + support_line['intercept']
            else:
                # Точка пересечения вне окна, используем упрощенную логику
                if resistance_line['slope'] < support_line['slope']:
                    direction = "up"
                    breakout_price = resistance_line['slope'] * window_size + resistance_line['intercept']
                else:
                    direction = "down"
                    breakout_price = support_line['slope'] * window_size + support_line['intercept']

                x_intersect = window_size

            # Рассчитываем расстояние до точки прорыва
            current_x = window_size - 1
            current_price = (resistance_line['slope'] + support_line['slope']) / 2 * current_x + \
                            (resistance_line['intercept'] + support_line['intercept']) / 2

            distance = abs(breakout_price - current_price) / current_price * 100

            return {
                'direction': direction,
                'price': breakout_price,
                'intersection_x': x_intersect,
                'distance_percent': distance,
                'distance': distance
            }

        except Exception as e:
            logger.error(f"Ошибка при расчете точки прорыва: {e}")
            return {
                'direction': 'unknown',
                'price': 0,
                'distance': 0
            }

    def _calculate_triangle_confidence(self, resistance_line: Dict[str, float],
                                       support_line: Dict[str, float],
                                       highs: np.ndarray, lows: np.ndarray) -> float:
        """
        Расчет уверенности в треугольном паттерне

        Args:
            resistance_line (dict): Линия сопротивления
            support_line (dict): Линия поддержки
            highs (np.ndarray): Максимумы
            lows (np.ndarray): Минимумы

        Returns:
            float: Уверенность (0-1)
        """
        try:
            confidence = 1.0

            # Учитываем качество линий (R²)
            confidence *= min(1.0, resistance_line['r_squared'])
            confidence *= min(1.0, support_line['r_squared'])

            # Учитываем количество касаний линий
            touch_counts = self._count_line_touches(resistance_line, support_line, highs, lows)
            touch_confidence = min(1.0, touch_counts['resistance'] / 3) * \
                               min(1.0, touch_counts['support'] / 3)
            confidence *= touch_confidence

            # Учитываем угол между линиями
            angle_diff = abs(np.degrees(np.arctan(resistance_line['slope'])) -
                             np.degrees(np.arctan(support_line['slope'])))
            angle_confidence = min(1.0, angle_diff / 30)  # Оптимальный угол 30 градусов
            confidence *= angle_confidence

            return min(1.0, confidence)

        except Exception as e:
            logger.error(f"Ошибка при расчете уверенности: {e}")
            return 0.5

    def _count_line_touches(self, resistance_line: Dict[str, float],
                            support_line: Dict[str, float],
                            highs: np.ndarray, lows: np.ndarray) -> Dict[str, int]:
        """
        Подсчет касаний линий

        Args:
            resistance_line (dict): Линия сопротивления
            support_line (dict): Линия поддержки
            highs (np.ndarray): Максимумы
            lows (np.ndarray): Минимумы

        Returns:
            dict: Количество касаний каждой линии
        """
        touches = {'resistance': 0, 'support': 0}

        try:
            tolerance = 0.01  # 1% допуск

            for i in range(len(highs)):
                # Рассчитываем ожидаемые значения линий
                resistance_value = resistance_line['slope'] * i + resistance_line['intercept']
                support_value = support_line['slope'] * i + support_line['intercept']

                # Проверяем касание сопротивления
                if abs(highs[i] - resistance_value) / resistance_value < tolerance:
                    touches['resistance'] += 1

                # Проверяем касание поддержки
                if abs(lows[i] - support_value) / support_value < tolerance:
                    touches['support'] += 1

        except Exception as e:
            logger.error(f"Ошибка при подсчете касаний: {e}")

        return touches

    def _analyze_triangle_volume(self, start_index: int, window_size: int) -> Dict[str, Any]:
        """
        Анализ объема в треугольном паттерне

        Args:
            start_index (int): Начальный индекс
            window_size (int): Размер окна

        Returns:
            dict: Статистика объема
        """
        result = {
            'avg_volume': 0,
            'volume_trend': 'stable',
            'breakout_volume_ratio': 1.0
        }

        if 'Volume' not in self.data.columns:
            return result

        try:
            # Получаем данные объема
            volume_data = self.data.iloc[start_index:start_index + window_size]['Volume']

            if len(volume_data) > 0:
                result['avg_volume'] = volume_data.mean()

                # Анализируем тренд объема
                if len(volume_data) >= 3:
                    first_half = volume_data.iloc[:len(volume_data) // 2].mean()
                    second_half = volume_data.iloc[len(volume_data) // 2:].mean()

                    if second_half > first_half * 1.2:
                        result['volume_trend'] = 'increasing'
                    elif second_half < first_half * 0.8:
                        result['volume_trend'] = 'decreasing'
                    else:
                        result['volume_trend'] = 'stable'

                # Соотношение объема при прорыве
                if len(volume_data) >= 5:
                    last_volume = volume_data.iloc[-1]
                    avg_last_5 = volume_data.iloc[-5:].mean()

                    if avg_last_5 > 0:
                        result['breakout_volume_ratio'] = last_volume / avg_last_5

        except Exception as e:
            logger.error(f"Ошибка при анализе объема: {e}")

        return result

    def _get_triangle_signal(self, triangle_type: str, breakout_direction: str) -> str:
        """
        Определение сигнала треугольника

        Args:
            triangle_type (str): Тип треугольника
            breakout_direction (str): Направление прорыва

        Returns:
            str: Сигнал (bullish/bearish/neutral)
        """
        # Логика сигналов на основе типа треугольника и направления прорыва
        if triangle_type == "ascending_triangle" and breakout_direction == "up":
            return "bullish"
        elif triangle_type == "descending_triangle" and breakout_direction == "down":
            return "bearish"
        elif triangle_type == "symmetrical_triangle":
            return breakout_direction  # up -> bullish, down -> bearish
        else:
            return "neutral"


class ChannelPattern(BasePattern):
    """
    Класс для обнаружения канальных паттернов
    """

    def detect(self, min_points: int = 5, sensitivity: float = 0.8) -> List[Dict[str, Any]]:
        """
        Обнаружение канальных паттернов

        Args:
            min_points (int): Минимальное количество точек для паттерна
            sensitivity (float): Чувствительность обнаружения

        Returns:
            list: Список обнаруженных канальных паттернов
        """
        patterns = []

        if self.data is None or self.data.empty:
            logger.warning("Нет данных для обнаружения канальных паттернов")
            return patterns

        if len(self.data) < min_points * 3:
            logger.warning(f"Недостаточно данных для канальных паттернов: {len(self.data)} < {min_points * 3}")
            return patterns

        try:
            # Получаем цены
            highs = self.data['High'].values
            lows = self.data['Low'].values

            # Ищем каналы в скользящем окне
            window_size = min_points * 3

            for i in range(len(self.data) - window_size + 1):
                # Извлекаем окно данных
                window_highs = highs[i:i + window_size]
                window_lows = lows[i:i + window_size]

                # Ищем параллельные линии
                channel_lines = self._find_channel_lines(window_highs, window_lows)

                if channel_lines:
                    upper_line = channel_lines['upper']
                    lower_line = channel_lines['lower']

                    # Проверяем параллельность линий
                    is_parallel = self._check_lines_parallel(upper_line, lower_line)

                    if is_parallel:
                        # Рассчитываем параметры канала
                        channel_info = self._calculate_channel_parameters(
                            upper_line, lower_line, window_size
                        )

                        # Рассчитываем уверенность
                        confidence = self._calculate_channel_confidence(
                            upper_line, lower_line, window_highs, window_lows
                        )

                        # Учитываем чувствительность
                        if confidence >= sensitivity:
                            # Создаем описание паттерна
                            pattern = {
                                'index': i + window_size - 1,
                                'pattern_type': 'price_channel',
                                'timestamp': self.data.index[i + window_size - 1],
                                'start_index': i,
                                'end_index': i + window_size - 1,
                                'upper_slope': upper_line['slope'],
                                'lower_slope': lower_line['slope'],
                                'upper_intercept': upper_line['intercept'],
                                'lower_intercept': lower_line['intercept'],
                                'channel_width': channel_info['width'],
                                'channel_height_pct': channel_info['height_pct'],
                                'trend_direction': channel_info['trend'],
                                'touch_points': channel_info['touch_counts'],
                                'current_position': channel_info['current_position'],
                                'confidence': confidence,
                                'signal': self._get_channel_signal(channel_info['current_position'])
                            }

                            patterns.append(pattern)

            logger.info(f"Обнаружено {len(patterns)} канальных паттернов")
            self.patterns = patterns
            return patterns

        except Exception as e:
            logger.error(f"Ошибка при обнаружении канальных паттернов: {e}")
            return []

    def _find_channel_lines(self, highs: np.ndarray, lows: np.ndarray) -> Optional[Dict[str, Dict]]:
        """
        Поиск параллельных линий канала

        Args:
            highs (np.ndarray): Массив максимумов
            lows (np.ndarray): Массив минимумов

        Returns:
            dict: Верхняя и нижняя линии канала или None
        """
        try:
            # Ищем линии через точки экстремумов
            upper_points = []
            lower_points = []

            # Находим локальные экстремумы
            for i in range(1, len(highs) - 1):
                # Локальные максимумы для верхней линии
                if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                    upper_points.append((i, highs[i]))

                # Локальные минимумы для нижней линии
                if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                    lower_points.append((i, lows[i]))

            if len(upper_points) < 2 or len(lower_points) < 2:
                return None

            # Аппроксимируем линии
            upper_x = [p[0] for p in upper_points]
            upper_y = [p[1] for p in upper_points]

            lower_x = [p[0] for p in lower_points]
            lower_y = [p[1] for p in lower_points]

            # Линейная регрессия для верхней линии
            upper_slope, upper_intercept, upper_r, _, _ = stats.linregress(upper_x, upper_y)

            # Линейная регрессия для нижней линии
            lower_slope, lower_intercept, lower_r, _, _ = stats.linregress(lower_x, lower_y)

            # Проверяем качество аппроксимации
            if upper_r ** 2 < 0.7 or lower_r ** 2 < 0.7:
                return None

            return {
                'upper': {
                    'slope': upper_slope,
                    'intercept': upper_intercept,
                    'r_squared': upper_r ** 2,
                    'type': 'upper_channel'
                },
                'lower': {
                    'slope': lower_slope,
                    'intercept': lower_intercept,
                    'r_squared': lower_r ** 2,
                    'type': 'lower_channel'
                }
            }

        except Exception as e:
            logger.error(f"Ошибка при поиске линий канала: {e}")
            return None

    def _check_lines_parallel(self, upper_line: Dict[str, float],
                              lower_line: Dict[str, float]) -> bool:
        """
        Проверка параллельности линий

        Args:
            upper_line (dict): Верхняя линия
            lower_line (dict): Нижняя линия

        Returns:
            bool: True если линии параллельны
        """
        try:
            # Допуск для параллельности (в радианах)
            angle_tolerance = np.radians(5)  # 5 градусов

            upper_angle = np.arctan(upper_line['slope'])
            lower_angle = np.arctan(lower_line['slope'])

            return abs(upper_angle - lower_angle) < angle_tolerance

        except Exception as e:
            logger.error(f"Ошибка при проверке параллельности: {e}")
            return False

    def _calculate_channel_parameters(self, upper_line: Dict[str, float],
                                      lower_line: Dict[str, float],
                                      window_size: int) -> Dict[str, Any]:
        """
        Расчет параметров канала

        Args:
            upper_line (dict): Верхняя линия
            lower_line (dict): Нижняя линия
            window_size (int): Размер окна

        Returns:
            dict: Параметры канала
        """
        try:
            # Ширина канала
            start_width = upper_line['intercept'] - lower_line['intercept']
            end_width = (upper_line['slope'] * window_size + upper_line['intercept']) - \
                        (lower_line['slope'] * window_size + lower_line['intercept'])

            avg_width = (start_width + end_width) / 2

            # Направление тренда
            avg_slope = (upper_line['slope'] + lower_line['slope']) / 2

            if avg_slope > 0.001:
                trend = 'up'
            elif avg_slope < -0.001:
                trend = 'down'
            else:
                trend = 'sideways'

            # Текущая позиция в канале
            current_x = window_size - 1
            current_price = (self.data.iloc[current_x]['High'] + self.data.iloc[current_x]['Low']) / 2

            upper_value = upper_line['slope'] * current_x + upper_line['intercept']
            lower_value = lower_line['slope'] * current_x + lower_line['intercept']

            channel_height = upper_value - lower_value

            if channel_height > 0:
                position = (current_price - lower_value) / channel_height
            else:
                position = 0.5

            # Подсчет касаний
            touch_counts = self._count_channel_touches(upper_line, lower_line, window_size)

            return {
                'width': avg_width,
                'height_pct': (avg_width / ((upper_value + lower_value) / 2)) * 100,
                'trend': trend,
                'touch_counts': touch_counts,
                'current_position': position
            }

        except Exception as e:
            logger.error(f"Ошибка при расчете параметров канала: {e}")
            return {
                'width': 0,
                'height_pct': 0,
                'trend': 'unknown',
                'touch_counts': {'upper': 0, 'lower': 0},
                'current_position': 0.5
            }

    def _count_channel_touches(self, upper_line: Dict[str, float],
                               lower_line: Dict[str, float],
                               window_size: int) -> Dict[str, int]:
        """
        Подсчет касаний границ канала

        Args:
            upper_line (dict): Верхняя линия
            lower_line (dict): Нижняя линия
            window_size (int): Размер окна

        Returns:
            dict: Количество касаний
        """
        touches = {'upper': 0, 'lower': 0}

        try:
            tolerance = 0.01  # 1% допуск

            for i in range(window_size):
                if i >= len(self.data):
                    break

                high = self.data.iloc[i]['High']
                low = self.data.iloc[i]['Low']

                upper_value = upper_line['slope'] * i + upper_line['intercept']
                lower_value = lower_line['slope'] * i + lower_line['intercept']

                # Проверяем касание верхней границы
                if abs(high - upper_value) / upper_value < tolerance:
                    touches['upper'] += 1

                # Проверяем касание нижней границы
                if abs(low - lower_value) / lower_value < tolerance:
                    touches['lower'] += 1

        except Exception as e:
            logger.error(f"Ошибка при подсчете касаний канала: {e}")

        return touches

    def _calculate_channel_confidence(self, upper_line: Dict[str, float],
                                      lower_line: Dict[str, float],
                                      highs: np.ndarray, lows: np.ndarray) -> float:
        """
        Расчет уверенности в канальном паттерне

        Args:
            upper_line (dict): Верхняя линия
            lower_line (dict): Нижняя линия
            highs (np.ndarray): Максимумы
            lows (np.ndarray): Минимумы

        Returns:
            float: Уверенность (0-1)
        """
        try:
            confidence = 1.0

            # Учитываем качество линий
            confidence *= min(1.0, upper_line['r_squared'])
            confidence *= min(1.0, lower_line['r_squared'])

            # Учитываем параллельность
            angle_diff = abs(np.arctan(upper_line['slope']) - np.arctan(lower_line['slope']))
            parallel_confidence = 1.0 - min(1.0, angle_diff / np.radians(10))
            confidence *= parallel_confidence

            # Учитываем ширину канала (должна быть значимой)
            avg_price = (highs.mean() + lows.mean()) / 2
            channel_width = abs(upper_line['intercept'] - lower_line['intercept'])
            width_ratio = channel_width / avg_price

            if width_ratio < 0.005:  # Слишком узкий канал
                confidence *= 0.5
            elif width_ratio > 0.05:  # Слишком широкий канал
                confidence *= 0.7

            return min(1.0, confidence)

        except Exception as e:
            logger.error(f"Ошибка при расчете уверенности канала: {e}")
            return 0.5

    def _get_channel_signal(self, current_position: float) -> str:
        """
        Определение сигнала канала

        Args:
            current_position (float): Текущая позиция в канале (0-1)

        Returns:
            str: Сигнал
        """
        if current_position > 0.7:  # Близко к верхней границе
            return "bearish"
        elif current_position < 0.3:  # Близко к нижней границе
            return "bullish"
        else:
            return "neutral"


# Создаем экземпляры классов для удобства
triangle_pattern = TrianglePattern(None)
channel_pattern = ChannelPattern(None)

