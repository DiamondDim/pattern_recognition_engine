from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class BasePattern(ABC):
    """
    Базовый абстрактный класс для всех паттернов
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Инициализация базового паттерна
        
        Args:
            data (pd.DataFrame): Финансовые данные для анализа
        """
        self.data = data
        self.patterns = []
        self.metrics = {}
        self.validation_rules = {}
        self._initialize()
        
    def _initialize(self):
        """Инициализация внутренних структур"""
        self.patterns = []
        self.metrics = {}
        self.validation_rules = {
            'min_confidence': 0.5,
            'min_volume_ratio': 0.8,
            'max_volatility': 0.05
        }
        
    @abstractmethod
    def detect(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Абстрактный метод для обнаружения паттернов
        
        Returns:
            list: Список обнаруженных паттернов
        """
        pass
        
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Расчет метрик для обнаруженных паттернов
        
        Returns:
            dict: Метрики паттернов
        """
        if not self.patterns:
            self.metrics = {
                'error': 'No patterns detected',
                'total_patterns': 0,
                'timestamp': datetime.now()
            }
            return self.metrics
            
        try:
            # Базовые метрики
            metrics = {
                'total_patterns': len(self.patterns),
                'pattern_types': {},
                'avg_confidence': 0,
                'min_confidence': 1,
                'max_confidence': 0,
                'success_rate': 0,
                'risk_reward_stats': {},
                'timestamp': datetime.now()
            }
            
            # Сбор данных для агрегации
            confidences = []
            success_flags = []
            risk_rewards = []
            volatilities = []
            volumes = []
            
            pattern_type_stats = {}
            
            for pattern in self.patterns:
                # Статистика по типам
                p_type = pattern.get('pattern_type', 'unknown')
                pattern_type_stats[p_type] = pattern_type_stats.get(p_type, 0) + 1
                
                # Сбор данных
                conf = pattern.get('confidence', 0)
                if conf is not None:
                    confidences.append(conf)
                
                success = pattern.get('success', None)
                if success is not None:
                    success_flags.append(success)
                
                rr = pattern.get('risk_reward_ratio', None)
                if rr is not None:
                    risk_rewards.append(rr)
                
                vol = pattern.get('volatility', None)
                if vol is not None:
                    volatilities.append(vol)
                
                vol_ratio = pattern.get('volume_ratio', None)
                if vol_ratio is not None:
                    volumes.append(vol_ratio)
            
            # Обновляем статистику по типам
            metrics['pattern_types'] = pattern_type_stats
            
            # Агрегация уверенности
            if confidences:
                metrics['avg_confidence'] = float(np.mean(confidences))
                metrics['min_confidence'] = float(np.min(confidences))
                metrics['max_confidence'] = float(np.max(confidences))
                metrics['confidence_std'] = float(np.std(confidences))
            
            # Статистика успешности
            if success_flags:
                success_rate = sum(success_flags) / len(success_flags) * 100
                metrics['success_rate'] = float(success_rate)
                metrics['success_count'] = int(sum(success_flags))
                metrics['failure_count'] = int(len(success_flags) - sum(success_flags))
            
            # Статистика риск/вознаграждение
            if risk_rewards:
                metrics['risk_reward_stats'] = {
                    'avg': float(np.mean(risk_rewards)),
                    'min': float(np.min(risk_rewards)),
                    'max': float(np.max(risk_rewards)),
                    'median': float(np.median(risk_rewards)),
                    'std': float(np.std(risk_rewards))
                }
            
            # Статистика волатильности
            if volatilities:
                metrics['volatility_stats'] = {
                    'avg': float(np.mean(volatilities)),
                    'max': float(np.max(volatilities)),
                    'min': float(np.min(volatilities))
                }
            
            # Статистика объема
            if volumes:
                metrics['volume_stats'] = {
                    'avg_ratio': float(np.mean(volumes)),
                    'max_ratio': float(np.max(volumes)),
                    'min_ratio': float(np.min(volumes))
                }
            
            # Временные метрики
            if len(self.patterns) > 1:
                try:
                    timestamps = []
                    for pattern in self.patterns:
                        ts = pattern.get('timestamp')
                        if isinstance(ts, datetime):
                            timestamps.append(ts.timestamp())
                        elif isinstance(ts, str):
                            try:
                                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                                timestamps.append(dt.timestamp())
                            except:
                                pass
                    
                    if len(timestamps) > 1:
                        timestamps.sort()
                        time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                        metrics['time_between_patterns_seconds'] = float(np.mean(time_diffs))
                        metrics['time_between_patterns_hours'] = float(np.mean(time_diffs) / 3600)
                except Exception as e:
                    logger.error(f"Ошибка расчета временных метрик: {e}")
            
            self.metrics = metrics
            logger.info(f"Рассчитаны метрики для {len(self.patterns)} паттернов")
            return metrics
            
        except Exception as e:
            error_msg = f"Ошибка расчета метрик: {e}"
            logger.error(error_msg)
            self.metrics = {
                'error': error_msg,
                'total_patterns': len(self.patterns),
                'timestamp': datetime.now()
            }
            return self.metrics
            
    def validate(self, min_confidence: float = 0.6, 
                min_volume_ratio: float = 0.8,
                max_volatility: float = 0.05) -> 'BasePattern':
        """
        Валидация паттернов по заданным критериям
        
        Args:
            min_confidence (float): Минимальная уверенность
            min_volume_ratio (float): Минимальное соотношение объема
            max_volatility (float): Максимальная волатильность
            
        Returns:
            BasePattern: Текущий экземпляр для цепочки вызовов
        """
        if not self.patterns:
            logger.warning("Нет паттернов для валидации")
            return self
            
        validated_patterns = []
        validation_errors = []
        
        # Обновляем правила валидации
        self.validation_rules = {
            'min_confidence': min_confidence,
            'min_volume_ratio': min_volume_ratio,
            'max_volatility': max_volatility,
            'validation_time': datetime.now()
        }
        
        for i, pattern in enumerate(self.patterns):
            is_valid = True
            errors = []
            
            try:
                # Проверка уверенности
                confidence = pattern.get('confidence', 0)
                if confidence < min_confidence:
                    is_valid = False
                    errors.append(f'Low confidence: {confidence:.2f} < {min_confidence}')
                
                # Проверка объема
                volume_ratio = pattern.get('volume_ratio', 1)
                if volume_ratio < min_volume_ratio:
                    is_valid = False
                    errors.append(f'Low volume: {volume_ratio:.2f} < {min_volume_ratio}')
                
                # Проверка волатильности
                volatility = pattern.get('volatility', 0)
                if volatility > max_volatility:
                    pattern['high_volatility_warning'] = True
                    errors.append(f'High volatility: {volatility:.3f} > {max_volatility}')
                    # Высокая волатильность не делает паттерн невалидным, только предупреждение
                
                # Проверка данных
                required_bars = pattern.get('required_bars', 0)
                if self.data is not None and len(self.data) < required_bars:
                    is_valid = False
                    errors.append(f'Insufficient data: {len(self.data)} < {required_bars}')
                
                # Проверка индекса
                pattern_index = pattern.get('index', -1)
                if self.data is not None and (pattern_index < 0 or pattern_index >= len(self.data)):
                    is_valid = False
                    errors.append(f'Invalid index: {pattern_index}')
                
                if is_valid:
                    pattern['is_validated'] = True
                    pattern['validation_errors'] = []
                    pattern['validation_time'] = datetime.now()
                    validated_patterns.append(pattern)
                else:
                    pattern['is_validated'] = False
                    pattern['validation_errors'] = errors
                    validation_errors.append({
                        'index': i,
                        'pattern_type': pattern.get('pattern_type', 'unknown'),
                        'errors': errors
                    })
                    
            except Exception as e:
                error_msg = f"Ошибка валидации паттерна {i}: {e}"
                logger.error(error_msg)
                pattern['is_validated'] = False
                pattern['validation_errors'] = [error_msg]
                validation_errors.append({
                    'index': i,
                    'pattern_type': pattern.get('pattern_type', 'unknown'),
                    'errors': [error_msg]
                })
        
        # Обновляем список паттернов
        self.patterns = validated_patterns
        
        # Логируем результаты валидации
        if validation_errors:
            logger.warning(f"Валидация: {len(validation_errors)} паттернов не прошли проверку")
            for error in validation_errors[:5]:  # Логируем первые 5 ошибок
                logger.debug(f"Ошибка паттерна {error['index']} ({error['pattern_type']}): {error['errors']}")
        else:
            logger.info(f"Все {len(validated_patterns)} паттернов прошли валидацию")
        
        # Обновляем статистику валидации
        self.validation_rules['total_validated'] = len(validated_patterns)
        self.validation_rules['total_rejected'] = len(validation_errors)
        self.validation_rules['validation_errors'] = validation_errors
        
        return self
        
    def get_patterns(self) -> List[Dict[str, Any]]:
        """
        Получение обнаруженных паттернов
        
        Returns:
            list: Список паттернов
        """
        return self.patterns.copy()
        
    def get_metrics(self) -> Dict[str, Any]:
        """
        Получение рассчитанных метрик
        
        Returns:
            dict: Метрики паттернов
        """
        return self.metrics.copy()
        
    def get_validation_rules(self) -> Dict[str, Any]:
        """
        Получение правил валидации
        
        Returns:
            dict: Правила валидации
        """
        return self.validation_rules.copy()
        
    def clear(self) -> 'BasePattern':
        """
        Очистка обнаруженных паттернов
        
        Returns:
            BasePattern: Текущий экземпляр для цепочки вызовов
        """
        self.patterns = []
        self.metrics = {}
        self.validation_rules = {}
        logger.info("Паттерны очищены")
        return self
        
    def filter_by_confidence(self, min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """
        Фильтрация паттернов по уверенности
        
        Args:
            min_confidence (float): Минимальная уверенность
            
        Returns:
            list: Отфильтрованные паттерны
        """
        filtered = [
            p for p in self.patterns 
            if p.get('confidence', 0) >= min_confidence
        ]
        logger.info(f"Отфильтровано {len(filtered)} паттернов с уверенностью >= {min_confidence}")
        return filtered
        
    def filter_by_type(self, pattern_type: str) -> List[Dict[str, Any]]:
        """
        Фильтрация паттернов по типу
        
        Args:
            pattern_type (str): Тип паттерна
            
        Returns:
            list: Отфильтрованные паттерны
        """
        filtered = [
            p for p in self.patterns 
            if p.get('pattern_type', '').lower() == pattern_type.lower()
        ]
        logger.info(f"Отфильтровано {len(filtered)} паттернов типа '{pattern_type}'")
        return filtered
        
    def save_patterns(self, filename: str) -> bool:
        """
        Сохранение паттернов в файл
        
        Args:
            filename (str): Имя файла
            
        Returns:
            bool: True если успешно
        """
        try:
            import json
            from datetime import datetime
            
            # Преобразуем datetime в строки
            def datetime_converter(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Type {type(obj)} not serializable")
            
            data = {
                'patterns': self.patterns,
                'metrics': self.metrics,
                'validation_rules': self.validation_rules,
                'save_time': datetime.now().isoformat()
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, default=datetime_converter, indent=2)
            
            logger.info(f"Паттерны сохранены в {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка сохранения паттернов в {filename}: {e}")
            return False
            
    def load_patterns(self, filename: str) -> bool:
        """
        Загрузка паттернов из файла
        
        Args:
            filename (str): Имя файла
            
        Returns:
            bool: True если успешно
        """
        try:
            import json
            from datetime import datetime
            
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Восстанавливаем datetime
            def restore_datetime(obj):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if isinstance(value, str):
                            try:
                                obj[key] = datetime.fromisoformat(value.replace('Z', '+00:00'))
                            except:
                                pass
                        elif isinstance(value, list):
                            for i, item in enumerate(value):
                                if isinstance(item, str):
                                    try:
                                        value[i] = datetime.fromisoformat(item.replace('Z', '+00:00'))
                                    except:
                                        pass
                return obj
            
            data = restore_datetime(data)
            
            self.patterns = data.get('patterns', [])
            self.metrics = data.get('metrics', {})
            self.validation_rules = data.get('validation_rules', {})
            
            logger.info(f"Загружено {len(self.patterns)} паттернов из {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки паттернов из {filename}: {e}")
            return False
            
    def __len__(self) -> int:
        """
        Возвращает количество обнаруженных паттернов
        
        Returns:
            int: Количество паттернов
        """
        return len(self.patterns)
        
    def __str__(self) -> str:
        """
        Строковое представление паттернов
        
        Returns:
            str: Информация о паттернах
        """
        if not self.patterns:
            return "No patterns detected"
            
        pattern_types = {}
        for pattern in self.patterns:
            p_type = pattern.get('pattern_type', 'unknown')
            pattern_types[p_type] = pattern_types.get(p_type, 0) + 1
        
        type_str = ', '.join([f"{k}: {v}" for k, v in pattern_types.items()])
        return f"Patterns: {len(self.patterns)} total, Types: {type_str}"

