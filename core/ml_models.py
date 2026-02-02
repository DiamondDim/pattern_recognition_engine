"""
Модуль машинного обучения для оценки паттернов
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import joblib
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ML библиотеки
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import lightgbm as lgb

from config import ANALYSIS_CONFIG, MODELS_DIR
from utils.logger import logger


@dataclass
class MLModelConfig:
    """Конфигурация ML моделей"""

    # Основные параметры
    MODEL_TYPE: str = 'random_forest'  # random_forest, xgboost, lightgbm, svm, neural_network
    TRAIN_TEST_SPLIT: float = 0.8
    RANDOM_STATE: int = 42
    CROSS_VALIDATION_FOLDS: int = 5

    # Параметры моделей
    RANDOM_FOREST_PARAMS: Dict = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42,
        'n_jobs': -1
    })

    XGBOOST_PARAMS: Dict = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    })

    LIGHTGBM_PARAMS: Dict = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    })

    SVM_PARAMS: Dict = field(default_factory=lambda: {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale',
        'probability': True,
        'random_state': 42
    })

    NEURAL_NETWORK_PARAMS: Dict = field(default_factory=lambda: {
        'hidden_layer_sizes': (100, 50),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.0001,
        'batch_size': 'auto',
        'learning_rate': 'constant',
        'learning_rate_init': 0.001,
        'max_iter': 200,
        'random_state': 42
    })

    # Признаки для обучения
    FEATURES: List[str] = field(default_factory=lambda: [
        # Геометрические признаки
        'pattern_height_normalized',
        'pattern_width_normalized',
        'symmetry_score',
        'complexity_score',

        # Фибоначчи признаки (для гармонических паттернов)
        'ab_retracement',
        'bc_retracement',
        'cd_extension',
        'xd_extension',

        # Контекстные признаки
        'trend_strength',
        'volatility',
        'volume_ratio',

        # Временные признаки
        'timeframe_multiplier',
        'hour_of_day',
        'day_of_week',

        # Индикаторы
        'rsi_value',
        'macd_value',
        'adx_value',

        # Качество
        'quality_score',
        'confidence'
    ])

    # Целевая переменная
    TARGET_COLUMN: str = 'outcome_success'  # 1 - успех, 0 - неудача


class PatternMLModel:
    """Класс для ML модели оценки паттернов"""

    def __init__(self, config: MLModelConfig = None):
        self.config = config or MLModelConfig()
        self.logger = logger.bind(name="PatternMLModel")

        # Модель и препроцессор
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

        # Статистика
        self.training_stats = {
            'training_samples': 0,
            'test_samples': 0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'cross_val_score': 0.0,
            'last_trained': None,
            'feature_importance': {}
        }

    def prepare_training_data(self, patterns: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготовка данных для обучения

        Args:
            patterns: Список паттернов с исходами

        Returns:
            X (признаки), y (целевая переменная)
        """
        try:
            features_list = []
            targets_list = []

            for pattern in patterns:
                # Извлекаем признаки
                features = self._extract_features_for_training(pattern)

                # Извлекаем целевую переменную
                target = self._extract_target(pattern)

                if features is not None and target is not None:
                    features_list.append(features)
                    targets_list.append(target)

            if not features_list:
                self.logger.warning("Нет данных для обучения")
                return np.array([]), np.array([])

            X = np.array(features_list)
            y = np.array(targets_list)

            self.logger.info(f"Подготовлено {len(X)} образцов для обучения")
            return X, y

        except Exception as e:
            self.logger.error(f"Ошибка подготовки данных: {e}")
            return np.array([]), np.array([])

    def _extract_features_for_training(self, pattern: Dict[str, Any]) -> Optional[np.ndarray]:
        """Извлечение признаков для обучения"""
        try:
            features = []

            # Геометрические признаки
            features.append(pattern.get('metadata', {}).get('quality_score', 0.5))
            features.append(pattern.get('metadata', {}).get('confidence', 0.5))

            # Размер паттерна
            points = pattern.get('points', [])
            if points:
                prices = [p['price'] for p in points]
                pattern_height = max(prices) - min(prices) if prices else 0
                avg_price = np.mean(prices) if prices else 1
                features.append(pattern_height / avg_price if avg_price > 0 else 0)
                features.append(len(points) / 100)  # Нормализованная ширина
            else:
                features.extend([0, 0])

            # Симметрия и сложность
            strength_analysis = pattern.get('strength_analysis', {})
            features.append(strength_analysis.get('geometric_quality', 0.5))
            features.append(pattern.get('complexity_level', 1) / 3)

            # Фибоначчи признаки (для гармонических паттернов)
            if pattern.get('type') == 'harmonic':
                # Используем расчеты из паттерна
                features.append(pattern.get('ab_retracement', 0))
                features.append(pattern.get('bc_retracement', 0))
                features.append(pattern.get('cd_extension', 0))
                features.append(pattern.get('xd_extension', 0))
            else:
                features.extend([0, 0, 0, 0])

            # Контекстные признаки
            metadata = pattern.get('metadata', {})
            features.append(metadata.get('adx_value', 0) / 100)
            features.append(metadata.get('volatility_pct', 0))
            features.append(metadata.get('average_volume', 1) / 1000000)

            # Временные признаки
            detected_time = datetime.fromisoformat(pattern.get('detection_time',
                                                               datetime.now().isoformat()))
            features.append(detected_time.hour / 24)
            features.append(detected_time.weekday() / 7)

            # Множитель таймфрейма
            timeframe = metadata.get('timeframe', 'H1')
            timeframe_map = {'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
                             'H1': 60, 'H4': 240, 'D1': 1440, 'W1': 10080}
            features.append(np.log(timeframe_map.get(timeframe, 60)) / np.log(10080))

            # Индикаторы
            features.append(metadata.get('rsi_value', 50) / 100)
            features.append(metadata.get('macd_value', 0) / 100)
            features.append(metadata.get('adx_value', 0) / 100)

            return np.array(features)

        except Exception as e:
            self.logger.error(f"Ошибка извлечения признаков: {e}")
            return None

    def _extract_target(self, pattern: Dict[str, Any]) -> Optional[int]:
        """Извлечение целевой переменной"""
        outcome = pattern.get('outcome')

        if outcome in ['success', 'profit', 'win']:
            return 1
        elif outcome in ['failure', 'loss']:
            return 0
        else:
            # Если нет явного исхода, используем историческую успешность
            success_rate = pattern.get('statistics', {}).get('historical_success_rate', 0.5)
            return 1 if success_rate > 0.6 else 0 if success_rate < 0.4 else None

    def train(self, patterns: List[Dict[str, Any]], validate: bool = True) -> bool:
        """
        Обучение модели

        Args:
            patterns: Данные для обучения
            validate: Проводить ли валидацию

        Returns:
            True если обучение успешно
        """
        try:
            # Подготовка данных
            X, y = self.prepare_training_data(patterns)

            if len(X) == 0 or len(y) == 0:
                self.logger.error("Нет данных для обучения")
                return False

            # Разделение на тренировочную и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=1 - self.config.TRAIN_TEST_SPLIT,
                random_state=self.config.RANDOM_STATE,
                stratify=y
            )

            # Масштабирование признаков
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Создание модели
            self.model = self._create_model()

            if self.model is None:
                self.logger.error("Не удалось создать модель")
                return False

            # Обучение модели
            self.logger.info(f"Обучение модели на {len(X_train)} образцах...")
            self.model.fit(X_train_scaled, y_train)

            # Оценка модели
            if validate:
                self._evaluate_model(X_test_scaled, y_test)

            # Кросс-валидация
            cv_scores = cross_val_score(
                self.model, X_train_scaled, y_train,
                cv=self.config.CROSS_VALIDATION_FOLDS,
                scoring='accuracy'
            )

            self.training_stats['training_samples'] = len(X_train)
            self.training_stats['test_samples'] = len(X_test)
            self.training_stats['cross_val_score'] = cv_scores.mean()
            self.training_stats['last_trained'] = datetime.now()

            # Важность признаков (если модель поддерживает)
            self._calculate_feature_importance(X_train_scaled, y_train)

            self.is_trained = True
            self.logger.info(f"Модель обучена. Accuracy: {self.training_stats['accuracy']:.3f}")

            # Сохранение модели
            self.save_model()

            return True

        except Exception as e:
            self.logger.error(f"Ошибка обучения модели: {e}")
            return False

    def _create_model(self):
        """Создание модели в зависимости от типа"""
        model_type = self.config.MODEL_TYPE

        if model_type == 'random_forest':
            return RandomForestClassifier(**self.config.RANDOM_FOREST_PARAMS)

        elif model_type == 'xgboost':
            return xgb.XGBClassifier(**self.config.XGBOOST_PARAMS)

        elif model_type == 'lightgbm':
            return lgb.LGBMClassifier(**self.config.LIGHTGBM_PARAMS)

        elif model_type == 'svm':
            return SVC(**self.config.SVM_PARAMS)

        elif model_type == 'neural_network':
            return MLPClassifier(**self.config.NEURAL_NETWORK_PARAMS)

        else:
            self.logger.warning(f"Неизвестный тип модели: {model_type}, используем Random Forest")
            return RandomForestClassifier(**self.config.RANDOM_FOREST_PARAMS)

    def _evaluate_model(self, X_test, y_test):
        """Оценка модели на тестовых данных"""
        try:
            y_pred = self.model.predict(X_test)

            self.training_stats['accuracy'] = accuracy_score(y_test, y_pred)
            self.training_stats['precision'] = precision_score(y_test, y_pred, zero_division=0)
            self.training_stats['recall'] = recall_score(y_test, y_pred, zero_division=0)
            self.training_stats['f1_score'] = f1_score(y_test, y_pred, zero_division=0)

            self.logger.info(f"Оценка модели:")
            self.logger.info(f"  Accuracy:  {self.training_stats['accuracy']:.3f}")
            self.logger.info(f"  Precision: {self.training_stats['precision']:.3f}")
            self.logger.info(f"  Recall:    {self.training_stats['recall']:.3f}")
            self.logger.info(f"  F1-score:  {self.training_stats['f1_score']:.3f}")

        except Exception as e:
            self.logger.error(f"Ошибка оценки модели: {e}")

    def _calculate_feature_importance(self, X_train, y_train):
        """Расчет важности признаков"""
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_

                # Создаем словарь важности признаков
                for i, feature_name in enumerate(self.config.FEATURES[:len(importances)]):
                    self.training_stats['feature_importance'][feature_name] = importances[i]

                # Сортируем по убыванию важности
                sorted_importance = sorted(
                    self.training_stats['feature_importance'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                self.logger.info("Важность признаков (топ-10):")
                for feature, importance in sorted_importance[:10]:
                    self.logger.info(f"  {feature}: {importance:.4f}")

        except Exception as e:
            self.logger.debug(f"Не удалось рассчитать важность признаков: {e}")

    def predict(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Предсказание успешности паттерна

        Args:
            pattern: Паттерн для предсказания

        Returns:
            Словарь с предсказанием и вероятностями
        """
        if not self.is_trained or self.model is None:
            self.logger.warning("Модель не обучена")
            return {
                'prediction': 0,
                'probability': 0.5,
                'confidence': 0.0,
                'error': 'Model not trained'
            }

        try:
            # Извлекаем признаки
            features = self._extract_features_for_training(pattern)

            if features is None:
                return {
                    'prediction': 0,
                    'probability': 0.5,
                    'confidence': 0.0,
                    'error': 'Feature extraction failed'
                }

            # Масштабирование
            features_scaled = self.scaler.transform([features])

            # Предсказание
            prediction = self.model.predict(features_scaled)[0]

            # Вероятности (если модель поддерживает)
            probability = 0.5
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features_scaled)[0]
                probability = proba[1] if len(proba) > 1 else proba[0]

            # Уверенность предсказания (на основе вероятности)
            confidence = abs(probability - 0.5) * 2  # 0-1, где 1 - максимальная уверенность

            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'confidence': float(confidence),
                'features_used': len(features)
            }

        except Exception as e:
            self.logger.error(f"Ошибка предсказания: {e}")
            return {
                'prediction': 0,
                'probability': 0.5,
                'confidence': 0.0,
                'error': str(e)
            }

    def predict_batch(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Пакетное предсказание для нескольких паттернов

        Args:
            patterns: Список паттернов

        Returns:
            Список предсказаний
        """
        predictions = []

        for pattern in patterns:
            prediction = self.predict(pattern)
            prediction['pattern_id'] = pattern.get('id', 'unknown')
            prediction['pattern_name'] = pattern.get('name', 'unknown')
            predictions.append(prediction)

        return predictions

    def save_model(self, filepath: str = None):
        """
        Сохранение модели

        Args:
            filepath: Путь для сохранения
        """
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = MODELS_DIR / f"pattern_model_{timestamp}.joblib"

        try:
            # Сохраняем модель и скейлер
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'config': self.config,
                'training_stats': self.training_stats,
                'is_trained': self.is_trained,
                'save_time': datetime.now()
            }

            joblib.dump(model_data, filepath)
            self.logger.info(f"Модель сохранена в {filepath}")

        except Exception as e:
            self.logger.error(f"Ошибка сохранения модели: {e}")

    def load_model(self, filepath: str):
        """
        Загрузка модели

        Args:
            filepath: Путь к файлу модели
        """
        try:
            if not Path(filepath).exists():
                self.logger.error(f"Файл модели не найден: {filepath}")
                return False

            model_data = joblib.load(filepath)

            self.model = model_data.get('model')
            self.scaler = model_data.get('scaler')
            self.config = model_data.get('config', self.config)
            self.training_stats = model_data.get('training_stats', self.training_stats)
            self.is_trained = model_data.get('is_trained', False)

            self.logger.info(f"Модель загружена из {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о модели"""
        info = {
            'is_trained': self.is_trained,
            'model_type': self.config.MODEL_TYPE,
            'training_stats': self.training_stats.copy(),
            'features_count': len(self.config.FEATURES)
        }

        if self.model is not None:
            info['model_params'] = self.model.get_params()

        return info

    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray):
        """
        Оптимизация гиперпараметров модели
        """
        # TODO: Реализовать оптимизацию с помощью GridSearchCV или Optuna
        self.logger.warning("Оптимизация гиперпараметров пока не реализована")

    def create_ensemble_model(self, models: List):
        """
        Создание ансамблевой модели

        Args:
            models: Список моделей для ансамбля
        """
        # TODO: Реализовать ансамбль моделей
        self.logger.warning("Ансамблевые модели пока не реализованы")


class PatternSuccessPredictor:
    """Предиктор успешности паттернов с использованием ML"""

    def __init__(self, ml_model: PatternMLModel = None):
        self.ml_model = ml_model or PatternMLModel()
        self.logger = logger.bind(name="PatternSuccessPredictor")

        # Кэш предсказаний
        self.prediction_cache = {}
        self.cache_max_size = 1000

        # Статистика предсказаний
        self.prediction_stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'avg_confidence': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }

    def predict_pattern_success(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Предсказание успешности паттерна

        Args:
            pattern: Паттерн для анализа

        Returns:
            Словарь с предсказанием
        """
        self.prediction_stats['total_predictions'] += 1

        # Проверяем кэш
        pattern_id = pattern.get('id', '')
        if pattern_id in self.prediction_cache:
            self.prediction_stats['cache_hits'] += 1
            return self.prediction_cache[pattern_id]

        self.prediction_stats['cache_misses'] += 1

        try:
            # Получаем предсказание от ML модели
            ml_prediction = self.ml_model.predict(pattern)

            # Комбинируем с другими методами оценки
            final_prediction = self._combine_predictions(pattern, ml_prediction)

            # Обновляем статистику
            if final_prediction.get('success', False):
                self.prediction_stats['successful_predictions'] += 1

            avg_conf = self.prediction_stats['avg_confidence']
            total = self.prediction_stats['total_predictions'] - 1
            new_conf = final_prediction.get('confidence', 0)

            self.prediction_stats['avg_confidence'] = (
                    (avg_conf * total + new_conf) / self.prediction_stats['total_predictions']
            )

            # Сохраняем в кэш
            if pattern_id:
                self._add_to_cache(pattern_id, final_prediction)

            return final_prediction

        except Exception as e:
            self.logger.error(f"Ошибка предсказания успешности: {e}")
            return {
                'success': False,
                'confidence': 0.0,
                'probability': 0.5,
                'method': 'error',
                'error': str(e)
            }

    def _combine_predictions(self, pattern: Dict[str, Any],
                             ml_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Комбинирование предсказаний разных методов"""
        predictions = []
        weights = []

        # 1. ML предсказание
        if ml_prediction.get('prediction') is not None:
            predictions.append(ml_prediction.get('probability', 0.5))
            weights.append(0.4)  # Вес ML модели

        # 2. Историческая успешность
        historical_stats = pattern.get('historical_statistics', {})
        if historical_stats.get('historical_success_rate'):
            predictions.append(historical_stats['historical_success_rate'])
            weights.append(0.3)

        # 3. Качество паттерна
        quality = pattern.get('metadata', {}).get('quality_score', 0.5)
        predictions.append(quality)
        weights.append(0.2)

        # 4. Риск/прибыль
        risk_reward = pattern.get('targets', {}).get('profit_risk_ratio', 1.0)
        rr_score = min(risk_reward / 3, 1.0)  # Нормализуем к 0-1
        predictions.append(rr_score)
        weights.append(0.1)

        # Взвешенное среднее
        if predictions:
            final_probability = sum(p * w for p, w in zip(predictions, weights)) / sum(weights)
        else:
            final_probability = 0.5

        # Определяем успешность
        success = final_probability > 0.6
        confidence = abs(final_probability - 0.5) * 2

        return {
            'success': success,
            'probability': final_probability,
            'confidence': confidence,
            'ml_prediction': ml_prediction,
            'combined_method': 'weighted_average',
            'weights': dict(zip(['ml', 'historical', 'quality', 'risk_reward'], weights))
        }

    def _add_to_cache(self, pattern_id: str, prediction: Dict[str, Any]):
        """Добавление предсказания в кэш"""
        if len(self.prediction_cache) >= self.cache_max_size:
            # Удаляем самое старое предсказание
            oldest_key = next(iter(self.prediction_cache))
            del self.prediction_cache[oldest_key]

        self.prediction_cache[pattern_id] = prediction

    def clear_cache(self):
        """Очистка кэша предсказаний"""
        self.prediction_cache.clear()
        self.logger.info("Кэш предсказаний очищен")

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики предсказаний"""
        stats = self.prediction_stats.copy()
        stats['cache_size'] = len(self.prediction_cache)
        stats['ml_model_info'] = self.ml_model.get_model_info()
        return stats

    def train_ml_model(self, historical_patterns: List[Dict[str, Any]]) -> bool:
        """
        Обучение ML модели на исторических данных

        Args:
            historical_patterns: Исторические паттерны с исходами

        Returns:
            True если обучение успешно
        """
        self.logger.info(f"Обучение ML модели на {len(historical_patterns)} исторических паттернах")

        success = self.ml_model.train(historical_patterns)

        if success:
            self.logger.info("ML модель успешно обучена")
        else:
            self.logger.warning("Не удалось обучить ML модель")

        return success

