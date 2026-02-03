"""
Модуль машинного обучения для распознавания паттернов
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import joblib
from pathlib import Path
import warnings

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from config import ML_CONFIG, DATA_DIR
from utils.logger import logger


class PatternClassifier:
    """Классификатор паттернов"""

    def __init__(self, model_type: str = 'random_forest', config: ML_CONFIG = None):
        self.config = config or ML_CONFIG
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importances_ = None
        self.classes_ = None
        self.logger = logger.bind(module="PatternClassifier")

        self._initialize_model()

    def _initialize_model(self):
        """Инициализация модели"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=self.config.RANDOM_FOREST.N_ESTIMATORS,
                max_depth=self.config.RANDOM_FOREST.MAX_DEPTH,
                min_samples_split=self.config.RANDOM_FOREST.MIN_SAMPLES_SPLIT,
                min_samples_leaf=self.config.RANDOM_FOREST.MIN_SAMPLES_LEAF,
                random_state=self.config.RANDOM_SEED,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=self.config.GRADIENT_BOOSTING.N_ESTIMATORS,
                learning_rate=self.config.GRADIENT_BOOSTING.LEARNING_RATE,
                max_depth=self.config.GRADIENT_BOOSTING.MAX_DEPTH,
                random_state=self.config.RANDOM_SEED
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                C=self.config.SVM.C,
                kernel=self.config.SVM.KERNEL,
                probability=True,
                random_state=self.config.RANDOM_SEED
            )
        elif self.model_type == 'mlp':
            self.model = MLPClassifier(
                hidden_layer_sizes=self.config.MLP.HIDDEN_LAYERS,
                activation=self.config.MLP.ACTIVATION,
                solver=self.config.MLP.SOLVER,
                alpha=self.config.MLP.ALPHA,
                learning_rate=self.config.MLP.LEARNING_RATE,
                max_iter=self.config.MLP.MAX_ITER,
                random_state=self.config.RANDOM_SEED
            )
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=self.config.XGBOOST.N_ESTIMATORS,
                max_depth=self.config.XGBOOST.MAX_DEPTH,
                learning_rate=self.config.XGBOOST.LEARNING_RATE,
                subsample=self.config.XGBOOST.SUBSAMPLE,
                colsample_bytree=self.config.XGBOOST.COLSAMPLE_BYTREE,
                random_state=self.config.RANDOM_SEED,
                n_jobs=-1
            )
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(
                n_estimators=self.config.LIGHTGBM.N_ESTIMATORS,
                max_depth=self.config.LIGHTGBM.MAX_DEPTH,
                learning_rate=self.config.LIGHTGBM.LEARNING_RATE,
                num_leaves=self.config.LIGHTGBM.NUM_LEAVES,
                random_state=self.config.RANDOM_SEED,
                n_jobs=-1
            )
        elif self.model_type == 'catboost':
            self.model = cb.CatBoostClassifier(
                iterations=self.config.CATBOOST.ITERATIONS,
                depth=self.config.CATBOOST.DEPTH,
                learning_rate=self.config.CATBOOST.LEARNING_RATE,
                random_seed=self.config.RANDOM_SEED,
                verbose=False
            )
        else:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}")

    def prepare_features(self, patterns: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготовка признаков для обучения

        Args:
            patterns: Список паттернов

        Returns:
            Матрица признаков и метки классов
        """
        if not patterns:
            return np.array([]), np.array([])

        features = []
        labels = []

        for pattern in patterns:
            # Извлекаем признаки
            pattern_features = self._extract_pattern_features(pattern)
            if pattern_features is not None:
                features.append(pattern_features)

                # Извлекаем метку (направление паттерна)
                direction = pattern.get('direction', 'neutral')
                label = self._encode_direction(direction)
                labels.append(label)

        if not features:
            return np.array([]), np.array([])

        X = np.array(features)
        y = np.array(labels)

        # Масштабирование признаков
        if len(X) > 0:
            X = self.scaler.fit_transform(X)

        return X, y

    def _extract_pattern_features(self, pattern: Dict[str, Any]) -> Optional[np.ndarray]:
        """Извлечение признаков из паттерна"""
        try:
            points = pattern.get('points', [])
            statistics = pattern.get('statistics', {})
            targets = pattern.get('targets', {})
            metadata = pattern.get('metadata', {})

            if len(points) < 2:
                return None

            # Координаты точек
            point_prices = [p.get('price', 0) for p in points]
            point_indices = [p.get('index', 0) for p in points]

            # Базовые статистики
            min_price = min(point_prices)
            max_price = max(point_prices)
            price_range = max_price - min_price
            avg_price = np.mean(point_prices)

            # Относительные координаты
            normalized_prices = [(p - min_price) / price_range if price_range > 0 else 0
                               for p in point_prices]

            # Углы между точками
            angles = []
            if len(points) >= 3:
                for i in range(len(points) - 2):
                    p1 = np.array([point_indices[i], point_prices[i]])
                    p2 = np.array([point_indices[i+1], point_prices[i+1]])
                    p3 = np.array([point_indices[i+2], point_prices[i+2]])

                    v1 = p2 - p1
                    v2 = p3 - p2

                    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        cos_angle = np.clip(cos_angle, -1, 1)
                        angle = np.arccos(cos_angle)
                        angles.append(angle)
                    else:
                        angles.append(0)

            # Признаки из статистики
            hist_matches = statistics.get('historical_matches', 0)
            hist_success = statistics.get('historical_success_rate', 0)
            quality_score = metadata.get('quality_score', 0)
            confidence = metadata.get('confidence', 0)

            # Признаки из целевых уровней
            entry_price = targets.get('entry_price', 0)
            stop_loss = targets.get('stop_loss', 0)
            take_profit = targets.get('take_profit', 0)

            if entry_price > 0 and stop_loss > 0:
                risk = abs(entry_price - stop_loss)
                reward = abs(take_profit - entry_price) if take_profit > 0 else 0
                risk_reward = reward / risk if risk > 0 else 0
            else:
                risk_reward = 0

            # Формируем вектор признаков
            feature_vector = [
                # Базовые характеристики
                len(points),
                min_price,
                max_price,
                price_range,
                avg_price,
                np.std(point_prices) if len(point_prices) > 1 else 0,

                # Относительные координаты
                np.mean(normalized_prices),
                np.std(normalized_prices),

                # Угловые характеристики
                np.mean(angles) if angles else 0,
                np.std(angles) if angles else 0,

                # Статистические признаки
                hist_matches,
                hist_success,
                quality_score,
                confidence,

                # Риск-менеджмент
                risk_reward,

                # Временные характеристики
                metadata.get('timeframe_multiplier', 1),
                metadata.get('pattern_age', 0)
            ]

            # Добавляем индивидуальные координаты точек
            for i in range(min(10, len(normalized_prices))):
                feature_vector.append(normalized_prices[i] if i < len(normalized_prices) else 0)

            return np.array(feature_vector)

        except Exception as e:
            self.logger.error(f"Ошибка извлечения признаков: {e}")
            return None

    def _encode_direction(self, direction: str) -> int:
        """Кодирование направления паттерна"""
        direction_map = {
            'bullish': 1,
            'bearish': 0,
            'neutral': 2
        }
        return direction_map.get(direction.lower(), 2)

    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Обучение модели

        Args:
            X: Матрица признаков
            y: Метки классов
            validation_split: Доля данных для валидации

        Returns:
            Словарь с метриками обучения
        """
        if len(X) == 0 or len(y) == 0:
            self.logger.error("Нет данных для обучения")
            return {}

        # Разделение данных
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=validation_split,
            random_state=self.config.RANDOM_SEED,
            stratify=y if len(np.unique(y)) > 1 else None
        )

        # Обучение модели
        self.model.fit(X_train, y_train)

        # Сохранение классов
        self.classes_ = self.model.classes_

        # Предсказания
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)

        # Вычисление метрик
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'train_precision': precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
            'val_precision': precision_score(y_val, y_val_pred, average='weighted', zero_division=0),
            'train_recall': recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
            'val_recall': recall_score(y_val, y_val_pred, average='weighted', zero_division=0),
            'train_f1': f1_score(y_train, y_train_pred, average='weighted', zero_division=0),
            'val_f1': f1_score(y_val, y_val_pred, average='weighted', zero_division=0),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'classes': list(self.classes_)
        }

        # Важность признаков
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
            metrics['feature_importances'] = self.feature_importances_.tolist()

        # Матрица ошибок
        metrics['confusion_matrix'] = confusion_matrix(y_val, y_val_pred).tolist()

        self.logger.info(f"Обучение завершено. Точность на валидации: {metrics['val_accuracy']:.4f}")

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание классов

        Args:
            X: Матрица признаков

        Returns:
            Предсказанные классы
        """
        if self.model is None:
            raise ValueError("Модель не обучена")

        if len(X) == 0:
            return np.array([])

        # Масштабирование
        X_scaled = self.scaler.transform(X)

        # Предсказание
        predictions = self.model.predict(X_scaled)

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание вероятностей классов

        Args:
            X: Матрица признаков

        Returns:
            Вероятности классов
        """
        if self.model is None:
            raise ValueError("Модель не обучена")

        if len(X) == 0:
            return np.array([])

        # Масштабирование
        X_scaled = self.scaler.transform(X)

        # Предсказание вероятностей
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)
        else:
            # Для моделей без predict_proba
            predictions = self.predict(X_scaled)
            probabilities = np.zeros((len(predictions), len(self.classes_)))
            for i, pred in enumerate(predictions):
                probabilities[i, pred] = 1.0

        return probabilities

    def save_model(self, filepath: str):
        """
        Сохранение модели

        Args:
            filepath: Путь для сохранения
        """
        if self.model is None:
            self.logger.warning("Модель не обучена, нечего сохранять")
            return

        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type,
                'feature_importances': self.feature_importances_,
                'classes': self.classes_
            }

            joblib.dump(model_data, filepath)
            self.logger.info(f"Модель сохранена: {filepath}")

        except Exception as e:
            self.logger.error(f"Ошибка сохранения модели: {e}")

    def load_model(self, filepath: str):
        """
        Загрузка модели

        Args:
            filepath: Путь к файлу модели
        """
        try:
            model_data = joblib.load(filepath)

            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.model_type = model_data['model_type']
            self.feature_importances_ = model_data.get('feature_importances')
            self.classes_ = model_data.get('classes')

            self.logger.info(f"Модель загружена: {filepath}")

        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели: {e}")


class PatternClusterer:
    """Кластеризация паттернов"""

    def __init__(self, method: str = 'kmeans', n_clusters: int = 5):
        self.method = method
        self.n_clusters = n_clusters
        self.model = None
        self.scaler = StandardScaler()
        self.labels_ = None
        self.logger = logger.bind(module="PatternClusterer")

        self._initialize_model()

    def _initialize_model(self):
        """Инициализация модели кластеризации"""
        if self.method == 'kmeans':
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )
        elif self.method == 'dbscan':
            self.model = DBSCAN(
                eps=0.5,
                min_samples=5
            )
        else:
            raise ValueError(f"Неизвестный метод кластеризации: {self.method}")

    def fit(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Кластеризация паттернов

        Args:
            patterns: Список паттернов

        Returns:
            Результаты кластеризации
        """
        if not patterns:
            return {}

        # Подготовка признаков
        classifier = PatternClassifier()
        X, _ = classifier.prepare_features(patterns)

        if len(X) == 0:
            return {}

        # Применение PCA для визуализации
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # Кластеризация
        self.labels_ = self.model.fit_predict(X)

        # Статистика по кластерам
        cluster_stats = {}
        unique_labels = np.unique(self.labels_)

        for label in unique_labels:
            if label == -1:  # Для DBSCAN - шум
                continue

            cluster_indices = np.where(self.labels_ == label)[0]
            cluster_patterns = [patterns[i] for i in cluster_indices]

            # Статистика паттернов в кластере
            directions = [p.get('direction', 'neutral') for p in cluster_patterns]
            bullish_count = directions.count('bullish')
            bearish_count = directions.count('bearish')

            success_rates = [p.get('statistics', {}).get('historical_success_rate', 0)
                           for p in cluster_patterns]

            cluster_stats[label] = {
                'size': len(cluster_indices),
                'bullish_ratio': bullish_count / len(cluster_indices) if cluster_indices.size > 0 else 0,
                'bearish_ratio': bearish_count / len(cluster_indices) if cluster_indices.size > 0 else 0,
                'avg_success_rate': np.mean(success_rates) if success_rates else 0,
                'patterns_indices': cluster_indices.tolist()
            }

        return {
            'labels': self.labels_.tolist(),
            'cluster_stats': cluster_stats,
            'pca_components': X_pca.tolist(),
            'explained_variance': pca.explained_variance_ratio_.tolist()
        }

