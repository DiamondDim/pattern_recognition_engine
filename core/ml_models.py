"""
Модуль машинного обучения для классификации паттернов
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import warnings
from datetime import datetime
import os

# Исправляем импорт для обратной совместимости
try:
    from config import config, ML_CONFIG, ML_MODEL_CONFIG
except ImportError:
    # Для обратной совместимости
    try:
        from config import ML_CONFIG as ML_MODEL_CONFIG
    except ImportError:
        # Создаем fallback конфиг
        ML_MODEL_CONFIG = type('Config', (), {
            'MODEL_TYPE': 'random_forest',
            'RANDOM_FOREST_PARAMS': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            'NEURAL_NETWORK_PARAMS': {
                'hidden_layer_sizes': (64, 32),
                'activation': 'relu',
                'solver': 'adam',
                'max_iter': 1000,
                'random_state': 42
            },
            'FEATURE_WINDOW': 20,
            'USE_TECHNICAL_INDICATORS': True
        })()


class MLModel:
    """Базовый класс модели машинного обучения"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.ml_config = ML_MODEL_CONFIG
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False

    def create_model(self) -> Any:
        """Создание модели в зависимости от конфигурации"""
        model_type = self.ml_config.MODEL_TYPE

        if model_type == 'random_forest':
            params = self.ml_config.RANDOM_FOREST_PARAMS
            self.model = RandomForestClassifier(**params)

        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            )

        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42
            )

        elif model_type == 'neural_network':
            params = self.ml_config.NEURAL_NETWORK_PARAMS
            self.model = MLPClassifier(**params)

        else:
            # По умолчанию используем Random Forest
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

        return self.model

    def extract_features(self, price_data: pd.DataFrame, pattern_data: Dict = None) -> pd.DataFrame:
        """
        Извлечение признаков из ценовых данных

        Args:
            price_data: DataFrame с OHLC данными
            pattern_data: Данные паттерна (опционально)

        Returns:
            DataFrame с признаками
        """
        features = pd.DataFrame(index=price_data.index)

        # Базовые ценовые признаки
        features['returns'] = price_data['close'].pct_change()
        features['log_returns'] = np.log(price_data['close'] / price_data['close'].shift(1))

        # Волатильность
        features['volatility'] = price_data['close'].rolling(20).std()
        features['atr'] = self._calculate_atr(price_data)

        # Технические индикаторы
        if self.ml_config.USE_TECHNICAL_INDICATORS:
            # Скользящие средние
            features['sma_10'] = price_data['close'].rolling(10).mean()
            features['sma_20'] = price_data['close'].rolling(20).mean()
            features['sma_50'] = price_data['close'].rolling(50).mean()

            # RSI
            features['rsi'] = self._calculate_rsi(price_data['close'])

            # MACD
            macd, signal = self._calculate_macd(price_data['close'])
            features['macd'] = macd
            features['macd_signal'] = signal
            features['macd_histogram'] = macd - signal

        # Объем
        if 'volume' in price_data.columns:
            features['volume'] = price_data['volume']
            features['volume_sma'] = price_data['volume'].rolling(20).mean()
            features['volume_ratio'] = price_data['volume'] / features['volume_sma']

        # Признаки из паттерна
        if pattern_data and self.ml_config.USE_PATTERN_FEATURES:
            features = self._add_pattern_features(features, pattern_data)

        # Лаговые признаки
        window = self.ml_config.FEATURE_WINDOW
        for lag in range(1, min(6, window)):
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)

        # Удаляем NaN значения
        features = features.dropna()

        # Сохраняем названия столбцов признаков
        self.feature_columns = features.columns.tolist()

        return features

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Расчет Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        return atr

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Расчет Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(self, prices: pd.Series,
                       fast_period: int = 12,
                       slow_period: int = 26,
                       signal_period: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Расчет MACD"""
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()

        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()

        return macd, signal

    def _add_pattern_features(self, features: pd.DataFrame, pattern_data: Dict) -> pd.DataFrame:
        """Добавление признаков из данных паттерна"""
        # Базовые признаки паттерна
        if 'quality' in pattern_data:
            features['pattern_quality'] = pattern_data['quality']

        if 'type' in pattern_data:
            # Кодируем тип паттерна
            pattern_type = pattern_data['type']
            type_mapping = {
                'candlestick': 1,
                'geometric': 2,
                'harmonic': 3
            }
            features['pattern_type'] = type_mapping.get(pattern_type, 0)

        if 'direction' in pattern_data:
            # Кодируем направление
            direction = pattern_data['direction']
            direction_mapping = {
                'bullish': 1,
                'bearish': -1,
                'neutral': 0
            }
            features['pattern_direction'] = direction_mapping.get(direction, 0)

        return features

    def prepare_training_data(self, features: pd.DataFrame,
                            target: pd.Series) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Подготовка данных для обучения

        Args:
            features: DataFrame с признаками
            target: Серия с целевой переменной

        Returns:
            Кортеж с признаками, целевой переменной и именами признаков
        """
        # Объединяем признаки и цель
        data = features.copy()
        data['target'] = target

        # Удаляем строки с NaN
        data = data.dropna()

        if len(data) == 0:
            return np.array([]), np.array([]), []

        # Разделяем признаки и цель
        X = data.drop('target', axis=1).values
        y = data['target'].values

        # Сохраняем имена признаков
        feature_names = data.drop('target', axis=1).columns.tolist()

        return X, y, feature_names

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Обучение модели

        Args:
            X: Матрица признаков
            y: Вектор целевой переменной

        Returns:
            Метрики обучения
        """
        if len(X) == 0 or len(y) == 0:
            return {}

        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

        # Масштабирование признаков
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Создание и обучение модели
        if self.model is None:
            self.create_model()

        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Оценка модели
        y_pred = self.model.predict(X_test_scaled)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }

        return metrics

    def predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Прогнозирование на новых данных

        Args:
            features: DataFrame с признаками

        Returns:
            Кортеж с предсказаниями и вероятностями
        """
        if not self.is_trained or self.model is None:
            return np.array([]), np.array([])

        # Проверяем, что есть все необходимые признаки
        missing_cols = set(self.feature_columns) - set(features.columns)
        if missing_cols:
            # Добавляем недостающие колонки с нулями
            for col in missing_cols:
                features[col] = 0

        # Упорядочиваем колонки как при обучении
        features = features[self.feature_columns]

        # Удаляем NaN
        features = features.dropna()

        if len(features) == 0:
            return np.array([]), np.array([])

        # Масштабирование
        X_scaled = self.scaler.transform(features.values)

        # Прогнозирование
        predictions = self.model.predict(X_scaled)

        # Вероятности (если модель их поддерживает)
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)
        else:
            probabilities = np.zeros((len(predictions), 2))

        return predictions, probabilities

    def save_model(self, filepath: str):
        """Сохранение модели на диск"""
        if self.model is not None:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'config': self.config
            }, filepath)

    def load_model(self, filepath: str) -> bool:
        """Загрузка модели с диска"""
        try:
            if os.path.exists(filepath):
                data = joblib.load(filepath)
                self.model = data['model']
                self.scaler = data['scaler']
                self.feature_columns = data['feature_columns']
                self.config = data.get('config', {})
                self.is_trained = True
                return True
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")

        return False


class PatternClassifier(MLModel):
    """Классификатор паттернов"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.pattern_classes = ['bullish', 'bearish', 'neutral']
        self.class_mapping = {cls: i for i, cls in enumerate(self.pattern_classes)}
        self.reverse_mapping = {i: cls for i, cls in enumerate(self.pattern_classes)}

    def prepare_pattern_data(self, patterns: List[Dict], price_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Подготовка данных паттернов для обучения

        Args:
            patterns: Список паттернов
            price_data: DataFrame с ценовыми данными

        Returns:
            Кортеж с признаками и целевой переменной
        """
        features_list = []
        targets_list = []

        for pattern in patterns:
            # Извлекаем признаки для паттерна
            pattern_features = self.extract_features(price_data, pattern)

            if len(pattern_features) > 0:
                # Берем последнюю строку (момент паттерна)
                latest_features = pattern_features.iloc[-1:]

                # Определяем целевую переменную
                pattern_direction = pattern.get('direction', 'neutral')
                target = self.class_mapping.get(pattern_direction, 2)  # neutral по умолчанию

                features_list.append(latest_features)
                targets_list.append(target)

        if features_list:
            all_features = pd.concat(features_list, ignore_index=True)
            all_targets = pd.Series(targets_list)
            return all_features, all_targets

        return pd.DataFrame(), pd.Series([])

    def classify_pattern(self, pattern: Dict, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Классификация паттерна

        Args:
            pattern: Данные паттерна
            price_data: Ценовые данные

        Returns:
            Результаты классификации
        """
        if not self.is_trained:
            return {'error': 'Модель не обучена'}

        # Извлекаем признаки
        features = self.extract_features(price_data, pattern)

        if len(features) == 0:
            return {'error': 'Не удалось извлечь признаки'}

        # Прогнозирование
        predictions, probabilities = self.predict(features)

        if len(predictions) == 0:
            return {'error': 'Не удалось сделать прогноз'}

        # Интерпретация результатов
        predicted_class_idx = predictions[-1]
        predicted_class = self.reverse_mapping.get(predicted_class_idx, 'neutral')

        result = {
            'predicted_class': predicted_class,
            'confidence': 0.0,
            'probabilities': {}
        }

        if len(probabilities) > 0:
            for i, cls in enumerate(self.pattern_classes):
                result['probabilities'][cls] = float(probabilities[-1, i])

            result['confidence'] = float(np.max(probabilities[-1]))

        return result

