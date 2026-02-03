import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PatternPredictor:
    """ML модель для предсказания паттернов и торговых сигналов"""
    
    def __init__(self, model_type: str = "random_forest", **kwargs):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.is_trained = False
        self.training_metrics = {}
        
        # Параметры по умолчанию
        default_params = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42
            },
            'svm': {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'probability': True,
                'random_state': 42
            },
            'neural_network': {
                'hidden_layer_sizes': (100, 50),
                'activation': 'relu',
                'alpha': 0.0001,
                'learning_rate': 'adaptive',
                'max_iter': 1000,
                'random_state': 42
            }
        }
        
        # Получаем параметры для выбранной модели
        params = default_params.get(model_type, {}).copy()
        params.update(kwargs)
        
        # Инициализация модели
        if model_type == "random_forest":
            self.model = RandomForestClassifier(**params)
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(**params)
        elif model_type == "svm":
            self.model = SVC(**params)
        elif model_type == "neural_network":
            self.model = MLPClassifier(**params)
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}")
    
    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Извлечение признаков из ценовых данных"""
        if len(data) < 20:
            return pd.DataFrame()
        
        features = pd.DataFrame(index=data.index)
        
        # Базовые ценовые признаки
        features['open'] = data['open']
        features['high'] = data['high']
        features['low'] = data['low']
        features['close'] = data['close']
        
        if 'volume' in data.columns:
            features['volume'] = data['volume']
        else:
            features['volume'] = 0
        
        # Технические индикаторы
        # 1. Простые скользящие средние
        features['sma_5'] = data['close'].rolling(window=5).mean()
        features['sma_10'] = data['close'].rolling(window=10).mean()
        features['sma_20'] = data['close'].rolling(window=20).mean()
        features['sma_50'] = data['close'].rolling(window=50).mean()
        
        # 2. Экспоненциальные скользящие средние
        features['ema_5'] = data['close'].ewm(span=5, adjust=False).mean()
        features['ema_10'] = data['close'].ewm(span=10, adjust=False).mean()
        features['ema_20'] = data['close'].ewm(span=20, adjust=False).mean()
        
        # 3. RSI (Relative Strength Index)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # 4. MACD
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # 5. Bollinger Bands
        sma_20 = data['close'].rolling(window=20).mean()
        std_20 = data['close'].rolling(window=20).std()
        features['bb_upper'] = sma_20 + (std_20 * 2)
        features['bb_lower'] = sma_20 - (std_20 * 2)
        features['bb_width'] = features['bb_upper'] - features['bb_lower']
        features['bb_position'] = (data['close'] - features['bb_lower']) / features['bb_width']
        
        # 6. ATR (Average True Range)
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        features['atr'] = true_range.rolling(window=14).mean()
        
        # 7. Volume indicators (если есть объем)
        if 'volume' in data.columns:
            features['volume_sma'] = data['volume'].rolling(window=20).mean()
            features['volume_ratio'] = data['volume'] / features['volume_sma']
            features['obv'] = (np.sign(data['close'].diff()) * data['volume']).fillna(0).cumsum()
        
        # 8. Price patterns features
        features['body_size'] = np.abs(data['close'] - data['open'])
        features['upper_shadow'] = data['high'] - np.maximum(data['open'], data['close'])
        features['lower_shadow'] = np.minimum(data['open'], data['close']) - data['low']
        features['total_range'] = data['high'] - data['low']
        features['body_ratio'] = features['body_size'] / features['total_range'].replace(0, 1)
        features['is_bullish'] = (data['close'] > data['open']).astype(int)
        
        # 9. Returns and volatility
        features['returns_1'] = data['close'].pct_change(1)
        features['returns_5'] = data['close'].pct_change(5)
        features['returns_10'] = data['close'].pct_change(10)
        features['volatility_5'] = features['returns_1'].rolling(window=5).std()
        features['volatility_20'] = features['returns_1'].rolling(window=20).std()
        
        # 10. Price position features
        features['price_vs_sma_20'] = data['close'] / features['sma_20']
        features['price_vs_sma_50'] = data['close'] / features['sma_50']
        features['sma_5_vs_20'] = features['sma_5'] / features['sma_20']
        features['sma_20_vs_50'] = features['sma_20'] / features['sma_50']
        
        # 11. Support/Resistance features
        features['high_20'] = data['high'].rolling(window=20).max()
        features['low_20'] = data['low'].rolling(window=20).min()
        features['dist_to_high'] = (features['high_20'] - data['close']) / features['high_20']
        features['dist_to_low'] = (data['close'] - features['low_20']) / features['low_20']
        
        # Заполнение пропусков
        features = features.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        # Удаляем строки с NaN после всех заполнений
        features = features.replace([np.inf, -np.inf], 0)
        
        return features
    
    def prepare_training_data(self, data: pd.DataFrame, target_col: str = 'signal',
                             lookback: int = 10, forecast_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Подготовка данных для обучения"""
        if len(data) < lookback + forecast_horizon + 10:
            logger.warning(f"Недостаточно данных для обучения: {len(data)} баров")
            return np.array([]), np.array([])
        
        # Извлечение признаков
        features = self._extract_features(data)
        
        if features.empty:
            return np.array([]), np.array([])
        
        # Создание целевой переменной
        # Предполагаем, что target_col содержит сигналы: 'buy', 'sell', 'hold'
        if target_col not in data.columns:
            logger.error(f"Целевая колонка {target_col} не найдена в данных")
            return np.array([]), np.array([])
        
        # Преобразуем сигналы в числовые значения
        signal_mapping = {'buy': 1, 'sell': 0, 'hold': 0}
        y_numeric = data[target_col].map(signal_mapping).fillna(0).astype(int)
        
        # Создание последовательностей
        X, y = [], []
        
        for i in range(lookback, len(features) - forecast_horizon):
            # Признаки - последовательность из lookback баров
            sequence = features.iloc[i-lookback:i].values
            
            # Проверяем, нет ли NaN в последовательности
            if not np.any(np.isnan(sequence)):
                X.append(sequence.flatten())  # Выпрямляем в вектор
                
                # Целевая переменная - сигнал через forecast_horizon баров
                if i + forecast_horizon < len(y_numeric):
                    target = y_numeric.iloc[i + forecast_horizon]
                    y.append(target)
        
        if not X or not y:
            logger.warning("Не удалось создать обучающие данные")
            return np.array([]), np.array([])
        
        X_array = np.array(X)
        y_array = np.array(y)
        
        logger.info(f"Подготовлены данные: X.shape={X_array.shape}, y.shape={y_array.shape}")
        logger.info(f"Распределение классов: {np.bincount(y_array)}")
        
        return X_array, y_array
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
             optimize: bool = False, cv_folds: int = 5) -> Dict[str, Any]:
        """Обучение модели"""
        if len(X) == 0 or len(y) == 0:
            logger.error("Нет данных для обучения")
            return {}
        
        if len(np.unique(y)) < 2:
            logger.warning("Только один класс в целевой переменной")
            return {}
        
        try:
            # Для временных рядов используем временное разделение
            if optimize and len(X) > 100:
                # Используем TimeSeriesSplit для кросс-валидации
                tscv = TimeSeriesSplit(n_splits=min(cv_folds, 5))
                
                # Разделение на train/validation
                split_idx = int(len(X) * (1 - test_size))
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]
                
                # Масштабирование признаков
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_val_scaled = self.scaler.transform(X_val)
                
                # Оптимизация гиперпараметров
                param_grid = self._get_param_grid()
                
                if param_grid:
                    grid_search = GridSearchCV(
                        self.model, param_grid, cv=tscv, 
                        scoring='f1', n_jobs=-1, verbose=0
                    )
                    grid_search.fit(X_train_scaled, y_train)
                    self.model = grid_search.best_estimator_
                    logger.info(f"Лучшие параметры: {grid_search.best_params_}")
                    logger.info(f"Лучший score: {grid_search.best_score_:.4f}")
                else:
                    self.model.fit(X_train_scaled, y_train)
            else:
                # Простое разделение на train/test
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                
                # Масштабирование признаков
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_val_scaled = self.scaler.transform(X_val)
                
                if optimize and hasattr(self.model, 'get_params'):
                    param_grid = self._get_param_grid()
                    if param_grid:
                        grid_search = GridSearchCV(
                            self.model, param_grid, cv=3, 
                            scoring='f1', n_jobs=-1, verbose=0
                        )
                        grid_search.fit(X_train_scaled, y_train)
                        self.model = grid_search.best_estimator_
                        logger.info(f"Лучшие параметры: {grid_search.best_params_}")
                    else:
                        self.model.fit(X_train_scaled, y_train)
                else:
                    self.model.fit(X_train_scaled, y_train)
            
            # Предсказание и оценка
            y_pred = self.model.predict(X_val_scaled)
            y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]
            
            # Метрики
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            # ROC-AUC если есть вероятности
            try:
                roc_auc = roc_auc_score(y_val, y_pred_proba)
            except:
                roc_auc = 0.0
            
            # Матрица ошибок
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_val, y_pred)
            
            # Важность признаков (если доступна)
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = dict(zip(
                    range(X.shape[1]), self.model.feature_importances_
                ))
            
            self.training_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'confusion_matrix': cm.tolist(),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'class_distribution_train': np.bincount(y_train).tolist(),
                'class_distribution_val': np.bincount(y_val).tolist()
            }
            
            self.is_trained = True
            
            logger.info(f"Модель обучена. Метрики на валидации:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1-Score: {f1:.4f}")
            logger.info(f"  ROC-AUC: {roc_auc:.4f}")
            
            return self.training_metrics
            
        except Exception as e:
            logger.error(f"Ошибка обучения модели: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def _get_param_grid(self) -> Dict:
        """Параметры для GridSearchCV"""
        if self.model_type == "random_forest":
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif self.model_type == "gradient_boosting":
            return {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
        elif self.model_type == "svm":
            return {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.1, 1]
            }
        elif self.model_type == "neural_network":
            return {
                'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        return {}
    
    def predict(self, X: np.ndarray, return_proba: bool = False) -> np.ndarray:
        """Предсказание на новых данных"""
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала вызовите метод train().")
        
        if len(X) == 0:
            return np.array([])
        
        X_scaled = self.scaler.transform(X)
        
        if return_proba:
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X_scaled)
            else:
                # Для моделей без predict_proba возвращаем псевдовероятности
                predictions = self.model.predict(X_scaled)
                proba = np.zeros((len(predictions), 2))
                proba[:, 1] = predictions
                proba[:, 0] = 1 - predictions
                return proba
        else:
            return self.model.predict(X_scaled)
    
    def predict_signal(self, data: pd.DataFrame, lookback: int = 10) -> Dict[str, Any]:
        """Предсказание торгового сигнала на основе последних данных"""
        if not self.is_trained:
            return {
                'signal': 'hold',
                'confidence': 0.0,
                'buy_probability': 0.5,
                'model_type': self.model_type,
                'error': 'Модель не обучена'
            }
        
        # Извлечение признаков
        features = self._extract_features(data)
        
        if features.empty or len(features) < lookback:
            return {
                'signal': 'hold',
                'confidence': 0.0,
                'buy_probability': 0.5,
                'model_type': self.model_type,
                'error': 'Недостаточно данных для предсказания'
            }
        
        # Последние lookback баров
        sequence = features.iloc[-lookback:].values.flatten().reshape(1, -1)
        
        # Предсказание
        try:
            proba = self.predict(sequence, return_proba=True)
            buy_probability = proba[0, 1]  # Вероятность класса "buy"
            
            # Формирование сигнала на основе порогов
            if buy_probability > 0.6:  # Порог для покупки
                signal = 'buy'
                confidence = buy_probability
            elif buy_probability < 0.4:  # Порог для продажи
                signal = 'sell'
                confidence = 1 - buy_probability
            else:
                signal = 'hold'
                confidence = 0.5
            
            return {
                'signal': signal,
                'confidence': float(confidence),
                'buy_probability': float(buy_probability),
                'model_type': self.model_type,
                'timestamp': data.index[-1] if not data.empty else None,
                'features_used': features.shape[1]
            }
        except Exception as e:
            logger.error(f"Ошибка предсказания сигнала: {e}")
            return {
                'signal': 'hold',
                'confidence': 0.0,
                'buy_probability': 0.5,
                'model_type': self.model_type,
                'error': str(e)
            }
    
    def save_model(self, path: str):
        """Сохранение модели"""
        if not self.is_trained:
            logger.warning("Модель не обучена, сохранение пропущено")
            return
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Модель сохранена: {path}")
    
    def load_model(self, path: str):
        """Загрузка модели"""
        if not os.path.exists(path):
            logger.error(f"Файл модели не найден: {path}")
            return
        
        try:
            model_data = joblib.load(path)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.model_type = model_data['model_type']
            self.is_trained = model_data['is_trained']
            self.feature_importance = model_data.get('feature_importance', {})
            self.training_metrics = model_data.get('training_metrics', {})
            
            logger.info(f"Модель загружена: {path}")
            logger.info(f"Тип модели: {self.model_type}")
            logger.info(f"Метрики обучения: {self.training_metrics.get('accuracy', 0):.4f}")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
    
    def get_feature_importance(self, top_n: int = 20) -> List[Dict]:
        """Получение важности признаков"""
        if not self.feature_importance:
            return []
        
        # Сортируем признаки по важности
        sorted_features = sorted(
            self.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_n]
        
        return [
            {'feature_index': idx, 'importance': importance}
            for idx, importance in sorted_features
        ]
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Оценка модели на тестовых данных"""
        if not self.is_trained:
            return {'error': 'Модель не обучена'}
        
        if len(X_test) == 0 or len(y_test) == 0:
            return {'error': 'Нет тестовых данных'}
        
        try:
            X_test_scaled = self.scaler.transform(X_test)
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
            
            # Метрики
            from sklearn.metrics import classification_report, roc_curve, auc
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'test_samples': len(X_test),
                'predictions': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist()
            }
            
        except Exception as e:
            logger.error(f"Ошибка оценки модели: {e}")
            return {'error': str(e)}

