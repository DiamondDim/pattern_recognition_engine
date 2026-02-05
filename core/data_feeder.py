import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib
import pickle
import os
from typing import Optional, Dict, Any
import logging

# Настройка логирования
logger = logging.getLogger(__name__)

try:
    from utils.mt5_connector import mt5_connector
except ImportError:
    # Для тестового режима
    mt5_connector = None
    logger.warning("MT5Connector не найден, используется тестовый режим")

class DataFeeder:
    """
    Класс для получения и подготовки данных для анализа
    """
    
    def __init__(self, cache_enabled: bool = True, cache_dir: str = "data_cache"):
        """
        Инициализация фидера данных
        
        Args:
            cache_enabled (bool): Включить кэширование данных
            cache_dir (str): Директория для кэша
        """
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir
        self.data_cache = {}
        
        # Создаем директорию для кэша, если нужно
        if cache_enabled and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Создана директория кэша: {cache_dir}")
            
    def get_data(self, symbol: str, timeframe: str, bars: int = 1000, 
                 from_date: Optional[datetime] = None, 
                 to_date: Optional[datetime] = None,
                 use_cache: bool = True) -> pd.DataFrame:
        """
        Получение данных для указанного символа и таймфрейма
        
        Args:
            symbol (str): Имя символа (например, 'EURUSD')
            timeframe (str): Таймфрейм ('M1', 'M5', 'H1', etc.)
            bars (int): Количество баров для получения
            from_date (datetime): Начальная дата (если указана, игнорируется bars)
            to_date (datetime): Конечная дата
            use_cache (bool): Использовать кэш
            
        Returns:
            pd.DataFrame: DataFrame с данными
        """
        cache_key = None
        
        # Генерируем ключ кэша
        if self.cache_enabled and use_cache:
            cache_key = self._generate_cache_key(symbol, timeframe, bars, from_date, to_date)
            
            # Проверяем кэш в памяти
            if cache_key in self.data_cache:
                logger.debug(f"Данные найдены в кэше памяти: {cache_key}")
                return self.data_cache[cache_key].copy()
                
            # Проверяем кэш на диске
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                logger.debug(f"Данные загружены из кэша на диске: {cache_key}")
                self.data_cache[cache_key] = cached_data
                return cached_data.copy()
        
        # Получаем данные из MT5
        logger.info(f"Загрузка данных: {symbol} {timeframe}, баров: {bars}")
        
        if mt5_connector is None:
            logger.warning("MT5Connector недоступен, генерируем тестовые данные")
            df = self._generate_test_data(symbol, timeframe, bars)
        else:
            df = mt5_connector.get_rates(symbol, timeframe, bars)
            
        if df.empty:
            logger.warning(f"Получены пустые данные для {symbol} {timeframe}")
            return df
            
        # Применяем фильтры по дате, если указаны
        if from_date is not None:
            df = df[df.index >= from_date]
        if to_date is not None:
            df = df[df.index <= to_date]
            
        # Добавляем технические индикаторы
        df = self._add_technical_indicators(df)
        
        # Кэшируем данные
        if self.cache_enabled and use_cache and cache_key is not None:
            self.data_cache[cache_key] = df.copy()
            self._save_to_cache(cache_key, df)
            logger.debug(f"Данные сохранены в кэш: {cache_key}")
            
        return df
        
    def _generate_cache_key(self, symbol: str, timeframe: str, bars: int,
                           from_date: Optional[datetime], 
                           to_date: Optional[datetime]) -> str:
        """
        Генерация ключа для кэша
        
        Args:
            symbol (str): Имя символа
            timeframe (str): Таймфрейм
            bars (int): Количество баров
            from_date (datetime): Начальная дата
            to_date (datetime): Конечная дата
            
        Returns:
            str: Ключ кэша
        """
        key_parts = [
            symbol,
            timeframe,
            str(bars),
            str(from_date) if from_date else 'None',
            str(to_date) if to_date else 'None',
            datetime.now().strftime('%Y%m%d')  # Дата для инвалидации кэша
        ]
        
        key_string = '_'.join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
        
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        Загрузка данных из кэша на диске
        
        Args:
            cache_key (str): Ключ кэша
            
        Returns:
            Optional[pd.DataFrame]: DataFrame из кэша или None
        """
        if not self.cache_enabled:
            return None
            
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            try:
                # Проверяем возраст файла (максимум 1 день)
                file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
                if file_age.days > 1:
                    logger.debug(f"Файл кэша устарел: {cache_file}")
                    return None
                    
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    
                if isinstance(data, pd.DataFrame) and not data.empty:
                    return data
                    
            except Exception as e:
                logger.warning(f"Ошибка загрузки из кэша {cache_file}: {e}")
                
        return None
        
    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """
        Сохранение данных в кэш на диске
        
        Args:
            cache_key (str): Ключ кэша
            data (pd.DataFrame): Данные для сохранения
        """
        if not self.cache_enabled:
            return
            
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Ошибка сохранения в кэш {cache_file}: {e}")
            
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление технических индикаторов к данным
        
        Args:
            df (pd.DataFrame): Исходные данные
            
        Returns:
            pd.DataFrame: Данные с индикаторами
        """
        if df.empty:
            return df
            
        # Создаем копию, чтобы избежать предупреждений
        df = df.copy()
        
        try:
            # Простые скользящие средние
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # Экспоненциальные скользящие средние
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
            
            # ATR (Average True Range)
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['ATR'] = true_range.rolling(window=14).mean()
            
            # Volume indicators
            if 'Volume' in df.columns:
                df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
                df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            # Price momentum
            df['Momentum'] = df['Close'].diff(5)
            df['ROC'] = (df['Close'].diff(10) / df['Close'].shift(10)) * 100
            
            # Volatility
            df['Volatility'] = df['Close'].rolling(window=20).std() / df['Close'].rolling(window=20).mean()
            
            # Support and Resistance levels (простой расчет)
            df['Resistance'] = df['High'].rolling(window=20).max()
            df['Support'] = df['Low'].rolling(window=20).min()
            
            # Заполняем NaN значения
            df = df.fillna(method='bfill').fillna(method='ffill')
            
            logger.debug(f"Добавлены технические индикаторы: {len(df.columns)} колонок")
            
        except Exception as e:
            logger.error(f"Ошибка при добавлении индикаторов: {e}")
            
        return df
        
    def _generate_test_data(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        """
        Генерация тестовых данных для отладки
        
        Args:
            symbol (str): Имя символа
            timeframe (str): Таймфрейм
            bars (int): Количество баров
            
        Returns:
            pd.DataFrame: Тестовые данные
        """
        logger.info(f"Генерация тестовых данных для {symbol} ({bars} баров)")
        
        # Создаем временную шкалу
        end_date = datetime.now()
        
        # Определяем интервал между барами
        timeframe_minutes = {
            'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
            'H1': 60, 'H4': 240, 'D1': 1440, 'W1': 10080
        }
        
        minutes_per_bar = timeframe_minutes.get(timeframe.upper(), 1)
        start_date = end_date - timedelta(minutes=bars * minutes_per_bar)
        
        dates = pd.date_range(start=start_date, end=end_date, periods=bars)
        
        # Генерируем случайные цены с трендом
        np.random.seed(42)  # Для воспроизводимости
        
        base_price = 100.0 if 'USD' in symbol else 1.0
        volatility = 0.01  # 1% волатильность
        
        # Создаем тренд
        trend = np.linspace(0, 0.1, bars)  # 10% тренд за весь период
        
        # Генерируем случайные изменения
        returns = np.random.normal(0, volatility, bars) + trend
        
        # Рассчитываем цены
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Создаем OHLC данные
        df = pd.DataFrame(index=dates)
        df['Close'] = prices
        
        # Генерируем High/Low на основе Close
        df['High'] = df['Close'] * (1 + np.abs(np.random.normal(0, volatility/2, bars)))
        df['Low'] = df['Close'] * (1 - np.abs(np.random.normal(0, volatility/2, bars)))
        
        # Open немного отличается от предыдущего Close
        df['Open'] = df['Close'].shift(1) * (1 + np.random.normal(0, volatility/4, bars))
        df['Open'].iloc[0] = df['Close'].iloc[0] * (1 + np.random.normal(0, volatility/4))
        
        # Объемы
        df['Volume'] = np.random.lognormal(mean=10, sigma=1, size=bars)
        
        # Заполняем NaN
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        logger.info(f"Сгенерировано {len(df)} тестовых баров для {symbol}")
        return df
        
    def clear_cache(self, older_than_days: int = 1):
        """
        Очистка кэша
        
        Args:
            older_than_days (int): Удалять файлы старше указанного количества дней
        """
        if not os.path.exists(self.cache_dir):
            return
            
        try:
            cutoff_time = datetime.now() - timedelta(days=older_than_days)
            deleted_count = 0
            
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    filepath = os.path.join(self.cache_dir, filename)
                    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    
                    if file_time < cutoff_time:
                        os.remove(filepath)
                        deleted_count += 1
                        
            logger.info(f"Очищен кэш: удалено {deleted_count} файлов")
            
        except Exception as e:
            logger.error(f"Ошибка при очистке кэша: {e}")
            
    def get_available_symbols(self) -> list:
        """
        Получение списка доступных символов
        
        Returns:
            list: Список символов
        """
        if mt5_connector is None or not mt5_connector.is_connected():
            return ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'BTCUSD']
            
        try:
            symbols = mt5.symbols_get()
            return [s.name for s in symbols[:50]]  # Ограничиваем для скорости
        except Exception as e:
            logger.error(f"Ошибка получения списка символов: {e}")
            return ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'BTCUSD']


# Создаем глобальный экземпляр для удобства
data_feeder = DataFeeder()

