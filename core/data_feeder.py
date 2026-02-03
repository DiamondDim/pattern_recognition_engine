# core/data_feeder.py

"""
Модуль загрузки и обработки данных
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import yfinance as yf
import requests
import io

from config import DATA_CONFIG
from utils.logger import logger
from utils.helpers import validate_ohlc_data, smooth_data


class DataFeeder:
    """Класс для загрузки и обработки данных"""

    def __init__(self, config: DATA_CONFIG = None):
        self.config = config or DATA_CONFIG
        self.logger = logger.bind(module="DataFeeder")
        self.cache = {}
        self.cache_size = self.config.CACHE_SIZE

    def load_ohlc_data(self,
                      source: str,
                      symbol: str,
                      timeframe: str = '1d',
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      bars: int = 1000) -> Dict[str, np.ndarray]:
        """
        Загрузка данных OHLC

        Args:
            source: Источник данных ('yfinance', 'csv', 'mt5', 'api')
            symbol: Торговый символ
            timeframe: Таймфрейм
            start_date: Начальная дата (YYYY-MM-DD)
            end_date: Конечная дата (YYYY-MM-DD)
            bars: Количество баров

        Returns:
            Словарь с данными OHLC
        """
        try:
            cache_key = f"{source}_{symbol}_{timeframe}_{start_date}_{end_date}_{bars}"
            if cache_key in self.cache:
                self.logger.debug(f"Используются кэшированные данные: {cache_key}")
                return self.cache[cache_key]

            if source == 'yfinance':
                data = self._load_from_yfinance(symbol, timeframe, start_date, end_date, bars)
            elif source == 'csv':
                data = self._load_from_csv(symbol, timeframe, start_date, end_date, bars)
            elif source == 'mt5':
                data = self._load_from_mt5(symbol, timeframe, start_date, end_date, bars)
            elif source == 'api':
                data = self._load_from_api(symbol, timeframe, start_date, end_date, bars)
            else:
                self.logger.error(f"Неизвестный источник данных: {source}")
                return {}

            if data:
                # Кэширование данных
                self._add_to_cache(cache_key, data)

            return data

        except Exception as e:
            self.logger.error(f"Ошибка загрузки данных: {e}")
            return {}

    def _load_from_yfinance(self,
                           symbol: str,
                           timeframe: str,
                           start_date: Optional[str],
                           end_date: Optional[str],
                           bars: int) -> Dict[str, np.ndarray]:
        """Загрузка данных из Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)

            # Определение периода
            if start_date and end_date:
                start = pd.Timestamp(start_date)
                end = pd.Timestamp(end_date)
                period = None
            else:
                # Автоматическое определение периода по количеству баров
                period = self._get_yfinance_period(bars, timeframe)
                start = end = None

            # Загрузка данных
            df = ticker.history(period=period, start=start, end=end, interval=timeframe)

            if df.empty:
                self.logger.error(f"Не удалось загрузить данные для {symbol}")
                return {}

            # Конвертация в нужный формат
            data = {
                'open': df['Open'].values.astype(np.float32),
                'high': df['High'].values.astype(np.float32),
                'low': df['Low'].values.astype(np.float32),
                'close': df['Close'].values.astype(np.float32),
                'volume': df['Volume'].values.astype(np.float32),
                'timestamp': df.index.values
            }

            self.logger.info(f"Загружено данных из Yahoo Finance: {symbol} - {len(data['close'])} баров")
            return data

        except Exception as e:
            self.logger.error(f"Ошибка загрузки из Yahoo Finance: {e}")
            return {}

    def _get_yfinance_period(self, bars: int, timeframe: str) -> str:
        """Определение периода для Yahoo Finance"""
        # Примерное соответствие количества баров периоду
        if timeframe in ['1m', '2m', '5m', '15m', '30m']:
            # Для минутных таймфреймов
            if bars <= 60:  # ~1 час
                return '1d'
            elif bars <= 390:  # ~1 день (6.5 часов)
                return '5d'
            else:
                return '1mo'
        elif timeframe in ['1h', '2h', '4h']:
            if bars <= 24:  # ~1 день
                return '5d'
            elif bars <= 168:  # ~1 неделя
                return '1mo'
            else:
                return '3mo'
        else:  # Дневные и выше
            if bars <= 30:  # ~1 месяц
                return '3mo'
            elif bars <= 90:  # ~3 месяца
                return '6mo'
            elif bars <= 180:  # ~6 месяцев
                return '1y'
            else:
                return '2y'

    def _load_from_csv(self,
                      symbol: str,
                      timeframe: str,
                      start_date: Optional[str],
                      end_date: Optional[str],
                      bars: int) -> Dict[str, np.ndarray]:
        """Загрузка данных из CSV файла"""
        try:
            # Поиск файла
            data_dir = Path(self.config.DATA_DIR)
            csv_files = list(data_dir.glob(f"**/*{symbol}*.csv"))

            if not csv_files:
                self.logger.error(f"CSV файл для {symbol} не найден")
                return {}

            # Используем первый найденный файл
            csv_path = csv_files[0]

            # Чтение CSV
            df = pd.read_csv(csv_path)

            # Проверка необходимых колонок
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                # Попробуем найти колонки с заглавными буквами
                df.columns = df.columns.str.lower()
                missing_cols = [col for col in required_cols if col not in df.columns]

                if missing_cols:
                    self.logger.error(f"Отсутствуют колонки в CSV: {missing_cols}")
                    return {}

            # Фильтрация по дате
            if 'timestamp' in df.columns or 'date' in df.columns:
                time_col = 'timestamp' if 'timestamp' in df.columns else 'date'
                df[time_col] = pd.to_datetime(df[time_col])

                if start_date:
                    start = pd.Timestamp(start_date)
                    df = df[df[time_col] >= start]

                if end_date:
                    end = pd.Timestamp(end_date)
                    df = df[df[time_col] <= end]

            # Ограничение количества баров
            if bars > 0:
                df = df.tail(bars)

            # Конвертация в numpy arrays
            data = {
                'open': df['open'].values.astype(np.float32),
                'high': df['high'].values.astype(np.float32),
                'low': df['low'].values.astype(np.float32),
                'close': df['close'].values.astype(np.float32),
                'volume': df['volume'].values.astype(np.float32),
                'timestamp': df[time_col].values if time_col in df.columns else np.arange(len(df))
            }

            self.logger.info(f"Загружено данных из CSV: {csv_path} - {len(data['close'])} баров")
            return data

        except Exception as e:
            self.logger.error(f"Ошибка загрузки из CSV: {e}")
            return {}

    def _load_from_mt5(self,
                      symbol: str,
                      timeframe: str,
                      start_date: Optional[str],
                      end_date: Optional[str],
                      bars: int) -> Dict[str, np.ndarray]:
        """Загрузка данных из MetaTrader 5"""
        try:
            # Проверяем доступность MT5
            try:
                from utils.mt5_connector import MT5Connector
            except ImportError:
                self.logger.error("MT5Connector не доступен")
                return {}

            # Подключение к MT5
            mt5 = MT5Connector()
            import asyncio

            # Асинхронная загрузка
            async def load():
                await mt5.connect()
                df = await mt5.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    bars=bars,
                    start_date=pd.Timestamp(start_date) if start_date else None,
                    end_date=pd.Timestamp(end_date) if end_date else None
                )
                await mt5.disconnect()
                return df

            df = asyncio.run(load())

            if df is None or df.empty:
                return {}

            # Конвертация в формат
            data = {
                'open': df['open'].values.astype(np.float32),
                'high': df['high'].values.astype(np.float32),
                'low': df['low'].values.astype(np.float32),
                'close': df['close'].values.astype(np.float32),
                'volume': df['tick_volume'].values.astype(np.float32) if 'tick_volume' in df.columns
                         else df['volume'].values.astype(np.float32),
                'timestamp': df['time'].values
            }

            self.logger.info(f"Загружено данных из MT5: {symbol} - {len(data['close'])} баров")
            return data

        except Exception as e:
            self.logger.error(f"Ошибка загрузки из MT5: {e}")
            return {}

    def _load_from_api(self,
                      symbol: str,
                      timeframe: str,
                      start_date: Optional[str],
                      end_date: Optional[str],
                      bars: int) -> Dict[str, np.ndarray]:
        """Загрузка данных через API"""
        try:
            # Здесь можно реализовать загрузку из различных API
            # Например: Alpha Vantage, Polygon, Twelve Data и т.д.

            # Временная заглушка
            self.logger.warning("Загрузка через API пока не реализована")
            return {}

        except Exception as e:
            self.logger.error(f"Ошибка загрузки через API: {e}")
            return {}

    def _add_to_cache(self, key: str, data: Dict[str, np.ndarray]):
        """Добавление данных в кэш"""
        self.cache[key] = data

        # Ограничение размера кэша
        if len(self.cache) > self.cache_size:
            # Удаляем самый старый ключ
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

    def calculate_indicators(self, ohlc_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Расчет технических индикаторов

        Args:
            ohlc_data: Данные OHLC

        Returns:
            Словарь с индикаторами
        """
        try:
            if not validate_ohlc_data(ohlc_data):
                self.logger.error("Невалидные данные для расчета индикаторов")
                return {}

            closes = ohlc_data['close']
            highs = ohlc_data['high']
            lows = ohlc_data['low']
            volumes = ohlc_data.get('volume', np.zeros_like(closes))

            indicators = {}

            # 1. Moving Averages
            indicators['sma_20'] = self._calculate_sma(closes, 20)
            indicators['sma_50'] = self._calculate_sma(closes, 50)
            indicators['sma_200'] = self._calculate_sma(closes, 200)

            indicators['ema_12'] = self._calculate_ema(closes, 12)
            indicators['ema_26'] = self._calculate_ema(closes, 26)

            # 2. RSI
            indicators['rsi'] = self._calculate_rsi(closes, 14)

            # 3. MACD
            macd, signal, hist = self._calculate_macd(closes)
            indicators['macd'] = macd
            indicators['macd_signal'] = signal
            indicators['macd_hist'] = hist

            # 4. Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(closes, 20)
            indicators['bb_upper'] = bb_upper
            indicators['bb_middle'] = bb_middle
            indicators['bb_lower'] = bb_lower

            # 5. Stochastic Oscillator
            stoch_k, stoch_d = self._calculate_stochastic(highs, lows, closes, 14)
            indicators['stoch_k'] = stoch_k
            indicators['stoch_d'] = stoch_d

            # 6. ATR (Average True Range)
            indicators['atr'] = self._calculate_atr(highs, lows, closes, 14)

            # 7. Volume indicators
            if len(volumes) > 0:
                indicators['volume_sma'] = self._calculate_sma(volumes, 20)
                indicators['obv'] = self._calculate_obv(closes, volumes)

            self.logger.debug(f"Рассчитано индикаторов: {len(indicators)}")
            return indicators

        except Exception as e:
            self.logger.error(f"Ошибка расчета индикаторов: {e}")
            return {}

    def _calculate_sma(self, data: np.ndarray, window: int) -> np.ndarray:
        """Расчет простой скользящей средней"""
        if len(data) < window:
            return np.zeros_like(data)

        sma = np.zeros_like(data)
        for i in range(len(data)):
            if i < window - 1:
                sma[i] = np.nan
            else:
                sma[i] = np.mean(data[i - window + 1:i + 1])

        return sma

    def _calculate_ema(self, data: np.ndarray, window: int) -> np.ndarray:
        """Расчет экспоненциальной скользящей средней"""
        if len(data) < window:
            return np.zeros_like(data)

        alpha = 2 / (window + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]

        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

        return ema

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Расчет RSI"""
        if len(prices) <= period:
            return np.zeros_like(prices)

        deltas = np.diff(prices)
        seed = deltas[:period + 1]

        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period

        if down == 0:
            rs = float('inf')
        else:
            rs = up / down

        rsi = np.zeros_like(prices)
        rsi[:period] = np.nan
        rsi[period] = 100 - 100 / (1 + rs)

        for i in range(period + 1, len(prices)):
            delta = deltas[i - 1]

            if delta > 0:
                up_val = delta
                down_val = 0
            else:
                up_val = 0
                down_val = -delta

            up = (up * (period - 1) + up_val) / period
            down = (down * (period - 1) + down_val) / period

            if down == 0:
                rs = float('inf')
            else:
                rs = up / down

            rsi[i] = 100 - 100 / (1 + rs)

        return rsi

    def _calculate_macd(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Расчет MACD"""
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)

        macd_line = ema_12 - ema_26
        signal_line = self._calculate_ema(macd_line, 9)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def _calculate_bollinger_bands(self,
                                  prices: np.ndarray,
                                  window: int = 20,
                                  num_std: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Расчет Bollinger Bands"""
        if len(prices) < window:
            return np.zeros_like(prices), np.zeros_like(prices), np.zeros_like(prices)

        sma = self._calculate_sma(prices, window)
        rolling_std = np.zeros_like(prices)

        for i in range(len(prices)):
            if i < window - 1:
                rolling_std[i] = np.nan
            else:
                rolling_std[i] = np.std(prices[i - window + 1:i + 1])

        upper_band = sma + (rolling_std * num_std)
        lower_band = sma - (rolling_std * num_std)

        return upper_band, sma, lower_band

    def _calculate_stochastic(self,
                             highs: np.ndarray,
                             lows: np.ndarray,
                             closes: np.ndarray,
                             period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
        """Расчет Stochastic Oscillator"""
        if len(closes) < period:
            return np.zeros_like(closes), np.zeros_like(closes)

        stoch_k = np.zeros_like(closes)
        stoch_d = np.zeros_like(closes)

        for i in range(period - 1, len(closes)):
            high_max = np.max(highs[i - period + 1:i + 1])
            low_min = np.min(lows[i - period + 1:i + 1])

            if high_max - low_min == 0:
                stoch_k[i] = 50  # Нейтральное значение
            else:
                stoch_k[i] = 100 * (closes[i] - low_min) / (high_max - low_min)

        # %D (сглаженный %K)
        for i in range(period - 1, len(closes)):
            if i >= 2:
                stoch_d[i] = np.mean(stoch_k[i - 2:i + 1])
            else:
                stoch_d[i] = stoch_k[i]

        return stoch_k, stoch_d

    def _calculate_atr(self,
                      highs: np.ndarray,
                      lows: np.ndarray,
                      closes: np.ndarray,
                      period: int = 14) -> np.ndarray:
        """Расчет Average True Range"""
        if len(closes) < period:
            return np.zeros_like(closes)

        tr = np.zeros_like(closes)
        atr = np.zeros_like(closes)

        # True Range
        for i in range(1, len(closes)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            tr[i] = max(hl, hc, lc)

        # ATR
        atr[period - 1] = np.mean(tr[1:period])

        for i in range(period, len(closes)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        atr[:period - 1] = np.nan
        return atr

    def _calculate_obv(self, closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Расчет On-Balance Volume"""
        if len(closes) < 2:
            return np.zeros_like(closes)

        obv = np.zeros_like(closes)
        obv[0] = volumes[0]

        for i in range(1, len(closes)):
            if closes[i] > closes[i - 1]:
                obv[i] = obv[i - 1] + volumes[i]
            elif closes[i] < closes[i - 1]:
                obv[i] = obv[i - 1] - volumes[i]
            else:
                obv[i] = obv[i - 1]

        return obv

    def preprocess_data(self,
                       ohlc_data: Dict[str, np.ndarray],
                       indicators: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Предобработка данных для ML

        Args:
            ohlc_data: Данные OHLC
            indicators: Технические индикаторы

        Returns:
            Предобработанные данные
        """
        try:
            if not validate_ohlc_data(ohlc_data):
                self.logger.error("Невалидные данные для предобработки")
                return {}

            # Копирование данных
            processed = {
                'ohlc': {k: v.copy() for k, v in ohlc_data.items()},
                'indicators': {},
                'features': {}
            }

            # Нормализация цен
            closes = ohlc_data['close']
            processed['ohlc']['close_norm'] = self._normalize_prices(closes)

            # Обработка индикаторов
            if indicators:
                for name, values in indicators.items():
                    if len(values) == len(closes):
                        # Замена NaN
                        values_clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                        processed['indicators'][name] = values_clean

            # Извлечение признаков
            processed['features'] = self._extract_features(processed['ohlc'], processed['indicators'])

            self.logger.debug(f"Предобработано данных: {len(closes)} баров")
            return processed

        except Exception as e:
            self.logger.error(f"Ошибка предобработки данных: {e}")
            return {}

    def _normalize_prices(self, prices: np.ndarray) -> np.ndarray:
        """Нормализация цен"""
        if len(prices) == 0:
            return prices

        min_val = np.min(prices)
        max_val = np.max(prices)

        if max_val - min_val == 0:
            return np.zeros_like(prices)

        return (prices - min_val) / (max_val - min_val)

    def _extract_features(self,
                         ohlc: Dict[str, np.ndarray],
                         indicators: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Извлечение признаков"""
        features = {}
        closes = ohlc['close']
        n_samples = len(closes)

        # 1. Returns features
        features['returns_1'] = self._calculate_returns(closes, 1)
        features['returns_5'] = self._calculate_returns(closes, 5)
        features['returns_20'] = self._calculate_returns(closes, 20)

        # 2. Volatility features
        features['volatility_20'] = self._calculate_rolling_volatility(closes, 20)
        features['volatility_50'] = self._calculate_rolling_volatility(closes, 50)

        # 3. Price position features
        features['price_position'] = self._calculate_price_position(closes, 20)

        # 4. Indicator-based features
        if 'rsi' in indicators:
            features['rsi_position'] = indicators['rsi'] / 100.0

        if 'macd' in indicators and 'macd_signal' in indicators:
            features['macd_cross'] = np.where(indicators['macd'] > indicators['macd_signal'], 1, -1)

        # 5. Volume features
        if 'volume' in ohlc:
            volumes = ohlc['volume']
            if len(volumes) > 0:
                features['volume_ratio'] = self._calculate_volume_ratio(volumes, 20)

        return features

    def _calculate_returns(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Расчет доходностей"""
        if len(prices) <= period:
            return np.zeros_like(prices)

        returns = np.zeros_like(prices)
        returns[period:] = (prices[period:] / prices[:-period]) - 1
        returns[:period] = 0

        return returns

    def _calculate_rolling_volatility(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Расчет скользящей волатильности"""
        if len(prices) <= window:
            return np.zeros_like(prices)

        returns = self._calculate_returns(prices, 1)
        volatility = np.zeros_like(prices)

        for i in range(window, len(prices)):
            volatility[i] = np.std(returns[i - window + 1:i + 1])

        volatility[:window] = 0
        return volatility

    def _calculate_price_position(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Расчет позиции цены в диапазоне"""
        if len(prices) <= window:
            return np.zeros_like(prices)

        position = np.zeros_like(prices)

        for i in range(window, len(prices)):
            window_prices = prices[i - window + 1:i + 1]
            min_price = np.min(window_prices)
            max_price = np.max(window_prices)

            if max_price - min_price > 0:
                position[i] = (prices[i] - min_price) / (max_price - min_price)
            else:
                position[i] = 0.5

        position[:window] = 0.5
        return position

    def _calculate_volume_ratio(self, volumes: np.ndarray, window: int) -> np.ndarray:
        """Расчет отношения объема к среднему"""
        if len(volumes) <= window:
            return np.ones_like(volumes)

        ratio = np.zeros_like(volumes)

        for i in range(window, len(volumes)):
            avg_volume = np.mean(volumes[i - window + 1:i + 1])
            if avg_volume > 0:
                ratio[i] = volumes[i] / avg_volume
            else:
                ratio[i] = 1.0

        ratio[:window] = 1.0
        return ratio

    def split_train_test(self,
                        data: Dict[str, np.ndarray],
                        test_size: float = 0.2,
                        time_series: bool = True) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Разделение данных на train и test

        Args:
            data: Данные для разделения
            test_size: Доля тестовых данных
            time_series: Флаг временного ряда (не перемешивать)

        Returns:
            train_data, test_data
        """
        try:
            if not data or len(data) == 0:
                return {}, {}

            # Определяем размер тестовой выборки
            sample_size = len(next(iter(data.values())))
            test_samples = int(sample_size * test_size)

            if time_series:
                # Для временных рядов берем последние данные для теста
                train_idx = slice(0, -test_samples)
                test_idx = slice(-test_samples, None)
            else:
                # Случайное разделение
                indices = np.arange(sample_size)
                np.random.shuffle(indices)
                train_indices = indices[:-test_samples]
                test_indices = indices[-test_samples:]

                train_idx = train_indices
                test_idx = test_indices

            # Разделение данных
            train_data = {}
            test_data = {}

            for key, values in data.items():
                if time_series:
                    train_data[key] = values[train_idx]
                    test_data[key] = values[test_idx]
                else:
                    train_data[key] = values[train_idx]
                    test_data[key] = values[test_idx]

            self.logger.debug(f"Данные разделены: train={len(train_data.get('close', []))}, "
                             f"test={len(test_data.get('close', []))}")

            return train_data, test_data

        except Exception as e:
            self.logger.error(f"Ошибка разделения данных: {e}")
            return {}, {}

    def clear_cache(self):
        """Очистка кэша данных"""
        self.cache.clear()
        self.logger.info("Кэш данных очищен")

