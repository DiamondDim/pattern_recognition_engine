"""
Коннектор для работы с MetaTrader 5
"""

import socket
import asyncio
import json
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime
import aiofiles
import websockets

from config import MT5_CONFIG, DATA_DIR, INPUT_DIR, OUTPUT_DIR
from utils.logger import logger


class MT5Connector:
    """Класс для подключения к MetaTrader 5"""

    def __init__(self, mode: str = None):
        self.mode = mode or MT5_CONFIG.CONNECTION_MODE
        self.logger = logger.bind(name="MT5Connector")

        # Настройки подключения
        self.host = MT5_CONFIG.SOCKET_HOST
        self.port = MT5_CONFIG.SOCKET_PORT

        # Состояние подключения
        self.is_connected = False
        self.socket = None
        self.ws = None

        # Буферы данных
        self.data_buffer = []
        self.command_buffer = []

        # Статистика
        self.stats = {
            'connection_attempts': 0,
            'successful_connections': 0,
            'data_received': 0,
            'data_sent': 0,
            'errors': 0,
            'last_connection': None
        }

    async def connect(self) -> bool:
        """
        Подключение к MT5

        Returns:
            True если подключение успешно
        """
        if self.mode == 'socket':
            return await self._connect_socket()
        elif self.mode == 'websocket':
            return await self._connect_websocket()
        else:
            self.logger.info("Режим файлового обмена не требует подключения")
            self.is_connected = True
            return True

    async def _connect_socket(self) -> bool:
        """Подключение через TCP сокет"""
        self.stats['connection_attempts'] += 1

        try:
            self.logger.info(f"Подключение к MT5 через сокет {self.host}:{self.port}")

            # Создаем сокет
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)

            # Подключаемся
            self.socket.connect((self.host, self.port))

            self.is_connected = True
            self.stats['successful_connections'] += 1
            self.stats['last_connection'] = datetime.now()

            self.logger.info("Подключение к MT5 установлено")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка подключения к MT5: {e}")
            self.stats['errors'] += 1
            self.is_connected = False
            return False

    async def _connect_websocket(self) -> bool:
        """Подключение через WebSocket"""
        self.stats['connection_attempts'] += 1

        try:
            self.logger.info(f"Подключение к MT5 через WebSocket ws://{self.host}:{self.port}")

            # Подключаемся через WebSocket
            self.ws = await websockets.connect(f"ws://{self.host}:{self.port}")

            self.is_connected = True
            self.stats['successful_connections'] += 1
            self.stats['last_connection'] = datetime.now()

            self.logger.info("WebSocket подключение к MT5 установлено")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка WebSocket подключения к MT5: {e}")
            self.stats['errors'] += 1
            self.is_connected = False
            return False

    async def send_data(self, data: Dict[str, Any]) -> bool:
        """
        Отправка данных в MT5

        Args:
            data: Данные для отправки

        Returns:
            True если отправка успешна
        """
        if not self.is_connected:
            self.logger.warning("Нет подключения к MT5")
            return False

        try:
            # Конвертируем данные в JSON
            json_data = json.dumps(data, default=str)

            if self.mode == 'socket':
                # Отправляем через сокет
                self.socket.sendall(json_data.encode('utf-8'))

            elif self.mode == 'websocket':
                # Отправляем через WebSocket
                await self.ws.send(json_data)

            self.stats['data_sent'] += 1
            self.logger.debug(f"Данные отправлены в MT5: {len(json_data)} байт")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка отправки данных в MT5: {e}")
            self.stats['errors'] += 1
            return False

    async def receive_data(self) -> Optional[Dict[str, Any]]:
        """
        Получение данных от MT5

        Returns:
            Полученные данные или None при ошибке
        """
        if not self.is_connected:
            self.logger.warning("Нет подключения к MT5")
            return None

        try:
            if self.mode == 'socket':
                # Получаем через сокет
                buffer_size = 4096
                data = b""

                while True:
                    chunk = self.socket.recv(buffer_size)
                    if not chunk:
                        break
                    data += chunk
                    if len(chunk) < buffer_size:
                        break

            elif self.mode == 'websocket':
                # Получаем через WebSocket
                data = await self.ws.recv()
                if isinstance(data, bytes):
                    data = data.decode('utf-8')

            # Парсим JSON
            json_data = json.loads(data)

            self.stats['data_received'] += 1
            self.logger.debug(f"Данные получены от MT5: {len(data)} байт")

            return json_data

        except Exception as e:
            self.logger.error(f"Ошибка получения данных от MT5: {e}")
            self.stats['errors'] += 1
            return None

    async def send_command(self, command: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Отправка команды в MT5 и получение ответа

        Args:
            command: Команда
            params: Параметры команды

        Returns:
            Ответ от MT5 или None при ошибке
        """
        # Формируем команду
        cmd_data = {
            'command': command,
            'timestamp': datetime.now().isoformat(),
            'params': params or {}
        }

        # Отправляем команду
        if not await self.send_data(cmd_data):
            return None

        # Ждем ответ (если команда требует ответа)
        if command in ['get_data', 'get_indicators', 'execute_trade']:
            response = await self.receive_data()
            return response

        return {'status': 'sent'}

    async def disconnect(self):
        """Отключение от MT5"""
        try:
            if self.mode == 'socket' and self.socket:
                self.socket.close()
                self.logger.info("Сокет подключение к MT5 закрыто")

            elif self.mode == 'websocket' and self.ws:
                await self.ws.close()
                self.logger.info("WebSocket подключение к MT5 закрыто")

        except Exception as e:
            self.logger.error(f"Ошибка отключения от MT5: {e}")

        finally:
            self.is_connected = False

    async def check_connection(self) -> bool:
        """Проверка подключения к MT5"""
        if not self.is_connected:
            return False

        try:
            # Отправляем ping-команду
            response = await self.send_command('ping')
            return response is not None

        except Exception as e:
            self.logger.error(f"Ошибка проверки подключения: {e}")
            self.is_connected = False
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики подключения"""
        return self.stats.copy()


class FileConnector:
    """Коннектор для файлового обмена с MT5"""

    def __init__(self):
        self.logger = logger.bind(name="FileConnector")

        # Пути к файлам
        self.input_file = INPUT_DIR / "mt5_data.csv"
        self.output_file = OUTPUT_DIR / "patterns.json"

        # Статистика
        self.stats = {
            'files_read': 0,
            'files_written': 0,
            'last_read': None,
            'last_write': None,
            'errors': 0
        }

    def read_data(self) -> Optional[pd.DataFrame]:
        """
        Чтение данных из файла

        Returns:
            DataFrame с данными или None при ошибке
        """
        try:
            if not self.input_file.exists():
                self.logger.warning(f"Входной файл не найден: {self.input_file}")
                return None

            # Читаем CSV
            df = pd.read_csv(self.input_file)

            # Проверяем необходимые колонки
            required_cols = ['timestamp', 'open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                self.logger.error(f"Отсутствуют колонки: {missing_cols}")
                return None

            # Конвертируем timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            self.stats['files_read'] += 1
            self.stats['last_read'] = datetime.now()

            self.logger.debug(f"Данные прочитаны из файла: {len(df)} строк")
            return df

        except Exception as e:
            self.logger.error(f"Ошибка чтения данных из файла: {e}")
            self.stats['errors'] += 1
            return None

    def write_patterns(self, patterns: List[Dict[str, Any]]) -> bool:
        """
        Запись паттернов в файл

        Args:
            patterns: Список паттернов

        Returns:
            True если запись успешна
        """
        try:
            # Подготавливаем данные для записи
            output_data = {
                'timestamp': datetime.now().isoformat(),
                'patterns_count': len(patterns),
                'patterns': patterns
            }

            # Записываем в JSON
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, default=str)

            self.stats['files_written'] += 1
            self.stats['last_write'] = datetime.now()

            self.logger.debug(f"Паттерны записаны в файл: {len(patterns)} паттернов")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка записи паттернов в файл: {e}")
            self.stats['errors'] += 1
            return False

    def write_signals(self, signals: List[Dict[str, Any]]) -> bool:
        """
        Запись торговых сигналов в файл

        Args:
            signals: Список сигналов

        Returns:
            True если запись успешна
        """
        try:
            signals_file = OUTPUT_DIR / "signals.json"

            output_data = {
                'timestamp': datetime.now().isoformat(),
                'signals_count': len(signals),
                'signals': signals
            }

            with open(signals_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, default=str)

            self.logger.debug(f"Сигналы записаны в файл: {len(signals)} сигналов")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка записи сигналов в файл: {e}")
            return False

    def cleanup_old_files(self, days_to_keep: int = 7) -> int:
        """
        Очистка старых файлов

        Args:
            days_to_keep: Количество дней для хранения

        Returns:
            Количество удаленных файлов
        """
        try:
            import os
            import time

            cutoff_time = time.time() - (days_to_keep * 24 * 3600)
            deleted_count = 0

            # Очищаем входные файлы
            for file_path in INPUT_DIR.glob("*.csv"):
                if os.path.getmtime(file_path) < cutoff_time:
                    os.remove(file_path)
                    deleted_count += 1

            # Очищаем выходные файлы
            for file_path in OUTPUT_DIR.glob("*.json"):
                if os.path.getmtime(file_path) < cutoff_time:
                    os.remove(file_path)
                    deleted_count += 1

            if deleted_count > 0:
                self.logger.info(f"Удалено {deleted_count} старых файлов")

            return deleted_count

        except Exception as e:
            self.logger.error(f"Ошибка очистки старых файлов: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики файлового обмена"""
        return self.stats.copy()


class MT5DataExporter:
    """Экспортер данных из MT5"""

    def __init__(self, connector_type: str = 'file'):
        self.connector_type = connector_type
        self.logger = logger.bind(name="MT5DataExporter")

        # Инициализация коннектора
        if connector_type == 'socket':
            self.connector = MT5Connector(mode='socket')
        elif connector_type == 'websocket':
            self.connector = MT5Connector(mode='websocket')
        else:
            self.connector = FileConnector()

        # Статистика экспорта
        self.export_stats = {
            'symbols_exported': 0,
            'timeframes_exported': 0,
            'total_candles': 0,
            'last_export': None,
            'export_errors': 0
        }

    async def export_data(self,
                          symbols: List[str] = None,
                          timeframes: List[str] = None,
                          bars_count: int = 1000) -> bool:
        """
        Экспорт данных из MT5

        Args:
            symbols: Список символов
            timeframes: Список таймфреймов
            bars_count: Количество свечей

        Returns:
            True если экспорт успешен
        """
        symbols = symbols or MT5_CONFIG.SYMBOLS
        timeframes = timeframes or MT5_CONFIG.EXPORT_TIMEFRAMES

        try:
            self.logger.info(f"Экспорт данных для {len(symbols)} символов")

            all_data = {}

            for symbol in symbols:
                for timeframe in timeframes:
                    try:
                        # Экспортируем данные для символа и таймфрейма
                        data = await self._export_symbol_data(
                            symbol, timeframe, bars_count
                        )

                        if data is not None:
                            key = f"{symbol}_{timeframe}"
                            all_data[key] = data

                            self.export_stats['symbols_exported'] = len(set(
                                key.split('_')[0] for key in all_data.keys()
                            ))
                            self.export_stats['timeframes_exported'] = len(set(
                                key.split('_')[1] for key in all_data.keys()
                            ))
                            self.export_stats['total_candles'] += len(data.get('close', []))

                    except Exception as e:
                        self.logger.error(f"Ошибка экспорта {symbol} {timeframe}: {e}")
                        self.export_stats['export_errors'] += 1

            # Сохраняем данные
            if all_data:
                await self._save_exported_data(all_data)
                self.export_stats['last_export'] = datetime.now()
                return True
            else:
                self.logger.warning("Нет данных для экспорта")
                return False

        except Exception as e:
            self.logger.error(f"Ошибка экспорта данных: {e}")
            self.export_stats['export_errors'] += 1
            return False

    async def _export_symbol_data(self,
                                  symbol: str,
                                  timeframe: str,
                                  bars_count: int) -> Optional[Dict[str, Any]]:
        """
        Экспорт данных для конкретного символа

        Args:
            symbol: Символ
            timeframe: Таймфрейм
            bars_count: Количество свечей

        Returns:
            Данные или None при ошибке
        """
        if self.connector_type in ['socket', 'websocket']:
            # Запрашиваем данные через сокет/WebSocket
            command_params = {
                'symbol': symbol,
                'timeframe': timeframe,
                'bars_count': bars_count
            }

            response = await self.connector.send_command('get_data', command_params)

            if response and 'data' in response:
                return response['data']
            else:
                return None

        else:
            # В файловом режиме данные уже должны быть в файле
            # MT5 скрипт должен записывать данные самостоятельно
            self.logger.debug(f"Ожидание данных от MT5 скрипта: {symbol} {timeframe}")
            return None

    async def _save_exported_data(self, data: Dict[str, Any]):
        """Сохранение экспортированных данных"""
        if self.connector_type in ['socket', 'websocket']:
            # Отправляем данные через коннектор
            await self.connector.send_data({
                'type': 'exported_data',
                'data': data,
                'timestamp': datetime.now().isoformat()
            })

        else:
            # Сохраняем в файл
            if isinstance(self.connector, FileConnector):
                # Для каждого символа/таймфрейма создаем отдельный файл
                for key, symbol_data in data.items():
                    symbol, timeframe = key.split('_')

                    # Конвертируем в DataFrame
                    df = pd.DataFrame(symbol_data)

                    # Сохраняем в CSV
                    filename = f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                    filepath = INPUT_DIR / filename

                    df.to_csv(filepath, index=False)
                    self.logger.debug(f"Данные сохранены в {filepath}")

    async def import_patterns_to_mt5(self, patterns: List[Dict[str, Any]]) -> bool:
        """
        Импорт паттернов обратно в MT5 для отрисовки

        Args:
            patterns: Список паттернов

        Returns:
            True если импорт успешен
        """
        try:
            if self.connector_type in ['socket', 'websocket']:
                # Отправляем паттерны через коннектор
                response = await self.connector.send_command('draw_patterns', {'patterns': patterns})
                return response is not None

            else:
                # Сохраняем паттерны в файл для MT5
                if isinstance(self.connector, FileConnector):
                    return self.connector.write_patterns(patterns)

            return False

        except Exception as e:
            self.logger.error(f"Ошибка импорта паттернов в MT5: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики экспорта"""
        return self.export_stats.copy()

    async def close(self):
        """Закрытие соединений"""
        if hasattr(self.connector, 'disconnect'):
            await self.connector.disconnect()

