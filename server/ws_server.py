"""
WebSocket сервер для Pattern Recognition Engine
"""

import asyncio
import json
import websockets
from typing import Dict, Any, Set, Optional
from datetime import datetime
import signal
import sys
from pathlib import Path

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent.parent))

from config import CONFIG, MT5_CONFIG
from core.pattern_detector import PatternDetector
from core.pattern_analyzer import PatternAnalyzer
from core.pattern_database import PatternDatabase
from core.ml_models import PatternSuccessPredictor
from utils.logger import logger


class WebSocketServer:
    """WebSocket сервер для реального времени"""

    def __init__(self, host: str = None, port: int = None):
        self.host = host or MT5_CONFIG.SOCKET_HOST
        self.port = port or MT5_CONFIG.SOCKET_PORT + 1  # Другой порт для сервера

        # Инициализация компонентов
        self.detector = PatternDetector()
        self.analyzer = PatternAnalyzer()
        self.database = PatternDatabase()
        self.predictor = PatternSuccessPredictor()

        # Клиенты
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.mt5_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.gui_clients: Set[websockets.WebSocketServerProtocol] = set()

        # Состояние сервера
        self.is_running = False
        self.server = None

        # Статистика
        self.stats = {
            'start_time': None,
            'clients_connected': 0,
            'messages_received': 0,
            'messages_sent': 0,
            'patterns_detected': 0,
            'errors': 0
        }

        self.logger = logger.bind(name="WebSocketServer")

    async def start(self):
        """Запуск WebSocket сервера"""
        self.logger.info(f"Запуск WebSocket сервера на {self.host}:{self.port}")

        # Загрузка ML модели
        await self._load_ml_model()

        # Запуск сервера
        self.server = await websockets.serve(
            self.handle_connection,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=10,
            close_timeout=10
        )

        self.is_running = True
        self.stats['start_time'] = datetime.now()

        self.logger.info("WebSocket сервер запущен")

        # Запуск фоновых задач
        asyncio.create_task(self._background_tasks())

        # Ожидание остановки
        await self.server.wait_closed()

    async def _load_ml_model(self):
        """Загрузка ML модели"""
        try:
            # Ищем последнюю сохраненную модель
            models_dir = Path(__file__).parent.parent / "models"
            if models_dir.exists():
                model_files = list(models_dir.glob("pattern_model_*.joblib"))
                if model_files:
                    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                    self.predictor.ml_model.load_model(str(latest_model))
                    self.logger.info(f"ML модель загружена: {latest_model.name}")
        except Exception as e:
            self.logger.error(f"Ошибка загрузки ML модели: {e}")

    async def handle_connection(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Обработка подключения клиента"""
        client_type = self._determine_client_type(path)
        client_id = f"{client_type}_{len(self.clients)}"

        self.clients.add(websocket)
        self.stats['clients_connected'] += 1

        self.logger.info(f"Новое подключение: {client_id} ({len(self.clients)} клиентов)")

        try:
            # Отправка приветственного сообщения
            await self._send_welcome(websocket, client_id, client_type)

            # Обработка сообщений от клиента
            async for message in websocket:
                self.stats['messages_received'] += 1
                await self.handle_message(websocket, message, client_id, client_type)

        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Клиент отключен: {client_id}")

        except Exception as e:
            self.logger.error(f"Ошибка обработки подключения {client_id}: {e}")
            self.stats['errors'] += 1

        finally:
            # Удаление клиента
            self.clients.remove(websocket)
            if client_type == 'mt5':
                self.mt5_clients.remove(websocket)
            elif client_type == 'gui':
                self.gui_clients.remove(websocket)

            self.logger.info(f"Клиент удален: {client_id} ({len(self.clients)} клиентов осталось)")

    def _determine_client_type(self, path: str) -> str:
        """Определение типа клиента по пути"""
        if path == '/mt5':
            return 'mt5'
        elif path == '/gui':
            return 'gui'
        else:
            return 'unknown'

    async def _send_welcome(self, websocket, client_id: str, client_type: str):
        """Отправка приветственного сообщения"""
        welcome_msg = {
            'type': 'welcome',
            'client_id': client_id,
            'client_type': client_type,
            'server_time': datetime.now().isoformat(),
            'version': CONFIG.VERSION,
            'supported_commands': ['analyze', 'get_patterns', 'get_stats', 'train_model']
        }

        await self._send_json(websocket, welcome_msg)

    async def handle_message(self, websocket, message: str, client_id: str, client_type: str):
        """Обработка сообщения от клиента"""
        try:
            data = json.loads(message)
            message_type = data.get('type', 'unknown')

            self.logger.debug(f"Сообщение от {client_id}: {message_type}")

            # Обработка в зависимости от типа сообщения
            if message_type == 'analyze':
                await self._handle_analyze(websocket, data, client_id)

            elif message_type == 'get_patterns':
                await self._handle_get_patterns(websocket, data)

            elif message_type == 'get_stats':
                await self._handle_get_stats(websocket)

            elif message_type == 'train_model':
                await self._handle_train_model(websocket, data)

            elif message_type == 'subscribe':
                await self._handle_subscribe(websocket, data, client_type)

            elif message_type == 'ping':
                await self._handle_ping(websocket)

            else:
                self.logger.warning(f"Неизвестный тип сообщения: {message_type}")
                await self._send_error(websocket, f"Unknown message type: {message_type}")

        except json.JSONDecodeError as e:
            self.logger.error(f"Ошибка парсинга JSON: {e}")
            await self._send_error(websocket, f"Invalid JSON: {str(e)}")

        except Exception as e:
            self.logger.error(f"Ошибка обработки сообщения: {e}")
            self.stats['errors'] += 1
            await self._send_error(websocket, f"Processing error: {str(e)}")

    async def _handle_analyze(self, websocket, data: Dict[str, Any], client_id: str):
        """Обработка запроса на анализ"""
        try:
            # Извлекаем данные
            symbol = data.get('symbol', 'UNKNOWN')
            timeframe = data.get('timeframe', 'H1')
            ohlc_data = data.get('data', {})

            if not ohlc_data:
                await self._send_error(websocket, "No data provided")
                return

            self.logger.info(f"Анализ данных: {symbol} {timeframe} от {client_id}")

            # Конвертация данных
            opens = np.array(ohlc_data.get('open', []), dtype=float)
            highs = np.array(ohlc_data.get('high', []), dtype=float)
            lows = np.array(ohlc_data.get('low', []), dtype=float)
            closes = np.array(ohlc_data.get('close', []), dtype=float)
            volumes = np.array(ohlc_data.get('volume', []), dtype=float)

            if len(closes) < 20:
                await self._send_error(websocket, "Insufficient data (minimum 20 candles)")
                return

            # Детектирование паттернов
            detection_result = self.detector.detect_all_patterns(
                symbol=symbol,
                timeframe=timeframe,
                data={
                    'open': opens,
                    'high': highs,
                    'low': lows,
                    'close': closes,
                    'volume': volumes
                }
            )

            # Анализ найденных паттернов
            analyzed_patterns = []
            for pattern in detection_result.patterns:
                # Получаем исторические аналоги
                historical_patterns = self.database.get_patterns(
                    symbol=symbol,
                    timeframe=timeframe,
                    pattern_type=pattern.get('type'),
                    direction=pattern.get('direction'),
                    limit=50
                )

                historical_dicts = [p.to_dict() for p in historical_patterns]

                # Анализ качества
                quality_analysis = self.analyzer.analyze_pattern_quality(pattern)
                pattern['quality_analysis'] = quality_analysis

                # ML предсказание
                ml_prediction = self.predictor.predict_pattern_success(pattern)
                pattern['ml_prediction'] = ml_prediction

                # Сохранение в базу данных
                self.database.save_pattern(pattern)

                analyzed_patterns.append(pattern)

                # Логирование
                self.logger.info(
                    f"Обнаружен паттерн: {pattern['name']} "
                    f"({pattern['direction']}) "
                    f"Качество: {quality_analysis.get('overall_score', 0):.2f}"
                )

            self.stats['patterns_detected'] += len(analyzed_patterns)

            # Отправка результатов
            response = {
                'type': 'analysis_result',
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'timeframe': timeframe,
                'patterns_count': len(analyzed_patterns),
                'patterns': analyzed_patterns[:10],  # Ограничиваем количество
                'statistics': detection_result.statistics
            }

            await self._send_json(websocket, response)

            # Рассылка GUI клиентам
            if analyzed_patterns:
                await self._broadcast_to_gui({
                    'type': 'new_patterns',
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'patterns_count': len(analyzed_patterns),
                    'sample_patterns': analyzed_patterns[:3]
                })

        except Exception as e:
            self.logger.error(f"Ошибка анализа: {e}")
            await self._send_error(websocket, f"Analysis error: {str(e)}")

    async def _handle_get_patterns(self, websocket, data: Dict[str, Any]):
        """Обработка запроса на получение паттернов"""
        try:
            symbol = data.get('symbol')
            timeframe = data.get('timeframe')
            pattern_type = data.get('pattern_type')
            direction = data.get('direction')
            limit = data.get('limit', 100)

            # Получаем паттерны из базы данных
            patterns = self.database.get_patterns(
                symbol=symbol,
                timeframe=timeframe,
                pattern_type=pattern_type,
                direction=direction,
                limit=limit
            )

            # Конвертируем в словари
            patterns_dict = [p.to_dict() for p in patterns]

            # Отправка ответа
            response = {
                'type': 'patterns_list',
                'timestamp': datetime.now().isoformat(),
                'count': len(patterns_dict),
                'patterns': patterns_dict
            }

            await self._send_json(websocket, response)

        except Exception as e:
            self.logger.error(f"Ошибка получения паттернов: {e}")
            await self._send_error(websocket, f"Get patterns error: {str(e)}")

    async def _handle_get_stats(self, websocket):
        """Обработка запроса статистики"""
        try:
            # Статистика сервера
            server_stats = self.stats.copy()
            server_stats['uptime'] = str(datetime.now() - server_stats['start_time'])

            # Статистика детектора
            detector_stats = self.detector.get_statistics()

            # Статистика анализатора
            analyzer_stats = self.analyzer.get_statistics()

            # Статистика базы данных
            db_stats = self.database.get_database_stats()

            # Статистика предиктора
            predictor_stats = self.predictor.get_stats()

            response = {
                'type': 'stats',
                'timestamp': datetime.now().isoformat(),
                'server': server_stats,
                'detector': detector_stats,
                'analyzer': analyzer_stats,
                'database': db_stats,
                'predictor': predictor_stats,
                'clients_count': len(self.clients),
                'mt5_clients_count': len(self.mt5_clients),
                'gui_clients_count': len(self.gui_clients)
            }

            await self._send_json(websocket, response)

        except Exception as e:
            self.logger.error(f"Ошибка получения статистики: {e}")
            await self._send_error(websocket, f"Get stats error: {str(e)}")

    async def _handle_train_model(self, websocket, data: Dict[str, Any]):
        """Обработка запроса на обучение модели"""
        try:
            days_back = data.get('days_back', 30)

            self.logger.info(f"Обучение ML модели на данных за {days_back} дней")

            # Получаем исторические данные
            historical_patterns = self.database.get_historical_patterns_for_analysis(
                days_back=days_back,
                min_quality=0.6
            )

            if len(historical_patterns) < 100:
                await self._send_error(websocket, f"Insufficient historical data: {len(historical_patterns)} patterns")
                return

            # Обучение модели
            success = self.predictor.train_ml_model(historical_patterns)

            response = {
                'type': 'training_result',
                'timestamp': datetime.now().isoformat(),
                'success': success,
                'patterns_count': len(historical_patterns),
                'model_info': self.predictor.ml_model.get_model_info() if success else None
            }

            await self._send_json(websocket, response)

        except Exception as e:
            self.logger.error(f"Ошибка обучения модели: {e}")
            await self._send_error(websocket, f"Training error: {str(e)}")

    async def _handle_subscribe(self, websocket, data: Dict[str, Any], client_type: str):
        """Обработка подписки на обновления"""
        try:
            subscription_type = data.get('subscription_type', 'all')

            # Добавляем клиента в соответствующую группу
            if client_type == 'mt5':
                self.mt5_clients.add(websocket)
            elif client_type == 'gui':
                self.gui_clients.add(websocket)

            response = {
                'type': 'subscription_confirmed',
                'timestamp': datetime.now().isoformat(),
                'subscription_type': subscription_type,
                'client_type': client_type
            }

            await self._send_json(websocket, response)

            self.logger.info(f"Клиент подписался: {client_type} -> {subscription_type}")

        except Exception as e:
            self.logger.error(f"Ошибка подписки: {e}")
            await self._send_error(websocket, f"Subscription error: {str(e)}")

    async def _handle_ping(self, websocket):
        """Обработка ping запроса"""
        response = {
            'type': 'pong',
            'timestamp': datetime.now().isoformat(),
            'server_time': datetime.now().isoformat()
        }

        await self._send_json(websocket, response)

    async def _send_json(self, websocket, data: Dict[str, Any]):
        """Отправка JSON сообщения"""
        try:
            message = json.dumps(data, default=str)
            await websocket.send(message)
            self.stats['messages_sent'] += 1
        except Exception as e:
            self.logger.error(f"Ошибка отправки сообщения: {e}")

    async def _send_error(self, websocket, error_message: str):
        """Отправка сообщения об ошибке"""
        error_response = {
            'type': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': error_message
        }

        await self._send_json(websocket, error_response)

    async def _broadcast_to_gui(self, message: Dict[str, Any]):
        """Рассылка сообщений всем GUI клиентам"""
        if not self.gui_clients:
            return

        message_json = json.dumps(message, default=str)

        tasks = []
        for client in self.gui_clients:
            try:
                tasks.append(client.send(message_json))
            except Exception as e:
                self.logger.error(f"Ошибка рассылки GUI клиенту: {e}")

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _background_tasks(self):
        """Фоновые задачи сервера"""
        self.logger.info("Запуск фоновых задач")

        while self.is_running:
            try:
                # Отправка heartbeat
                await self._send_heartbeat()

                # Очистка старых данных
                await self._cleanup_old_data()

                # Сохранение статистики
                await self._save_statistics()

                # Пауза между итерациями
                await asyncio.sleep(60)  # Каждую минуту

            except Exception as e:
                self.logger.error(f"Ошибка в фоновой задаче: {e}")
                await asyncio.sleep(10)

    async def _send_heartbeat(self):
        """Отправка heartbeat сообщений"""
        heartbeat = {
            'type': 'heartbeat',
            'timestamp': datetime.now().isoformat(),
            'clients_count': len(self.clients),
            'patterns_detected': self.stats['patterns_detected']
        }

        heartbeat_json = json.dumps(heartbeat, default=str)

        tasks = []
        for client in self.clients:
            try:
                tasks.append(client.send(heartbeat_json))
            except Exception:
                pass  # Игнорируем отключенных клиентов

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _cleanup_old_data(self):
        """Очистка старых данных"""
        try:
            # Очистка старых паттернов
            deleted = self.database.cleanup_old_patterns(days_to_keep=90)
            if deleted > 0:
                self.logger.info(f"Очищено {deleted} старых паттернов")

            # Очистка кэша предиктора
            self.predictor.clear_cache()

        except Exception as e:
            self.logger.error(f"Ошибка очистки данных: {e}")

    async def _save_statistics(self):
        """Сохранение статистики сервера"""
        try:
            stats_file = Path(__file__).parent.parent / "data" / "server_stats.json"

            stats_data = {
                'server': self.stats,
                'detector': self.detector.get_statistics(),
                'analyzer': self.analyzer.get_statistics(),
                'predictor': self.predictor.get_stats(),
                'timestamp': datetime.now().isoformat()
            }

            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_data, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Ошибка сохранения статистики: {e}")

    async def stop(self):
        """Остановка сервера"""
        self.logger.info("Остановка WebSocket сервера...")

        self.is_running = False

        if self.server:
            self.server.close()
            await self.server.wait_closed()

        # Закрытие всех подключений
        tasks = []
        for client in self.clients:
            tasks.append(client.close())

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self.logger.info("WebSocket сервер остановлен")


async def main():
    """Основная функция запуска сервера"""
    server = WebSocketServer()

    # Обработка сигналов для корректного завершения
    loop = asyncio.get_running_loop()

    def signal_handler():
        loop.create_task(server.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Сервер остановлен по запросу пользователя")
    except Exception as e:
        logger.error(f"Ошибка запуска сервера: {e}")
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())

