"""
WebSocket сервер для Pattern Recognition Engine
"""

import asyncio
import json
import websockets
from typing import Dict, Any, Optional, Set, List
from datetime import datetime
from dataclasses import dataclass, asdict

from config import config
from utils.logger import logger
from core.pattern_detector import PatternDetector
from core.pattern_analyzer import PatternAnalyzer
from core.ml_models import PatternMLModel

@dataclass
class ClientInfo:
    """Информация о подключенном клиенте"""
    websocket: websockets.WebSocketServerProtocol
    client_id: str
    subscribed_symbols: Set[str]
    connected_at: datetime
    last_activity: datetime

class WebSocketServer:
    """WebSocket сервер для передачи данных в реальном времени"""

    def __init__(self,
                 host: str = "127.0.0.1",
                 port: int = 8765,
                 detector: Optional[PatternDetector] = None,
                 analyzer: Optional[PatternAnalyzer] = None,
                 ml_model: Optional[PatternMLModel] = None):

        self.host = host
        self.port = port
        self.logger = logger.bind(module="ws_server")

        # Компоненты системы
        self.detector = detector or PatternDetector()
        self.analyzer = analyzer or PatternAnalyzer()
        self.ml_model = ml_model

        # Клиенты
        self.clients: Dict[str, ClientInfo] = {}
        self.server: Optional[websockets.WebSocketServer] = None

        # Статистика
        self.stats = {
            'connections_total': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'errors': 0
        }

    async def start(self):
        """Запуск WebSocket сервера"""
        try:
            self.server = await websockets.serve(
                self._handle_client,
                self.host,
                self.port
            )

            self.logger.info(f"WebSocket сервер запущен на ws://{self.host}:{self.port}")

            # Запуск фоновых задач
            asyncio.create_task(self._broadcast_market_data())
            asyncio.create_task(self._cleanup_inactive_clients())

            # Ожидание завершения
            await self.server.wait_closed()

        except Exception as e:
            self.logger.error(f"Ошибка запуска WebSocket сервера: {e}")

    async def stop(self):
        """Остановка WebSocket сервера"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.logger.info("WebSocket сервер остановлен")

    async def _handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Обработка подключения клиента"""
        client_id = f"client_{self.stats['connections_total'] + 1}"

        # Создаем информацию о клиенте
        client_info = ClientInfo(
            websocket=websocket,
            client_id=client_id,
            subscribed_symbols=set(),
            connected_at=datetime.now(),
            last_activity=datetime.now()
        )

        self.clients[client_id] = client_info
        self.stats['connections_total'] += 1

        self.logger.info(f"Клиент подключен: {client_id}")

        try:
            # Отправляем приветственное сообщение
            await self._send_to_client(client_id, {
                'type': 'welcome',
                'client_id': client_id,
                'timestamp': datetime.now().isoformat(),
                'message': 'Connected to Pattern Recognition Engine WebSocket Server'
            })

            # Обработка сообщений от клиента
            async for message in websocket:
                await self._process_client_message(client_id, message)

        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Клиент отключен: {client_id}")
        except Exception as e:
            self.logger.error(f"Ошибка обработки клиента {client_id}: {e}")
            self.stats['errors'] += 1
        finally:
            # Удаляем клиента
            if client_id in self.clients:
                del self.clients[client_id]

    async def _process_client_message(self, client_id: str, message: str):
        """Обработка сообщения от клиента"""
        try:
            data = json.loads(message)
            self.stats['messages_received'] += 1

            client_info = self.clients.get(client_id)
            if not client_info:
                return

            # Обновляем время последней активности
            client_info.last_activity = datetime.now()

            message_type = data.get('type', 'unknown')

            if message_type == 'subscribe':
                # Подписка на символы
                symbols = data.get('symbols', [])
                if isinstance(symbols, list):
                    client_info.subscribed_symbols.update(symbols)

                    await self._send_to_client(client_id, {
                        'type': 'subscription_confirmed',
                        'symbols': list(client_info.subscribed_symbols),
                        'timestamp': datetime.now().isoformat()
                    })

                    self.logger.info(f"Клиент {client_id} подписался на: {symbols}")

            elif message_type == 'unsubscribe':
                # Отписка от символов
                symbols = data.get('symbols', [])
                if isinstance(symbols, list):
                    for symbol in symbols:
                        client_info.subscribed_symbols.discard(symbol)

                    await self._send_to_client(client_id, {
                        'type': 'unsubscription_confirmed',
                        'symbols': list(client_info.subscribed_symbols),
                        'timestamp': datetime.now().isoformat()
                    })

            elif message_type == 'get_patterns':
                # Запрос на получение паттернов
                symbol = data.get('symbol', 'EURUSD')
                timeframe = data.get('timeframe', 'H1')
                bars = data.get('bars', 100)

                await self._process_pattern_request(client_id, symbol, timeframe, bars)

            elif message_type == 'analyze':
                # Запрос на анализ
                symbol = data.get('symbol', 'EURUSD')
                pattern_data = data.get('pattern', {})

                await self._process_analysis_request(client_id, symbol, pattern_data)

            elif message_type == 'ping':
                # Ping-запрос
                await self._send_to_client(client_id, {
                    'type': 'pong',
                    'timestamp': datetime.now().isoformat()
                })

            else:
                self.logger.warning(f"Неизвестный тип сообщения от клиента {client_id}: {message_type}")

        except json.JSONDecodeError:
            self.logger.error(f"Невалидный JSON от клиента {client_id}")
        except Exception as e:
            self.logger.error(f"Ошибка обработки сообщения от клиента {client_id}: {e}")

    async def _process_pattern_request(self, client_id: str, symbol: str, timeframe: str, bars: int):
        """Обработка запроса на получение паттернов"""
        try:
            # Здесь должна быть логика загрузки данных и детектирования паттернов
            # Для примера отправляем заглушку

            await self._send_to_client(client_id, {
                'type': 'patterns',
                'symbol': symbol,
                'timeframe': timeframe,
                'patterns': [],
                'timestamp': datetime.now().isoformat(),
                'message': 'Pattern detection not implemented in WebSocket example'
            })

        except Exception as e:
            self.logger.error(f"Ошибка обработки запроса паттернов: {e}")

            await self._send_to_client(client_id, {
                'type': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })

    async def _process_analysis_request(self, client_id: str, symbol: str, pattern_data: Dict[str, Any]):
        """Обработка запроса на анализ"""
        try:
            # Здесь должна быть логика анализа паттерна
            # Для примера отправляем заглушку

            await self._send_to_client(client_id, {
                'type': 'analysis',
                'symbol': symbol,
                'pattern': pattern_data,
                'analysis': {
                    'quality': 0.75,
                    'confidence': 0.8,
                    'direction': 'bullish',
                    'targets': {
                        'entry': 1.1000,
                        'stop_loss': 1.0950,
                        'take_profit': 1.1100
                    }
                },
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            self.logger.error(f"Ошибка обработки запроса анализа: {e}")

            await self._send_to_client(client_id, {
                'type': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })

    async def _broadcast_market_data(self):
        """Рассылка рыночных данных всем подписанным клиентам"""
        import time
        import random

        while True:
            try:
                if not self.clients:
                    await asyncio.sleep(1)
                    continue

                # Собираем все подписанные символы
                all_symbols = set()
                for client_info in self.clients.values():
                    all_symbols.update(client_info.subscribed_symbols)

                # Для каждого символа генерируем обновление цены
                for symbol in all_symbols:
                    # Генерация случайного изменения цены
                    price_change = random.uniform(-0.001, 0.001)

                    # Создаем сообщение
                    message = {
                        'type': 'price_update',
                        'symbol': symbol,
                        'bid': 1.1000 + price_change,
                        'ask': 1.1005 + price_change,
                        'high': 1.1020 + price_change,
                        'low': 1.0980 + price_change,
                        'volume': random.randint(1000, 10000),
                        'timestamp': datetime.now().isoformat()
                    }

                    # Отправляем всем клиентам, подписанным на этот символ
                    for client_id, client_info in self.clients.items():
                        if symbol in client_info.subscribed_symbols:
                            await self._send_to_client(client_id, message)

                # Интервал обновления
                await asyncio.sleep(1)

            except Exception as e:
                self.logger.error(f"Ошибка рассылки рыночных данных: {e}")
                await asyncio.sleep(5)

    async def _cleanup_inactive_clients(self):
        """Очистка неактивных клиентов"""
        while True:
            try:
                current_time = datetime.now()
                inactive_clients = []

                for client_id, client_info in self.clients.items():
                    # Клиент считается неактивным если не было активности 5 минут
                    inactivity_period = current_time - client_info.last_activity
                    if inactivity_period.total_seconds() > 300:  # 5 минут
                        inactive_clients.append(client_id)

                # Закрываем соединения с неактивными клиентами
                for client_id in inactive_clients:
                    client_info = self.clients.get(client_id)
                    if client_info:
                        try:
                            await client_info.websocket.close()
                            self.logger.info(f"Закрыто соединение с неактивным клиентом: {client_id}")
                        except:
                            pass

                await asyncio.sleep(60)  # Проверка каждую минуту

            except Exception as e:
                self.logger.error(f"Ошибка очистки неактивных клиентов: {e}")
                await asyncio.sleep(60)

    async def _send_to_client(self, client_id: str, data: Dict[str, Any]):
        """Отправка данных конкретному клиенту"""
        try:
            client_info = self.clients.get(client_id)
            if client_info and not client_info.websocket.closed:
                await client_info.websocket.send(json.dumps(data))
                self.stats['messages_sent'] += 1
        except Exception as e:
            self.logger.error(f"Ошибка отправки данных клиенту {client_id}: {e}")

    async def broadcast_to_all(self, data: Dict[str, Any]):
        """Широковещательная рассылка всем клиентам"""
        tasks = []
        for client_id in list(self.clients.keys()):
            tasks.append(self._send_to_client(client_id, data))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики сервера"""
        return {
            **self.stats,
            'connected_clients': len(self.clients),
            'timestamp': datetime.now().isoformat()
        }

    async def send_pattern_detection(self,
                                    symbol: str,
                                    timeframe: str,
                                    patterns: List[Dict[str, Any]]):
        """Отправка обнаруженных паттернов всем подписанным клиентам"""
        message = {
            'type': 'pattern_detection',
            'symbol': symbol,
            'timeframe': timeframe,
            'patterns': patterns,
            'timestamp': datetime.now().isoformat()
        }

        await self.broadcast_to_all(message)

    async def send_trading_signal(self,
                                 symbol: str,
                                 signal: Dict[str, Any]):
        """Отправка торгового сигнала всем подписанным клиентам"""
        message = {
            'type': 'trading_signal',
            'symbol': symbol,
            'signal': signal,
            'timestamp': datetime.now().isoformat()
        }

        await self.broadcast_to_all(message)

def run_websocket_server():
    """Запуск WebSocket сервера (точка входа)"""
    import argparse

    parser = argparse.ArgumentParser(description="WebSocket Server for Pattern Recognition Engine")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8765, help="Port number")

    args = parser.parse_args()

    server = WebSocketServer(host=args.host, port=args.port)

    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\nСервер остановлен")
    except Exception as e:
        logger.error(f"Ошибка запуска сервера: {e}")

if __name__ == "__main__":
    run_websocket_server()

