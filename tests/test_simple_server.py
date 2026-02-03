"""
Тесты для простого сервера
"""

import unittest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path
import sys

# Добавляем путь к корневой директории проекта
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from server.ws_server import WebSocketServer, ClientInfo


class TestWebSocketServer(unittest.TestCase):
    """Тесты для WebSocket сервера"""

    def setUp(self):
        """Подготовка тестов"""
        self.server = WebSocketServer(host="127.0.0.1", port=8765)

        # Мокаем зависимости
        self.server.detector = MagicMock()
        self.server.analyzer = MagicMock()

        # Создаем mock клиента
        self.mock_websocket = AsyncMock()
        self.mock_websocket.closed = False
        self.mock_websocket.send = AsyncMock()
        self.mock_websocket.close = AsyncMock()

    def test_client_info_creation(self):
        """Тест создания информации о клиенте"""
        client_info = ClientInfo(
            websocket=self.mock_websocket,
            client_id="test_client",
            subscribed_symbols=set(["EURUSD", "GBPUSD"]),
            connected_at=None,
            last_activity=None
        )

        self.assertEqual(client_info.client_id, "test_client")
        self.assertEqual(len(client_info.subscribed_symbols), 2)
        self.assertIn("EURUSD", client_info.subscribed_symbols)

    def test_add_client(self):
        """Тест добавления клиента"""
        client_id = "test_client_1"

        client_info = ClientInfo(
            websocket=self.mock_websocket,
            client_id=client_id,
            subscribed_symbols=set(),
            connected_at=None,
            last_activity=None
        )

        self.server.clients[client_id] = client_info

        self.assertIn(client_id, self.server.clients)
        self.assertEqual(self.server.clients[client_id].client_id, client_id)

    def test_remove_client(self):
        """Тест удаления клиента"""
        client_id = "test_client_2"

        client_info = ClientInfo(
            websocket=self.mock_websocket,
            client_id=client_id,
            subscribed_symbols=set(),
            connected_at=None,
            last_activity=None
        )

        self.server.clients[client_id] = client_info
        self.assertIn(client_id, self.server.clients)

        del self.server.clients[client_id]
        self.assertNotIn(client_id, self.server.clients)

    async def test_process_subscribe_message(self):
        """Тест обработки сообщения подписки"""
        client_id = "test_client_3"

        client_info = ClientInfo(
            websocket=self.mock_websocket,
            client_id=client_id,
            subscribed_symbols=set(),
            connected_at=None,
            last_activity=None
        )

        self.server.clients[client_id] = client_info

        # Создаем сообщение подписки
        message = json.dumps({
            'type': 'subscribe',
            'symbols': ['EURUSD', 'GBPUSD']
        })

        # Мокаем отправку сообщения
        self.mock_websocket.send = AsyncMock()

        # Обрабатываем сообщение
        await self.server._process_client_message(client_id, message)

        # Проверяем что символы добавлены
        self.assertEqual(len(client_info.subscribed_symbols), 2)
        self.assertIn('EURUSD', client_info.subscribed_symbols)
        self.assertIn('GBPUSD', client_info.subscribed_symbols)

        # Проверяем что отправлено подтверждение
        self.mock_websocket.send.assert_called()

    async def test_process_unsubscribe_message(self):
        """Тест обработки сообщения отписки"""
        client_id = "test_client_4"

        client_info = ClientInfo(
            websocket=self.mock_websocket,
            client_id=client_id,
            subscribed_symbols=set(['EURUSD', 'GBPUSD', 'USDJPY']),
            connected_at=None,
            last_activity=None
        )

        self.server.clients[client_id] = client_info

        # Создаем сообщение отписки
        message = json.dumps({
            'type': 'unsubscribe',
            'symbols': ['EURUSD', 'GBPUSD']
        })

        # Мокаем отправку сообщения
        self.mock_websocket.send = AsyncMock()

        # Обрабатываем сообщение
        await self.server._process_client_message(client_id, message)

        # Проверяем что символы удалены
        self.assertEqual(len(client_info.subscribed_symbols), 1)
        self.assertNotIn('EURUSD', client_info.subscribed_symbols)
        self.assertNotIn('GBPUSD', client_info.subscribed_symbols)
        self.assertIn('USDJPY', client_info.subscribed_symbols)

        # Проверяем что отправлено подтверждение
        self.mock_websocket.send.assert_called()

    async def test_process_ping_message(self):
        """Тест обработки ping сообщения"""
        client_id = "test_client_5"

        client_info = ClientInfo(
            websocket=self.mock_websocket,
            client_id=client_id,
            subscribed_symbols=set(),
            connected_at=None,
            last_activity=None
        )

        self.server.clients[client_id] = client_info

        # Создаем ping сообщение
        message = json.dumps({
            'type': 'ping'
        })

        # Мокаем отправку сообщения
        self.mock_websocket.send = AsyncMock()

        # Обрабатываем сообщение
        await self.server._process_client_message(client_id, message)

        # Проверяем что отправлен pong
        self.mock_websocket.send.assert_called_once()

        # Проверяем что отправлен правильный ответ
        call_args = self.mock_websocket.send.call_args[0][0]
        response = json.loads(call_args)
        self.assertEqual(response['type'], 'pong')

    def test_get_stats(self):
        """Тест получения статистики"""
        stats = self.server.get_stats()

        self.assertIn('connections_total', stats)
        self.assertIn('messages_sent', stats)
        self.assertIn('messages_received', stats)
        self.assertIn('errors', stats)
        self.assertIn('connected_clients', stats)
        self.assertIn('timestamp', stats)

        # Проверяем начальные значения
        self.assertEqual(stats['connections_total'], 0)
        self.assertEqual(stats['connected_clients'], 0)

    async def test_send_to_client(self):
        """Тест отправки сообщения клиенту"""
        client_id = "test_client_6"

        client_info = ClientInfo(
            websocket=self.mock_websocket,
            client_id=client_id,
            subscribed_symbols=set(),
            connected_at=None,
            last_activity=None
        )

        self.server.clients[client_id] = client_info

        # Тестовое сообщение
        test_message = {'type': 'test', 'data': 'test_data'}

        # Мокаем отправку
        self.mock_websocket.send = AsyncMock()

        # Отправляем сообщение
        await self.server._send_to_client(client_id, test_message)

        # Проверяем что сообщение отправлено
        self.mock_websocket.send.assert_called_once_with(json.dumps(test_message))

        # Увеличился счетчик сообщений
        self.assertEqual(self.server.stats['messages_sent'], 1)

    async def test_send_to_nonexistent_client(self):
        """Тест отправки сообщения несуществующему клиенту"""
        # Пытаемся отправить сообщение несуществующему клиенту
        await self.server._send_to_client('nonexistent', {'type': 'test'})

        # Не должно быть ошибок
        # Счетчик сообщений не должен увеличиться
        self.assertEqual(self.server.stats['messages_sent'], 0)


class TestServerCleanup(unittest.TestCase):
    """Тесты очистки сервера"""

    def setUp(self):
        """Подготовка тестов"""
        self.server = WebSocketServer()

    async def test_cleanup_inactive_clients(self):
        """Тест очистки неактивных клиентов"""
        # Создаем mock клиентов
        active_client = AsyncMock()
        active_client.closed = False
        active_client.close = AsyncMock()

        inactive_client = AsyncMock()
        inactive_client.closed = False
        inactive_client.close = AsyncMock()

        # Добавляем клиентов
        self.server.clients['active'] = ClientInfo(
            websocket=active_client,
            client_id='active',
            subscribed_symbols=set(),
            connected_at=None,
            last_activity=None  # Будет обновлено
        )

        self.server.clients['inactive'] = ClientInfo(
            websocket=inactive_client,
            client_id='inactive',
            subscribed_symbols=set(),
            connected_at=None,
            last_activity=None  # Остается None (очень старое)
        )

        # Обновляем время активности активного клиента
        import datetime
        self.server.clients['active'].last_activity = datetime.datetime.now()

        # Запускаем очистку
        await self.server._cleanup_inactive_clients()

        # Проверяем что неактивный клиент был закрыт
        inactive_client.close.assert_called_once()

        # Проверяем что активный клиент не был закрыт
        active_client.close.assert_not_called()


if __name__ == '__main__':
    # Запуск асинхронных тестов
    import asyncio

    # Создаем event loop для асинхронных тестов
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Запускаем тесты
    unittest.main()

