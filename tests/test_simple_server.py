# test_simple_server.py
import socket
import json
import threading
from datetime import datetime


class SimpleTestServer:
    def __init__(self, host='127.0.0.1', port=5555):
        self.host = host
        self.port = port
        self.server = None
        self.running = False

    def start(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(5)
        self.running = True

        print(f"Тестовый сервер запущен на {self.host}:{self.port}")
        print("Ожидание подключений от MT5...")

        try:
            while self.running:
                client, address = self.server.accept()
                print(f"\nНовое подключение от {address}")

                # Запускаем обработку в отдельном потоке
                thread = threading.Thread(target=self.handle_client, args=(client, address))
                thread.daemon = True
                thread.start()

        except KeyboardInterrupt:
            print("\nСервер остановлен")
        finally:
            self.stop()

    def handle_client(self, client, address):
        try:
            # Получаем данные
            data = b""
            client.settimeout(5.0)

            while True:
                chunk = client.recv(4096)
                if not chunk:
                    break
                data += chunk

                # Проверяем конец JSON (ищем закрывающую скобку)
                if b'}' in data:
                    break

            if data:
                try:
                    # Декодируем UTF-8
                    json_str = data.decode('utf-8', errors='ignore')
                    print(f"Получено {len(json_str)} символов")

                    # Пытаемся распарсить JSON
                    try:
                        parsed = json.loads(json_str)
                        print("✅ JSON успешно распарсен")
                        print(f"   Символ: {parsed.get('symbol')}")
                        print(f"   Таймфрейм: {parsed.get('timeframe')}")
                        print(f"   Баров: {parsed.get('count')}")

                        if 'data' in parsed and len(parsed['data']) > 0:
                            print(f"   Первый бар: {parsed['data'][0]}")
                            print(f"   Последний бар: {parsed['data'][-1]}")

                    except json.JSONDecodeError as e:
                        print(f"❌ Ошибка парсинга JSON: {e}")
                        print(f"Первые 500 символов данных:")
                        print(json_str[:500])

                    # Отправляем ответ
                    response = {
                        "status": "success",
                        "message": "Данные успешно получены",
                        "patterns_count": 3,
                        "patterns": [
                            {"type": "head_shoulders", "direction": "bearish", "quality": 0.85},
                            {"type": "double_top", "direction": "bearish", "quality": 0.72},
                            {"type": "abcd", "direction": "bullish", "quality": 0.68}
                        ],
                        "timestamp": datetime.now().isoformat(),
                        "received_bytes": len(data),
                        "symbol": parsed.get('symbol', 'unknown')
                    }

                    response_json = json.dumps(response, indent=2)
                    client.send(response_json.encode('utf-8'))
                    print(f"Отправлен ответ: {len(response_json)} символов")

                except Exception as e:
                    print(f"Ошибка обработки данных: {e}")

        except socket.timeout:
            print("Таймаут ожидания данных")
        except Exception as e:
            print(f"Ошибка обработки клиента: {e}")
        finally:
            client.close()
            print(f"Соединение с {address} закрыто")

    def stop(self):
        self.running = False
        if self.server:
            self.server.close()


if __name__ == "__main__":
    server = SimpleTestServer()
    server.start()

