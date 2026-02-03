"""
Основной модуль Pattern Recognition Engine
"""

import sys
import asyncio
from pathlib import Path

# Добавляем корневую директорию в путь Python
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from config import config
from utils.logger import setup_logger, logger
from core.pattern_detector import PatternDetector
from core.pattern_analyzer import PatternAnalyzer
from core.pattern_database import PatternDatabase
from core.data_feeder import DataFeeder
from core.ml_models import PatternMLModel
from utils.mt5_connector import MT5Connector
from gui.main_window import MainWindow

class PatternRecognitionEngine:
    """Основной класс Pattern Recognition Engine"""

    def __init__(self, mode: str = None):
        """
        Инициализация движка

        Args:
            mode: Режим работы (file, socket, websocket, api)
        """
        self.mode = mode or config.MODE
        self.logger = logger.bind(engine="main")

        # Инициализация компонентов
        self.detector = None
        self.analyzer = None
        self.database = None
        self.data_feeder = None
        self.ml_model = None
        self.mt5_connector = None

        # Состояние
        self.is_running = False
        self.components_initialized = False

    async def initialize(self):
        """Асинхронная инициализация компонентов"""
        try:
            self.logger.info(f"Инициализация Pattern Recognition Engine в режиме {self.mode}")

            # Валидация конфигурации
            errors = config.validate()
            if errors:
                self.logger.error(f"Ошибки конфигурации: {errors}")
                return False

            # Инициализация базы данных
            self.database = PatternDatabase()
            await self.database.initialize()

            # Инициализация DataFeeder
            self.data_feeder = DataFeeder()

            # Инициализация ML модели
            if config.ML.ENABLED:
                self.ml_model = PatternMLModel(model_type=config.ML.MODEL_TYPE)
                # Загрузка обученной модели если существует
                model_files = list(Path(config.PATHS["models_dir"]).glob("*.pkl"))
                if model_files:
                    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                    self.ml_model.load_model(str(latest_model))
                    self.logger.info(f"Загружена ML модель: {latest_model}")

            # Инициализация PatternDetector
            self.detector = PatternDetector()

            # Инициализация PatternAnalyzer
            self.analyzer = PatternAnalyzer()

            # Подключение к MT5 если настроено
            if config.MT5.ENABLED and config.MT5.AUTO_CONNECT:
                self.mt5_connector = MT5Connector()
                if await self.mt5_connector.connect():
                    self.logger.info("Успешное подключение к MT5")
                else:
                    self.logger.warning("Не удалось подключиться к MT5")

            self.components_initialized = True
            self.logger.info("Инициализация компонентов завершена")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка инициализации: {e}", exc_info=True)
            return False

    async def run_file_mode(self):
        """Запуск в файловом режиме"""
        self.logger.info("Запуск в файловом режиме")

        # Проверка наличия входного файла
        input_file = Path(config.MT5.INPUT_FILE_PATH)
        if not input_file.exists():
            self.logger.error(f"Входной файл не найден: {input_file}")
            return

        try:
            # Чтение данных из файла
            data = await self.data_feeder.read_from_file(str(input_file))
            if not data:
                self.logger.error("Не удалось прочитать данные из файла")
                return

            # Детектирование паттернов
            detection_result = await self.detector.detect_all_patterns(
                symbol="EURUSD",  # Извлекаем из файла или конфига
                timeframe="H1",
                data=data
            )

            # Анализ паттернов
            analyzed_patterns = await self.analyzer.analyze_patterns(
                detection_result.patterns,
                data
            )

            # Сохранение результатов
            output_file = Path(config.MT5.OUTPUT_FILE_PATH)
            await self.data_feeder.save_to_file(
                analyzed_patterns,
                str(output_file)
            )

            # Визуализация
            if config.VISUALIZATION.ENABLE_PLOTTING:
                from utils.visualization import PatternVisualizer
                visualizer = PatternVisualizer()
                plot_file = output_file.with_suffix('.png')
                visualizer.plot_patterns(data, analyzed_patterns, str(plot_file))

            self.logger.info(f"Анализ завершен. Результаты сохранены в {output_file}")

        except Exception as e:
            self.logger.error(f"Ошибка в файловом режиме: {e}", exc_info=True)

    async def run_socket_mode(self):
        """Запуск в режиме сокетов"""
        self.logger.info("Запуск в режиме сокетов")

        try:
            from server.socket_server import SocketServer
            server = SocketServer(
                host="127.0.0.1",
                port=5555,
                detector=self.detector,
                analyzer=self.analyzer,
                data_feeder=self.data_feeder
            )

            await server.start()

        except Exception as e:
            self.logger.error(f"Ошибка в режиме сокетов: {e}", exc_info=True)

    async def run_websocket_mode(self):
        """Запуск в режиме WebSocket"""
        self.logger.info("Запуск в режиме WebSocket")

        try:
            from server.ws_server import WebSocketServer
            server = WebSocketServer(
                host="127.0.0.1",
                port=8765,
                detector=self.detector,
                analyzer=self.analyzer,
                ml_model=self.ml_model
            )

            await server.start()

        except Exception as e:
            self.logger.error(f"Ошибка в режиме WebSocket: {e}", exc_info=True)

    async def run_api_mode(self):
        """Запуск в режиме API"""
        self.logger.info("Запуск в режиме API")

        try:
            from server.api_server import APIServer
            server = APIServer(
                detector=self.detector,
                analyzer=self.analyzer,
                database=self.database,
                ml_model=self.ml_model
            )

            await server.start()

        except Exception as e:
            self.logger.error(f"Ошибка в режиме API: {e}", exc_info=True)

    async def run_gui_mode(self):
        """Запуск в GUI режиме"""
        self.logger.info("Запуск в GUI режиме")

        try:
            app = MainWindow(self)
            app.run()

        except Exception as e:
            self.logger.error(f"Ошибка в GUI режиме: {e}", exc_info=True)

    async def run(self):
        """Основной цикл работы"""
        if not self.components_initialized:
            initialized = await self.initialize()
            if not initialized:
                self.logger.error("Не удалось инициализировать компоненты")
                return

        self.is_running = True

        try:
            # Выбор режима работы
            if self.mode == "file":
                await self.run_file_mode()

            elif self.mode == "socket":
                await self.run_socket_mode()

            elif self.mode == "websocket":
                await self.run_websocket_mode()

            elif self.mode == "api":
                await self.run_api_mode()

            elif self.mode == "gui":
                await self.run_gui_mode()

            else:
                self.logger.error(f"Неизвестный режим работы: {self.mode}")
                return

        except KeyboardInterrupt:
            self.logger.info("Получен сигнал прерывания")
        except Exception as e:
            self.logger.error(f"Критическая ошибка: {e}", exc_info=True)
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Корректное завершение работы"""
        self.logger.info("Завершение работы Pattern Recognition Engine")
        self.is_running = False

        # Закрытие соединений
        if self.mt5_connector:
            await self.mt5_connector.disconnect()

        if self.database:
            await self.database.close()

        self.logger.info("Работа завершена")

def main():
    """Точка входа в приложение"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Pattern Recognition Engine - профессиональная система распознавания паттернов"
    )

    parser.add_argument(
        "--mode",
        choices=["file", "socket", "websocket", "api", "gui"],
        default=None,
        help="Режим работы"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Путь к конфигурационному файлу"
    )

    parser.add_argument(
        "--symbol",
        type=str,
        help="Торговый символ для анализа"
    )

    parser.add_argument(
        "--timeframe",
        type=str,
        default="H1",
        help="Таймфрейм для анализа"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Уровень логирования"
    )

    args = parser.parse_args()

    # Настройка логгера
    setup_logger(level=args.log_level)
    logger.info(f"Запуск Pattern Recognition Engine с аргументами: {args}")

    # Загрузка конфигурации если указана
    if args.config:
        from config import CONFIG
        global config
        config = CONFIG.load(args.config)
        logger.info(f"Загружена конфигурация из {args.config}")

    # Обновление конфигурации аргументами
    if args.mode:
        config.MODE = args.mode

    if args.symbol:
        config.MT5.SYMBOLS = [args.symbol]

    # Создание и запуск движка
    engine = PatternRecognitionEngine(mode=args.mode)

    # Запуск асинхронного цикла
    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        logger.info("Приложение остановлено пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

