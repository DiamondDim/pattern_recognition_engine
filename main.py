"""
Основной файл Pattern Recognition Engine
"""

import asyncio
import sys
import signal
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path

from config import CONFIG, MT5_CONFIG, DATA_DIR, OUTPUT_DIR
from core.pattern_detector import PatternDetector
from core.pattern_analyzer import PatternAnalyzer
from core.pattern_database import PatternDatabase
from utils.logger import logger, pattern_logger
from utils.mt5_connector import FileConnector, MT5DataExporter
from utils.visualization import PatternVisualizer, create_pattern_report


class PatternRecognitionEngine:
    """Основной класс движка распознавания паттернов"""

    def __init__(self):
        self.logger = logger.bind(name="PREngine")
        self.is_running = False

        # Инициализация компонентов
        self.detector = PatternDetector()
        self.analyzer = PatternAnalyzer()
        self.database = PatternDatabase()
        self.visualizer = PatternVisualizer()
        self.exporter = MT5DataExporter(connector_type=MT5_CONFIG.CONNECTION_MODE)

        # Файловый коннектор для режима файлового обмена
        self.file_connector = FileConnector()

        # Состояние движка
        self.current_symbol = "EURUSD"
        self.current_timeframe = "H1"
        self.last_processed_time = None

        # Статистика
        self.engine_stats = {
            'start_time': None,
            'total_cycles': 0,
            'patterns_found_total': 0,
            'signals_generated': 0,
            'errors': 0
        }

    async def initialize(self):
        """Инициализация движка"""
        self.logger.info("=" * 60)
        self.logger.info("PATTERN RECOGNITION ENGINE")
        self.logger.info(f"Version: {CONFIG.VERSION}")
        self.logger.info(f"Mode: {MT5_CONFIG.CONNECTION_MODE}")
        self.logger.info("=" * 60)

        # Загружаем исторические данные для анализа
        await self._load_historical_data()

        # Проверяем подключение к MT5
        if MT5_CONFIG.CONNECTION_MODE in ['socket', 'websocket']:
            connected = await self.exporter.connector.connect()
            if not connected:
                self.logger.warning("Не удалось подключиться к MT5, переходим в автономный режим")

        self.engine_stats['start_time'] = datetime.now()
        self.is_running = True

        self.logger.info("Движок инициализирован и готов к работе")

    async def _load_historical_data(self):
        """Загрузка исторических данных для анализа"""
        try:
            self.logger.info("Загрузка исторических данных...")

            # Получаем исторические паттерны из базы данных
            historical_patterns = self.database.get_historical_patterns_for_analysis(
                days_back=30, min_quality=0.6
            )

            if historical_patterns:
                # Строим модель для анализа
                self.analyzer.build_prediction_model(historical_patterns)
                self.logger.info(f"Загружено {len(historical_patterns)} исторических паттернов")
            else:
                self.logger.warning("Исторические данные не найдены, анализ будет ограничен")

        except Exception as e:
            self.logger.error(f"Ошибка загрузки исторических данных: {e}")

    async def run_cycle(self):
        """Выполнение одного цикла анализа"""
        try:
            self.engine_stats['total_cycles'] += 1
            cycle_start = datetime.now()

            self.logger.debug(f"Начало цикла анализа #{self.engine_stats['total_cycles']}")

            # 1. Получение данных
            data = await self._get_data_from_source()
            if data is None or data.empty:
                self.logger.warning("Нет данных для анализа, пропускаем цикл")
                await asyncio.sleep(MT5_CONFIG.UPDATE_INTERVAL_SEC)
                return

            # 2. Конвертация данных в numpy arrays
            ohlc_data = self._convert_to_ohlc(data)

            # 3. Детектирование паттернов
            detection_result = self.detector.detect_all_patterns(
                symbol=self.current_symbol,
                timeframe=self.current_timeframe,
                data=ohlc_data
            )

            # 4. Анализ найденных паттернов
            analyzed_patterns = []
            for pattern in detection_result.patterns:
                # Поиск исторических аналогов
                historical_patterns = self.database.get_patterns(
                    symbol=self.current_symbol,
                    timeframe=self.current_timeframe,
                    pattern_type=pattern.get('type'),
                    direction=pattern.get('direction'),
                    limit=100
                )

                historical_dicts = [p.to_dict() for p in historical_patterns]

                # Анализ качества
                quality_analysis = self.analyzer.analyze_pattern_quality(pattern)
                pattern['quality_analysis'] = quality_analysis

                # Прогнозирование исхода
                prediction = self.analyzer.predict_pattern_outcome(
                    pattern, historical_dicts
                )
                pattern['prediction'] = prediction

                # Статистика из исторических аналогов
                if historical_dicts:
                    similar_patterns = self.analyzer.find_similar_patterns(
                        pattern, historical_dicts, n_neighbors=10
                    )

                    if similar_patterns:
                        success_rate = self.analyzer.calculate_success_rate(similar_patterns)
                        avg_profit = self.analyzer.calculate_average_profit(similar_patterns)

                        pattern['historical_statistics'] = {
                            'similar_patterns_count': len(similar_patterns),
                            'historical_success_rate': success_rate,
                            'average_profit': avg_profit,
                            'most_similar_patterns': [
                                {'id': p['id'], 'similarity': s}
                                for p, s in similar_patterns[:3]
                            ]
                        }

                analyzed_patterns.append(pattern)

                # Логирование обнаруженного паттерна
                pattern_logger.pattern_detected(
                    pattern_name=pattern['name'],
                    symbol=pattern['metadata']['symbol'],
                    timeframe=pattern['metadata']['timeframe'],
                    confidence=pattern['metadata']['confidence']
                )

            # 5. Сохранение паттернов в базу данных
            for pattern in analyzed_patterns:
                self.database.save_pattern(pattern)

            # 6. Генерация торговых сигналов
            signals = await self._generate_trading_signals(analyzed_patterns)

            # 7. Визуализация
            await self._visualize_results(analyzed_patterns, ohlc_data)

            # 8. Экспорт результатов в MT5
            await self._export_results_to_mt5(analyzed_patterns, signals)

            # 9. Обновление статистики
            self.engine_stats['patterns_found_total'] += len(analyzed_patterns)
            self.engine_stats['signals_generated'] += len(signals)

            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            self.logger.info(
                f"Цикл #{self.engine_stats['total_cycles']} завершен: "
                f"{len(analyzed_patterns)} паттернов, "
                f"{len(signals)} сигналов, "
                f"время: {cycle_duration:.2f}с"
            )

            # Ждем перед следующим циклом
            await asyncio.sleep(MT5_CONFIG.UPDATE_INTERVAL_SEC)

        except Exception as e:
            self.logger.error(f"Ошибка в цикле анализа: {e}", exc_info=True)
            self.engine_stats['errors'] += 1
            await asyncio.sleep(MT5_CONFIG.UPDATE_INTERVAL_SEC * 2)  # Удваиваем задержку при ошибке

    async def _get_data_from_source(self):
        """Получение данных из источника (MT5 или файла)"""
        if MT5_CONFIG.CONNECTION_MODE == 'file':
            # Чтение из файла
            data = self.file_connector.read_data()
            return data

        else:
            # Запрос данных через сокет/WebSocket
            # TODO: Реализовать получение данных через сокет
            self.logger.warning("Режим сокета/WebSocket пока не реализован полностью")
            return None

    def _convert_to_ohlc(self, data):
        """Конвертация DataFrame в формат OHLC"""
        ohlc_data = {
            'timestamp': data['timestamp'].values if 'timestamp' in data.columns else None,
            'open': data['open'].values.astype(float),
            'high': data['high'].values.astype(float),
            'low': data['low'].values.astype(float),
            'close': data['close'].values.astype(float)
        }

        if 'volume' in data.columns:
            ohlc_data['volume'] = data['volume'].values.astype(float)

        return ohlc_data

    async def _generate_trading_signals(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Генерация торговых сигналов на основе паттернов"""
        signals = []

        for pattern in patterns:
            # Фильтруем паттерны по качеству
            quality_analysis = pattern.get('quality_analysis', {})
            overall_score = quality_analysis.get('overall_score', 0)

            if overall_score >= 0.6:  # Минимальный порог качества
                # Создаем сигнал
                signal = {
                    'id': f"signal_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(signals)}",
                    'pattern_id': pattern.get('id'),
                    'symbol': pattern.get('metadata', {}).get('symbol'),
                    'timeframe': pattern.get('metadata', {}).get('timeframe'),
                    'signal_type': 'pattern_based',
                    'direction': pattern.get('direction'),
                    'pattern_name': pattern.get('name'),
                    'quality_score': overall_score,
                    'entry_price': pattern.get('targets', {}).get('entry_price'),
                    'stop_loss': pattern.get('targets', {}).get('stop_loss'),
                    'take_profit': pattern.get('targets', {}).get('take_profit'),
                    'risk_amount': pattern.get('targets', {}).get('risk_amount'),
                    'reward_amount': pattern.get('targets', {}).get('reward_amount'),
                    'profit_risk_ratio': pattern.get('targets', {}).get('profit_risk_ratio'),
                    'confidence': pattern.get('metadata', {}).get('confidence', 0),
                    'generated_time': datetime.now().isoformat()
                }

                signals.append(signal)

                # Логирование сигнала
                pattern_logger.trading_signal(
                    symbol=signal['symbol'],
                    pattern=signal['pattern_name'],
                    direction=signal['direction'],
                    entry=signal['entry_price'],
                    stop_loss=signal['stop_loss'],
                    take_profit=signal['take_profit']
                )

        # Сохраняем сигналы в базу данных
        for signal in signals:
            self.database.save_trading_signal(signal['pattern_id'], signal)

        # Сохраняем сигналы в файл
        if MT5_CONFIG.CONNECTION_MODE == 'file':
            self.file_connector.write_signals(signals)

        return signals

    async def _visualize_results(self, patterns: List[Dict[str, Any]], ohlc_data: Dict[str, Any]):
        """Визуализация результатов анализа"""
        if not patterns or not self.config.ENABLE_PLOTTING:
            return

        try:
            # Создаем график для каждого паттерна
            for i, pattern in enumerate(patterns):
                # Сохраняем график
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"pattern_{pattern['name']}_{timestamp}_{i}.png"
                save_path = OUTPUT_DIR / filename

                fig = self.visualizer.plot_pattern(pattern, ohlc_data, save_path=str(save_path))

                if fig:
                    plt.close(fig)  # Закрываем фигуру чтобы освободить память

            # Создаем сводный график для всех паттернов
            if len(patterns) > 1:
                summary_filename = f"patterns_summary_{timestamp}.png"
                summary_path = OUTPUT_DIR / summary_filename

                fig = self.visualizer.plot_multiple_patterns(
                    patterns, ohlc_data, save_path=str(summary_path)
                )

                if fig:
                    plt.close(fig)

        except Exception as e:
            self.logger.error(f"Ошибка визуализации: {e}")

    async def _export_results_to_mt5(self, patterns: List[Dict[str, Any]], signals: List[Dict[str, Any]]):
        """Экспорт результатов в MT5"""
        if not patterns:
            return

        try:
            # Подготавливаем данные для экспорта
            export_data = {
                'patterns': patterns,
                'signals': signals,
                'timestamp': datetime.now().isoformat(),
                'symbol': self.current_symbol,
                'timeframe': self.current_timeframe
            }

            # Экспортируем через соответствующий коннектор
            if MT5_CONFIG.CONNECTION_MODE == 'file':
                # Сохраняем в JSON файл
                filename = f"patterns_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                filepath = OUTPUT_DIR / filename

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str)

                self.logger.debug(f"Результаты экспортированы в файл: {filepath}")

            else:
                # Отправляем через сокет/WebSocket
                await self.exporter.import_patterns_to_mt5(patterns)

        except Exception as e:
            self.logger.error(f"Ошибка экспорта результатов в MT5: {e}")

    async def run(self):
        """Основной цикл работы движка"""
        await self.initialize()

        self.logger.info("Запуск основного цикла анализа...")

        # Регистрием обработчик сигналов для корректного завершения
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Основной цикл
        while self.is_running:
            try:
                await self.run_cycle()
            except KeyboardInterrupt:
                self.logger.info("Получен сигнал прерывания")
                break
            except Exception as e:
                self.logger.error(f"Критическая ошибка в основном цикле: {e}")
                self.engine_stats['errors'] += 1

                # Ждем перед повторной попыткой
                await asyncio.sleep(30)

        await self.shutdown()

    def _signal_handler(self, signum, frame):
        """Обработчик сигналов для корректного завершения"""
        self.logger.info(f"Получен сигнал {signum}, завершение работы...")
        self.is_running = False

    async def shutdown(self):
        """Корректное завершение работы движка"""
        self.logger.info("Завершение работы Pattern Recognition Engine...")

        # Закрываем соединения
        await self.exporter.close()

        # Сохраняем статистику
        await self._save_statistics()

        # Закрываем базу данных
        self.database.close()

        self.logger.info("Работа движка завершена")
        self._print_summary()

    async def _save_statistics(self):
        """Сохранение статистики работы"""
        try:
            stats = {
                'engine': self.engine_stats,
                'detector': self.detector.get_statistics(),
                'analyzer': self.analyzer.get_statistics(),
                'database': self.database.get_database_stats(),
                'file_connector': self.file_connector.get_stats() if hasattr(self, 'file_connector') else {},
                'exporter': self.exporter.get_stats() if hasattr(self, 'exporter') else {},
                'end_time': datetime.now().isoformat()
            }

            # Конвертируем datetime в строки
            for key in ['start_time', 'end_time', 'last_processed_time']:
                if key in stats['engine'] and stats['engine'][key]:
                    stats['engine'][key] = stats['engine'][key].isoformat()

            # Сохраняем в файл
            filename = f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = OUTPUT_DIR / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, default=str)

            self.logger.info(f"Статистика сохранена в {filepath}")

        except Exception as e:
            self.logger.error(f"Ошибка сохранения статистики: {e}")

    def _print_summary(self):
        """Вывод сводки работы"""
        duration = None
        if self.engine_stats['start_time']:
            end_time = datetime.now()
            duration = end_time - self.engine_stats['start_time']

        summary = f"""
{'=' * 60}
Итог работы Pattern Recognition Engine:
{'=' * 60}
Общее время работы: {duration}
Всего циклов анализа: {self.engine_stats['total_cycles']}
Всего найдено паттернов: {self.engine_stats['patterns_found_total']}
Сгенерировано сигналов: {self.engine_stats['signals_generated']}
Ошибок: {self.engine_stats['errors']}

Детектор:
  Всего обработано: {self.detector.detection_stats['total_processed']}
  Найдено паттернов: {self.detector.detection_stats['patterns_found']}
  Среднее качество: {self.detector.detection_stats['avg_quality']:.2f}

Анализатор:
  Проанализировано паттернов: {self.analyzer.analysis_stats['total_patterns_analyzed']}
  Найдено аналогов: {self.analyzer.analysis_stats['similar_patterns_found']}

База данных:
  Всего паттернов: {self.database.get_database_stats().get('total_patterns', 0)}
{'=' * 60}
        """

        self.logger.info(summary)


async def main():
    """Основная функция"""
    # Создаем и запускаем движок
    engine = PatternRecognitionEngine()

    try:
        await engine.run()
    except Exception as e:
        logger.critical(f"Критическая ошибка в main: {e}")
        await engine.shutdown()
        raise


if __name__ == "__main__":
    # Запуск асинхронного event loop
    asyncio.run(main())

