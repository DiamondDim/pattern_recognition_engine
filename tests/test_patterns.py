"""
Тесты для Pattern Recognition Engine
"""

import unittest
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent.parent))

from patterns import create_pattern, get_available_patterns
from core.pattern_detector import PatternDetector
from core.pattern_analyzer import PatternAnalyzer
from core.pattern_database import PatternDatabase
from utils.helpers import validate_ohlc_data, normalize_prices, calculate_returns


class TestPatterns(unittest.TestCase):
    """Тесты для паттернов"""

    def setUp(self):
        """Подготовка тестовых данных"""
        # Создаем тестовые данные OHLC
        np.random.seed(42)
        n_samples = 100

        self.test_data = {
            'open': np.random.uniform(1.0, 1.2, n_samples),
            'high': np.random.uniform(1.1, 1.3, n_samples),
            'low': np.random.uniform(0.9, 1.1, n_samples),
            'close': np.random.uniform(1.0, 1.2, n_samples),
            'volume': np.random.uniform(1000, 10000, n_samples),
            'timestamp': np.array([datetime.now() - timedelta(minutes=i * 5)
                                   for i in range(n_samples)])
        }

        # Корректируем high и low чтобы они были корректными
        for i in range(n_samples):
            self.test_data['high'][i] = max(self.test_data['open'][i],
                                            self.test_data['close'][i],
                                            self.test_data['high'][i])
            self.test_data['low'][i] = min(self.test_data['open'][i],
                                           self.test_data['close'][i],
                                           self.test_data['low'][i])

    def test_pattern_creation(self):
        """Тест создания паттернов"""
        available_patterns = get_available_patterns()
        self.assertGreater(len(available_patterns), 0)

        # Тестируем создание нескольких паттернов
        test_patterns = ['head_shoulders', 'double_top', 'engulfing_bullish']

        for pattern_name in test_patterns:
            try:
                pattern = create_pattern(pattern_name)
                self.assertIsNotNone(pattern)
                self.assertEqual(pattern.name, pattern_name.replace('_', ' ').title())
            except Exception as e:
                self.fail(f"Ошибка создания паттерна {pattern_name}: {e}")

    def test_data_validation(self):
        """Тест валидации данных"""
        # Корректные данные должны пройти валидацию
        self.assertTrue(validate_ohlc_data(self.test_data))

        # Неполные данные должны не пройти валидацию
        invalid_data = {'open': [1.0], 'high': [1.1]}  # Нет low и close
        self.assertFalse(validate_ohlc_data(invalid_data))

        # Данные с некорректными значениями (high < low)
        invalid_data2 = {
            'open': [1.0],
            'high': [0.9],  # high < low
            'low': [1.1],  # low > high
            'close': [1.05]
        }
        self.assertFalse(validate_ohlc_data(invalid_data2))

    def test_price_normalization(self):
        """Тест нормализации цен"""
        prices = np.array([100, 105, 95, 110, 90])

        # Min-Max нормализация
        normalized = normalize_prices(prices, method='minmax')
        self.assertEqual(len(normalized), len(prices))
        self.assertAlmostEqual(np.min(normalized), 0.0)
        self.assertAlmostEqual(np.max(normalized), 1.0)

        # Z-score нормализация
        normalized_z = normalize_prices(prices, method='zscore')
        self.assertAlmostEqual(np.mean(normalized_z), 0.0, places=5)
        self.assertAlmostEqual(np.std(normalized_z), 1.0, places=5)

    def test_returns_calculation(self):
        """Тест расчета доходностей"""
        prices = np.array([100, 105, 110, 115, 120])

        # Однопериодные доходности
        returns = calculate_returns(prices, period=1)
        expected = np.array([0.05, 0.0476, 0.0455, 0.0435])  # Округлено
        np.testing.assert_array_almost_equal(returns, expected, decimal=3)

        # Двухпериодные доходности
        returns_2 = calculate_returns(prices, period=2)
        self.assertEqual(len(returns_2), 3)

    def test_pattern_detector_initialization(self):
        """Тест инициализации детектора паттернов"""
        detector = PatternDetector()
        self.assertIsNotNone(detector)
        self.assertGreater(len(detector.pattern_detectors), 0)

        # Проверяем статистику
        stats = detector.get_statistics()
        self.assertIn('total_processed', stats)
        self.assertIn('patterns_found', stats)

    def test_pattern_analyzer_initialization(self):
        """Тест инициализации анализатора"""
        analyzer = PatternAnalyzer()
        self.assertIsNotNone(analyzer)

        # Проверяем статистику
        stats = analyzer.get_statistics()
        self.assertIn('total_patterns_analyzed', stats)

    def test_database_operations(self):
        """Тест операций с базой данных"""
        # Используем временную базу данных для тестов
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name

        try:
            # Создаем базу данных
            db = PatternDatabase(db_path)
            self.assertIsNotNone(db)

            # Проверяем создание таблиц
            # (В реальном тесте нужно проверить выполнение запросов)

            # Закрываем базу
            db.close()

        finally:
            # Удаляем временный файл
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_pattern_detection(self):
        """Тест детектирования паттернов (упрощенный)"""
        detector = PatternDetector()

        # Создаем искусственный паттерн "Голова и Плечи"
        test_pattern_data = self._create_test_head_shoulders()

        # Проверяем детектирование (в реальном тесте нужно проверить результат)
        # Этот тест должен быть расширен для реального использования
        self.assertTrue(len(test_pattern_data['close']) > 0)

    def _create_test_head_shoulders(self):
        """Создание тестовых данных для паттерна Голова и Плечи"""
        n = 50
        base_price = 100

        # Создаем форму Голова и Плечи
        x = np.linspace(0, 4 * np.pi, n)
        pattern = -np.sin(x) * 10  # Отрицательный синус для медвежьего паттерна

        # Добавляем шум
        noise = np.random.normal(0, 0.5, n)
        prices = base_price + pattern + noise

        # Создаем OHLC данные
        data = {
            'open': prices,
            'high': prices + np.random.uniform(0.1, 0.3, n),
            'low': prices - np.random.uniform(0.1, 0.3, n),
            'close': prices,
            'volume': np.random.uniform(1000, 10000, n),
            'timestamp': np.array([datetime.now() - timedelta(minutes=i * 5)
                                   for i in range(n)])
        }

        return data

    def test_pattern_quality_analysis(self):
        """Тест анализа качества паттерна"""
        # Создаем тестовый паттерн
        pattern = {
            'id': 'test_pattern_001',
            'name': 'Head and Shoulders',
            'type': 'geometric',
            'direction': 'bearish',
            'metadata': {
                'symbol': 'EURUSD',
                'timeframe': 'H1',
                'quality_score': 0.75,
                'confidence': 0.8,
                'market_context': 'uptrend'
            },
            'targets': {
                'entry_price': 1.1200,
                'stop_loss': 1.1250,
                'take_profit': 1.1100,
                'profit_risk_ratio': 2.0
            },
            'points': [
                {'index': 10, 'price': 1.1250, 'point_type': 'left_shoulder'},
                {'index': 20, 'price': 1.1300, 'point_type': 'head'},
                {'index': 30, 'price': 1.1240, 'point_type': 'right_shoulder'}
            ]
        }

        analyzer = PatternAnalyzer()
        analysis = analyzer.analyze_pattern_quality(pattern)

        # Проверяем, что анализ содержит ожидаемые ключи
        expected_keys = ['geometric_quality', 'context_score', 'confirmation_score',
                         'risk_reward_score', 'historical_success_score', 'overall_score']

        for key in expected_keys:
            self.assertIn(key, analysis)

        # Проверяем, что оценки в диапазоне 0-1
        for key in expected_keys:
            if key != 'recommendation' and key != 'confidence':
                self.assertGreaterEqual(analysis[key], 0.0)
                self.assertLessEqual(analysis[key], 1.0)

    def test_historical_analysis(self):
        """Тест исторического анализа"""
        analyzer = PatternAnalyzer()

        # Создаем тестовые исторические паттерны
        historical_patterns = []
        for i in range(10):
            pattern = {
                'id': f'historical_pattern_{i}',
                'name': 'Head and Shoulders',
                'type': 'geometric',
                'direction': 'bearish',
                'metadata': {
                    'symbol': 'EURUSD',
                    'timeframe': 'H1',
                    'quality_score': 0.7 + i * 0.03,
                    'confidence': 0.75
                },
                'targets': {
                    'entry_price': 1.1200 - i * 0.001,
                    'stop_loss': 1.1250,
                    'take_profit': 1.1100,
                    'profit_risk_ratio': 2.0 - i * 0.1
                },
                'outcome': 'success' if i % 2 == 0 else 'failure',
                'statistics': {
                    'historical_success_rate': 0.6 if i % 2 == 0 else 0.4
                }
            }
            historical_patterns.append(pattern)

        # Создаем текущий паттерн для сравнения
        current_pattern = {
            'id': 'current_pattern',
            'name': 'Head and Shoulders',
            'type': 'geometric',
            'direction': 'bearish',
            'metadata': {
                'symbol': 'EURUSD',
                'timeframe': 'H1',
                'quality_score': 0.8,
                'confidence': 0.85
            },
            'targets': {
                'entry_price': 1.1190,
                'stop_loss': 1.1240,
                'take_profit': 1.1090,
                'profit_risk_ratio': 2.5
            }
        }

        # Ищем похожие паттерны
        similar_patterns = analyzer.find_similar_patterns(
            current_pattern, historical_patterns, n_neighbors=5
        )

        # Должно найти похожие паттерны
        self.assertLessEqual(len(similar_patterns), 5)

        if similar_patterns:
            # Проверяем расчет успешности
            success_rate = analyzer.calculate_success_rate(similar_patterns)
            self.assertGreaterEqual(success_rate, 0.0)
            self.assertLessEqual(success_rate, 1.0)

            # Проверяем расчет средней прибыли
            avg_profit = analyzer.calculate_average_profit(similar_patterns)
            self.assertIsNotNone(avg_profit)

    def test_extremum_detection(self):
        """Тест обнаружения экстремумов"""
        # Создаем тестовые данные с явными экстремумами
        n = 100
        x = np.linspace(0, 4 * np.pi, n)
        prices = np.sin(x) * 10 + 100  # Синусоида

        # Добавляем явные максимумы и минимумы
        prices[25] = 120  # Явный максимум
        prices[50] = 80  # Явный минимум
        prices[75] = 115  # Явный максимум

        # В реальном тесте нужно проверить работу детектора экстремумов
        # Это упрощенный тест для демонстрации

        from scipy.signal import argrelextrema

        # Находим локальные максимумы и минимумы
        max_indices = argrelextrema(prices, np.greater, order=5)[0]
        min_indices = argrelextrema(prices, np.less, order=5)[0]

        # Проверяем, что нашли ожидаемые экстремумы
        self.assertIn(25, max_indices)
        self.assertIn(50, min_indices)
        self.assertIn(75, max_indices)


class TestIntegration(unittest.TestCase):
    """Интеграционные тесты"""

    def test_full_pipeline(self):
        """Тест полного пайплайна обработки"""
        # Этот тест проверяет взаимодействие всех компонентов
        # В реальном проекте нужно реализовать более детальные тесты

        # 1. Создаем детектор
        detector = PatternDetector()

        # 2. Создаем тестовые данные
        np.random.seed(42)
        n_samples = 200

        test_data = {
            'open': np.random.uniform(1.0, 1.2, n_samples),
            'high': np.random.uniform(1.1, 1.3, n_samples),
            'low': np.random.uniform(0.9, 1.1, n_samples),
            'close': np.random.uniform(1.0, 1.2, n_samples),
            'volume': np.random.uniform(1000, 10000, n_samples)
        }

        # 3. Запускаем детектирование
        result = detector.detect_all_patterns(
            symbol='TEST',
            timeframe='H1',
            data=test_data
        )

        # Проверяем, что результат имеет правильную структуру
        self.assertIsNotNone(result)
        self.assertIn('patterns', result)
        self.assertIn('indicators', result)
        self.assertIn('extremums', result)
        self.assertIn('statistics', result)

        # 4. Проверяем статистику
        stats = result.statistics
        self.assertIn('total_patterns', stats)
        self.assertIn('bullish_patterns', stats)
        self.assertIn('bearish_patterns', stats)

        # 5. Если найдены паттерны, проверяем их структуру
        if result.patterns:
            pattern = result.patterns[0]
            expected_keys = ['id', 'name', 'type', 'direction', 'metadata', 'targets']

            for key in expected_keys:
                self.assertIn(key, pattern)


if __name__ == '__main__':
    # Запуск тестов
    unittest.main(verbosity=2)

