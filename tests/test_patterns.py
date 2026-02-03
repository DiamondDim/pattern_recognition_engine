"""
Тесты для модуля паттернов
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Добавляем путь к корневой директории проекта
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from patterns.candlestick_patterns import CandlestickPatterns
from patterns.geometric_patterns import GeometricPatterns
from patterns.harmonic_patterns import HarmonicPatterns

class TestCandlestickPatterns(unittest.TestCase):
    """Тесты для свечных паттернов"""

    def setUp(self):
        """Подготовка тестовых данных"""
        self.patterns = CandlestickPatterns()

        # Создаем тестовые данные
        n = 100
        self.test_data = {
            'open': np.random.randn(n).cumsum() + 100,
            'high': np.random.randn(n).cumsum() + 101,
            'low': np.random.randn(n).cumsum() + 99,
            'close': np.random.randn(n).cumsum() + 100,
            'volume': np.random.rand(n) * 1000
        }

        # Создаем искусственные паттерны для тестирования
        # Дожи на индексе 50
        self.test_data['open'][50] = 100.0
        self.test_data['close'][50] = 100.0
        self.test_data['high'][50] = 101.0
        self.test_data['low'][50] = 99.0

    def test_detection(self):
        """Тест детектирования паттернов"""
        results = self.patterns.detect(self.test_data)

        # Проверяем что функция возвращает список
        self.assertIsInstance(results, list)

        # Проверяем структуру результатов
        if results:
            result = results[0]
            self.assertIn('name', result)
            self.assertIn('direction', result)
            self.assertIn('points', result)
            self.assertIn('quality', result)
            self.assertIn('confidence', result)

    def test_doji_detection(self):
        """Тест детектирования Дожи"""
        # Создаем данные с явным Дожи
        test_data = {
            'open': np.array([100.0, 100.0]),
            'high': np.array([101.0, 101.0]),
            'low': np.array([99.0, 99.0]),
            'close': np.array([102.0, 100.0]),  # Второй бар - Дожи
            'volume': np.array([1000, 1000])
        }

        results = self.patterns.detect(test_data)

        # Должен найти хотя бы один паттерн
        self.assertGreater(len(results), 0)

        # Проверяем что найден Дожи
        doji_found = any(r['name'] == 'doji' for r in results)
        self.assertTrue(doji_found, "Дожи должен быть обнаружен")

class TestGeometricPatterns(unittest.TestCase):
    """Тесты для геометрических паттернов"""

    def setUp(self):
        """Подготовка тестовых данных"""
        self.patterns = GeometricPatterns()

        # Создаем тестовые данные с паттерном "Голова и Плечи"
        n = 100
        self.test_data = {
            'high': np.ones(n) * 100,
            'low': np.ones(n) * 90
        }

        # Создаем паттерн "Голова и Плечи"
        # Левое плечо
        self.test_data['high'][10] = 105
        self.test_data['low'][12] = 95

        # Голова
        self.test_data['high'][20] = 110
        self.test_data['low'][22] = 100

        # Правое плечо
        self.test_data['high'][30] = 105
        self.test_data['low'][32] = 95

    def test_detection(self):
        """Тест детектирования геометрических паттернов"""
        results = self.patterns.detect(self.test_data)

        # Проверяем что функция возвращает список
        self.assertIsInstance(results, list)

        # Проверяем структуру результатов
        if results:
            result = results[0]
            self.assertIn('name', result)
            self.assertIn('direction', result)
            self.assertIn('points', result)
            self.assertIn('quality', result)
            self.assertIn('confidence', result)

    def test_head_shoulders_detection(self):
        """Тест детектирования паттерна Голова и Плечи"""
        # Создаем более четкий паттерн
        test_data = {
            'high': np.array([100, 105, 110, 105, 100]),
            'low': np.array([95, 100, 95, 100, 95])
        }

        results = self.patterns.detect(test_data)

        # Должен найти хотя бы один паттерн
        self.assertGreater(len(results), 0)

        # Проверяем что найден Head and Shoulders
        hs_found = any('head_shoulders' in r['name'] for r in results)
        self.assertTrue(hs_found, "Head and Shoulders должен быть обнаружен")

class TestHarmonicPatterns(unittest.TestCase):
    """Тесты для гармонических паттернов"""

    def setUp(self):
        """Подготовка тестовых данных"""
        self.patterns = HarmonicPatterns()

        # Создаем тестовые данные с гармоническим паттерном
        n = 50
        self.test_data = {
            'high': np.ones(n) * 100,
            'low': np.ones(n) * 90
        }

        # Создаем точки для паттерна Gartley
        # X, A, B, C, D точки
        self.test_data['high'][5] = 110   # A (высокая)
        self.test_data['low'][10] = 95    # B (низкая)
        self.test_data['high'][15] = 105  # C (высокая)
        self.test_data['low'][20] = 97    # D (низкая)

    def test_detection(self):
        """Тест детектирования гармонических паттернов"""
        results = self.patterns.detect(self.test_data)

        # Проверяем что функция возвращает список
        self.assertIsInstance(results, list)

        # Проверяем структуру результатов
        if results:
            result = results[0]
            self.assertIn('name', result)
            self.assertIn('direction', result)
            self.assertIn('points', result)
            self.assertIn('quality', result)
            self.assertIn('confidence', result)

    def test_gartley_detection(self):
        """Тест детектирования паттерна Gartley"""
        # Создаем более четкий паттерн Gartley
        test_data = {
            'high': np.array([100, 110, 105, 108, 103]),
            'low': np.array([90, 95, 93, 96, 91])
        }

        results = self.patterns.detect(test_data)

        # Должен найти хотя бы один паттерн
        self.assertGreater(len(results), 0)

class TestPatternIntegration(unittest.TestCase):
    """Интеграционные тесты паттернов"""

    def test_all_patterns_together(self):
        """Тест работы всех типов паттернов вместе"""
        from core.pattern_detector import PatternDetector

        detector = PatternDetector()

        # Создаем тестовые данные
        n = 200
        test_data = {
            'open': np.random.randn(n).cumsum() + 100,
            'high': np.random.randn(n).cumsum() + 102,
            'low': np.random.randn(n).cumsum() + 98,
            'close': np.random.randn(n).cumsum() + 100,
            'volume': np.random.rand(n) * 1000
        }

        # Добавляем несколько паттернов
        # Дожи
        test_data['open'][50] = 100.0
        test_data['close'][50] = 100.0

        # Head and Shoulders
        test_data['high'][80] = 110
        test_data['high'][90] = 115
        test_data['high'][100] = 110

        # Проверяем что детектор не падает
        try:
            result = detector.detect_all_patterns(
                symbol="TEST",
                timeframe="H1",
                data=test_data
            )

            self.assertIsNotNone(result)
            self.assertIn('patterns', result)
            self.assertIsInstance(result.patterns, list)

        except Exception as e:
            self.fail(f"Детектор паттернов упал с ошибкой: {e}")

if __name__ == '__main__':
    unittest.main()

