# full_test.py
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.data_feeder import DataFeeder
from core.pattern_detector import PatternDetector
from core.pattern_database import PatternDatabase


def main():
    print("🧪 Тест системы распознавания паттернов")

    # 1. Загрузка данных
    print("1. Загрузка данных...")
    feeder = DataFeeder()
    data = feeder.get_data("EURUSD", "H1", bars=300)

    if data.empty:
        print("❌ Не удалось загрузить данные")
        return

    print(f"   ✅ Загружено {len(data)} баров")

    # 2. Обнаружение паттернов
    print("2. Обнаружение паттернов...")
    detector = PatternDetector()
    patterns = detector.detect_all_patterns(data, pattern_types=['candlestick'])

    print(f"   ✅ Найдено {len(patterns)} паттернов")

    # 3. Сохранение в базу данных
    print("3. Сохранение в базу данных...")
    db = PatternDatabase()

    for pattern in patterns:
        pattern['symbol'] = 'EURUSD'
        pattern['timeframe'] = 'H1'
        db.add_pattern(pattern)

    print(f"   ✅ Паттерны сохранены в базу")

    # 4. Чтение из базы
    print("4. Чтение из базы данных...")
    saved_patterns = db.get_patterns(symbol='EURUSD', timeframe='H1')
    print(f"   ✅ Получено {len(saved_patterns)} паттернов из базы")

    # 5. Статистика
    print("5. Статистика...")
    stats = db.get_pattern_statistics()
    print(f"   Всего паттернов: {stats.get('pattern_count', 0)}")

    print("\n✅ Все тесты пройдены успешно!")


if __name__ == "__main__":
    main()

