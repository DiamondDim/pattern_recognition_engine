# Pattern Recognition Engine

Движок для распознавания торговых паттернов с использованием машинного обучения и технического анализа.

## 🚀 Основные возможности

- **Распознавание паттернов**: свечные, геометрические, гармонические паттерны
- **Технический анализ**: 20+ технических индикаторов (RSI, MACD, Bollinger Bands и др.)
- **Машинное обучение**: классификация и кластеризация паттернов
- **Бэктестинг**: полная система тестирования торговых стратегий
- **Визуализация**: интерактивные графики с Plotly и Matplotlib
- **База данных**: хранение и поиск исторических паттернов
- **Интеграция**: Yahoo Finance, MetaTrader 5, CSV файлы

## 📋 Системные требования

### Обязательные
- Python 3.8+
- 4GB RAM минимум
- 2GB свободного места на диске

### Опциональные
- MetaTrader 5 (для работы с реальными данными)
- GPU с поддержкой CUDA (для ускорения ML)

## ⚡ Быстрый старт

### 1. Клонирование репозитория

```bash
git clone https://github.com/DiamondDim/pattern_recognition_engine.git
cd pattern_recognition_engine
```

### 2. Установка зависимостей

#### Способ 1: Использование pip (рекомендуется)
```bash
pip install -r requirements.txt
```

#### Способ 2: Использование Poetry
```bash
poetry install
```

### 3. Настройка окружения

Создайте файл `.env` в корне проекта:

```env
# Общие настройки
LOG_LEVEL=INFO
DATA_DIR=./data
RESULTS_DIR=./results

# Настройки MetaTrader 5 (опционально)
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_server
MT5_PATH=C:/Program Files/MetaTrader 5/terminal64.exe

# Настройки Yahoo Finance (для получения данных)
YFINANCE_TIMEOUT=10
YFINANCE_MAX_RETRIES=3
```

### 4. Первоначальная настройка базы данных

```bash
python scripts/init_database.py
```

### 5. Запуск тестовой системы

```bash
python run.py --mode test --symbol EURUSD --timeframe H1 --bars 1000
```

## 🛠 Конфигурация

### Основные настройки (config.py)

```python
# Основные параметры системы
DEBUG = False
LOG_LEVEL = "INFO"
DATA_DIR = Path("./data")
RESULTS_DIR = Path("./results")
MODELS_DIR = Path("./models")

# Параметры детекции паттернов
PATTERN_MIN_CONFIDENCE = 0.6
PATTERN_MIN_BARS = 5
MAX_PATTERNS_PER_SYMBOL = 100

# Параметры машинного обучения
ML_BATCH_SIZE = 32
ML_EPOCHS = 100
ML_VALIDATION_SPLIT = 0.2
```

### Настройки торговли

```python
# Параметры риск-менеджмента
RISK_PER_TRADE = 0.01  # 1% от депозита на сделку
MAX_POSITION_SIZE = 1.0  # Максимальный размер позиции в лотах
LEVERAGE = 100  # Кредитное плечо
STOP_LOSS_PERCENT = 2.0  # Стоп-лосс в процентах
TAKE_PROFIT_PERCENT = 4.0  # Тейк-профит в процентах
```

## 🚀 Запуск в различных режимах

### 1. Режим анализа (рекомендуется для начала)

```bash
python main.py --mode analyze --symbol AAPL --timeframe 1d --period 1y
```

Параметры:
- `--mode`: режим работы (analyze, backtest, trade, monitor)
- `--symbol`: торговый символ (AAPL, EURUSD, BTCUSDT)
- `--timeframe`: таймфрейм (1m, 5m, 15m, 1h, 4h, 1d)
- `--period`: период данных (1d, 1w, 1m, 1y)
- `--output`: путь для сохранения результатов

### 2. Режим бэктестинга

```bash
python main.py --mode backtest \
    --symbol EURUSD \
    --timeframe H1 \
    --start 2023-01-01 \
    --end 2023-12-31 \
    --strategy pattern_based \
    --initial_balance 10000
```

### 3. Режим реальной торговли (требует настройки MT5)

```bash
python main.py --mode trade \
    --symbol EURUSD \
    --timeframe M15 \
    --risk 0.02 \
    --max_trades 3
```

### 4. Мониторинг в реальном времени

```bash
python main.py --mode monitor \
    --symbols EURUSD,GBPUSD,USDJPY \
    --timeframe M5 \
    --interval 60 \
    --notifications
```

## 📊 Примеры использования

### Пример 1: Анализ паттернов на исторических данных

```python
from core.pattern_detector import PatternDetector
from core.data_feeder import DataFeeder
from utils.visualization import PatternVisualizer

# Загрузка данных
feeder = DataFeeder()
data = feeder.load_ohlc_data(
    source='yfinance',
    symbol='AAPL',
    timeframe='1d',
    bars=1000
)

# Расчет индикаторов
indicators = feeder.calculate_indicators(data)

# Детекция паттернов
detector = PatternDetector()
patterns = detector.detect_all_patterns(data, indicators)

# Визуализация
visualizer = PatternVisualizer()
for pattern in patterns[:5]:  # Показать первые 5 паттернов
    fig = visualizer.plot_pattern(
        pattern.pattern_data,
        data,
        indicators,
        save_path=f'./results/pattern_{pattern.pattern_name}.png'
    )
```

### Пример 2: Бэктестинг стратегии

```python
from core.backtesting import BacktestEngine
from core.pattern_analyzer import PatternAnalyzer

# Инициализация движка бэктестинга
engine = BacktestEngine()

# Загрузка и анализ паттернов
analyzer = PatternAnalyzer()
analysis = analyzer.analyze_pattern(pattern, data)

# Размещение ордеров на основе анализа
if analysis['recommendations']['primary_action'] == 'strong_buy':
    order = Order(
        id=generate_id(),
        symbol='AAPL',
        order_type=OrderType.BUY,
        volume=engine.calculate_position_size(
            entry_price=analysis['trading_signals']['entry_price'],
            stop_loss=analysis['trading_signals']['stop_loss'],
            risk_percent=1.0
        ),
        entry_price=analysis['trading_signals']['entry_price'],
        stop_loss=analysis['trading_signals']['stop_loss'],
        take_profit=analysis['trading_signals']['take_profit']
    )
    engine.place_order(order)

# Генерация отчета
report = engine.generate_report()
print(f"Итоговая прибыль: ${report['summary']['net_profit']:.2f}")
```

### Пример 3: Обучение модели машинного обучения

```python
from core.ml_models import PatternClassifier
from core.pattern_database import PatternDatabase

# Загрузка исторических паттернов из базы данных
db = PatternDatabase()
historical_patterns = db.find_patterns(
    symbol='EURUSD',
    timeframe='H1',
    min_confidence=0.6,
    limit=1000
)

# Подготовка данных для обучения
classifier = PatternClassifier(model_type='xgboost')
X, y = classifier.prepare_features(historical_patterns)

# Обучение модели
metrics = classifier.train(X, y, validation_split=0.2)
print(f"Точность модели: {metrics['val_accuracy']:.2%}")

# Сохранение модели
classifier.save_model('./models/pattern_classifier_v1.joblib')
```

## 📁 Структура проекта

```
pattern_recognition_engine/
├── core/                    # Основные модули
│   ├── backtesting.py      # Движок бэктестинга
│   ├── data_feeder.py      # Загрузка и обработка данных
│   ├── ml_models.py        # Модели машинного обучения
│   ├── pattern_analyzer.py # Анализ паттернов
│   ├── pattern_database.py # База данных паттернов
│   ├── pattern_detector.py # Детектор паттернов
│   └── statistics.py       # Статистические расчеты
├── patterns/               # Определения паттернов
│   ├── base_pattern.py     # Базовый класс паттерна
│   ├── candlestick_patterns.py  # Свечные паттерны
│   ├── geometric_patterns.py    # Геометрические паттерны
│   └── harmonic_patterns.py     # Гармонические паттерны
├── utils/                  # Вспомогательные модули
│   ├── helpers.py          # Вспомогательные функции
│   ├── logger.py           # Логирование
│   ├── mt5_connector.py    # Подключение к MT5
│   └── visualization.py    # Визуализация
├── config.py              # Конфигурация приложения
├── main.py               # Основной скрипт запуска
├── run.py               # Скрипт для быстрого запуска
├── requirements.txt     # Зависимости Python
├── pyproject.toml      # Конфигурация Poetry
└── README.md           # Документация
```

## 🔧 Расширенные настройки

### Настройка MetaTrader 5

1. Установите MetaTrader 5
2. Создайте демо-счет или используйте реальный
3. Обновите настройки в `.env`:

```env
MT5_PATH=C:/Program Files/MetaTrader 5/terminal64.exe
MT5_LOGIN=1234567
MT5_PASSWORD=your_password
MT5_SERVER=MetaQuotes-Demo
```

4. Проверьте подключение:
```bash
python utils/mt5_connector.py --test
```

### Настройка базы данных

Проект использует SQLite по умолчанию. Для использования PostgreSQL:

1. Установите psycopg2:
```bash
pip install psycopg2-binary
```

2. Обновите конфигурацию:
```python
DATABASE_CONFIG = {
    'ENGINE': 'postgresql',
    'NAME': 'pattern_db',
    'USER': 'postgres',
    'PASSWORD': 'your_password',
    'HOST': 'localhost',
    'PORT': 5432
}
```

### Оптимизация производительности

Для больших объемов данных:

1. Включите кэширование в `config.py`:
```python
CACHE_ENABLED = True
CACHE_MAX_SIZE = 10000
CACHE_TTL = 3600  # 1 час
```

2. Используйте многопоточность:
```python
MAX_WORKERS = 4
PARALLEL_PROCESSING = True
```

3. Оптимизируйте параметры ML:
```python
ML_USE_GPU = True
ML_BATCH_SIZE = 64
ML_OPTIMIZER = 'adam'
```

## 🧪 Тестирование

### Запуск тестов

```bash
# Все тесты
pytest tests/ -v

# Тесты паттернов
pytest tests/test_patterns.py -v

# Тесты с покрытием
pytest --cov=core --cov-report=html tests/
```

### Интеграционные тесты

```bash
# Тест загрузки данных
python tests/test_data_loading.py

# Тест бэктестинга
python tests/test_backtesting.py --symbol EURUSD --bars 500

# Тест визуализации
python tests/test_visualization.py --save
```

## 📈 Примеры выходных данных

### Отчет бэктестинга

```json
{
  "summary": {
    "initial_balance": 10000.0,
    "final_balance": 11250.75,
    "net_profit": 1250.75,
    "total_return_pct": 12.51,
    "total_trades": 45,
    "winning_trades": 28,
    "losing_trades": 17,
    "win_rate_pct": 62.22,
    "profit_factor": 1.85,
    "max_drawdown_pct": 8.42,
    "sharpe_ratio": 1.24
  }
}
```

### Статистика паттернов

```
Тип паттерна      Количество  Успешность  Сред. доходность
---------------  -----------  ----------  ----------------
Head & Shoulders           12        58.3%             2.1%
Double Top                  8        62.5%             1.8%
Triangle                   15        66.7%             2.4%
Bull Flag                   9        77.8%             3.2%
```

## 🔍 Отладка и логирование

### Уровни логирования

```bash
# DEBUG - подробная отладка
python main.py --log-level DEBUG

# INFO - обычная информация
python main.py --log-level INFO

# WARNING - только предупреждения
python main.py --log-level WARNING
```

### Просмотр логов

```bash
# В реальном времени
tail -f logs/pattern_engine.log

# Поиск ошибок
grep "ERROR" logs/pattern_engine.log

# Анализ производительности
grep "Execution time" logs/pattern_engine.log
```

## 🚀 Производительность

### Бенчмарки

| Операция | Время (1000 баров) | Время (10000 баров) |
|----------|-------------------|---------------------|
| Загрузка данных | 0.5 сек | 3.2 сек |
| Расчет индикаторов | 1.2 сек | 8.7 сек |
| Детекция паттернов | 2.1 сек | 18.5 сек |
| Обучение ML модели | 15 сек | 120 сек |

### Оптимизация

1. Используйте кэширование:
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def calculate_indicators(data):
    # Кэшированные расчеты
```

2. Векторизация NumPy:
```python
# Вместо циклов используйте векторизацию
returns = np.diff(prices) / prices[:-1]
```

3. Асинхронные операции:
```python
async def process_multiple_symbols(symbols):
    tasks = [load_data(symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks)
```

## 📚 API документация

### Основные классы

#### PatternDetector
```python
detector = PatternDetector()
patterns = detector.detect_all_patterns(data, indicators)
```

#### BacktestEngine
```python
engine = BacktestEngine(initial_balance=10000)
engine.place_order(order)
report = engine.generate_report()
```

#### PatternClassifier
```python
classifier = PatternClassifier(model_type='random_forest')
classifier.train(X, y)
predictions = classifier.predict(new_data)
```

## 🤝 Вклад в проект

### Установка для разработки

```bash
git clone https://github.com/DiamondDim/pattern_recognition_engine.git
cd pattern_recognition_engine
pip install -r requirements-dev.txt
pre-commit install
```

### Правила коммитов

```bash
# Типы коммитов:
# feat: Новая функциональность
# fix: Исправление ошибок
# docs: Изменение документации
# style: Форматирование кода
# refactor: Рефакторинг кода
# test: Добавление тестов
# chore: Обновление зависимостей

git commit -m "feat: add new harmonic pattern detection"
```

### Тестирование изменений

```bash
# Перед коммитом
pytest tests/
flake8 core/ utils/ patterns/
black --check .
```

## 🆘 Поиск и устранение неисправностей

### Распространенные проблемы

1. **Ошибка подключения к MT5**
   - Проверьте путь к терминалу в `.env`
   - Убедитесь, что MT5 запущен
   - Проверьте логин и пароль

2. **Проблемы с зависимостями**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt --force-reinstall
   ```

3. **Недостаточно памяти**
   - Уменьшите размер батча в `config.py`
   - Включите `USE_INCREMENTAL_LEARNING = True`

### Получение помощи

1. Проверьте логи: `logs/pattern_engine.log`
2. Создайте issue на GitHub с деталями ошибки
3. Используйте отладочный режим:
   ```bash
   python main.py --debug --log-level DEBUG
   ```

## 📄 Лицензия

MIT License - подробности в файле LICENSE

## 📞 Контакты

- Автор: DiamondDim
- GitHub: [https://github.com/DiamondDim](https://github.com/DiamondDim)
- Issues: [GitHub Issues](https://github.com/DiamondDim/pattern_recognition_engine/issues)

---

## 🔄 Обновление проекта

Для обновления до последней версии:

```bash
git pull origin main
pip install -r requirements.txt --upgrade
python scripts/update_database.py
```

---

**Примечание**: Этот проект предназначен для образовательных и исследовательских целей. Используйте торговые стратегии на свой страх и риск. Автор не несет ответственности за финансовые потери.