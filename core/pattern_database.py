import sqlite3
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import logging
import hashlib
import os

logger = logging.getLogger(__name__)


class PatternDatabase:
    """
    Класс для работы с базой данных паттернов
    """

    def __init__(self, db_path: str = 'patterns.db'):
        """
        Инициализация базы данных паттернов

        Args:
            db_path (str): Путь к файлу базы данных
        """
        self.db_path = db_path
        self.conn = None
        self._initialize_database()

    def _initialize_database(self):
        """Инициализация структуры базы данных"""
        try:
            # Создаем директорию для БД, если нужно
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)

            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # Для доступа к колонкам по имени

            cursor = self.conn.cursor()

            # Таблица паттернов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_hash TEXT UNIQUE,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    pattern_family TEXT NOT NULL,
                    detection_time TIMESTAMP NOT NULL,
                    data_start TIMESTAMP NOT NULL,
                    data_end TIMESTAMP NOT NULL,
                    confidence REAL,
                    metrics TEXT,
                    raw_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Таблица результатов паттернов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pattern_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_id INTEGER NOT NULL,
                    outcome TEXT,
                    pnl REAL,
                    pnl_percent REAL,
                    duration_hours REAL,
                    exit_reason TEXT,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (pattern_id) REFERENCES patterns (id) ON DELETE CASCADE
                )
            ''')

            # Таблица статистики
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pattern_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    timeframe TEXT,
                    pattern_type TEXT,
                    period TEXT,
                    total_count INTEGER,
                    success_count INTEGER,
                    avg_confidence REAL,
                    avg_pnl REAL,
                    avg_pnl_percent REAL,
                    win_rate REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, pattern_type, period)
                )
            ''')

            # Индексы для ускорения запросов
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_symbol_timeframe ON patterns (symbol, timeframe)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_detection_time ON patterns (detection_time)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_pattern_type ON patterns (pattern_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_confidence ON patterns (confidence)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_pattern_id ON pattern_results (pattern_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_stats_symbol_timeframe ON pattern_stats (symbol, timeframe)')

            self.conn.commit()
            logger.info(f"База данных инициализирована: {self.db_path}")

        except Exception as e:
            logger.error(f"Ошибка инициализации базы данных: {e}")
            raise

    def _generate_pattern_hash(self, pattern_data: Dict[str, Any]) -> str:
        """
        Генерация хэша для паттерна

        Args:
            pattern_data (dict): Данные паттерна

        Returns:
            str: Хэш паттерна
        """
        # Создаем строку для хэширования
        hash_string = f"{pattern_data.get('symbol', '')}_{pattern_data.get('timeframe', '')}_"
        hash_string += f"{pattern_data.get('pattern_type', '')}_{pattern_data.get('detection_time', '')}_"

        # Добавляем основные параметры
        if 'prices' in pattern_data:
            prices = pattern_data['prices']
            hash_string += f"{prices.get('X', 0)}_{prices.get('A', 0)}_{prices.get('B', 0)}_{prices.get('C', 0)}_{prices.get('D', 0)}"
        elif 'index' in pattern_data:
            hash_string += f"{pattern_data.get('index', 0)}"

        # Генерируем MD5 хэш
        return hashlib.md5(hash_string.encode()).hexdigest()

    def add_pattern(self, pattern_data: Dict[str, Any]) -> Optional[int]:
        """
        Добавление паттерна в базу данных

        Args:
            pattern_data (dict): Данные паттерна

        Returns:
            int: ID добавленного паттерна или None при ошибке
        """
        try:
            # Генерируем хэш паттерна
            pattern_hash = self._generate_pattern_hash(pattern_data)

            # Проверяем, существует ли уже такой паттерн
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT id FROM patterns WHERE pattern_hash = ?",
                (pattern_hash,)
            )
            existing = cursor.fetchone()

            if existing:
                logger.debug(f"Паттерн уже существует: {pattern_hash}")
                return existing['id']

            # Подготавливаем данные для вставки
            symbol = pattern_data.get('symbol', 'UNKNOWN')
            timeframe = pattern_data.get('timeframe', 'UNKNOWN')
            pattern_type = pattern_data.get('pattern_type', 'unknown')
            pattern_family = pattern_data.get('pattern_family', 'unknown')
            detection_time = pattern_data.get('detection_time', datetime.now())
            confidence = pattern_data.get('confidence', 0.0)

            # Определяем временной диапазон данных
            if 'points_indices' in pattern_data and self.data is not None:
                indices = pattern_data['points_indices']
                if indices:
                    data_start = self.data.index[min(indices)] if min(indices) < len(self.data) else detection_time
                    data_end = self.data.index[max(indices)] if max(indices) < len(self.data) else detection_time
                else:
                    data_start = data_end = detection_time
            else:
                data_start = data_end = detection_time

            # Сериализуем метрики и сырые данные
            metrics = pattern_data.get('metrics', {})
            raw_data = {k: v for k, v in pattern_data.items() if k not in ['symbol', 'timeframe', 'detection_time']}

            metrics_json = json.dumps(metrics, default=str)
            raw_data_json = json.dumps(raw_data, default=str)

            # Вставляем паттерн
            cursor.execute('''
                INSERT INTO patterns 
                (pattern_hash, symbol, timeframe, pattern_type, pattern_family,
                 detection_time, data_start, data_end, confidence,
                 metrics, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pattern_hash,
                symbol,
                timeframe,
                pattern_type,
                pattern_family,
                detection_time.isoformat() if isinstance(detection_time, datetime) else detection_time,
                data_start.isoformat() if isinstance(data_start, datetime) else data_start,
                data_end.isoformat() if isinstance(data_end, datetime) else data_end,
                confidence,
                metrics_json,
                raw_data_json
            ))

            pattern_id = cursor.lastrowid
            self.conn.commit()

            logger.debug(f"Паттерн добавлен с ID: {pattern_id}")

            # Обновляем статистику
            self._update_statistics(symbol, timeframe, pattern_type)

            return pattern_id

        except Exception as e:
            logger.error(f"Ошибка добавления паттерна: {e}")
            self.conn.rollback()
            return None

    def get_patterns(self, symbol: Optional[str] = None,
                     timeframe: Optional[str] = None,
                     pattern_type: Optional[str] = None,
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None,
                     min_confidence: float = 0.0,
                     limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Получение паттернов из базы данных

        Args:
            symbol (str): Фильтр по символу
            timeframe (str): Фильтр по таймфрейму
            pattern_type (str): Фильтр по типу паттерна
            start_date (datetime): Начальная дата
            end_date (datetime): Конечная дата
            min_confidence (float): Минимальная уверенность
            limit (int): Ограничение количества результатов

        Returns:
            list: Список паттернов
        """
        try:
            cursor = self.conn.cursor()

            # Строим запрос
            query = "SELECT * FROM patterns WHERE 1=1"
            params = []

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)

            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)

            if pattern_type:
                query += " AND pattern_type = ?"
                params.append(pattern_type)

            if start_date:
                query += " AND detection_time >= ?"
                params.append(start_date.isoformat())

            if end_date:
                query += " AND detection_time <= ?"
                params.append(end_date.isoformat())

            query += " AND confidence >= ?"
            params.append(min_confidence)

            query += " ORDER BY detection_time DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Преобразуем результаты
            patterns = []
            for row in rows:
                pattern = dict(row)

                # Десериализуем JSON данные
                if pattern['metrics']:
                    pattern['metrics'] = json.loads(pattern['metrics'])
                if pattern['raw_data']:
                    pattern['raw_data'] = json.loads(pattern['raw_data'])

                # Преобразуем строки в datetime
                for date_field in ['detection_time', 'data_start', 'data_end', 'created_at', 'updated_at']:
                    if pattern.get(date_field):
                        try:
                            pattern[date_field] = datetime.fromisoformat(pattern[date_field].replace('Z', '+00:00'))
                        except:
                            pass

                patterns.append(pattern)

            logger.debug(f"Получено {len(patterns)} паттернов из БД")
            return patterns

        except Exception as e:
            logger.error(f"Ошибка получения паттернов: {e}")
            return []

    def add_pattern_result(self, pattern_id: int,
                           result_data: Dict[str, Any]) -> bool:
        """
        Добавление результата торговли по паттерну

        Args:
            pattern_id (int): ID паттерна
            result_data (dict): Данные результата

        Returns:
            bool: True если успешно
        """
        try:
            cursor = self.conn.cursor()

            cursor.execute('''
                INSERT INTO pattern_results 
                (pattern_id, outcome, pnl, pnl_percent, duration_hours, exit_reason, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                pattern_id,
                result_data.get('outcome', 'unknown'),
                result_data.get('pnl', 0.0),
                result_data.get('pnl_percent', 0.0),
                result_data.get('duration_hours', 0.0),
                result_data.get('exit_reason', ''),
                result_data.get('notes', '')
            ))

            self.conn.commit()

            # Обновляем статистику паттерна
            pattern = self.get_pattern_by_id(pattern_id)
            if pattern:
                self._update_statistics(
                    pattern['symbol'],
                    pattern['timeframe'],
                    pattern['pattern_type']
                )

            logger.debug(f"Результат добавлен для паттерна {pattern_id}")
            return True

        except Exception as e:
            logger.error(f"Ошибка добавления результата: {e}")
            self.conn.rollback()
            return False

    def get_pattern_results(self, pattern_id: Optional[int] = None,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Получение результатов торговли по паттернам

        Args:
            pattern_id (int): Фильтр по ID паттерна
            start_date (datetime): Начальная дата
            end_date (datetime): Конечная дата

        Returns:
            list: Список результатов
        """
        try:
            cursor = self.conn.cursor()

            query = """
                SELECT pr.*, p.symbol, p.timeframe, p.pattern_type 
                FROM pattern_results pr
                JOIN patterns p ON pr.pattern_id = p.id
                WHERE 1=1
            """
            params = []

            if pattern_id:
                query += " AND pr.pattern_id = ?"
                params.append(pattern_id)

            if start_date:
                query += " AND pr.created_at >= ?"
                params.append(start_date.isoformat())

            if end_date:
                query += " AND pr.created_at <= ?"
                params.append(end_date.isoformat())

            query += " ORDER BY pr.created_at DESC"

            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Преобразуем результаты
            results = []
            for row in rows:
                result = dict(row)

                # Преобразуем datetime
                if result.get('created_at'):
                    try:
                        result['created_at'] = datetime.fromisoformat(result['created_at'].replace('Z', '+00:00'))
                    except:
                        pass

                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Ошибка получения результатов: {e}")
            return []

    def get_pattern_by_id(self, pattern_id: int) -> Optional[Dict[str, Any]]:
        """
        Получение паттерна по ID

        Args:
            pattern_id (int): ID паттерна

        Returns:
            dict: Данные паттерна или None
        """
        try:
            cursor = self.conn.cursor()

            cursor.execute("SELECT * FROM patterns WHERE id = ?", (pattern_id,))
            row = cursor.fetchone()

            if row:
                pattern = dict(row)

                # Десериализуем JSON
                if pattern['metrics']:
                    pattern['metrics'] = json.loads(pattern['metrics'])
                if pattern['raw_data']:
                    pattern['raw_data'] = json.loads(pattern['raw_data'])

                # Преобразуем datetime
                for date_field in ['detection_time', 'data_start', 'data_end', 'created_at', 'updated_at']:
                    if pattern.get(date_field):
                        try:
                            pattern[date_field] = datetime.fromisoformat(pattern[date_field].replace('Z', '+00:00'))
                        except:
                            pass

                return pattern

            return None

        except Exception as e:
            logger.error(f"Ошибка получения паттерна по ID: {e}")
            return None

    def get_pattern_statistics(self, symbol: Optional[str] = None,
                               timeframe: Optional[str] = None,
                               pattern_type: Optional[str] = None,
                               period: str = 'all') -> Dict[str, Any]:
        """
        Получение статистики по паттернам

        Args:
            symbol (str): Фильтр по символу
            timeframe (str): Фильтр по таймфрейму
            pattern_type (str): Фильтр по типу паттерна
            period (str): Период ('day', 'week', 'month', 'all')

        Returns:
            dict: Статистика
        """
        try:
            cursor = self.conn.cursor()

            # Определяем условия периода
            period_conditions = {
                'day': "detection_time >= datetime('now', '-1 day')",
                'week': "detection_time >= datetime('now', '-7 days')",
                'month': "detection_time >= datetime('now', '-30 days')",
                'all': "1=1"
            }

            period_condition = period_conditions.get(period, '1=1')

            # Строим запрос для статистики паттернов
            query = f"""
                SELECT 
                    COUNT(*) as total_count,
                    AVG(confidence) as avg_confidence,
                    SUM(CASE WHEN confidence >= 0.7 THEN 1 ELSE 0 END) as high_confidence_count
                FROM patterns
                WHERE {period_condition}
            """

            params = []

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)

            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)

            if pattern_type:
                query += " AND pattern_type = ?"
                params.append(pattern_type)

            cursor.execute(query, params)
            pattern_stats = dict(cursor.fetchone())

            # Статистика по результатам
            query = f"""
                SELECT 
                    COUNT(*) as total_results,
                    SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as win_count,
                    SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as loss_count,
                    AVG(pnl) as avg_pnl,
                    AVG(pnl_percent) as avg_pnl_percent
                FROM pattern_results pr
                JOIN patterns p ON pr.pattern_id = p.id
                WHERE {period_condition.replace('detection_time', 'pr.created_at')}
            """

            params = []

            if symbol:
                query += " AND p.symbol = ?"
                params.append(symbol)

            if timeframe:
                query += " AND p.timeframe = ?"
                params.append(timeframe)

            if pattern_type:
                query += " AND p.pattern_type = ?"
                params.append(pattern_type)

            cursor.execute(query, params)
            result_stats = dict(cursor.fetchone())

            # Рассчитываем винрейт
            total_results = result_stats.get('total_results', 0)
            win_count = result_stats.get('win_count', 0)

            if total_results > 0:
                win_rate = (win_count / total_results) * 100
            else:
                win_rate = 0

            # Формируем итоговую статистику
            statistics = {
                'pattern_count': pattern_stats.get('total_count', 0),
                'avg_confidence': pattern_stats.get('avg_confidence', 0),
                'high_confidence_count': pattern_stats.get('high_confidence_count', 0),
                'result_count': total_results,
                'win_count': win_count,
                'loss_count': result_stats.get('loss_count', 0),
                'win_rate_percent': win_rate,
                'avg_pnl': result_stats.get('avg_pnl', 0),
                'avg_pnl_percent': result_stats.get('avg_pnl_percent', 0),
                'period': period,
                'symbol': symbol or 'all',
                'timeframe': timeframe or 'all',
                'pattern_type': pattern_type or 'all',
                'calculated_at': datetime.now().isoformat()
            }

            return statistics

        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {}

    def _update_statistics(self, symbol: str, timeframe: str, pattern_type: str):
        """
        Обновление агрегированной статистики

        Args:
            symbol (str): Символ
            timeframe (str): Таймфрейм
            pattern_type (str): Тип паттерна
        """
        try:
            cursor = self.conn.cursor()

            # Периоды для агрегации
            periods = ['day', 'week', 'month', 'all']

            for period in periods:
                # Получаем статистику за период
                stats = self.get_pattern_statistics(symbol, timeframe, pattern_type, period)

                if stats['result_count'] > 0:
                    # Вставляем или обновляем статистику
                    cursor.execute('''
                        INSERT OR REPLACE INTO pattern_stats 
                        (symbol, timeframe, pattern_type, period,
                         total_count, success_count, avg_confidence,
                         avg_pnl, avg_pnl_percent, win_rate)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol,
                        timeframe,
                        pattern_type,
                        period,
                        stats['pattern_count'],
                        stats['win_count'],
                        stats['avg_confidence'],
                        stats['avg_pnl'],
                        stats['avg_pnl_percent'],
                        stats['win_rate_percent']
                    ))

            self.conn.commit()

        except Exception as e:
            logger.error(f"Ошибка обновления статистики: {e}")
            self.conn.rollback()

    def cleanup_old_patterns(self, days_to_keep: int = 30) -> int:
        """
        Очистка старых паттернов

        Args:
            days_to_keep (int): Количество дней для хранения

        Returns:
            int: Количество удаленных паттернов
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()

            cursor = self.conn.cursor()

            # Получаем ID паттернов для удаления
            cursor.execute('''
                SELECT id FROM patterns 
                WHERE detection_time < ?
            ''', (cutoff_date,))

            old_patterns = cursor.fetchall()

            if not old_patterns:
                logger.info(f"Нет паттернов старше {days_to_keep} дней")
                return 0

            # Удаляем результаты паттернов
            pattern_ids = [p['id'] for p in old_patterns]
            placeholders = ','.join(['?'] * len(pattern_ids))

            cursor.execute(f'''
                DELETE FROM pattern_results 
                WHERE pattern_id IN ({placeholders})
            ''', pattern_ids)

            # Удаляем паттерны
            cursor.execute(f'''
                DELETE FROM patterns 
                WHERE id IN ({placeholders})
            ''', pattern_ids)

            deleted_count = cursor.rowcount
            self.conn.commit()

            logger.info(f"Удалено {deleted_count} паттернов старше {days_to_keep} дней")

            # Обновляем статистику
            self._cleanup_old_statistics(days_to_keep)

            return deleted_count

        except Exception as e:
            logger.error(f"Ошибка очистки старых паттернов: {e}")
            self.conn.rollback()
            return 0

    def _cleanup_old_statistics(self, days_to_keep: int):
        """
        Очистка старой статистики

        Args:
            days_to_keep (int): Количество дней для хранения
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()

            cursor = self.conn.cursor()
            cursor.execute('''
                DELETE FROM pattern_stats 
                WHERE created_at < ?
            ''', (cutoff_date,))

            self.conn.commit()

        except Exception as e:
            logger.error(f"Ошибка очистки статистики: {e}")

    def export_to_csv(self, filename: str,
                      symbol: Optional[str] = None,
                      timeframe: Optional[str] = None,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None):
        """
        Экспорт паттернов в CSV файл

        Args:
            filename (str): Имя файла
            symbol (str): Фильтр по символу
            timeframe (str): Фильтр по таймфрейму
            start_date (datetime): Начальная дата
            end_date (datetime): Конечная дата
        """
        try:
            # Получаем паттерны
            patterns = self.get_patterns(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                limit=10000  # Ограничение для экспорта
            )

            if not patterns:
                logger.warning("Нет данных для экспорта")
                return

            # Преобразуем в DataFrame
            df_data = []

            for pattern in patterns:
                row = {
                    'id': pattern['id'],
                    'symbol': pattern['symbol'],
                    'timeframe': pattern['timeframe'],
                    'pattern_type': pattern['pattern_type'],
                    'detection_time': pattern['detection_time'],
                    'confidence': pattern['confidence'],
                    'created_at': pattern['created_at']
                }

                # Добавляем метрики
                metrics = pattern.get('metrics', {})
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        if isinstance(value, (int, float, str, bool)):
                            row[f'metric_{key}'] = value

                df_data.append(row)

            df = pd.DataFrame(df_data)

            # Сохраняем в CSV
            df.to_csv(filename, index=False, encoding='utf-8')

            logger.info(f"Экспортировано {len(df)} паттернов в {filename}")

        except Exception as e:
            logger.error(f"Ошибка экспорта в CSV: {e}")

    def close(self):
        """Закрытие соединения с базой данных"""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Соединение с базой данных закрыто")

    def __del__(self):
        """Деструктор - закрытие соединения"""
        self.close()


# Создаем глобальный экземпляр для удобства
pattern_database = PatternDatabase()

