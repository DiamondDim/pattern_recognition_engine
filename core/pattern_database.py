"""
Модуль базы данных паттернов
"""

import sqlite3
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

from config import DATABASE_CONFIG
from utils.logger import logger
from utils.helpers import generate_id


class PatternDatabase:
    """База данных для хранения паттернов"""

    def __init__(self, db_path: str = None):
        self.config = DATABASE_CONFIG
        self.db_path = db_path or self.config.DB_PATH
        self.logger = logger.bind(module="PatternDatabase")
        self.connection = None
        self._initialize_database()

    def _initialize_database(self):
        """Инициализация базы данных"""
        try:
            # Создаем директорию если не существует
            db_file = Path(self.db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)

            # Подключаемся к БД
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row

            # Создаем таблицы
            self._create_tables()

            self.logger.info(f"База данных инициализирована: {self.db_path}")

        except Exception as e:
            self.logger.error(f"Ошибка инициализации БД: {e}")
            raise

    def _create_tables(self):
        """Создание таблиц базы данных"""
        cursor = self.connection.cursor()

        # Таблица паттернов
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                id TEXT PRIMARY KEY,
                pattern_type TEXT NOT NULL,
                pattern_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                direction TEXT NOT NULL,
                start_index INTEGER NOT NULL,
                end_index INTEGER NOT NULL,
                confidence REAL NOT NULL,
                quality_score REAL,
                data_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Таблица исторических результатов
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_history (
                id TEXT PRIMARY KEY,
                pattern_id TEXT NOT NULL,
                entry_price REAL,
                exit_price REAL,
                profit REAL,
                outcome TEXT,  -- 'success', 'failure', 'neutral'
                duration_hours INTEGER,
                max_profit REAL,
                max_loss REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (pattern_id) REFERENCES patterns (id)
            )
        ''')

        # Таблица статистики
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_statistics (
                pattern_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                total_occurrences INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                avg_confidence REAL,
                avg_profit REAL,
                avg_duration_hours REAL,
                success_rate REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (pattern_name, symbol, timeframe)
            )
        ''')

        # Индексы для оптимизации поиска
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_symbol ON patterns(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_timeframe ON patterns(timeframe)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_confidence ON patterns(confidence)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_history_pattern_id ON pattern_history(pattern_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_history_outcome ON pattern_history(outcome)')

        self.connection.commit()
        self.logger.debug("Таблицы базы данных созданы")

    def save_pattern(self, pattern_data: Dict[str, Any]) -> str:
        """
        Сохранение паттерна в БД

        Args:
            pattern_data: Данные паттерна

        Returns:
            ID сохраненного паттерна
        """
        try:
            # Генерация ID
            pattern_id = generate_id(prefix="pat")

            # Извлечение данных
            pattern_type = pattern_data.get('pattern_type', 'unknown')
            pattern_name = pattern_data.get('name', 'unknown')
            symbol = pattern_data.get('metadata', {}).get('symbol', 'UNKNOWN')
            timeframe = pattern_data.get('metadata', {}).get('timeframe', 'UNKNOWN')
            direction = pattern_data.get('direction', 'neutral')
            start_index = pattern_data.get('start_index', 0)
            end_index = pattern_data.get('end_index', 0)
            confidence = pattern_data.get('confidence', 0.5)
            quality_score = pattern_data.get('quality_score', 0.5)

            # Сериализация данных
            data_json = json.dumps(pattern_data, default=str)

            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT INTO patterns 
                (id, pattern_type, pattern_name, symbol, timeframe, direction, 
                 start_index, end_index, confidence, quality_score, data_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (pattern_id, pattern_type, pattern_name, symbol, timeframe, direction,
                  start_index, end_index, confidence, quality_score, data_json))

            self.connection.commit()

            # Обновление статистики
            self._update_statistics(pattern_name, symbol, timeframe, confidence)

            self.logger.debug(f"Паттерн сохранен: {pattern_id} ({pattern_name})")
            return pattern_id

        except Exception as e:
            self.logger.error(f"Ошибка сохранения паттерна: {e}")
            return ""

    def save_patterns_batch(self, patterns: List[Dict[str, Any]]) -> List[str]:
        """
        Пакетное сохранение паттернов

        Args:
            patterns: Список паттернов

        Returns:
            Список ID сохраненных паттернов
        """
        ids = []
        for pattern in patterns:
            pattern_id = self.save_pattern(pattern)
            if pattern_id:
                ids.append(pattern_id)

        self.logger.info(f"Сохранено паттернов: {len(ids)}")
        return ids

    def get_pattern_by_id(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение паттерна по ID

        Args:
            pattern_id: ID паттерна

        Returns:
            Данные паттерна или None
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute('SELECT * FROM patterns WHERE id = ?', (pattern_id,))
            row = cursor.fetchone()

            if row:
                pattern_data = json.loads(row['data_json'])
                pattern_data['db_id'] = row['id']
                pattern_data['created_at'] = row['created_at']
                pattern_data['updated_at'] = row['updated_at']
                return pattern_data

            return None

        except Exception as e:
            self.logger.error(f"Ошибка получения паттерна: {e}")
            return None

    def find_patterns(self,
                     pattern_type: Optional[str] = None,
                     pattern_name: Optional[str] = None,
                     symbol: Optional[str] = None,
                     timeframe: Optional[str] = None,
                     direction: Optional[str] = None,
                     min_confidence: float = 0.0,
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None,
                     limit: int = 100) -> List[Dict[str, Any]]:
        """
        Поиск паттернов по критериям

        Args:
            pattern_type: Тип паттерна
            pattern_name: Имя паттерна
            symbol: Символ
            timeframe: Таймфрейм
            direction: Направление
            min_confidence: Минимальная уверенность
            start_date: Начальная дата
            end_date: Конечная дата
            limit: Ограничение количества

        Returns:
            Список найденных паттернов
        """
        try:
            query = "SELECT * FROM patterns WHERE 1=1"
            params = []

            if pattern_type:
                query += " AND pattern_type = ?"
                params.append(pattern_type)

            if pattern_name:
                query += " AND pattern_name = ?"
                params.append(pattern_name)

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)

            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)

            if direction:
                query += " AND direction = ?"
                params.append(direction)

            if min_confidence > 0:
                query += " AND confidence >= ?"
                params.append(min_confidence)

            if start_date:
                query += " AND created_at >= ?"
                params.append(start_date.isoformat())

            if end_date:
                query += " AND created_at <= ?"
                params.append(end_date.isoformat())

            query += " ORDER BY confidence DESC, created_at DESC LIMIT ?"
            params.append(limit)

            cursor = self.connection.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

            patterns = []
            for row in rows:
                pattern_data = json.loads(row['data_json'])
                pattern_data['db_id'] = row['id']
                pattern_data['created_at'] = row['created_at']
                pattern_data['updated_at'] = row['updated_at']
                patterns.append(pattern_data)

            self.logger.debug(f"Найдено паттернов: {len(patterns)}")
            return patterns

        except Exception as e:
            self.logger.error(f"Ошибка поиска паттернов: {e}")
            return []

    def add_pattern_outcome(self,
                           pattern_id: str,
                           entry_price: float,
                           exit_price: float,
                           outcome: str,
                           duration_hours: float = 0.0) -> str:
        """
        Добавление результата паттерна

        Args:
            pattern_id: ID паттерна
            entry_price: Цена входа
            exit_price: Цена выхода
            outcome: Результат ('success', 'failure', 'neutral')
            duration_hours: Продолжительность в часах

        Returns:
            ID записи истории
        """
        try:
            history_id = generate_id(prefix="hist")

            profit = exit_price - entry_price if outcome != 'neutral' else 0

            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT INTO pattern_history 
                (id, pattern_id, entry_price, exit_price, profit, outcome, duration_hours)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (history_id, pattern_id, entry_price, exit_price, profit, outcome, duration_hours))

            self.connection.commit()

            # Обновление статистики паттерна
            pattern = self.get_pattern_by_id(pattern_id)
            if pattern:
                pattern_name = pattern.get('name', 'unknown')
                symbol = pattern.get('metadata', {}).get('symbol', 'UNKNOWN')
                timeframe = pattern.get('metadata', {}).get('timeframe', 'UNKNOWN')

                self._update_statistics_with_outcome(
                    pattern_name, symbol, timeframe, outcome, profit, duration_hours)

            self.logger.debug(f"Результат паттерна сохранен: {history_id}")
            return history_id

        except Exception as e:
            self.logger.error(f"Ошибка добавления результата: {e}")
            return ""

    def _update_statistics(self,
                          pattern_name: str,
                          symbol: str,
                          timeframe: str,
                          confidence: float):
        """Обновление статистики при сохранении паттерна"""
        try:
            cursor = self.connection.cursor()

            # Проверяем существование записи
            cursor.execute('''
                SELECT total_occurrences, avg_confidence 
                FROM pattern_statistics 
                WHERE pattern_name = ? AND symbol = ? AND timeframe = ?
            ''', (pattern_name, symbol, timeframe))

            row = cursor.fetchone()

            if row:
                # Обновляем существующую запись
                total = row['total_occurrences'] + 1
                avg_conf = (row['avg_confidence'] * row['total_occurrences'] + confidence) / total

                cursor.execute('''
                    UPDATE pattern_statistics 
                    SET total_occurrences = ?, avg_confidence = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE pattern_name = ? AND symbol = ? AND timeframe = ?
                ''', (total, avg_conf, pattern_name, symbol, timeframe))
            else:
                # Создаем новую запись
                cursor.execute('''
                    INSERT INTO pattern_statistics 
                    (pattern_name, symbol, timeframe, total_occurrences, avg_confidence)
                    VALUES (?, ?, ?, ?, ?)
                ''', (pattern_name, symbol, timeframe, 1, confidence))

            self.connection.commit()

        except Exception as e:
            self.logger.error(f"Ошибка обновления статистики: {e}")

    def _update_statistics_with_outcome(self,
                                       pattern_name: str,
                                       symbol: str,
                                       timeframe: str,
                                       outcome: str,
                                       profit: float,
                                       duration_hours: float):
        """Обновление статистики с учетом результата"""
        try:
            cursor = self.connection.cursor()

            cursor.execute('''
                SELECT success_count, failure_count, avg_profit, avg_duration_hours
                FROM pattern_statistics 
                WHERE pattern_name = ? AND symbol = ? AND timeframe = ?
            ''', (pattern_name, symbol, timeframe))

            row = cursor.fetchone()

            if row:
                success_count = row['success_count']
                failure_count = row['failure_count']
                total_outcomes = success_count + failure_count

                if outcome == 'success':
                    success_count += 1
                elif outcome == 'failure':
                    failure_count += 1

                total_outcomes_new = success_count + failure_count

                # Обновляем средние значения
                if total_outcomes > 0:
                    avg_profit = (row['avg_profit'] * total_outcomes + profit) / total_outcomes_new
                    avg_duration = (row['avg_duration_hours'] * total_outcomes + duration_hours) / total_outcomes_new
                else:
                    avg_profit = profit
                    avg_duration = duration_hours

                success_rate = success_count / total_outcomes_new if total_outcomes_new > 0 else 0

                cursor.execute('''
                    UPDATE pattern_statistics 
                    SET success_count = ?, failure_count = ?, 
                        avg_profit = ?, avg_duration_hours = ?, success_rate = ?,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE pattern_name = ? AND symbol = ? AND timeframe = ?
                ''', (success_count, failure_count, avg_profit, avg_duration, success_rate,
                      pattern_name, symbol, timeframe))

            self.connection.commit()

        except Exception as e:
            self.logger.error(f"Ошибка обновления статистики с результатом: {e}")

    def get_pattern_statistics(self,
                              pattern_name: Optional[str] = None,
                              symbol: Optional[str] = None,
                              timeframe: Optional[str] = None) -> Dict[str, Any]:
        """
        Получение статистики паттернов

        Args:
            pattern_name: Имя паттерна
            symbol: Символ
            timeframe: Таймфрейм

        Returns:
            Статистика паттернов
        """
        try:
            query = "SELECT * FROM pattern_statistics WHERE 1=1"
            params = []

            if pattern_name:
                query += " AND pattern_name = ?"
                params.append(pattern_name)

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)

            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)

            cursor = self.connection.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

            statistics = []
            for row in rows:
                stats = {
                    'pattern_name': row['pattern_name'],
                    'symbol': row['symbol'],
                    'timeframe': row['timeframe'],
                    'total_occurrences': row['total_occurrences'],
                    'success_count': row['success_count'],
                    'failure_count': row['failure_count'],
                    'avg_confidence': row['avg_confidence'],
                    'avg_profit': row['avg_profit'],
                    'avg_duration_hours': row['avg_duration_hours'],
                    'success_rate': row['success_rate'],
                    'last_updated': row['last_updated']
                }
                statistics.append(stats)

            # Агрегированная статистика
            if statistics:
                total_stats = {
                    'total_patterns': len(statistics),
                    'avg_success_rate': np.mean([s['success_rate'] for s in statistics if s['success_rate']]),
                    'total_occurrences': sum(s['total_occurrences'] for s in statistics),
                    'total_successes': sum(s['success_count'] for s in statistics),
                    'total_failures': sum(s['failure_count'] for s in statistics)
                }
            else:
                total_stats = {}

            return {
                'detailed': statistics,
                'aggregated': total_stats
            }

        except Exception as e:
            self.logger.error(f"Ошибка получения статистики: {e}")
            return {}

    def get_similar_patterns(self,
                            pattern_data: Dict[str, Any],
                            max_distance: float = 0.1) -> List[Dict[str, Any]]:
        """
        Поиск похожих паттернов

        Args:
            pattern_data: Данные паттерна для сравнения
            max_distance: Максимальное расстояние для схожести

        Returns:
            Список похожих паттернов
        """
        try:
            pattern_name = pattern_data.get('name', '')
            symbol = pattern_data.get('metadata', {}).get('symbol', '')
            timeframe = pattern_data.get('metadata', {}).get('timeframe', '')

            # Получаем все паттерны того же типа
            similar_patterns = self.find_patterns(
                pattern_name=pattern_name,
                symbol=symbol,
                timeframe=timeframe,
                limit=1000
            )

            # Если нет достаточно данных
            if len(similar_patterns) < 5:
                return []

            # Извлекаем характеристики для сравнения
            target_points = pattern_data.get('points', [])
            if not target_points:
                return []

            target_prices = [p.get('price', 0) for p in target_points]
            target_range = max(target_prices) - min(target_prices)

            if target_range == 0:
                return []

            # Нормализация цен целевого паттерна
            target_norm = [(p - min(target_prices)) / target_range for p in target_prices]

            results = []
            for pattern in similar_patterns:
                # Пропускаем сам паттерн если он уже в БД
                if pattern.get('db_id') == pattern_data.get('db_id'):
                    continue

                points = pattern.get('points', [])
                if len(points) != len(target_points):
                    continue

                # Нормализация цен паттерна из БД
                pattern_prices = [p.get('price', 0) for p in points]
                pattern_range = max(pattern_prices) - min(pattern_prices)

                if pattern_range == 0:
                    continue

                pattern_norm = [(p - min(pattern_prices)) / pattern_range for p in pattern_prices]

                # Расчет расстояния (евклидово)
                distance = np.sqrt(np.sum((np.array(target_norm) - np.array(pattern_norm)) ** 2))

                if distance <= max_distance:
                    pattern['similarity_distance'] = distance
                    results.append(pattern)

            # Сортировка по схожести
            results.sort(key=lambda x: x.get('similarity_distance', 1.0))

            self.logger.debug(f"Найдено похожих паттернов: {len(results)}")
            return results[:10]  # Возвращаем топ-10

        except Exception as e:
            self.logger.error(f"Ошибка поиска похожих паттернов: {e}")
            return []

    def cleanup_old_patterns(self, days_to_keep: int = 30) -> int:
        """
        Очистка старых паттернов

        Args:
            days_to_keep: Количество дней для хранения

        Returns:
            Количество удаленных записей
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            cursor = self.connection.cursor()

            # Сначала удаляем связанные записи истории
            cursor.execute('''
                DELETE FROM pattern_history 
                WHERE pattern_id IN (
                    SELECT id FROM patterns 
                    WHERE created_at < ?
                )
            ''', (cutoff_date.isoformat(),))

            # Затем удаляем старые паттерны
            cursor.execute('DELETE FROM patterns WHERE created_at < ?', (cutoff_date.isoformat(),))

            deleted_count = cursor.rowcount
            self.connection.commit()

            # Вакуумирование базы данных
            cursor.execute('VACUUM')

            self.logger.info(f"Удалено старых паттернов: {deleted_count}")
            return deleted_count

        except Exception as e:
            self.logger.error(f"Ошибка очистки старых паттернов: {e}")
            return 0

    def export_to_csv(self, filepath: str, table_name: str = 'patterns') -> bool:
        """
        Экспорт таблицы в CSV

        Args:
            filepath: Путь для сохранения
            table_name: Имя таблицы

        Returns:
            Успешность экспорта
        """
        try:
            if table_name not in ['patterns', 'pattern_history', 'pattern_statistics']:
                self.logger.error(f"Неподдерживаемая таблица: {table_name}")
                return False

            cursor = self.connection.cursor()
            cursor.execute(f'SELECT * FROM {table_name}')
            rows = cursor.fetchall()

            if not rows:
                self.logger.warning(f"Таблица {table_name} пуста")
                return False

            # Конвертация в DataFrame
            df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])

            # Сохранение в CSV
            df.to_csv(filepath, index=False)

            self.logger.info(f"Таблица {table_name} экспортирована: {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка экспорта в CSV: {e}")
            return False

    def close(self):
        """Закрытие соединения с БД"""
        if self.connection:
            self.connection.close()
            self.connection = None
            self.logger.info("Соединение с БД закрыто")

