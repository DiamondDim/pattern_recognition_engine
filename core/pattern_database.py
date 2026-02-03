"""
Модуль работы с базой данных паттернов
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd

try:
    from config import config
except ImportError:
    # Для обратной совместимости
    from config import config


class PatternDatabase:
    """Класс для работы с базой данных паттернов"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.DATABASE.NAME
        self.connection = None
        self._initialize_database()

    def _initialize_database(self):
        """Инициализация базы данных"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            cursor = self.connection.cursor()

            # Создаем таблицу для паттернов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    pattern_name TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    quality REAL,
                    price REAL,
                    direction TEXT,
                    timestamp DATETIME,
                    detection_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    UNIQUE(symbol, timeframe, pattern_name, timestamp)
                )
            ''')

            # Создаем таблицу для статистики
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    total_patterns INTEGER DEFAULT 0,
                    avg_quality REAL DEFAULT 0,
                    last_update DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Создаем таблицу для backtesting результатов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    start_date DATETIME,
                    end_date DATETIME,
                    initial_balance REAL,
                    final_balance REAL,
                    total_trades INTEGER,
                    profitable_trades INTEGER,
                    win_rate REAL,
                    profit_factor REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    parameters TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            self.connection.commit()
            print(f"База данных инициализирована: {self.db_path}")

        except Exception as e:
            print(f"Ошибка инициализации базы данных: {e}")
            if self.connection:
                self.connection.close()

    def save_pattern(self, symbol: str, timeframe: str, pattern_name: str,
                    pattern_type: str, quality: float, metadata: Dict = None,
                    price: Optional[float] = None, direction: Optional[str] = None,
                    timestamp: Optional[datetime] = None):
        """Сохранение паттерна в базу данных"""
        try:
            cursor = self.connection.cursor()

            # Конвертируем metadata в JSON
            metadata_json = json.dumps(metadata) if metadata else None

            # Используем текущее время если timestamp не указан
            if timestamp is None:
                timestamp = datetime.now()

            cursor.execute('''
                INSERT OR REPLACE INTO patterns 
                (symbol, timeframe, pattern_name, pattern_type, quality, price, direction, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, timeframe, pattern_name, pattern_type, quality, price, direction, timestamp, metadata_json))

            self.connection.commit()
            return cursor.lastrowid

        except Exception as e:
            print(f"Ошибка сохранения паттерна: {e}")
            return None

    def get_patterns(self, symbol: Optional[str] = None, timeframe: Optional[str] = None,
                    pattern_type: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Получение паттернов из базы данных"""
        try:
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

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor = self.connection.cursor()
            cursor.execute(query, params)

            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()

            patterns = []
            for row in rows:
                pattern = dict(zip(columns, row))

                # Парсим metadata если есть
                if pattern.get('metadata'):
                    try:
                        pattern['metadata'] = json.loads(pattern['metadata'])
                    except:
                        pass

                patterns.append(pattern)

            return patterns

        except Exception as e:
            print(f"Ошибка получения паттернов: {e}")
            return []

    def get_statistics(self, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> Dict:
        """Получение статистики по паттернам"""
        try:
            query = """
                SELECT 
                    COUNT(*) as total_patterns,
                    AVG(quality) as avg_quality,
                    pattern_type,
                    direction
                FROM patterns 
                WHERE 1=1
            """

            params = []

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)

            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)

            query += " GROUP BY pattern_type, direction"

            cursor = self.connection.cursor()
            cursor.execute(query, params)

            stats = {
                'total': 0,
                'avg_quality': 0,
                'by_type': {},
                'by_direction': {}
            }

            for row in cursor.fetchall():
                total, avg_quality, pattern_type, direction = row

                stats['total'] += total
                stats['avg_quality'] = avg_quality if avg_quality else 0

                if pattern_type:
                    stats['by_type'][pattern_type] = stats['by_type'].get(pattern_type, 0) + total

                if direction:
                    stats['by_direction'][direction] = stats['by_direction'].get(direction, 0) + total

            return stats

        except Exception as e:
            print(f"Ошибка получения статистики: {e}")
            return {}

    def save_backtest_result(self, symbol: str, timeframe: str, start_date: datetime,
                           end_date: datetime, initial_balance: float, final_balance: float,
                           total_trades: int, profitable_trades: int, parameters: Dict = None):
        """Сохранение результатов backtesting"""
        try:
            cursor = self.connection.cursor()

            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            profit_factor = final_balance / initial_balance if initial_balance > 0 else 0

            parameters_json = json.dumps(parameters) if parameters else None

            cursor.execute('''
                INSERT INTO backtest_results 
                (symbol, timeframe, start_date, end_date, initial_balance, final_balance,
                 total_trades, profitable_trades, win_rate, profit_factor, parameters)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, timeframe, start_date, end_date, initial_balance, final_balance,
                  total_trades, profitable_trades, win_rate, profit_factor, parameters_json))

            self.connection.commit()
            return cursor.lastrowid

        except Exception as e:
            print(f"Ошибка сохранения результатов backtesting: {e}")
            return None

    def get_backtest_results(self, symbol: Optional[str] = None,
                           timeframe: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Получение результатов backtesting"""
        try:
            query = "SELECT * FROM backtest_results WHERE 1=1"
            params = []

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)

            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor = self.connection.cursor()
            cursor.execute(query, params)

            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()

            results = []
            for row in rows:
                result = dict(zip(columns, row))

                # Парсим parameters если есть
                if result.get('parameters'):
                    try:
                        result['parameters'] = json.loads(result['parameters'])
                    except:
                        pass

                results.append(result)

            return results

        except Exception as e:
            print(f"Ошибка получения результатов backtesting: {e}")
            return []

    def close(self):
        """Закрытие соединения с базой данных"""
        if self.connection:
            self.connection.close()
            print("Соединение с базой данных закрыто")

    def __del__(self):
        self.close()

