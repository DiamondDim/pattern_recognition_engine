import sqlite3
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
import logging
import os
from ..patterns.base_pattern import Pattern

logger = logging.getLogger(__name__)

class PatternDatabase:
    """База данных для хранения и анализа паттернов"""
    
    def __init__(self, db_path: str = "patterns.db"):
        self.db_path = db_path
        self.conn = None
        self._init_database()
    
    def _init_database(self):
        """Инициализация базы данных"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            cursor = self.conn.cursor()
            
            # Таблица паттернов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    signal TEXT NOT NULL,
                    confidence REAL,
                    data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, pattern_type, timeframe, timestamp)
                )
            ''')
            
            # Таблица производительности паттернов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pattern_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_id INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    profit REAL NOT NULL,
                    profit_pct REAL NOT NULL,
                    duration_hours INTEGER NOT NULL,
                    success BOOLEAN NOT NULL,
                    exit_reason TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (pattern_id) REFERENCES patterns (id) ON DELETE CASCADE
                )
            ''')
            
            # Таблица статистики паттернов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pattern_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    total_occurrences INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    total_profit REAL DEFAULT 0,
                    avg_profit_pct REAL DEFAULT 0,
                    avg_confidence REAL DEFAULT 0,
                    last_occurrence DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, pattern_type, timeframe)
                )
            ''')
            
            # Создаем индексы для быстрого поиска
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_symbol ON patterns (symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns (pattern_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_timeframe ON patterns (timeframe)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_timestamp ON patterns (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_pattern ON pattern_performance (pattern_id)')
            
            self.conn.commit()
            logger.info(f"База данных инициализирована: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации БД: {e}")
            raise
    
    def add_pattern(self, pattern: Pattern, timeframe: str = "D1") -> Optional[int]:
        """Добавление паттерна в БД"""
        try:
            cursor = self.conn.cursor()
            
            # Проверяем, нет ли уже такого паттерна
            cursor.execute('''
                SELECT id FROM patterns 
                WHERE symbol = ? AND pattern_type = ? AND timeframe = ? AND timestamp = ?
            ''', (pattern.symbol, pattern.pattern_type, timeframe, pattern.timestamp.isoformat()))
            
            existing = cursor.fetchone()
            if existing:
                logger.debug(f"Паттерн уже существует в БД: ID {existing['id']}")
                return existing['id']
            
            # Добавляем новый паттерн
            cursor.execute('''
                INSERT INTO patterns (symbol, pattern_type, timeframe, timestamp, signal, confidence, data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                pattern.symbol,
                pattern.pattern_type,
                timeframe,
                pattern.timestamp.isoformat() if hasattr(pattern.timestamp, 'isoformat') else str(pattern.timestamp),
                pattern.get_signal(),
                pattern.confidence,
                json.dumps(pattern.data) if pattern.data else '{}'
            ))
            
            pattern_id = cursor.lastrowid
            
            # Обновляем статистику
            self._update_statistics(pattern.symbol, pattern.pattern_type, timeframe)
            
            self.conn.commit()
            logger.debug(f"Паттерн добавлен в БД с ID: {pattern_id}")
            return pattern_id
            
        except Exception as e:
            logger.error(f"Ошибка добавления паттерна в БД: {e}")
            self.conn.rollback()
            return None
    
    def _update_statistics(self, symbol: str, pattern_type: str, timeframe: str):
        """Обновление статистики паттерна"""
        try:
            cursor = self.conn.cursor()
            
            # Получаем текущую статистику
            cursor.execute('''
                SELECT COUNT(*) as total, AVG(confidence) as avg_conf, MAX(timestamp) as last_occ
                FROM patterns 
                WHERE symbol = ? AND pattern_type = ? AND timeframe = ?
            ''', (symbol, pattern_type, timeframe))
            
            stats = cursor.fetchone()
            
            # Обновляем или вставляем статистику
            cursor.execute('''
                INSERT OR REPLACE INTO pattern_statistics 
                (symbol, pattern_type, timeframe, total_occurrences, avg_confidence, last_occurrence, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                symbol,
                pattern_type,
                timeframe,
                stats['total'] if stats else 0,
                stats['avg_conf'] if stats and stats['avg_conf'] else 0,
                stats['last_occ'] if stats else None
            ))
            
        except Exception as e:
            logger.error(f"Ошибка обновления статистики: {e}")
    
    def add_performance_record(self, pattern_id: int, entry_price: float, 
                              exit_price: float, profit: float, 
                              profit_pct: float, duration_hours: int, 
                              success: bool, exit_reason: str = "") -> Optional[int]:
        """Добавление записи о производительности паттерна"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                INSERT INTO pattern_performance 
                (pattern_id, entry_price, exit_price, profit, profit_pct, duration_hours, success, exit_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (pattern_id, entry_price, exit_price, profit, profit_pct, duration_hours, success, exit_reason))
            
            record_id = cursor.lastrowid
            
            # Обновляем статистику успешности
            self._update_success_statistics(pattern_id)
            
            self.conn.commit()
            return record_id
            
        except Exception as e:
            logger.error(f"Ошибка добавления записи производительности: {e}")
            self.conn.rollback()
            return None
    
    def _update_success_statistics(self, pattern_id: int):
        """Обновление статистики успешности для паттерна"""
        try:
            cursor = self.conn.cursor()
            
            # Получаем информацию о паттерне
            cursor.execute('''
                SELECT symbol, pattern_type, timeframe 
                FROM patterns WHERE id = ?
            ''', (pattern_id,))
            
            pattern_info = cursor.fetchone()
            if not pattern_info:
                return
            
            # Получаем статистику производительности
            cursor.execute('''
                SELECT COUNT(*) as total, 
                       SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count,
                       AVG(profit_pct) as avg_profit,
                       SUM(profit) as total_profit
                FROM pattern_performance pp
                JOIN patterns p ON pp.pattern_id = p.id
                WHERE p.symbol = ? AND p.pattern_type = ? AND p.timeframe = ?
            ''', (pattern_info['symbol'], pattern_info['pattern_type'], pattern_info['timeframe']))
            
            perf_stats = cursor.fetchone()
            
            # Обновляем статистику
            cursor.execute('''
                UPDATE pattern_statistics 
                SET success_count = ?, 
                    total_profit = ?,
                    avg_profit_pct = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE symbol = ? AND pattern_type = ? AND timeframe = ?
            ''', (
                perf_stats['success_count'] if perf_stats else 0,
                perf_stats['total_profit'] if perf_stats else 0,
                perf_stats['avg_profit'] if perf_stats else 0,
                pattern_info['symbol'],
                pattern_info['pattern_type'],
                pattern_info['timeframe']
            ))
            
        except Exception as e:
            logger.error(f"Ошибка обновления статистики успешности: {e}")
    
    def get_patterns(self, symbol: Optional[str] = None, 
                     pattern_type: Optional[str] = None,
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None,
                     timeframe: Optional[str] = None,
                     min_confidence: float = 0.0,
                     limit: int = 1000) -> List[Dict]:
        """Получение паттернов из БД с фильтрами"""
        try:
            cursor = self.conn.cursor()
            
            query = '''
                SELECT p.*, 
                       ps.total_occurrences,
                       ps.success_count,
                       ps.avg_profit_pct
                FROM patterns p
                LEFT JOIN pattern_statistics ps ON 
                    p.symbol = ps.symbol AND 
                    p.pattern_type = ps.pattern_type AND 
                    p.timeframe = ps.timeframe
                WHERE 1=1
            '''
            
            params = []
            
            if symbol:
                query += " AND p.symbol = ?"
                params.append(symbol)
            
            if pattern_type:
                query += " AND p.pattern_type = ?"
                params.append(pattern_type)
            
            if timeframe:
                query += " AND p.timeframe = ?"
                params.append(timeframe)
            
            if start_date:
                query += " AND p.timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND p.timestamp <= ?"
                params.append(end_date.isoformat())
            
            if min_confidence > 0:
                query += " AND p.confidence >= ?"
                params.append(min_confidence)
            
            query += " ORDER BY p.timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Преобразуем в словари
            patterns = []
            for row in rows:
                pattern = dict(row)
                
                # Парсим JSON данные
                if pattern['data']:
                    try:
                        pattern['data'] = json.loads(pattern['data'])
                    except:
                        pattern['data'] = {}
                
                # Преобразуем timestamp
                if pattern['timestamp']:
                    pattern['timestamp'] = datetime.fromisoformat(pattern['timestamp'])
                
                if pattern['created_at']:
                    pattern['created_at'] = datetime.fromisoformat(pattern['created_at'])
                
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Ошибка получения паттернов из БД: {e}")
            return []
    
    def get_pattern_statistics(self, symbol: Optional[str] = None,
                              pattern_type: Optional[str] = None,
                              timeframe: Optional[str] = None) -> Dict[str, Any]:
        """Получение статистики по паттернам"""
        try:
            cursor = self.conn.cursor()
            
            query = '''
                SELECT 
                    symbol,
                    pattern_type,
                    timeframe,
                    total_occurrences,
                    success_count,
                    total_profit,
                    avg_profit_pct,
                    avg_confidence,
                    last_occurrence
                FROM pattern_statistics
                WHERE 1=1
            '''
            
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if pattern_type:
                query += " AND pattern_type = ?"
                params.append(pattern_type)
            
            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)
            
            query += " ORDER BY total_occurrences DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Общая статистика
            total_patterns = 0
            total_success = 0
            total_profit = 0.0
            
            pattern_stats = []
            for row in rows:
                stats = dict(row)
                
                # Расчет процента успешности
                if stats['total_occurrences'] > 0:
                    success_rate = (stats['success_count'] / stats['total_occurrences']) * 100
                else:
                    success_rate = 0
                
                stats['success_rate'] = success_rate
                
                # Преобразуем дату
                if stats['last_occurrence']:
                    stats['last_occurrence'] = datetime.fromisoformat(stats['last_occurrence'])
                
                pattern_stats.append(stats)
                
                # Суммируем общую статистику
                total_patterns += stats['total_occurrences']
                total_success += stats['success_count']
                total_profit += stats['total_profit'] if stats['total_profit'] else 0
            
            # Расчет общей статистики
            overall_success_rate = (total_success / total_patterns * 100) if total_patterns > 0 else 0
            
            return {
                'pattern_statistics': pattern_stats,
                'overall_statistics': {
                    'total_patterns': total_patterns,
                    'total_success': total_success,
                    'total_profit': total_profit,
                    'success_rate': overall_success_rate
                }
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {}
    
    def get_best_patterns(self, symbol: Optional[str] = None, 
                         min_occurrences: int = 5,
                         min_success_rate: float = 60.0) -> List[Dict]:
        """Получение лучших паттернов по успешности"""
        try:
            cursor = self.conn.cursor()
            
            query = '''
                SELECT 
                    symbol,
                    pattern_type,
                    timeframe,
                    total_occurrences,
                    success_count,
                    (success_count * 100.0 / total_occurrences) as success_rate,
                    avg_profit_pct,
                    last_occurrence
                FROM pattern_statistics
                WHERE total_occurrences >= ?
            '''
            
            params = [min_occurrences]
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            query += " HAVING success_rate >= ? ORDER BY success_rate DESC, total_occurrences DESC"
            params.append(min_success_rate)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            best_patterns = []
            for row in rows:
                pattern = dict(row)
                
                if pattern['last_occurrence']:
                    pattern['last_occurrence'] = datetime.fromisoformat(pattern['last_occurrence'])
                
                best_patterns.append(pattern)
            
            return best_patterns
            
        except Exception as e:
            logger.error(f"Ошибка получения лучших паттернов: {e}")
            return []
    
    def cleanup_old_patterns(self, days_to_keep: int = 90):
        """Очистка старых паттернов"""
        try:
            cursor = self.conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Получаем ID паттернов для удаления
            cursor.execute('''
                SELECT id FROM patterns WHERE timestamp < ?
            ''', (cutoff_date.isoformat(),))
            
            old_patterns = cursor.fetchall()
            pattern_ids = [p['id'] for p in old_patterns]
            
            # Удаляем записи производительности для старых паттернов
            if pattern_ids:
                placeholders = ','.join(['?'] * len(pattern_ids))
                cursor.execute(f'''
                    DELETE FROM pattern_performance 
                    WHERE pattern_id IN ({placeholders})
                ''', pattern_ids)
            
            # Удаляем старые паттерны
            deleted_count = cursor.execute('''
                DELETE FROM patterns WHERE timestamp < ?
            ''', (cutoff_date.isoformat(),)).rowcount
            
            # Обновляем статистику для затронутых символов/паттернов
            cursor.execute('''
                SELECT DISTINCT symbol, pattern_type, timeframe 
                FROM patterns WHERE timestamp >= ?
            ''', (cutoff_date.isoformat(),))
            
            affected_patterns = cursor.fetchall()
            for pattern in affected_patterns:
                self._update_statistics(pattern['symbol'], pattern['pattern_type'], pattern['timeframe'])
            
            self.conn.commit()
            logger.info(f"Удалено {deleted_count} старых паттернов (старше {days_to_keep} дней)")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Ошибка очистки БД: {e}")
            self.conn.rollback()
            return 0
    
    def export_to_csv(self, filepath: str, symbol: Optional[str] = None):
        """Экспорт данных в CSV"""
        try:
            patterns = self.get_patterns(symbol=symbol, limit=10000)
            
            if not patterns:
                logger.warning("Нет данных для экспорта")
                return False
            
            df = pd.DataFrame(patterns)
            
            # Убираем столбец с данными (может быть большим)
            if 'data' in df.columns:
                df = df.drop('data', axis=1)
            
            df.to_csv(filepath, index=False, encoding='utf-8')
            logger.info(f"Данные экспортированы в {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка экспорта в CSV: {e}")
            return False
    
    def get_database_size(self) -> Dict[str, Any]:
        """Получение информации о размере БД"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("SELECT COUNT(*) as count FROM patterns")
            patterns_count = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM pattern_performance")
            performance_count = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM pattern_statistics")
            statistics_count = cursor.fetchone()['count']
            
            # Размер файла БД
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            
            return {
                'patterns_count': patterns_count,
                'performance_count': performance_count,
                'statistics_count': statistics_count,
                'database_size_mb': db_size / (1024 * 1024),
                'database_path': self.db_path
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения размера БД: {e}")
            return {}
    
    def close(self):
        """Закрытие соединения с БД"""
        if self.conn:
            self.conn.close()
            logger.info("Соединение с БД закрыто")

