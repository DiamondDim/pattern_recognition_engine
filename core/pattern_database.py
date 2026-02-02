"""
База данных для хранения и управления паттернами
"""

import sqlite3
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import pandas as pd
from pathlib import Path

from config import DB_CONFIG
from utils.logger import logger


@dataclass
class PatternRecord:
    """Запись паттерна в базе данных"""

    id: str
    symbol: str
    timeframe: str
    pattern_type: str
    pattern_name: str
    direction: str
    detected_time: datetime
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    profit_risk_ratio: Optional[float] = None
    quality_score: float = 0.0
    confidence: float = 0.0
    market_context: str = "neutral"
    metadata: Dict[str, Any] = field(default_factory=dict)
    points: List[Dict[str, Any]] = field(default_factory=list)
    outcome: Optional[str] = None
    outcome_time: Optional[datetime] = None
    realized_profit: Optional[float] = None
    realized_risk: Optional[float] = None
    holding_period: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'pattern_type': self.pattern_type,
            'pattern_name': self.pattern_name,
            'direction': self.direction,
            'detected_time': self.detected_time.isoformat(),
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'profit_risk_ratio': self.profit_risk_ratio,
            'quality_score': self.quality_score,
            'confidence': self.confidence,
            'market_context': self.market_context,
            'metadata': self.metadata,
            'points': self.points,
            'outcome': self.outcome,
            'outcome_time': self.outcome_time.isoformat() if self.outcome_time else None,
            'realized_profit': self.realized_profit,
            'realized_risk': self.realized_risk,
            'holding_period': self.holding_period
        }


class PatternDatabase:
    """Класс для работы с базой данных паттернов"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_CONFIG.connection_string
        self.logger = logger.bind(name="PatternDatabase")

        # Подключение к базе данных
        self.connection = None
        self.cursor = None

        # Инициализация базы данных
        self._initialize_database()

    def _initialize_database(self):
        """Инициализация базы данных"""
        try:
            # Создаем подключение
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            self.cursor = self.connection.cursor()

            # Создаем таблицы
            self._create_tables()

            self.logger.info(f"База данных инициализирована: {self.db_path}")

        except Exception as e:
            self.logger.error(f"Ошибка инициализации базы данных: {e}")
            raise

    def _create_tables(self):
        """Создание таблиц базы данных"""

        # Таблица паттернов
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                pattern_name TEXT NOT NULL,
                direction TEXT NOT NULL,
                detected_time DATETIME NOT NULL,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                profit_risk_ratio REAL,
                quality_score REAL,
                confidence REAL,
                market_context TEXT,
                metadata TEXT,
                points TEXT,
                outcome TEXT,
                outcome_time DATETIME,
                realized_profit REAL,
                realized_risk REAL,
                holding_period REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Индексы для ускорения поиска
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_patterns_symbol 
            ON patterns (symbol)
        ''')

        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_patterns_timeframe 
            ON patterns (timeframe)
        ''')

        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_patterns_type 
            ON patterns (pattern_type)
        ''')

        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_patterns_direction 
            ON patterns (direction)
        ''')

        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_patterns_detected_time 
            ON patterns (detected_time)
        ''')

        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_patterns_outcome 
            ON patterns (outcome)
        ''')

        # Таблица статистики паттернов
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_statistics (
                pattern_name TEXT,
                symbol TEXT,
                timeframe TEXT,
                direction TEXT,
                total_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                avg_quality REAL DEFAULT 0,
                avg_profit REAL DEFAULT 0,
                avg_risk REAL DEFAULT 0,
                avg_holding_period REAL DEFAULT 0,
                success_rate REAL DEFAULT 0,
                profit_factor REAL DEFAULT 0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (pattern_name, symbol, timeframe, direction)
            )
        ''')

        # Таблица торговых сигналов
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_signals (
                id TEXT PRIMARY KEY,
                pattern_id TEXT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                risk_amount REAL,
                reward_amount REAL,
                profit_risk_ratio REAL,
                confidence REAL,
                status TEXT DEFAULT 'pending',
                created_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                executed_time DATETIME,
                closed_time DATETIME,
                result TEXT,
                pnl REAL,
                FOREIGN KEY (pattern_id) REFERENCES patterns (id)
            )
        ''')

        self.connection.commit()

        self.logger.debug("Таблицы базы данных созданы")

    def save_pattern(self, pattern_data: Dict[str, Any]) -> bool:
        """
        Сохранение паттерна в базу данных

        Args:
            pattern_data: Данные паттерна

        Returns:
            True если успешно сохранено
        """
        try:
            # Извлекаем данные
            pattern_id = pattern_data.get('id', '')
            symbol = pattern_data.get('metadata', {}).get('symbol', 'UNKNOWN')
            timeframe = pattern_data.get('metadata', {}).get('timeframe', 'UNKNOWN')
            pattern_name = pattern_data.get('name', '')
            pattern_type = pattern_data.get('type', '')
            direction = pattern_data.get('direction', 'neutral')

            # Время обнаружения
            detection_time_str = pattern_data.get('detection_time', '')
            if detection_time_str:
                detected_time = datetime.fromisoformat(detection_time_str.replace('Z', '+00:00'))
            else:
                detected_time = datetime.now()

            # Целевые уровни
            targets = pattern_data.get('targets', {})
            entry_price = targets.get('entry')
            stop_loss = targets.get('stop_loss')
            take_profit = targets.get('take_profit')
            profit_risk_ratio = targets.get('profit_risk_ratio')

            # Качество и уверенность
            metadata = pattern_data.get('metadata', {})
            quality_score = metadata.get('quality_score', 0)
            confidence = metadata.get('confidence', 0)
            market_context = metadata.get('market_context', 'neutral')

            # Точки и дополнительные метаданные
            points = pattern_data.get('points', [])
            full_metadata = {
                'indicators': metadata.get('indicators', {}),
                'volume_confirmation': metadata.get('volume_confirmation', False),
                'trend_confirmation': metadata.get('trend_confirmation', False),
                'strength_analysis': pattern_data.get('strength_analysis', {})
            }

            # Проверяем, существует ли уже такой паттерн
            self.cursor.execute(
                'SELECT id FROM patterns WHERE id = ?',
                (pattern_id,)
            )
            existing = self.cursor.fetchone()

            if existing:
                # Обновляем существующий паттерн
                self.cursor.execute('''
                    UPDATE patterns SET
                        entry_price = ?,
                        stop_loss = ?,
                        take_profit = ?,
                        profit_risk_ratio = ?,
                        quality_score = ?,
                        confidence = ?,
                        market_context = ?,
                        metadata = ?,
                        points = ?
                    WHERE id = ?
                ''', (
                    entry_price,
                    stop_loss,
                    take_profit,
                    profit_risk_ratio,
                    quality_score,
                    confidence,
                    market_context,
                    json.dumps(full_metadata),
                    json.dumps(points),
                    pattern_id
                ))
            else:
                # Вставляем новый паттерн
                self.cursor.execute('''
                    INSERT INTO patterns (
                        id, symbol, timeframe, pattern_type, pattern_name,
                        direction, detected_time, entry_price, stop_loss,
                        take_profit, profit_risk_ratio, quality_score,
                        confidence, market_context, metadata, points
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pattern_id,
                    symbol,
                    timeframe,
                    pattern_type,
                    pattern_name,
                    direction,
                    detected_time.isoformat(),
                    entry_price,
                    stop_loss,
                    take_profit,
                    profit_risk_ratio,
                    quality_score,
                    confidence,
                    market_context,
                    json.dumps(full_metadata),
                    json.dumps(points)
                ))

            self.connection.commit()

            # Обновляем статистику
            self._update_pattern_statistics(
                pattern_name, symbol, timeframe, direction,
                quality_score, entry_price, stop_loss, take_profit
            )

            self.logger.debug(f"Паттерн сохранен: {pattern_name} ({pattern_id})")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка сохранения паттерна: {e}")
            self.connection.rollback()
            return False

    def update_pattern_outcome(self,
                               pattern_id: str,
                               outcome: str,
                               realized_profit: Optional[float] = None,
                               realized_risk: Optional[float] = None,
                               holding_period: Optional[float] = None) -> bool:
        """
        Обновление исхода паттерна

        Args:
            pattern_id: ID паттерна
            outcome: Исход ('success', 'failure', 'neutral')
            realized_profit: Реализованная прибыль
            realized_risk: Реализованный риск
            holding_period: Период удержания в часах

        Returns:
            True если успешно обновлено
        """
        try:
            outcome_time = datetime.now()

            self.cursor.execute('''
                UPDATE patterns SET
                    outcome = ?,
                    outcome_time = ?,
                    realized_profit = ?,
                    realized_risk = ?,
                    holding_period = ?
                WHERE id = ?
            ''', (
                outcome,
                outcome_time.isoformat(),
                realized_profit,
                realized_risk,
                holding_period,
                pattern_id
            ))

            self.connection.commit()

            # Получаем данные паттерна для обновления статистики
            pattern = self.get_pattern_by_id(pattern_id)
            if pattern:
                self._update_statistics_with_outcome(pattern, outcome, realized_profit)

            self.logger.info(f"Исход паттерна обновлен: {pattern_id} -> {outcome}")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка обновления исхода: {e}")
            self.connection.rollback()
            return False

    def _update_pattern_statistics(self,
                                   pattern_name: str,
                                   symbol: str,
                                   timeframe: str,
                                   direction: str,
                                   quality_score: float,
                                   entry_price: Optional[float],
                                   stop_loss: Optional[float],
                                   take_profit: Optional[float]):
        """Обновление статистики паттерна"""
        try:
            # Рассчитываем потенциальную прибыль и риск
            potential_profit = 0
            potential_risk = 0

            if entry_price and stop_loss and take_profit:
                if direction == 'bullish':
                    potential_profit = take_profit - entry_price
                    potential_risk = entry_price - stop_loss
                else:
                    potential_profit = entry_price - take_profit
                    potential_risk = stop_loss - entry_price

            # Проверяем существующую статистику
            self.cursor.execute('''
                SELECT total_count, success_count, avg_quality, avg_profit
                FROM pattern_statistics
                WHERE pattern_name = ? AND symbol = ? AND timeframe = ? AND direction = ?
            ''', (pattern_name, symbol, timeframe, direction))

            existing = self.cursor.fetchone()

            if existing:
                # Обновляем существующую статистику
                total_count = existing['total_count'] + 1
                success_count = existing['success_count']
                old_avg_quality = existing['avg_quality']
                old_avg_profit = existing['avg_profit']

                # Новые средние значения
                new_avg_quality = (old_avg_quality * existing['total_count'] + quality_score) / total_count
                new_avg_profit = (old_avg_profit * existing['total_count'] + potential_profit) / total_count

                # Обновляем успешность
                success_rate = success_count / total_count if total_count > 0 else 0

                self.cursor.execute('''
                    UPDATE pattern_statistics SET
                        total_count = ?,
                        avg_quality = ?,
                        avg_profit = ?,
                        avg_risk = ?,
                        success_rate = ?,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE pattern_name = ? AND symbol = ? AND timeframe = ? AND direction = ?
                ''', (
                    total_count,
                    new_avg_quality,
                    new_avg_profit,
                    potential_risk,
                    success_rate,
                    pattern_name, symbol, timeframe, direction
                ))

            else:
                # Создаем новую запись статистики
                self.cursor.execute('''
                    INSERT INTO pattern_statistics (
                        pattern_name, symbol, timeframe, direction,
                        total_count, avg_quality, avg_profit, avg_risk,
                        success_rate
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pattern_name,
                    symbol,
                    timeframe,
                    direction,
                    1,
                    quality_score,
                    potential_profit,
                    potential_risk,
                    0.0
                ))

            self.connection.commit()

        except Exception as e:
            self.logger.error(f"Ошибка обновления статистики: {e}")

    def _update_statistics_with_outcome(self,
                                        pattern: PatternRecord,
                                        outcome: str,
                                        realized_profit: Optional[float]):
        """Обновление статистики с учетом исхода"""
        try:
            is_success = outcome in ['success', 'profit', 'win']

            self.cursor.execute('''
                SELECT success_count, total_count, avg_profit
                FROM pattern_statistics
                WHERE pattern_name = ? AND symbol = ? AND timeframe = ? AND direction = ?
            ''', (
                pattern.pattern_name,
                pattern.symbol,
                pattern.timeframe,
                pattern.direction
            ))

            existing = self.cursor.fetchone()

            if existing:
                total_count = existing['total_count']
                success_count = existing['success_count'] + (1 if is_success else 0)
                success_rate = success_count / total_count if total_count > 0 else 0

                # Обновляем среднюю прибыль
                old_avg_profit = existing['avg_profit']
                realized_profit_val = realized_profit or 0
                new_avg_profit = (old_avg_profit * (total_count - 1) + realized_profit_val) / total_count

                self.cursor.execute('''
                    UPDATE pattern_statistics SET
                        success_count = ?,
                        success_rate = ?,
                        avg_profit = ?,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE pattern_name = ? AND symbol = ? AND timeframe = ? AND direction = ?
                ''', (
                    success_count,
                    success_rate,
                    new_avg_profit,
                    pattern.pattern_name,
                    pattern.symbol,
                    pattern.timeframe,
                    pattern.direction
                ))

                self.connection.commit()

        except Exception as e:
            self.logger.error(f"Ошибка обновления статистики с исходом: {e}")

    def get_pattern_by_id(self, pattern_id: str) -> Optional[PatternRecord]:
        """Получение паттерна по ID"""
        try:
            self.cursor.execute(
                'SELECT * FROM patterns WHERE id = ?',
                (pattern_id,)
            )

            row = self.cursor.fetchone()

            if row:
                return self._row_to_pattern_record(row)

            return None

        except Exception as e:
            self.logger.error(f"Ошибка получения паттерна: {e}")
            return None

    def get_patterns(self,
                     symbol: Optional[str] = None,
                     timeframe: Optional[str] = None,
                     pattern_type: Optional[str] = None,
                     direction: Optional[str] = None,
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None,
                     limit: int = 100) -> List[PatternRecord]:
        """
        Получение паттернов с фильтрами

        Args:
            symbol: Фильтр по символу
            timeframe: Фильтр по таймфрейму
            pattern_type: Фильтр по типу паттерна
            direction: Фильтр по направлению
            start_date: Начальная дата
            end_date: Конечная дата
            limit: Ограничение количества

        Returns:
            Список паттернов
        """
        try:
            query = 'SELECT * FROM patterns WHERE 1=1'
            params = []

            if symbol:
                query += ' AND symbol = ?'
                params.append(symbol)

            if timeframe:
                query += ' AND timeframe = ?'
                params.append(timeframe)

            if pattern_type:
                query += ' AND pattern_type = ?'
                params.append(pattern_type)

            if direction:
                query += ' AND direction = ?'
                params.append(direction)

            if start_date:
                query += ' AND detected_time >= ?'
                params.append(start_date.isoformat())

            if end_date:
                query += ' AND detected_time <= ?'
                params.append(end_date.isoformat())

            query += ' ORDER BY detected_time DESC LIMIT ?'
            params.append(limit)

            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()

            return [self._row_to_pattern_record(row) for row in rows]

        except Exception as e:
            self.logger.error(f"Ошибка получения паттернов: {e}")
            return []

    def get_pattern_statistics(self,
                               pattern_name: Optional[str] = None,
                               symbol: Optional[str] = None,
                               timeframe: Optional[str] = None,
                               direction: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Получение статистики паттернов

        Returns:
            Список статистики
        """
        try:
            query = 'SELECT * FROM pattern_statistics WHERE 1=1'
            params = []

            if pattern_name:
                query += ' AND pattern_name = ?'
                params.append(pattern_name)

            if symbol:
                query += ' AND symbol = ?'
                params.append(symbol)

            if timeframe:
                query += ' AND timeframe = ?'
                params.append(timeframe)

            if direction:
                query += ' AND direction = ?'
                params.append(direction)

            query += ' ORDER BY success_rate DESC'

            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()

            statistics = []
            for row in rows:
                stat = dict(row)
                statistics.append(stat)

            return statistics

        except Exception as e:
            self.logger.error(f"Ошибка получения статистики: {e}")
            return []

    def get_historical_patterns_for_analysis(self,
                                             days_back: int = 30,
                                             min_quality: float = 0.6) -> List[Dict[str, Any]]:
        """
        Получение исторических паттернов для анализа

        Args:
            days_back: Количество дней назад
            min_quality: Минимальное качество

        Returns:
            Список паттернов в формате словарей
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)

            self.cursor.execute('''
                SELECT * FROM patterns 
                WHERE detected_time >= ? 
                AND quality_score >= ?
                AND outcome IS NOT NULL
                ORDER BY detected_time DESC
                LIMIT 1000
            ''', (cutoff_date.isoformat(), min_quality))

            rows = self.cursor.fetchall()

            patterns = []
            for row in rows:
                pattern_record = self._row_to_pattern_record(row)
                patterns.append(pattern_record.to_dict())

            return patterns

        except Exception as e:
            self.logger.error(f"Ошибка получения исторических паттернов: {e}")
            return []

    def save_trading_signal(self,
                            pattern_id: str,
                            signal_data: Dict[str, Any]) -> bool:
        """
        Сохранение торгового сигнала

        Args:
            pattern_id: ID паттерна
            signal_data: Данные сигнала

        Returns:
            True если успешно сохранено
        """
        try:
            signal_id = signal_data.get('id', '')
            symbol = signal_data.get('symbol', '')
            timeframe = signal_data.get('timeframe', '')
            signal_type = signal_data.get('signal_type', 'pattern')
            direction = signal_data.get('direction', 'neutral')

            entry_price = signal_data.get('entry_price')
            stop_loss = signal_data.get('stop_loss')
            take_profit = signal_data.get('take_profit')
            risk_amount = signal_data.get('risk_amount')
            reward_amount = signal_data.get('reward_amount')
            profit_risk_ratio = signal_data.get('profit_risk_ratio')
            confidence = signal_data.get('confidence', 0)

            self.cursor.execute('''
                INSERT INTO trading_signals (
                    id, pattern_id, symbol, timeframe, signal_type,
                    direction, entry_price, stop_loss, take_profit,
                    risk_amount, reward_amount, profit_risk_ratio,
                    confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_id,
                pattern_id,
                symbol,
                timeframe,
                signal_type,
                direction,
                entry_price,
                stop_loss,
                take_profit,
                risk_amount,
                reward_amount,
                profit_risk_ratio,
                confidence
            ))

            self.connection.commit()

            self.logger.info(f"Торговый сигнал сохранен: {signal_id}")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка сохранения торгового сигнала: {e}")
            self.connection.rollback()
            return False

    def _row_to_pattern_record(self, row) -> PatternRecord:
        """Конвертация строки БД в PatternRecord"""
        # Парсим JSON поля
        metadata = json.loads(row['metadata']) if row['metadata'] else {}
        points = json.loads(row['points']) if row['points'] else []

        # Парсим даты
        detected_time = datetime.fromisoformat(row['detected_time'])
        outcome_time = (
            datetime.fromisoformat(row['outcome_time'])
            if row['outcome_time'] else None
        )

        return PatternRecord(
            id=row['id'],
            symbol=row['symbol'],
            timeframe=row['timeframe'],
            pattern_type=row['pattern_type'],
            pattern_name=row['pattern_name'],
            direction=row['direction'],
            detected_time=detected_time,
            entry_price=row['entry_price'],
            stop_loss=row['stop_loss'],
            take_profit=row['take_profit'],
            profit_risk_ratio=row['profit_risk_ratio'],
            quality_score=row['quality_score'],
            confidence=row['confidence'],
            market_context=row['market_context'],
            metadata=metadata,
            points=points,
            outcome=row['outcome'],
            outcome_time=outcome_time,
            realized_profit=row['realized_profit'],
            realized_risk=row['realized_risk'],
            holding_period=row['holding_period']
        )

    def export_to_csv(self, filepath: str,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> bool:
        """
        Экспорт паттернов в CSV

        Args:
            filepath: Путь к файлу
            start_date: Начальная дата
            end_date: Конечная дата

        Returns:
            True если успешно экспортировано
        """
        try:
            # Получаем паттерны
            patterns = self.get_patterns(start_date=start_date, end_date=end_date, limit=10000)

            if not patterns:
                self.logger.warning("Нет данных для экспорта")
                return False

            # Конвертируем в DataFrame
            data = []
            for pattern in patterns:
                row = pattern.to_dict()
                data.append(row)

            df = pd.DataFrame(data)

            # Сохраняем в CSV
            df.to_csv(filepath, index=False, encoding='utf-8')

            self.logger.info(f"Экспортировано {len(patterns)} паттернов в {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка экспорта в CSV: {e}")
            return False

    def cleanup_old_patterns(self, days_to_keep: int = 365) -> int:
        """
        Очистка старых паттернов

        Args:
            days_to_keep: Количество дней для хранения

        Returns:
            Количество удаленных записей
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            self.cursor.execute(
                'DELETE FROM patterns WHERE detected_time < ?',
                (cutoff_date.isoformat(),)
            )

            deleted_count = self.cursor.rowcount
            self.connection.commit()

            if deleted_count > 0:
                self.logger.info(f"Удалено {deleted_count} старых паттернов")

            return deleted_count

        except Exception as e:
            self.logger.error(f"Ошибка очистки старых паттернов: {e}")
            self.connection.rollback()
            return 0

    def get_database_stats(self) -> Dict[str, Any]:
        """Получение статистики базы данных"""
        try:
            stats = {}

            # Количество паттернов
            self.cursor.execute('SELECT COUNT(*) as count FROM patterns')
            stats['total_patterns'] = self.cursor.fetchone()['count']

            # Количество по типам
            self.cursor.execute('''
                SELECT pattern_type, COUNT(*) as count 
                FROM patterns 
                GROUP BY pattern_type
            ''')
            stats['by_type'] = dict(self.cursor.fetchall())

            # Количество по направлениям
            self.cursor.execute('''
                SELECT direction, COUNT(*) as count 
                FROM patterns 
                GROUP BY direction
            ''')
            stats['by_direction'] = dict(self.cursor.fetchall())

            # Количество с исходом
            self.cursor.execute('''
                SELECT outcome, COUNT(*) as count 
                FROM patterns 
                WHERE outcome IS NOT NULL
                GROUP BY outcome
            ''')
            stats['by_outcome'] = dict(self.cursor.fetchall())

            # Последний паттерн
            self.cursor.execute('''
                SELECT MAX(detected_time) as last_pattern 
                FROM patterns
            ''')
            stats['last_pattern_time'] = self.cursor.fetchone()['last_pattern']

            return stats

        except Exception as e:
            self.logger.error(f"Ошибка получения статистики БД: {e}")
            return {}

    def close(self):
        """Закрытие соединения с базой данных"""
        if self.connection:
            self.connection.close()
            self.logger.info("Соединение с базой данных закрыто")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

