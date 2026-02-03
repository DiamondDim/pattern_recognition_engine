"""
Модуль работы с базой данных паттернов
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd

from config import config
from utils.logger import logger

# Создание базового класса для моделей
Base = declarative_base()

class PatternRecord(Base):
    """Модель записи паттерна в базе данных"""

    __tablename__ = 'patterns'

    id = Column(Integer, primary_key=True, autoincrement=True)
    pattern_id = Column(String(100), unique=True, index=True)
    name = Column(String(100))
    type = Column(String(50))  # geometric, candlestick, harmonic
    symbol = Column(String(50))
    timeframe = Column(String(10))
    direction = Column(String(20))  # bullish, bearish, neutral
    quality_score = Column(Float)
    confidence_score = Column(Float)
    detected_at = Column(DateTime)
    points = Column(JSON)  # Координаты точек паттерна
    metadata = Column(JSON)  # Дополнительные метаданные
    is_active = Column(Boolean, default=True)
    is_traded = Column(Boolean, default=False)
    trade_result = Column(Float, nullable=True)  # Результат сделки
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            'id': self.id,
            'pattern_id': self.pattern_id,
            'name': self.name,
            'type': self.type,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'direction': self.direction,
            'quality_score': self.quality_score,
            'confidence_score': self.confidence_score,
            'detected_at': self.detected_at.isoformat() if self.detected_at else None,
            'points': self.points,
            'metadata': self.metadata,
            'is_active': self.is_active,
            'is_traded': self.is_traded,
            'trade_result': self.trade_result,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class PatternDatabase:
    """Класс для работы с базой данных паттернов"""

    def __init__(self):
        self.logger = logger.bind(module="pattern_database")
        self.engine = None
        self.SessionLocal = None
        self.initialized = False

    async def initialize(self) -> bool:
        """Инициализация подключения к базе данных"""
        try:
            # Создание URL для подключения
            db_url = config.DATABASE.url

            # Создание движка SQLAlchemy
            self.engine = create_engine(
                db_url,
                echo=config.ENVIRONMENT == "development",
                pool_size=config.DATABASE.POOL_SIZE,
                max_overflow=config.DATABASE.MAX_OVERFLOW,
                pool_pre_ping=True
            )

            # Создание фабрики сессий
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )

            # Создание таблиц если их нет
            Base.metadata.create_all(bind=self.engine)

            # Проверка подключения
            with self.SessionLocal() as session:
                session.execute("SELECT 1")

            self.initialized = True
            self.logger.info(f"База данных инициализирована: {db_url}")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка инициализации базы данных: {e}")
            return False

    def get_session(self) -> Session:
        """Получение сессии базы данных"""
        if not self.initialized:
            raise RuntimeError("База данных не инициализирована")
        return self.SessionLocal()

    async def save_pattern(self, pattern_data: Dict[str, Any]) -> Optional[str]:
        """
        Сохранение паттерна в базу данных

        Args:
            pattern_data: Данные паттерна

        Returns:
            ID сохраненного паттерна или None
        """
        try:
            # Генерация уникального ID паттерна
            pattern_id = self._generate_pattern_id(pattern_data)

            with self.get_session() as session:
                # Проверяем, существует ли уже такой паттерн
                existing = session.query(PatternRecord).filter_by(
                    pattern_id=pattern_id
                ).first()

                if existing:
                    # Обновляем существующий паттерн
                    existing.quality_score = pattern_data.get('quality_score', existing.quality_score)
                    existing.confidence_score = pattern_data.get('confidence_score', existing.confidence_score)
                    existing.points = pattern_data.get('points', existing.points)
                    existing.metadata = pattern_data.get('metadata', existing.metadata)
                    existing.updated_at = datetime.now()

                    self.logger.debug(f"Паттерн обновлен: {pattern_id}")

                else:
                    # Создаем новый паттерн
                    pattern = PatternRecord(
                        pattern_id=pattern_id,
                        name=pattern_data.get('name', 'unknown'),
                        type=pattern_data.get('type', 'unknown'),
                        symbol=pattern_data.get('symbol', 'UNKNOWN'),
                        timeframe=pattern_data.get('timeframe', 'H1'),
                        direction=pattern_data.get('direction', 'neutral'),
                        quality_score=pattern_data.get('quality_score', 0.5),
                        confidence_score=pattern_data.get('confidence_score', 0.5),
                        detected_at=pattern_data.get('detected_at', datetime.now()),
                        points=pattern_data.get('points', []),
                        metadata=pattern_data.get('metadata', {}),
                        is_active=True,
                        is_traded=False
                    )

                    session.add(pattern)
                    self.logger.debug(f"Паттерн сохранен: {pattern_id}")

                session.commit()
                return pattern_id

        except SQLAlchemyError as e:
            self.logger.error(f"Ошибка сохранения паттерна: {e}")
            session.rollback()
            return None
        except Exception as e:
            self.logger.error(f"Неожиданная ошибка при сохранении паттерна: {e}")
            return None

    async def save_patterns_batch(self, patterns_data: List[Dict[str, Any]]) -> List[str]:
        """
        Пакетное сохранение паттернов

        Args:
            patterns_data: Список данных паттернов

        Returns:
            Список ID сохраненных паттернов
        """
        saved_ids = []

        try:
            with self.get_session() as session:
                for pattern_data in patterns_data:
                    pattern_id = self._generate_pattern_id(pattern_data)

                    # Проверяем существование
                    existing = session.query(PatternRecord).filter_by(
                        pattern_id=pattern_id
                    ).first()

                    if existing:
                        existing.updated_at = datetime.now()
                    else:
                        pattern = PatternRecord(
                            pattern_id=pattern_id,
                            name=pattern_data.get('name', 'unknown'),
                            type=pattern_data.get('type', 'unknown'),
                            symbol=pattern_data.get('symbol', 'UNKNOWN'),
                            timeframe=pattern_data.get('timeframe', 'H1'),
                            direction=pattern_data.get('direction', 'neutral'),
                            quality_score=pattern_data.get('quality_score', 0.5),
                            confidence_score=pattern_data.get('confidence_score', 0.5),
                            detected_at=pattern_data.get('detected_at', datetime.now()),
                            points=pattern_data.get('points', []),
                            metadata=pattern_data.get('metadata', {}),
                            is_active=True,
                            is_traded=False
                        )
                        session.add(pattern)

                    saved_ids.append(pattern_id)

                session.commit()
                self.logger.info(f"Пакетно сохранено паттернов: {len(saved_ids)}")

        except SQLAlchemyError as e:
            self.logger.error(f"Ошибка пакетного сохранения паттернов: {e}")
            session.rollback()
            return []

        return saved_ids

    async def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение паттерна по ID

        Args:
            pattern_id: ID паттерна

        Returns:
            Данные паттерна или None
        """
        try:
            with self.get_session() as session:
                pattern = session.query(PatternRecord).filter_by(
                    pattern_id=pattern_id
                ).first()

                if pattern:
                    return pattern.to_dict()
                else:
                    return None

        except Exception as e:
            self.logger.error(f"Ошибка получения паттерна {pattern_id}: {e}")
            return None

    async def get_patterns(self,
                         symbol: Optional[str] = None,
                         timeframe: Optional[str] = None,
                         pattern_type: Optional[str] = None,
                         direction: Optional[str] = None,
                         min_quality: float = 0.0,
                         min_confidence: float = 0.0,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         active_only: bool = True,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        Получение паттернов по фильтрам

        Args:
            symbol: Фильтр по символу
            timeframe: Фильтр по таймфрейму
            pattern_type: Фильтр по типу паттерна
            direction: Фильтр по направлению
            min_quality: Минимальный качественный балл
            min_confidence: Минимальная уверенность
            start_date: Начальная дата
            end_date: Конечная дата
            active_only: Только активные паттерны
            limit: Максимальное количество записей

        Returns:
            Список паттернов
        """
        try:
            with self.get_session() as session:
                query = session.query(PatternRecord)

                # Применяем фильтры
                if symbol:
                    query = query.filter_by(symbol=symbol)

                if timeframe:
                    query = query.filter_by(timeframe=timeframe)

                if pattern_type:
                    query = query.filter_by(type=pattern_type)

                if direction:
                    query = query.filter_by(direction=direction)

                if active_only:
                    query = query.filter_by(is_active=True)

                # Фильтры по качеству
                query = query.filter(
                    PatternRecord.quality_score >= min_quality,
                    PatternRecord.confidence_score >= min_confidence
                )

                # Фильтры по дате
                if start_date:
                    query = query.filter(PatternRecord.detected_at >= start_date)

                if end_date:
                    query = query.filter(PatternRecord.detected_at <= end_date)

                # Сортировка и лимит
                patterns = query.order_by(
                    PatternRecord.quality_score.desc(),
                    PatternRecord.detected_at.desc()
                ).limit(limit).all()

                return [p.to_dict() for p in patterns]

        except Exception as e:
            self.logger.error(f"Ошибка получения паттернов: {e}")
            return []

    async def get_pattern_statistics(self,
                                   symbol: Optional[str] = None,
                                   timeframe: Optional[str] = None) -> Dict[str, Any]:
        """
        Получение статистики по паттернам

        Args:
            symbol: Фильтр по символу
            timeframe: Фильтр по таймфрейму

        Returns:
            Статистика паттернов
        """
        try:
            with self.get_session() as session:
                query = session.query(PatternRecord)

                if symbol:
                    query = query.filter_by(symbol=symbol)

                if timeframe:
                    query = query.filter_by(timeframe=timeframe)

                # Общая статистика
                total_count = query.count()
                active_count = query.filter_by(is_active=True).count()
                traded_count = query.filter_by(is_traded=True).count()

                # Статистика по типам
                type_stats = {}
                for pattern_type in session.query(PatternRecord.type).distinct():
                    if pattern_type[0]:
                        count = query.filter_by(type=pattern_type[0]).count()
                        type_stats[pattern_type[0]] = count

                # Статистика по направлениям
                direction_stats = {}
                for direction in session.query(PatternRecord.direction).distinct():
                    if direction[0]:
                        count = query.filter_by(direction=direction[0]).count()
                        direction_stats[direction[0]] = count

                # Средние значения
                avg_quality = session.query(
                    func.avg(PatternRecord.quality_score)
                ).scalar() or 0

                avg_confidence = session.query(
                    func.avg(PatternRecord.confidence_score)
                ).scalar() or 0

                # Статистика торгов
                profitable_trades = query.filter(
                    PatternRecord.is_traded == True,
                    PatternRecord.trade_result > 0
                ).count()

                losing_trades = query.filter(
                    PatternRecord.is_traded == True,
                    PatternRecord.trade_result <= 0
                ).count()

                avg_trade_result = session.query(
                    func.avg(PatternRecord.trade_result)
                ).filter(PatternRecord.is_traded == True).scalar() or 0

                return {
                    'total_patterns': total_count,
                    'active_patterns': active_count,
                    'traded_patterns': traded_count,
                    'by_type': type_stats,
                    'by_direction': direction_stats,
                    'avg_quality_score': float(avg_quality),
                    'avg_confidence_score': float(avg_confidence),
                    'trade_statistics': {
                        'profitable_trades': profitable_trades,
                        'losing_trades': losing_trades,
                        'total_trades': traded_count,
                        'win_rate': profitable_trades / traded_count if traded_count > 0 else 0,
                        'avg_trade_result': float(avg_trade_result)
                    },
                    'timestamp': datetime.now().isoformat()
                }

        except Exception as e:
            self.logger.error(f"Ошибка получения статистики: {e}")
            return {}

    async def update_pattern_status(self,
                                  pattern_id: str,
                                  is_active: Optional[bool] = None,
                                  is_traded: Optional[bool] = None,
                                  trade_result: Optional[float] = None) -> bool:
        """
        Обновление статуса паттерна

        Args:
            pattern_id: ID паттерна
            is_active: Активен ли паттерн
            is_traded: Совершена ли сделка
            trade_result: Результат сделки

        Returns:
            Успешность обновления
        """
        try:
            with self.get_session() as session:
                pattern = session.query(PatternRecord).filter_by(
                    pattern_id=pattern_id
                ).first()

                if not pattern:
                    self.logger.warning(f"Паттерн не найден: {pattern_id}")
                    return False

                if is_active is not None:
                    pattern.is_active = is_active

                if is_traded is not None:
                    pattern.is_traded = is_traded

                if trade_result is not None:
                    pattern.trade_result = trade_result

                pattern.updated_at = datetime.now()
                session.commit()

                self.logger.debug(f"Статус паттерна обновлен: {pattern_id}")
                return True

        except Exception as e:
            self.logger.error(f"Ошибка обновления статуса паттерна: {e}")
            session.rollback()
            return False

    async def delete_old_patterns(self, days_old: int = 30) -> int:
        """
        Удаление старых паттернов

        Args:
            days_old: Удалять паттерны старше N дней

        Returns:
            Количество удаленных паттернов
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)

            with self.get_session() as session:
                deleted_count = session.query(PatternRecord).filter(
                    PatternRecord.detected_at < cutoff_date,
                    PatternRecord.is_traded == False
                ).delete()

                session.commit()

                self.logger.info(f"Удалено старых паттернов: {deleted_count}")
                return deleted_count

        except Exception as e:
            self.logger.error(f"Ошибка удаления старых паттернов: {e}")
            session.rollback()
            return 0

    async def export_to_csv(self, filepath: str, filters: Dict[str, Any] = None) -> bool:
        """
        Экспорт паттернов в CSV файл

        Args:
            filepath: Путь к файлу
            filters: Фильтры для экспорта

        Returns:
            Успешность экспорта
        """
        try:
            # Получаем данные с фильтрами
            patterns = await self.get_patterns(
                symbol=filters.get('symbol') if filters else None,
                timeframe=filters.get('timeframe') if filters else None,
                pattern_type=filters.get('type') if filters else None,
                min_quality=filters.get('min_quality', 0) if filters else 0,
                limit=filters.get('limit', 1000) if filters else 1000
            )

            if not patterns:
                self.logger.warning("Нет данных для экспорта")
                return False

            # Конвертируем в DataFrame
            df = pd.DataFrame(patterns)

            # Сохраняем в CSV
            df.to_csv(filepath, index=False, encoding='utf-8')

            self.logger.info(f"Экспортировано паттернов в CSV: {len(patterns)}")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка экспорта в CSV: {e}")
            return False

    async def import_from_csv(self, filepath: str) -> int:
        """
        Импорт паттернов из CSV файла

        Args:
            filepath: Путь к файлу

        Returns:
            Количество импортированных паттернов
        """
        try:
            # Читаем CSV
            df = pd.read_csv(filepath)

            # Конвертируем в список словарей
            patterns_data = df.to_dict('records')

            # Сохраняем в базу
            imported_count = len(await self.save_patterns_batch(patterns_data))

            self.logger.info(f"Импортировано паттернов из CSV: {imported_count}")
            return imported_count

        except Exception as e:
            self.logger.error(f"Ошибка импорта из CSV: {e}")
            return 0

    async def backup_database(self, backup_dir: Optional[str] = None) -> str:
        """
        Создание резервной копии базы данных

        Args:
            backup_dir: Директория для бэкапа

        Returns:
            Путь к файлу бэкапа
        """
        try:
            if backup_dir is None:
                backup_dir = Path(config.PATHS["database_dir"]) / "backups"

            backup_dir = Path(backup_dir)
            backup_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"patterns_backup_{timestamp}.db"

            if config.DATABASE.TYPE == "sqlite":
                # Копируем SQLite файл
                import shutil
                source_file = Path(config.DATABASE.NAME)
                shutil.copy2(source_file, backup_file)
            else:
                # Для других БД экспортируем данные
                patterns = await self.get_patterns(limit=10000)
                with open(backup_file, 'w', encoding='utf-8') as f:
                    json.dump(patterns, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Создана резервная копия: {backup_file}")
            return str(backup_file)

        except Exception as e:
            self.logger.error(f"Ошибка создания резервной копии: {e}")
            return ""

    async def close(self):
        """Закрытие соединения с базой данных"""
        if self.engine:
            self.engine.dispose()
            self.logger.info("Соединение с базой данных закрыто")

    def _generate_pattern_id(self, pattern_data: Dict[str, Any]) -> str:
        """Генерация уникального ID паттерна"""
        symbol = pattern_data.get('symbol', 'UNKNOWN')
        pattern_type = pattern_data.get('type', 'unknown')
        pattern_name = pattern_data.get('name', 'unknown')
        detected_at = pattern_data.get('detected_at', datetime.now())

        if isinstance(detected_at, str):
            timestamp = detected_at
        else:
            timestamp = detected_at.strftime("%Y%m%d_%H%M%S")

        return f"{symbol}_{pattern_type}_{pattern_name}_{timestamp}"

