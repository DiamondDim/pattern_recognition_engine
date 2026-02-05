#!/usr/bin/env python3
"""
Main entry point for Pattern Recognition Engine.
"""

import sys
import os
import logging

# Настраиваем базовое логирование до импорта модулей
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pattern_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function."""
    try:
        logger.info("=" * 60)
        logger.info("PATTERN RECOGNITION ENGINE - STARTING")
        logger.info("=" * 60)
        
        # Пытаемся импортировать основные модули
        try:
            import config
            logger.info(f"Config loaded: {config.SYMBOL} {config.TIMEFRAME}")
        except ImportError as e:
            logger.error(f"Failed to load config: {e}")
            return
        
        try:
            from utils.mt5_connector import MT5Connector
            mt5 = MT5Connector()
            if mt5.connect():
                logger.info("✅ MT5 connected successfully")
                mt5.disconnect()
            else:
                logger.warning("⚠️  MT5 connection failed, continuing in offline mode")
        except ImportError as e:
            logger.warning(f"MT5 not available: {e}")
        
        try:
            from core.data_feeder import DataFeeder
            feeder = DataFeeder(cache_enabled=True)
            logger.info("✅ DataFeeder initialized")
        except ImportError as e:
            logger.error(f"Failed to load DataFeeder: {e}")
            return
        
        try:
            from core.pattern_detector import PatternDetector
            detector = PatternDetector()
            logger.info("✅ PatternDetector initialized")
        except ImportError as e:
            logger.error(f"Failed to load PatternDetector: {e}")
            return
        
        # Основной цикл работы
        logger.info("\n" + "=" * 60)
        logger.info("SYSTEM READY - Starting pattern recognition")
        logger.info("=" * 60)
        
        # Здесь будет основной код работы
        # Пока просто тестовая заглушка
        logger.info("System initialized successfully!")
        logger.info("Run 'python run.py --demo' for full functionality")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

