"""
Test configuration to avoid issues with main config.
"""

import os

# Test configuration
TEST_CONFIG = {
    'MT5_LOGIN': 12345678,
    'MT5_PASSWORD': 'your_password',
    'MT5_SERVER': 'MetaQuotes-Demo',
    'MT5_TIMEOUT': 30000,

    'SYMBOL': 'EURUSD',
    'TIMEFRAME': 'H1',
    'BARS_TO_LOAD': 100,

    'PATTERN_TYPES': ['candlestick'],
    'MIN_CONFIDENCE': 0.6,

    'INITIAL_CAPITAL': 10000.0,
    'STOP_LOSS_PCT': 0.02,
    'TAKE_PROFIT_PCT': 0.04,
    'POSITION_SIZE_PCT': 0.1,

    'LOG_LEVEL': 'INFO',
    'LOG_FILE': 'logs/test.log',

    'DB_PATH': 'test_patterns.db',
    'CACHE_ENABLED': False,
    'CACHE_DIR': 'test_cache',

    'SERVER_HOST': '127.0.0.1',
    'SERVER_PORT': 8765,

    'VERBOSE': True,
    'DEBUG_MODE': True
}


def setup_test_environment():
    """Setup test environment variables."""
    for key, value in TEST_CONFIG.items():
        if isinstance(value, list):
            os.environ[key] = ','.join(value)
        else:
            os.environ[key] = str(value)

    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('test_cache', exist_ok=True)

    print("✅ Test environment configured")
    return True

