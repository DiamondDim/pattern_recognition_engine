"""
Configuration module for Pattern Recognition Engine.
Uses environment variables with fallback to defaults.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MT5 Configuration
MT5_LOGIN = int(os.getenv("MT5_LOGIN", 12345678))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "your_password")
MT5_SERVER = os.getenv("MT5_SERVER", "MetaQuotes-Demo")
MT5_TIMEOUT = int(os.getenv("MT5_TIMEOUT", 30000))

# Project Settings
SYMBOL = os.getenv("SYMBOL", "EURUSD")
TIMEFRAME = os.getenv("TIMEFRAME", "H1")
BARS_TO_LOAD = int(os.getenv("BARS_TO_LOAD", 1000))

# Pattern Recognition
PATTERN_TYPES = os.getenv("PATTERN_TYPES", "candlestick,geometric,harmonic").split(",")
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", 0.6))

# Backtesting
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", 10000.0))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", 0.02))
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", 0.04))
POSITION_SIZE_PCT = float(os.getenv("POSITION_SIZE_PCT", 0.1))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/pattern_engine.log")

# Database
DB_PATH = os.getenv("DB_PATH", "patterns.db")
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "True") == "True"
CACHE_DIR = os.getenv("CACHE_DIR", "data_cache")

# Server Configuration
SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT", 8765))

# Additional Settings
VERBOSE = os.getenv("VERBOSE", "False") == "True"
DEBUG_MODE = os.getenv("DEBUG_MODE", "False") == "True"

# Timeframes mapping for MT5
TIMEFRAME_MAP = {
    'M1': 1,
    'M5': 5,
    'M15': 15,
    'M30': 30,
    'H1': 60,
    'H4': 240,
    'D1': 1440,
    'W1': 10080,
    'MN1': 43200
}

def get_timeframe_mt5(timeframe_str):
    """Convert timeframe string to MT5 constant."""
    from MetaTrader5 import (
        TIMEFRAME_M1, TIMEFRAME_M5, TIMEFRAME_M15, TIMEFRAME_M30,
        TIMEFRAME_H1, TIMEFRAME_H4, TIMEFRAME_D1, TIMEFRAME_W1, TIMEFRAME_MN1
    )

    timeframe_map = {
        'M1': TIMEFRAME_M1,
        'M5': TIMEFRAME_M5,
        'M15': TIMEFRAME_M15,
        'M30': TIMEFRAME_M30,
        'H1': TIMEFRAME_H1,
        'H4': TIMEFRAME_H4,
        'D1': TIMEFRAME_D1,
        'W1': TIMEFRAME_W1,
        'MN1': TIMEFRAME_MN1
    }

    return timeframe_map.get(timeframe_str.upper(), TIMEFRAME_H1)

def get_config_summary():
    """Return a summary of the current configuration."""
    return {
        "mt5_login": MT5_LOGIN,
        "mt5_server": MT5_SERVER,
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "bars_to_load": BARS_TO_LOAD,
        "pattern_types": PATTERN_TYPES,
        "min_confidence": MIN_CONFIDENCE,
        "initial_capital": INITIAL_CAPITAL,
        "cache_enabled": CACHE_ENABLED,
        "db_path": DB_PATH
    }

