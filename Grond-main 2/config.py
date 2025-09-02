"""Configuration constants for the trading system.

This module centralizes environment-driven settings, default values,
and constants used throughout the application. It avoids repeated
lookups and ensures consistent configuration handling.
"""

import os
import pytz

# === API KEYS & TOKENS ===
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# === Service Identity & Modes ===
SERVICE_NAME = os.getenv("SERVICE_NAME", "grond")
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "SCALP")
RISK_MODE = os.getenv("RISK_MODE", "AGGRESSIVE")

# === Trading Parameters ===
ORDER_SIZE = float(os.getenv("ORDER_SIZE", "1.0"))
BANDIT_EPSILON = float(os.getenv("BANDIT_EPSILON", "0.02"))

# === Underlying Symbols ===
TICKERS = [
    "TSLA", "AAPL", "MSFT", "NVDA",
    "NFLX", "AMZN", "META", "GOOG",
    "IBIT", "BABA", "VEX",
]
OPTIONS_TICKERS = TICKERS.copy()

# === Risk-Free Rate & Dividends ===
RISK_FREE_RATE = 0.048
DIVIDEND_YIELDS = {
    "TSLA": 0.00,
    "AAPL": 0.005,
    "MSFT": 0.007,
    "NVDA": 0.001,
    "NFLX": 0.00,
    "AMZN": 0.00,
    "META": 0.002,
    "GOOG": 0.00,
    "IBIT": 0.00,
    "BABA": 0.00,
    "VEX": 0.00,
}

# === Single Fallback Volatility ===
# Used only if realized sigma cannot be computed
DEFAULT_VOLATILITY_FALLBACK = float(
    os.getenv("DEFAULT_VOLATILITY_FALLBACK", "0.2")
)

# === Timezone & Market-Hour Labels ===
tz = pytz.timezone("US/Eastern")
TIME_OF_DAY_LABELS = (
    "PRE_MARKET",
    "MORNING",
    "MIDDAY",
    "AFTERNOON",
    "AFTER_HOURS",
    "OFF_HOURS",
)

# === Rate Limiting (Polygon API) ===
RATE_LIMIT_PER_SEC = 5.0
BURST_CAPACITY_SEC = 10
RATE_LIMIT_PER_MIN = 200.0
BURST_CAPACITY_MIN = 200

# === API & Logging File Paths ===
SNAPSHOT_FILE = "snapshots/options_oi_snapshots.json"
SIGNAL_TRACKER_FILE = "logs/alladin_signal_performance_log.csv"
EXIT_LOG_FILE = "logs/alladin_exit_log.csv"
STATUS_FILE = "logs/alladin_status.txt"

# === Metrics & HTTP Ports ===
METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))
HTTP_PORT = int(os.getenv("HTTP_PORT", "10000"))

# === Telegram Notifications ===
ENABLE_TELEGRAM = True
TELEGRAM_COOLDOWN_SECONDS = 60

# === Strategy & Classifier Parameters ===
MIN_BREAKOUT_PROBABILITY = 0.5
EXIT_BARS = 3

MOVEMENT_CONFIG_FILE = "movement_config.json"
MOVEMENT_LOGIC_CONFIG_FILE = "movement_logic_config.json"

# === Data-Source & Ingestion Settings ===
DATA_SOURCE_MODE = "hybrid"
REST_POLL_INTERVAL = 10
WEBHOOK_INITIAL_DELAY = 300

# === Historical Lookbacks ===
LOOKBACK_BREAKOUT = 5
LOOKBACK_RISK_REWARD = 20
DEFAULT_CANDLE_LIMIT = 500

# === TTL (Time-to-Live) Mappings ===
TTL_MAP = {
    "SHORT": 3,
    "MEDIUM": 10,
    "EXPIRY": 10000,
}

# === Exit-Level Parameters ===
EXIT_PROFIT_TARGET = 0.06
EXIT_STOP_LOSS = -0.035
