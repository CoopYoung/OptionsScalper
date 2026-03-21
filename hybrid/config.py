"""Configuration for the hybrid Claude + Alpaca trading system.

Loads from .env file. Simple and flat — no nested settings.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)


def _get(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _float(key: str, default: float = 0.0) -> float:
    return float(_get(key, str(default)))


def _int(key: str, default: int = 0) -> int:
    return int(_get(key, str(default)))


def _bool(key: str, default: bool = False) -> bool:
    return _get(key, str(default)).lower() in ("true", "1", "yes")


# ── Alpaca ──────────────────────────────────────────────────
ALPACA_API_KEY = _get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = _get("ALPACA_SECRET_KEY")
ALPACA_PAPER = _bool("ALPACA_PAPER", True)
ALPACA_BASE_URL = (
    "https://paper-api.alpaca.markets"
    if ALPACA_PAPER
    else "https://api.alpaca.markets"
)
ALPACA_DATA_URL = "https://data.alpaca.markets"

# ── Claude API ──────────────────────────────────────────────
ANTHROPIC_API_KEY = _get("ANTHROPIC_API_KEY")
CLAUDE_MODEL = _get("CLAUDE_MODEL", "claude-sonnet-4-20250514")

# ── Telegram ────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = _get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = _get("TELEGRAM_CHAT_ID")

# ── Trading Rules (hard limits — validator enforces) ────────
UNDERLYINGS = _get("UNDERLYINGS", "SPY,QQQ,IWM").split(",")
MAX_RISK_PER_TRADE = _float("MAX_RISK_PER_TRADE", 150.0)    # Max $ loss per spread
MAX_DAILY_LOSS = _float("MAX_DAILY_LOSS", 500.0)             # Stop trading after this
MAX_CONCURRENT_POSITIONS = _int("MAX_CONCURRENT_POSITIONS", 3)
MAX_CONTRACTS_PER_TRADE = _int("MAX_CONTRACTS_PER_TRADE", 5)
MIN_REWARD_RISK_RATIO = _float("MIN_REWARD_RISK_RATIO", 1.5)
ALLOWED_STRATEGIES = [
    "bull_put_spread", "bear_call_spread",
    "iron_condor", "long_call", "long_put",
    "call_debit_spread", "put_debit_spread",
]

# ── Timing ──────────────────────────────────────────────────
CRON_INTERVAL_MINUTES = _int("CRON_INTERVAL_MINUTES", 10)
ENTRY_START_ET = _get("ENTRY_START_ET", "09:45")
ENTRY_CUTOFF_ET = _get("ENTRY_CUTOFF_ET", "15:00")
HARD_CLOSE_ET = _get("HARD_CLOSE_ET", "15:45")
MIN_DTE = _int("MIN_DTE", 0)   # 0 = same-day allowed
MAX_DTE = _int("MAX_DTE", 3)   # Up to 3 DTE

# ── Logging ─────────────────────────────────────────────────
LOG_DIR = Path(_get("LOG_DIR", str(Path(__file__).parent / "trade_logs")))
LOG_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_LOG = LOG_DIR / "audit.jsonl"
