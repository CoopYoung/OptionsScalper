"""Configuration for the hybrid Claude + Alpaca trading system.

Loads from .env file. Simple and flat — no nested settings.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env — root .env has real broker keys, hybrid/.env adds extras
# (dotenv override=False by default, so first loaded values win)
_root_env = Path(__file__).parent.parent / ".env"
_hybrid_env = Path(__file__).parent / ".env"
load_dotenv(_root_env)       # root .env has real Alpaca/Telegram keys
load_dotenv(_hybrid_env)     # hybrid/.env adds Finnhub/Public keys (won't override existing)


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
CLAUDE_CODE_MODEL = _get("CLAUDE_CODE_MODEL", "")

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
EXIT_CHECK_SECONDS = _int("EXIT_CHECK_SECONDS", 15)
ENTRY_START_ET = _get("ENTRY_START_ET", "09:45")
ENTRY_CUTOFF_ET = _get("ENTRY_CUTOFF_ET", "15:00")
HARD_CLOSE_ET = _get("HARD_CLOSE_ET", "15:45")
MIN_DTE = _int("MIN_DTE", 0)   # 0 = same-day allowed
MAX_DTE = _int("MAX_DTE", 3)   # Up to 3 DTE

# ── Public.com ─────────────────────────────────────────────
PUBLIC_SECRET_KEY = _get("PUBLIC_SECRET_KEY")

# ── LLM Provider ──────────────────────────────────────────
LLM_PROVIDER = _get("LLM_PROVIDER", "ollama")       # ollama | anthropic | openai
OLLAMA_URL = _get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = _get("OLLAMA_MODEL", "0xroyce/plutus")

# ── OpenAI-compatible (also Groq, Together, etc.) ─────────
OPENAI_API_KEY = _get("OPENAI_API_KEY")
OPENAI_BASE_URL = _get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = _get("OPENAI_MODEL", "gpt-4o-mini")

# ── Exit Management ──────────────────────────────────────
PROFIT_TARGET_PCT = _float("PROFIT_TARGET_PCT", 50.0)
STOP_LOSS_PCT = _float("STOP_LOSS_PCT", 30.0)
TRAILING_STOP_ACTIVATE_PCT = _float("TRAILING_STOP_ACTIVATE_PCT", 30.0)
TRAILING_STOP_PCT = _float("TRAILING_STOP_PCT", 15.0)

# ── Signal Thresholds ────────────────────────────────────
SIGNAL_CONFIDENCE_THRESHOLD = _int("SIGNAL_CONFIDENCE_THRESHOLD", 55)
MAX_OPTIONS_PER_TYPE = _int("MAX_OPTIONS_PER_TYPE", 4)

# ── Logging ─────────────────────────────────────────────────
LOG_DIR = Path(_get("LOG_DIR", str(Path(__file__).parent / "trade_logs")))
LOG_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_LOG = LOG_DIR / "audit.jsonl"
