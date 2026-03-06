"""Application configuration for Zero-DTE Options Scalper.

Loading order:
    1. .env              — shared credentials & infrastructure
    2. .env.{mode}       — mode-specific tuning (paper / live)
    3. Real env vars     — final override (Docker, CI)
"""

import os
from decimal import Decimal
from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


def _env_files() -> tuple[str, ...]:
    base = Path(".env")
    mode = os.environ.get("TRADING_MODE", "").lower()
    if not mode and base.exists():
        for line in base.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("TRADING_MODE"):
                _, _, value = stripped.partition("=")
                mode = value.strip().strip('"').strip("'").lower()
                break
    overlay = Path(f".env.{mode}") if mode else None
    if overlay and overlay.exists():
        return (".env", str(overlay))
    return (".env",)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_env_files(),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Alpaca ──────────────────────────────────────────────────
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper: bool = True
    alpaca_data_feed: str = "iex"  # "iex" (free) or "sip" (paid, real-time)

    # ── Underlyings ─────────────────────────────────────────────
    underlyings: str = "SPY,QQQ,IWM"  # Comma-separated

    @property
    def underlying_list(self) -> list[str]:
        return [s.strip() for s in self.underlyings.split(",") if s.strip()]

    # ── Telegram ────────────────────────────────────────────────
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # ── Redis ───────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"

    # ── Trading Mode ────────────────────────────────────────────
    trading_mode: TradingMode = TradingMode.PAPER

    # ── 0DTE Strategy Parameters ────────────────────────────────
    target_delta: float = 0.30        # Strike selection: buy ~30 delta options
    max_contracts_per_trade: int = 10
    min_premium: float = 0.50         # Don't buy options < $0.50
    max_premium: float = 5.00         # Don't buy options > $5.00
    min_spread_ratio: float = 0.90    # bid/ask ratio >= 0.90 (tight spread)

    # ── Timing (all ET) ─────────────────────────────────────────
    entry_start: str = "09:45"        # 15 min after open for vol to settle
    entry_cutoff: str = "14:30"       # Stop new entries 1 hr before cutoff
    hard_close: str = "15:15"         # Close all 15 min before 3:30 cutoff

    # ── Risk ────────────────────────────────────────────────────
    kelly_fraction: Decimal = Field(default=Decimal("0.20"))
    max_position_pct: Decimal = Field(default=Decimal("0.05"))
    max_portfolio_exposure: Decimal = Field(default=Decimal("0.30"))
    daily_drawdown_halt: Decimal = Field(default=Decimal("0.08"))
    signal_confidence_threshold: int = 55
    min_edge: Decimal = Field(default=Decimal("0.03"))

    # Greeks portfolio limits
    max_portfolio_delta: float = 50.0
    max_portfolio_gamma: float = 20.0
    max_portfolio_theta: float = -100.0
    max_portfolio_vega: float = 30.0

    # Per-underlying limits
    max_positions_per_underlying: int = 3
    max_same_strike: int = 1

    # ── Profit-taking / Stop-loss ───────────────────────────────
    pt_profit_target_pct: float = 0.50   # Take profit at 50% premium gain
    sl_stop_loss_pct: float = 0.30       # Stop loss at 30% premium loss
    sl_trailing_pct: float = 0.20        # Trail 20% below peak premium
    pt_time_exit_minutes: int = 15       # Exit all positions 15 min before cutoff

    # ── Quant Layer ─────────────────────────────────────────────
    vix_high_threshold: float = 30.0     # Reduce size when VIX > 30
    vix_low_threshold: float = 12.0      # Increase size when VIX < 12
    vix_crisis_threshold: float = 35.0   # Halt all entries when VIX > 35
    iv_percentile_max: float = 85.0      # Don't buy when IV pctile > 85

    # GEX
    gex_source: str = "squeezemetrics"
    squeezemetrics_api_key: str = ""

    # Macro calendar
    macro_blackout_minutes: int = 60     # Pause trading ±60 min around FOMC/CPI

    # ── Sentiment ───────────────────────────────────────────────
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "zero-dte-scalper/1.0"
    sentiment_weight: float = 0.05       # Weight in ensemble (0-1)

    # ── Technical Indicators ────────────────────────────────────
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0

    # ── Engine ──────────────────────────────────────────────────
    fast_loop_seconds: int = 5           # Exit checking, tick momentum
    quant_loop_seconds: int = 30         # VIX, GEX, sentiment refresh
    strategy_loop_seconds: int = 15      # Entry signal evaluation
    candle_interval_minutes: int = 1     # 1-min candles for intraday
    candle_cache_size: int = 500         # Keep 500 1-min candles (~8 hours)

    # Circuit breaker
    max_consecutive_losses_window: int = 3
    consecutive_loss_window_minutes: int = 30
    circuit_breaker_cooldown_hours: int = 2

    # ── Database ────────────────────────────────────────────────
    sqlite_db_path: str = "data/bot.db"

    # ── Web Dashboard ───────────────────────────────────────────
    web_port: int = 8090
    web_enabled: bool = True

    # ── Signal Ensemble Weights ─────────────────────────────────
    weight_technical: float = 0.25
    weight_tick_momentum: float = 0.20
    weight_gex: float = 0.15
    weight_flow: float = 0.15
    weight_vix: float = 0.10
    weight_internals: float = 0.10
    weight_sentiment: float = 0.05


def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
