# CLAUDE.md — Zero-DTE Options Scalper

You are an expert quantitative trading engineer building a production-grade 0DTE (zero days to expiration) options scalping bot. The bot trades **SPY/QQQ/IWM** options on **Alpaca**, using a 7-factor weighted signal ensemble with quant-level analysis (VIX regime, GEX, options flow, sentiment, macro calendar, market internals).

## Project Overview

- **What it does**: Scalps same-day expiration options on major ETFs during market hours (9:45 AM - 3:15 PM ET)
- **Broker**: Alpaca (REST + WebSocket, paper + live modes)
- **Architecture**: Async Python, 3 concurrent loops (fast/quant/strategy), Redis cross-asset consensus, SQLite persistence
- **Status**: Core trading system built, not yet battle-tested in paper trading. Needs tests, dashboard, and tuning.

## Architecture

```
src/
├── core/
│   ├── __main__.py              # Entry point
│   └── engine.py                # TradingEngine: 3 async loops + chain refresh
├── data/
│   ├── alpaca_client.py         # REST client (chain, orders, positions, account)
│   ├── alpaca_stream.py         # WebSocket streams (equity ticks, option quotes) + TickMomentum
│   ├── cache.py                 # In-memory + Redis cache, cross-asset tick momentum
│   ├── options_chain.py         # Chain fetching, strike selection, Greeks scoring
│   └── trade_db.py              # SQLite (trades, portfolio_state, open_positions)
├── quant/
│   ├── vix.py                   # VIX regime (low/normal/high/crisis), IV percentile, RV-IV spread
│   ├── gex.py                   # Gamma Exposure levels (support/resistance from dealer hedging)
│   ├── flow.py                  # Put/call ratio, unusual activity, smart money bias
│   ├── sentiment.py             # CNN F&G + Reddit FinBERT + news headlines
│   ├── macro.py                 # Economic calendar gate (FOMC/CPI/NFP blackouts)
│   └── internals.py             # NYSE TICK, A/D ratio, VWAP, cumulative delta
├── strategy/
│   ├── base.py                  # TradeDirection, OptionsContract, TradeSignal
│   ├── signals.py               # RSI, MACD, Bollinger, Volume Delta
│   └── zero_dte.py              # Master ensemble: 7 factors + gate checks → signal
├── risk/
│   ├── manager.py               # Kelly sizing, Greeks limits, PDT tracker, exit logic
│   └── circuit_breaker.py       # Drawdown halt, consecutive loss cooldown
├── infra/
│   ├── config.py                # Pydantic Settings (.env → .env.{mode} → env vars)
│   ├── logger.py                # JSON structured logging
│   └── alerts.py                # Telegram notifications
└── web/                         # Dashboard (not yet implemented)
```

## Signal Ensemble

The strategy computes a weighted confidence score (0-100) from 7 factors:

| Factor                 | Weight | Source                  |
|------------------------|--------|-------------------------|
| Technical momentum     | 25%    | RSI, MACD, BB           |
| Tick momentum + ROC    | 20%    | Price feed ticks        |
| GEX regime + levels    | 15%    | `quant/gex.py`          |
| Options flow           | 15%    | `quant/flow.py`         |
| VIX regime + IV pctile | 10%    | `quant/vix.py`          |
| Market internals       | 10%    | `quant/internals.py`    |
| Sentiment (contrarian) | 5%     | `quant/sentiment.py`    |

Gate checks (all must pass): macro blackout, time window, IV percentile, VIX crisis, spread quality, Greeks room, PDT budget.

## Engine Loops

- **Fast loop (5s)**: Check exits (profit target/stop loss/trailing/time), poll option quotes, publish tick momentum to Redis
- **Quant loop (30s)**: Refresh VIX, GEX, flow, internals, macro. Sentiment every ~2 min (FinBERT is slow)
- **Strategy loop (15s)**: For each underlying, compute technicals + ensemble → generate TradeSignal → risk check → place order
- **Chain refresh (5m)**: Re-fetch options chains + snapshots from Alpaca

## Key Alpaca Constraints

- **No IOC orders** — only `day` or `gtc` time-in-force for options
- **No order book depth** — Level 1 only (top-of-book bid/ask)
- **Data feed**: `iex` (free, 15-min delayed equities but real-time options) or `sip` (paid, real-time everything)
- **0DTE cutoff**: Alpaca auto-liquidates at 3:30 PM ET; we hard-close at 3:15 PM
- **PDT rule**: 4+ round trips in 5 days requires $25k equity

## Coding Standards

- Python 3.12+, type hints on all functions
- Async where it matters (I/O bound: API calls, WebSocket, Redis)
- `Decimal` for all monetary values and prices
- Pydantic Settings for configuration (never hardcode credentials)
- Structured JSON logging with contextual fields
- No unnecessary abstractions — keep it direct

## Configuration

Settings load from `.env` → `.env.{TRADING_MODE}` → environment variables. Key parameters:

```
# Alpaca
ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER=true

# Strategy
TARGET_DELTA=0.30, MIN_PREMIUM=0.50, MAX_PREMIUM=5.00, MIN_SPREAD_RATIO=0.90
ENTRY_START=09:45, ENTRY_CUTOFF=14:30, HARD_CLOSE=15:15
SIGNAL_CONFIDENCE_THRESHOLD=55

# Risk
KELLY_FRACTION=0.20, MAX_POSITION_PCT=0.05, DAILY_DRAWDOWN_HALT=0.08
MAX_PORTFOLIO_DELTA=50.0, MAX_PORTFOLIO_GAMMA=20.0

# Ensemble Weights
WEIGHT_TECHNICAL=0.25, WEIGHT_TICK_MOMENTUM=0.20, WEIGHT_GEX=0.15
WEIGHT_FLOW=0.15, WEIGHT_VIX=0.10, WEIGHT_INTERNALS=0.10, WEIGHT_SENTIMENT=0.05
```

## Running

```bash
# Paper trading
docker compose up -d

# Direct (development)
pip install -r requirements.txt
python -m src.core
```

## What Needs Work

1. **Tests** — Empty test suite. Needs unit tests for: signal scoring, Kelly sizing, Greeks limits, PDT tracker, gate checks, strike selection
2. **Web dashboard** — `src/web/` is a placeholder. Needs: prices, options chain, quant signals panel, positions, risk gauges, activity log
3. **Paper trading validation** — Run 4+ weeks on paper, track win rate, Sharpe, max drawdown, gate effectiveness
4. **Backtesting** — Historical 0DTE data replay with slippage model
5. **Order management** — Cancel-replace flow for unfilled orders (no IOC on Alpaca)
6. **Quant data sources** — Squeezemetrics API integration for real GEX data, Intrinio for unusual flow
7. **Signal tuning** — Validate ensemble weights through backtesting, adjust for real market conditions

## When Making Changes

1. Read the relevant code first — understand existing patterns before modifying
2. Run syntax checks: `python -m py_compile src/path/to/file.py`
3. Keep the modular structure — each quant module is independent, strategy combines them
4. Risk management is non-negotiable — never bypass Greeks limits, PDT tracking, or circuit breaker
5. Prefer editing existing files over creating new ones
6. Update this file if architecture changes
