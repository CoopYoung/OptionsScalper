# CLAUDE.md — Zero-DTE Options Scalper

You are an expert quantitative trading engineer building a production-grade 0DTE (zero days to expiration) options scalping bot. The bot trades **SPY/QQQ/IWM** options using a local LLM (Ollama/Plutus) as the decision brain, with Python handling all data gathering, signal computation, and trade execution.

## Project Overview

- **What it does**: Scalps same-day expiration options on major ETFs during market hours (9:45 AM - 3:15 PM ET)
- **Brokers**: Alpaca (paper trading) + Public.com (live trading) — dual-broker architecture
- **Brain**: Local Ollama model (0xroyce/plutus, Llama 3.1 8B fine-tuned on 394 finance books) running on Orange Pi 6 Plus. Python pre-digests all data into a single prompt; model outputs one JSON decision. Easy to swap to a frontier API (Claude/GPT/Gemini) later.
- **Architecture**: Python orchestrator (cron every 10 min) → gather data → compute signals → format digest → Ollama inference → validate → execute
- **Status**: Migrating from `claude -p` subprocess architecture to direct Ollama/API integration. Paper testing phase.

## Active Architecture (v2 — Local LLM)

```
┌────────────────────────────────────────────────┐
│  Cron (every 10 min, market hours)             │
├────────────────────────────────────────────────┤
│  Python orchestrator                            │
│  ├── Fetch quotes + chain + Greeks              │
│  │   ├── Alpaca (paper mode)                   │
│  │   └── Public.com (live mode)                │
│  ├── Compute RSI, MACD, Bollinger, VWAP        │
│  ├── Get VIX, F&G, news (yfinance, CNN, etc.)  │
│  ├── Format single digest prompt                │
│  ├── POST to Ollama (Plutus) ← one call        │
│  ├── Parse JSON response                        │
│  ├── Validate (risk checks in Python)           │
│  └── Execute via broker API                     │
├────────────────────────────────────────────────┤
│  Broker: Alpaca (paper) / Public.com (live)    │
│  Data:   Public.com (Greeks, indices)           │
│          yfinance, CNN F&G, Finnhub             │
│  Brain:  Ollama Plutus (local, $0/month)        │
│          ↑ swap to Claude/GPT API for prod      │
└────────────────────────────────────────────────┘
```

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
│   ├── sentiment.py             # CNN F&G + X/Twitter FinBERT + news headlines
│   ├── macro.py                 # Economic calendar gate (FOMC/CPI/NFP blackouts)
│   ├── internals.py             # NYSE TICK, A/D ratio, VWAP, cumulative delta
│   └── optionsai.py             # OptionsAI: IV skew, expected move, AI strategy bias, earnings
├── strategy/
│   ├── base.py                  # TradeDirection, OptionsContract, TradeSignal
│   ├── signals.py               # RSI, MACD, Bollinger, Volume Delta
│   └── zero_dte.py              # Master ensemble: 8 factors + gate checks → signal
├── risk/
│   ├── manager.py               # Kelly sizing, Greeks limits, PDT tracker, exit logic
│   └── circuit_breaker.py       # Drawdown halt, consecutive loss cooldown
├── infra/
│   ├── config.py                # Pydantic Settings (.env → .env.{mode} → env vars)
│   ├── logger.py                # JSON structured logging
│   └── alerts.py                # Telegram notifications
└── web/
    └── app.py                   # Aiohttp dashboard: SSE, API, dark-themed HTML
```

## Signal Ensemble

The strategy computes a weighted confidence score (0-100) from 8 factors:

| Factor                 | Weight | Source                  |
|------------------------|--------|-------------------------|
| Technical momentum     | 22%    | RSI, MACD, BB           |
| Tick momentum + ROC    | 18%    | Price feed ticks        |
| GEX regime + levels    | 13%    | `quant/gex.py`          |
| Options flow           | 14%    | `quant/flow.py`         |
| OptionsAI              | 10%    | `quant/optionsai.py`    |
| VIX regime + IV pctile | 8%     | `quant/vix.py`          |
| Market internals       | 10%    | `quant/internals.py`    |
| Sentiment (contrarian) | 5%     | `quant/sentiment.py`    |

Gate checks (all must pass): macro blackout, earnings blackout (per-underlying), time window, IV percentile, VIX crisis, spread quality, Greeks room, PDT budget.

## Engine Loops

- **Fast loop (5s)**: Check exits (profit target/stop loss/trailing/time), poll option quotes, publish tick momentum to Redis
- **Quant loop (30s)**: Refresh VIX, GEX, flow, internals, macro, OptionsAI. Sentiment every ~2 min (FinBERT is slow)
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

# Ensemble Weights (8 factors, sum to 1.0)
WEIGHT_TECHNICAL=0.22, WEIGHT_TICK_MOMENTUM=0.18, WEIGHT_GEX=0.13
WEIGHT_FLOW=0.14, WEIGHT_OPTIONSAI=0.10, WEIGHT_VIX=0.08
WEIGHT_INTERNALS=0.10, WEIGHT_SENTIMENT=0.05
```

## Broker Details

### Alpaca (paper trading)
- REST + WebSocket, real-time options quotes
- No IOC orders — only `day` or `gtc` for options
- 0DTE cutoff: auto-liquidates at 3:30 PM ET; we hard-close at 3:15 PM
- PDT rule: 4+ round trips in 5 days requires $25k equity

### Public.com (live trading)
- REST only (no WebSocket), 10 req/s rate limit
- Commission-free options + rebates ($0.06-$0.18/contract)
- Python SDK: `publicdotcom-py`
- MCP server available: `publicdotcom-mcp-server`
- No paper trading environment — use Alpaca for testing
- Greeks endpoint: batch up to 250 contracts with IV

## LLM Integration

### Local (default): Ollama + Plutus
- Model: `0xroyce/plutus` (Llama 3.1 8B, fine-tuned on finance books)
- Hardware: Orange Pi 6 Plus (RK3588, 22.59 tok/s prompt eval, ~5 tok/s generation)
- Endpoint: `http://localhost:11434/api/generate`
- Python does ALL data gathering and signal computation
- Model receives ONE pre-digested prompt, returns ONE JSON decision
- No tool calling — keep it simple for 8B model

### Cloud (upgrade path): Anthropic API
- Swap `OLLAMA_URL` for `ANTHROPIC_API_KEY` in config
- Claude Haiku: ~$8/month for 39 calls/day
- Claude Sonnet: ~$60/month (recommended for live money)
- Can use tool calling for more autonomous operation

## Running

```bash
# Paper trading (Alpaca + local Ollama)
python -m hybrid.orchestrator --mode paper

# Live trading (Public.com + local Ollama)
python -m hybrid.orchestrator --mode live

# With cloud LLM instead
ANTHROPIC_API_KEY=sk-... python -m hybrid.orchestrator --mode paper --llm anthropic

# Legacy (original async architecture)
pip install -r requirements.txt
python -m src.core
```

## What Needs Work

1. **Build v2 orchestrator** — Python gathers data, formats digest, calls Ollama, validates, executes
2. **Paper trading validation** — Run 4+ weeks on Alpaca paper, track win rate, Sharpe, max drawdown
3. **Backtesting** — Historical data replay using same digest→Ollama→decision pipeline
4. **Public.com live integration** — Test order flow, handle 10 req/s limit, confirm 0DTE support
5. **Model comparison** — Log Plutus decisions vs what Claude/Sonnet would decide on same data
6. **Signal tuning** — Validate ensemble weights through paper trading results

## When Making Changes

1. Read the relevant code first — understand existing patterns before modifying
2. Run syntax checks: `python -m py_compile src/path/to/file.py`
3. Keep the modular structure — each quant module is independent, strategy combines them
4. Risk management is non-negotiable — never bypass Greeks limits, PDT tracking, or circuit breaker
5. Prefer editing existing files over creating new ones
6. Update this file if architecture changes
