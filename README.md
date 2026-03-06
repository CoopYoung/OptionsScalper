# OptionsScalper

Automated zero-DTE options scalping bot for SPY/QQQ/IWM on Alpaca. Uses a 7-factor quant signal ensemble with VIX regime detection, gamma exposure analysis, options flow, market sentiment, and macro calendar awareness.

## How It Works

The bot runs 3 concurrent async loops during market hours:

- **Fast loop (5s)** — Monitors open positions for exit conditions (profit target, stop loss, trailing stop, time-based close)
- **Quant loop (30s)** — Refreshes market signals: VIX regime, GEX levels, options flow, market internals, sentiment, macro calendar
- **Strategy loop (15s)** — Evaluates entry signals using a weighted ensemble of 7 factors, selects optimal strikes, and places orders through Alpaca

### Signal Ensemble

| Factor | Weight | What It Measures |
|--------|--------|------------------|
| Technical | 25% | RSI, MACD, Bollinger Bands, volume delta |
| Tick Momentum | 20% | Short-term price direction and acceleration |
| GEX Regime | 15% | Dealer hedging flow — mean-reverting vs trending |
| Options Flow | 15% | Put/call ratio, unusual activity, smart money |
| VIX / IV | 10% | Volatility regime, IV percentile, RV-IV spread |
| Market Internals | 10% | NYSE TICK, advance/decline, VWAP deviation |
| Sentiment | 5% | CNN Fear & Greed, Reddit, news (contrarian) |

Entries require passing gate checks: macro blackout, time window, IV percentile, VIX crisis, spread quality, portfolio Greeks limits, and PDT budget.

### Risk Management

- **Kelly criterion sizing** with VIX-adjusted multiplier
- **Portfolio Greeks limits** — delta, gamma, theta, vega caps
- **PDT rule tracking** — 3 day-trade limit under $25k
- **Circuit breaker** — auto-halt on drawdown or consecutive losses
- **Hard close** — all positions closed at 3:15 PM ET (before Alpaca's 3:30 PM 0DTE cutoff)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your Alpaca API keys

# 3. Run (paper trading)
docker compose up -d
```

See [INSTRUCTIONS.md](INSTRUCTIONS.md) for full setup, configuration reference, and troubleshooting.

## Project Structure

```
src/
├── core/engine.py          # Trading engine (3 async loops)
├── data/                   # Alpaca client, WebSocket streams, chain manager, cache, DB
├── quant/                  # VIX, GEX, flow, sentiment, macro calendar, market internals
├── strategy/               # Technical signals, 0DTE ensemble strategy
├── risk/                   # Kelly sizing, Greeks limits, PDT tracker, circuit breaker
└── infra/                  # Config, logging, Telegram alerts
```

## Requirements

- Python 3.12+
- Docker & Docker Compose
- Alpaca account (paper trading is free)
- Redis (included in Docker Compose)

## Status

Core trading system is built. Still needed:
- [ ] Unit & integration tests
- [ ] Web dashboard (real-time quant signals, positions, risk gauges)
- [ ] Paper trading validation (4+ weeks)
- [ ] Backtesting framework with historical 0DTE data
- [ ] Order management improvements (cancel-replace for unfilled orders)
