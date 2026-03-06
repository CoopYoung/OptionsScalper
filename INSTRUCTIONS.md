# Instructions — Zero-DTE Options Scalper Setup & Operation

## Prerequisites

- Python 3.12+
- Docker & Docker Compose
- An Alpaca account (paper trading is free: https://alpaca.markets)
- Redis (included in Docker Compose, or install locally)

## Initial Setup

### 1. Clone and Install

```bash
cd ~/git-projects/OptionsScalper
pip install -r requirements.txt
```

### 2. Create Environment Files

Copy the example and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your keys:

```env
# Required
ALPACA_API_KEY=your_paper_api_key
ALPACA_SECRET_KEY=your_paper_secret_key
ALPACA_PAPER=true

# Optional — Telegram alerts
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Optional — Reddit sentiment
REDDIT_CLIENT_ID=your_reddit_app_id
REDDIT_CLIENT_SECRET=your_reddit_secret

# Optional — GEX data
SQUEEZEMETRICS_API_KEY=your_key
```

For separate paper/live configs, create `.env.paper` and `.env.live` overlay files:

```bash
# .env.paper
TRADING_MODE=paper
SIGNAL_CONFIDENCE_THRESHOLD=55
KELLY_FRACTION=0.20
WEB_PORT=8090

# .env.live
TRADING_MODE=live
SIGNAL_CONFIDENCE_THRESHOLD=65
KELLY_FRACTION=0.10
WEB_PORT=8091
```

### 3. Get Alpaca API Keys

1. Sign up at https://app.alpaca.markets
2. Go to Paper Trading dashboard
3. Generate API keys (Key ID + Secret Key)
4. Options trading is available on paper accounts by default

## Running

### Docker (Recommended)

```bash
# Start paper trading + Redis
docker compose up -d

# View logs
docker compose logs -f paper

# Stop
docker compose down
```

### Direct (Development)

```bash
# Start Redis separately
docker run -d --name redis -p 6389:6379 redis:7-alpine

# Run the bot
TRADING_MODE=paper python -m src.core
```

### Verify It's Working

Successful startup looks like:

```
============================================================
Zero-DTE Scalper starting (paper mode)
Underlyings: ['SPY', 'QQQ', 'IWM']
============================================================
Alpaca connected: equity=$100000, buying_power=$200000, paper=True
Pre-market setup...
Macro calendar loaded: 2 events (0 high, 1 medium)
VIX regime: normal (18.5)
Chain refreshed: SPY has 342 contracts
Chain refreshed: QQQ has 280 contracts
Chain refreshed: IWM has 195 contracts
Starting trading loops...
Equity stream connecting...
Option stream connecting...
```

## Trading Schedule (All Times ET)

| Time | Action |
|------|--------|
| Pre-market | Load macro calendar, check VIX, refresh chains |
| 09:30 | Market open — streams start, ticks begin flowing |
| 09:45 | **Entry window opens** — strategy loop starts evaluating |
| 09:45-14:30 | Active trading — entries evaluated every 15s |
| 14:30 | **Entry cutoff** — no new positions after this |
| 15:15 | **Hard close** — all remaining positions closed |
| 15:30 | Alpaca 0DTE auto-liquidation deadline |
| 16:00 | Market close — daily summary persisted |

## Configuration Reference

### Strategy Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TARGET_DELTA` | 0.30 | Buy options near 30 delta |
| `MIN_PREMIUM` | 0.50 | Minimum option premium ($) |
| `MAX_PREMIUM` | 5.00 | Maximum option premium ($) |
| `MIN_SPREAD_RATIO` | 0.90 | Bid/ask ratio >= 0.90 (tight spread) |
| `SIGNAL_CONFIDENCE_THRESHOLD` | 55 | Minimum confidence to trade (0-100) |

### Risk Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `KELLY_FRACTION` | 0.20 | Fraction of Kelly criterion to use |
| `MAX_POSITION_PCT` | 0.05 | Max 5% of portfolio per trade |
| `MAX_PORTFOLIO_EXPOSURE` | 0.30 | Max 30% of portfolio in options |
| `DAILY_DRAWDOWN_HALT` | 0.08 | Halt at 8% daily drawdown |
| `MAX_CONTRACTS_PER_TRADE` | 10 | Hard cap on contracts per order |

### Exit Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PT_PROFIT_TARGET_PCT` | 0.50 | Take profit at 50% premium gain |
| `SL_STOP_LOSS_PCT` | 0.30 | Stop loss at 30% premium loss |
| `SL_TRAILING_PCT` | 0.20 | Trailing stop 20% below peak |
| `HARD_CLOSE` | 15:15 | Close all positions by 3:15 PM ET |

### Greeks Portfolio Limits

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_PORTFOLIO_DELTA` | 50.0 | Max absolute portfolio delta |
| `MAX_PORTFOLIO_GAMMA` | 20.0 | Max absolute portfolio gamma |
| `MAX_PORTFOLIO_THETA` | -100.0 | Min portfolio theta (max decay cost) |
| `MAX_PORTFOLIO_VEGA` | 30.0 | Max absolute portfolio vega |

### Ensemble Weights

All weights should sum to 1.0:

| Factor | Weight | Config Key |
|--------|--------|------------|
| Technical (RSI/MACD/BB) | 0.25 | `WEIGHT_TECHNICAL` |
| Tick momentum | 0.20 | `WEIGHT_TICK_MOMENTUM` |
| GEX regime | 0.15 | `WEIGHT_GEX` |
| Options flow | 0.15 | `WEIGHT_FLOW` |
| VIX/IV analysis | 0.10 | `WEIGHT_VIX` |
| Market internals | 0.10 | `WEIGHT_INTERNALS` |
| Sentiment | 0.05 | `WEIGHT_SENTIMENT` |

## Monitoring

### Logs

```bash
# All logs
docker compose logs -f paper

# Filter for trades
docker compose logs paper | grep -E "SIGNAL|ORDER|FILLED|CLOSED"

# Filter for risk events
docker compose logs paper | grep -E "CIRCUIT|BLACKOUT|PDT|HALT"
```

### SQLite Database

The bot persists all trades and portfolio state to `data/bot.db`:

```bash
sqlite3 data/bot.db "SELECT * FROM trades ORDER BY entry_time DESC LIMIT 10;"
sqlite3 data/bot.db "SELECT * FROM portfolio_state;"
```

### Telegram Alerts

If configured, the bot sends Telegram messages for:
- Trade opened (underlying, strike, premium, confidence)
- Trade closed (P&L, hold time, exit reason)
- Circuit breaker triggered
- High-impact macro events today

## Troubleshooting

### "Failed to connect to Alpaca"
- Check `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in `.env`
- Verify keys are for paper (not live) if `ALPACA_PAPER=true`
- Alpaca may be down — check https://status.alpaca.markets

### "No viable strikes found"
- Market may be closed (options chains empty outside RTH)
- Premium bounds too tight — try widening `MIN_PREMIUM` / `MAX_PREMIUM`
- Expiration might not have 0DTE contracts today (check if it's a trading day)

### "PDT restricted"
- Account under $25k with 3+ day trades in 5 days
- Wait for oldest day trade to roll off, or add funds to exceed $25k

### "VIX crisis" / "Macro blackout"
- These are protective gates — the bot is correctly avoiding high-risk periods
- To override (not recommended): adjust `VIX_CRISIS_THRESHOLD` or `MACRO_BLACKOUT_MINUTES`

### Redis connection failed
- Check Redis is running: `docker compose ps redis`
- The bot degrades gracefully — runs with in-memory cache only (no cross-asset consensus)
