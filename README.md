# OptionsScalper — AI-Powered 0DTE Options Trader

AI-powered options trading system that uses a **local LLM (Ollama) or cloud API (Claude/GPT)** as the trading brain, with **Python handling all data gathering, signal computation, and risk validation**. Supports dual brokers: **Alpaca** (paper trading) and **Public.com** (live trading).

## Architecture

```
Cron (every 10 min, market hours)
  └─ python -m hybrid.orchestrator --mode paper
       ├─ Python gathers all data:
       │   ├─ Alpaca/Public.com: quotes, bars, option chains, Greeks
       │   ├─ yfinance: VIX, sector performance
       │   ├─ CNN: Fear & Greed index
       │   └─ Finnhub: news, earnings calendar
       ├─ Python computes technicals:
       │   ├─ RSI(14), MACD(12/26/9), Bollinger Bands(20,2)
       │   ├─ VWAP, volume ratio, momentum(5/20)
       │   └─ Support/resistance levels
       ├─ Formats ONE digest prompt (all data pre-digested)
       ├─ Sends to LLM (Ollama local or API) → gets JSON decision
       ├─ Validates via Python risk layer (cannot be bypassed)
       └─ Executes via broker API + Telegram alert
```

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/CoopYoung/OptionsScalper.git
cd OptionsScalper
pip install -r hybrid/requirements.txt

# 2. Configure
cp .env.example .env    # Edit with your Alpaca + Telegram credentials
# Also edit hybrid/.env for Finnhub, Public.com keys

# 3. Verify connections
python -m hybrid.cli account          # Check Alpaca
python -m hybrid.cli quotes SPY QQQ   # Check market data
python -m hybrid.cli vix              # Check VIX data
python -m hybrid.cli market-overview  # VIX + F&G + sectors in one call
```

## Orchestrator Commands

```bash
# ── Testing & Development ────────────────────────────────────
# See the digest prompt (no LLM call, no execution)
python -m hybrid.orchestrator --mode paper --digest-only --force

# Full pipeline with LLM but don't execute trades
python -m hybrid.orchestrator --mode paper --dry-run --force

# Full pipeline with execution (during market hours)
python -m hybrid.orchestrator --mode paper

# Force run outside market hours (for testing)
python -m hybrid.orchestrator --mode paper --force

# ── LLM Provider Options ─────────────────────────────────────
# Local Ollama (default, $0/month)
python -m hybrid.orchestrator --mode paper --provider ollama

# Claude API (~$8/month with Haiku, ~$60/month with Sonnet)
LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-... python -m hybrid.orchestrator --mode paper

# OpenAI-compatible (OpenAI, Groq, Together, etc.)
LLM_PROVIDER=openai OPENAI_API_KEY=sk-... python -m hybrid.orchestrator --mode paper

# ── Broker Modes ──────────────────────────────────────────────
# Paper trading on Alpaca (default)
python -m hybrid.orchestrator --mode paper

# Live trading on Public.com
python -m hybrid.orchestrator --mode live

# ── Cron Setup (autonomous trading) ──────────────────────────
# Every 10 min during market hours (Mon-Fri 9:45 AM - 3:15 PM ET)
*/10 9-15 * * 1-5 cd /path/to/OptionsScalper && python -m hybrid.orchestrator --mode paper >> /var/log/trader.log 2>&1

# ── CLI Data Commands (26 commands) ──────────────────────────
python -m hybrid.cli account             # Account info
python -m hybrid.cli positions           # Open positions
python -m hybrid.cli orders              # Order status
python -m hybrid.cli quotes SPY QQQ IWM  # Real-time quotes
python -m hybrid.cli bars SPY            # Price bars (5min default)
python -m hybrid.cli chain SPY           # Options chain + Greeks
python -m hybrid.cli expirations SPY     # Available expirations
python -m hybrid.cli vix                 # VIX + regime classification
python -m hybrid.cli fear-greed          # CNN Fear & Greed index
python -m hybrid.cli sectors             # Sector performance + breadth
python -m hybrid.cli market-overview     # VIX + F&G + sectors combined
python -m hybrid.cli news                # Market headlines
python -m hybrid.cli calendar            # Economic calendar
python -m hybrid.cli earnings            # Earnings calendar
python -m hybrid.cli indices VIX SPX     # Index quotes (Public.com)
python -m hybrid.cli chain-greeks SPY    # Chain + Greeks (Public.com)
```

## Project Structure

```
hybrid/                              # Active trading system (v2)
├── orchestrator.py                  # Main entry point — cron-friendly, one cycle
├── digest.py                        # Data gathering + technicals + prompt formatting
├── llm.py                           # LLM abstraction (Ollama, Anthropic, OpenAI)
├── config.py                        # Settings from .env
├── cli.py                           # 26 CLI commands (data inspection)
├── trading_prompt.md                # Trading instructions (for claude -p mode)
├── broker/
│   ├── broker_base.py               # Broker Protocol + AlpacaBroker wrapper
│   ├── alpaca.py                    # Alpaca REST wrapper (paper trading)
│   ├── public_broker.py             # Public.com SDK wrapper (live trading)
│   ├── public_data.py               # Public.com data (Greeks, indices)
│   └── market_data.py               # External data (VIX, F&G, sectors, news)
├── risk/
│   └── validator.py                 # Hard rule enforcement (cannot be bypassed)
├── alerts/
│   └── telegram.py                  # Trade alerts + daily summaries
├── logs/
│   └── audit.py                     # JSONL audit trail
└── backtest/                        # Backtesting infrastructure
    ├── run_backtest.py              # Claude-in-the-loop backtester
    ├── data_loader.py               # Historical data + Black-Scholes pricing
    ├── snapshot_builder.py          # Synthesize CLI responses from history
    └── mock_cli.py                  # Drop-in CLI for backtest mode

src/                                 # Original async Python bot (legacy)
```

## Digest Signals (18 total — what the LLM sees each cycle)

**Market Context:**
VIX + regime, Fear & Greed (contrarian), sector breadth, economic calendar, earnings, news headlines, time-of-day regime (opening/trend/lunch/power hour + theta decay)

**Per-Underlying Technical Analysis:**
RSI(14), MACD(12/26/9), Bollinger Bands, VWAP, volume ratio, momentum(5/20), support/resistance, overnight gap, intraday narrative (open → high → low → current story)

**Options Intelligence:**
Put/Call ratio (volume + OI), IV rank/percentile, net delta flow + unusual activity detection, cross-underlying correlation/divergence, round-number pinning magnet

**Context Awareness:**
Recent trade history (last 5 trades — avoids repeating mistakes), price-level proximity alerts (near support/resistance/round numbers)

## Setup for Automated Trading

```bash
# 1. Configure .env
OLLAMA_URL=http://<orange-pi-ip>:11434    # Point to Orange Pi running Ollama
OLLAMA_MODEL=0xroyce/plutus               # Or any Ollama model
LLM_PROVIDER=ollama                       # ollama | anthropic | openai

# 2. Test connectivity
python -m hybrid.orchestrator --mode paper --digest-only --force  # Generate digest
python -m hybrid.orchestrator --mode paper --dry-run --force      # Full pipeline, no execution

# 3. Install cron (every 10 min during market hours)
crontab -e
# Add: */10 9-15 * * 1-5 cd /path/to/OptionsScalper && python -m hybrid.orchestrator --mode paper >> /var/log/trader.log 2>&1
```

## Risk Rules (Hard-Coded, Validator-Enforced)

| Rule | Default | Configurable |
|------|---------|-------------|
| Max risk per trade | $150 | MAX_RISK_PER_TRADE |
| Max daily loss | $500 | MAX_DAILY_LOSS |
| Max concurrent positions | 3 | MAX_CONCURRENT_POSITIONS |
| Max contracts per trade | 5 | MAX_CONTRACTS_PER_TRADE |
| Confidence threshold | 55 | SIGNAL_CONFIDENCE_THRESHOLD |
| Entry window | 9:45 AM - 3:00 PM ET | ENTRY_START_ET, ENTRY_CUTOFF_ET |
| Hard close | 3:45 PM ET | HARD_CLOSE_ET |
| Profit target | 50% | PROFIT_TARGET_PCT |
| Stop loss | 30% | STOP_LOSS_PCT |
| Trailing stop | 15% from peak (after 30% gain) | TRAILING_STOP_PCT |

## LLM Options

| Provider | Model | Monthly Cost (39 calls/day) | Quality |
|----------|-------|---------------------------|---------|
| **Ollama (local)** | 0xroyce/plutus | **$0** | Good for PoC |
| Anthropic | Claude Haiku | ~$8 | Great |
| Anthropic | Claude Sonnet | ~$60 | Best |
| Google | Gemini 2.5 Flash | ~$3 | Good, fastest |
| OpenAI | GPT-4o-mini | ~$2.50 | Good |
| DeepSeek | V3 | ~$1.50 | Cheapest cloud |

## Broker Comparison

| Feature | Alpaca (paper) | Public.com (live) |
|---------|---------------|-------------------|
| Options commissions | $0 | $0 + rebates ($0.06-$0.18/contract) |
| WebSocket streaming | Yes | No (REST only) |
| Rate limit | 200 req/min | 10 req/s |
| Paper trading | Yes | No |
| Historical bars | Yes | No (use yfinance fallback) |

## Phase Plan

- **Phase 1 (current):** Paper trading on Alpaca with local Ollama — validate decision-making
- **Phase 2:** Upgrade to Claude API for better reasoning, continue paper trading
- **Phase 3:** Switch to Public.com for live trading (commission rebates)
