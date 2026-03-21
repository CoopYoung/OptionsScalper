# OptionsScalper — Hybrid AI Options Trader

AI-powered options trading system that uses **Claude Code as the trading brain** and **Alpaca as the broker**. Claude analyzes the market, manages positions, and executes trades autonomously during market hours — with a Python validation layer that enforces hard risk rules before any order reaches the broker.

**Cost: $0 extra** — runs on your existing Claude Max subscription.

## Architecture

```
Cron (every 10 min)
  └─ run_cycle.sh
       └─ claude -p "trading prompt" --allowedTools "Bash(...)"
            ├─ python3 -m hybrid.cli account         → account state
            ├─ python3 -m hybrid.cli positions        → open P&L
            ├─ python3 -m hybrid.cli quotes SPY QQQ   → real-time prices
            ├─ python3 -m hybrid.cli chain SPY ...     → options chain + Greeks
            ├─ python3 -m hybrid.cli validate buy ...  → risk check (dry run)
            ├─ python3 -m hybrid.cli order buy ...     → validated execution
            └─ Telegram alert                          → trade notification
```

**Each cycle is stateless** — Claude reads fresh portfolio state from Alpaca every time. No position tracking dicts, no stale state, no accumulation bugs.

### Three Validation Layers

1. **Claude's prompt rules** — trading instructions, analysis process, risk awareness
2. **Python validator** — hard limits on size, daily loss, timing, position count (cannot be bypassed)
3. **Alpaca broker** — buying power, options approval level, PDT enforcement

### What Claude Analyzes Each Cycle

| Step | What It Does |
|------|-------------|
| Check state | Account equity, open positions, pending orders, daily P&L |
| Manage positions | Current option quotes, take profit/stop loss/time exit decisions |
| Scan setups | Price bars (OHLCV, VWAP), support/resistance, volume analysis |
| Evaluate options | Chains with Greeks (delta, gamma, theta, vega, IV), spreads, liquidity |
| Validate & execute | Dry-run validation → limit order placement → Telegram alert |

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/CoopYoung/OptionsScalper.git
cd OptionsScalper
pip install -r hybrid/requirements.txt

# 2. Configure (edit .env with your Alpaca + Telegram credentials)
cp hybrid/.env.example .env

# 3. Test it
python -m hybrid.cli account          # Verify Alpaca connection
python -m hybrid.cli quotes SPY QQQ   # Check quotes
./hybrid/run_cycle.sh --force         # Run one cycle (even outside market hours)

# 4. Install cron for autonomous trading
./hybrid/setup_cron.sh
```

### Requirements

- **Claude Code** CLI installed and authenticated (Max subscription recommended)
- **Alpaca** account (paper trading is free — no deposit needed)
- **Python 3.10+**
- Telegram bot (optional, for trade alerts)

## Project Structure

```
hybrid/                         # Active trading system
├── run_cycle.sh                # Cron entry point — invokes Claude Code
├── setup_cron.sh               # One-command cron installer
├── trading_prompt.md           # Claude's analysis instructions + rules
├── cli.py                      # 14 CLI commands wrapping Alpaca API
├── config.py                   # Settings from .env
├── broker/
│   ├── alpaca.py               # Alpaca REST wrapper
│   └── tools.py                # Tool definitions (API fallback mode)
├── ai/
│   ├── analyst.py              # Claude API caller (fallback mode)
│   └── prompts.py              # System prompt builder (fallback mode)
├── risk/
│   └── validator.py            # Hard rule enforcement
├── alerts/
│   └── telegram.py             # Trade alerts + daily summaries
└── logs/
    └── audit.py                # JSONL audit trail

src/                            # Original Python bot (deprecated)
```

## Risk Rules (Hard-Coded, Validator-Enforced)

| Rule | Default | Configurable |
|------|---------|-------------|
| Max risk per trade | $150 | MAX_RISK_PER_TRADE |
| Max daily loss | $500 | MAX_DAILY_LOSS |
| Max concurrent positions | 3 | MAX_CONCURRENT_POSITIONS |
| Max contracts per trade | 5 | MAX_CONTRACTS_PER_TRADE |
| Min reward:risk ratio | 1.5:1 | MIN_REWARD_RISK_RATIO |
| Entry window | 9:45 AM - 3:00 PM ET | ENTRY_START_ET, ENTRY_CUTOFF_ET |
| Hard close | 3:45 PM ET | HARD_CLOSE_ET |
| Strategies allowed | Spreads + long options only | ALLOWED_STRATEGIES |

## Modes

| Mode | Command | Cost |
|------|---------|------|
| **Claude Code (recommended)** | `./hybrid/run_cycle.sh` | Free (Max subscription) |
| Claude API (fallback) | `python -m hybrid --api` | ~$5-15/day pay-as-you-go |
| Daily summary | `python -m hybrid --summary` | Free |
| Direct CLI | `python -m hybrid.cli [command]` | Free |

## Phase Plan

- **Phase 1 (current):** Paper trading on Alpaca — validate Claude's decision-making
- **Phase 2:** Swap to Public.com for live trading (commission rebates, native multi-leg support)
- **Future:** Add market data tools (VIX, sentiment, economic calendar, options flow) for richer analysis

## Background

This project started as a complex async Python bot with an 8-factor signal ensemble, 3 concurrent loops, and 3000+ lines of stateful code. After weeks of debugging — including a $30.5k paper trading loss from a position accumulation bug — the architecture was redesigned around a simpler principle: **let Claude reason about the market, let Python enforce the rules, let the broker enforce reality.**
