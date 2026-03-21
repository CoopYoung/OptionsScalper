#!/usr/bin/env python3
"""Claude-in-the-loop backtester.

Replays historical market data through Claude, having it make real trading
decisions exactly as it would in live trading — but against mock CLI that
serves pre-built snapshots.

Usage:
    # Backtest last 5 trading days on SPY
    python -m hybrid.backtest.run_backtest --underlying SPY --days 5

    # Specific date range
    python -m hybrid.backtest.run_backtest --start 2025-03-10 --end 2025-03-14

    # Faster: skip bars (every other cycle)
    python -m hybrid.backtest.run_backtest --underlying SPY --days 3 --step 2

    # Save results
    python -m hybrid.backtest.run_backtest --days 5 --output results.json
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

from hybrid.backtest.data_loader import load_days, DaySnapshot
from hybrid.backtest.snapshot_builder import build_snapshot, save_snapshot
from dataclasses import asdict
from hybrid.backtest.bt_state import (
    BacktestState, load_state, save_state, init_day, record_decision,
)

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
SNAPSHOT_PATH = Path("/tmp/bt_snapshot.json")
STATE_PATH = Path("/tmp/bt_state.json")
PROMPT_TEMPLATE = Path(__file__).parent / "backtest_prompt.md"


def _build_prompt(simulated_dt: datetime, expiry_date: str) -> str:
    """Load backtest prompt template and inject simulated date/time."""
    template = PROMPT_TEMPLATE.read_text()
    dt_str = simulated_dt.strftime("%Y-%m-%d %H:%M ET")
    prompt = template.replace("{SIMULATED_DATETIME}", dt_str)
    prompt = prompt.replace("{EXPIRY_DATE}", expiry_date)
    return prompt


def _invoke_claude(prompt: str, model: str = "sonnet", max_turns: int = 25) -> dict:
    """Invoke Claude via CLI and capture output.

    Returns dict with:
        output: str — Claude's full response text
        decision: dict — parsed decision JSON if found
        error: str — error message if invocation failed
        duration_s: float — wall-clock seconds
    """
    env = os.environ.copy()
    env["BACKTEST_SNAPSHOT"] = str(SNAPSHOT_PATH)
    env["BACKTEST_STATE"] = str(STATE_PATH)
    env["BACKTEST_MAX_RISK"] = "500"          # $500 max risk per trade (realistic for SPY 0DTE)
    env["BACKTEST_MAX_DAILY_LOSS"] = "1000"   # $1000 daily loss limit

    cmd = [
        "claude", "-p", prompt,
        "--allowedTools",
        "Bash(python3 -m hybrid.backtest.mock_cli *)",
        "--model", model,
        "--max-turns", str(max_turns),
        "--output-format", "text",
    ]

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min max per cycle
            env=env,
            cwd=str(PROJECT_ROOT),
        )
        duration = time.time() - start
        output = result.stdout or ""
        stderr = result.stderr or ""

        if result.returncode != 0 and not output:
            return {
                "output": "",
                "decision": {},
                "error": f"Claude exited {result.returncode}: {stderr[:500]}",
                "duration_s": duration,
            }

        return {
            "output": output,
            "decision": {},  # Decision extracted in run_day() with state diff
            "error": "",
            "duration_s": duration,
        }

    except subprocess.TimeoutExpired:
        return {
            "output": "",
            "decision": {},
            "error": "Claude invocation timed out (300s)",
            "duration_s": 300.0,
        }
    except FileNotFoundError:
        return {
            "output": "",
            "decision": {},
            "error": "claude CLI not found — install Claude Code first",
            "duration_s": 0,
        }


def _extract_decision(text: str, state_before: dict, state_after: dict) -> dict:
    """Extract decision from Claude's response using state diff + text analysis.

    Primary method: compare state before/after Claude ran. If positions or
    trades changed, we know exactly what happened regardless of output format.

    Fallback: parse structured JSON or infer from natural language.
    """
    # ── Primary: state-based detection ──
    trades_before = state_before.get("daily_trades", 0)
    trades_after = state_after.get("daily_trades", 0)
    pos_before = len(state_before.get("positions", []))
    pos_after = len(state_after.get("positions", []))
    pnl_before = state_before.get("daily_pnl", 0)
    pnl_after = state_after.get("daily_pnl", 0)

    # New trade was placed
    if trades_after > trades_before and pos_after > pos_before:
        new_positions = state_after.get("positions", [])[pos_before:]
        trade_info = new_positions[0] if new_positions else {}
        reasoning = _extract_reasoning(text)
        return {
            "decision": "TRADE",
            "reasoning": reasoning or f"Opened {trade_info.get('symbol', '?')} @ ${trade_info.get('avg_entry_price', 0):.2f}",
            "trade": trade_info,
        }

    # Position was closed
    if pos_after < pos_before:
        pnl_change = round(pnl_after - pnl_before, 2)
        reasoning = _extract_reasoning(text)
        return {
            "decision": "EXIT",
            "reasoning": reasoning or f"Closed position, P&L ${pnl_change:+.2f}",
        }

    # Has positions but didn't change them
    if pos_after > 0 and pos_after == pos_before:
        reasoning = _extract_reasoning(text)
        return {
            "decision": "HOLD",
            "reasoning": reasoning or "Holding existing positions",
        }

    # ── Fallback: try to parse structured JSON from output ──
    if text:
        patterns = [
            r'```json\s*(\{[^`]*?"decision"[^`]*?\})\s*```',
            r'```\s*(\{[^`]*?"decision"[^`]*?\})\s*```',
            r'(\{[^{}]*"decision"\s*:\s*"[^"]*"[^{}]*\})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue

    # ── Fallback: infer from natural language ──
    reasoning = _extract_reasoning(text)
    if text:
        text_lower = text.lower()
        no_trade_phrases = [
            "no trade", "no_trade", "standing aside", "no entry",
            "no new positions", "not entering", "pass on",
            "wait for", "sitting out", "no compelling", "no clear",
            "conditions aren't", "conditions are not", "nothing meets",
            "no actionable", "do not trade", "don't trade",
            "no setups", "choppy", "indecisive", "low conviction",
            "staying flat", "remain flat", "no opportunities",
        ]
        if any(w in text_lower for w in no_trade_phrases):
            return {"decision": "NO_TRADE", "reasoning": reasoning or "No compelling setup found"}

        trade_phrases = [
            "order buy", "order sell", "executed", "filled",
            "placing order", "entering", "bought", "opening position",
        ]
        if any(w in text_lower for w in trade_phrases):
            return {"decision": "TRADE", "reasoning": reasoning or "Trade executed"}

        exit_phrases = [
            "closing", "closed position", "exit", "taking profit",
            "stop loss hit", "cutting", "sold",
        ]
        if any(w in text_lower for w in exit_phrases):
            return {"decision": "EXIT", "reasoning": reasoning or "Position closed"}

    # No positions, no trades, no text match = most likely NO_TRADE
    if pos_after == 0 and trades_after == trades_before:
        return {"decision": "NO_TRADE", "reasoning": reasoning or "No action taken"}

    return {"decision": "UNKNOWN", "reasoning": reasoning or "Could not determine decision"}


def _extract_reasoning(text: str) -> str:
    """Try to pull a reasoning summary from Claude's output."""
    if not text:
        return ""

    # Look for explicit reasoning patterns
    patterns = [
        r'[Rr]eason(?:ing)?:\s*(.+?)(?:\n|$)',
        r'[Dd]ecision:\s*(.+?)(?:\n|$)',
        r'[Ss]ummary:\s*(.+?)(?:\n|$)',
        r'[Aa]ction:\s*(.+?)(?:\n|$)',
        r'[Cc]onclusion:\s*(.+?)(?:\n|$)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()[:200]

    # Take the last meaningful paragraph as summary
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip() and len(p.strip()) > 20]
    if paragraphs:
        # Prefer shorter final paragraphs (likely the summary)
        last = paragraphs[-1]
        # Skip if it's a code block
        if not last.startswith("```"):
            return last[:200]

    return ""


def run_day(
    day: DaySnapshot,
    state: BacktestState,
    step: int = 2,
    model: str = "sonnet",
    max_turns: int = 25,
    verbose: bool = False,
) -> dict:
    """Run all cycles for a single day.

    Args:
        day: Historical day data
        state: Backtest account state
        step: Bar step (2 = every other 5-min bar = ~10 min intervals)
        model: Claude model to use
        max_turns: Max tool-use turns per cycle
        verbose: Print Claude's full output

    Returns:
        Summary dict for the day.
    """
    init_day(state, day.date.isoformat())
    save_state(state, STATE_PATH)

    # Determine cycle bars (skip pre-market, start at 9:45 ET)
    market_bars = []
    for i, bar in enumerate(day.bars):
        et = bar.timestamp.astimezone(ET)
        if dtime(9, 45) <= et.time() <= dtime(15, 50):
            market_bars.append(i)

    if not market_bars:
        logger.warning(f"No market-hours bars for {day.date}")
        return {"date": day.date.isoformat(), "cycles": 0, "trades": 0, "pnl": 0}

    # Run cycles at intervals
    cycle_indices = market_bars[::step]
    expiry_date = day.date.isoformat()
    day_results = {
        "date": day.date.isoformat(),
        "underlying": day.underlying,
        "vix": day.vix,
        "open_price": day.bars[market_bars[0]].close,
        "cycles": [],
        "total_cycles": len(cycle_indices),
    }

    print(f"\n{'='*60}")
    print(f"  {day.date.isoformat()} | {day.underlying} | VIX {day.vix:.1f}")
    print(f"  {len(cycle_indices)} cycles (step={step})")
    print(f"{'='*60}")

    for cycle_num, bar_idx in enumerate(cycle_indices, 1):
        bar = day.bars[bar_idx]
        et = bar.timestamp.astimezone(ET)
        time_str = et.strftime("%H:%M")

        # Update state with simulated time
        state.current_time_et = time_str
        state.current_date = day.date.isoformat()
        save_state(state, STATE_PATH)

        # Build and save snapshot
        snapshot = build_snapshot(day, bar_idx, state)
        save_snapshot(snapshot, str(SNAPSHOT_PATH))

        # Build prompt
        prompt = _build_prompt(et, expiry_date)

        print(f"\n  Cycle {cycle_num}/{len(cycle_indices)} | {time_str} ET | "
              f"SPY ${bar.close:.2f} | P&L ${state.daily_pnl:.2f} | "
              f"Positions: {len(state.positions)}")

        # Capture state BEFORE Claude runs
        state_before = asdict(state)

        # Invoke Claude
        result = _invoke_claude(prompt, model=model, max_turns=max_turns)

        # Reload state AFTER Claude ran (it may have modified via mock_cli)
        state = load_state(STATE_PATH)
        state_after = asdict(state)

        # Detect decision by diffing state + parsing output
        decision = _extract_decision(
            result.get("output", ""), state_before, state_after
        )
        decision_str = decision.get("decision", "UNKNOWN")
        reasoning = decision.get("reasoning", "")

        record_decision(state, decision_str, reasoning,
                       decision.get("trade"))
        save_state(state, STATE_PATH)

        cycle_result = {
            "cycle": cycle_num,
            "time": time_str,
            "price": bar.close,
            "decision": decision_str,
            "reasoning": reasoning[:200],
            "daily_pnl": state.daily_pnl,
            "positions": len(state.positions),
            "duration_s": result["duration_s"],
            "error": result.get("error", ""),
        }
        day_results["cycles"].append(cycle_result)

        icon = {"TRADE": "\u2705", "EXIT": "\U0001f4b0", "NO_TRADE": "\u23f8\ufe0f",
                "HOLD": "\u23f3", "UNKNOWN": "\u2753"}.get(decision_str, "\u2753")
        print(f"  {icon} {decision_str}: {reasoning[:80]}")
        print(f"     ({result['duration_s']:.1f}s)")

        if result.get("error"):
            print(f"     \u26a0\ufe0f {result['error'][:100]}")

        if verbose and result.get("output"):
            print(f"\n--- Claude Output ---\n{result['output'][:2000]}\n---\n")

        # Check daily loss limit
        if state.daily_pnl <= -1000:
            print(f"  \U0001f6d1 Daily loss limit hit (${state.daily_pnl:.2f}). Stopping day.")
            break

    # Force close any remaining positions at end of day
    if state.positions:
        print(f"\n  \u23f0 End of day — force closing {len(state.positions)} positions")
        # Build final snapshot for close
        final_idx = market_bars[-1] if market_bars else len(day.bars) - 1
        snapshot = build_snapshot(day, final_idx, state)
        save_snapshot(snapshot, str(SNAPSHOT_PATH))

        close_prompt = (
            f"It is {day.date.isoformat()} 15:50 ET. Market is closing. "
            f"Close ALL open positions immediately.\n\n"
        )
        for pos in state.positions:
            close_prompt += (
                f"python3 -m hybrid.backtest.mock_cli close {pos['symbol']}\n"
            )
        close_prompt += "\nClose each position above using the Bash tool."

        result = _invoke_claude(close_prompt, model=model, max_turns=10)
        state = load_state(STATE_PATH)

    # Day summary
    day_results["close_price"] = day.bars[market_bars[-1]].close if market_bars else 0
    day_results["daily_pnl"] = round(state.daily_pnl, 2)
    day_results["total_trades"] = state.daily_trades
    day_results["wins"] = state.daily_wins
    day_results["losses"] = state.daily_losses
    day_results["ending_equity"] = round(state.equity, 2)

    print(f"\n  {'─'*40}")
    print(f"  Day Summary: P&L ${state.daily_pnl:+.2f} | "
          f"Trades: {state.daily_trades} | W/L: {state.daily_wins}/{state.daily_losses}")

    return day_results


def run_backtest(
    underlying: str = "SPY",
    start: date = None,
    end: date = None,
    days: int = 5,
    step: int = 2,
    model: str = "sonnet",
    max_turns: int = 25,
    initial_capital: float = 69_500.0,
    use_alpaca: bool = True,
    verbose: bool = False,
    output_file: str = None,
) -> dict:
    """Run the full Claude-in-the-loop backtest.

    Args:
        underlying: Symbol to trade (SPY, QQQ, IWM)
        start: Start date (default: end - days)
        end: End date (default: today)
        days: Number of trading days if start not specified
        step: Bar step for cycles (2 = ~10 min)
        model: Claude model (sonnet, opus, haiku)
        max_turns: Max tool-use turns per cycle
        initial_capital: Starting account equity
        use_alpaca: Try Alpaca data first (else yfinance)
        verbose: Print Claude's full output
        output_file: Save results JSON here

    Returns:
        Complete backtest results dict.
    """
    if end is None:
        end = date.today() - timedelta(days=1)  # Yesterday to ensure data
    if start is None:
        start = end - timedelta(days=days + 5)  # Extra days for weekends

    print(f"\n{'='*60}")
    print(f"  Claude-in-the-Loop Backtest")
    print(f"{'='*60}")
    print(f"  Underlying: {underlying}")
    print(f"  Period:     {start} → {end}")
    print(f"  Model:      {model}")
    print(f"  Capital:    ${initial_capital:,.0f}")
    print(f"  Step:       {step} bars (~{step * 5} min cycles)")
    print(f"{'='*60}")

    # Load historical data
    print(f"\n  Loading historical data...")
    day_snapshots = load_days(underlying, start, end, "5Min", use_alpaca)

    if not day_snapshots:
        print("  ERROR: No historical data loaded. Check API keys and date range.")
        return {"error": "No data loaded"}

    # Limit to requested number of days
    if len(day_snapshots) > days:
        day_snapshots = day_snapshots[-days:]

    print(f"  Loaded {len(day_snapshots)} trading days "
          f"({sum(len(d.bars) for d in day_snapshots)} total bars)")

    # Initialize state
    state = BacktestState(
        initial_capital=initial_capital,
        equity=initial_capital,
        cash=initial_capital,
        buying_power=initial_capital * 2,
    )
    save_state(state, STATE_PATH)

    # Run each day
    all_results = {
        "config": {
            "underlying": underlying,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "model": model,
            "initial_capital": initial_capital,
            "step": step,
            "max_turns": max_turns,
        },
        "days": [],
    }

    total_pnl = 0.0
    total_trades = 0
    total_wins = 0
    total_losses = 0
    equity = initial_capital

    for day_snap in day_snapshots:
        # Reset daily state but keep running equity
        state = BacktestState(
            initial_capital=initial_capital,
            equity=equity,
            cash=equity,
            buying_power=equity * 2,
        )

        day_result = run_day(
            day_snap, state, step=step, model=model,
            max_turns=max_turns, verbose=verbose,
        )
        all_results["days"].append(day_result)

        pnl = day_result.get("daily_pnl", 0)
        total_pnl += pnl
        total_trades += day_result.get("total_trades", 0)
        total_wins += day_result.get("wins", 0)
        total_losses += day_result.get("losses", 0)
        equity += pnl

    # Overall summary
    win_rate = total_wins / max(1, total_wins + total_losses) * 100
    avg_pnl_per_day = total_pnl / max(1, len(day_snapshots))
    return_pct = total_pnl / initial_capital * 100

    summary = {
        "total_pnl": round(total_pnl, 2),
        "return_pct": round(return_pct, 2),
        "total_trades": total_trades,
        "wins": total_wins,
        "losses": total_losses,
        "win_rate_pct": round(win_rate, 1),
        "avg_daily_pnl": round(avg_pnl_per_day, 2),
        "ending_equity": round(equity, 2),
        "trading_days": len(day_snapshots),
        "total_cycles": sum(d.get("total_cycles", 0) for d in all_results["days"]),
    }
    all_results["summary"] = summary

    # Print final summary
    print(f"\n\n{'='*60}")
    print(f"  BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"  Period:         {day_snapshots[0].date} → {day_snapshots[-1].date}")
    print(f"  Trading days:   {len(day_snapshots)}")
    print(f"  Total cycles:   {summary['total_cycles']}")
    print(f"  ──────────────────────────")
    print(f"  Total P&L:      ${total_pnl:+,.2f} ({return_pct:+.2f}%)")
    print(f"  Avg daily P&L:  ${avg_pnl_per_day:+,.2f}")
    print(f"  Ending equity:  ${equity:,.2f}")
    print(f"  ──────────────────────────")
    print(f"  Total trades:   {total_trades}")
    print(f"  Wins / Losses:  {total_wins} / {total_losses}")
    print(f"  Win rate:       {win_rate:.1f}%")
    print(f"{'='*60}")

    # Per-day breakdown
    print(f"\n  Daily Breakdown:")
    print(f"  {'Date':<12} {'P&L':>10} {'Trades':>8} {'W/L':>8}")
    print(f"  {'─'*42}")
    for d in all_results["days"]:
        pnl = d.get("daily_pnl", 0)
        trades = d.get("total_trades", 0)
        wl = f"{d.get('wins', 0)}/{d.get('losses', 0)}"
        print(f"  {d['date']:<12} ${pnl:>+9,.2f} {trades:>8} {wl:>8}")
    print()

    # Save results
    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Results saved to: {out_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Claude-in-the-loop backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m hybrid.backtest.run_backtest --underlying SPY --days 3
  python -m hybrid.backtest.run_backtest --start 2025-03-10 --end 2025-03-14
  python -m hybrid.backtest.run_backtest --days 5 --model haiku --output results.json
  python -m hybrid.backtest.run_backtest --days 1 --step 4 --verbose
        """,
    )
    parser.add_argument("--underlying", default="SPY",
                        help="Underlying to trade (default: SPY)")
    parser.add_argument("--days", type=int, default=5,
                        help="Number of trading days (default: 5)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--step", type=int, default=2,
                        help="Bar step: 1=every 5min, 2=every 10min (default: 2)")
    parser.add_argument("--model", default="sonnet",
                        choices=["sonnet", "opus", "haiku"],
                        help="Claude model (default: sonnet)")
    parser.add_argument("--max-turns", type=int, default=25,
                        help="Max tool-use turns per cycle (default: 25)")
    parser.add_argument("--capital", type=float, default=69_500.0,
                        help="Initial capital (default: 69500)")
    parser.add_argument("--no-alpaca", action="store_true",
                        help="Use yfinance instead of Alpaca for historical data")
    parser.add_argument("--verbose", action="store_true",
                        help="Print Claude's full output each cycle")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results JSON to this file")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None

    run_backtest(
        underlying=args.underlying,
        start=start,
        end=end,
        days=args.days,
        step=args.step,
        model=args.model,
        max_turns=args.max_turns,
        initial_capital=args.capital,
        use_alpaca=not args.no_alpaca,
        verbose=args.verbose,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
