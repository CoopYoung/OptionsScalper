"""Delta experiment: test different option deltas across underlyings.

Answers: does trading closer-to-ATM options reduce theta drag enough
to improve P&L, even though individual contracts cost more?

Usage:
    python -m src.backtest.delta_experiment
"""

import asyncio
import logging
import os
from datetime import date
from pathlib import Path

# Setup before imports
project_root = Path(__file__).parent.parent.parent
os.chdir(project_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("peewee").setLevel(logging.WARNING)

from src.infra.config import get_settings
from src.backtest.engine import BacktestEngine, SlippageModel


DELTAS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
UNDERLYINGS = ["SPY", "QQQ", "IWM"]
START = date(2026, 1, 15)
END = date(2026, 3, 20)
INTERVAL = "5m"
CAPITAL = 100_000


async def run_single(underlying: str, target_delta: float) -> dict:
    """Run one backtest with a specific delta."""
    settings = get_settings()

    # Override delta and premium cap for this experiment
    settings.target_delta = target_delta
    # ATM options (delta ~0.50) on SPY $570 can be $5-8
    # Scale max premium with delta to allow ATM contracts
    settings.max_premium = 3.0 + target_delta * 10.0  # 0.25→5.5, 0.50→8.0

    slippage = SlippageModel(slippage_pct=0.005)
    engine = BacktestEngine(settings, slippage, CAPITAL)

    result = await engine.run(underlying, START, END, INTERVAL)

    return {
        "underlying": underlying,
        "delta": target_delta,
        "pnl": result.total_pnl,
        "trades": result.total_trades,
        "wins": result.win_count,
        "losses": result.loss_count,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "sharpe": result.sharpe_ratio,
        "avg_pnl": result.avg_trade_pnl,
        "avg_hold": result.avg_hold_minutes,
        "max_dd": result.max_drawdown,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "wl_ratio": 0.0,
    }


async def main() -> None:
    print(f"\n{'=' * 80}")
    print(f"  DELTA EXPERIMENT: {START} to {END}")
    print(f"  Deltas: {DELTAS}")
    print(f"  Underlyings: {UNDERLYINGS}")
    print(f"  Capital: ${CAPITAL:,.0f}  |  Interval: {INTERVAL}")
    print(f"{'=' * 80}\n")

    results: list[dict] = []

    for underlying in UNDERLYINGS:
        for delta in DELTAS:
            print(f"  Running {underlying} δ={delta:.2f}...", end=" ", flush=True)
            try:
                r = await run_single(underlying, delta)

                # Compute avg winner/loser from the engine run
                # (we lost the detail, but can derive from aggregate)
                if r["wins"] > 0 and r["losses"] > 0:
                    # total_pnl = wins*avg_win + losses*avg_loss
                    # We need the full result for this — re-run to get detail
                    pass

                results.append(r)
                marker = "+" if r["pnl"] > 0 else "-"
                print(
                    f"[{marker}] P&L=${r['pnl']:>8.2f}  "
                    f"Trades={r['trades']:>3d}  "
                    f"WR={r['win_rate']:.1%}  "
                    f"PF={r['profit_factor']:.2f}  "
                    f"Sharpe={r['sharpe']:.2f}"
                )
            except Exception as e:
                print(f"FAILED: {e}")
                continue

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"  RESULTS MATRIX")
    print(f"{'=' * 80}")
    print(
        f"\n  {'Underlying':<10} {'Delta':>6} {'P&L':>10} {'Trades':>7} "
        f"{'WR':>7} {'PF':>7} {'Sharpe':>8} {'AvgPnL':>8} {'MaxDD':>7}"
    )
    print(f"  {'-' * 75}")

    for underlying in UNDERLYINGS:
        for r in results:
            if r["underlying"] != underlying:
                continue
            marker = ">>>" if r["pnl"] > 0 else "   "
            print(
                f"{marker}{r['underlying']:<7} "
                f"{r['delta']:>6.2f} "
                f"${r['pnl']:>9.2f} "
                f"{r['trades']:>7d} "
                f"{r['win_rate']:>6.1%} "
                f"{r['profit_factor']:>7.2f} "
                f"{r['sharpe']:>8.2f} "
                f"${r['avg_pnl']:>7.2f} "
                f"{r['max_dd']:>6.2%}"
            )
        print()

    # Best per underlying
    print(f"\n  BEST DELTA PER UNDERLYING:")
    print(f"  {'-' * 50}")
    for underlying in UNDERLYINGS:
        uresults = [r for r in results if r["underlying"] == underlying]
        if uresults:
            best = max(uresults, key=lambda r: r["pnl"])
            print(
                f"  {underlying}: δ={best['delta']:.2f} → "
                f"P&L=${best['pnl']:>8.2f}  WR={best['win_rate']:.1%}  "
                f"PF={best['profit_factor']:.2f}  ({best['trades']} trades)"
            )

    # Overall best
    if results:
        best_overall = max(results, key=lambda r: r["pnl"])
        print(f"\n  OVERALL BEST: {best_overall['underlying']} δ={best_overall['delta']:.2f}")
        print(f"  P&L=${best_overall['pnl']:.2f}  Sharpe={best_overall['sharpe']:.2f}")
    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    asyncio.run(main())
