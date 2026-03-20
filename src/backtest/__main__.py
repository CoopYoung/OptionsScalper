"""CLI entry point for running backtests.

Usage:
    python -m src.backtest --underlying SPY --start 2026-03-01 --end 2026-03-14
    python -m src.backtest --underlying SPY --days 5
    python -m src.backtest --underlying SPY,QQQ --days 10 --interval 2m
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-DTE Options Scalper Backtester")
    parser.add_argument("--underlying", type=str, default="SPY",
                        help="Underlying symbol(s), comma-separated (default: SPY)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=5,
                        help="Number of trading days to backtest (default: 5)")
    parser.add_argument("--interval", type=str, default="2m",
                        choices=["1m", "2m", "5m"],
                        help="Bar interval (default: 2m)")
    parser.add_argument("--capital", type=float, default=100_000,
                        help="Initial capital (default: 100000)")
    parser.add_argument("--slippage", type=float, default=0.005,
                        help="Slippage percentage (default: 0.005 = 0.5%%)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Suppress noisy loggers
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)

    # Change to project root for .env loading
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)

    from src.infra.config import get_settings
    from src.backtest.engine import BacktestEngine, SlippageModel

    settings = get_settings()
    slippage = SlippageModel(slippage_pct=args.slippage)

    # Parse dates
    if args.end:
        end_date = date.fromisoformat(args.end)
    else:
        end_date = date.today() - timedelta(days=1)

    if args.start:
        start_date = date.fromisoformat(args.start)
    else:
        start_date = end_date - timedelta(days=args.days + 3)  # Extra for weekends

    underlyings = [s.strip() for s in args.underlying.split(",")]

    print(f"\n  Zero-DTE Options Scalper Backtester")
    print(f"  {'=' * 45}")
    print(f"  Underlyings:  {', '.join(underlyings)}")
    print(f"  Period:       {start_date} to {end_date}")
    print(f"  Interval:     {args.interval}")
    print(f"  Capital:      ${args.capital:,.2f}")
    print(f"  Slippage:     {args.slippage:.1%}")
    print(f"  {'=' * 45}\n")

    all_results = []
    for underlying in underlyings:
        engine = BacktestEngine(settings, slippage, args.capital)

        try:
            result = asyncio.run(engine.run(underlying, start_date, end_date, args.interval))
        except Exception:
            logging.exception("Backtest failed for %s", underlying)
            continue

        print(result.summary())
        print(result.daily_summary())
        print()

        all_results.append(result)

    # Save results
    if args.output and all_results:
        output_data = []
        for result in all_results:
            output_data.append({
                "underlying": result.underlying,
                "start_date": result.start_date.isoformat(),
                "end_date": result.end_date.isoformat(),
                "initial_capital": result.initial_capital,
                "final_capital": result.final_capital,
                "total_pnl": result.total_pnl,
                "total_trades": result.total_trades,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "max_drawdown": result.max_drawdown,
                "sharpe_ratio": result.sharpe_ratio,
                "avg_trade_pnl": result.avg_trade_pnl,
                "avg_hold_minutes": result.avg_hold_minutes,
                "config": result.config,
                "days": [
                    {
                        "date": d.date.isoformat(),
                        "pnl": d.total_pnl,
                        "trades": len(d.trades),
                        "wins": d.wins,
                        "losses": d.losses,
                        "vix": d.vix,
                        "trade_details": [
                            {
                                "symbol": t.symbol,
                                "direction": t.direction,
                                "contracts": t.contracts,
                                "entry_price": t.entry_price,
                                "exit_price": t.exit_price,
                                "pnl": t.pnl,
                                "hold_minutes": t.hold_minutes,
                                "exit_reason": t.exit_reason,
                                "confidence": t.entry_confidence,
                            }
                            for t in d.trades
                        ],
                    }
                    for d in result.days
                ],
            })

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
