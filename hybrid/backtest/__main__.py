"""CLI for running hybrid backtests.

Usage:
    python -m hybrid.backtest --underlying SPY --days 10
    python -m hybrid.backtest --underlying SPY,QQQ --start 2026-03-01 --end 2026-03-20
    python -m hybrid.backtest --underlying SPY --days 5 --output results.json
    python -m hybrid.backtest --underlying SPY --days 10 --profit-target 0.25 --stop-loss 0.35
"""

import argparse
import json
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Trader Backtester — simulate Claude's trading decisions"
    )

    # Core params
    parser.add_argument("--underlying", type=str, default="SPY",
                        help="Underlying(s), comma-separated (default: SPY)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None,
                        help="End date YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=10,
                        help="Number of calendar days to look back (default: 10)")
    parser.add_argument("--timeframe", type=str, default="5Min",
                        choices=["1Min", "5Min", "15Min"],
                        help="Bar timeframe (default: 5Min)")

    # Capital
    parser.add_argument("--capital", type=float, default=69_500,
                        help="Initial capital (default: 69500)")

    # Strategy tuning
    parser.add_argument("--profit-target", type=float, default=0.30,
                        help="Profit target %% (default: 0.30 = 30%%)")
    parser.add_argument("--stop-loss", type=float, default=0.40,
                        help="Stop loss %% (default: 0.40 = 40%%)")
    parser.add_argument("--max-hold", type=int, default=45,
                        help="Max hold time minutes (default: 45)")
    parser.add_argument("--confidence", type=float, default=0.55,
                        help="Confidence threshold 0-1 (default: 0.55)")
    parser.add_argument("--max-risk", type=float, default=150,
                        help="Max risk per trade $ (default: 150)")
    parser.add_argument("--max-positions", type=int, default=3,
                        help="Max concurrent positions (default: 3)")
    parser.add_argument("--slippage", type=float, default=0.02,
                        help="Slippage %% (default: 0.02 = 2%%)")

    # Output
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--trades", action="store_true",
                        help="Show individual trade log")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging")

    args = parser.parse_args()

    # Logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Change to project root
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)

    from hybrid.backtest.engine import HybridBacktester, BacktestConfig

    # Build config
    config = BacktestConfig(
        initial_capital=args.capital,
        max_risk_per_trade=args.max_risk,
        max_concurrent_positions=args.max_positions,
        profit_target_pct=args.profit_target,
        stop_loss_pct=args.stop_loss,
        time_exit_minutes=args.max_hold,
        confidence_threshold=args.confidence,
        slippage_pct=args.slippage,
    )

    # Parse dates
    if args.end:
        end_date = date.fromisoformat(args.end)
    else:
        end_date = date.today() - timedelta(days=1)

    if args.start:
        start_date = date.fromisoformat(args.start)
    else:
        start_date = end_date - timedelta(days=args.days)

    underlyings = [s.strip() for s in args.underlying.split(",")]

    # Header
    print(f"\n  Hybrid Trader Backtester")
    print(f"  {'=' * 50}")
    print(f"  Underlyings:    {', '.join(underlyings)}")
    print(f"  Period:         {start_date} to {end_date}")
    print(f"  Timeframe:      {args.timeframe}")
    print(f"  Capital:        ${args.capital:,.2f}")
    print(f"  Max risk/trade: ${args.max_risk:,.2f}")
    print(f"  Profit target:  {args.profit_target:.0%}")
    print(f"  Stop loss:      {args.stop_loss:.0%}")
    print(f"  Max hold:       {args.max_hold} min")
    print(f"  Confidence:     {args.confidence:.0%}")
    print(f"  Slippage:       {args.slippage:.0%}")
    print(f"  {'=' * 50}\n")

    # Run
    backtester = HybridBacktester(config)
    result = backtester.run(underlyings, start_date, end_date, args.timeframe)

    # Output
    print(result.summary())
    print(result.daily_table())

    if args.trades:
        print(result.trade_log())

    print()

    # Save JSON
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"  Results saved to {args.output}")


if __name__ == "__main__":
    main()
