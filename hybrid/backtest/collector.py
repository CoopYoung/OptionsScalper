"""Market data snapshot collector.

Captures real market snapshots every N minutes during market hours.
Stores to JSONL files for later replay through Claude or analysis.

This builds a high-quality dataset of exactly what Claude would see
during each trading cycle — bars, quotes, chains, VIX, sentiment.

Usage:
    # Collect snapshots every 10 min (same as trading cron)
    python -m hybrid.backtest.collector

    # Collect every 5 min with custom output dir
    python -m hybrid.backtest.collector --interval 5 --output-dir data/snapshots

    # Install as cron job
    python -m hybrid.backtest.collector --install-cron
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


def collect_snapshot(
    underlyings: list[str] = None,
    include_chains: bool = True,
    include_market_data: bool = True,
) -> dict:
    """Capture a complete market snapshot — everything Claude would see."""
    if underlyings is None:
        underlyings = ["SPY", "QQQ", "IWM"]

    now = datetime.now(ET)
    snapshot = {
        "timestamp": now.isoformat(),
        "timestamp_utc": datetime.now(tz=ZoneInfo("UTC")).isoformat(),
    }

    # 1. Account state
    try:
        from hybrid.broker import alpaca
        snapshot["account"] = alpaca.get_account()
        snapshot["positions"] = alpaca.get_positions()
    except Exception as e:
        snapshot["account_error"] = str(e)

    # 2. Equity quotes
    try:
        from hybrid.broker import alpaca
        snapshot["quotes"] = alpaca.get_stock_quotes(underlyings)
    except Exception as e:
        snapshot["quotes_error"] = str(e)

    # 3. Price bars (last 50 × 5min)
    bars = {}
    for sym in underlyings:
        try:
            from hybrid.broker import alpaca
            bars[sym] = alpaca.get_stock_bars(sym, timeframe="5Min", limit=50)
        except Exception as e:
            bars[sym] = {"error": str(e)}
    snapshot["bars"] = bars

    # 4. Options chains (nearest expiry)
    if include_chains:
        chains = {}
        for sym in underlyings:
            try:
                from hybrid.broker import alpaca
                expirations = alpaca.get_option_expirations(sym)
                if expirations:
                    # Get nearest expiry
                    nearest = expirations[0]
                    chain = alpaca.get_option_chain(sym, expiration_date=nearest)
                    chains[sym] = {
                        "expiry": nearest,
                        "contracts": len(chain),
                        "chain": chain[:40],  # Near-money only to save space
                    }
            except Exception as e:
                chains[sym] = {"error": str(e)}
        snapshot["chains"] = chains

    # 5. Market context
    if include_market_data:
        try:
            from hybrid.broker.market_data import get_vix, get_fear_greed
            snapshot["vix"] = get_vix()
            snapshot["fear_greed"] = get_fear_greed()
        except Exception as e:
            snapshot["market_data_error"] = str(e)

        # Public.com indices
        try:
            from hybrid.broker.public_data import get_index_quotes
            snapshot["indices"] = get_index_quotes(["VIX", "SPX"])
        except Exception as e:
            snapshot["indices_error"] = str(e)

        # Finnhub news (if key available)
        try:
            from hybrid.broker.market_data import get_market_news
            news = get_market_news()
            if news and not isinstance(news[0], dict) or "error" not in news[0]:
                snapshot["news"] = news[:5]  # Top 5 headlines
        except Exception:
            pass

    return snapshot


def save_snapshot(snapshot: dict, output_dir: Path) -> Path:
    """Save snapshot to JSONL file (one file per day)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(ET).strftime("%Y-%m-%d")
    filepath = output_dir / f"snapshots_{today}.jsonl"

    with open(filepath, "a") as f:
        f.write(json.dumps(snapshot, default=str) + "\n")

    return filepath


def is_market_hours() -> bool:
    """Check if within market hours (9:30 AM - 4:00 PM ET, Mon-Fri)."""
    now = datetime.now(ET)
    if now.weekday() >= 5:  # Weekend
        return False
    market_open = now.replace(hour=9, minute=25, second=0)
    market_close = now.replace(hour=16, minute=5, second=0)
    return market_open <= now <= market_close


def run_collector(
    interval_minutes: int = 10,
    output_dir: Path = None,
    underlyings: list[str] = None,
    force: bool = False,
):
    """Run continuous snapshot collection."""
    import time

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "backtest_data"

    if underlyings is None:
        underlyings = ["SPY", "QQQ", "IWM"]

    print(f"  Snapshot Collector")
    print(f"  Interval:    {interval_minutes} min")
    print(f"  Output:      {output_dir}")
    print(f"  Underlyings: {', '.join(underlyings)}")
    print(f"  Market hours only: {'No (forced)' if force else 'Yes'}")
    print()

    while True:
        if force or is_market_hours():
            try:
                snapshot = collect_snapshot(underlyings)
                filepath = save_snapshot(snapshot, output_dir)
                ts = snapshot["timestamp"]
                n_quotes = len(snapshot.get("quotes", {}))
                vix = snapshot.get("vix", {}).get("vix", "?")
                print(f"  [{ts}] Snapshot saved → {filepath.name} "
                      f"(quotes={n_quotes}, VIX={vix})")
            except Exception as e:
                print(f"  [ERROR] Snapshot failed: {e}")
        else:
            now = datetime.now(ET)
            print(f"  [{now.strftime('%H:%M:%S')} ET] Outside market hours — waiting...")

        time.sleep(interval_minutes * 60)


def install_cron(interval_minutes: int = 10, output_dir: str = None):
    """Print crontab line for snapshot collection."""
    project = Path(__file__).parent.parent.parent.resolve()
    if output_dir is None:
        output_dir = project / "hybrid" / "backtest_data"

    script = project / "hybrid" / "backtest" / "collector.py"
    python = sys.executable

    # Cron: every N minutes during market hours
    cron_line = (
        f"*/{interval_minutes} 9-16 * * 1-5 "
        f"cd {project} && {python} -c "
        f"\"from hybrid.backtest.collector import collect_snapshot, save_snapshot; "
        f"from pathlib import Path; "
        f"s = collect_snapshot(); "
        f"save_snapshot(s, Path('{output_dir}'))\""
    )

    print(f"\n  Add this to your crontab (crontab -e):")
    print(f"  {cron_line}")
    print(f"\n  Or run the continuous collector:")
    print(f"  python -m hybrid.backtest.collector --interval {interval_minutes}")


def main():
    parser = argparse.ArgumentParser(description="Market data snapshot collector")
    parser.add_argument("--interval", type=int, default=10,
                        help="Collection interval in minutes (default: 10)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for snapshots")
    parser.add_argument("--underlying", type=str, default="SPY,QQQ,IWM",
                        help="Underlyings to collect")
    parser.add_argument("--once", action="store_true",
                        help="Collect a single snapshot and exit")
    parser.add_argument("--force", action="store_true",
                        help="Collect even outside market hours")
    parser.add_argument("--install-cron", action="store_true",
                        help="Print crontab line for collection")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)

    underlyings = [s.strip() for s in args.underlying.split(",")]
    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.install_cron:
        install_cron(args.interval, args.output_dir)
        return

    if args.once:
        snapshot = collect_snapshot(underlyings)
        if output_dir:
            filepath = save_snapshot(snapshot, output_dir)
            print(f"Snapshot saved to {filepath}")
        else:
            print(json.dumps(snapshot, indent=2, default=str))
        return

    run_collector(
        interval_minutes=args.interval,
        output_dir=output_dir,
        underlyings=underlyings,
        force=args.force,
    )


if __name__ == "__main__":
    main()
