#!/usr/bin/env python3
"""Hybrid Trader — Main entry point.

This is what the cron job calls every 10-15 minutes during market hours.
It's intentionally simple:

1. Check if market is open
2. Call Claude to analyze + trade
3. Log the result
4. Send Telegram notification
5. Exit

No loops, no state machines, no complex async. Each run is independent.

Usage:
    # Single cycle (cron calls this)
    python -m hybrid.main

    # Continuous mode (runs its own loop — alternative to cron)
    python -m hybrid.main --continuous

    # Dry run (no trades, just analysis)
    python -m hybrid.main --dry-run

    # Daily summary
    python -m hybrid.main --summary
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from hybrid.ai.analyst import run_analysis_cycle
from hybrid.alerts.telegram import (
    notify_cycle_result,
    notify_daily_summary,
    notify_error,
    notify_shutdown,
    notify_startup,
)
from hybrid.broker import alpaca
from hybrid.config import CRON_INTERVAL_MINUTES
from hybrid.logs.audit import get_todays_trades, log_cycle
from hybrid.risk.validator import get_daily_state, is_market_hours

ET = ZoneInfo("America/New_York")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("hybrid/trade_logs/hybrid.log"),
    ],
)
logger = logging.getLogger("hybrid")


def run_single_cycle() -> dict:
    """Run a single analysis + trading cycle."""
    logger.info("=" * 60)
    logger.info("Starting analysis cycle at %s ET",
                datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S"))

    if not is_market_hours():
        logger.info("Market is closed — skipping cycle")
        return {"action": "MARKET_CLOSED", "trades": [], "errors": []}

    try:
        result = run_analysis_cycle()

        # Log to audit trail
        log_cycle(result)

        # Notify via Telegram
        notify_cycle_result(result)

        # Log summary
        action = result.get("action", "UNKNOWN")
        trades = result.get("trades", [])
        cost = result.get("token_usage", {}).get("estimated_cost", 0)
        logger.info(
            "Cycle complete: action=%s trades=%d cost=$%.4f",
            action, len(trades), cost,
        )

        return result

    except Exception as e:
        logger.exception("Cycle failed: %s", e)
        notify_error(str(e))
        return {"action": "ERROR", "trades": [], "errors": [str(e)]}


def run_continuous() -> None:
    """Run in continuous mode — own event loop instead of cron."""
    logger.info("Starting continuous mode (interval: %d min)", CRON_INTERVAL_MINUTES)
    notify_startup()

    try:
        while True:
            now_et = datetime.now(ET)

            if is_market_hours():
                run_single_cycle()
            else:
                # Check if it's end of day — send summary
                current_time = now_et.strftime("%H:%M")
                if current_time == "16:05":
                    send_daily_summary()

                logger.info(
                    "Market closed (%s ET) — sleeping %d min",
                    current_time, CRON_INTERVAL_MINUTES,
                )

            # Sleep until next cycle
            time.sleep(CRON_INTERVAL_MINUTES * 60)

    except KeyboardInterrupt:
        logger.info("Shutting down (keyboard interrupt)")
        notify_shutdown("Keyboard interrupt")
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        notify_shutdown(f"Error: {e}")
        raise


def send_daily_summary() -> None:
    """Send end-of-day summary via Telegram."""
    try:
        daily_state = get_daily_state()
        positions = alpaca.get_positions()
        account = alpaca.get_account()
        notify_daily_summary(daily_state, positions, account)
        logger.info("Daily summary sent")
    except Exception as e:
        logger.error("Failed to send daily summary: %s", e)


def main():
    parser = argparse.ArgumentParser(description="Hybrid Claude + Alpaca Trader")
    parser.add_argument(
        "--continuous", action="store_true",
        help="Run in continuous mode instead of single cycle",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Send daily summary and exit",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Analyze but don't execute trades (not yet implemented)",
    )
    args = parser.parse_args()

    if args.summary:
        send_daily_summary()
        return

    if args.continuous:
        run_continuous()
    else:
        result = run_single_cycle()
        sys.exit(0 if result.get("action") != "ERROR" else 1)


if __name__ == "__main__":
    main()
