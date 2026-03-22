#!/usr/bin/env python3
"""Hybrid Trader — Entry point for API mode (fallback).

Primary mode: Claude Code via run_cycle.sh (uses Max subscription, free)
Fallback mode: Anthropic API via this script (pay-as-you-go)

Usage:
    # Claude Code mode (recommended — free with Max subscription):
    ./hybrid/run_cycle.sh

    # API mode (if you prefer pay-as-you-go):
    python -m hybrid --api

    # Daily summary (either mode):
    python -m hybrid --summary
"""

import argparse
import logging
import signal
import sys
import time

from hybrid.alerts.telegram import notify_daily_summary, notify_error
from hybrid.broker import alpaca
from hybrid.config import CRON_INTERVAL_MINUTES
from hybrid.risk.validator import get_daily_state, is_market_hours

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hybrid")


def send_daily_summary():
    """Send end-of-day summary via Telegram."""
    try:
        daily_state = get_daily_state()
        positions = alpaca.get_positions()
        account = alpaca.get_account()
        notify_daily_summary(daily_state, positions, account)
        logger.info("Daily summary sent")
    except Exception as e:
        logger.error("Failed to send daily summary: %s", e)


def run_api_cycle():
    """Run a cycle using the Anthropic API (pay-as-you-go)."""
    if not is_market_hours():
        logger.info("Market is closed — skipping cycle")
        return

    try:
        from hybrid.ai.analyst import run_analysis_cycle
        from hybrid.alerts.telegram import notify_cycle_result
        from hybrid.logs.audit import log_cycle

        result = run_analysis_cycle()
        log_cycle(result)
        notify_cycle_result(result)
        logger.info("API cycle complete: %s", result.get("action"))
    except Exception as e:
        logger.exception("API cycle failed: %s", e)
        notify_error(str(e))


def run_serve():
    """Run continuously — loop cycles at CRON_INTERVAL_MINUTES, wait when market closed."""
    interval = CRON_INTERVAL_MINUTES * 60
    running = True

    def _stop(sig, frame):
        nonlocal running
        logger.info("Shutting down (signal %s)", sig)
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    logger.info("Hybrid Trader — serve mode (interval=%dm)", CRON_INTERVAL_MINUTES)
    logger.info("Waiting for market hours to begin cycles...")

    while running:
        if is_market_hours():
            logger.info("Market open — running cycle")
            run_api_cycle()
        else:
            logger.info("Market closed — sleeping %dm", CRON_INTERVAL_MINUTES)

        # Sleep in small increments so we can catch signals
        for _ in range(interval):
            if not running:
                break
            time.sleep(1)

    logger.info("Hybrid Trader stopped")


def main():
    parser = argparse.ArgumentParser(description="Hybrid Claude + Alpaca Trader")
    parser.add_argument("--summary", action="store_true", help="Send daily summary")
    parser.add_argument("--api", action="store_true",
                        help="Run single cycle via Anthropic API (costs money)")
    parser.add_argument("--serve", action="store_true",
                        help="Run continuously — loop cycles, wait when market closed")
    args = parser.parse_args()

    if args.summary:
        send_daily_summary()
    elif args.serve:
        run_serve()
    elif args.api:
        run_api_cycle()
    else:
        print("Hybrid Trader")
        print("")
        print("Recommended (free with Max subscription):")
        print("  ./hybrid/run_cycle.sh          # Single cycle via Claude Code")
        print("  ./hybrid/run_cycle.sh --force   # Run even outside market hours")
        print("  ./hybrid/setup_cron.sh          # Install cron for auto-trading")
        print("")
        print("Fallback (pay-as-you-go API):")
        print("  python -m hybrid --api          # Single cycle via Anthropic API")
        print("  python -m hybrid --serve        # Continuous mode (loops forever)")
        print("")
        print("Utilities:")
        print("  python -m hybrid --summary      # Send daily Telegram summary")
        print("  python -m hybrid.cli account    # Check account directly")
        print("  python -m hybrid.cli positions  # Check positions directly")


if __name__ == "__main__":
    main()
