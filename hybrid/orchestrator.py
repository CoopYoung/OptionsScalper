"""Main orchestrator — cron-friendly, runs one cycle and exits.

Usage:
    python -m hybrid.orchestrator --mode paper              # Paper trade on Alpaca
    python -m hybrid.orchestrator --mode live                # Live trade on Public.com
    python -m hybrid.orchestrator --mode paper --dry-run     # Full pipeline, no execution
    python -m hybrid.orchestrator --mode paper --digest-only # Print prompt and exit
    python -m hybrid.orchestrator --mode paper --force       # Run outside market hours

Designed for cron:
    */10 9-15 * * 1-5 cd /path/to/OptionsScalper && python -m hybrid.orchestrator --mode paper
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from hybrid import config
from hybrid.broker.broker_base import AlpacaBroker, Broker
from hybrid.digest import (
    build_digest,
    gather_market_context,
    gather_underlying_analysis,
    _find_todays_expiry,
)
from hybrid.llm import get_llm_client
from hybrid.risk.validator import (
    get_daily_state,
    is_market_hours,
    record_trade_pnl,
    should_force_close_all,
    validate_new_order,
)

logger = logging.getLogger("hybrid")
ET = ZoneInfo("America/New_York")

# Trailing stop state — persists across cycles within a day
_PEAKS_FILE = config.LOG_DIR / ".position_peaks.json"


# ── Logging Setup ───────────────────────────────────────────

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(config.LOG_DIR / "orchestrator.log"),
        ],
    )


# ── Broker Factory ──────────────────────────────────────────

def _get_broker(mode: str) -> Broker:
    if mode == "paper":
        return AlpacaBroker()
    elif mode == "live":
        from hybrid.broker.public_broker import PublicBroker
        return PublicBroker()
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ── Trailing Stop Peaks ────────────────────────────────────

def _load_peaks() -> dict:
    today = datetime.now(ET).strftime("%Y-%m-%d")
    if _PEAKS_FILE.exists():
        try:
            data = json.loads(_PEAKS_FILE.read_text())
            if data.get("date") == today:
                return data.get("peaks", {})
        except (json.JSONDecodeError, KeyError):
            pass
    return {}


def _save_peaks(peaks: dict) -> None:
    today = datetime.now(ET).strftime("%Y-%m-%d")
    _PEAKS_FILE.write_text(json.dumps({"date": today, "peaks": peaks}))


# ── VIX-Adaptive Exit Thresholds ──────────────────────────

def _get_vix_exit_params() -> dict:
    """Return exit thresholds adapted to last known VIX regime.

    Reads last_vix from daily state (set during market context gathering).
    Falls back to static config values if VIX unknown.
    """
    state = get_daily_state()
    vix = state.get("last_vix", 0)

    if vix <= 0:
        # No VIX data — use static config defaults
        return {
            "profit_target": config.PROFIT_TARGET_PCT,
            "stop_loss": config.STOP_LOSS_PCT,
            "trailing_activate": config.TRAILING_STOP_ACTIVATE_PCT,
            "trailing_stop": config.TRAILING_STOP_PCT,
        }

    if vix < 15:
        return {"profit_target": 40.0, "stop_loss": 25.0,
                "trailing_activate": 25.0, "trailing_stop": 12.0}
    elif vix < 20:
        return {"profit_target": 50.0, "stop_loss": 30.0,
                "trailing_activate": 30.0, "trailing_stop": 15.0}
    elif vix < 30:
        return {"profit_target": 60.0, "stop_loss": 40.0,
                "trailing_activate": 35.0, "trailing_stop": 18.0}
    else:
        return {"profit_target": 70.0, "stop_loss": 50.0,
                "trailing_activate": 40.0, "trailing_stop": 22.0}


# ── Exit Management (deterministic, no LLM) ────────────────

def _manage_exits(broker: Broker, positions: list[dict],
                  dry_run: bool = False) -> list[dict]:
    """Check all positions for mechanical exit conditions.

    Uses VIX-adaptive thresholds when VIX data is available (stored in
    daily state by the market context gathering phase of a prior cycle).
    Falls back to static config values otherwise.
    """
    exits = []
    peaks = _load_peaks()
    now_et = datetime.now(ET)
    vix_params = _get_vix_exit_params()
    profit_target = vix_params["profit_target"]
    stop_loss = vix_params["stop_loss"]
    trailing_activate = vix_params["trailing_activate"]
    trailing_stop = vix_params["trailing_stop"]

    for pos in positions:
        if pos.get("asset_class") != "us_option":
            continue

        symbol = pos["symbol"]
        entry = pos["avg_entry_price"]
        current = pos["current_price"]
        qty = pos["qty"]

        if entry <= 0:
            continue

        pnl_pct = (current - entry) / entry * 100

        # Update peak price for trailing stop
        peak = peaks.get(symbol, current)
        if current > peak:
            peak = current
            peaks[symbol] = peak

        exit_reason = None

        # Profit target (VIX-adaptive)
        if pnl_pct >= profit_target:
            exit_reason = f"PROFIT_TARGET: +{pnl_pct:.1f}% (threshold {profit_target:.0f}%)"

        # Stop loss (VIX-adaptive)
        elif pnl_pct <= -stop_loss:
            exit_reason = f"STOP_LOSS: {pnl_pct:.1f}% (threshold -{stop_loss:.0f}%)"

        # Trailing stop (VIX-adaptive activation and drawdown)
        elif pnl_pct >= trailing_activate and peak > entry:
            drawdown_from_peak = (peak - current) / peak * 100
            if drawdown_from_peak >= trailing_stop:
                exit_reason = (f"TRAILING_STOP: peak ${peak:.2f} → "
                               f"now ${current:.2f} ({drawdown_from_peak:.1f}% drawdown)")

        # Time stop
        elif now_et.strftime("%H:%M") >= "15:15":
            exit_reason = f"TIME_STOP: {now_et.strftime('%H:%M')} ET"

        if exit_reason:
            realized_pnl = (current - entry) * qty * 100
            logger.info("EXIT %s: %s → P&L $%.2f", symbol, exit_reason, realized_pnl)

            if not dry_run:
                try:
                    broker.close_position(symbol)
                    record_trade_pnl(realized_pnl)
                except Exception as e:
                    logger.error("Failed to close %s: %s", symbol, e)

            exits.append({
                "symbol": symbol,
                "reason": exit_reason,
                "pnl": round(realized_pnl, 2),
                "pnl_pct": round(pnl_pct, 1),
            })
            # Remove from peaks tracking
            peaks.pop(symbol, None)

    _save_peaks(peaks)
    return exits


# ── Force Close All ─────────────────────────────────────────

def _force_close_all(broker: Broker, positions: list[dict],
                     dry_run: bool = False) -> list[dict]:
    """Close all positions — end of day."""
    exits = []
    for pos in positions:
        if pos.get("asset_class") != "us_option":
            continue
        symbol = pos["symbol"]
        entry = pos["avg_entry_price"]
        current = pos["current_price"]
        qty = pos["qty"]
        pnl = (current - entry) * qty * 100

        logger.info("FORCE CLOSE %s: P&L $%.2f", symbol, pnl)
        if not dry_run:
            try:
                broker.close_position(symbol)
                record_trade_pnl(pnl)
            except Exception as e:
                logger.error("Failed to force close %s: %s", symbol, e)

        exits.append({"symbol": symbol, "reason": "FORCE_CLOSE", "pnl": round(pnl, 2)})

    return exits


# ── Telegram Notification ──────────────────────────────────

def _notify(message: str) -> None:
    """Send Telegram notification if configured."""
    token = config.TELEGRAM_BOT_TOKEN
    chat_id = config.TELEGRAM_CHAT_ID
    if not token or not chat_id:
        return
    try:
        import requests
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        logger.warning("Telegram notification failed: %s", e)


# ── Audit Log ───────────────────────────────────────────────

def _audit_log(entry: dict) -> None:
    """Append to JSONL audit log."""
    try:
        with open(config.AUDIT_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error("Audit log failed: %s", e)


# ── Main Cycle ──────────────────────────────────────────────

def run_cycle(
    mode: str = "paper",
    provider: str | None = None,
    dry_run: bool = False,
    digest_only: bool = False,
    force: bool = False,
    verbose: bool = False,
) -> dict:
    """Run one complete trading cycle. Returns result dict."""
    _setup_logging(verbose)
    now_et = datetime.now(ET)
    result: dict = {
        "timestamp": now_et.isoformat(),
        "mode": mode,
        "exits": [],
        "decision": {},
        "trade": None,
        "error": None,
    }

    # ── Market hours check ──
    if not force and not is_market_hours():
        logger.info("Market is closed — skipping cycle")
        result["skipped"] = "market_closed"
        return result

    # ── Initialize ──
    try:
        broker = _get_broker(mode)
        logger.info("Broker: %s mode", mode)
    except Exception as e:
        logger.error("Broker init failed: %s", e)
        result["error"] = f"Broker init: {e}"
        return result

    # ── Phase 1: Exit Management ──
    logger.info("=== PHASE 1: EXIT MANAGEMENT ===")
    try:
        positions = broker.get_positions()
        option_positions = [p for p in positions if p.get("asset_class") == "us_option"]

        if should_force_close_all() and not digest_only:
            logger.info("Past hard close — force closing all positions")
            exits = _force_close_all(broker, option_positions, dry_run)
            result["exits"] = exits
            if exits:
                summary = ", ".join(f"{e['symbol']} ${e['pnl']}" for e in exits)
                _notify(f"🔴 FORCE CLOSE: {summary}")
            _audit_log(result)
            return result

        exits = _manage_exits(broker, option_positions, dry_run)
        result["exits"] = exits
        if exits:
            for e in exits:
                _notify(f"{'🟢' if e['pnl'] > 0 else '🔴'} EXIT {e['symbol']}: "
                        f"{e['reason']} → ${e['pnl']:+.2f}")
    except Exception as e:
        logger.error("Exit management failed: %s", e)
        result["error"] = f"Exit management: {e}"

    # ── Phase 2: Entry Decision ──
    logger.info("=== PHASE 2: ENTRY DECISION ===")

    # Gate checks (all in Python, no LLM needed)
    daily_state = get_daily_state()

    current_time = now_et.strftime("%H:%M")
    if not force and not digest_only and current_time > config.ENTRY_CUTOFF_ET:
        logger.info("Past entry cutoff (%s) — no new entries", config.ENTRY_CUTOFF_ET)
        result["decision"] = {"decision": "NO_TRADE", "reasoning": "Past entry cutoff"}
        _audit_log(result)
        return result

    if not force and not digest_only and current_time < config.ENTRY_START_ET:
        logger.info("Before entry start (%s) — no new entries", config.ENTRY_START_ET)
        result["decision"] = {"decision": "NO_TRADE", "reasoning": "Before entry start"}
        _audit_log(result)
        return result

    if daily_state.get("realized_pnl", 0) <= -config.MAX_DAILY_LOSS:
        logger.info("Daily loss limit reached ($%.2f) — halting",
                     daily_state["realized_pnl"])
        result["decision"] = {"decision": "NO_TRADE", "reasoning": "Daily loss limit"}
        _audit_log(result)
        return result

    # Refresh positions after exits
    try:
        positions = broker.get_positions()
        option_positions = [p for p in positions if p.get("asset_class") == "us_option"]
    except Exception:
        option_positions = []

    if len(option_positions) >= config.MAX_CONCURRENT_POSITIONS:
        logger.info("Max positions reached (%d/%d) — no new entries",
                     len(option_positions), config.MAX_CONCURRENT_POSITIONS)
        result["decision"] = {"decision": "NO_TRADE", "reasoning": "Max positions reached"}
        _audit_log(result)
        return result

    # ── Gather Data ──
    logger.info("Gathering market context...")
    t0 = time.time()
    market_context = gather_market_context()
    logger.info("Market context gathered in %.1fs", time.time() - t0)

    # VIX crisis gate + store VIX for adaptive exits
    vix_data = market_context.get("vix", {})
    vix_val = vix_data.get("vix", 0)
    if isinstance(vix_val, (int, float)) and vix_val > 0:
        # Persist VIX for VIX-adaptive exit thresholds in next cycle
        state = get_daily_state()
        state["last_vix"] = round(float(vix_val), 2)
        from hybrid.risk.validator import _save_daily_state
        _save_daily_state(state)
        logger.info("Stored VIX %.2f for adaptive exits", vix_val)
    if isinstance(vix_val, (int, float)) and vix_val > 35:
        logger.info("VIX crisis (%.1f) — no trading", vix_val)
        result["decision"] = {"decision": "NO_TRADE", "reasoning": f"VIX crisis: {vix_val}"}
        _audit_log(result)
        return result

    logger.info("Gathering per-underlying analysis...")
    t0 = time.time()
    analyses: dict[str, dict] = {}
    for symbol in config.UNDERLYINGS:
        expiry = _find_todays_expiry(broker, symbol)
        if not expiry:
            logger.info("%s: no 0DTE expiry today — skipping", symbol)
            continue
        analyses[symbol] = gather_underlying_analysis(broker, symbol, expiry)
    logger.info("Analysis gathered for %d underlyings in %.1fs",
                len(analyses), time.time() - t0)

    if not analyses:
        logger.info("No underlyings with 0DTE expiry today")
        result["decision"] = {"decision": "NO_TRADE", "reasoning": "No 0DTE expiry today"}
        _audit_log(result)
        return result

    # ── Build Digest ──
    try:
        account = broker.get_account()
    except Exception as e:
        logger.error("Account fetch failed: %s", e)
        account = {"equity": 0, "cash": 0, "buying_power": 0}

    digest = build_digest(
        account=account,
        positions=option_positions,
        daily_state=daily_state,
        market_context=market_context,
        analyses=analyses,
        now_et=now_et,
    )

    if digest_only:
        print(digest)
        print(f"\n--- Digest: {len(digest)} chars, ~{len(digest.split())//4*3} tokens ---")
        return result

    # ── Call LLM ──
    logger.info("Calling LLM for decision...")
    try:
        llm = get_llm_client(provider)
        logger.info("LLM client: %s", llm)
    except Exception as e:
        logger.error("LLM init failed: %s", e)
        result["error"] = f"LLM init: {e}"
        _audit_log(result)
        return result

    decision = llm.decide(digest)
    result["decision"] = decision

    logger.info("Decision: %s (confidence: %s) — %s",
                decision.get("decision"), decision.get("confidence"),
                decision.get("reasoning", "")[:100])

    if decision.get("decision") != "TRADE":
        logger.info("No trade this cycle")
        _audit_log(result)
        return result

    # ── Validate & Execute ──
    symbol = decision.get("contract_symbol", "")
    qty = decision.get("qty", 1)
    limit_price = decision.get("limit_price")
    confidence = decision.get("confidence", 0)

    if not symbol or not limit_price:
        logger.warning("TRADE decision missing contract_symbol or limit_price")
        result["error"] = "Incomplete trade decision"
        _audit_log(result)
        return result

    # Confidence gate
    if confidence < config.SIGNAL_CONFIDENCE_THRESHOLD:
        logger.info("Confidence %d < threshold %d — blocking trade",
                     confidence, config.SIGNAL_CONFIDENCE_THRESHOLD)
        result["decision"]["blocked"] = "low_confidence"
        _audit_log(result)
        return result

    # Risk validation
    validation = validate_new_order(
        symbol=symbol,
        qty=qty,
        side="buy",
        order_type="limit",
        limit_price=limit_price,
        current_positions=option_positions,
        account=account,
    )

    if not validation["approved"]:
        logger.warning("Trade BLOCKED by validator: %s", validation["reason"])
        result["decision"]["blocked"] = validation["reason"]
        _audit_log(result)
        return result

    # Execute
    if dry_run:
        logger.info("DRY RUN — would place: buy %d x %s @ $%.2f", qty, symbol, limit_price)
        result["trade"] = {"dry_run": True, "symbol": symbol, "qty": qty, "price": limit_price}
    else:
        try:
            order = broker.place_order(
                symbol=symbol,
                qty=qty,
                side="buy",
                order_type="limit",
                time_in_force="day",
                limit_price=limit_price,
            )
            logger.info("ORDER PLACED: %s", order)
            result["trade"] = order

            underlying = decision.get("underlying", "")
            direction = decision.get("direction", "")
            strike = decision.get("strike", "")
            _notify(
                f"🟡 ORDER: {direction} {underlying} ${strike} x{qty} "
                f"@ ${limit_price:.2f} | Conf: {confidence}"
            )
        except Exception as e:
            logger.error("Order placement failed: %s", e)
            result["error"] = f"Order failed: {e}"
            _notify(f"❌ ORDER FAILED: {e}")

    _audit_log(result)
    return result


# ── CLI Entry Point ─────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="0DTE Options Scalper — Orchestrator")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper",
                        help="Paper (Alpaca) or live (Public.com)")
    parser.add_argument("--provider", choices=["ollama", "anthropic", "openai"],
                        default=None, help="LLM provider (default: from config)")
    parser.add_argument("--force", action="store_true",
                        help="Run outside market hours")
    parser.add_argument("--dry-run", action="store_true",
                        help="Full pipeline but don't execute trades")
    parser.add_argument("--digest-only", action="store_true",
                        help="Print digest prompt and exit (no LLM call)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Debug logging")

    args = parser.parse_args()

    result = run_cycle(
        mode=args.mode,
        provider=args.provider,
        dry_run=args.dry_run,
        digest_only=args.digest_only,
        force=args.force,
        verbose=args.verbose,
    )

    # Print summary to stdout
    decision = result.get("decision", {})
    exits = result.get("exits", [])

    if result.get("skipped"):
        print(f"⏸ Skipped: {result['skipped']}")
    elif exits:
        for e in exits:
            emoji = "🟢" if e["pnl"] > 0 else "🔴"
            print(f"{emoji} EXIT {e['symbol']}: {e['reason']} → ${e['pnl']:+.2f}")

    if decision.get("decision") == "TRADE":
        trade = result.get("trade", {})
        if trade:
            if trade.get("dry_run"):
                print(f"🧪 DRY RUN: buy {trade['qty']}x {trade['symbol']} @ ${trade['price']:.2f}")
            else:
                print(f"🟡 ORDER: {trade.get('symbol')} → {trade.get('status')}")
    elif decision.get("decision") == "NO_TRADE":
        print(f"⏸ NO_TRADE: {decision.get('reasoning', 'unknown')[:80]}")

    if result.get("error"):
        print(f"❌ Error: {result['error']}")

    # Non-zero exit on errors (useful for cron alerting)
    if result.get("error"):
        sys.exit(1)


if __name__ == "__main__":
    main()
