"""Trade analytics for signal evaluation and performance attribution.

Answers the questions that matter for tuning:
    - Which factors actually predict profitable trades?
    - What time of day produces the best/worst results?
    - Is the confidence score well-calibrated?
    - What exit reasons dominate and what does that imply?
    - How does hold time correlate with P&L?
"""

import json
import logging
from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from src.data.trade_db import TradeDB

logger = logging.getLogger(__name__)

FACTOR_NAMES = [
    "technical", "tick_momentum", "gex", "flow",
    "vix", "internals", "sentiment", "optionsai",
]


class TradeAnalyzer:
    """Analyzes closed trades to evaluate signal and factor performance."""

    def __init__(self, db: TradeDB) -> None:
        self._db = db

    def daily_report(self, trade_date: Optional[date] = None) -> dict:
        """Generate a comprehensive daily analytics report.

        Returns a dict suitable for JSON storage and Telegram summary.
        """
        trade_date = trade_date or date.today()
        trades = self._get_closed_trades(trade_date)

        if not trades:
            return {
                "date": trade_date.isoformat(),
                "total_trades": 0,
                "summary": "No trades today.",
            }

        report = {
            "date": trade_date.isoformat(),
            "total_trades": len(trades),
            "performance": self._compute_performance(trades),
            "factor_attribution": self._compute_factor_attribution(trades),
            "time_of_day": self._compute_time_buckets(trades),
            "confidence_calibration": self._compute_confidence_calibration(trades),
            "exit_reasons": self._compute_exit_reasons(trades),
            "hold_time_analysis": self._compute_hold_time_analysis(trades),
        }

        # Store in DB
        self._store_report(trade_date, report)

        return report

    def format_telegram_summary(self, report: dict) -> str:
        """Format analytics report for Telegram message."""
        if report.get("total_trades", 0) == 0:
            return f"*Analytics ({report['date']})*\nNo trades today."

        perf = report.get("performance", {})
        lines = [
            f"*Daily Analytics ({report['date']})*",
            f"",
            f"*Performance*",
            f"Trades: {report['total_trades']}",
            f"Win Rate: {perf.get('win_rate', 0):.1%}",
            f"Total P&L: ${perf.get('total_pnl', 0):+,.2f}",
            f"Avg P&L: ${perf.get('avg_pnl', 0):+,.2f}",
            f"Best: ${perf.get('best_trade', 0):+,.2f}",
            f"Worst: ${perf.get('worst_trade', 0):+,.2f}",
            f"Avg Hold: {perf.get('avg_hold_minutes', 0):.1f}min",
        ]

        # Factor attribution (top 3 by win rate)
        factors = report.get("factor_attribution", {})
        if factors:
            sorted_factors = sorted(
                factors.items(),
                key=lambda x: x[1].get("win_rate", 0),
                reverse=True,
            )
            lines.append("")
            lines.append("*Factor Win Rates*")
            for name, data in sorted_factors[:4]:
                count = data.get("dominant_count", 0)
                wr = data.get("win_rate", 0)
                if count > 0:
                    lines.append(f"  {name}: {wr:.0%} ({count} trades)")

        # Exit reasons
        exits = report.get("exit_reasons", {})
        if exits:
            lines.append("")
            lines.append("*Exit Reasons*")
            for reason, count in sorted(exits.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  {reason}: {count}")

        # Confidence calibration
        cal = report.get("confidence_calibration", {})
        if cal:
            lines.append("")
            lines.append("*Confidence Calibration*")
            for bucket, data in sorted(cal.items()):
                actual = data.get("actual_win_rate", 0)
                count = data.get("count", 0)
                if count > 0:
                    lines.append(f"  {bucket}: {actual:.0%} actual ({count} trades)")

        return "\n".join(lines)

    def get_factor_performance(
        self, factor: str, lookback_days: int = 30,
    ) -> dict:
        """Get performance stats for a specific factor."""
        trades = self._get_recent_closed_trades(lookback_days)
        wins = 0
        losses = 0
        total_pnl = 0.0

        for trade in trades:
            breakdown = self._parse_quant_signals(trade)
            if not breakdown:
                continue

            score = breakdown.get(factor, 0)
            pnl = float(trade.get("pnl") or 0)

            # Factor was "right" if score direction matches P&L direction
            if (score > 0 and pnl > 0) or (score < 0 and pnl > 0):
                wins += 1
            elif pnl < 0:
                losses += 1
            total_pnl += pnl

        total = wins + losses
        return {
            "factor": factor,
            "trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total if total > 0 else 0,
            "total_pnl": round(total_pnl, 2),
        }

    def get_rolling_factor_sharpe(self, lookback_trades: int = 20) -> dict[str, float]:
        """Compute rolling Sharpe-like ratio per factor.

        For each factor: sharpe ≈ mean(factor_score × sign(pnl)) / std(factor_score × sign(pnl))
        High Sharpe = factor consistently predicts winners.
        """
        trades = self._get_recent_closed_trades(days=60)
        trades = trades[:lookback_trades]  # Most recent N

        factor_returns: dict[str, list[float]] = defaultdict(list)

        for trade in trades:
            breakdown = self._parse_quant_signals(trade)
            if not breakdown:
                continue

            pnl = float(trade.get("pnl") or 0)
            pnl_sign = 1.0 if pnl > 0 else (-1.0 if pnl < 0 else 0.0)

            for factor in FACTOR_NAMES:
                score = breakdown.get(factor, 0)
                factor_returns[factor].append(score * pnl_sign)

        result = {}
        for factor in FACTOR_NAMES:
            returns = factor_returns.get(factor, [])
            if len(returns) < 5:
                result[factor] = 0.0
                continue
            mean = sum(returns) / len(returns)
            variance = sum((r - mean) ** 2 for r in returns) / len(returns)
            std = variance ** 0.5
            result[factor] = round(mean / std, 4) if std > 0.001 else 0.0

        return result

    # ── Private helpers ────────────────────────────────────────

    def _get_closed_trades(self, trade_date: date) -> list[dict]:
        """Get all closed trades for a specific date."""
        assert self._db._conn is not None
        date_str = trade_date.isoformat()
        cur = self._db._conn.execute(
            """SELECT * FROM trades
               WHERE exit_time IS NOT NULL
                 AND date(entry_time) = ?
               ORDER BY entry_time""",
            (date_str,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def _get_recent_closed_trades(self, days: int = 30) -> list[dict]:
        """Get recent closed trades across multiple days."""
        assert self._db._conn is not None
        cur = self._db._conn.execute(
            """SELECT * FROM trades
               WHERE exit_time IS NOT NULL
               ORDER BY entry_time DESC
               LIMIT 500""",
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def _parse_quant_signals(self, trade: dict) -> dict:
        """Parse the quant_signals JSON column."""
        raw = trade.get("quant_signals", "")
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}

    def _compute_performance(self, trades: list[dict]) -> dict:
        """Basic P&L stats."""
        pnls = [float(t.get("pnl") or 0) for t in trades]
        hold_secs = [int(t.get("hold_seconds") or 0) for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p < 0)
        total = wins + losses

        return {
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total if total > 0 else 0,
            "total_pnl": round(sum(pnls), 2),
            "avg_pnl": round(sum(pnls) / len(pnls), 2) if pnls else 0,
            "best_trade": round(max(pnls), 2) if pnls else 0,
            "worst_trade": round(min(pnls), 2) if pnls else 0,
            "avg_hold_minutes": round(sum(hold_secs) / len(hold_secs) / 60, 1) if hold_secs else 0,
        }

    def _compute_factor_attribution(self, trades: list[dict]) -> dict:
        """For each factor, compute win rate when it was the dominant signal."""
        factor_stats: dict[str, dict] = {f: {"dominant_count": 0, "wins": 0} for f in FACTOR_NAMES}

        for trade in trades:
            breakdown = self._parse_quant_signals(trade)
            if not breakdown:
                continue

            # Find dominant factor (highest absolute score)
            dominant = max(
                FACTOR_NAMES,
                key=lambda f: abs(breakdown.get(f, 0)),
            )
            pnl = float(trade.get("pnl") or 0)

            factor_stats[dominant]["dominant_count"] += 1
            if pnl > 0:
                factor_stats[dominant]["wins"] += 1

        # Compute win rates
        result = {}
        for factor, stats in factor_stats.items():
            count = stats["dominant_count"]
            result[factor] = {
                "dominant_count": count,
                "wins": stats["wins"],
                "win_rate": stats["wins"] / count if count > 0 else 0,
            }
        return result

    def _compute_time_buckets(self, trades: list[dict]) -> dict:
        """P&L by 30-minute time bucket (ET)."""
        buckets: dict[str, dict] = defaultdict(lambda: {"count": 0, "total_pnl": 0.0, "wins": 0})

        for trade in trades:
            entry_str = trade.get("entry_time", "")
            if not entry_str:
                continue
            try:
                entry_dt = datetime.fromisoformat(entry_str)
                # Round to 30-min bucket
                minute = (entry_dt.minute // 30) * 30
                bucket = f"{entry_dt.hour:02d}:{minute:02d}"
            except (ValueError, TypeError):
                continue

            pnl = float(trade.get("pnl") or 0)
            buckets[bucket]["count"] += 1
            buckets[bucket]["total_pnl"] += pnl
            if pnl > 0:
                buckets[bucket]["wins"] += 1

        return {
            k: {**v, "avg_pnl": round(v["total_pnl"] / v["count"], 2) if v["count"] > 0 else 0}
            for k, v in sorted(buckets.items())
        }

    def _compute_confidence_calibration(self, trades: list[dict]) -> dict:
        """Check if confidence scores predict win rate.

        Groups trades into confidence buckets (55-65, 65-75, 75-85, 85+)
        and checks actual win rate vs predicted.
        """
        buckets = {
            "55-64": {"count": 0, "wins": 0},
            "65-74": {"count": 0, "wins": 0},
            "75-84": {"count": 0, "wins": 0},
            "85+": {"count": 0, "wins": 0},
        }

        for trade in trades:
            conf = int(trade.get("confidence") or 0)
            pnl = float(trade.get("pnl") or 0)

            if conf < 55:
                continue
            elif conf < 65:
                bucket = "55-64"
            elif conf < 75:
                bucket = "65-74"
            elif conf < 85:
                bucket = "75-84"
            else:
                bucket = "85+"

            buckets[bucket]["count"] += 1
            if pnl > 0:
                buckets[bucket]["wins"] += 1

        return {
            k: {
                **v,
                "actual_win_rate": v["wins"] / v["count"] if v["count"] > 0 else 0,
            }
            for k, v in buckets.items()
        }

    def _compute_exit_reasons(self, trades: list[dict]) -> dict:
        """Count exit reasons — helps identify if stops are too tight/loose."""
        reasons: dict[str, int] = defaultdict(int)
        for trade in trades:
            reason = trade.get("exit_reason", "unknown")
            # Simplify reason to category
            if "Profit target" in reason:
                reasons["Profit target"] += 1
            elif "Catastrophic" in reason:
                reasons["Catastrophic stop"] += 1
            elif "Greeks stop" in reason or "stop" in reason.lower():
                reasons["Stop loss"] += 1
            elif "trail" in reason.lower() or "retrace" in reason.lower():
                reasons["Trailing stop"] += 1
            elif "hold" in reason.lower() or "timeout" in reason.lower():
                reasons["Hold timeout"] += 1
            elif "Time exit" in reason:
                reasons["Time exit"] += 1
            elif "End of day" in reason:
                reasons["End of day"] += 1
            else:
                reasons[reason[:30]] += 1
        return dict(reasons)

    def _compute_hold_time_analysis(self, trades: list[dict]) -> dict:
        """Analyze P&L vs hold time buckets."""
        buckets = {
            "0-5min": {"count": 0, "total_pnl": 0.0},
            "5-10min": {"count": 0, "total_pnl": 0.0},
            "10-20min": {"count": 0, "total_pnl": 0.0},
            "20-30min": {"count": 0, "total_pnl": 0.0},
            "30min+": {"count": 0, "total_pnl": 0.0},
        }
        for trade in trades:
            hold_sec = int(trade.get("hold_seconds") or 0)
            hold_min = hold_sec / 60
            pnl = float(trade.get("pnl") or 0)

            if hold_min < 5:
                bucket = "0-5min"
            elif hold_min < 10:
                bucket = "5-10min"
            elif hold_min < 20:
                bucket = "10-20min"
            elif hold_min < 30:
                bucket = "20-30min"
            else:
                bucket = "30min+"

            buckets[bucket]["count"] += 1
            buckets[bucket]["total_pnl"] += pnl

        return {
            k: {**v, "avg_pnl": round(v["total_pnl"] / v["count"], 2) if v["count"] > 0 else 0}
            for k, v in buckets.items()
        }

    def _store_report(self, trade_date: date, report: dict) -> None:
        """Store daily report in DB."""
        try:
            assert self._db._conn is not None
            self._db._conn.execute(
                """INSERT OR REPLACE INTO daily_reports (trade_date, report_json)
                   VALUES (?, ?)""",
                (trade_date.isoformat(), json.dumps(report)),
            )
            self._db._conn.commit()
        except Exception:
            logger.exception("Failed to store daily report")
