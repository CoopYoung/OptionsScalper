"""Backtesting engine for the hybrid Claude + Alpaca trading system.

Simulates what Claude would see and decide during each 10-minute cycle:
  1. Check VIX regime → adjust position sizing or stand aside
  2. Analyze price bars for setups (momentum, S/R, VWAP, volume)
  3. Select options from simulated chains (target delta, spread quality)
  4. Apply validator rules (same as live: max risk, daily loss, position count)
  5. Manage exits (profit target, stop loss, trailing stop, time exit)

Uses coded heuristics that approximate Claude's decision-making — fast
enough to run months of data in seconds.

Usage:
    python -m hybrid.backtest --underlying SPY --days 10
    python -m hybrid.backtest --underlying SPY,QQQ --start 2026-03-01 --end 2026-03-20
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np

from hybrid.backtest.data_loader import (
    Bar, DaySnapshot, SimOption, generate_chain, load_days,
)

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


# ── Data Classes ─────────────────────────────────────────────

@dataclass
class Trade:
    """A single simulated trade."""
    symbol: str
    underlying: str
    option_type: str
    strike: float
    contracts: int
    entry_price: float
    entry_time: datetime
    entry_spot: float
    entry_delta: float = 0.0
    entry_gamma: float = 0.0
    entry_theta: float = 0.0
    entry_iv: float = 0.0
    entry_reason: str = ""
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    pnl: float = 0.0
    peak_price: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    @property
    def hold_minutes(self) -> float:
        if not self.exit_time:
            return 0
        return (self.exit_time - self.entry_time).total_seconds() / 60

    @property
    def risk_amount(self) -> float:
        """Max risk = premium × contracts × 100."""
        return self.entry_price * self.contracts * 100


@dataclass
class DayResult:
    """Results for a single trading day."""
    date: date
    underlying: str
    trades: list[Trade]
    vix: float
    pnl: float = 0.0
    wins: int = 0
    losses: int = 0
    signals_seen: int = 0
    signals_skipped: int = 0
    bars_count: int = 0


@dataclass
class BacktestConfig:
    """Backtest parameters — mirrors hybrid validator rules."""
    initial_capital: float = 69_500.0  # Match paper account
    max_risk_per_trade: float = 150.0
    max_daily_loss: float = 500.0
    max_concurrent_positions: int = 3
    max_contracts_per_trade: int = 5

    # Entry criteria
    entry_start: time = field(default_factory=lambda: time(9, 45))
    entry_cutoff: time = field(default_factory=lambda: time(15, 0))
    hard_close: time = field(default_factory=lambda: time(15, 45))
    confidence_threshold: float = 0.55  # Minimum signal strength
    target_delta: float = 0.30
    min_premium: float = 0.50
    max_premium: float = 5.00
    min_spread_ratio: float = 0.90  # bid/ask >= 0.90

    # Exit rules (from trading_prompt.md)
    profit_target_pct: float = 0.30   # +30% take profit
    stop_loss_pct: float = 0.40       # -40% stop loss
    trailing_trigger_pct: float = 0.20  # Activate trailing after +20%
    trailing_stop_pct: float = 0.40   # Give back 40% from peak = exit
    time_exit_minutes: int = 45       # Max hold time

    # Slippage
    slippage_pct: float = 0.02  # 2% slippage on fills

    # VIX regime adjustments
    vix_crisis_threshold: float = 35.0  # Stand aside entirely
    vix_high_threshold: float = 25.0    # Half position size
    vix_low_threshold: float = 15.0     # Can be more aggressive

    # Cycle interval (simulates cron every 10 min)
    cycle_interval_minutes: int = 10


@dataclass
class BacktestResult:
    """Full backtest results."""
    underlyings: list[str]
    start_date: date
    end_date: date
    config: BacktestConfig
    days: list[DayResult]

    @property
    def initial_capital(self) -> float:
        return self.config.initial_capital

    @property
    def total_pnl(self) -> float:
        return sum(d.pnl for d in self.days)

    @property
    def final_capital(self) -> float:
        return self.initial_capital + self.total_pnl

    @property
    def total_trades(self) -> int:
        return sum(len(d.trades) for d in self.days)

    @property
    def wins(self) -> int:
        return sum(d.wins for d in self.days)

    @property
    def losses(self) -> int:
        return sum(d.losses for d in self.days)

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl for d in self.days for t in d.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for d in self.days for t in d.trades if t.pnl < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    @property
    def max_drawdown(self) -> float:
        equity = self.initial_capital
        peak = equity
        max_dd = 0
        for d in self.days:
            equity += d.pnl
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        return max_dd

    @property
    def sharpe_ratio(self) -> float:
        daily_pnls = [d.pnl for d in self.days]
        if len(daily_pnls) < 2:
            return 0
        mean = np.mean(daily_pnls)
        std = np.std(daily_pnls, ddof=1)
        return (mean / std) * np.sqrt(252) if std > 0 else 0

    @property
    def avg_trade_pnl(self) -> float:
        all_trades = [t for d in self.days for t in d.trades]
        return np.mean([t.pnl for t in all_trades]) if all_trades else 0

    @property
    def avg_hold_minutes(self) -> float:
        all_trades = [t for d in self.days for t in d.trades if t.exit_time]
        return np.mean([t.hold_minutes for t in all_trades]) if all_trades else 0

    @property
    def avg_win(self) -> float:
        winners = [t.pnl for d in self.days for t in d.trades if t.pnl > 0]
        return np.mean(winners) if winners else 0

    @property
    def avg_loss(self) -> float:
        losers = [t.pnl for d in self.days for t in d.trades if t.pnl < 0]
        return np.mean(losers) if losers else 0

    @property
    def best_day(self) -> float:
        return max((d.pnl for d in self.days), default=0)

    @property
    def worst_day(self) -> float:
        return min((d.pnl for d in self.days), default=0)

    @property
    def no_trade_days(self) -> int:
        return sum(1 for d in self.days if len(d.trades) == 0)

    def summary(self) -> str:
        lines = [
            f"{'=' * 65}",
            f"  HYBRID BACKTEST RESULTS",
            f"  {', '.join(self.underlyings)} | {self.start_date} to {self.end_date} ({len(self.days)} days)",
            f"{'=' * 65}",
            f"",
            f"  Capital:      ${self.initial_capital:>10,.2f}  →  ${self.final_capital:>10,.2f}",
            f"  Total P&L:    ${self.total_pnl:>10,.2f}  ({self.total_pnl/self.initial_capital*100:+.2f}%)",
            f"",
            f"  Trades:       {self.total_trades:>6d}   ({self.total_trades/max(1,len(self.days)):.1f}/day avg)",
            f"  Wins:         {self.wins:>6d}   ({self.win_rate:.1%} win rate)",
            f"  Losses:       {self.losses:>6d}",
            f"  No-trade days:{self.no_trade_days:>6d}   ({self.no_trade_days/max(1,len(self.days)):.0%})",
            f"",
            f"  Avg Win:      ${self.avg_win:>10,.2f}",
            f"  Avg Loss:     ${self.avg_loss:>10,.2f}",
            f"  Avg Trade:    ${self.avg_trade_pnl:>10,.2f}",
            f"  Avg Hold:     {self.avg_hold_minutes:>6.1f} min",
            f"",
            f"  Profit Factor:{self.profit_factor:>9.2f}",
            f"  Sharpe Ratio: {self.sharpe_ratio:>9.2f}",
            f"  Max Drawdown: {self.max_drawdown:>9.2%}",
            f"  Best Day:     ${self.best_day:>10,.2f}",
            f"  Worst Day:    ${self.worst_day:>10,.2f}",
            f"{'=' * 65}",
        ]
        return "\n".join(lines)

    def daily_table(self) -> str:
        lines = [
            f"\n{'Date':<12} {'VIX':>5} {'Trades':>7} {'W/L':>6} {'P&L':>10} {'Equity':>12}",
            f"{'-' * 55}",
        ]
        equity = self.initial_capital
        for d in self.days:
            equity += d.pnl
            wl = f"{d.wins}/{d.losses}" if d.trades else "-"
            lines.append(
                f"{d.date!s:<12} {d.vix:>5.1f} {len(d.trades):>7d} {wl:>6} "
                f"${d.pnl:>9,.2f} ${equity:>11,.2f}"
            )
        return "\n".join(lines)

    def trade_log(self) -> str:
        lines = [
            f"\n{'Time':<20} {'Symbol':<25} {'Type':>5} {'Qty':>4} "
            f"{'Entry':>7} {'Exit':>7} {'P&L':>9} {'Hold':>6} {'Reason':<30}",
            f"{'-' * 120}",
        ]
        for d in self.days:
            for t in d.trades:
                et = t.entry_time.astimezone(ET).strftime("%m/%d %H:%M")
                lines.append(
                    f"{et:<20} {t.symbol:<25} {t.option_type:>5} {t.contracts:>4} "
                    f"${t.entry_price:>6.2f} ${t.exit_price:>6.2f} "
                    f"${t.pnl:>8,.2f} {t.hold_minutes:>5.0f}m {t.exit_reason:<30}"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize results for JSON output."""
        return {
            "summary": {
                "underlyings": self.underlyings,
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat(),
                "initial_capital": self.initial_capital,
                "final_capital": self.final_capital,
                "total_pnl": round(self.total_pnl, 2),
                "total_trades": self.total_trades,
                "win_rate": round(self.win_rate, 4),
                "profit_factor": round(self.profit_factor, 4),
                "sharpe_ratio": round(self.sharpe_ratio, 4),
                "max_drawdown": round(self.max_drawdown, 4),
                "avg_trade_pnl": round(self.avg_trade_pnl, 2),
                "avg_hold_minutes": round(self.avg_hold_minutes, 1),
            },
            "days": [
                {
                    "date": d.date.isoformat(),
                    "vix": d.vix,
                    "trades": len(d.trades),
                    "pnl": round(d.pnl, 2),
                    "trade_details": [
                        {
                            "symbol": t.symbol,
                            "type": t.option_type,
                            "contracts": t.contracts,
                            "entry_price": t.entry_price,
                            "exit_price": t.exit_price,
                            "pnl": round(t.pnl, 2),
                            "hold_minutes": round(t.hold_minutes, 1),
                            "entry_reason": t.entry_reason,
                            "exit_reason": t.exit_reason,
                            "entry_delta": t.entry_delta,
                            "entry_iv": t.entry_iv,
                        }
                        for t in d.trades
                    ],
                }
                for d in self.days
            ],
        }


# ── Technical Analysis (approximates Claude's chart reading) ─

class TechnicalAnalyzer:
    """Compute technical signals from price bars."""

    def __init__(self):
        self._closes = deque(maxlen=200)
        self._volumes = deque(maxlen=50)
        self._highs = deque(maxlen=50)
        self._lows = deque(maxlen=50)

    def update(self, bar: Bar):
        self._closes.append(bar.close)
        self._volumes.append(bar.volume)
        self._highs.append(bar.high)
        self._lows.append(bar.low)

    def reset(self):
        self._closes.clear()
        self._volumes.clear()
        self._highs.clear()
        self._lows.clear()

    def has_enough_data(self) -> bool:
        return len(self._closes) >= 30

    def rsi(self, period: int = 14) -> Optional[float]:
        if len(self._closes) < period + 1:
            return None
        closes = list(self._closes)
        deltas = [closes[i] - closes[i-1] for i in range(-period, 0)]
        gains = [d for d in deltas if d > 0]
        losses = [-d for d in deltas if d < 0]
        avg_gain = sum(gains) / period if gains else 0
        avg_loss = sum(losses) / period if losses else 0.001
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def vwap_position(self) -> Optional[float]:
        """Price position relative to VWAP: >0 = above, <0 = below."""
        if len(self._closes) < 10:
            return None
        closes = list(self._closes)[-20:]
        vols = list(self._volumes)[-20:]
        if sum(vols) == 0:
            return None
        vwap = sum(c * v for c, v in zip(closes, vols)) / sum(vols)
        current = closes[-1]
        return (current - vwap) / vwap * 100

    def momentum(self, lookback: int = 5) -> Optional[float]:
        """Price momentum: % change over lookback bars."""
        if len(self._closes) < lookback + 1:
            return None
        current = self._closes[-1]
        prior = self._closes[-(lookback + 1)]
        return (current - prior) / prior * 100

    def volume_surge(self, lookback: int = 20) -> Optional[float]:
        """Current volume relative to average."""
        if len(self._volumes) < lookback:
            return None
        avg_vol = np.mean(list(self._volumes)[-lookback:])
        current_vol = self._volumes[-1]
        return current_vol / avg_vol if avg_vol > 0 else 1.0

    def support_resistance_bounce(self) -> Optional[str]:
        """Detect if price is bouncing off support or resistance."""
        if len(self._closes) < 20 or len(self._lows) < 20:
            return None

        closes = list(self._closes)
        lows = list(self._lows)[-20:]
        highs = list(self._highs)[-20:]
        current = closes[-1]

        # Recent low = potential support
        recent_low = min(lows[:-2]) if len(lows) > 2 else min(lows)
        recent_high = max(highs[:-2]) if len(highs) > 2 else max(highs)

        # Near support + bouncing up
        if abs(current - recent_low) / current < 0.003 and closes[-1] > closes[-2]:
            return "SUPPORT_BOUNCE"
        # Near resistance + failing
        if abs(current - recent_high) / current < 0.003 and closes[-1] < closes[-2]:
            return "RESISTANCE_REJECTION"

        return None

    def evaluate_setup(self, vix: float) -> tuple[Optional[str], float, str]:
        """Evaluate for a trade setup.

        Returns: (direction, confidence, reason) or (None, 0, "") if no setup.
        direction is "call" or "put".
        confidence is 0-1 scale.
        """
        if not self.has_enough_data():
            return None, 0, "Insufficient data"

        rsi = self.rsi()
        mom = self.momentum()
        vwap = self.vwap_position()
        vol_surge = self.volume_surge()
        sr_bounce = self.support_resistance_bounce()

        if rsi is None or mom is None:
            return None, 0, "Indicators not ready"

        score = 0.0
        reasons = []

        # RSI
        if rsi < 30:
            score += 0.25
            reasons.append(f"RSI oversold ({rsi:.0f})")
        elif rsi > 70:
            score -= 0.25
            reasons.append(f"RSI overbought ({rsi:.0f})")
        elif rsi < 40:
            score += 0.10
        elif rsi > 60:
            score -= 0.10

        # Momentum
        if mom > 0.15:
            score += 0.20
            reasons.append(f"Momentum +{mom:.2f}%")
        elif mom < -0.15:
            score -= 0.20
            reasons.append(f"Momentum {mom:.2f}%")

        # VWAP position
        if vwap is not None:
            if vwap > 0.05:
                score += 0.15
                reasons.append("Above VWAP")
            elif vwap < -0.05:
                score -= 0.15
                reasons.append("Below VWAP")

        # Volume confirmation
        if vol_surge is not None and vol_surge > 1.5:
            score *= 1.2  # Amplify signal on high volume
            reasons.append(f"Volume surge ({vol_surge:.1f}x)")
        elif vol_surge is not None and vol_surge < 0.5:
            score *= 0.5  # Dampen signal on low volume
            reasons.append("Low volume — weak signal")

        # Support/Resistance
        if sr_bounce == "SUPPORT_BOUNCE":
            score += 0.20
            reasons.append("Support bounce")
        elif sr_bounce == "RESISTANCE_REJECTION":
            score -= 0.20
            reasons.append("Resistance rejection")

        # VIX regime adjustment
        if vix > 35:
            return None, 0, "VIX CRISIS — standing aside"
        elif vix > 25:
            score *= 0.7  # Reduce conviction in high vol
            reasons.append(f"VIX high ({vix:.1f}) — reduced confidence")

        # Determine direction and confidence
        abs_score = abs(score)
        if abs_score < 0.30:
            return None, abs_score, "No clear setup"

        direction = "call" if score > 0 else "put"
        confidence = min(1.0, abs_score)

        return direction, confidence, " | ".join(reasons)


# ── Main Backtest Engine ─────────────────────────────────────

class HybridBacktester:
    """Simulates the hybrid trading system over historical data."""

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        underlyings: list[str],
        start_date: date,
        end_date: date,
        timeframe: str = "5Min",
    ) -> BacktestResult:
        """Run backtest across multiple underlyings."""
        all_day_results = []

        for underlying in underlyings:
            logger.info(f"Loading data for {underlying}...")
            days = load_days(underlying, start_date, end_date, timeframe)

            if not days:
                logger.warning(f"No data for {underlying}")
                continue

            for day_data in days:
                result = self._backtest_day(day_data)
                all_day_results.append(result)

        # Sort by date
        all_day_results.sort(key=lambda d: d.date)

        return BacktestResult(
            underlyings=underlyings,
            start_date=start_date,
            end_date=end_date,
            config=self.config,
            days=all_day_results,
        )

    def _backtest_day(self, day: DaySnapshot) -> DayResult:
        """Simulate one trading day."""
        cfg = self.config
        tech = TechnicalAnalyzer()
        open_trades: list[Trade] = []
        closed_trades: list[Trade] = []
        daily_pnl = 0.0
        signals_seen = 0
        signals_skipped = 0

        # Track when we last "ran a cycle" (every 10 min like cron)
        last_cycle_time = None

        for bar in day.bars:
            et_time = bar.timestamp.astimezone(ET)
            current_time = et_time.time()
            tech.update(bar)

            # ── Check exits every bar ────────────────────────
            for trade in list(open_trades):
                current_price = self._reprice(trade, bar.close, et_time)
                trade.peak_price = max(trade.peak_price, current_price)

                should_exit, reason = self._check_exit(trade, current_price, et_time)
                if should_exit:
                    exit_price = current_price * (1 - cfg.slippage_pct)
                    trade.exit_price = round(exit_price, 2)
                    trade.exit_time = bar.timestamp
                    trade.exit_reason = reason
                    trade.pnl = round(
                        (exit_price - trade.entry_price) * trade.contracts * 100, 2
                    )
                    daily_pnl += trade.pnl
                    open_trades.remove(trade)
                    closed_trades.append(trade)

            # ── Simulate 10-min cycle for entries ────────────
            if last_cycle_time is None or (et_time - last_cycle_time).total_seconds() >= cfg.cycle_interval_minutes * 60:
                last_cycle_time = et_time

                # Gate checks
                if current_time < cfg.entry_start or current_time > cfg.entry_cutoff:
                    continue
                if daily_pnl <= -cfg.max_daily_loss:
                    continue
                if len(open_trades) >= cfg.max_concurrent_positions:
                    continue
                if day.vix >= cfg.vix_crisis_threshold:
                    continue

                # Evaluate setup
                direction, confidence, reason = tech.evaluate_setup(day.vix)
                if direction is None or confidence < cfg.confidence_threshold:
                    continue

                signals_seen += 1

                # Generate chain and select option
                minutes_left = self._minutes_until_close(et_time)
                chain = generate_chain(
                    spot=bar.close,
                    vix=day.vix,
                    minutes_to_close=minutes_left,
                    underlying=day.underlying,
                    expiry=day.date.isoformat(),
                )

                option = self._select_option(chain, direction, bar.close)
                if option is None:
                    signals_skipped += 1
                    continue

                # Size the position
                contracts = self._size_position(option, day.vix)
                if contracts == 0:
                    signals_skipped += 1
                    continue

                # Validate risk
                total_risk = option.mid * contracts * 100
                if total_risk > cfg.max_risk_per_trade:
                    contracts = max(1, int(cfg.max_risk_per_trade / (option.mid * 100)))
                    total_risk = option.mid * contracts * 100

                # Enter trade
                entry_price = round(option.mid * (1 + cfg.slippage_pct), 2)
                trade = Trade(
                    symbol=option.symbol,
                    underlying=day.underlying,
                    option_type=direction,
                    strike=option.strike,
                    contracts=contracts,
                    entry_price=entry_price,
                    entry_time=bar.timestamp,
                    entry_spot=bar.close,
                    entry_delta=option.delta,
                    entry_gamma=option.gamma,
                    entry_theta=option.theta,
                    entry_iv=option.iv,
                    entry_reason=reason,
                    peak_price=entry_price,
                )
                open_trades.append(trade)

        # Force close any remaining at end of day
        if day.bars and open_trades:
            last_bar = day.bars[-1]
            last_et = last_bar.timestamp.astimezone(ET)
            for trade in open_trades:
                price = self._reprice(trade, last_bar.close, last_et)
                exit_price = price * (1 - cfg.slippage_pct)
                trade.exit_price = round(exit_price, 2)
                trade.exit_time = last_bar.timestamp
                trade.exit_reason = "End of day close"
                trade.pnl = round(
                    (exit_price - trade.entry_price) * trade.contracts * 100, 2
                )
                daily_pnl += trade.pnl
                closed_trades.append(trade)

        wins = sum(1 for t in closed_trades if t.pnl > 0)
        losses = sum(1 for t in closed_trades if t.pnl < 0)

        return DayResult(
            date=day.date,
            underlying=day.underlying,
            trades=closed_trades,
            vix=day.vix,
            pnl=round(daily_pnl, 2),
            wins=wins,
            losses=losses,
            signals_seen=signals_seen,
            signals_skipped=signals_skipped,
            bars_count=len(day.bars),
        )

    def _select_option(self, chain: list[SimOption], direction: str, spot: float) -> Optional[SimOption]:
        """Select the best option for a given direction."""
        cfg = self.config
        candidates = [o for o in chain if o.option_type == direction]

        # Filter
        filtered = []
        for o in candidates:
            if o.mid < cfg.min_premium or o.mid > cfg.max_premium:
                continue
            if o.bid <= 0.01 or o.ask <= 0:
                continue
            spread_ratio = o.bid / o.ask if o.ask > 0 else 0
            if spread_ratio < cfg.min_spread_ratio:
                continue
            filtered.append(o)

        if not filtered:
            return None

        # Sort by distance from target delta
        target = cfg.target_delta if direction == "call" else -cfg.target_delta
        filtered.sort(key=lambda o: abs(abs(o.delta) - cfg.target_delta))

        return filtered[0]

    def _size_position(self, option: SimOption, vix: float) -> int:
        """Position sizing — buying-power based with VIX adjustment."""
        cfg = self.config
        max_risk = cfg.max_risk_per_trade
        cost_per_contract = option.mid * 100

        if cost_per_contract <= 0:
            return 0

        contracts = int(max_risk / cost_per_contract)
        contracts = min(contracts, cfg.max_contracts_per_trade)

        # VIX adjustment
        if vix > cfg.vix_high_threshold:
            contracts = max(1, contracts // 2)

        return max(1, contracts) if contracts > 0 else 0

    def _check_exit(self, trade: Trade, current_price: float, et_time: datetime) -> tuple[bool, str]:
        """Check exit conditions per trading_prompt.md rules."""
        cfg = self.config
        entry = trade.entry_price
        if entry <= 0:
            return False, ""

        pnl_pct = (current_price - entry) / entry
        hold_min = (et_time - trade.entry_time.astimezone(ET)).total_seconds() / 60

        # Profit target: +30%
        if pnl_pct >= cfg.profit_target_pct:
            return True, f"Profit target ({pnl_pct:.0%})"

        # Stop loss: -40%
        if pnl_pct <= -cfg.stop_loss_pct:
            return True, f"Stop loss ({pnl_pct:.0%})"

        # Trailing stop
        if trade.peak_price > entry * (1 + cfg.trailing_trigger_pct):
            retrace = (trade.peak_price - current_price) / trade.peak_price
            if retrace >= cfg.trailing_stop_pct:
                return True, f"Trailing stop ({retrace:.0%} from peak)"

        # Time exit
        if hold_min >= cfg.time_exit_minutes:
            return True, f"Time exit ({hold_min:.0f}m)"

        # Hard close
        if et_time.time() >= cfg.hard_close:
            return True, "Hard close (3:45 PM)"

        return False, ""

    def _reprice(self, trade: Trade, current_spot: float, et_time: datetime) -> float:
        """Greeks-based repricing (Taylor expansion)."""
        ds = current_spot - trade.entry_spot
        hold_min = (et_time - trade.entry_time.astimezone(ET)).total_seconds() / 60
        dt_days = hold_min / 390  # Trading minutes per day

        delta_pnl = trade.entry_delta * ds
        gamma_pnl = 0.5 * trade.entry_gamma * ds * ds
        theta_pnl = trade.entry_theta * dt_days

        estimated = trade.entry_price + delta_pnl + gamma_pnl + theta_pnl
        return max(0.01, estimated)

    def _minutes_until_close(self, et_time: datetime) -> float:
        close = et_time.replace(hour=16, minute=0, second=0, microsecond=0)
        return max(1, (close - et_time).total_seconds() / 60)
