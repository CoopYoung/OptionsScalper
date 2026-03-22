"""Backtesting engine for Zero-DTE options scalping strategy.

Replays historical intraday data through the signal ensemble and risk
manager, simulating order fills with configurable slippage and latency.

Usage:
    python -m src.backtest --underlying SPY --start 2026-03-01 --end 2026-03-15
"""

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np

from src.backtest.data_loader import (
    BacktestDay, HistoricalBar, HistoricalDataLoader,
    OptionPricer, SimulatedOption,
)
from src.infra.config import Settings
from src.quant.flow import FlowAnalyzer
from src.quant.gex import GEXAnalyzer
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.manager import OptionsRiskManager
from src.strategy.base import BaseStrategy, OptionsContract, TradeDirection, TradeSignal
from src.strategy.signals import compute_all_signals, SignalBundle

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

# Market hours (ET)
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


@dataclass
class BacktestTrade:
    """A single simulated trade."""
    underlying: str
    symbol: str
    option_type: str
    strike: float
    direction: str
    contracts: int
    entry_price: float
    entry_time: datetime
    entry_confidence: int
    entry_spot: float = 0.0       # Underlying price at entry (for directional P&L)
    entry_delta: float = 0.0      # Option delta at entry
    entry_gamma: float = 0.0      # Option gamma at entry
    entry_theta: float = 0.0      # Option theta at entry (per day)
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    pnl: float = 0.0
    peak_price: float = 0.0
    peak_spot: float = 0.0        # Highest favorable underlying price seen
    score_breakdown: dict = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    @property
    def hold_minutes(self) -> float:
        if not self.exit_time:
            return 0
        return (self.exit_time - self.entry_time).total_seconds() / 60


@dataclass
class DayResult:
    """Results for a single backtested day."""
    date: date
    underlying: str
    trades: list[BacktestTrade]
    total_pnl: float = 0.0
    wins: int = 0
    losses: int = 0
    max_drawdown: float = 0.0
    signals_generated: int = 0
    signals_blocked: int = 0
    vix: float = 0.0


@dataclass
class BacktestResult:
    """Full backtest results across all days."""
    underlying: str
    start_date: date
    end_date: date
    initial_capital: float
    final_capital: float
    days: list[DayResult]
    config: dict

    @property
    def total_pnl(self) -> float:
        return sum(d.total_pnl for d in self.days)

    @property
    def total_trades(self) -> int:
        return sum(len(d.trades) for d in self.days)

    @property
    def win_count(self) -> int:
        return sum(d.wins for d in self.days)

    @property
    def loss_count(self) -> int:
        return sum(d.losses for d in self.days)

    @property
    def win_rate(self) -> float:
        total = self.win_count + self.loss_count
        return self.win_count / total if total > 0 else 0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(
            t.pnl for d in self.days for t in d.trades if t.pnl > 0
        )
        gross_loss = abs(sum(
            t.pnl for d in self.days for t in d.trades if t.pnl < 0
        ))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    @property
    def max_drawdown(self) -> float:
        equity_curve = []
        equity = self.initial_capital
        for day in self.days:
            equity += day.total_pnl
            equity_curve.append(equity)
        if not equity_curve:
            return 0
        peak = equity_curve[0]
        max_dd = 0
        for eq in equity_curve:
            peak = max(peak, eq)
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        return max_dd

    @property
    def sharpe_ratio(self) -> float:
        daily_returns = [d.total_pnl for d in self.days]
        if len(daily_returns) < 2:
            return 0
        mean_r = np.mean(daily_returns)
        std_r = np.std(daily_returns, ddof=1)
        if std_r == 0:
            return 0
        return (mean_r / std_r) * np.sqrt(252)

    @property
    def avg_trade_pnl(self) -> float:
        all_trades = [t for d in self.days for t in d.trades]
        if not all_trades:
            return 0
        return np.mean([t.pnl for t in all_trades])

    @property
    def avg_hold_minutes(self) -> float:
        all_trades = [t for d in self.days for t in d.trades if t.exit_time]
        if not all_trades:
            return 0
        return np.mean([t.hold_minutes for t in all_trades])

    def summary(self) -> str:
        lines = [
            f"{'=' * 60}",
            f"  BACKTEST RESULTS: {self.underlying}",
            f"  {self.start_date} to {self.end_date} ({len(self.days)} trading days)",
            f"{'=' * 60}",
            f"",
            f"  Initial Capital:  ${self.initial_capital:>12,.2f}",
            f"  Final Capital:    ${self.final_capital:>12,.2f}",
            f"  Total P&L:        ${self.total_pnl:>12,.2f} ({self.total_pnl/self.initial_capital*100:+.2f}%)",
            f"",
            f"  Total Trades:     {self.total_trades:>6d}",
            f"  Wins:             {self.win_count:>6d}",
            f"  Losses:           {self.loss_count:>6d}",
            f"  Win Rate:         {self.win_rate:>6.1%}",
            f"  Profit Factor:    {self.profit_factor:>9.2f}",
            f"",
            f"  Avg Trade P&L:    ${self.avg_trade_pnl:>12,.2f}",
            f"  Avg Hold Time:    {self.avg_hold_minutes:>6.1f} min",
            f"  Max Drawdown:     {self.max_drawdown:>6.2%}",
            f"  Sharpe Ratio:     {self.sharpe_ratio:>9.2f}",
            f"{'=' * 60}",
        ]
        # Exit reason breakdown
        from collections import Counter
        all_trades = [t for d in self.days for t in d.trades]
        if all_trades:
            reasons = Counter()
            reason_pnl: dict[str, float] = {}
            for t in all_trades:
                # Normalize reason to category
                r = t.exit_reason.split("(")[0].strip()
                reasons[r] += 1
                reason_pnl[r] = reason_pnl.get(r, 0) + t.pnl
            lines.append("")
            lines.append("  Exit Reasons:")
            for reason, count in reasons.most_common():
                avg = reason_pnl[reason] / count
                lines.append(
                    f"    {reason:<30s} {count:>4d} trades  avg ${avg:>7.2f}"
                )
            # Avg winner / avg loser
            winners = [t.pnl for t in all_trades if t.pnl > 0]
            losers = [t.pnl for t in all_trades if t.pnl < 0]
            if winners:
                lines.append(f"\n  Avg Winner:       ${np.mean(winners):>12,.2f}")
            if losers:
                lines.append(f"  Avg Loser:        ${np.mean(losers):>12,.2f}")
            if winners and losers:
                lines.append(f"  Win/Loss Ratio:   {abs(np.mean(winners)/np.mean(losers)):>9.2f}")
            lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    def daily_summary(self) -> str:
        lines = [
            f"\n{'Date':<12} {'P&L':>10} {'Trades':>7} {'Wins':>5} {'Losses':>7} {'VIX':>6}",
            f"{'-'*50}",
        ]
        for d in self.days:
            lines.append(
                f"{d.date!s:<12} ${d.total_pnl:>9,.2f} {len(d.trades):>7d} "
                f"{d.wins:>5d} {d.losses:>7d} {d.vix:>6.1f}"
            )
        return "\n".join(lines)


class SlippageModel:
    """Models execution slippage for backtesting."""

    def __init__(self, slippage_pct: float = 0.005, latency_bars: int = 1) -> None:
        self.slippage_pct = slippage_pct
        self.latency_bars = latency_bars

    def apply_entry(self, mid_price: float, option_type: str) -> float:
        """Entry fills slightly worse than mid (pay more)."""
        return mid_price * (1 + self.slippage_pct)

    def apply_exit(self, mid_price: float, option_type: str) -> float:
        """Exit fills slightly worse than mid (receive less)."""
        return mid_price * (1 - self.slippage_pct)


class BacktestEngine:
    """Replay historical data through the strategy ensemble."""

    def __init__(
        self,
        settings: Settings,
        slippage: Optional[SlippageModel] = None,
        initial_capital: float = 100_000,
    ) -> None:
        self._settings = settings
        self._slippage = slippage or SlippageModel()
        self._initial_capital = initial_capital
        self._capital = initial_capital
        self._pricer = OptionPricer()
        self._loader = HistoricalDataLoader(settings.underlying_list)
        self._gex = GEXAnalyzer(settings)
        self._flow = FlowAnalyzer(settings)

    async def run(
        self,
        underlying: str,
        start_date: date,
        end_date: date,
        interval: str = "2m",
    ) -> BacktestResult:
        """Run full backtest over a date range."""
        logger.info("Starting backtest: %s %s to %s", underlying, start_date, end_date)

        days_data = self._loader.load_days(underlying, start_date, end_date, interval)
        if not days_data:
            logger.error("No data loaded for backtest")
            return BacktestResult(
                underlying=underlying,
                start_date=start_date,
                end_date=end_date,
                initial_capital=self._initial_capital,
                final_capital=self._initial_capital,
                days=[],
                config=self._config_dict(),
            )

        self._capital = self._initial_capital
        # Reset GEX/Flow for fresh backtest
        self._gex = GEXAnalyzer(self._settings)
        self._flow = FlowAnalyzer(self._settings)
        day_results = []

        for day_data in days_data:
            result = await self._backtest_day(day_data)
            day_results.append(result)
            self._capital += result.total_pnl

        return BacktestResult(
            underlying=underlying,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self._initial_capital,
            final_capital=self._capital,
            days=day_results,
            config=self._config_dict(),
        )

    async def _backtest_day(self, day: BacktestDay) -> DayResult:
        """Simulate one trading day."""
        cb = CircuitBreaker(self._settings)
        risk = OptionsRiskManager(self._settings, cb)
        risk.set_portfolio_value(Decimal(str(self._capital)))
        risk.set_day_start_value(Decimal(str(self._capital)))

        # Filter bars to entry window
        entry_start_h, entry_start_m = map(int, self._settings.entry_start.split(":"))
        entry_cutoff_h, entry_cutoff_m = map(int, self._settings.entry_cutoff.split(":"))
        hard_close_h, hard_close_m = map(int, self._settings.hard_close.split(":"))

        candles = deque(maxlen=self._settings.candle_cache_size)
        open_trades: list[BacktestTrade] = []
        closed_trades: list[BacktestTrade] = []
        signals_generated = 0
        signals_blocked = 0
        last_entry_time: Optional[datetime] = None
        entry_cooldown_minutes = 3  # Short cooldown — exits manage risk now
        max_concurrent = 3  # Allow up to 3 simultaneous positions
        day_pnl = 0.0  # Track intraday P&L for daily stop
        # Reset momentum cache for each day
        self._momentum_cache = {}

        # GEX/Flow update tracking — refresh every 5 bars (~10min on 2m bars)
        bars_since_quant_update = 0
        gex_update_interval = 5

        # Build candle history from bars
        for bar in day.bars:
            candles.append({
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            })

            et_time = bar.timestamp.astimezone(ET)
            current_time = et_time.time()
            bars_since_quant_update += 1

            # ── Update GEX/Flow periodically ──────────────────────
            if bars_since_quant_update >= gex_update_interval:
                bars_since_quant_update = 0
                minutes_to_close = self._minutes_until(et_time, hard_close_h, hard_close_m)
                quant_chain = self._pricer.generate_chain(
                    spot=bar.close, vix=day.vix_close,
                    minutes_to_close=minutes_to_close,
                    underlying=day.underlying,
                    expiration=day.date.isoformat(),
                )
                await self._gex.update(day.underlying, quant_chain, bar.close)
                await self._flow.update(quant_chain)

            # ── Check exits for open positions ────────────────────
            for trade in list(open_trades):
                # Estimate current option price using Greeks (more realistic
                # than full BS reprice, which overestimates 0DTE theta decay)
                current_price = self._greeks_reprice(trade, bar.close, et_time)
                trade.peak_price = max(trade.peak_price, current_price)

                # Check exit conditions
                should_exit, reason = self._check_exit(
                    trade, current_price, bar.close, et_time, hard_close_h, hard_close_m,
                )
                if should_exit:
                    exit_price = self._slippage.apply_exit(current_price, trade.option_type)
                    trade.exit_price = exit_price
                    trade.exit_time = bar.timestamp
                    trade.exit_reason = reason
                    trade.pnl = (exit_price - trade.entry_price) * trade.contracts * 100
                    logger.debug(
                        "EXIT %s: reason=%s spot=%.2f→%.2f option=%.2f→%.2f pnl=$%.2f hold=%.0fm",
                        trade.symbol, reason, trade.entry_spot, bar.close,
                        trade.entry_price, exit_price, trade.pnl, trade.hold_minutes,
                    )
                    open_trades.remove(trade)
                    closed_trades.append(trade)
                    day_pnl += trade.pnl
                    risk.record_close(trade.symbol, Decimal(str(trade.pnl)))

            # ── Check entry signals ───────────────────────────────
            # Cooldown check: wait N minutes between entries to avoid overtrading
            cooldown_ok = (
                last_entry_time is None or
                (bar.timestamp - last_entry_time).total_seconds() / 60 >= entry_cooldown_minutes
            )

            if (current_time >= time(entry_start_h, entry_start_m) and
                current_time <= time(entry_cutoff_h, entry_cutoff_m) and
                len(candles) >= 35 and
                len(open_trades) < max_concurrent and  # Up to 3 concurrent
                cooldown_ok and
                day_pnl > -(self._capital * 0.005) and  # Daily stop: 0.5% of capital
                not cb.is_halted):

                closes = [Decimal(str(c["close"])) for c in candles]
                candle_list = list(candles)

                # Skip entries on crisis/high-VIX days (>28)
                if day.vix_close > 28:
                    continue

                tech_signals = compute_all_signals(
                    closes, candle_list,
                    rsi_period=self._settings.rsi_period,
                    rsi_overbought=self._settings.rsi_overbought,
                    rsi_oversold=self._settings.rsi_oversold,
                    macd_fast=self._settings.macd_fast,
                    macd_slow=self._settings.macd_slow,
                    macd_signal=self._settings.macd_signal,
                    bb_period=self._settings.bb_period,
                    bb_std=self._settings.bb_std,
                )

                # Simplified signal scoring (technical only for backtest)
                signal = self._evaluate_signal(
                    underlying=day.underlying,
                    price=bar.close,
                    tech_signals=tech_signals,
                    vix=day.vix_close,
                    bar=bar,
                    et_time=et_time,
                )

                if signal.should_trade:
                    signals_generated += 1

                    # Find matching option
                    minutes_to_close = self._minutes_until(et_time, hard_close_h, hard_close_m)
                    chain = self._pricer.generate_chain(
                        spot=bar.close, vix=day.vix_close,
                        minutes_to_close=minutes_to_close,
                        underlying=day.underlying,
                        expiration=day.date.isoformat(),
                    )

                    option_type = "call" if signal.direction == TradeDirection.BUY_CALL else "put"
                    target_delta = self._settings.target_delta
                    best = self._select_strike(chain, option_type, target_delta)

                    if best and best.mid >= self._settings.min_premium and best.mid <= self._settings.max_premium:
                        # Build OptionsContract for risk check
                        contract = OptionsContract(
                            symbol=best.symbol, underlying=day.underlying,
                            option_type=option_type, strike=Decimal(str(best.strike)),
                            expiration=best.expiration,
                            bid=Decimal(str(best.bid)), ask=Decimal(str(best.ask)),
                            last=Decimal(str(best.mid)),
                            delta=best.delta, gamma=best.gamma,
                            theta=best.theta, vega=best.vega, iv=best.iv,
                        )
                        signal_with_contract = TradeSignal(
                            direction=signal.direction,
                            confidence=signal.confidence,
                            underlying=day.underlying,
                            contract=contract,
                            target_price=Decimal(str(best.mid)),
                            reason=signal.reason,
                            score_breakdown=signal.score_breakdown,
                        )

                        can_trade, block_reason = risk.can_trade(signal_with_contract)
                        if can_trade:
                            vix_mult = self._vix_size_mult(day.vix_close)
                            contracts = risk.compute_position_size(signal_with_contract, vix_mult)
                            if contracts > 0:
                                entry_price = self._slippage.apply_entry(best.mid, option_type)
                                trade = BacktestTrade(
                                    underlying=day.underlying,
                                    symbol=best.symbol,
                                    option_type=option_type,
                                    strike=best.strike,
                                    direction=signal.direction.value,
                                    contracts=contracts,
                                    entry_price=entry_price,
                                    entry_time=bar.timestamp,
                                    entry_confidence=signal.confidence,
                                    entry_spot=bar.close,
                                    entry_delta=best.delta,
                                    entry_gamma=best.gamma,
                                    entry_theta=best.theta,
                                    peak_price=entry_price,
                                    peak_spot=bar.close,
                                    score_breakdown=signal.score_breakdown,
                                )
                                open_trades.append(trade)
                                last_entry_time = bar.timestamp
                                risk.record_open(
                                    best.symbol, day.underlying, contracts,
                                    Decimal(str(entry_price)), contract,
                                )
                        else:
                            signals_blocked += 1
                    else:
                        signals_blocked += 1

        # Force-close any remaining positions at end of day
        if day.bars:
            last_bar = day.bars[-1]
            last_et = last_bar.timestamp.astimezone(ET)
            for trade in open_trades:
                current_price = self._greeks_reprice(trade, last_bar.close, last_et)
                trade.exit_price = self._slippage.apply_exit(current_price, trade.option_type)
                trade.exit_time = last_bar.timestamp
                trade.exit_reason = "End of day forced close"
                trade.pnl = (trade.exit_price - trade.entry_price) * trade.contracts * 100
                closed_trades.append(trade)
        else:
            for trade in open_trades:
                trade.exit_price = 0.01
                trade.exit_reason = "End of day forced close (no bars)"
                trade.pnl = (0.01 - trade.entry_price) * trade.contracts * 100
                closed_trades.append(trade)

        all_trades = closed_trades
        total_pnl = sum(t.pnl for t in all_trades)
        wins = sum(1 for t in all_trades if t.pnl > 0)
        losses = sum(1 for t in all_trades if t.pnl < 0)

        return DayResult(
            date=day.date,
            underlying=day.underlying,
            trades=all_trades,
            total_pnl=total_pnl,
            wins=wins,
            losses=losses,
            signals_generated=signals_generated,
            signals_blocked=signals_blocked,
            vix=day.vix_close,
        )

    def _evaluate_signal(
        self,
        underlying: str,
        price: float,
        tech_signals: SignalBundle,
        vix: float,
        bar: HistoricalBar,
        et_time: datetime,
    ) -> TradeSignal:
        """Signal evaluation for backtesting.

        Uses multiple factors matching the live ensemble weights:
        - Technical (RSI, MACD, BB, Volume Delta) — 22%
        - Price momentum — 18%
        - GEX regime + levels — 13%
        - Options flow — 14%
        - VIX regime — 8%
        Requires directional agreement across primary factors.
        """
        tech_score = self._score_technicals(tech_signals)

        # VIX regime scoring
        vix_score = 0.0
        if vix < 12:
            vix_score = 0.3
        elif vix < 15:
            vix_score = 0.1
        elif vix > 35:
            vix_score = -0.8
        elif vix > 30:
            vix_score = -0.3

        # Short-term momentum: 5-bar price trend
        momentum_score = self._score_momentum(bar, et_time, underlying)

        # GEX scores (directional)
        gex_call_score = self._gex.get_score(underlying, "call")
        gex_put_score = self._gex.get_score(underlying, "put")

        # Flow scores (directional)
        flow_call_score = self._flow.get_score("call")
        flow_put_score = self._flow.get_score("put")

        # Combined score using ensemble weights (normalized to sum ~1.0)
        # tech=0.29, momentum=0.24, gex=0.17, flow=0.19, vix=0.11
        call_score = (
            tech_score * 0.29
            + momentum_score * 0.24
            + gex_call_score * 0.17
            + flow_call_score * 0.19
            + vix_score * 0.11
        )
        put_score = (
            -tech_score * 0.29
            + (-momentum_score) * 0.24
            + gex_put_score * 0.17
            + flow_put_score * 0.19
            + (-vix_score) * 0.11
        )

        # Scale to 0-100
        call_conf = int(max(0, min(100, (call_score + 1) / 2 * 100)))
        put_conf = int(max(0, min(100, (put_score + 1) / 2 * 100)))

        # Per-underlying thresholds: SPY is most efficiently priced,
        # needs slightly higher bar. Exits manage risk, so don't over-filter.
        underlying_thresholds = {"SPY": 60, "QQQ": 58, "IWM": 57}
        threshold = max(
            self._settings.signal_confidence_threshold,
            underlying_thresholds.get(underlying, 58),
        )

        breakdown = {
            "technical": tech_score,
            "momentum": momentum_score,
            "gex": gex_call_score,
            "flow": flow_call_score,
            "vix": vix_score,
        }

        # Directional agreement: require at least one primary factor strongly
        # directional (>0.2) OR both mildly directional (>0.05).
        # Old gate (both > 0.1) blocked 60%+ of valid signals.
        bullish_agreement = (
            (tech_score > 0.2 or momentum_score > 0.2) and
            tech_score > -0.05 and momentum_score > -0.05
        )
        bearish_agreement = (
            (tech_score < -0.2 or momentum_score < -0.2) and
            tech_score < 0.05 and momentum_score < 0.05
        )

        if call_conf >= threshold and bullish_agreement:
            return TradeSignal(
                direction=TradeDirection.BUY_CALL,
                confidence=call_conf,
                underlying=underlying,
                contract=None,
                target_price=Decimal(str(price)),
                reason=(
                    f"Backtest: call conf={call_conf} tech={tech_score:.2f} "
                    f"mom={momentum_score:.2f} gex={gex_call_score:.2f} flow={flow_call_score:.2f}"
                ),
                score_breakdown=breakdown,
            )
        elif put_conf >= threshold and bearish_agreement:
            breakdown["gex"] = gex_put_score
            breakdown["flow"] = flow_put_score
            return TradeSignal(
                direction=TradeDirection.BUY_PUT,
                confidence=put_conf,
                underlying=underlying,
                contract=None,
                target_price=Decimal(str(price)),
                reason=(
                    f"Backtest: put conf={put_conf} tech={-tech_score:.2f} "
                    f"mom={-momentum_score:.2f} gex={gex_put_score:.2f} flow={flow_put_score:.2f}"
                ),
                score_breakdown=breakdown,
            )
        else:
            return TradeSignal(
                direction=TradeDirection.HOLD,
                confidence=0,
                underlying=underlying,
                contract=None,
                target_price=Decimal("0"),
                reason="No directional agreement or below threshold",
            )

    def _score_technicals(self, signals: SignalBundle) -> float:
        """Momentum-following technical scoring for 0DTE.

        0DTE options profit from trend continuation, not reversals.
        RSI oversold doesn't mean "buy" — it means the move is strong.
        Enter WITH momentum, not against it.
        """
        score = 0.0

        if signals.rsi:
            # Momentum interpretation: strong RSI = strong trend
            # RSI > 60 = bullish momentum, RSI < 40 = bearish momentum
            # RSI 40-60 = no clear momentum (weak signal)
            if signals.rsi.value > 65:
                score += 0.4   # Strong bullish momentum
            elif signals.rsi.value > 55:
                score += 0.15  # Moderate bullish
            elif signals.rsi.value < 35:
                score -= 0.4   # Strong bearish momentum
            elif signals.rsi.value < 45:
                score -= 0.15  # Moderate bearish
            # RSI 45-55 = neutral, no signal

        if signals.macd:
            # MACD crossover + histogram direction = trend confirmation
            if signals.macd.is_bullish:
                score += 0.25
            else:
                score -= 0.25

            # Histogram growing = strengthening momentum
            if signals.macd.histogram > 0.3:
                score += 0.15
            elif signals.macd.histogram < -0.3:
                score -= 0.15

        if signals.bollinger:
            # For momentum: above upper band = strong bullish, below lower = strong bearish
            # Squeeze = low vol, don't trade (no momentum)
            if signals.bollinger.is_squeeze:
                score *= 0.3  # Heavily reduce signal during squeeze
            elif signals.bollinger.pct_b > 0.8:
                score += 0.15  # Momentum pushing upper band
            elif signals.bollinger.pct_b < 0.2:
                score -= 0.15  # Momentum pushing lower band

        if signals.volume_delta:
            if signals.volume_delta.ratio > 0.6:
                score += 0.1
            elif signals.volume_delta.ratio < 0.4:
                score -= 0.1

        return max(-1.0, min(1.0, score))

    def _score_momentum(self, bar: HistoricalBar, et_time: datetime, underlying: str = "SPY") -> float:
        """Score -1 to +1 from short-term price momentum.

        Normalizes by rolling realized volatility so that a 0.1% SPY move
        and a 0.2% QQQ move produce the same score if they represent the
        same number of standard deviations. This prevents high-beta assets
        from systematically scoring higher.
        """
        cache = getattr(self, '_momentum_cache', {})

        # Use the candle history we already have
        history = cache.get(underlying, [])
        history.append(bar.close)
        if len(history) > 20:
            history = history[-20:]
        cache[underlying] = history
        self._momentum_cache = cache

        if len(history) < 5:
            return 0.0

        # Short-term momentum: compare last 3 bars vs previous 3
        recent = np.mean(history[-3:])
        earlier = np.mean(history[-6:-3]) if len(history) >= 6 else np.mean(history[:3])

        if earlier == 0:
            return 0.0

        pct_change = (recent - earlier) / earlier

        # Normalize by rolling realized volatility (bar-to-bar returns std)
        # This ensures high-beta and low-beta assets score equally for
        # the same number of standard deviations of movement.
        if len(history) >= 8:
            returns = [(history[i] - history[i-1]) / history[i-1]
                       for i in range(1, len(history)) if history[i-1] > 0]
            if returns:
                rv = float(np.std(returns))
                if rv > 1e-6:
                    z_move = pct_change / rv
                    # ~1 sigma → 0.3, ~2 sigma → 0.6, ~3 sigma → capped near 1.0
                    score = float(np.tanh(z_move * 0.35))
                    return max(-1.0, min(1.0, score))

        # Fallback: fixed scaling if not enough history for vol estimate
        score = pct_change * 300
        return max(-1.0, min(1.0, score))

    def _check_exit(
        self,
        trade: BacktestTrade,
        current_price: float,
        current_spot: float,
        et_time: datetime,
        hard_close_h: int,
        hard_close_m: int,
    ) -> tuple[bool, str]:
        """Check exit conditions for 0DTE scalps.

        v6 exit logic — fixes asymmetric reward/risk:
        - Tighter stop (-18%) to cut losers fast
        - Higher profit target (+20%) to let winners run
        - Trailing stop activates at +8% gain, trails at 40% giveback
        - Scaled time exits: take any profit >3% after 8 min, force at 12 min
        - Early momentum stop: if down >10% in first 4 min, trend is wrong
        """
        entry = trade.entry_price
        if entry <= 0:
            return False, ""

        actual_pnl_pct = (current_price - entry) / entry

        # Directional P&L for secondary checks
        underlying_move = current_spot - trade.entry_spot
        direction_sign = 1.0 if trade.option_type == "call" else -1.0
        favorable_move = underlying_move * direction_sign
        delta_abs = abs(trade.entry_delta) if trade.entry_delta != 0 else 0.30
        directional_pnl = favorable_move * delta_abs
        directional_pnl_pct = directional_pnl / entry if entry > 0 else 0

        hold_minutes = (et_time - trade.entry_time.astimezone(ET)).total_seconds() / 60

        # Track peaks
        trade.peak_price = max(trade.peak_price, current_price)
        if trade.option_type == "call":
            trade.peak_spot = max(trade.peak_spot, current_spot)
        else:
            trade.peak_spot = min(trade.peak_spot, current_spot)

        peak_pnl_pct = (trade.peak_price - entry) / entry

        # ── Hard close ──
        current_t = et_time.time()
        if current_t >= time(hard_close_h, hard_close_m):
            return True, f"Hard close ({self._settings.hard_close} ET)"

        # ── Stop loss: 18% hard stop ──
        if actual_pnl_pct <= -0.18:
            return True, f"Stop loss ({actual_pnl_pct:.1%})"

        # ── Early momentum stop: down >10% in first 4 min = wrong direction ──
        if hold_minutes <= 4 and actual_pnl_pct <= -0.10:
            return True, f"Early stop ({actual_pnl_pct:.1%} @ {hold_minutes:.0f}m)"

        # ── Profit target: 20% premium gain (was 12% — let winners run) ──
        if actual_pnl_pct >= 0.20:
            return True, f"Profit target ({actual_pnl_pct:.1%})"

        # ── Trailing stop: activates at +8% gain, 40% giveback ──
        if peak_pnl_pct >= 0.08:
            giveback = (peak_pnl_pct - actual_pnl_pct) / peak_pnl_pct if peak_pnl_pct > 0 else 0
            if giveback >= 0.40:
                return True, f"Trail stop (peak {peak_pnl_pct:.1%}, now {actual_pnl_pct:.1%})"

        # ── Directional profit: underlying moved well but theta ate some ──
        if directional_pnl_pct >= 0.20 and actual_pnl_pct > -0.03:
            return True, f"Directional profit (dir {directional_pnl_pct:.1%}, opt {actual_pnl_pct:.1%})"

        # ── Scratch exit: near breakeven after 6 min = no edge, close before theta eats it ──
        if hold_minutes >= 6 and -0.03 <= actual_pnl_pct <= 0.03:
            return True, f"Scratch ({actual_pnl_pct:.1%} @ {hold_minutes:.0f}m)"

        # ── Adverse time stop: losing after 8 min = trend didn't develop, cut it ──
        if hold_minutes >= 8 and actual_pnl_pct < -0.05:
            return True, f"Time stop ({actual_pnl_pct:.1%} @ {hold_minutes:.0f}m)"

        # ── Trailing on underlying ──
        if trade.option_type == "call":
            peak_move = trade.peak_spot - trade.entry_spot
            current_move = current_spot - trade.entry_spot
        else:
            peak_move = trade.entry_spot - trade.peak_spot
            current_move = trade.entry_spot - current_spot

        if peak_move > trade.entry_spot * 0.001 and peak_move > 0:
            retracement = 1.0 - (current_move / peak_move) if current_move < peak_move else 0
            if retracement >= 0.50 and actual_pnl_pct > -0.05:
                return True, f"Trail (retrace {retracement:.0%}, peak ${peak_move:.2f})"

        # ── Time take profit: any gain >3% after 8 min ──
        if hold_minutes >= 8 and actual_pnl_pct > 0.03:
            return True, f"Time take profit ({actual_pnl_pct:.1%} @ {hold_minutes:.0f}m)"

        # ── Hard time exit: 15 min max (was 12 — give profitable trades more room) ──
        if hold_minutes >= 15:
            return True, f"Time exit ({actual_pnl_pct:.1%} @ {hold_minutes:.0f}m)"

        return False, ""

    def _greeks_reprice(
        self, trade: BacktestTrade, current_spot: float, et_time: datetime,
    ) -> float:
        """Estimate current option price using entry Greeks with adjustments.

        Uses Taylor expansion with two 0DTE-specific improvements:
        1. Accelerating theta: real 0DTE theta scales as 1/√T, so decay
           accelerates into the close. We scale entry theta by √(T_entry/T_now).
        2. Delta adjustment: as spot moves, delta shifts by gamma×ΔS.
           Use adjusted delta for more accurate P&L on larger moves.
        """
        ds = current_spot - trade.entry_spot
        delta = trade.entry_delta
        gamma = trade.entry_gamma if trade.entry_gamma != 0 else 0.01
        theta_per_day = trade.entry_theta if trade.entry_theta != 0 else -0.15

        # Hold time
        hold_seconds = (et_time - trade.entry_time.astimezone(ET)).total_seconds()
        hold_minutes = hold_seconds / 60
        dt_days = hold_minutes / 390

        # ── Accelerating theta for 0DTE ──
        # Theta scales as 1/√T. If entry was at T_entry minutes to close
        # and now T_now minutes to close, actual theta ≈ entry_theta × √(T_entry/T_now)
        entry_et = trade.entry_time.astimezone(ET)
        close_today = entry_et.replace(hour=16, minute=0, second=0)
        t_entry_min = max(1, (close_today - entry_et).total_seconds() / 60)
        t_now_min = max(1, (close_today - et_time).total_seconds() / 60)

        # Average theta acceleration over the hold period
        # integral of 1/√t from t_now to t_entry, divided by (t_entry - t_now)
        if t_now_min < t_entry_min:
            theta_accel = (t_entry_min / t_now_min) ** 0.5
            theta_accel = min(theta_accel, 3.0)  # Cap at 3x to avoid explosion
        else:
            theta_accel = 1.0

        theta_pnl = theta_per_day * dt_days * theta_accel

        # ── Adjusted delta: δ_adj = δ + γ·ΔS ──
        adjusted_delta = delta + gamma * ds
        delta_pnl = adjusted_delta * ds

        # Gamma P&L (second-order, using entry gamma)
        gamma_pnl = 0.5 * gamma * ds * ds

        # Total: use max of (delta_pnl, delta_pnl + gamma_pnl) to avoid
        # double-counting since adjusted_delta already captures some convexity
        # Actually: delta_adj*ds = delta*ds + gamma*ds², and gamma_pnl = 0.5*gamma*ds²
        # So combined = delta*ds + 1.5*gamma*ds² which overcounts.
        # Correct approach: just use adjusted_delta * ds (which = delta*ds + gamma*ds²)
        # plus theta. The 0.5*gamma*ds² is already embedded.
        price_change = delta_pnl + theta_pnl

        estimated_price = trade.entry_price + price_change
        return max(0.01, estimated_price)

    def _find_option(
        self, chain: list[SimulatedOption], strike: float, option_type: str,
    ) -> Optional[SimulatedOption]:
        for opt in chain:
            if abs(opt.strike - strike) < 0.01 and opt.option_type == option_type:
                return opt
        return None

    def _select_strike(
        self, chain: list[SimulatedOption], option_type: str, target_delta: float,
    ) -> Optional[SimulatedOption]:
        """Select the strike closest to target delta."""
        candidates = [o for o in chain if o.option_type == option_type]
        if not candidates:
            return None

        # Sort by distance from target delta
        target = target_delta if option_type == "call" else -target_delta
        candidates.sort(key=lambda o: abs(abs(o.delta) - target_delta))
        return candidates[0] if candidates else None

    def _vix_size_mult(self, vix: float) -> float:
        if vix > 35:
            return 0.0
        if vix > 30:
            return 0.5
        if vix < 12:
            return 1.3
        return 1.0

    def _minutes_until(self, et_time: datetime, h: int, m: int) -> float:
        target = et_time.replace(hour=h, minute=m, second=0, microsecond=0)
        delta = (target - et_time).total_seconds() / 60
        return max(1, delta)

    def _config_dict(self) -> dict:
        s = self._settings
        return {
            "signal_confidence_threshold": s.signal_confidence_threshold,
            "target_delta": s.target_delta,
            "min_premium": s.min_premium,
            "max_premium": s.max_premium,
            "pt_profit_target_pct": s.pt_profit_target_pct,
            "sl_stop_loss_pct": s.sl_stop_loss_pct,
            "sl_trailing_pct": s.sl_trailing_pct,
            "kelly_fraction": str(s.kelly_fraction),
            "max_position_pct": str(s.max_position_pct),
            "daily_drawdown_halt": str(s.daily_drawdown_halt),
        }
