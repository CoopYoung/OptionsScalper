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

    def __init__(self, slippage_pct: float = 0.02, latency_bars: int = 1) -> None:
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

    def run(
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
        day_results = []

        for day_data in days_data:
            result = self._backtest_day(day_data)
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

    def _backtest_day(self, day: BacktestDay) -> DayResult:
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
                    risk.record_close(trade.symbol, Decimal(str(trade.pnl)))

            # ── Check entry signals ───────────────────────────────
            if (current_time >= time(entry_start_h, entry_start_m) and
                current_time <= time(entry_cutoff_h, entry_cutoff_m) and
                len(candles) >= 35 and
                len(open_trades) < self._settings.max_positions_per_underlying and
                not cb.is_halted):

                closes = [Decimal(str(c["close"])) for c in candles]
                candle_list = list(candles)

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
        """Simplified signal evaluation for backtesting (technical + VIX only)."""
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

        # Combined score
        call_score = tech_score * 0.70 + vix_score * 0.30
        put_score = -tech_score * 0.70 + (-vix_score) * 0.30

        call_conf = int(max(0, min(100, (call_score + 1) / 2 * 100)))
        put_conf = int(max(0, min(100, (put_score + 1) / 2 * 100)))

        threshold = self._settings.signal_confidence_threshold

        if call_conf >= put_conf and call_conf >= threshold:
            return TradeSignal(
                direction=TradeDirection.BUY_CALL,
                confidence=call_conf,
                underlying=underlying,
                contract=None,
                target_price=Decimal(str(price)),
                reason=f"Backtest signal: call conf={call_conf}",
                score_breakdown={"technical": tech_score, "vix": vix_score},
            )
        elif put_conf > call_conf and put_conf >= threshold:
            return TradeSignal(
                direction=TradeDirection.BUY_PUT,
                confidence=put_conf,
                underlying=underlying,
                contract=None,
                target_price=Decimal(str(price)),
                reason=f"Backtest signal: put conf={put_conf}",
                score_breakdown={"technical": -tech_score, "vix": -vix_score},
            )
        else:
            return TradeSignal(
                direction=TradeDirection.HOLD,
                confidence=0,
                underlying=underlying,
                contract=None,
                target_price=Decimal("0"),
                reason="Below threshold",
            )

    def _score_technicals(self, signals: SignalBundle) -> float:
        """Same scoring as zero_dte.py for consistency."""
        score = 0.0
        if signals.rsi:
            if signals.rsi.is_oversold:
                score += 0.4
            elif signals.rsi.is_overbought:
                score -= 0.4
            else:
                rsi_norm = (signals.rsi.value - 50) / 50
                score -= rsi_norm * 0.2

        if signals.macd:
            score += 0.3 if signals.macd.is_bullish else -0.3
            if abs(signals.macd.histogram) > 0.5:
                score += 0.1 if signals.macd.histogram > 0 else -0.1

        if signals.bollinger:
            if signals.bollinger.pct_b < 0.1:
                score += 0.2
            elif signals.bollinger.pct_b > 0.9:
                score -= 0.2
            if signals.bollinger.is_squeeze:
                score *= 0.5

        if signals.volume_delta:
            if signals.volume_delta.ratio > 0.6:
                score += 0.1
            elif signals.volume_delta.ratio < 0.4:
                score -= 0.1

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

        Uses a hybrid approach:
        - Actual option price P&L for profit target and stop loss (realistic)
        - Directional P&L (delta × underlying move) as a secondary check
        - Aggressive time management: max 20 min hold to avoid theta decay
        - 0DTE-appropriate thresholds: 20% profit target, 25% stop loss
        """
        entry = trade.entry_price
        if entry <= 0:
            return False, ""

        # Actual option price P&L
        actual_pnl_pct = (current_price - entry) / entry

        # Directional P&L (underlying move × delta)
        underlying_move = current_spot - trade.entry_spot
        direction_sign = 1.0 if trade.option_type == "call" else -1.0
        favorable_move = underlying_move * direction_sign
        delta_abs = abs(trade.entry_delta) if trade.entry_delta != 0 else 0.30
        directional_pnl = favorable_move * delta_abs
        directional_pnl_pct = directional_pnl / entry if entry > 0 else 0

        hold_minutes = (et_time - trade.entry_time.astimezone(ET)).total_seconds() / 60

        # ── Profit target (actual option price) ──
        # 0DTE scalps target 20% option premium gain (quick in-and-out)
        if actual_pnl_pct >= 0.20:
            return True, f"Profit target ({actual_pnl_pct:.1%})"

        # ── Directional profit exit ──
        # If underlying has moved strongly in our favor (>35%), take it
        # even if theta has eaten some premium
        if directional_pnl_pct >= 0.35 and actual_pnl_pct > -0.10:
            return True, f"Directional profit (dir {directional_pnl_pct:.1%}, opt {actual_pnl_pct:.1%})"

        # ── Stop loss (directional) ──
        # Stop at 25% adverse underlying move (not raw option price, which
        # includes natural theta decay)
        if directional_pnl_pct <= -0.25:
            return True, f"Stop loss (dir {directional_pnl_pct:.1%})"

        # ── Catastrophic stop (actual option price) ──
        # If option has lost >50% of value for any reason, cut it
        if actual_pnl_pct <= -0.50:
            return True, f"Catastrophic stop ({actual_pnl_pct:.1%})"

        # ── Trailing stop on actual option price ──
        # Only activate when we've had meaningful gains
        trade.peak_price = max(trade.peak_price, current_price)
        if trade.peak_price > entry * 1.10:  # Peak was >10% above entry
            retrace_from_peak = (trade.peak_price - current_price) / trade.peak_price
            if retrace_from_peak >= 0.30:  # Gave back 30% from peak
                return True, f"Trailing stop ({retrace_from_peak:.0%} from peak)"

        # ── Max hold time ──
        # 0DTE scalps: exit after 20 min to avoid theta decay eating edge
        if hold_minutes >= 20:
            if actual_pnl_pct > 0:
                return True, f"Time exit +profit ({actual_pnl_pct:.1%} @ {hold_minutes:.0f}m)"
            elif hold_minutes >= 30:
                # Give losing trades a bit more time but not too long
                return True, f"Time exit ({actual_pnl_pct:.1%} @ {hold_minutes:.0f}m)"

        # ── Hard close ──
        current_t = et_time.time()
        if current_t >= time(hard_close_h, hard_close_m):
            return True, f"Hard close ({self._settings.hard_close} ET)"

        return False, ""

    def _greeks_reprice(
        self, trade: BacktestTrade, current_spot: float, et_time: datetime,
    ) -> float:
        """Estimate current option price using entry Greeks.

        Instead of full Black-Scholes repricing (which overestimates 0DTE
        theta decay), use the Taylor expansion:
            ΔP ≈ δ·ΔS + ½γ·(ΔS)² + θ·Δt

        This matches how traders estimate intraday P&L and produces more
        realistic 0DTE simulation results.
        """
        ds = current_spot - trade.entry_spot  # Underlying move
        delta = trade.entry_delta
        # Use reasonable defaults if Greeks weren't captured
        gamma = getattr(trade, 'entry_gamma', 0.01)
        theta_per_day = getattr(trade, 'entry_theta', -0.15)

        # Hold time in trading days (for theta)
        hold_minutes = (et_time - trade.entry_time.astimezone(ET)).total_seconds() / 60
        dt_days = hold_minutes / 390  # Trading minutes per day

        # Delta P&L (directional)
        delta_pnl = delta * ds

        # Gamma P&L (convexity — amplifies moves, especially for 0DTE)
        gamma_pnl = 0.5 * gamma * ds * ds

        # Theta P&L (time decay — capped to prevent unrealistic decay)
        theta_pnl = theta_per_day * dt_days

        # Total estimated price change
        price_change = delta_pnl + gamma_pnl + theta_pnl

        # For puts, delta is negative, so delta_pnl is negative when spot rises
        # (which is correct — put loses value when underlying rises)
        estimated_price = trade.entry_price + price_change

        # Floor at $0.01 (options can't go negative)
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
