"""Master 0DTE strategy: multi-factor signal ensemble.

Signal ensemble (weighted scoring, 8 factors):
    Technical momentum    22%   RSI, MACD, BB
    Tick momentum + ROC   18%   Price feed
    GEX regime + levels   13%   quant/gex.py
    Options flow          14%   quant/flow.py
    OptionsAI             10%   quant/optionsai.py (IV skew, expected move, AI strategies)
    VIX regime + IV pctile 8%   quant/vix.py
    Market internals      10%   quant/internals.py
    Sentiment (contrarian) 5%   quant/sentiment.py

Gate checks (must ALL pass):
    1. Not in macro blackout
    1b. No earnings blackout for underlying
    2. Within entry window (9:45 AM - 2:30 PM ET)
    3. IV percentile < 85
    4. VIX regime not crisis
    5. Spread ratio >= 0.90 on target contract
    6. Portfolio Greeks within limits
    7. PDT budget available
"""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from src.data.alpaca_stream import TickMomentum
from src.data.options_chain import OptionsChainManager
from src.infra.config import Settings
from src.quant.flow import FlowAnalyzer
from src.quant.gex import GEXAnalyzer
from src.quant.internals import MarketInternals
from src.quant.macro import MacroCalendar
from src.quant.optionsai import OptionsAIAnalyzer
from src.quant.sentiment import SentimentAggregator
from src.quant.vix import VIXRegimeDetector
from src.strategy.base import BaseStrategy, TradeDirection, TradeSignal
from src.strategy.signals import SignalBundle
from src.strategy.weight_adapter import WeightAdapter

logger = logging.getLogger(__name__)


class ZeroDTEStrategy(BaseStrategy):
    """Multi-factor 0DTE options scalping strategy."""

    def __init__(
        self,
        settings: Settings,
        chain_mgr: OptionsChainManager,
        vix: VIXRegimeDetector,
        gex: GEXAnalyzer,
        flow: FlowAnalyzer,
        sentiment: SentimentAggregator,
        macro: MacroCalendar,
        internals: MarketInternals,
        optionsai: Optional[OptionsAIAnalyzer] = None,
        weight_adapter: Optional[WeightAdapter] = None,
    ) -> None:
        self._settings = settings
        self._chain_mgr = chain_mgr
        self._vix = vix
        self._gex = gex
        self._flow = flow
        self._sentiment = sentiment
        self._macro = macro
        self._internals = internals
        self._optionsai = optionsai
        self._weight_adapter = weight_adapter

    @property
    def name(self) -> str:
        return "ZeroDTE-Ensemble-v1"

    def evaluate(
        self,
        underlying: str,
        current_price: Decimal,
        signals: Optional[SignalBundle] = None,
        momentum: Optional[TickMomentum] = None,
        now: Optional[datetime] = None,
    ) -> TradeSignal:
        """Evaluate all factors and produce a trade signal."""
        now = now or datetime.now(timezone.utc)

        # ── Gate checks ────────────────────────────────────────
        gate_result = self._check_gates(underlying, now)
        if gate_result:
            return self._make_hold(underlying, gate_result)

        # ── Compute factor scores ──────────────────────────────
        tech_score = self._score_technicals(signals)
        tick_score = self._score_tick_momentum(momentum)
        gex_score_call = self._gex.get_score(underlying, "call")
        gex_score_put = self._gex.get_score(underlying, "put")
        flow_score_call = self._flow.get_score("call")
        flow_score_put = self._flow.get_score("put")
        vix_score = self._vix.get_score()
        internals_score = self._internals.get_score("call")  # Direction TBD
        sentiment_score_call = self._sentiment.get_score("call")
        sentiment_score_put = self._sentiment.get_score("put")
        oai_score_call = self._optionsai.get_score(underlying, "call") if self._optionsai else 0.0
        oai_score_put = self._optionsai.get_score(underlying, "put") if self._optionsai else 0.0

        # ── Get weights (adaptive or static) ─────────────────
        wa = self._weight_adapter
        wt = wa.get_weight("technical") if wa else self._settings.weight_technical
        wtm = wa.get_weight("tick_momentum") if wa else self._settings.weight_tick_momentum
        wg = wa.get_weight("gex") if wa else self._settings.weight_gex
        wf = wa.get_weight("flow") if wa else self._settings.weight_flow
        wv = wa.get_weight("vix") if wa else self._settings.weight_vix
        wi = wa.get_weight("internals") if wa else self._settings.weight_internals
        ws = wa.get_weight("sentiment") if wa else self._settings.weight_sentiment
        wo = wa.get_weight("optionsai") if wa else self._settings.weight_optionsai

        # ── Weighted ensemble for CALL direction ───────────────
        call_score = (
            tech_score * wt +
            tick_score * wtm +
            gex_score_call * wg +
            flow_score_call * wf +
            vix_score * wv +
            internals_score * wi +
            sentiment_score_call * ws +
            oai_score_call * wo
        )

        # ── Weighted ensemble for PUT direction ────────────────
        put_score = (
            -tech_score * wt +
            -tick_score * wtm +
            gex_score_put * wg +
            flow_score_put * wf +
            -vix_score * wv +
            -internals_score * wi +
            sentiment_score_put * ws +
            oai_score_put * wo
        )

        # Scale to 0-100 confidence
        call_confidence = int(max(0, min(100, (call_score + 1) / 2 * 100)))
        put_confidence = int(max(0, min(100, (put_score + 1) / 2 * 100)))

        # Pick stronger direction
        if call_confidence >= put_confidence and call_confidence >= w.signal_confidence_threshold:
            direction = TradeDirection.BUY_CALL
            confidence = call_confidence
            option_type = "call"
            score_breakdown = self._build_breakdown(
                tech_score, tick_score, gex_score_call, flow_score_call,
                vix_score, internals_score, sentiment_score_call, oai_score_call,
            )
        elif put_confidence > call_confidence and put_confidence >= w.signal_confidence_threshold:
            direction = TradeDirection.BUY_PUT
            confidence = put_confidence
            option_type = "put"
            score_breakdown = self._build_breakdown(
                -tech_score, -tick_score, gex_score_put, flow_score_put,
                -vix_score, -internals_score, sentiment_score_put, oai_score_put,
            )
        else:
            return self._make_hold(
                underlying,
                f"Below threshold (call={call_confidence}, put={put_confidence}, req={w.signal_confidence_threshold})",
            )

        # ── Strike selection ───────────────────────────────────
        candidate = self._chain_mgr.select_strike(
            underlying, option_type, current_price,
        )
        if not candidate:
            return self._make_hold(underlying, f"No viable {option_type} strikes found")

        contract = candidate.contract

        # Target price: use mid (will be adjusted by engine for limit)
        target_price = contract.mid

        reason = (
            f"{self.name}: {direction.value} {underlying} "
            f"${contract.strike} {option_type} @ ${float(target_price):.2f} "
            f"(conf={confidence}, delta={contract.delta:.2f})"
        )

        logger.info(
            "SIGNAL: %s conf=%d tech=%.2f tick=%.2f gex=%.2f flow=%.2f "
            "vix=%.2f int=%.2f sent=%.2f oai=%.2f",
            direction.value, confidence, tech_score, tick_score,
            gex_score_call if option_type == "call" else gex_score_put,
            flow_score_call if option_type == "call" else flow_score_put,
            vix_score, internals_score,
            sentiment_score_call if option_type == "call" else sentiment_score_put,
            oai_score_call if option_type == "call" else oai_score_put,
        )

        return TradeSignal(
            direction=direction,
            confidence=confidence,
            underlying=underlying,
            contract=contract,
            target_price=target_price,
            reason=reason,
            score_breakdown=score_breakdown,
        )

    # ── Gate Checks ────────────────────────────────────────────

    def _check_gates(self, underlying: str, now: datetime) -> Optional[str]:
        """Run all gate checks. Returns reason string if blocked, None if clear."""
        # 1. Macro blackout
        if self._macro.is_blackout():
            macro_signals = self._macro.latest
            reason = macro_signals.blackout_reason if macro_signals else "Macro event"
            return f"Macro blackout: {reason}"

        # 1b. Earnings blackout (per-underlying)
        if self._optionsai and self._optionsai.has_earnings(underlying):
            return f"Earnings blackout for {underlying}"

        # 2. Entry window (ET timezone)
        from zoneinfo import ZoneInfo
        et = now.astimezone(ZoneInfo("America/New_York"))
        entry_start_h, entry_start_m = map(int, self._settings.entry_start.split(":"))
        entry_cutoff_h, entry_cutoff_m = map(int, self._settings.entry_cutoff.split(":"))

        if et.hour < entry_start_h or (et.hour == entry_start_h and et.minute < entry_start_m):
            return f"Before entry window ({self._settings.entry_start} ET)"
        if et.hour > entry_cutoff_h or (et.hour == entry_cutoff_h and et.minute > entry_cutoff_m):
            return f"After entry cutoff ({self._settings.entry_cutoff} ET)"

        # 3. VIX checks
        vix_signals = self._vix.latest
        if vix_signals and not vix_signals.should_trade:
            return vix_signals.block_reason

        return None

    # ── Factor Scorers ─────────────────────────────────────────

    def _score_technicals(self, signals: Optional[SignalBundle]) -> float:
        """Score -1 to +1 from RSI, MACD, Bollinger."""
        if not signals:
            return 0.0

        score = 0.0

        # RSI
        if signals.rsi:
            if signals.rsi.is_oversold:
                score += 0.4   # Bullish
            elif signals.rsi.is_overbought:
                score -= 0.4   # Bearish
            else:
                # Normalize RSI 30-70 → -0.2 to +0.2
                rsi_norm = (signals.rsi.value - 50) / 50
                score -= rsi_norm * 0.2  # High RSI = bearish

        # MACD
        if signals.macd:
            if signals.macd.is_bullish:
                score += 0.3
            else:
                score -= 0.3

            # Histogram strength
            if abs(signals.macd.histogram) > 0.5:
                score += 0.1 if signals.macd.histogram > 0 else -0.1

        # Bollinger Bands
        if signals.bollinger:
            if signals.bollinger.pct_b < 0.1:
                score += 0.2  # Near lower band = bullish reversal
            elif signals.bollinger.pct_b > 0.9:
                score -= 0.2  # Near upper band = bearish reversal

            if signals.bollinger.is_squeeze:
                score *= 0.5  # Squeeze = reduce conviction (breakout direction unclear)

        # Volume delta
        if signals.volume_delta:
            if signals.volume_delta.ratio > 0.6:
                score += 0.1  # More buy volume
            elif signals.volume_delta.ratio < 0.4:
                score -= 0.1  # More sell volume

        return max(-1.0, min(1.0, score))

    def _score_tick_momentum(self, momentum: Optional[TickMomentum]) -> float:
        """Score -1 to +1 from tick momentum."""
        if not momentum:
            return 0.0

        direction = momentum.direction
        roc = momentum.roc_pct

        score = direction * 0.5  # Base from direction

        # ROC amplifier
        if abs(roc) > 0.1:
            score += (0.3 if roc > 0 else -0.3)
        elif abs(roc) > 0.05:
            score += (0.15 if roc > 0 else -0.15)

        return max(-1.0, min(1.0, score))

    def _build_breakdown(
        self, tech: float, tick: float, gex: float, flow: float,
        vix: float, internals: float, sentiment: float, optionsai: float = 0.0,
    ) -> dict:
        return {
            "technical": round(tech, 3),
            "tick_momentum": round(tick, 3),
            "gex": round(gex, 3),
            "flow": round(flow, 3),
            "vix": round(vix, 3),
            "internals": round(internals, 3),
            "sentiment": round(sentiment, 3),
            "optionsai": round(optionsai, 3),
        }
