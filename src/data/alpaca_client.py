"""Alpaca REST client for options trading.

Handles authentication, options chain fetching, order placement,
position tracking, and account queries via the alpaca-py SDK.
"""

import logging
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Optional

from src.infra.config import Settings
from src.strategy.base import OptionsContract

logger = logging.getLogger(__name__)


class AlpacaClient:
    """REST client wrapping alpaca-py for options trading."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._trading_client = None
        self._data_client = None
        self._connected = False

    async def connect(self) -> bool:
        """Initialize Alpaca SDK clients."""
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical.option import OptionHistoricalDataClient

            self._trading_client = TradingClient(
                api_key=self._settings.alpaca_api_key,
                secret_key=self._settings.alpaca_secret_key,
                paper=self._settings.alpaca_paper,
            )
            self._data_client = OptionHistoricalDataClient(
                api_key=self._settings.alpaca_api_key,
                secret_key=self._settings.alpaca_secret_key,
            )
            # Verify connection
            account = self._trading_client.get_account()
            self._connected = True
            logger.info(
                "Alpaca connected: equity=$%s, buying_power=$%s, paper=%s",
                account.equity, account.buying_power, self._settings.alpaca_paper,
            )
            return True
        except Exception:
            logger.exception("Failed to connect to Alpaca")
            self._connected = False
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Options Chain ────────────────────────────────────────────

    async def get_options_chain(
        self,
        underlying: str,
        expiration: Optional[date] = None,
        option_type: Optional[str] = None,
    ) -> list[OptionsContract]:
        """Fetch options chain for an underlying.

        Args:
            underlying: Ticker symbol (SPY, QQQ, IWM).
            expiration: Expiration date (defaults to today for 0DTE).
            option_type: "call" or "put" (None = both).

        Returns:
            List of OptionsContract with current quotes + Greeks.
        """
        if not self._trading_client:
            return []

        try:
            from alpaca.trading.requests import GetOptionContractsRequest
            from alpaca.trading.enums import ContractType

            exp = expiration or date.today()

            params = GetOptionContractsRequest(
                underlying_symbols=[underlying],
                expiration_date=exp.isoformat(),
                status="active",
            )
            if option_type == "call":
                params.type = ContractType.CALL
            elif option_type == "put":
                params.type = ContractType.PUT

            response = self._trading_client.get_option_contracts(params)
            contracts = []

            if response and hasattr(response, 'option_contracts'):
                for c in response.option_contracts:
                    contracts.append(OptionsContract(
                        symbol=c.symbol,
                        underlying=underlying,
                        option_type="call" if str(c.type).lower() == "call" else "put",
                        strike=Decimal(str(c.strike_price)),
                        expiration=str(c.expiration_date),
                    ))

            logger.info("Fetched %d contracts for %s exp=%s", len(contracts), underlying, exp)
            return contracts

        except Exception:
            logger.exception("Failed to fetch options chain for %s", underlying)
            return []

    async def get_snapshots(
        self, contract_symbols: list[str],
    ) -> dict[str, OptionsContract]:
        """Get real-time snapshots (quote + Greeks) for specific contracts.

        Returns dict mapping symbol -> OptionsContract with populated fields.
        """
        if not self._data_client or not contract_symbols:
            return {}

        try:
            from alpaca.data.requests import OptionSnapshotRequest

            # Alpaca limits batch size — chunk if needed
            results: dict[str, OptionsContract] = {}
            chunk_size = 100

            for i in range(0, len(contract_symbols), chunk_size):
                chunk = contract_symbols[i:i + chunk_size]
                request = OptionSnapshotRequest(symbol_or_symbols=chunk)
                snapshots = self._data_client.get_option_snapshot(request)

                for sym, snap in snapshots.items():
                    quote = snap.latest_quote
                    greeks = snap.greeks if hasattr(snap, 'greeks') and snap.greeks else None

                    contract = OptionsContract(
                        symbol=sym,
                        underlying=sym[:3] if len(sym) > 3 else sym,  # Extract underlying
                        option_type="call" if "C" in sym else "put",
                        strike=Decimal("0"),  # Parsed from symbol if needed
                        expiration="",
                        bid=Decimal(str(quote.bid_price)) if quote else Decimal("0"),
                        ask=Decimal(str(quote.ask_price)) if quote else Decimal("0"),
                        last=Decimal(str(snap.latest_trade.price)) if snap.latest_trade else Decimal("0"),
                        iv=float(snap.implied_volatility) if hasattr(snap, 'implied_volatility') and snap.implied_volatility else 0.0,
                        delta=float(greeks.delta) if greeks and greeks.delta else 0.0,
                        gamma=float(greeks.gamma) if greeks and greeks.gamma else 0.0,
                        theta=float(greeks.theta) if greeks and greeks.theta else 0.0,
                        vega=float(greeks.vega) if greeks and greeks.vega else 0.0,
                        rho=float(greeks.rho) if greeks and greeks.rho else 0.0,
                    )
                    results[sym] = contract

            return results

        except Exception:
            logger.exception("Failed to get option snapshots")
            return {}

    # ── Order Management ─────────────────────────────────────────

    async def place_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        limit_price: float,
        time_in_force: str = "day",
    ) -> Optional[dict]:
        """Place a limit order for an option contract.

        Args:
            symbol: Option contract symbol.
            side: "buy" or "sell".
            qty: Number of contracts.
            limit_price: Limit price per contract.
            time_in_force: "day" or "gtc".

        Returns:
            Order dict or None on failure.
        """
        if not self._trading_client:
            return None

        try:
            from alpaca.trading.requests import LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass

            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
                type="limit",
                time_in_force=TimeInForce.DAY if time_in_force == "day" else TimeInForce.GTC,
                limit_price=round(limit_price, 2),
            )

            order = self._trading_client.submit_order(order_request)
            logger.info(
                "Order placed: %s %s %d @ $%.2f (id=%s)",
                side, symbol, qty, limit_price, order.id,
            )
            return {
                "id": str(order.id),
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "limit_price": limit_price,
                "status": str(order.status),
                "created_at": str(order.created_at),
            }

        except Exception:
            logger.exception("Failed to place order: %s %s %d @ $%.2f", side, symbol, qty, limit_price)
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if not self._trading_client:
            return False
        try:
            self._trading_client.cancel_order_by_id(order_id)
            logger.info("Cancelled order %s", order_id)
            return True
        except Exception:
            logger.exception("Failed to cancel order %s", order_id)
            return False

    async def get_order(self, order_id: str) -> Optional[dict]:
        """Get order status."""
        if not self._trading_client:
            return None
        try:
            order = self._trading_client.get_order_by_id(order_id)
            return {
                "id": str(order.id),
                "status": str(order.status),
                "filled_qty": str(order.filled_qty),
                "filled_avg_price": str(order.filled_avg_price) if order.filled_avg_price else None,
            }
        except Exception:
            logger.exception("Failed to get order %s", order_id)
            return None

    # ── Positions ────────────────────────────────────────────────

    async def get_positions(self) -> list[dict]:
        """Get all open options positions."""
        if not self._trading_client:
            return []
        try:
            positions = self._trading_client.get_all_positions()
            return [
                {
                    "symbol": str(p.symbol),
                    "qty": int(p.qty),
                    "side": str(p.side),
                    "avg_entry_price": str(p.avg_entry_price),
                    "current_price": str(p.current_price),
                    "unrealized_pl": str(p.unrealized_pl),
                    "asset_class": str(p.asset_class),
                }
                for p in positions
                if str(p.asset_class) == "us_option"
            ]
        except Exception:
            logger.exception("Failed to get positions")
            return []

    async def close_position(self, symbol: str) -> Optional[dict]:
        """Close an open position by symbol."""
        if not self._trading_client:
            return None
        try:
            order = self._trading_client.close_position(symbol)
            logger.info("Closing position: %s", symbol)
            return {"id": str(order.id), "status": str(order.status)}
        except Exception:
            logger.exception("Failed to close position %s", symbol)
            return None

    # ── Account ──────────────────────────────────────────────────

    async def get_account(self) -> Optional[dict]:
        """Get account info."""
        if not self._trading_client:
            return None
        try:
            account = self._trading_client.get_account()
            return {
                "equity": str(account.equity),
                "buying_power": str(account.buying_power),
                "cash": str(account.cash),
                "portfolio_value": str(account.portfolio_value),
                "pattern_day_trader": account.pattern_day_trader,
                "daytrade_count": account.daytrade_count,
            }
        except Exception:
            logger.exception("Failed to get account info")
            return None
