"""Abstract broker interface — Alpaca (paper) and Public.com (live) share this contract.

The orchestrator only talks to the Broker protocol. Swap implementations
by changing --mode paper|live.
"""

from typing import Protocol


class Broker(Protocol):
    """Minimal interface for trading operations."""

    def get_account(self) -> dict: ...
    def get_positions(self) -> list[dict]: ...
    def get_orders(self, status: str = "open") -> list[dict]: ...

    # Market data
    def get_stock_quotes(self, symbols: list[str]) -> dict[str, dict]: ...
    def get_stock_bars(self, symbol: str, timeframe: str = "5Min",
                       limit: int = 50) -> list[dict]: ...
    def get_option_chain(self, underlying: str, expiration_date: str | None = None,
                         option_type: str | None = None) -> list[dict]: ...
    def get_option_expirations(self, underlying: str) -> list[str]: ...
    def get_option_quote(self, symbol: str) -> dict: ...

    # Execution
    def place_order(self, symbol: str, qty: int, side: str, order_type: str,
                    time_in_force: str = "day",
                    limit_price: float | None = None) -> dict: ...
    def close_position(self, symbol: str, qty: int | None = None) -> dict: ...
    def cancel_order(self, order_id: str) -> dict: ...


class AlpacaBroker:
    """Wraps hybrid.broker.alpaca module functions into the Broker protocol."""

    def get_account(self) -> dict:
        from hybrid.broker import alpaca
        return alpaca.get_account()

    def get_positions(self) -> list[dict]:
        from hybrid.broker import alpaca
        return alpaca.get_positions()

    def get_orders(self, status: str = "open") -> list[dict]:
        from hybrid.broker import alpaca
        return alpaca.get_orders(status)

    def get_stock_quotes(self, symbols: list[str]) -> dict[str, dict]:
        from hybrid.broker import alpaca
        return alpaca.get_stock_quotes(symbols)

    def get_stock_bars(self, symbol: str, timeframe: str = "5Min",
                       limit: int = 50) -> list[dict]:
        from hybrid.broker import alpaca
        return alpaca.get_stock_bars(symbol, timeframe, limit)

    def get_option_chain(self, underlying: str, expiration_date: str | None = None,
                         option_type: str | None = None) -> list[dict]:
        from hybrid.broker import alpaca
        return alpaca.get_option_chain(underlying, expiration_date, option_type)

    def get_option_expirations(self, underlying: str) -> list[str]:
        from hybrid.broker import alpaca
        return alpaca.get_option_expirations(underlying)

    def get_option_quote(self, symbol: str) -> dict:
        from hybrid.broker import alpaca
        return alpaca.get_option_quote(symbol)

    def place_order(self, symbol: str, qty: int, side: str, order_type: str,
                    time_in_force: str = "day",
                    limit_price: float | None = None) -> dict:
        from hybrid.broker import alpaca
        return alpaca.place_order(symbol, qty, side, order_type,
                                 time_in_force, limit_price)

    def close_position(self, symbol: str, qty: int | None = None) -> dict:
        from hybrid.broker import alpaca
        return alpaca.close_position(symbol, qty)

    def cancel_order(self, order_id: str) -> dict:
        from hybrid.broker import alpaca
        return alpaca.cancel_order(order_id)
