"""Public.com API — supplemental market data.

Uses Public.com's free API for data that Alpaca doesn't provide:
    - INDEX quotes: VIX, SPX directly (no yfinance dependency)
    - Option Greeks with IV: delta, gamma, theta, vega, rho, implied volatility
    - Portfolio view: positions with cost basis and daily gain

Requires PUBLIC_SECRET_KEY in .env (free at public.com/api).
"""

import logging
from typing import Optional

from hybrid.config import _get

logger = logging.getLogger(__name__)

# Lazy client singleton
_client = None
_account_id = None


def _get_client():
    """Create and cache a Public.com API client."""
    global _client, _account_id

    if _client is not None:
        return _client

    secret = _get("PUBLIC_SECRET_KEY")
    if not secret:
        return None

    try:
        from public_api_sdk import (
            ApiKeyAuthConfig,
            PublicApiClient,
            PublicApiClientConfiguration,
        )

        auth = ApiKeyAuthConfig(api_secret_key=secret)

        # Get account ID on first connect
        temp_client = PublicApiClient(auth_config=auth)
        accounts = temp_client.get_accounts()
        if not accounts.accounts:
            logger.error("No Public.com accounts found")
            return None

        _account_id = accounts.accounts[0].account_id
        config = PublicApiClientConfiguration(default_account_number=_account_id)
        _client = PublicApiClient(auth_config=auth, config=config)
        logger.info(f"Public.com client initialized (account: {_account_id})")
        return _client

    except Exception as e:
        logger.error(f"Public.com client init failed: {e}")
        return None


# ── Index Quotes (VIX, SPX, DJX, etc.) ──────────────────────

def get_index_quotes(symbols: list[str] = None) -> dict:
    """Get index quotes — VIX, SPX, etc. directly from Public.com.

    These are not available on Alpaca. No yfinance needed.
    """
    if symbols is None:
        symbols = ["VIX", "SPX"]

    client = _get_client()
    if not client:
        return {"error": "PUBLIC_SECRET_KEY not set — get free key at public.com/api"}

    try:
        from public_api_sdk import OrderInstrument, InstrumentType

        instruments = [
            OrderInstrument(symbol=s, type=InstrumentType.INDEX)
            for s in symbols
        ]
        quotes = client.get_quotes(instruments)

        results = {}
        for q in quotes:
            sym = q.instrument.symbol
            data = {
                "last": q.last,
                "volume": q.volume,
            }
            if q.bid is not None:
                data["bid"] = q.bid
            if q.ask is not None:
                data["ask"] = q.ask

            # Add VIX regime classification
            if sym == "VIX" and q.last:
                vix_val = float(q.last)
                if vix_val < 15:
                    data["regime"] = "LOW_VOL"
                elif vix_val < 20:
                    data["regime"] = "NORMAL"
                elif vix_val < 25:
                    data["regime"] = "ELEVATED"
                elif vix_val < 35:
                    data["regime"] = "HIGH"
                else:
                    data["regime"] = "CRISIS"

            results[sym] = data

        return results

    except Exception as e:
        return {"error": f"Public.com index quotes failed: {e}"}


# ── Option Greeks + IV (batch, up to 250) ────────────────────

def get_option_greeks(osi_symbols: list[str]) -> dict:
    """Get Greeks + implied volatility for option contracts.

    Public.com provides delta, gamma, theta, vega, rho, and IV
    per contract. Alpaca does NOT provide this natively.

    Args:
        osi_symbols: List of OCC/OSI option symbols (max 250)
                     e.g. ["SPY260323C00650000", "SPY260323P00640000"]
    """
    client = _get_client()
    if not client:
        return {"error": "PUBLIC_SECRET_KEY not set"}

    if not osi_symbols:
        return {"error": "No symbols provided"}

    # Cap at 250 (API limit)
    osi_symbols = osi_symbols[:250]

    try:
        response = client.get_option_greeks(osi_symbols)

        results = []
        for g in response.greeks:
            results.append({
                "symbol": g.symbol,
                "delta": g.greeks.delta,
                "gamma": g.greeks.gamma,
                "theta": g.greeks.theta,
                "vega": g.greeks.vega,
                "rho": g.greeks.rho,
                "iv": g.greeks.implied_volatility,
            })

        return {"greeks": results, "count": len(results)}

    except Exception as e:
        return {"error": f"Public.com Greeks fetch failed: {e}"}


# ── Enhanced Option Chain (chain + Greeks in one call) ───────

def get_option_chain_with_greeks(
    symbol: str,
    expiry: str,
    option_type: str = None,
    near_money_range: int = 10,
) -> dict:
    """Get option chain with Greeks + IV from Public.com.

    Fetches the chain, then batch-fetches Greeks for near-money strikes.
    Returns a combined view with quotes AND Greeks/IV in one response.

    Args:
        symbol: Underlying (e.g. "SPY")
        expiry: Expiration date "YYYY-MM-DD"
        option_type: "call", "put", or None for both
        near_money_range: Number of strikes above/below ATM to include
    """
    client = _get_client()
    if not client:
        return {"error": "PUBLIC_SECRET_KEY not set"}

    try:
        from public_api_sdk import (
            OptionChainRequest,
            OrderInstrument,
            InstrumentType,
        )

        # Get current price for ATM reference
        equity_quotes = client.get_quotes([
            OrderInstrument(symbol=symbol, type=InstrumentType.EQUITY)
        ])
        spot_price = float(equity_quotes[0].last) if equity_quotes else None

        # Fetch chain
        req = OptionChainRequest(
            instrument=OrderInstrument(symbol=symbol, type=InstrumentType.EQUITY),
            expiration_date=expiry,
        )
        chain = client.get_option_chain(req)

        # Filter to near-money strikes
        def _extract_strike(osi_sym: str) -> float:
            """Extract strike price from OSI symbol."""
            try:
                # Format: SPY260323C00650000 — last 8 digits are strike × 1000
                return int(osi_sym[-8:]) / 1000.0
            except (ValueError, IndexError):
                return 0.0

        def _filter_near_money(contracts, spot):
            """Keep contracts within near_money_range strikes of ATM."""
            if not spot:
                return contracts[:near_money_range * 2]
            filtered = []
            for c in contracts:
                strike = _extract_strike(c.instrument.symbol)
                if abs(strike - spot) / spot <= (near_money_range * 0.005):
                    filtered.append(c)
            return filtered or contracts[:near_money_range * 2]

        calls = _filter_near_money(chain.calls, spot_price) if option_type != "put" else []
        puts = _filter_near_money(chain.puts, spot_price) if option_type != "call" else []

        # Batch fetch Greeks for all near-money contracts
        all_symbols = [c.instrument.symbol for c in calls + puts]
        greeks_map = {}
        if all_symbols:
            try:
                greeks_resp = client.get_option_greeks(all_symbols[:250])
                for g in greeks_resp.greeks:
                    greeks_map[g.symbol] = {
                        "delta": g.greeks.delta,
                        "gamma": g.greeks.gamma,
                        "theta": g.greeks.theta,
                        "vega": g.greeks.vega,
                        "rho": g.greeks.rho,
                        "iv": g.greeks.implied_volatility,
                    }
            except Exception as e:
                logger.warning(f"Greeks fetch failed, continuing without: {e}")

        # Build combined response
        def _format_contract(c):
            sym = c.instrument.symbol
            strike = _extract_strike(sym)
            entry = {
                "symbol": sym,
                "strike": strike,
                "last": c.last,
                "bid": c.bid,
                "ask": c.ask,
                "volume": c.volume,
                "open_interest": c.open_interest,
            }
            if c.bid and c.ask:
                mid = (float(c.bid) + float(c.ask)) / 2
                spread = float(c.ask) - float(c.bid)
                entry["mid"] = round(mid, 2)
                entry["spread"] = round(spread, 2)
                entry["spread_pct"] = round(spread / mid * 100, 1) if mid > 0 else None

            # Merge Greeks if available
            if sym in greeks_map:
                entry.update(greeks_map[sym])

            return entry

        result = {
            "underlying": symbol,
            "expiry": expiry,
            "spot_price": spot_price,
        }

        if option_type != "put":
            result["calls"] = [_format_contract(c) for c in calls]
        if option_type != "call":
            result["puts"] = [_format_contract(c) for c in puts]

        total = len(result.get("calls", [])) + len(result.get("puts", []))
        result["contracts_returned"] = total
        result["greeks_available"] = len(greeks_map)

        return result

    except Exception as e:
        return {"error": f"Public.com chain+greeks failed: {e}"}


# ── Portfolio View ───────────────────────────────────────────

def get_public_portfolio() -> dict:
    """Get Public.com portfolio — positions with cost basis and daily gain.

    Useful for seeing the Public.com account state if/when we
    transition to live trading there.
    """
    client = _get_client()
    if not client:
        return {"error": "PUBLIC_SECRET_KEY not set"}

    try:
        portfolio = client.get_portfolio()

        result = {
            "account_id": portfolio.account_id,
            "buying_power": {},
            "equity_breakdown": [],
            "positions": [],
        }

        if portfolio.buying_power:
            bp = portfolio.buying_power
            result["buying_power"] = {
                "cash_only": bp.cash_only_buying_power,
                "buying_power": bp.buying_power,
                "options_buying_power": bp.options_buying_power,
            }

        if portfolio.equity:
            for e in portfolio.equity:
                result["equity_breakdown"].append({
                    "type": e.type.value if e.type else "UNKNOWN",
                    "value": e.value,
                    "pct_of_portfolio": e.percentage_of_portfolio,
                })

        if portfolio.positions:
            for p in portfolio.positions:
                pos = {
                    "symbol": p.instrument.symbol if p.instrument else "?",
                    "type": p.instrument.type.value if p.instrument and p.instrument.type else "?",
                    "quantity": p.quantity,
                    "current_value": p.current_value,
                    "pct_of_portfolio": p.percent_of_portfolio,
                }
                if p.last_price:
                    pos["last_price"] = p.last_price.last_price
                if p.instrument_gain:
                    pos["total_gain"] = p.instrument_gain.gain_value
                    pos["total_gain_pct"] = p.instrument_gain.gain_percentage
                if p.position_daily_gain:
                    pos["daily_gain"] = p.position_daily_gain.gain_value
                    pos["daily_gain_pct"] = p.position_daily_gain.gain_percentage
                if p.cost_basis:
                    pos["cost_basis"] = p.cost_basis.total_cost
                    pos["unit_cost"] = p.cost_basis.unit_cost
                result["positions"].append(pos)

        return result

    except Exception as e:
        return {"error": f"Public.com portfolio failed: {e}"}
