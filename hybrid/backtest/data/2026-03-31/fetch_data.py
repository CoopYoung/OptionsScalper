#!/usr/bin/env python3
"""Fetch live market data from Alpaca paper trading API — 2026-03-31 monitor run.

Also uses yfinance as a fallback/supplement for market data when Alpaca
endpoints are unreachable (e.g., network egress restrictions).
"""

import json
import os
import sys
from datetime import datetime, timezone
from collections import OrderedDict

import requests

# ── Credentials & Headers ──────────────────────────────────────────
ALPACA_API_KEY = "PKZBHWZGQWW33A6UBJBYN23ATR"
ALPACA_SECRET_KEY = "5NQnREMU3s3nY7uqzwHWiRzNawBKX2hJehyPp8zXh47o"

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

PAPER_BASE = "https://paper-api.alpaca.markets"
DATA_BASE = "https://data.alpaca.markets"

results = OrderedDict()


def fetch(label, url, params=None):
    """GET a URL, store result, print status."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  GET {url}")
    if params:
        print(f"  Params: {params}")
    print(f"{'='*70}")
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
        print(f"  Status: {resp.status_code}")
        try:
            data = resp.json()
        except Exception:
            data = resp.text
        results[label] = {
            "url": url,
            "params": params,
            "status_code": resp.status_code,
            "data": data,
        }
        dumped = json.dumps(data, indent=2, default=str)
        print(dumped[:3000])
        if len(dumped) > 3000:
            print("  ... (truncated in console, full data saved to file)")
        return data
    except Exception as e:
        print(f"  ERROR: {e}")
        results[label] = {
            "url": url,
            "params": params,
            "status_code": None,
            "error": str(e),
        }
        return None


def fetch_yfinance_ticker(symbol: str) -> dict | None:
    """Fetch latest data for a symbol via yfinance. Returns dict or None."""
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        hist = t.history(period="5d")
        if hist.empty:
            return {"error": f"No data for {symbol}"}
        last = hist.iloc[-1]
        info = {
            "symbol": symbol,
            "close": float(last["Close"]),
            "open": float(last["Open"]),
            "high": float(last["High"]),
            "low": float(last["Low"]),
            "volume": int(last["Volume"]) if "Volume" in hist.columns else 0,
            "date": str(hist.index[-1]),
        }
        # Include last 5 rows for context
        rows = []
        for idx, row in hist.iterrows():
            rows.append({
                "date": str(idx),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]) if "Volume" in hist.columns else 0,
            })
        info["history_5d"] = rows
        return info
    except Exception as e:
        return {"error": str(e)}


# ── 1. Account ─────────────────────────────────────────────────────
fetch("1_account", f"{PAPER_BASE}/v2/account")

# ── 2. Positions ───────────────────────────────────────────────────
fetch("2_positions", f"{PAPER_BASE}/v2/positions")

# ── 3. Today's Orders ─────────────────────────────────────────────
fetch("3_orders_today", f"{PAPER_BASE}/v2/orders", params={
    "status": "all",
    "after": "2026-03-31",
})

# ── 4. Stock Snapshots (SPY, QQQ, IWM) ────────────────────────────
stock_data = fetch("4_stock_snapshots", f"{DATA_BASE}/v2/stocks/snapshots", params={
    "symbols": "SPY,QQQ,IWM",
})

# Determine SPY price for options symbol construction
spy_price = None
if stock_data and isinstance(stock_data, dict) and "SPY" in stock_data:
    try:
        spy_snap = stock_data["SPY"]
        if "latestTrade" in spy_snap and spy_snap["latestTrade"]:
            spy_price = float(spy_snap["latestTrade"].get("p", 0))
        if not spy_price and "latestQuote" in spy_snap and spy_snap["latestQuote"]:
            bp = float(spy_snap["latestQuote"].get("bp", 0))
            ap = float(spy_snap["latestQuote"].get("ap", 0))
            if bp and ap:
                spy_price = (bp + ap) / 2
        if not spy_price and "dailyBar" in spy_snap and spy_snap["dailyBar"]:
            spy_price = float(spy_snap["dailyBar"].get("c", 0))
    except Exception as e:
        print(f"  Could not parse SPY price: {e}")

if spy_price:
    print(f"\n  >>> SPY price detected: ${spy_price:.2f}")
else:
    spy_price = 560.00
    print(f"\n  >>> SPY price fallback: ${spy_price:.2f}")

# ── 5. Options Snapshots (test endpoint) ──────────────────────────
strike_round = round(spy_price)
strike_str = f"{strike_round * 1000:08d}"
call_sym = f"SPY260331C{strike_str}"
put_sym = f"SPY260331P{strike_str}"
print(f"\n  >>> Testing options symbols: {call_sym}, {put_sym}")

fetch("5_options_snapshots", f"{DATA_BASE}/v1beta1/options/snapshots", params={
    "symbols": f"{call_sym},{put_sym}",
})

# ── 6. VIX Proxy (VIXY) ──────────────────────────────────────────
fetch("6_vixy_snapshot", f"{DATA_BASE}/v2/stocks/snapshots", params={
    "symbols": "VIXY",
})

# ── 7. VIX via yfinance ──────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  7_vix_yfinance")
print(f"{'='*70}")
try:
    import yfinance as yf
    vix = yf.Ticker("^VIX")
    hist = vix.history(period="1d")
    if not hist.empty:
        vix_data = {
            "close": float(hist["Close"].iloc[-1]),
            "high": float(hist["High"].iloc[-1]),
            "low": float(hist["Low"].iloc[-1]),
            "open": float(hist["Open"].iloc[-1]),
            "volume": int(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else 0,
            "date": str(hist.index[-1]),
        }
    else:
        vix_data = {"error": "No VIX data returned"}
    results["7_vix_yfinance"] = {"data": vix_data}
    print(json.dumps(vix_data, indent=2, default=str))
except ImportError:
    msg = "yfinance not installed"
    print(f"  {msg}")
    results["7_vix_yfinance"] = {"error": msg}
except Exception as e:
    print(f"  ERROR: {e}")
    results["7_vix_yfinance"] = {"error": str(e)}

# ── 8. SPY Options Chain (today's expiry) ─────────────────────────
fetch("8_spy_options_chain", f"{DATA_BASE}/v1beta1/options/contracts", params={
    "underlying_symbols": "SPY",
    "expiration_date": "2026-03-31",
    "status": "active",
    "limit": 100,
})

# ── 9. QQQ Options Chain (today's expiry) ─────────────────────────
fetch("9_qqq_options_chain", f"{DATA_BASE}/v1beta1/options/contracts", params={
    "underlying_symbols": "QQQ",
    "expiration_date": "2026-03-31",
    "status": "active",
    "limit": 100,
})

# ── 10. IWM Options Chain (today's expiry) ────────────────────────
fetch("10_iwm_options_chain", f"{DATA_BASE}/v1beta1/options/contracts", params={
    "underlying_symbols": "IWM",
    "expiration_date": "2026-03-31",
    "status": "active",
    "limit": 100,
})

# ── 11. yfinance fallback: SPY, QQQ, IWM, VIX ────────────────────
print(f"\n{'='*70}")
print(f"  11_yfinance_fallback (SPY, QQQ, IWM prices + VIX)")
print(f"{'='*70}")
yf_results = {}
for sym in ["SPY", "QQQ", "IWM", "^VIX"]:
    print(f"  Fetching {sym}...")
    yf_results[sym] = fetch_yfinance_ticker(sym)
    if yf_results[sym] and "close" in yf_results[sym]:
        print(f"    {sym}: ${yf_results[sym]['close']:.2f} (date: {yf_results[sym]['date']})")
    else:
        print(f"    {sym}: {yf_results[sym]}")
results["11_yfinance_fallback"] = {"data": yf_results}

# Update spy_price from yfinance if Alpaca failed
if not spy_price or spy_price == 560.00:
    if yf_results.get("SPY") and "close" in yf_results["SPY"]:
        spy_price = yf_results["SPY"]["close"]
        print(f"\n  >>> SPY price updated from yfinance: ${spy_price:.2f}")

# ── Summary ────────────────────────────────────────────────────────
print(f"\n\n{'#'*70}")
print(f"  SUMMARY")
print(f"{'#'*70}")
for label, res in results.items():
    status = res.get("status_code", "N/A")
    error = res.get("error", "")
    data = res.get("data")
    if isinstance(data, list):
        count = f"{len(data)} items"
    elif isinstance(data, dict):
        count = f"{len(data)} keys"
    else:
        count = "n/a"
    err_str = f" | ERROR: {error}" if error else ""
    status_str = str(status) if status else "N/A"
    print(f"  {label:30s} -> HTTP {status_str:>5s} | {count}{err_str}")

# ── Save to file ──────────────────────────────────────────────────
output_path = "/home/user/OptionsScalper/hybrid/backtest/data/2026-03-31/1100_raw_api.json"
output = {
    "fetched_at": datetime.now(timezone.utc).isoformat(),
    "spy_price_used": spy_price,
    "results": results,
}
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=str)

file_size = len(json.dumps(output, default=str))
print(f"\n  Saved full output to: {output_path}")
print(f"  File size: {file_size:,} bytes")
