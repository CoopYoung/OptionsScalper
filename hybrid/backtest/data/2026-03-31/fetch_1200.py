#!/usr/bin/env python3
"""Fetch all Alpaca paper trading data for 2026-03-31 12:00 snapshot."""

import json
import requests
from datetime import date, datetime

API_KEY = "PKZBHWZGQWW33A6UBJBYN23ATR"
SECRET_KEY = "5NQnREMU3s3nY7uqzwHWiRzNawBKX2hJehyPp8zXh47o"
TRADE_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"
HEADERS = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": SECRET_KEY}
TODAY = "2026-03-31"
OUTPUT_FILE = f"/home/user/OptionsScalper/hybrid/backtest/data/{TODAY}/1200_raw_api.json"

results = {}

def fetch(label, url, params=None):
    """Fetch a URL and store results, handling errors gracefully."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            return data
        else:
            err = {"error": resp.status_code, "message": resp.text[:500]}
            print(f"  Error: {resp.text[:300]}")
            return err
    except Exception as e:
        err = {"error": str(e)}
        print(f"  Exception: {e}")
        return err

# 1. Account info
data = fetch("1. Account Info", f"{TRADE_URL}/v2/account")
results["account"] = data
if isinstance(data, dict) and "error" not in data:
    print(f"  Equity: ${data.get('equity', 'N/A')}")
    print(f"  Buying Power: ${data.get('buying_power', 'N/A')}")
    print(f"  Cash: ${data.get('cash', 'N/A')}")
    print(f"  Portfolio Value: ${data.get('portfolio_value', 'N/A')}")
    print(f"  Day Trade Count: {data.get('daytrade_count', 'N/A')}")
    print(f"  PDT Flagged: {data.get('pattern_day_trader', 'N/A')}")

# 2. Open positions
data = fetch("2. Open Positions", f"{TRADE_URL}/v2/positions")
results["positions"] = data
if isinstance(data, list):
    print(f"  Count: {len(data)}")
    for p in data:
        print(f"    {p.get('symbol')}: qty={p.get('qty')} side={p.get('side')} "
              f"avg_entry={p.get('avg_entry_price')} current={p.get('current_price')} "
              f"unrealized_pl={p.get('unrealized_pl')}")

# 3. Today's orders
data = fetch("3. Today's Orders", f"{TRADE_URL}/v2/orders",
             params={"status": "all", "after": f"{TODAY}T00:00:00Z", "limit": 100})
results["orders"] = data
if isinstance(data, list):
    print(f"  Count: {len(data)}")
    for o in data[:10]:
        print(f"    {o.get('symbol')}: {o.get('side')} {o.get('qty')} @ {o.get('limit_price', 'market')} "
              f"status={o.get('status')} type={o.get('order_type')}")

# 4. Current quotes for SPY, QQQ, IWM
data = fetch("4. Stock Quotes (SPY, QQQ, IWM)",
             f"{DATA_URL}/v2/stocks/quotes/latest",
             params={"symbols": "SPY,QQQ,IWM", "feed": "iex"})
results["stock_quotes"] = data
if isinstance(data, dict) and "quotes" in data:
    for sym, q in data["quotes"].items():
        print(f"  {sym}: bid={q.get('bp')} ask={q.get('ap')} bid_sz={q.get('bs')} ask_sz={q.get('as')}")

# 5. VIX proxy (VIXY)
data = fetch("5. VIX Proxy (VIXY)",
             f"{DATA_URL}/v2/stocks/quotes/latest",
             params={"symbols": "VIXY", "feed": "iex"})
results["vix_proxy"] = data
if isinstance(data, dict) and "quotes" in data:
    for sym, q in data["quotes"].items():
        print(f"  {sym}: bid={q.get('bp')} ask={q.get('ap')}")

# 6. 5-min bars (last 50)
data = fetch("6. 5-Min Bars (SPY, QQQ, IWM)",
             f"{DATA_URL}/v2/stocks/bars",
             params={"symbols": "SPY,QQQ,IWM", "timeframe": "5Min", "limit": 50, "feed": "iex"})
results["bars_5min"] = data
if isinstance(data, dict) and "bars" in data:
    for sym, bars in data["bars"].items():
        if bars:
            last = bars[-1]
            print(f"  {sym}: {len(bars)} bars, last: o={last.get('o')} h={last.get('h')} "
                  f"l={last.get('l')} c={last.get('c')} v={last.get('v')} t={last.get('t')}")

# 7-9. Options chains for SPY, QQQ, IWM 0DTE
all_option_symbols = []
for underlying in ["SPY", "QQQ", "IWM"]:
    label = f"7-9. Options Chain {underlying} 0DTE ({TODAY})"
    data = fetch(label, f"{TRADE_URL}/v2/options/contracts",
                 params={
                     "underlying_symbols": underlying,
                     "expiration_date": TODAY,
                     "status": "active",
                     "limit": 250
                 })
    key = f"options_chain_{underlying.lower()}"
    results[key] = data

    if isinstance(data, dict) and "option_contracts" in data:
        contracts = data["option_contracts"]
        print(f"  Contracts found: {len(contracts)}")
        # Collect symbols for snapshot
        for c in contracts:
            all_option_symbols.append(c["symbol"])
        # Print sample
        if contracts:
            calls = [c for c in contracts if c.get("type") == "call"]
            puts = [c for c in contracts if c.get("type") == "put"]
            print(f"  Calls: {len(calls)}, Puts: {len(puts)}")
            # Show a few near-ATM
            sample = contracts[:5]
            for c in sample:
                print(f"    {c['symbol']}: {c['type']} strike={c['strike_price']} "
                      f"exp={c['expiration_date']} status={c['status']}")
    elif isinstance(data, list):
        print(f"  Contracts found: {len(data)}")
        for c in data:
            all_option_symbols.append(c.get("symbol", ""))
        for c in data[:5]:
            print(f"    {c.get('symbol')}: strike={c.get('strike_price')}")

# 10. Options snapshots (batch up to 100 at a time)
print(f"\n{'='*60}")
print(f"  10. Options Snapshots")
print(f"{'='*60}")
print(f"  Total option symbols to fetch: {len(all_option_symbols)}")

results["options_snapshots"] = {}
batch_size = 100
for i in range(0, len(all_option_symbols), batch_size):
    batch = all_option_symbols[i:i+batch_size]
    symbols_str = ",".join(batch)
    batch_num = i // batch_size + 1
    print(f"\n  Batch {batch_num}: {len(batch)} symbols")
    try:
        resp = requests.get(
            f"{DATA_URL}/v1beta1/options/snapshots",
            headers=HEADERS,
            params={"symbols": symbols_str, "feed": "indicative"},
            timeout=30
        )
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 200:
            snap_data = resp.json()
            # The response has a "snapshots" key
            snapshots = snap_data.get("snapshots", snap_data)
            if isinstance(snapshots, dict):
                results["options_snapshots"].update(snapshots)
                print(f"  Got {len(snapshots)} snapshots in this batch")
                # Print a few samples with greeks
                for sym, snap in list(snapshots.items())[:3]:
                    greeks = snap.get("greeks", {})
                    quote = snap.get("latestQuote", snap.get("latest_quote", {}))
                    trade = snap.get("latestTrade", snap.get("latest_trade", {}))
                    print(f"    {sym}:")
                    print(f"      Greeks: delta={greeks.get('delta')} gamma={greeks.get('gamma')} "
                          f"theta={greeks.get('theta')} vega={greeks.get('vega')} iv={greeks.get('implied_volatility')}")
                    print(f"      Quote: bid={quote.get('bp')} ask={quote.get('ap')}")
                    if trade:
                        print(f"      Last: price={trade.get('p')} size={trade.get('s')}")
            else:
                results["options_snapshots"][f"batch_{batch_num}"] = snap_data
                print(f"  Unexpected format: {str(snap_data)[:200]}")
        else:
            print(f"  Error: {resp.text[:300]}")
            results["options_snapshots"][f"batch_{batch_num}_error"] = {
                "status": resp.status_code, "message": resp.text[:500]
            }
    except Exception as e:
        print(f"  Exception: {e}")
        results["options_snapshots"][f"batch_{batch_num}_error"] = {"error": str(e)}

# Summary
print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"  Account: {'OK' if 'error' not in (results.get('account') or {}) else 'ERROR'}")
print(f"  Positions: {len(results.get('positions', [])) if isinstance(results.get('positions'), list) else 'ERROR'}")
print(f"  Orders: {len(results.get('orders', [])) if isinstance(results.get('orders'), list) else 'ERROR'}")
print(f"  Stock Quotes: {'OK' if results.get('stock_quotes') else 'ERROR'}")
print(f"  VIX Proxy: {'OK' if results.get('vix_proxy') else 'ERROR'}")
print(f"  5-Min Bars: {'OK' if results.get('bars_5min') else 'ERROR'}")
for u in ["spy", "qqq", "iwm"]:
    chain = results.get(f"options_chain_{u}")
    if isinstance(chain, dict) and "option_contracts" in chain:
        cnt = len(chain["option_contracts"])
    elif isinstance(chain, list):
        cnt = len(chain)
    else:
        cnt = "ERROR"
    print(f"  Options Chain {u.upper()}: {cnt} contracts")
print(f"  Options Snapshots: {len(results.get('options_snapshots', {}))} total")

# Save to file
print(f"\n  Saving to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"  Saved! File size: ", end="")

import os
size = os.path.getsize(OUTPUT_FILE)
if size > 1024 * 1024:
    print(f"{size / 1024 / 1024:.1f} MB")
elif size > 1024:
    print(f"{size / 1024:.1f} KB")
else:
    print(f"{size} bytes")

print("\nDone!")
