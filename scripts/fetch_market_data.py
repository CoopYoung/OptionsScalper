#!/usr/bin/env python3
"""Fetch current market data: VIX, CNN Fear & Greed, SPY/QQQ/IWM prices.

Run: python3 scripts/fetch_market_data.py
Requires: pip install yfinance requests
"""

import json
import sys
from datetime import datetime

import requests
import yfinance as yf


def fetch_vix() -> None:
    """Fetch VIX current level and 5-day history from yfinance."""
    print("\n--- VIX (CBOE Volatility Index) ---")
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="5d")
        if hist.empty:
            print("  No VIX data returned from yfinance")
            return
        latest = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else None
        print(f"  Current Level:  {latest['Close']:.2f}")
        if prev is not None:
            change = latest["Close"] - prev["Close"]
            pct = (change / prev["Close"]) * 100
            print(f"  Day Change:     {change:+.2f} ({pct:+.2f}%)")
        print(f"\n  5-Day History:")
        for date, row in hist.iterrows():
            print(
                f"    {date.strftime('%Y-%m-%d')}  "
                f"O:{row['Open']:.2f}  H:{row['High']:.2f}  "
                f"L:{row['Low']:.2f}  C:{row['Close']:.2f}"
            )
    except Exception as e:
        print(f"  ERROR: {e}")


def fetch_fear_greed() -> None:
    """Fetch CNN Fear & Greed Index."""
    print("\n--- CNN Fear & Greed Index ---")
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Referer": "https://www.cnn.com/markets/fear-and-greed",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        fg = data.get("fear_and_greed", {})

        def fmt(val: object) -> str:
            return f"{val:.1f}" if isinstance(val, (int, float)) else str(val)

        print(f"  Current Score:  {fmt(fg.get('score', 'N/A'))}")
        print(f"  Rating:         {fg.get('rating', 'N/A')}")
        print(f"  Previous Close: {fmt(fg.get('previous_close', 'N/A'))}")
        print(f"  1 Week Ago:     {fmt(fg.get('previous_1_week', 'N/A'))}")
        print(f"  1 Month Ago:    {fmt(fg.get('previous_1_month', 'N/A'))}")
        print(f"  1 Year Ago:     {fmt(fg.get('previous_1_year', 'N/A'))}")
    except requests.RequestException as e:
        print(f"  CNN endpoint failed: {e}")
        print("  Trying alternative (crypto) Fear & Greed from alternative.me...")
        try:
            resp2 = requests.get(
                "https://api.alternative.me/fng/?limit=1", timeout=10
            )
            resp2.raise_for_status()
            alt = resp2.json()
            entry = alt.get("data", [{}])[0]
            print(f"  Crypto F&G Score:  {entry.get('value', 'N/A')}")
            print(f"  Classification:    {entry.get('value_classification', 'N/A')}")
            print("  (Note: This is crypto F&G, not the CNN equity index)")
        except Exception as e2:
            print(f"  Alternative also failed: {e2}")


def fetch_etf_prices() -> None:
    """Fetch SPY, QQQ, IWM current prices and day change."""
    print("\n--- ETF Prices ---")
    tickers = ["SPY", "QQQ", "IWM"]
    for symbol in tickers:
        try:
            t = yf.Ticker(symbol)
            hist = t.history(period="5d")
            if hist.empty:
                print(f"\n  {symbol}: No data returned")
                continue
            latest = hist.iloc[-1]
            prev = hist.iloc[-2] if len(hist) > 1 else None
            price = latest["Close"]
            print(f"\n  {symbol}:")
            print(f"    Price:      ${price:.2f}")
            print(f"    Open:       ${latest['Open']:.2f}")
            print(f"    High:       ${latest['High']:.2f}")
            print(f"    Low:        ${latest['Low']:.2f}")
            print(f"    Volume:     {latest['Volume']:,.0f}")
            if prev is not None:
                change = price - prev["Close"]
                pct = (change / prev["Close"]) * 100
                print(f"    Day Change: {change:+.2f} ({pct:+.2f}%)")
        except Exception as e:
            print(f"\n  {symbol}: ERROR - {e}")


def main() -> None:
    print("=" * 60)
    print(f"  MARKET DATA SNAPSHOT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (local time)")
    print("=" * 60)

    fetch_vix()
    fetch_fear_greed()
    fetch_etf_prices()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
