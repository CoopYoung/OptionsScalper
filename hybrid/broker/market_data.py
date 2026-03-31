"""External market data sources for Claude's analysis.

These supplement the Alpaca broker data with macro context, sentiment,
and volatility information. All free-tier or no-auth-required.

Sources:
    - Yahoo Finance (yfinance): VIX, sector performance, volume — no key needed
    - CNN Fear & Greed Index: market regime — no key needed
    - Finnhub: news sentiment, economic calendar, earnings — free key
    - FRED: economic release dates — free key
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any

import requests

from hybrid.config import _get

logger = logging.getLogger(__name__)

def _finnhub_key():
    """Lazy load so .env is fully loaded before reading."""
    return _get("FINNHUB_API_KEY")

def _fred_key():
    return _get("FRED_API_KEY")


# ── VIX & Volatility (no API key needed) ─────────────────────

def get_vix() -> dict:
    """Get current VIX level and regime classification via yfinance."""
    try:
        import yfinance as yf

        vix = yf.Ticker("^VIX")
        hist = vix.history(period="5d", interval="1d")

        if hist.empty:
            return {"error": "No VIX data available"}

        current = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current
        change = current - prev
        change_pct = (change / prev * 100) if prev > 0 else 0

        # Classify regime
        if current < 15:
            regime = "LOW_VOL"
            description = "Low volatility — calm market, tight spreads, smaller moves expected"
        elif current < 20:
            regime = "NORMAL"
            description = "Normal volatility — standard conditions"
        elif current < 25:
            regime = "ELEVATED"
            description = "Elevated volatility — wider moves, consider smaller position sizes"
        elif current < 35:
            regime = "HIGH"
            description = "High volatility — significant moves likely, reduce size, wider stops"
        else:
            regime = "CRISIS"
            description = "Crisis volatility — extreme risk, consider standing aside entirely"

        # 5-day history for trend
        history = [
            {"date": str(hist.index[i].date()), "close": round(float(hist["Close"].iloc[i]), 2)}
            for i in range(len(hist))
        ]

        return {
            "vix": round(current, 2),
            "previous_close": round(prev, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 1),
            "regime": regime,
            "description": description,
            "history_5d": history,
        }
    except Exception as e:
        return {"error": f"VIX fetch failed: {e}"}


# ── Fear & Greed Index (no API key needed) ────────────────────

def get_fear_greed() -> dict:
    """Get CNN Fear & Greed Index — market sentiment regime."""
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.cnn.com/markets/fear-and-greed",
        }
        resp = requests.get(url, headers=headers, timeout=10)

        if resp.status_code != 200:
            return {"error": f"CNN F&G returned {resp.status_code}"}

        data = resp.json()
        fg = data.get("fear_and_greed", {})
        score = int(fg.get("score", 50))
        rating = fg.get("rating", "Neutral")

        # Raw classification — no directional bias, let the LLM decide
        if score <= 25:
            signal = "EXTREME_FEAR"
            advice = "Extreme fear — heavy selling pressure, high panic"
        elif score <= 40:
            signal = "FEAR"
            advice = "Fear — sellers dominating, risk-off sentiment"
        elif score <= 60:
            signal = "NEUTRAL"
            advice = "Neutral — no strong sentiment signal"
        elif score <= 75:
            signal = "GREED"
            advice = "Greed — buyers dominating, risk-on sentiment"
        else:
            signal = "EXTREME_GREED"
            advice = "Extreme greed — euphoric buying, elevated complacency"

        return {
            "score": score,
            "rating": rating,
            "signal": signal,
            "advice": advice,
        }
    except Exception as e:
        return {"error": f"Fear & Greed fetch failed: {e}"}


# ── Sector Performance (no API key needed) ────────────────────

def get_sector_performance() -> dict:
    """Get sector ETF performance — market breadth proxy."""
    try:
        import yfinance as yf

        sectors = {
            "XLK": "Technology", "XLF": "Financials", "XLV": "Healthcare",
            "XLE": "Energy", "XLI": "Industrials", "XLP": "Consumer Staples",
            "XLY": "Consumer Discretionary", "XLB": "Materials",
            "XLU": "Utilities", "XLRE": "Real Estate", "XLC": "Communications",
        }

        tickers = yf.download(
            list(sectors.keys()), period="2d", interval="1d",
            progress=False, group_by="ticker",
        )

        results = []
        up_count = 0
        down_count = 0

        for sym, name in sectors.items():
            try:
                if sym in tickers.columns.get_level_values(0):
                    close_today = float(tickers[sym]["Close"].iloc[-1])
                    close_prev = float(tickers[sym]["Close"].iloc[-2]) if len(tickers[sym]) > 1 else close_today
                    change_pct = round((close_today - close_prev) / close_prev * 100, 2)
                    results.append({"symbol": sym, "sector": name, "change_pct": change_pct})
                    if change_pct > 0:
                        up_count += 1
                    else:
                        down_count += 1
            except (KeyError, IndexError):
                continue

        results.sort(key=lambda x: x["change_pct"], reverse=True)

        breadth = "BROAD_RALLY" if up_count >= 9 else (
            "BROAD_DECLINE" if down_count >= 9 else (
                "ROTATION" if (up_count >= 4 and down_count >= 4) else "MIXED"
            )
        )

        return {
            "sectors": results,
            "up_count": up_count,
            "down_count": down_count,
            "breadth": breadth,
            "description": f"{up_count} sectors up, {down_count} down — {breadth.lower().replace('_', ' ')}",
        }
    except Exception as e:
        return {"error": f"Sector performance fetch failed: {e}"}


# ── Finnhub News Sentiment (free API key) ─────────────────────

def get_news_sentiment(symbol: str = "SPY") -> dict:
    """Get pre-scored news sentiment from Finnhub."""
    if not _finnhub_key():
        return {"error": "FINNHUB_API_KEY not set in .env — get free key at finnhub.io/register"}

    try:
        url = "https://finnhub.io/api/v1/news-sentiment"
        params = {"symbol": symbol, "token": _finnhub_key()}
        resp = requests.get(url, params=params, timeout=10)

        if resp.status_code != 200:
            return {"error": f"Finnhub returned {resp.status_code}"}

        data = resp.json()
        sentiment = data.get("sentiment", {})
        buzz = data.get("buzz", {})

        bullish = float(sentiment.get("bullishPercent", 0.5))
        bearish = float(sentiment.get("bearishPercent", 0.5))
        articles = int(buzz.get("articlesInLastWeek", 0))
        weekly_avg = float(buzz.get("weeklyAverage", 1))

        net_sentiment = round(bullish - bearish, 3)
        buzz_ratio = round(articles / weekly_avg, 2) if weekly_avg > 0 else 0

        if net_sentiment > 0.2:
            signal = "BULLISH"
        elif net_sentiment < -0.2:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        return {
            "symbol": symbol,
            "bullish_pct": round(bullish * 100, 1),
            "bearish_pct": round(bearish * 100, 1),
            "net_sentiment": net_sentiment,
            "signal": signal,
            "articles_this_week": articles,
            "buzz_ratio": buzz_ratio,
            "buzz_description": "Above average news coverage" if buzz_ratio > 1.5 else "Normal coverage",
        }
    except Exception as e:
        return {"error": f"Finnhub sentiment fetch failed: {e}"}


# ── Finnhub Market News (free API key) ────────────────────────

def get_market_news(category: str = "general") -> list[dict]:
    """Get latest market news headlines from Finnhub."""
    if not _finnhub_key():
        return [{"error": "FINNHUB_API_KEY not set"}]

    try:
        url = "https://finnhub.io/api/v1/news"
        params = {"category": category, "token": _finnhub_key()}
        resp = requests.get(url, params=params, timeout=10)

        if resp.status_code != 200:
            return [{"error": f"Finnhub news returned {resp.status_code}"}]

        data = resp.json()
        headlines = []
        for item in data[:10]:  # Top 10 headlines
            headlines.append({
                "headline": item.get("headline", ""),
                "source": item.get("source", ""),
                "summary": item.get("summary", "")[:200],
                "datetime": item.get("datetime", 0),
                "url": item.get("url", ""),
            })
        return headlines
    except Exception as e:
        return [{"error": f"Finnhub news fetch failed: {e}"}]


# ── Finnhub Economic Calendar (free API key) ──────────────────

def get_economic_calendar() -> list[dict]:
    """Get today's and this week's economic events from Finnhub."""
    if not _finnhub_key():
        return [{"error": "FINNHUB_API_KEY not set"}]

    try:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        end = (datetime.utcnow() + timedelta(days=5)).strftime("%Y-%m-%d")

        url = "https://finnhub.io/api/v1/calendar/economic"
        params = {"from": today, "to": end, "token": _finnhub_key()}
        resp = requests.get(url, params=params, timeout=10)

        if resp.status_code != 200:
            return [{"error": f"Finnhub calendar returned {resp.status_code}"}]

        data = resp.json()
        events = []
        for item in data.get("economicCalendar", []):
            country = item.get("country", "")
            if country and country != "US":
                continue

            impact = item.get("impact", "").lower()
            event_name = item.get("event", "")

            # Classify for trading
            is_high_impact = any(kw in event_name.lower() for kw in [
                "fomc", "cpi", "nonfarm", "non-farm", "interest rate",
                "employment situation", "consumer price",
            ])

            events.append({
                "event": event_name,
                "date": item.get("date", today),
                "time": item.get("time", ""),
                "impact": "HIGH" if is_high_impact else impact,
                "estimate": item.get("estimate", ""),
                "previous": item.get("prev", ""),
                "actual": item.get("actual", ""),
                "trading_note": "BLACKOUT — avoid trading ±60 min" if is_high_impact else "",
            })

        return events
    except Exception as e:
        return [{"error": f"Economic calendar fetch failed: {e}"}]


# ── Finnhub Earnings Calendar (free API key) ──────────────────

def get_earnings_calendar() -> list[dict]:
    """Get upcoming earnings for our underlyings and major companies."""
    if not _finnhub_key():
        return [{"error": "FINNHUB_API_KEY not set"}]

    try:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        end = (datetime.utcnow() + timedelta(days=7)).strftime("%Y-%m-%d")

        url = "https://finnhub.io/api/v1/calendar/earnings"
        params = {"from": today, "to": end, "token": _finnhub_key()}
        resp = requests.get(url, params=params, timeout=10)

        if resp.status_code != 200:
            return [{"error": f"Finnhub earnings returned {resp.status_code}"}]

        data = resp.json()
        earnings = []
        for item in data.get("earningsCalendar", []):
            symbol = item.get("symbol", "")
            earnings.append({
                "symbol": symbol,
                "date": item.get("date", ""),
                "hour": item.get("hour", ""),  # bmo/amc
                "eps_estimate": item.get("epsEstimate"),
                "eps_actual": item.get("epsActual"),
                "revenue_estimate": item.get("revenueEstimate"),
            })

        # Sort: our underlyings first, then by date
        priority = {"SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"}
        earnings.sort(key=lambda x: (0 if x["symbol"] in priority else 1, x["date"]))

        return earnings[:20]  # Top 20
    except Exception as e:
        return [{"error": f"Earnings calendar fetch failed: {e}"}]
