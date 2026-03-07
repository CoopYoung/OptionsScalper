"""Multi-source market sentiment scoring.

Sources:
    1. CNN Fear & Greed Index (every 30 min)
    2. X/Twitter cashtag sentiment (every 15 min) — FinBERT NLP
    3. Financial news headlines (every 15 min) — FinBERT NLP
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from src.infra.config import Settings

logger = logging.getLogger(__name__)


class SentimentRegime(str, Enum):
    EXTREME_FEAR = "extreme_fear"    # 0-25 → contrarian bullish
    FEAR = "fear"                     # 25-45 → lean bullish
    NEUTRAL = "neutral"               # 45-55
    GREED = "greed"                   # 55-75 → lean bearish
    EXTREME_GREED = "extreme_greed"  # 75-100 → contrarian bearish


@dataclass
class SentimentSignals:
    """Composite sentiment signals."""
    composite_score: float           # -1.0 (fear) to +1.0 (greed)
    regime: SentimentRegime
    fear_greed_index: int            # 0-100 (CNN)
    x_sentiment: float               # -1 to +1 (X/Twitter cashtags)
    news_sentiment: float            # -1 to +1
    contrarian_signal: float         # -1 to +1 (fade extremes)
    news_catalyst: bool              # Is there a breaking event?
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SentimentAggregator:
    """Multi-source market sentiment scoring."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._latest: Optional[SentimentSignals] = None
        self._finbert_model = None
        self._finbert_tokenizer = None

    async def update(self) -> SentimentSignals:
        """Refresh all sentiment sources and compute composite."""
        fg_index = await self._fetch_fear_greed()
        x_sent = await self._fetch_x_sentiment()
        news_sent = await self._fetch_news_sentiment()

        # Composite: weighted average
        composite = (
            self._fg_to_score(fg_index) * 0.40 +
            x_sent * 0.30 +
            news_sent * 0.30
        )

        # Regime classification
        regime = self._classify_regime(fg_index)

        # Contrarian signal: fade extremes
        contrarian = 0.0
        if regime == SentimentRegime.EXTREME_FEAR:
            contrarian = 0.5   # Contrarian bullish
        elif regime == SentimentRegime.EXTREME_GREED:
            contrarian = -0.5  # Contrarian bearish
        elif regime == SentimentRegime.FEAR:
            contrarian = 0.2
        elif regime == SentimentRegime.GREED:
            contrarian = -0.2

        # News catalyst detection (high absolute sentiment = event)
        news_catalyst = abs(news_sent) > 0.6

        signals = SentimentSignals(
            composite_score=round(composite, 3),
            regime=regime,
            fear_greed_index=fg_index,
            x_sentiment=round(x_sent, 3),
            news_sentiment=round(news_sent, 3),
            contrarian_signal=round(contrarian, 3),
            news_catalyst=news_catalyst,
        )
        self._latest = signals

        logger.info(
            "Sentiment: F&G=%d regime=%s composite=%.2f contrarian=%.2f catalyst=%s",
            fg_index, regime.value, composite, contrarian, news_catalyst,
        )
        return signals

    async def _fetch_fear_greed(self) -> int:
        """Fetch CNN Fear & Greed Index."""
        try:
            import aiohttp

            url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
            headers = {"User-Agent": "Mozilla/5.0"}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        return 50
                    data = await resp.json()
                    score = data.get("fear_and_greed", {}).get("score", 50)
                    return int(score)

        except Exception:
            logger.debug("CNN Fear & Greed fetch failed")
            return 50

    async def _fetch_x_sentiment(self) -> float:
        """Analyze X/Twitter cashtag sentiment using FinBERT.

        Searches recent tweets mentioning $SPY, $QQQ, $IWM and scores
        them through the NLP pipeline. Requires X_BEARER_TOKEN.
        """
        if not self._settings.x_bearer_token:
            return 0.0

        try:
            import aiohttp

            # Search for cashtags related to our underlyings
            cashtags = " OR ".join(
                f"${sym}" for sym in self._settings.underlying_list
            )
            query = f"({cashtags}) lang:en -is:retweet"

            url = "https://api.x.com/2/tweets/search/recent"
            headers = {
                "Authorization": f"Bearer {self._settings.x_bearer_token}",
            }
            params = {
                "query": query,
                "max_results": 50,
                "tweet.fields": "text",
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=headers, params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        logger.debug("X API returned %d", resp.status)
                        return 0.0

                    data = await resp.json()

            tweets = data.get("data", [])
            if not tweets:
                return 0.0

            texts = [t["text"] for t in tweets]
            sentiments = self._analyze_texts(texts)
            return sum(sentiments) / len(sentiments) if sentiments else 0.0

        except Exception:
            logger.debug("X sentiment fetch failed")
            return 0.0

    async def _fetch_news_sentiment(self) -> float:
        """Analyze financial news headlines using FinBERT."""
        try:
            import feedparser

            feeds = [
                "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY",
                "https://www.marketwatch.com/rss/topstories",
            ]

            headlines = []
            for feed_url in feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries[:10]:
                        headlines.append(entry.title)
                except Exception:
                    continue

            if not headlines:
                return 0.0

            sentiments = self._analyze_texts(headlines)
            return sum(sentiments) / len(sentiments) if sentiments else 0.0

        except Exception:
            logger.debug("News sentiment fetch failed")
            return 0.0

    def _analyze_texts(self, texts: list[str]) -> list[float]:
        """Run FinBERT sentiment analysis on texts. Falls back to TextBlob."""
        scores = []

        # Try FinBERT first
        try:
            if self._finbert_model is None:
                self._load_finbert()

            if self._finbert_model and self._finbert_tokenizer:
                import torch

                for text in texts:
                    inputs = self._finbert_tokenizer(
                        text, return_tensors="pt", truncation=True, max_length=128,
                    )
                    with torch.no_grad():
                        outputs = self._finbert_model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)[0]
                    # FinBERT: [negative, neutral, positive]
                    neg, neu, pos = probs.tolist()
                    scores.append(pos - neg)  # -1 to +1

                return scores

        except Exception:
            logger.debug("FinBERT failed, falling back to TextBlob")

        # Fallback: TextBlob
        try:
            from textblob import TextBlob

            for text in texts:
                blob = TextBlob(text)
                scores.append(blob.sentiment.polarity)

        except Exception:
            logger.debug("TextBlob fallback also failed")

        return scores

    def _load_finbert(self) -> None:
        """Lazy-load FinBERT model."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model_name = "ProsusAI/finbert"
            self._finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            logger.info("FinBERT model loaded")

        except Exception:
            logger.warning("FinBERT not available, using TextBlob fallback")
            self._finbert_model = None
            self._finbert_tokenizer = None

    def _fg_to_score(self, fg_index: int) -> float:
        """Convert Fear & Greed Index (0-100) to -1 to +1 score."""
        return (fg_index - 50) / 50

    def _classify_regime(self, fg_index: int) -> SentimentRegime:
        if fg_index <= 25:
            return SentimentRegime.EXTREME_FEAR
        if fg_index <= 45:
            return SentimentRegime.FEAR
        if fg_index <= 55:
            return SentimentRegime.NEUTRAL
        if fg_index <= 75:
            return SentimentRegime.GREED
        return SentimentRegime.EXTREME_GREED

    @property
    def latest(self) -> Optional[SentimentSignals]:
        return self._latest

    def get_score(self, direction: str) -> float:
        """Score for the ensemble. Uses CONTRARIAN logic at extremes."""
        if not self._latest:
            return 0.0

        signals = self._latest

        # At extremes, use contrarian signal
        if signals.regime in (SentimentRegime.EXTREME_FEAR, SentimentRegime.EXTREME_GREED):
            if direction == "call":
                return signals.contrarian_signal  # Positive when extreme fear
            return -signals.contrarian_signal      # Positive when extreme greed

        # In normal regimes, sentiment alignment gives mild boost
        if direction == "call":
            return signals.composite_score * 0.3
        return -signals.composite_score * 0.3
