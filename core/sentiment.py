"""
sentiment.py
------------
Fetches news headlines and scores them with FinBERT.
Falls back to a lightweight VADER scorer if FinBERT isn't available
(useful for low-RAM environments or quick testing).
"""

import os
import datetime
import logging
from typing import Optional
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# News fetching
# --------------------------------------------------------------------------- #

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")  # set in .env

def fetch_headlines(ticker: str, days_back: int = 1) -> list[dict]:
    """
    Pull headlines from NewsAPI for a given ticker symbol.
    Returns a list of {"title": ..., "publishedAt": ..., "source": ...}
    """
    if not NEWSAPI_KEY:
        logger.warning("NEWSAPI_KEY not set — returning empty headlines.")
        return []

    from_date = (datetime.date.today() - datetime.timedelta(days=days_back)).isoformat()
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": ticker,
        "from": from_date,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 20,
        "apiKey": NEWSAPI_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        articles = r.json().get("articles", [])
        return [
            {
                "ticker": ticker,
                "title": a["title"],
                "publishedAt": a["publishedAt"],
                "source": a["source"]["name"],
            }
            for a in articles
            if a.get("title")
        ]
    except Exception as e:
        logger.error(f"NewsAPI error for {ticker}: {e}")
        return []


# --------------------------------------------------------------------------- #
# Scoring — FinBERT (primary) or VADER (fallback)
# --------------------------------------------------------------------------- #

_finbert_pipe = None

def _load_finbert():
    global _finbert_pipe
    if _finbert_pipe is None:
        try:
            from transformers import pipeline
            _finbert_pipe = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                top_k=None,          # return all labels
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load FinBERT ({e}). Falling back to VADER.")
            _finbert_pipe = "vader"
    return _finbert_pipe


def _vader_score(text: str) -> float:
    """Returns a score in [-1, 1] using VADER."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        return sia.polarity_scores(text)["compound"]
    except ImportError:
        # last resort: neutral
        return 0.0


def score_text(text: str) -> float:
    """
    Returns a sentiment score in [-1, 1].
    FinBERT: positive → +1, negative → -1, neutral → 0 (weighted by prob).
    VADER: compound score directly.
    """
    pipe = _load_finbert()
    if pipe == "vader":
        return _vader_score(text)

    try:
        results = pipe(text[:512])[0]  # list of {label, score}
        label_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        weighted = sum(label_map.get(r["label"], 0) * r["score"] for r in results)
        return round(weighted, 4)
    except Exception as e:
        logger.error(f"FinBERT scoring error: {e}")
        return _vader_score(text)


# --------------------------------------------------------------------------- #
# Aggregate daily sentiment per ticker
# --------------------------------------------------------------------------- #

def compute_daily_sentiment(tickers: list[str], days_back: int = 1) -> pd.DataFrame:
    """
    For each ticker, fetch headlines and compute a mean sentiment score.
    Returns a DataFrame with columns: ticker, score, n_articles, date
    """
    rows = []
    today = datetime.date.today().isoformat()

    for ticker in tickers:
        headlines = fetch_headlines(ticker, days_back=days_back)
        if not headlines:
            rows.append({"ticker": ticker, "score": 0.0, "n_articles": 0, "date": today})
            continue

        scores = [score_text(h["title"]) for h in headlines]
        mean_score = round(sum(scores) / len(scores), 4)
        rows.append({
            "ticker": ticker,
            "score": mean_score,
            "n_articles": len(headlines),
            "date": today,
        })
        logger.info(f"{ticker}: score={mean_score:.3f} over {len(headlines)} articles")

    return pd.DataFrame(rows)
