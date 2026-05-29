"""
backtest.py
-----------
Historical backtest of the sentiment strategy.
Uses a pre-built CSV of sentiment scores (or regenerates them).

Outputs:
  - Equity curve
  - Sharpe ratio
  - Max drawdown
  - Win rate
  - Per-ticker attribution
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def load_price_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Returns daily close prices as a DataFrame (dates × tickers)."""
    raw = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])
    return raw.dropna(how="all")


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().shift(-1)  # next-day returns


def run_backtest(
    sentiment_path: str,
    long_threshold: float = 0.35,
    short_threshold: float = 0.35,
    max_positions: int = 10,
    start: str = "2022-01-01",
    end: str = "2024-01-01",
) -> dict:
    """
    Runs a vectorised backtest.

    sentiment_path: CSV with columns [date, ticker, score, n_articles]
    Returns a dict of performance metrics + equity_curve (pd.Series).
    """
    sent_df = pd.read_csv(sentiment_path, parse_dates=["date"])
    tickers = sent_df["ticker"].unique().tolist()

    prices = load_price_data(tickers, start, end)
    fwd_returns = compute_returns(prices)

    daily_pnl = []
    equity = 1.0

    for date, day_sent in sent_df.groupby("date"):
        if date not in fwd_returns.index:
            continue

        day_sent = day_sent[day_sent["n_articles"] >= 2].copy()

        longs  = day_sent[day_sent["score"] >= long_threshold]["ticker"].tolist()
        shorts = day_sent[day_sent["score"] <= -short_threshold]["ticker"].tolist()

        # Sort by |score|, take top max_positions
        active = day_sent[day_sent["ticker"].isin(longs + shorts)]
        active = active.sort_values("score", key=abs, ascending=False).head(max_positions)

        if active.empty:
            daily_pnl.append({"date": date, "pnl": 0.0, "n_trades": 0})
            continue

        day_rets = fwd_returns.loc[date]
        position_return = 0.0
        n = len(active)

        for _, row in active.iterrows():
            t = row["ticker"]
            if t not in day_rets or pd.isna(day_rets[t]):
                continue
            direction = 1.0 if row["score"] >= long_threshold else -1.0
            position_return += direction * day_rets[t] / n  # equal weight

        daily_pnl.append({"date": date, "pnl": position_return, "n_trades": n})
        equity *= (1 + position_return)

    results = pd.DataFrame(daily_pnl).set_index("date")
    equity_curve = (1 + results["pnl"]).cumprod()

    # ── Metrics ──────────────────────────────────────────────────────────── #
    rets = results["pnl"]
    ann_return = rets.mean() * 252
    ann_vol    = rets.std() * np.sqrt(252)
    sharpe     = ann_return / ann_vol if ann_vol > 0 else 0
    drawdown   = (equity_curve / equity_curve.cummax() - 1).min()
    win_rate   = (rets > 0).mean()

    metrics = {
        "annualized_return": round(ann_return * 100, 2),
        "annualized_vol":    round(ann_vol * 100, 2),
        "sharpe_ratio":      round(sharpe, 3),
        "max_drawdown":      round(drawdown * 100, 2),
        "win_rate":          round(win_rate * 100, 2),
        "total_trading_days": len(results),
        "equity_curve":      equity_curve,
        "daily_pnl":         results,
    }
    return metrics


def print_summary(metrics: dict):
    ec = metrics.pop("equity_curve")
    dp = metrics.pop("daily_pnl")
    print("\n── Backtest Results ─────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<25} {v}")
    print(f"  {'final_equity':<25} {ec.iloc[-1]:.4f}x")
    print("─────────────────────────────────────────────────\n")
    metrics["equity_curve"] = ec
    metrics["daily_pnl"]    = dp
