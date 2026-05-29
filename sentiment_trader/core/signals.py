"""
signals.py
----------
Converts daily sentiment scores into trade signals.
Signal logic:
  score >  LONG_THRESHOLD  → LONG
  score < -SHORT_THRESHOLD → SHORT
  otherwise                → FLAT
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)

LONG_THRESHOLD  = 0.35   # tune these after backtesting
SHORT_THRESHOLD = 0.35
MAX_POSITIONS   = 10     # max concurrent positions


def generate_signals(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:  DataFrame with columns [ticker, score, n_articles, date]
    Output: DataFrame with an added 'signal' column: LONG | SHORT | FLAT
    """
    df = sentiment_df.copy()

    def _signal(row):
        if row["n_articles"] < 2:
            return "FLAT"     # not enough data — stay out
        if row["score"] >= LONG_THRESHOLD:
            return "LONG"
        if row["score"] <= -SHORT_THRESHOLD:
            return "SHORT"
        return "FLAT"

    df["signal"] = df.apply(_signal, axis=1)

    # Cap total active positions
    active = df[df["signal"] != "FLAT"].copy()
    active = active.sort_values("score", key=abs, ascending=False).head(MAX_POSITIONS)
    df.loc[~df.index.isin(active.index), "signal"] = "FLAT"

    logger.info(
        f"Signals: {(df['signal']=='LONG').sum()} LONG, "
        f"{(df['signal']=='SHORT').sum()} SHORT, "
        f"{(df['signal']=='FLAT').sum()} FLAT"
    )
    return df


def compute_position_sizes(signal_df: pd.DataFrame, portfolio_value: float) -> pd.DataFrame:
    """
    Equal dollar sizing across active positions.
    Returns DataFrame with 'dollar_size' column added.
    """
    df = signal_df.copy()
    active = df[df["signal"] != "FLAT"]
    n = len(active)
    per_position = portfolio_value / n if n > 0 else 0

    df["dollar_size"] = df["signal"].apply(lambda s: per_position if s != "FLAT" else 0.0)
    return df
