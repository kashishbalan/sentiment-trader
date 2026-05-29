"""
execution.py
------------
Handles order execution via Alpaca.
Uses paper trading by default — flip ALPACA_BASE_URL to go live.

Env vars required:
  ALPACA_API_KEY
  ALPACA_SECRET_KEY
  ALPACA_BASE_URL  (default: paper trading endpoint)
"""

import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")


def _get_client():
    try:
        from alpaca.trading.client import TradingClient
        return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=(
            "paper" in ALPACA_BASE_URL
        ))
    except ImportError:
        raise ImportError("Run: pip install alpaca-py")


def get_account() -> dict:
    """Returns account info (portfolio value, cash, etc.)"""
    client = _get_client()
    acct = client.get_account()
    return {
        "portfolio_value": float(acct.portfolio_value),
        "cash": float(acct.cash),
        "buying_power": float(acct.buying_power),
        "equity": float(acct.equity),
    }


def get_current_positions() -> dict[str, float]:
    """Returns {ticker: qty} for all open positions."""
    client = _get_client()
    positions = client.get_all_positions()
    return {p.symbol: float(p.qty) for p in positions}


def close_all_positions():
    """Liquidates all open positions (called before rebalancing)."""
    client = _get_client()
    client.close_all_positions(cancel_orders=True)
    logger.info("All positions closed.")


def place_orders(signal_df: pd.DataFrame, portfolio_value: float):
    """
    Given a signal DataFrame with columns [ticker, signal, dollar_size],
    places market orders for all non-FLAT signals.

    Closes existing positions first, then opens new ones.
    """
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    import yfinance as yf

    client = _get_client()
    active = signal_df[signal_df["signal"] != "FLAT"].copy()

    if active.empty:
        logger.info("No signals to trade today.")
        return []

    # Get current prices to compute share quantities
    tickers = active["ticker"].tolist()
    prices = yf.download(tickers, period="1d", progress=False)["Close"].iloc[-1]

    orders_placed = []
    for _, row in active.iterrows():
        ticker = row["ticker"]
        side = OrderSide.BUY if row["signal"] == "LONG" else OrderSide.SELL
        price = prices.get(ticker)

        if not price or price <= 0:
            logger.warning(f"No price for {ticker}, skipping.")
            continue

        qty = max(1, int(row["dollar_size"] / price))

        try:
            req = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
            order = client.submit_order(req)
            logger.info(f"Order placed: {side.value} {qty} {ticker} @ ~${price:.2f}")
            orders_placed.append({
                "ticker": ticker,
                "side": side.value,
                "qty": qty,
                "est_price": round(price, 2),
                "order_id": str(order.id),
            })
        except Exception as e:
            logger.error(f"Order failed for {ticker}: {e}")

    return orders_placed


def rebalance(signal_df: pd.DataFrame):
    """
    Full rebalance: close all → open new positions based on signals.
    """
    acct = get_account()
    portfolio_value = acct["portfolio_value"]
    logger.info(f"Portfolio value: ${portfolio_value:,.2f}")

    close_all_positions()

    from core.signals import compute_position_sizes
    sized_df = compute_position_sizes(signal_df, portfolio_value * 0.95)  # keep 5% cash buffer

    orders = place_orders(sized_df, portfolio_value)
    return orders, acct
