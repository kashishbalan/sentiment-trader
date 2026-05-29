"""
main.py
-------
Entry point. Run daily (e.g. via cron at 9:25 AM ET before market open).

Modes:
  python main.py --mode live       # score → signal → execute on Alpaca
  python main.py --mode dry-run    # score → signal → print orders (no execution)
  python main.py --mode backtest   # run historical backtest
"""

import argparse
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Ticker universe ──────────────────────────────────────────────────────── #
# S&P 100 subset — liquid, well-covered by news
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM",
    "V", "UNH", "JNJ", "XOM", "WMT", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "PEP", "KO", "BAC", "LLY", "AVGO", "COST",
]


def run_live(dry_run: bool = False):
    from core.sentiment import compute_daily_sentiment
    from core.signals import generate_signals

    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Starting daily pipeline...")

    # 1. Sentiment
    sent_df = compute_daily_sentiment(UNIVERSE, days_back=1)
    logger.info(f"\n{sent_df.sort_values('score', ascending=False).to_string(index=False)}\n")

    # 2. Signals
    signal_df = generate_signals(sent_df)
    active = signal_df[signal_df["signal"] != "FLAT"]
    logger.info(f"\nActive signals:\n{active[['ticker','score','signal']].to_string(index=False)}\n")

    if dry_run:
        logger.info("[DRY RUN] No orders placed.")
        return

    # 3. Execute
    from core.execution import rebalance
    orders, acct = rebalance(signal_df)

    logger.info(f"\nOrders placed: {len(orders)}")
    for o in orders:
        logger.info(f"  {o['side']:5s} {o['qty']:>4} {o['ticker']} @ ~${o['est_price']:.2f}")

    logger.info(f"\nAccount: equity=${acct['equity']:,.2f}  cash=${acct['cash']:,.2f}")

    # Save run log
    signal_df.to_csv(f"data/signals_{signal_df['date'].iloc[0]}.csv", index=False)


def run_backtest():
    from core.backtest import run_backtest as _bt, print_summary

    sent_path = "data/historical_sentiment.csv"
    if not Path(sent_path).exists():
        logger.error(
            f"Missing {sent_path}. "
            "Generate it by running sentiment scoring over historical data first.\n"
            "See README.md for instructions."
        )
        sys.exit(1)

    metrics = _bt(
        sentiment_path=sent_path,
        long_threshold=0.35,
        short_threshold=0.35,
        max_positions=10,
        start="2022-01-01",
        end="2024-01-01",
    )
    print_summary(metrics)

    # Save equity curve
    metrics["equity_curve"].to_csv("data/equity_curve.csv", header=["equity"])
    logger.info("Equity curve saved to data/equity_curve.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment Alpha Trader")
    parser.add_argument(
        "--mode",
        choices=["live", "dry-run", "backtest"],
        default="dry-run",
        help="Execution mode",
    )
    args = parser.parse_args()

    if args.mode == "backtest":
        run_backtest()
    elif args.mode == "live":
        run_live(dry_run=False)
    else:
        run_live(dry_run=True)
