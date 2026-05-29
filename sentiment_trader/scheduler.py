"""
scheduler.py
------------
Runs the trading pipeline on a schedule.
Fires at 9:28 AM ET every weekday (2 min before market open).

Run with: python scheduler.py
Or use cron: 28 9 * * 1-5 /path/to/venv/python /path/to/scheduler.py
"""

import schedule
import time
import logging
from datetime import datetime
import pytz

logger = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")


def is_market_day() -> bool:
    """Basic check — skips weekends. Doesn't handle holidays."""
    return datetime.now(ET).weekday() < 5  # Mon–Fri


def daily_job():
    if not is_market_day():
        logger.info("Weekend — skipping.")
        return

    logger.info("=== Daily sentiment pipeline starting ===")
    try:
        from main import run_live
        run_live(dry_run=False)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)


# Schedule for 9:28 AM ET
schedule.every().day.at("09:28").do(daily_job)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    logger.info("Scheduler started. Waiting for 9:28 AM ET...")
    while True:
        schedule.run_pending()
        time.sleep(30)
