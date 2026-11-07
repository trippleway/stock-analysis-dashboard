# -*- coding: utf-8 -*-
"""
Daily schedule entry point:
- python run_daily.py --once   run once immediately
- python run_daily.py          enter the schedule loop (09:30 US Eastern every day)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import schedule
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from daily_picker import run_daily_pipeline  # noqa: E402

LOG = logging.getLogger("run_daily")
if not LOG.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

TZ_ET = ZoneInfo("US/Eastern")


def _run_once_job() -> None:
    """Wrap one job run so a failure does not stop the scheduler."""
    try:
        LOG.info("Starting the daily pick job (includes Prophet; 50 names can take about 5-10 minutes)")
        run_daily_pipeline(force_refresh=False)
        LOG.info("Daily pick job finished")
    except Exception as e:  # noqa: BLE001
        LOG.exception("Daily pick job failed: %s", e)


def _schedule_loop() -> None:
    """Schedule loop for 09:30 US Eastern every day."""
    schedule.every().day.at("09:30", "US/Eastern").do(_run_once_job)
    LOG.info("Schedule started: runs every day at 09:30 US Eastern.")
    while True:
        schedule.run_pending()
        time.sleep(30)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Daily stock-pick scheduler")
    parser.add_argument("--once", action="store_true", help="Run once immediately and then exit")
    args = parser.parse_args()

    now_et = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    LOG.info("Current local time: %s", now_et)
    LOG.info("Note: the first full run of 50 names with Prophet can take about 5-10 minutes. The cache speeds up later runs.")

    if args.once:
        _run_once_job()
        return
    _schedule_loop()


if __name__ == "__main__":
    main()
