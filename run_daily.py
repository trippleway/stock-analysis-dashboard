# -*- coding: utf-8 -*-
"""
每日排程入口：
- python run_daily.py --once   立即執行一次
- python run_daily.py          進入排程循環（每日美東 09:30）
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
    """封裝一次任務執行，避免排程中斷。"""
    try:
        LOG.info("開始每日選股任務（含 Prophet，50 支可能約 5-10 分鐘）")
        run_daily_pipeline(force_refresh=False)
        LOG.info("每日選股任務完成")
    except Exception as e:  # noqa: BLE001
        LOG.exception("每日選股任務失敗：%s", e)


def _schedule_loop() -> None:
    """每日美東 09:30 排程循環。"""
    schedule.every().day.at("09:30", "US/Eastern").do(_run_once_job)
    LOG.info("排程已啟動：每天美東 09:30 執行。")
    while True:
        schedule.run_pending()
        time.sleep(30)


def main() -> None:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="每日選股排程器")
    parser.add_argument("--once", action="store_true", help="立即執行一次後結束")
    args = parser.parse_args()

    now_et = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    LOG.info("目前時間（本機）：%s", now_et)
    LOG.info("提示：首次完整跑 50 支含 Prophet 可能約 5-10 分鐘，快取可加速後續執行。")

    if args.once:
        _run_once_job()
        return
    _schedule_loop()


if __name__ == "__main__":
    main()
