# -*- coding: utf-8 -*-
"""
檢查組 A 月度基準對起始月敏不敏感，以及報酬是否集中在少數月份。

滾動測試以 config.BACKTEST_FIRST_REBALANCE_MONTH 為中心，往前、往後各 12 個月，
共 25 個起始月。結束月固定是 config.BACKTEST_LAST_MONTH。
選股、持有、成本都沿用 run_group_backtest，這裡不改規則。

月報酬集中度只看目前這段基準：36 個持有月各自複利，再拿掉報酬最高的月份。
SPY 用同一套持有區間，但是拿掉的是 SPY 自己最高的月份。

執行方式（於專案根目錄）：
    python src/backtest_robustness.py
"""

from __future__ import annotations

import logging
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_SRC = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from backtest_compare_groups import define_three_groups, run_group_backtest  # noqa: E402
from config import (  # noqa: E402
    BACKTEST_END,
    BACKTEST_FIRST_REBALANCE_MONTH,
    BACKTEST_LAST_MONTH,
)

_log = logging.getLogger("backtest_robustness")
if not _log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# 中心起始月的前後各 12 個月，含中心自己，一共 25 個。
_ROLL_RADIUS = 12
_PERCENTILES = (5, 25, 50, 75, 95)
_DROP_BEST = (0, 1, 3, 5)
_DAYS_PER_YEAR = 365.25


def _parse_year_month(token: str) -> tuple[int, int]:
    """把 YYYY-MM 拆成年、月。月份超出 1–12 就拒絕。"""
    year_text, month_text = str(token).split("-", 1)
    year, month = int(year_text), int(month_text)
    if month < 1 or month > 12:
        raise ValueError(f"月份必須介於 1 到 12，收到 {token}")
    return year, month


def _shift_month(token: str, offset: int) -> str:
    """把 YYYY-MM 平移 offset 個月。負數是往前。"""
    year, month = _parse_year_month(token)
    index = year * 12 + (month - 1) + offset
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def _month_span(start_month: str, end_month: str) -> int:
    """兩個 YYYY-MM 相差幾個月。同一個月是 0。"""
    start_year, start = _parse_year_month(start_month)
    end_year, end = _parse_year_month(end_month)
    return (end_year * 12 + end) - (start_year * 12 + start)


def _month_first_day(token: str) -> date:
    """該月 1 日。回測用它當日曆邊界，換股日仍是月底交易日。"""
    year, month = _parse_year_month(token)
    return date(year, month, 1)


def _annualized(cumulative: float, start: date, end: date) -> float:
    """
    把一段累積報酬換成年化複利。

    年數用持有區間的日曆天除以 365.25。長度不同的起始月才能放在一起比。

    參數:
        cumulative: 這一段的累積報酬。
        start: 第一個持有日。
        end: 最後一個持有日。

    回傳:
        年化報酬。天數不是正數，或累積報酬不是有限值時為 NaN。
    """
    days = (end - start).days
    if days <= 0 or not math.isfinite(cumulative) or cumulative <= -1.0:
        return float("nan")
    return float((1.0 + cumulative) ** (_DAYS_PER_YEAR / days) - 1.0)


def _window_return(daily: pd.Series, start: str, end: str) -> float:
    """持有區間的複利報酬。區間含起日與迄日。沒有日報酬時為 NaN。"""
    window = daily.loc[pd.Timestamp(start) : pd.Timestamp(end)]
    if window.empty or window.isna().all():
        return float("nan")
    return float((1.0 + window.fillna(0.0)).prod() - 1.0)


def _pct(value: float) -> str:
    """把比率格式化成百分比。無效值顯示為破折號。"""
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value * 100:.1f}%"


def _num(value: float) -> str:
    """把 Sharpe 或百分位格式化成兩位小數。"""
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:.2f}"


def _percentile_rank(values: list[float], point: float) -> float:
    """
    基準落在這組數字的經驗百分位。

    定義是「不高於基準的比例」。100 代表基準是這組裡最高的，
    50 左右代表有一半的起始月不高於基準。

    參數:
        values: 完整跑完的起始月指標。
        point: 目前基準的同一個指標。

    回傳:
        0 到 100 的百分位。沒有可比樣本時為 NaN。
    """
    usable = [value for value in values if math.isfinite(value)]
    if not usable or not math.isfinite(point):
        return float("nan")
    return float(100.0 * sum(1 for value in usable if value <= point) / len(usable))


def _start_months() -> list[str]:
    """中心起始月往前 12 個月到往後 12 個月，由早到晚。"""
    return [
        _shift_month(BACKTEST_FIRST_REBALANCE_MONTH, offset)
        for offset in range(-_ROLL_RADIUS, _ROLL_RADIUS + 1)
    ]


def _row_from_result(start_month: str, result: dict[str, Any]) -> dict[str, Any]:
    """
    把一次月度回測收成列。起始月或結束月對不上時標成資料不足。

    參數:
        start_month: 這次指定的第一次換股月，YYYY-MM。
        result: run_group_backtest 的回傳。

    回傳:
        含年化報酬、Sharpe、最大回撤、期間月數的字典。資料不足時 ok 為 False。
    """
    months = list(result.get("months") or [])
    expected = _month_span(start_month, BACKTEST_LAST_MONTH)
    reasons: list[str] = []
    if not months:
        reasons.append("沒有任何持有月")
    else:
        first_signal = str(months[0]["signal"])[:7]
        last_hold = str(months[-1]["hold_end"])[:7]
        if first_signal != start_month:
            reasons.append(f"第一個換股日是 {months[0]['signal']}，不是 {start_month}")
        if last_hold != BACKTEST_LAST_MONTH:
            reasons.append(f"最後一個持有日是 {months[-1]['hold_end']}，不是 {BACKTEST_LAST_MONTH}")
    if int(result["n_rebalances"]) != expected:
        reasons.append(f"換股 {int(result['n_rebalances'])} 次，這段跨度應為 {expected} 個月")
    hold_start = date.fromisoformat(str(result["start"]))
    hold_end = date.fromisoformat(str(result["end"]))
    cumulative = float(result["cumulative_return_net"])
    row = {
        "start_month": start_month,
        "ok": not reasons,
        "reason": "；".join(reasons),
        "cumulative": cumulative,
        "annualized": _annualized(cumulative, hold_start, hold_end),
        "sharpe": float(result["sharpe_net"]),
        "max_drawdown": float(result["max_drawdown_net"]),
        "n_months": int(result["n_rebalances"]),
        "expected_months": expected,
        "hold_start": result["start"],
        "hold_end": result["end"],
        "result": result if not reasons else None,
    }
    return row


def run_rolling_starts(tickers: list[str]) -> list[dict[str, Any]]:
    """
    對 25 個起始月各跑一次組 A 月度基準，結束月都是 BACKTEST_LAST_MONTH。

    由早到晚跑。較早的起始月會把價量快取往前補齊，後面的月份就不用再抓同一段。
    某一個月失敗或區間對不上，這一列標成資料不足，其餘月份繼續。

    參數:
        tickers: 組 A 代號。

    回傳:
        25 列，順序與起始月相同。目前基準那一列另外留著完整回測結果。
    """
    rows: list[dict[str, Any]] = []
    for start_month in _start_months():
        start_day = _month_first_day(start_month)
        _log.info("滾動起始月 %s，結束月 %s", start_month, BACKTEST_LAST_MONTH)
        try:
            result = run_group_backtest(tickers, start_date=start_day, end_date=BACKTEST_END)
        except Exception as exc:  # noqa: BLE001 — 單月失敗要留在表上
            _log.warning("起始月 %s 資料不足: %s", start_month, exc)
            rows.append(
                {
                    "start_month": start_month,
                    "ok": False,
                    "reason": str(exc),
                    "cumulative": float("nan"),
                    "annualized": float("nan"),
                    "sharpe": float("nan"),
                    "max_drawdown": float("nan"),
                    "n_months": _month_span(start_month, BACKTEST_LAST_MONTH),
                    "expected_months": _month_span(start_month, BACKTEST_LAST_MONTH),
                    "hold_start": "",
                    "hold_end": "",
                    "result": None,
                }
            )
            continue
        row = _row_from_result(start_month, result)
        if row["ok"] and start_month == BACKTEST_FIRST_REBALANCE_MONTH:
            row["result"] = result
        elif not row["ok"]:
            _log.warning("起始月 %s 資料不足: %s", start_month, row["reason"])
            row["result"] = None
        else:
            row["result"] = None
        rows.append(row)
    return rows


def _print_rolling(rows: list[dict[str, Any]]) -> None:
    """印出 25 個起始月，以及年化報酬、Sharpe 的百分位。"""
    print(
        f"\n=== 滾動起始月（結束月固定 {BACKTEST_LAST_MONTH}，組 A 月度基準，扣成本）===",
        flush=True,
    )
    print(
        f"中心是 {BACKTEST_FIRST_REBALANCE_MONTH}，往前與往後各 {_ROLL_RADIUS} 個月，共 {len(rows)} 個起始月。",
        flush=True,
    )
    header = (
        f"{'起始月':<10}{'期間月數':>8}{'累積報酬':>10}{'年化報酬':>10}"
        f"{'Sharpe':>8}{'最大回撤':>10}  狀態"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for row in rows:
        mark = "基準" if row["start_month"] == BACKTEST_FIRST_REBALANCE_MONTH else ""
        status = "資料不足" if not row["ok"] else mark
        print(
            f"{row['start_month']:<10}"
            f"{int(row['n_months']):>8}"
            f"{_pct(row['cumulative']):>10}"
            f"{_pct(row['annualized']):>10}"
            f"{_num(row['sharpe']):>8}"
            f"{_pct(row['max_drawdown']):>10}  {status}",
            flush=True,
        )
        if not row["ok"]:
            print(f"           原因：{row['reason']}", flush=True)

    usable = [row for row in rows if row["ok"]]
    print(
        f"\n百分位只用完整跑完的 {len(usable)} 個起始月。資料不足的列留在上面，沒有拿來算分布。",
        flush=True,
    )
    if not usable:
        print("沒有任何一個起始月跑完，無法計算百分位。", flush=True)
        return
    baseline = next(
        (row for row in usable if row["start_month"] == BACKTEST_FIRST_REBALANCE_MONTH),
        None,
    )
    for key, title in (("annualized", "年化報酬"), ("sharpe", "Sharpe")):
        values = [float(row[key]) for row in usable]
        print(f"{title} 百分位：", flush=True)
        for pct in _PERCENTILES:
            level = float(np.percentile(values, pct))
            shown = _pct(level) if key == "annualized" else _num(level)
            print(f"  {pct:>2} 分位  {shown}", flush=True)
        if baseline is None:
            print(f"  目前基準 {BACKTEST_FIRST_REBALANCE_MONTH} 沒有完整結果，不能排百分位。", flush=True)
            continue
        rank = _percentile_rank(values, float(baseline[key]))
        shown = _pct(float(baseline[key])) if key == "annualized" else _num(float(baseline[key]))
        print(
            f"  目前基準 {baseline['start_month']} 的{title}是 {shown}，"
            f"落在第 {_num(rank)} 百分位"
            f"（{len(usable)} 個完整起始月裡，不高於基準的比例）。",
            flush=True,
        )


def _monthly_returns(result: dict[str, Any], field: str) -> list[dict[str, Any]]:
    """
    把每個持有月的日報酬複利成一個月報酬。

    參數:
        result: 目前基準的回測結果，需含 months 與日報酬。
        field: net_returns 或 spy_returns。

    回傳:
        每個月一列，含換股月、持有區間與月報酬。區間沒有日報酬時月報酬為 NaN。
    """
    daily = result[field]
    rows: list[dict[str, Any]] = []
    for month in result["months"]:
        month_return = _window_return(daily, str(month["hold_start"]), str(month["hold_end"]))
        rows.append(
            {
                "signal": str(month["signal"])[:7],
                "period": f"{month['hold_start']} ~ {month['hold_end']}",
                "return": month_return,
            }
        )
    return rows


def _compound_without_best(returns: list[float], drop: int) -> float:
    """拿掉最高的 drop 個月之後，其餘月份複利。drop 為 0 就是全部。"""
    ordered = sorted(returns, reverse=True)
    kept = ordered[drop:]
    if not kept or any(not math.isfinite(value) for value in kept):
        return float("nan")
    return float(math.prod(1.0 + value for value in kept) - 1.0)


def _print_concentration(result: dict[str, Any]) -> None:
    """印出基準 36 個月的報酬排序，以及拿掉最好幾個月之後的累積報酬。"""
    strategy = _monthly_returns(result, "net_returns")
    spy = _monthly_returns(result, "spy_returns")
    print(
        f"\n=== 月報酬集中度（{BACKTEST_FIRST_REBALANCE_MONTH} 至 {BACKTEST_LAST_MONTH}，扣成本）===",
        flush=True,
    )
    print(
        "一個月是一次換股後的持有區間。組 A 拿掉的是組 A 報酬最高的月份；"
        "SPY 拿掉的是 SPY 自己報酬最高的月份。",
        flush=True,
    )
    missing = [row["signal"] for row in strategy + spy if not math.isfinite(row["return"])]
    if missing:
        print("這些月份沒有日報酬，集中度不算完整：" + ", ".join(missing), flush=True)

    ranked = sorted(strategy, key=lambda row: row["return"], reverse=True)
    print(f"\n組 A 月報酬由高到低（{len(ranked)} 個月）", flush=True)
    header = f"{'名次':>4}  {'換股月':<10}{'持有期間':<27}{'月報酬':>10}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for rank, row in enumerate(ranked, start=1):
        print(
            f"{rank:>4}  {row['signal']:<10}{row['period']:<27}{_pct(row['return']):>10}",
            flush=True,
        )

    strategy_returns = [float(row["return"]) for row in strategy]
    spy_returns = [float(row["return"]) for row in spy]
    product = _compound_without_best(strategy_returns, 0)
    reported = float(result["cumulative_return_net"])
    if math.isfinite(product) and abs(product - reported) > 1e-8:
        print(
            f"36 個月複利是 {_pct(product)}，全程扣成本累積是 {_pct(reported)}，兩者沒有對上。",
            flush=True,
        )

    print("\n拿掉報酬最高的月份之後", flush=True)
    summary = f"{'範圍':<16}{'組 A 累積':>12}{'SPY 累積':>12}"
    print(summary, flush=True)
    print("-" * len(summary), flush=True)
    strategy_order = [row["signal"] for row in ranked]
    spy_order = [row["signal"] for row in sorted(spy, key=lambda item: item["return"], reverse=True)]
    for drop in _DROP_BEST:
        label = "全部" if drop == 0 else f"拿掉最好 {drop} 個月"
        print(
            f"{label:<16}"
            f"{_pct(_compound_without_best(strategy_returns, drop)):>12}"
            f"{_pct(_compound_without_best(spy_returns, drop)):>12}",
            flush=True,
        )
        if drop:
            print(f"  組 A 拿掉：{', '.join(strategy_order[:drop])}", flush=True)
            print(f"  SPY 拿掉：{', '.join(spy_order[:drop])}", flush=True)


def main() -> None:
    """跑滾動起始月與月報酬集中度，並印出結果。"""
    # 回測每個月份都會記一筆選股，25 次會把表格淹沒。失敗仍由本模組記 warning。
    logging.getLogger("backtest_compare_groups").setLevel(logging.WARNING)
    logging.getLogger("backtest_data").setLevel(logging.WARNING)
    logging.getLogger("backtest_universe_sp500").setLevel(logging.WARNING)
    groups = define_three_groups()
    tickers = groups["A"]
    print(f"組 A（{len(tickers)} 檔）", flush=True)
    rows = run_rolling_starts(tickers)
    _print_rolling(rows)
    baseline = next(
        (
            row["result"]
            for row in rows
            if row["start_month"] == BACKTEST_FIRST_REBALANCE_MONTH and row.get("result") is not None
        ),
        None,
    )
    if baseline is None:
        print(
            f"\n目前基準 {BACKTEST_FIRST_REBALANCE_MONTH} 沒有完整結果，月報酬集中度沒有計算。",
            flush=True,
        )
        return
    _print_concentration(baseline)


if __name__ == "__main__":
    main()
