# -*- coding: utf-8 -*-
"""
檢查組 A 月度基準對起始月敏不敏感，以及報酬是否集中在少數月份。

滾動測試以 config.BACKTEST_FIRST_REBALANCE_MONTH 為中心，往前、往後各 12 個月，
共 25 個起始月。結束月固定是 config.BACKTEST_LAST_MONTH。
選股、持有、成本都沿用 run_group_backtest，這裡不改規則。

月報酬集中度只看目前這段基準：36 個持有月各自複利，再拿掉報酬最高的月份。
SPY 用同一套持有區間，拿掉的也是組 A 報酬最高的那些月份。

另外用同一段固定區間、同一套換手成本，做四個對照：
等權持有全部 34 檔、從合格股票裡隨機抽 3 檔重複 1000 次、
月底分數與下個月報酬的 Spearman 排名相關，以及前後各 18 個月。

執行方式（於專案根目錄）：
    python src/backtest_robustness.py
"""

from __future__ import annotations

import logging
import math
import sys
from datetime import date, timedelta
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

from backtest_compare_groups import (  # noqa: E402
    _BENCHMARK,
    _ONE_WAY_COST,
    _TOP_N,
    _WARMUP_DAYS,
    _as_timestamp_index,
    _cumulative,
    _month_end_signals,
    _net_returns,
    _schedule_trade_costs,
    _score_frame,
    _sharpe,
    _target_weights,
    define_three_groups,
    run_group_backtest,
)
from backtest_data import build_ohlcv_panel, load_ohlcv  # noqa: E402
from config import (  # noqa: E402
    BACKTEST_END,
    BACKTEST_FIRST_REBALANCE_MONTH,
    BACKTEST_LAST_MONTH,
    BACKTEST_START,
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
_RANDOM_REPS = 1000
_RANDOM_SEED = 42
_HALF_MONTHS = 18


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
                "hold_start": str(month["hold_start"]),
                "hold_end": str(month["hold_end"]),
                "picked": list(month.get("picked") or []),
                "return": month_return,
            }
        )
    return rows


def _compound_except(rows: list[dict[str, Any]], excluded: set[str]) -> float:
    """拿掉指定換股月之後，其餘月份複利。excluded 為空就是全部。"""
    kept = [float(row["return"]) for row in rows if row["signal"] not in excluded]
    if not kept or any(not math.isfinite(value) for value in kept):
        return float("nan")
    return float(math.prod(1.0 + value for value in kept) - 1.0)


def _stock_closes(tickers: list[str]) -> dict[str, pd.Series]:
    """
    讀取這些代號在基準區間的收盤價。

    多留基準起日以前的快取，持有首日的漲跌才有前一日收盤可除。
    單一代號沒有價量時略過，呼叫端把該檔標成資料不足。

    參數:
        tickers: 要查的代號。

    回傳:
        代號對到收盤價。索引是交易日。
    """
    closes: dict[str, pd.Series] = {}
    for ticker in sorted(set(tickers)):
        try:
            frame = load_ohlcv(ticker, BACKTEST_START, BACKTEST_END)
        except Exception as exc:  # noqa: BLE001 — 單檔失敗不中斷其餘代號
            _log.warning("%s 價量讀取失敗: %s", ticker, exc)
            continue
        if frame is None or frame.empty or "Close" not in frame.columns:
            _log.warning("%s 沒有收盤價，當月報酬標成資料不足", ticker)
            continue
        close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
        stamps = pd.to_datetime(close.index)
        if getattr(stamps, "tz", None) is not None:
            stamps = stamps.tz_localize(None)
        close.index = pd.DatetimeIndex(stamps).normalize()
        close = close[~close.index.duplicated(keep="last")].sort_index()
        closes[ticker] = close
    return closes


def _price_return(close: pd.Series, hold_start: str, hold_end: str) -> float:
    """
    一檔股票在持有區間的價格報酬。

    用持有首日到持有末日的日報酬複利，含持有首日相對前一日收盤的漲跌。
    這是價格報酬，不含換股成本。

    參數:
        close: 該檔收盤價。
        hold_start: 持有首日。
        hold_end: 持有末日。

    回傳:
        區間價格報酬。沒有可用收盤價時為 NaN。
    """
    daily = close.pct_change().loc[pd.Timestamp(hold_start) : pd.Timestamp(hold_end)]
    if daily.empty or daily.isna().all():
        return float("nan")
    return float((1.0 + daily.fillna(0.0)).prod() - 1.0)


def _print_top_holdings(ranked: list[dict[str, Any]], n_best: int = 5) -> None:
    """
    印出報酬最高的幾個月各自選了哪 3 檔，以及這批持股出現幾次。

    參數:
        ranked: 組 A 月報酬由高到低的列，需含 picked 與持有區間。
        n_best: 要展開的月份數。
    """
    chosen = ranked[:n_best]
    print(f"\n報酬最高的 {len(chosen)} 個月：當月選股與個股價格報酬", flush=True)
    counts: dict[str, int] = {}
    closes = _stock_closes([ticker for row in chosen for ticker in row["picked"]])
    for row in chosen:
        print(
            f"{row['signal']}  持有 {row['period']}  組 A 月報酬 {_pct(row['return'])}",
            flush=True,
        )
        if not row["picked"]:
            print("  這個月沒有選出股票", flush=True)
            continue
        for ticker in row["picked"]:
            counts[ticker] = counts.get(ticker, 0) + 1
            close = closes.get(ticker)
            stock_return = _price_return(close, row["hold_start"], row["hold_end"]) if close is not None else float("nan")
            label = _pct(stock_return) if math.isfinite(stock_return) else "資料不足"
            print(f"  {ticker:<6}{label}", flush=True)
    print(f"\n這 {sum(counts.values())} 個持股出現次數", flush=True)
    if not counts:
        print("沒有持股可統計。", flush=True)
        return
    for ticker, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {ticker:<6}{count} 次", flush=True)


def _print_concentration(result: dict[str, Any]) -> None:
    """印出基準 36 個月的報酬排序，以及拿掉最好幾個月之後的累積報酬。"""
    strategy = _monthly_returns(result, "net_returns")
    spy = _monthly_returns(result, "spy_returns")
    print(
        f"\n=== 月報酬集中度（{BACKTEST_FIRST_REBALANCE_MONTH} 至 {BACKTEST_LAST_MONTH}，扣成本）===",
        flush=True,
    )
    print(
        "一個月是一次換股後的持有區間。組 A 拿掉自己報酬最高的月份，"
        "SPY 拿掉的是同一段期間，不是 SPY 自己報酬最高的月份。",
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

    product = _compound_except(strategy, set())
    reported = float(result["cumulative_return_net"])
    if math.isfinite(product) and abs(product - reported) > 1e-8:
        print(
            f"36 個月複利是 {_pct(product)}，全程扣成本累積是 {_pct(reported)}，兩者沒有對上。",
            flush=True,
        )

    print("\n拿掉組 A 報酬最高的月份之後（SPY 排除同一段期間）", flush=True)
    summary = f"{'範圍':<16}{'組 A 累積':>12}{'SPY 累積':>12}"
    print(summary, flush=True)
    print("-" * len(summary), flush=True)
    strategy_order = [row["signal"] for row in ranked]
    for drop in _DROP_BEST:
        excluded = set(strategy_order[:drop])
        label = "全部" if drop == 0 else f"拿掉最好 {drop} 個月"
        spy_kept = _compound_except(spy, excluded)
        print(
            f"{label:<16}"
            f"{_pct(_compound_except(strategy, excluded)):>12}"
            f"{_pct(spy_kept):>12}",
            flush=True,
        )
        if drop:
            print(f"  同一段期間：{', '.join(strategy_order[:drop])}", flush=True)
            if drop == 5:
                print(f"  修正後的 SPY 累積報酬：{_pct(spy_kept)}", flush=True)
    _print_top_holdings(ranked, n_best=5)


def _load_pool(tickers: list[str]) -> dict[str, Any]:
    """
    用跟月度回測相同的暖機與月底，準備收盤價、分數和持有區間。

    參數:
        tickers: 組 A 代號。

    回傳:
        含 close、score、ready、持有區間與每月合格名單的字典。
    """
    fetch_start = BACKTEST_START - timedelta(days=_WARMUP_DAYS)
    panel = build_ohlcv_panel([*tickers, _BENCHMARK], fetch_start, BACKTEST_END)
    spy = panel.get(_BENCHMARK)
    if spy is None or spy.empty:
        raise RuntimeError("沒有 SPY 價量，無法做股票池對照")
    master = _as_timestamp_index(spy.index)
    master = master[(master >= pd.Timestamp(fetch_start)) & (master <= pd.Timestamp(BACKTEST_END))]
    stock_panel = {ticker: frame for ticker, frame in panel.items() if ticker != _BENCHMARK}
    close, score, ready = _score_frame(stock_panel, master)
    signals = _month_end_signals(master, BACKTEST_START, BACKTEST_END)
    windows: list[dict[str, Any]] = []
    for pos, signal_day in enumerate(signals):
        if signal_day not in close.index:
            continue
        nxt = signals[pos + 1] if pos + 1 < len(signals) else close.index[-1]
        hold = close.index[(close.index > signal_day) & (close.index <= nxt)]
        if len(hold) == 0:
            continue
        tradable = (
            score.loc[signal_day].notna()
            & close.loc[signal_day].reindex(score.columns).notna()
            & ready.loc[signal_day].fillna(False).astype(bool)
        )
        eligible = sorted(score.loc[signal_day].loc[tradable].dropna().index.astype(str).tolist())
        windows.append(
            {
                "signal": signal_day,
                "signal_month": signal_day.date().isoformat()[:7],
                "hold": hold,
                "hold_start": pd.Timestamp(hold[0]).date().isoformat(),
                "hold_end": pd.Timestamp(hold[-1]).date().isoformat(),
                "eligible": eligible,
            }
        )
    return {"close": close, "score": score, "windows": windows, "tickers": list(close.columns)}


def _simulate_paths(close: pd.DataFrame, windows: list[dict[str, Any]], paths: list[list[list[str]]]) -> np.ndarray:
    """
    把多條選股路徑換成扣成本後的累積報酬。

    成本沿用月度回測：單邊費率乘上權重變動的絕對值合計。
    第一個月從現金買進，成本落在持有首日；之後的調倉成本落在訊號日。
    當天沒有成交的持股，權重分給其餘仍持有的名字。

    參數:
        close: 收盤價。
        windows: 每個持有月的區間與訊號日。
        paths: 每條路徑是一串名單，順序與 windows 相同。

    回傳:
        每條路徑的累積報酬。
    """
    if not paths or not windows:
        return np.array([], dtype=float)
    daily = close.pct_change()
    columns = {str(name): pos for pos, name in enumerate(close.columns)}
    values = daily.to_numpy(dtype=float)
    n_paths = len(paths)
    n_names = values.shape[1]
    wealth = np.ones(n_paths, dtype=float)
    previous = np.zeros((n_paths, n_names), dtype=float)
    for month_pos, window in enumerate(windows):
        hold_pos = close.index.get_indexer(window["hold"])
        if np.any(hold_pos < 0):
            raise RuntimeError(f"{window['signal_month']} 的持有日對不上收盤價")
        block = values[hold_pos]  # (天數, 股票)
        gross = np.full((len(hold_pos), n_paths), np.nan, dtype=float)
        current = np.zeros((n_paths, n_names), dtype=float)
        for path_pos, picks in enumerate(path[month_pos] for path in paths):
            cols = [columns[name] for name in picks if name in columns]
            if not cols:
                continue
            weight = 1.0 / len(cols)
            current[path_pos, cols] = weight
            taken = block[:, cols]
            present = np.isfinite(taken)
            filled = np.where(present, taken, 0.0)
            counts = present.sum(axis=1)
            month_gross = np.full(len(hold_pos), np.nan, dtype=float)
            usable = counts > 0
            month_gross[usable] = filled[usable].sum(axis=1) / counts[usable]
            gross[:, path_pos] = month_gross
        traded = np.abs(current - previous).sum(axis=1)
        cost = _ONE_WAY_COST * traded
        # 第一個月成本在持有首日；之後在訊號日，也就是上一段的最後一天。
        if month_pos == 0:
            gross[0, :] = (1.0 - cost) * (1.0 + np.nan_to_num(gross[0, :], nan=0.0)) - 1.0
        else:
            wealth *= 1.0 - cost
        finite = np.isfinite(gross)
        growth = np.ones_like(gross)
        growth[finite] = 1.0 + gross[finite]
        wealth *= growth.prod(axis=0)
        previous = current
    return wealth - 1.0


def _strategy_picks(baseline: dict[str, Any], windows: list[dict[str, Any]]) -> list[list[str]]:
    """依基準回測的換股月，取出每個月實際選到的代號。"""
    by_month = {str(month["signal"])[:7]: list(month["picked"]) for month in baseline["months"]}
    missing = [window["signal_month"] for window in windows if window["signal_month"] not in by_month]
    if missing:
        raise RuntimeError("基準選股對不上持有區間：" + ", ".join(missing))
    return [by_month[window["signal_month"]] for window in windows]


def _print_equal_and_random(
    close: pd.DataFrame,
    windows: list[dict[str, Any]],
    strategy_return: float,
    strategy_picks: list[list[str]],
) -> tuple[pd.Series, list[list[str]]]:
    """
    印出等權基準與 1000 次隨機抽 3 檔的分布。

    參數:
        close: 收盤價。
        windows: 持有區間，含每月合格名單。
        strategy_return: 策略扣成本後的累積報酬。沿用正式回測，不在這裡重算選股。
        strategy_picks: 策略每個月的持股，用來核對成本計算是否與正式回測一致。

    回傳:
        (等權的扣成本日報酬, 等權每月持股)。日報酬供前後半段使用。
    """
    equal_picks = [list(window["eligible"]) for window in windows]
    counts = [len(names) for names in equal_picks]
    print("\n=== 等權基準（同一批股票，每月底全部等權，扣同樣的換手成本）===", flush=True)
    print(
        "合格條件與策略相同：當天有技術面分數、有收盤價，而且已滿最短歷史。"
        f"每月合格檔數 {min(counts)}–{max(counts)}。",
        flush=True,
    )
    thin = [window["signal_month"] for window, names in zip(windows, equal_picks) if len(names) < len(close.columns)]
    if thin:
        print("這些月份沒有把全部代號都算進等權：" + ", ".join(thin), flush=True)
    checked = float(_simulate_paths(close, windows, [strategy_picks])[0])
    if abs(checked - strategy_return) > 1e-6:
        _log.warning(
            "同一套成本重算策略得到 %.4f%%，正式回測是 %.4f%%",
            checked * 100,
            strategy_return * 100,
        )
    equal_return = float(_simulate_paths(close, windows, [equal_picks])[0])
    print(
        f"策略累積 {_pct(strategy_return)}，等權累積 {_pct(equal_return)}，"
        f"差距 {_pct(strategy_return - equal_return)}。",
        flush=True,
    )

    print(
        f"\n=== 隨機基準（每月從合格股票抽 {_TOP_N} 檔，{_RANDOM_REPS} 次，seed={_RANDOM_SEED}）===",
        flush=True,
    )
    short = [window["signal_month"] for window in windows if len(window["eligible"]) < _TOP_N]
    if short:
        print("這些月份合格股票少於 3 檔，隨機基準改持有全部合格股票：" + ", ".join(short), flush=True)
    rng = np.random.default_rng(_RANDOM_SEED)
    random_paths: list[list[list[str]]] = []
    for _rep in range(_RANDOM_REPS):
        path: list[list[str]] = []
        for window in windows:
            eligible = window["eligible"]
            draw = min(_TOP_N, len(eligible))
            if draw == 0:
                path.append([])
                continue
            picked = rng.choice(eligible, size=draw, replace=False)
            path.append(sorted(str(name) for name in picked))
        random_paths.append(path)
    random_returns = _simulate_paths(close, windows, random_paths)
    print("累積報酬百分位：", flush=True)
    for pct in _PERCENTILES:
        print(f"  {pct:>2} 分位  {_pct(float(np.percentile(random_returns, pct)))}", flush=True)
    rank = _percentile_rank(random_returns.tolist(), strategy_return)
    print(
        f"策略 {_pct(strategy_return)} 落在第 {_num(rank)} 百分位"
        f"（{_RANDOM_REPS} 次裡，累積報酬不高於策略的比例）。",
        flush=True,
    )
    equal_daily = _path_daily_returns(close, windows, equal_picks)
    return equal_daily, equal_picks


def _path_daily_returns(
    close: pd.DataFrame,
    windows: list[dict[str, Any]],
    picks: list[list[str]],
) -> pd.Series:
    """
    單一路徑的扣成本日報酬。演算法與 _simulate_paths 相同，保留日期索引。

    參數:
        close: 收盤價。
        windows: 持有區間。
        picks: 每個持有月的名單。

    回傳:
        扣成本後的日報酬。
    """
    cumulative = _simulate_paths(close, windows, [picks])
    daily_ret = close.pct_change()
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    books: list[dict[str, Any]] = []
    for window, names in zip(windows, picks):
        if names:
            weights.loc[window["hold"], names] = 1.0 / len(names)
        cost_day = window["hold"][0] if not books else window["signal"]
        books.append({"first_hold_day": cost_day, "weights": _target_weights(names)})
    valid = daily_ret.notna() & weights.gt(0)
    effective = weights.where(valid, 0.0)
    scale = effective.sum(axis=1).replace(0.0, np.nan)
    renormalized = effective.div(scale, axis=0).fillna(0.0)
    portfolio = (renormalized * daily_ret.fillna(0.0)).sum(axis=1)
    invested = effective.sum(axis=1).gt(0)
    gross = portfolio.where(invested, np.nan).dropna()
    cost_on_day, _stats = _schedule_trade_costs(books)
    costs = pd.Series(0.0, index=gross.index, dtype=float)
    for day, cost in cost_on_day.items():
        stamp = pd.Timestamp(day)
        if stamp in costs.index:
            costs.loc[stamp] = cost
    net = _net_returns(gross, costs)
    # 兩種寫法都在，用來確認日報酬複利和批次累積是同一條路徑。
    if abs(float(_cumulative(net)) - float(cumulative[0])) > 1e-6:
        _log.warning("等權日報酬複利與批次累積不一致")
    return net


def _holding_price_returns(close: pd.DataFrame, hold: pd.DatetimeIndex) -> pd.Series:
    """每檔股票在這段持有期的價格報酬。沒有成交的代號是 NaN。"""
    daily = close.pct_change().loc[hold]
    returns: dict[str, float] = {}
    for ticker in close.columns:
        window = daily[ticker].dropna()
        if window.empty:
            returns[str(ticker)] = float("nan")
        else:
            returns[str(ticker)] = float((1.0 + window).prod() - 1.0)
    return pd.Series(returns, dtype=float)


def _spearman(scores: pd.Series, returns: pd.Series) -> tuple[float, int, list[str]]:
    """
    月底分數與該月持有期報酬的 Spearman 相關。

    參數:
        scores: 訊號日的技術面分數。
        returns: 同一批股票的持有期價格報酬。

    回傳:
        (相關係數, 有配對的檔數, 沒有配對的代號)。檔數少於 3 時相關係數為 NaN。
    """
    paired = pd.DataFrame({"score": scores, "ret": returns}).dropna()
    universe = sorted(set(scores.index.astype(str)) | set(returns.index.astype(str)))
    missing = [name for name in universe if name not in set(paired.index.astype(str))]
    if len(paired) < 3:
        return float("nan"), int(len(paired)), missing
    # 名次的 Pearson 相關就是 Spearman。并列名次用平均名次。
    ranked = paired.rank(method="average")
    return float(ranked["score"].corr(ranked["ret"])), int(len(paired)), missing


def _print_rank_ic(score: pd.DataFrame, close: pd.DataFrame, windows: list[dict[str, Any]]) -> None:
    """印出每個月的 Rank IC，以及平均、標準差、正值比例和 t 統計量。"""
    print("\n=== Rank IC（月底技術面分數 vs 該月持有期報酬，Spearman）===", flush=True)
    print("用當月有分數也有報酬的股票，不是只看選中的 3 檔。", flush=True)
    header = f"{'換股月':<10}{'檔數':>6}{'IC':>8}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    correlations: list[float] = []
    for window in windows:
        scores = score.loc[window["signal"]]
        returns = _holding_price_returns(close, window["hold"])
        ic, used, missing = _spearman(scores, returns)
        print(f"{window['signal_month']:<10}{used:>6}{_num(ic):>8}", flush=True)
        if missing:
            print(f"           未納入：{', '.join(missing)}", flush=True)
        if math.isfinite(ic):
            correlations.append(ic)
    if len(correlations) < 2:
        print("有效月份少於 2，無法計算 IC 標準差和 t 統計量。", flush=True)
        return
    values = np.array(correlations, dtype=float)
    mean_ic = float(values.mean())
    std_ic = float(values.std(ddof=1))
    stderr = std_ic / math.sqrt(len(values))
    t_stat = mean_ic / stderr if stderr > 0 else float("nan")
    positive = float(np.mean(values > 0))
    print(
        f"平均 IC {_num(mean_ic)}，標準差 {_num(std_ic)}，"
        f"IC > 0 的月份 {_pct(positive)}（{int(np.sum(values > 0))}/{len(values)}），"
        f"t 統計量 {_num(t_stat)}（平均 IC / 標準誤）。",
        flush=True,
    )


def _slice_half(daily: pd.Series, windows: list[dict[str, Any]]) -> pd.Series:
    """把日報酬依持有區間接成一段。區間沒有重疊。"""
    parts = [daily.loc[pd.Timestamp(window["hold_start"]) : pd.Timestamp(window["hold_end"])] for window in windows]
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts)


def _print_halves(
    baseline: dict[str, Any],
    windows: list[dict[str, Any]],
    equal_daily: pd.Series,
) -> None:
    """把 36 個月切成前後各 18 個月，比較策略、等權和 SPY。"""
    print("\n=== 前後各 18 個月（扣成本）===", flush=True)
    ordered = sorted(windows, key=lambda window: window["hold_start"])
    if len(ordered) != _HALF_MONTHS * 2:
        print(f"持有月共 {len(ordered)} 個，不是 36，前後半段仍照時間對半切。", flush=True)
    first = ordered[:_HALF_MONTHS]
    second = ordered[_HALF_MONTHS : _HALF_MONTHS * 2]
    strategy = baseline["net_returns"]
    spy = baseline["spy_returns"]
    header = f"{'區段':<28}{'策略累積':>10}{'策略 Sharpe':>12}{'等權累積':>10}{'等權 Sharpe':>12}{'SPY 累積':>10}{'SPY Sharpe':>12}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for title, group in ((f"{first[0]['hold_start'][:7]} 至 {first[-1]['hold_end'][:7]}", first), (f"{second[0]['hold_start'][:7]} 至 {second[-1]['hold_end'][:7]}", second)):
        strategy_part = _slice_half(strategy, group)
        equal_part = _slice_half(equal_daily, group)
        spy_part = _slice_half(spy, group)
        print(
            f"{title:<28}"
            f"{_pct(_cumulative(strategy_part)):>10}{_num(_sharpe(strategy_part)):>12}"
            f"{_pct(_cumulative(equal_part)):>10}{_num(_sharpe(equal_part)):>12}"
            f"{_pct(_cumulative(spy_part)):>10}{_num(_sharpe(spy_part)):>12}",
            flush=True,
        )


def run_pool_benchmarks(tickers: list[str], baseline: dict[str, Any]) -> None:
    """
    用固定區間比較等權、隨機、Rank IC 和前後半段。

    參數:
        tickers: 組 A 代號。
        baseline: 目前基準的月度回測。策略報酬用這一條，不另外改選股規則。
    """
    pool = _load_pool(tickers)
    windows = pool["windows"]
    if len(windows) != len(baseline["months"]):
        _log.warning(
            "股票池持有月 %s 段，基準回測 %s 段",
            len(windows),
            len(baseline["months"]),
        )
    picks = _strategy_picks(baseline, windows)
    equal_daily, _equal_picks = _print_equal_and_random(
        pool["close"],
        windows,
        float(baseline["cumulative_return_net"]),
        picks,
    )
    _print_rank_ic(pool["score"], pool["close"], windows)
    _print_halves(baseline, windows, equal_daily)


def main() -> None:
    """跑滾動起始月、月報酬集中度，以及股票池對照。"""
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
    run_pool_benchmarks(tickers, baseline)


if __name__ == "__main__":
    main()
