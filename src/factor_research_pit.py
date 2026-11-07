# -*- coding: utf-8 -*-
"""
時點 S&P 500 上的因子 rank IC。只算 IC，不做回測、也不選股。

因子定義、持有區間與 Spearman 算法沿用 factor_research.py。
差別是每個月底的橫斷面：只用 backtest_universe_sp500.get_point_in_time_sp500()
在該月底當天的成分股，不用固定清單，也不用整段聯集。

已下市的代號 yfinance 常常沒有報價。抓不到的股票不會被補進橫斷面，
所以結果偏向還能被抓到的公司。抓不到不等於全部都已下市。

下載幾百檔歷史股價很慢。不加參數時只印預估檔數與時間，不下載。
確認後才抓資料並計算。價量已經在本地時，加上 --cache-only 就不再打 yfinance：

    python src/factor_research_pit.py
    python src/factor_research_pit.py --confirm
    python src/factor_research_pit.py --confirm --cache-only
"""

from __future__ import annotations

import logging
import math
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_SRC = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from backtest_compare_groups import define_three_groups  # noqa: E402
from backtest_data import (  # noqa: E402
    _cache_covers,
    _cache_path,
    _read_cache,
    build_ohlcv_panel,
    load_ohlcv,
)
from backtest_robustness import _holding_price_returns, _spearman  # noqa: E402
from backtest_universe_sp500 import get_point_in_time_sp500  # noqa: E402
from config import BACKTEST_END, BACKTEST_START  # noqa: E402
from factor_research import (  # noqa: E402
    _CALENDAR,
    _FACTORS,
    _WARMUP_DAYS,
    _check_indicator_api,
    _cross_section,
    _fmt_num,
    _fmt_ratio,
    _holding_windows,
    _ic_stats,
    _load_histories,
    _monthly_ic,
    _prepare_ohlcv,
    compute_factors,
    verify_no_lookahead,
)

_log = logging.getLogger("factor_research_pit")
if not _log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# 預估下載時間用的假設，不是實際迴圈的 sleep。實際間隔在 backtest_data，是 0.7 秒。
# 2.5 秒涵蓋這 0.7 秒，再加上一次成功的歷史股價請求。失敗會重試，時間更長。
_ESTIMATE_SECONDS_PER_FETCH = 2.5
# 前視偏差檢查會對每個訊號月截斷重算。以單檔約 0.3 秒估算整段檢查。
_ESTIMATE_SECONDS_PER_STOCK_CHECK = 0.3
_SURVIVOR_BANNER = "已下市股票無法取得，結果偏向倖存者。"
# 下載前已完成下市或被併購、Yahoo 因此沒有這段歷史的代號。
# 只列有成交紀錄的案子，不用來猜測仍在交易的公司。
_ACQUIRED_BEFORE_DOWNLOAD = frozenset(
    {
        "ANSS",  # Synopsys，2025-07
        "ATVI",  # Microsoft，2023-10
        "CDAY",  # 舊代號；發行公司 Dayforce 於 2026-02 被 Thoma Bravo 併購
        "CMA",  # Fifth Third，2026-02
        "CTLT",  # Novo Holdings，2024-12
        "DAY",  # Thoma Bravo，2026-02
        "DFS",  # Capital One，2025-05
        "HES",  # Chevron，2025-07
        "JNPR",  # HPE，2025-07
        "MRO",  # ConocoPhillips，2024-11
        "PXD",  # Exxon，2024-05
        "QRVO",  # Skyworks，2026-10（樣本結束後、這次下載前）
        "WBA",  # Sycamore，2025-08
        "WRK",  # Smurfit Westrock，2024-07
    }
)
# 使用者指出這些代號目前應仍上市，抓不到不當成下市。
_STILL_LISTED_UNCLEAR = ("MMC", "EA", "BK", "K", "EQR", "AVB", "CTRA", "HOLX", "IPG")


def _yahoo_symbol(ticker: str) -> str:
    """
    歷史成分股代號轉成 Yahoo 報價代號。

    S&P 歷史檔使用 BRK.B 這種寫法，yfinance 要的是 BRK-B。
    規則與 backtest_engine_v2._yahoo_symbol 相同。這只改請求用的代號，
    不把股票加進或移出當月成分股。

    參數:
        ticker: 時點名單上的代號。

    回傳:
        用來向 yfinance 要價量的代號。
    """
    return str(ticker).upper().strip().replace(".", "-")


def _fetch_bounds() -> tuple[date, date]:
    """
    價量請求的起迄日。起日含暖機，讓第一個訊號月的 252 日窗口有機會算得出來。

    回傳:
        (請求起日, 請求迄日)。迄日是 config 的回測迄日。
    """
    return BACKTEST_START - timedelta(days=_WARMUP_DAYS), BACKTEST_END


def _print_banner() -> None:
    """研究輸出的第一句。下市股票沒有報價，橫斷面會少掉它們。"""
    print(_SURVIVOR_BANNER, flush=True)


def _signal_windows() -> list[dict[str, Any]]:
    """
    用 SPY 交易日排出與 factor_research 相同的 36 個訊號月。

    SPY 若已在本地快取就不會重新下載。

    回傳:
        每個月一列，含 signal、signal_month、hold。
    """
    fetch_start, fetch_end = _fetch_bounds()
    spy = load_ohlcv(_CALENDAR, fetch_start, fetch_end)
    if spy is None or spy.empty:
        raise RuntimeError("沒有 SPY 價量，無法沿用現有的月底日曆")
    prepared = _prepare_ohlcv(spy)
    windows = _holding_windows(prepared.index)
    if not windows:
        raise RuntimeError("沒有任何訊號月")
    return windows


def _members_by_month(windows: list[dict[str, Any]]) -> dict[str, set[str]]:
    """
    每個訊號月底的 S&P 500 成分股。

    一次取出區間內的每日時點名單，再只留下訊號日那天的集合。
    每日名單來自 get_point_in_time_sp500，用的是小於等於當天的最近一筆完整名單。
    IC 不使用這整段的聯集。

    參數:
        windows: 訊號月。

    回傳:
        訊號月 YYYY-MM 對到該月底的成分股代號。
    """
    daily = get_point_in_time_sp500(BACKTEST_START, BACKTEST_END, freq="daily")
    members: dict[str, set[str]] = {}
    for window in windows:
        day = pd.Timestamp(window["signal"]).date()
        if day not in daily:
            raise RuntimeError(f"{window['signal_month']} 的訊號日 {day.isoformat()} 沒有時點成分股")
        members[str(window["signal_month"])] = set(daily[day])
    return members


def _needs_download(symbol: str, start: date, end: date) -> bool:
    """
    這個 Yahoo 代號的本地價量是否還不夠這段區間。

    判斷與 backtest_data.load_ohlcv 的快取涵蓋規則相同。只讀本地檔，不發請求。

    參數:
        symbol: Yahoo 代號。
        start: 請求起日。
        end: 請求迄日。

    回傳:
        True 表示執行時會向 yfinance 要資料。
    """
    if not _cache_path(symbol).is_file():
        return True
    try:
        cached = _read_cache(symbol)
    except Exception as exc:  # noqa: BLE001 — 壞快取視為需要重抓
        _log.warning("讀取 %s 價量快取失敗，預估時視為需要下載: %s", symbol, exc)
        return True
    if cached is None or cached.empty:
        return True
    return not _cache_covers(cached, start, end)


def _shift_month(token: str, offset: int) -> str:
    """把 YYYY-MM 平移 offset 個月。"""
    year, month = (int(part) for part in token.split("-", 1))
    index = year * 12 + (month - 1) + offset
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def _format_month_span(months: list[str]) -> str:
    """
    把已排序的 YYYY-MM 收成連續區間。

    參數:
        months: 由早到晚的月份。

    回傳:
        例如「2023-09 至 2024-02、2024-06」。
    """
    if not months:
        return ""
    ranges: list[tuple[str, str]] = []
    start = previous = months[0]
    for month in months[1:]:
        if month == _shift_month(previous, 1):
            previous = month
            continue
        ranges.append((start, previous))
        start = previous = month
    ranges.append((start, previous))
    parts = [start if start == end else f"{start} 至 {end}" for start, end in ranges]
    return "、".join(parts)


def _format_duration(seconds: float) -> str:
    """把秒數說成幾分鐘又幾秒。不滿一分鐘只說秒。"""
    total = max(0, int(round(seconds)))
    minutes, remain = divmod(total, 60)
    if minutes == 0:
        return f"{remain} 秒"
    if remain == 0:
        return f"{minutes} 分鐘"
    return f"{minutes} 分鐘 {remain} 秒"


def build_plan() -> dict[str, Any]:
    """
    只用時點名單與本地快取做下載預估。不向 yfinance 要個股歷史。

    回傳:
        訊號月、每月成分股、要請求的 Yahoo 代號，以及快取是否已涵蓋。
    """
    windows = _signal_windows()
    members = _members_by_month(windows)
    fetch_start, fetch_end = _fetch_bounds()
    yahoo_of: dict[str, str] = {}
    months_of: dict[str, list[str]] = {}
    for month, names in members.items():
        for raw in names:
            yahoo = _yahoo_symbol(raw)
            yahoo_of[raw] = yahoo
            months_of.setdefault(raw, []).append(month)
    symbols = sorted(set(yahoo_of.values()))
    pending = [symbol for symbol in symbols if _needs_download(symbol, fetch_start, fetch_end)]
    return {
        "windows": windows,
        "members": members,
        "yahoo_of": yahoo_of,
        "months_of": months_of,
        "symbols": symbols,
        "pending": pending,
        "fetch_start": fetch_start,
        "fetch_end": fetch_end,
    }


def _print_estimate(plan: dict[str, Any]) -> None:
    """印出每月成分股數、不重複檔數，以及預估下載與檢查時間。不下載。"""
    members: dict[str, set[str]] = plan["members"]
    windows: list[dict[str, Any]] = plan["windows"]
    symbols: list[str] = plan["symbols"]
    pending: list[str] = plan["pending"]
    sizes = [len(members[window["signal_month"]]) for window in windows]
    print(
        f"固定區間 {BACKTEST_START.isoformat()} 至 {BACKTEST_END.isoformat()}。"
        f"訊號月 {windows[0]['signal_month']} 至 {windows[-1]['signal_month']}，共 {len(windows)} 個。",
        flush=True,
    )
    print(
        "每個月只用該月底的 S&P 500 成分股。下面的檔數是時點名單，還沒向 yfinance 要股價。",
        flush=True,
    )
    print(f"{'訊號月':<10}{'當時成分股':>10}", flush=True)
    for window in windows:
        month = window["signal_month"]
        print(f"{month:<10}{len(members[month]):>10}", flush=True)
    print(
        f"每月成分股 {min(sizes)}–{max(sizes)} 檔，平均 {sum(sizes) / len(sizes):.1f} 檔。",
        flush=True,
    )
    dotted = sorted(raw for raw in plan["yahoo_of"] if "." in raw)
    print(
        f"36 個月出現過的不重複代號 {len(plan['yahoo_of'])} 個，"
        f"轉成 Yahoo 代號後是 {len(symbols)} 檔。",
        flush=True,
    )
    if dotted:
        print(
            f"其中 {len(dotted)} 個代號含有句點，請求時改成連字號：{', '.join(dotted)}。",
            flush=True,
        )
    print(f"本地快取已涵蓋這段區間：{len(symbols) - len(pending)} 檔。", flush=True)
    print(f"需要下載：{len(pending)} 檔。", flush=True)
    download_seconds = len(pending) * _ESTIMATE_SECONDS_PER_FETCH
    check_seconds = len(symbols) * _ESTIMATE_SECONDS_PER_STOCK_CHECK
    print(
        f"預估下載時間約 {_format_duration(download_seconds)}"
        f"（每檔 {_ESTIMATE_SECONDS_PER_FETCH:.1f} 秒，含呼叫前 0.7 秒間隔；"
        "抓不到而重試 3 次的代號會更久）。",
        flush=True,
    )
    print(
        f"預估前視偏差檢查約 {_format_duration(check_seconds)}"
        f"（已下載的 {len(symbols)} 檔都要截斷重算）。",
        flush=True,
    )
    print(
        f"合計約 {_format_duration(download_seconds + check_seconds)}。這是成功路徑的粗估，不是保證。",
        flush=True,
    )
    print("尚未下載。確認後再執行：", flush=True)
    print("    python src/factor_research_pit.py --confirm", flush=True)


def _print_gap_report(
    plan: dict[str, Any],
    loaded: set[str],
) -> None:
    """
    印出每個月的成分股、實際抓到的檔數、缺漏比例，以及整段抓不到的代號。

    參數:
        plan: build_plan 的結果。
        loaded: 成功取得價量的 Yahoo 代號。
    """
    members: dict[str, set[str]] = plan["members"]
    yahoo_of: dict[str, str] = plan["yahoo_of"]
    months_of: dict[str, list[str]] = plan["months_of"]
    windows: list[dict[str, Any]] = plan["windows"]
    print("\n=== 資料缺口（時點成分股 vs 實際取得的價量）===", flush=True)
    print("缺漏比例 = （當時成分股 − 抓到的檔數） / 當時成分股。", flush=True)
    header = f"{'訊號月':<10}{'當時成分股':>10}{'抓到':>8}{'缺漏比例':>10}"
    print(header, flush=True)
    for window in windows:
        month = window["signal_month"]
        names = members[month]
        fetched = sum(1 for raw in names if yahoo_of[raw] in loaded)
        ratio = (len(names) - fetched) / len(names) if names else float("nan")
        print(f"{month:<10}{len(names):>10}{fetched:>8}{ratio * 100:>9.1f}%", flush=True)

    missing = [raw for raw in months_of if yahoo_of[raw] not in loaded]
    print(f"\n整段抓不到的代號：{len(missing)} 個。", flush=True)
    if not missing:
        print("每個時點代號都有價量。", flush=True)
        return
    for raw in sorted(missing):
        months = sorted(set(months_of[raw]))
        yahoo = yahoo_of[raw]
        label = raw if yahoo == raw else f"{raw}（請求 {yahoo}）"
        print(f"  {label}：{_format_month_span(months)}", flush=True)
    acquired = [raw for raw in sorted(missing) if raw in _ACQUIRED_BEFORE_DOWNLOAD]
    unknown = [raw for raw in sorted(missing) if raw not in _ACQUIRED_BEFORE_DOWNLOAD]
    still_listed = [raw for raw in _STILL_LISTED_UNCLEAR if raw in unknown]
    print(
        "確定下市或被併購："
        + ("、".join(acquired) if acquired else "無")
        + "。這些公司在這次下載前已完成下市或被併購。",
        flush=True,
    )
    unknown_text = "、".join(unknown) if unknown else "無"
    note = ""
    if still_listed:
        note = (
            "其中 "
            + "、".join(still_listed)
            + " 目前應仍上市，抓不到較像 Yahoo 代號或報價問題，不是下市。"
        )
    print(f"原因不明：{unknown_text}。{note}", flush=True)


def _pit_monthly_ic(
    factors: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    windows: list[dict[str, Any]],
    members: dict[str, set[str]],
    yahoo_of: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    """
    每個月只用該月底成分股計算 rank IC。

    算法與 factor_research._monthly_ic 相同：持有期報酬用 _holding_price_returns，
    相關用 _spearman。沒有價量的成分股不進入這個橫斷面。
    這裡不把 36 個月的聯集拿來排名。

    參數:
        factors: 已取得價量的股票的因子表，鍵是 Yahoo 代號。
        close: 對齊到 SPY 日曆的收盤價。
        windows: 訊號月。
        members: 每月的原始成分股代號。
        yahoo_of: 原始代號對到 Yahoo 代號。

    回傳:
        (月度 IC, 每個因子在各有效月份的橫斷面檔數)。
    """
    ic = pd.DataFrame(
        index=[window["signal_month"] for window in windows],
        columns=[key for key, _label in _FACTORS],
        dtype=float,
    )
    used_counts: dict[str, list[int]] = {key: [] for key, _label in _FACTORS}
    for window in windows:
        month = str(window["signal_month"])
        loaded_members = sorted(
            {yahoo_of[raw] for raw in members[month] if yahoo_of[raw] in factors}
        )
        returns = _holding_price_returns(close, window["hold"]).reindex(loaded_members)
        for key, _label in _FACTORS:
            scores = _cross_section(factors, loaded_members, window["signal"], key)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                value, used, _missing = _spearman(scores, returns)
            ic.loc[month, key] = value
            if math.isfinite(value):
                used_counts[key].append(int(used))
    ic.index.name = "signal_month"
    return ic, used_counts


def _print_pit_table(ic: pd.DataFrame, used_counts: dict[str, list[int]]) -> list[dict[str, Any]]:
    """
    印出與 factor_research 相同的 IC 表，並多一欄平均橫斷面檔數。

    平均橫斷面檔數是有效月份裡進入 Spearman 的平均檔數。

    參數:
        ic: 月度 IC。
        used_counts: 每個因子的有效月份檔數。

    回傳:
        依 |t| 由大到小排好的列，供 t 值對照使用。
    """
    rows: list[dict[str, Any]] = []
    for key, label in _FACTORS:
        stats = _ic_stats(ic[key])
        counts = used_counts[key]
        average = float(sum(counts) / len(counts)) if counts else float("nan")
        t_value = float(stats["t"])
        rows.append(
            {
                "key": key,
                "因子": label,
                "平均 IC": float(stats["mean"]),
                "IC 標準差": float(stats["std"]),
                "t 統計量": t_value,
                "abs_t": abs(t_value) if math.isfinite(t_value) else -1.0,
                "IC>0 月份比例": _fmt_ratio(
                    float(stats["positive_ratio"]),
                    int(stats["n_positive"]),
                    int(stats["n"]),
                ),
                "有效月數": int(stats["n"]),
                "平均橫斷面檔數": average,
            }
        )
    rows.sort(key=lambda row: (-float(row["abs_t"]), str(row["因子"])))
    display = pd.DataFrame(
        {
            "因子": [row["因子"] for row in rows],
            "平均 IC": [_fmt_num(float(row["平均 IC"]), 3) for row in rows],
            "IC 標準差": [_fmt_num(float(row["IC 標準差"]), 3) for row in rows],
            "t 統計量": [_fmt_num(float(row["t 統計量"]), 2) for row in rows],
            "IC>0 月份比例": [row["IC>0 月份比例"] for row in rows],
            "有效月數": [row["有效月數"] for row in rows],
            "平均橫斷面檔數": [
                _fmt_num(float(row["平均橫斷面檔數"]), 1) for row in rows
            ],
        }
    )
    print("\n=== 月度 rank IC（時點 S&P 500，Spearman，依 |t| 由大到小）===", flush=True)
    print(
        "t = 平均 IC / (IC 樣本標準差 / sqrt(有效月數))。"
        "平均橫斷面檔數是這些有效月份裡進入 Spearman 的平均檔數。",
        flush=True,
    )
    print(display.to_string(index=False), flush=True)
    return rows


def _group_a_t(*, cache_only: bool) -> dict[str, float]:
    """
    用 factor_research 的同一套算法，重算組 A 34 檔的 t 值。

    這份名單是固定清單。對照用，不拿來做時點 IC。

    參數:
        cache_only: True 時組 A 也只讀本地快取。

    回傳:
        因子代號對到 t 統計量。
    """
    tickers = list(define_three_groups()["A"])
    if cache_only:
        fetch_start, fetch_end = _fetch_bounds()
        histories = _load_prices(tickers, fetch_start, fetch_end, cache_only=True)
        spy = _slice_cache(_CALENDAR, fetch_start, fetch_end)
        if spy is None or spy.empty:
            raise RuntimeError("本地沒有 SPY 價量，無法重算 34 檔")
        master = _prepare_ohlcv(spy).index
        close = pd.DataFrame(
            {ticker: history["Close"].reindex(master) for ticker, history in histories.items()},
            index=master,
        )
    else:
        histories, master, close = _load_histories(tickers)
    windows = _holding_windows(master)
    signals = [window["signal"] for window in windows]
    score_dates = pd.DatetimeIndex([*signals, *(history.index[-1] for history in histories.values())])
    factors: dict[str, pd.DataFrame] = {}
    for ticker, history in histories.items():
        factor_frame, _missing = compute_factors(history, score_dates=score_dates)
        factors[ticker] = factor_frame
    ic, _shortfalls = _monthly_ic(factors, close, windows, tickers)
    return {key: float(_ic_stats(ic[key])["t"]) for key, _label in _FACTORS}


def _print_t_comparison(pit_rows: list[dict[str, Any]], group_a_t: dict[str, float]) -> None:
    """
    把 34 檔固定清單與時點 S&P 500 的 t 值放在同一張表。

    參數:
        pit_rows: PIT 表的列，已依 |t| 排序。
        group_a_t: 組 A 的 t 值。
    """
    print("\n=== t 值對照（34 檔固定清單 vs 時點 S&P 500）===", flush=True)
    print("34 檔是組 A，每個月用同一份名單。PIT 每個月只用該月底的成分股。", flush=True)
    header = f"{'因子':<16}{'34 檔 t':>10}{'PIT t':>10}"
    print(header, flush=True)
    for row in pit_rows:
        print(
            f"{row['因子']:<16}"
            f"{_fmt_num(group_a_t.get(str(row['key']), float('nan')), 2):>10}"
            f"{_fmt_num(float(row['t 統計量']), 2):>10}",
            flush=True,
        )


def _slice_cache(symbol: str, start: date, end: date) -> pd.DataFrame | None:
    """
    只讀本地價量快取，不向 yfinance 發請求。

    參數:
        symbol: Yahoo 代號。
        start: 切片起日（含）。
        end: 切片迄日（含）。

    回傳:
        區間內的 OHLCV。沒有快取或切出來是空的則為 None。
    """
    try:
        cached = _read_cache(symbol)
    except Exception as exc:  # noqa: BLE001 — 單檔壞快取不中斷其餘代號
        _log.warning("讀取 %s 價量快取失敗: %s", symbol, exc)
        return None
    if cached is None or cached.empty:
        return None
    sliced = cached.loc[pd.Timestamp(start) : pd.Timestamp(end)]
    if sliced.empty:
        return None
    return sliced


def _load_prices(
    symbols: list[str],
    start: date,
    end: date,
    *,
    cache_only: bool,
) -> dict[str, pd.DataFrame]:
    """
    取得這些代號的價量。單一代號失敗就略過，留給缺口報告。

    參數:
        symbols: Yahoo 代號。
        start: 請求起日。
        end: 請求迄日。
        cache_only: True 時只讀本地快取，不補下載。

    回傳:
        成功取得、且已整理過的 OHLCV。
    """
    panel: dict[str, pd.DataFrame] = {}
    if cache_only:
        for symbol in symbols:
            frame = _slice_cache(symbol, start, end)
            if frame is None:
                _log.warning("%s 沒有可用的本地價量", symbol)
                continue
            panel[symbol] = frame
    else:
        try:
            downloaded = build_ohlcv_panel([*symbols, _CALENDAR], start, end)
        except Exception:
            _log.exception("價量面板建立失敗")
            raise
        panel = {symbol: frame for symbol, frame in downloaded.items() if symbol != _CALENDAR}
    loaded: dict[str, pd.DataFrame] = {}
    for symbol, frame in panel.items():
        if frame is None or frame.empty:
            _log.warning("%s 沒有可用價量", symbol)
            continue
        try:
            loaded[symbol] = _prepare_ohlcv(frame)
        except Exception:
            _log.exception("%s 價量整理失敗", symbol)
    return loaded


def run_confirmed(plan: dict[str, Any], *, cache_only: bool) -> int:
    """
    讀取價量、做前視偏差檢查，再印 PIT 的 IC 與 34 檔 t 值對照。

    參數:
        plan: build_plan 的結果。確認後才呼叫。
        cache_only: True 時不向 yfinance 補下載。

    回傳:
        0 表示檢查通過且表已印出；1 表示中止。
    """
    symbols: list[str] = plan["symbols"]
    if cache_only:
        print(f"\n使用本地價量快取，不重新下載。要讀取 {len(symbols)} 檔。", flush=True)
    else:
        print(
            f"\n開始下載。要處理 {len(symbols)} 檔，其中 {len(plan['pending'])} 檔不在可用快取裡。",
            flush=True,
        )
    try:
        histories = _load_prices(
            symbols,
            plan["fetch_start"],
            plan["fetch_end"],
            cache_only=cache_only,
        )
    except Exception as exc:
        print(f"價量讀取失敗，研究中止：{exc}", flush=True)
        return 1
    if not histories:
        print("沒有任何一檔取得價量，研究中止。", flush=True)
        return 1
    _print_gap_report(plan, set(histories))

    windows: list[dict[str, Any]] = plan["windows"]
    signals = [window["signal"] for window in windows]
    score_dates = pd.DatetimeIndex(
        [*signals, *(history.index[-1] for history in histories.values() if len(history))]
    )
    factors: dict[str, pd.DataFrame] = {}
    for symbol, history in histories.items():
        try:
            factor_frame, _missing = compute_factors(history, score_dates=score_dates)
        except Exception:
            _log.exception("%s 因子計算失敗，這檔不進入橫斷面", symbol)
            continue
        factors[symbol] = factor_frame
    if not factors:
        print("因子全部計算失敗，研究中止。", flush=True)
        return 1
    dropped = sorted(set(histories) - set(factors))
    if dropped:
        print("這些代號有價量，但因子計算失敗，不進入橫斷面：" + ", ".join(dropped), flush=True)
    checked = {symbol: histories[symbol] for symbol in factors}

    if not _check_indicator_api(checked, factors):
        print("子分數與 indicators.calculate_tech_score 不一致，研究中止。", flush=True)
        return 1

    print(
        f"\n開始前視偏差檢查：{len(factors)} 檔 × {len(signals)} 個訊號月。"
        "截斷到該月底的因子值必須與完整資料相同。",
        flush=True,
    )
    try:
        mismatches = verify_no_lookahead(checked, factors, signals)
    except Exception:
        _log.exception("前視偏差檢查發生例外")
        print("前視偏差檢查發生例外，研究中止。", flush=True)
        return 1
    if mismatches:
        print(
            f"前視偏差檢查失敗：{len(mismatches)} 處不一致。研究中止，不計算 IC。",
            flush=True,
        )
        for row in mismatches[:12]:
            detail = f"，{row['detail']}" if row.get("detail") else ""
            print(
                f"  {row['factor']}  {row['month']}  {row['ticker']}"
                f"  完整資料={row['full']}  截斷至該月={row['truncated']}{detail}",
                flush=True,
            )
        if len(mismatches) > 12:
            print(f"  …其餘 {len(mismatches) - 12} 筆省略", flush=True)
        return 1
    print(
        f"前視偏差檢查通過：{len(_FACTORS)} 個因子 × {len(signals)} 個訊號月 × {len(factors)} 檔。",
        flush=True,
    )

    fetch_start, fetch_end = plan["fetch_start"], plan["fetch_end"]
    if cache_only:
        spy = _slice_cache(_CALENDAR, fetch_start, fetch_end)
    else:
        spy = load_ohlcv(_CALENDAR, fetch_start, fetch_end)
    if spy is None or spy.empty:
        print("沒有 SPY 價量，無法計算持有期報酬。", flush=True)
        return 1
    master = _prepare_ohlcv(spy).index
    # 因子表沒有 Close。收盤價用同一批代號的價量，而且只留因子有算出來的股票。
    closes = {symbol: histories[symbol]["Close"].reindex(master) for symbol in factors}
    close = pd.DataFrame(closes, index=master)
    ic, used_counts = _pit_monthly_ic(
        factors,
        close,
        windows,
        plan["members"],
        plan["yahoo_of"],
    )
    pit_rows = _print_pit_table(ic, used_counts)

    try:
        group_a_t = _group_a_t(cache_only=cache_only)
    except Exception:
        _log.exception("34 檔 t 值重算失敗")
        print("34 檔 t 值重算失敗，PIT 的表已印出，並排對照沒有完成。", flush=True)
        return 1
    _print_t_comparison(pit_rows, group_a_t)
    return 0


def main(argv: list[str] | None = None) -> int:
    """
    先印預估。只有明確帶 --confirm 才下載並計算。

    參數:
        argv: 參數列。None 時用 sys.argv。

    回傳:
        0 表示預估已印出，或研究完成；1 表示中止。
    """
    logging.getLogger("backtest_universe_sp500").setLevel(logging.WARNING)
    arguments = list(sys.argv[1:] if argv is None else argv)
    confirmed = "--confirm" in arguments
    cache_only = "--cache-only" in arguments
    _print_banner()
    try:
        plan = build_plan()
    except Exception as exc:
        _log.exception("時點名單或預估失敗")
        print(f"無法完成預估，研究中止：{exc}", flush=True)
        return 1
    _print_estimate(plan)
    if not confirmed:
        return 0
    if not cache_only:
        logging.getLogger("backtest_data").setLevel(logging.INFO)
    return run_confirmed(plan, cache_only=cache_only)


if __name__ == "__main__":
    sys.exit(main())
