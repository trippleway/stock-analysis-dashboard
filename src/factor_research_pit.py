# -*- coding: utf-8 -*-
"""
Factor rank IC on the point-in-time S&P 500. This computes IC only. It does not backtest and it does not pick stocks.

Factor definitions, holding windows, and the Spearman method follow factor_research.py.
The difference is the month-end cross section: only the constituents from
backtest_universe_sp500.get_point_in_time_sp500() on that month-end date.
It does not use a fixed list, and it does not use the union over the whole window.

Delisted symbols often have no yfinance quote. Names that cannot be fetched are not filled into the cross section,
so the result leans toward companies that can still be fetched. A missing quote does not mean every such name is delisted.

Downloading a few hundred histories is slow. With no arguments this only prints the estimated name count and time. It does not download.
Data is fetched and computed only after confirmation. When prices are already local, --cache-only does not call yfinance:

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

# Assumptions used only to estimate download time, not the sleep inside the real loop. The real gap is 0.7 seconds in backtest_data.
# 2.5 seconds covers that 0.7 seconds plus one successful history request. A failure retries and takes longer.
_ESTIMATE_SECONDS_PER_FETCH = 2.5
# The lookahead check truncates and recomputes every signal month. The whole check is estimated at about 0.3 seconds per name.
_ESTIMATE_SECONDS_PER_STOCK_CHECK = 0.3
_SURVIVOR_BANNER = "Delisted stocks cannot be fetched, so the result leans toward survivors."
# Symbols that finished a delisting or acquisition before this download, so Yahoo has no history for this window.
# Only deals with a recorded close are listed. This set is not used to guess about companies that are still trading.
_ACQUIRED_BEFORE_DOWNLOAD = frozenset(
    {
        "ANSS",  # Synopsys, 2025-07
        "ATVI",  # Microsoft, 2023-10
        "CDAY",  # Old symbol; the issuer Dayforce was acquired by Thoma Bravo in 2026-02
        "CMA",  # Fifth Third, 2026-02
        "CTLT",  # Novo Holdings, 2024-12
        "DAY",  # Thoma Bravo, 2026-02
        "DFS",  # Capital One, 2025-05
        "HES",  # Chevron, 2025-07
        "JNPR",  # HPE, 2025-07
        "MRO",  # ConocoPhillips, 2024-11
        "PXD",  # Exxon, 2024-05
        "QRVO",  # Skyworks, 2026-10 (after the sample ended, before this download)
        "WBA",  # Sycamore, 2025-08
        "WRK",  # Smurfit Westrock, 2024-07
    }
)
# The user noted these symbols should still be listed. A miss is not treated as a delisting.
_STILL_LISTED_UNCLEAR = ("MMC", "EA", "BK", "K", "EQR", "AVB", "CTRA", "HOLX", "IPG")


def _yahoo_symbol(ticker: str) -> str:
    """
    Turn a historical constituent symbol into the Yahoo quote symbol.

    The S&P history file writes BRK.B. yfinance wants BRK-B.
    The rule matches backtest_engine_v2._yahoo_symbol. This only changes the symbol used for the request.
    It does not add a stock to, or remove a stock from, that month's constituents.

    Args:
        ticker: Symbol on the point-in-time list.

    Returns:
        The symbol used to request prices from yfinance.
    """
    return str(ticker).upper().strip().replace(".", "-")


def _fetch_bounds() -> tuple[date, date]:
    """
    Start and end dates for the price request. The start includes warmup so the first signal month's 252-day window can be computed.

    Returns:
        (request start, request end). The end is the backtest end date from config.
    """
    return BACKTEST_START - timedelta(days=_WARMUP_DAYS), BACKTEST_END


def _print_banner() -> None:
    """The first line of the research output. Delisted stocks have no quote, so the cross section loses them."""
    print(_SURVIVOR_BANNER, flush=True)


def _signal_windows() -> list[dict[str, Any]]:
    """
    Use SPY trading days to build the same 36 signal months as factor_research.

    SPY is not downloaded again when it is already in the local cache.

    Returns:
        One row per month, with signal, signal_month, and hold.
    """
    fetch_start, fetch_end = _fetch_bounds()
    spy = load_ohlcv(_CALENDAR, fetch_start, fetch_end)
    if spy is None or spy.empty:
        raise RuntimeError("No SPY prices, so the existing month-end calendar cannot be reused")
    prepared = _prepare_ohlcv(spy)
    windows = _holding_windows(prepared.index)
    if not windows:
        raise RuntimeError("There are no signal months")
    return windows


def _members_by_month(windows: list[dict[str, Any]]) -> dict[str, set[str]]:
    """
    S&P 500 constituents at each signal month-end.

    The daily point-in-time lists for the range are loaded once, then only the set on the signal date is kept.
    The daily lists come from get_point_in_time_sp500 and use the latest full list on or before that day.
    IC does not use the union over the whole range.

    Args:
        windows: Signal months.

    Returns:
        Signal month YYYY-MM mapped to the constituent symbols at that month-end.
    """
    daily = get_point_in_time_sp500(BACKTEST_START, BACKTEST_END, freq="daily")
    members: dict[str, set[str]] = {}
    for window in windows:
        day = pd.Timestamp(window["signal"]).date()
        if day not in daily:
            raise RuntimeError(f"Signal date {day.isoformat()} for {window['signal_month']} has no point-in-time constituents")
        members[str(window["signal_month"])] = set(daily[day])
    return members


def _needs_download(symbol: str, start: date, end: date) -> bool:
    """
    Whether this Yahoo symbol's local prices still do not cover the range.

    The check matches the cache-coverage rule in backtest_data.load_ohlcv. It only reads the local file and does not send a request.

    Args:
        symbol: Yahoo symbol.
        start: Request start.
        end: Request end.

    Returns:
        True when the run will ask yfinance for data.
    """
    if not _cache_path(symbol).is_file():
        return True
    try:
        cached = _read_cache(symbol)
    except Exception as exc:  # noqa: BLE001 — a broken cache is treated as needing a refetch
        _log.warning("Failed to read the price cache for %s; the estimate treats it as a download: %s", symbol, exc)
        return True
    if cached is None or cached.empty:
        return True
    return not _cache_covers(cached, start, end)


def _shift_month(token: str, offset: int) -> str:
    """Shift a YYYY-MM token by offset months."""
    year, month = (int(part) for part in token.split("-", 1))
    index = year * 12 + (month - 1) + offset
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def _format_month_span(months: list[str]) -> str:
    """
    Collapse sorted YYYY-MM values into contiguous ranges.

    Args:
        months: Months from earliest to latest.

    Returns:
        For example "2023-09 to 2024-02, 2024-06".
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
    parts = [start if start == end else f"{start} to {end}" for start, end in ranges]
    return ", ".join(parts)


def _format_duration(seconds: float) -> str:
    """Say a number of seconds as minutes and seconds. Under one minute, only seconds are shown."""
    total = max(0, int(round(seconds)))
    minutes, remain = divmod(total, 60)
    if minutes == 0:
        return f"{remain} seconds"
    if remain == 0:
        return f"{minutes} minutes"
    return f"{minutes} minutes {remain} seconds"


def build_plan() -> dict[str, Any]:
    """
    Estimate the download from the point-in-time list and the local cache only. Individual histories are not requested from yfinance.

    Returns:
        Signal months, monthly constituents, Yahoo symbols to request, and whether the cache already covers them.
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
    """Print the monthly constituent count, the unique name count, and the estimated download and check time. Nothing is downloaded."""
    members: dict[str, set[str]] = plan["members"]
    windows: list[dict[str, Any]] = plan["windows"]
    symbols: list[str] = plan["symbols"]
    pending: list[str] = plan["pending"]
    sizes = [len(members[window["signal_month"]]) for window in windows]
    print(
        f"Fixed window {BACKTEST_START.isoformat()} to {BACKTEST_END.isoformat()}. "
        f"Signal months {windows[0]['signal_month']} to {windows[-1]['signal_month']}, {len(windows)} in total.",
        flush=True,
    )
    print(
        "Each month uses only that month-end's S&P 500 constituents. The counts below are the point-in-time list. Prices have not been requested from yfinance yet.",
        flush=True,
    )
    print(f"{'Month':<10}{'Members':>10}", flush=True)
    for window in windows:
        month = window["signal_month"]
        print(f"{month:<10}{len(members[month]):>10}", flush=True)
    print(
        f"Monthly constituents {min(sizes)}-{max(sizes)} names, average {sum(sizes) / len(sizes):.1f}.",
        flush=True,
    )
    dotted = sorted(raw for raw in plan["yahoo_of"] if "." in raw)
    print(
        f"Unique symbols across 36 months: {len(plan['yahoo_of'])}. "
        f"After converting to Yahoo symbols: {len(symbols)}.",
        flush=True,
    )
    if dotted:
        print(
            f"{len(dotted)} of those symbols contain a dot and are requested with a hyphen: {', '.join(dotted)}.",
            flush=True,
        )
    print(f"Local cache already covers this window: {len(symbols) - len(pending)} names.", flush=True)
    print(f"Need to download: {len(pending)} names.", flush=True)
    download_seconds = len(pending) * _ESTIMATE_SECONDS_PER_FETCH
    check_seconds = len(symbols) * _ESTIMATE_SECONDS_PER_STOCK_CHECK
    print(
        f"Estimated download time about {_format_duration(download_seconds)} "
        f"({_ESTIMATE_SECONDS_PER_FETCH:.1f} seconds per name, including the 0.7 second gap before each call; "
        "a symbol that misses and retries 3 times takes longer).",
        flush=True,
    )
    print(
        f"Estimated lookahead check about {_format_duration(check_seconds)} "
        f"(all {len(symbols)} downloaded names are truncated and recomputed).",
        flush=True,
    )
    print(
        f"About {_format_duration(download_seconds + check_seconds)} in total. This is a rough estimate of the success path, not a guarantee.",
        flush=True,
    )
    print("Nothing has been downloaded yet. Run this after you confirm:", flush=True)
    print("    python src/factor_research_pit.py --confirm", flush=True)


def _print_gap_report(
    plan: dict[str, Any],
    loaded: set[str],
) -> None:
    """
    Print each month's constituent count, how many were actually fetched, the gap ratio, and symbols missing for the whole window.

    Args:
        plan: The result of build_plan.
        loaded: Yahoo symbols whose prices were fetched.
    """
    members: dict[str, set[str]] = plan["members"]
    yahoo_of: dict[str, str] = plan["yahoo_of"]
    months_of: dict[str, list[str]] = plan["months_of"]
    windows: list[dict[str, Any]] = plan["windows"]
    print("\n=== Data gaps (point-in-time constituents vs prices actually fetched) ===", flush=True)
    print("Gap ratio = (constituents then - names fetched) / constituents then.", flush=True)
    header = f"{'Month':<10}{'Members':>10}{'Fetched':>8}{'Gap':>10}"
    print(header, flush=True)
    for window in windows:
        month = window["signal_month"]
        names = members[month]
        fetched = sum(1 for raw in names if yahoo_of[raw] in loaded)
        ratio = (len(names) - fetched) / len(names) if names else float("nan")
        print(f"{month:<10}{len(names):>10}{fetched:>8}{ratio * 100:>9.1f}%", flush=True)

    missing = [raw for raw in months_of if yahoo_of[raw] not in loaded]
    print(f"\nSymbols missing for the whole window: {len(missing)}.", flush=True)
    if not missing:
        print("Every point-in-time symbol has prices.", flush=True)
        return
    for raw in sorted(missing):
        months = sorted(set(months_of[raw]))
        yahoo = yahoo_of[raw]
        label = raw if yahoo == raw else f"{raw} (requested {yahoo})"
        print(f"  {label}: {_format_month_span(months)}", flush=True)
    acquired = [raw for raw in sorted(missing) if raw in _ACQUIRED_BEFORE_DOWNLOAD]
    unknown = [raw for raw in sorted(missing) if raw not in _ACQUIRED_BEFORE_DOWNLOAD]
    still_listed = [raw for raw in _STILL_LISTED_UNCLEAR if raw in unknown]
    print(
        "Confirmed delisted or acquired: "
        + (", ".join(acquired) if acquired else "none")
        + ". These companies finished a delisting or acquisition before this download.",
        flush=True,
    )
    unknown_text = ", ".join(unknown) if unknown else "none"
    note = ""
    if still_listed:
        note = (
            "Of those, "
            + ", ".join(still_listed)
            + " should still be listed. A miss looks more like a Yahoo symbol or quote problem than a delisting."
        )
    print(f"Unknown: {unknown_text}. {note}", flush=True)


def _pit_monthly_ic(
    factors: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    windows: list[dict[str, Any]],
    members: dict[str, set[str]],
    yahoo_of: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    """
    Compute rank IC using only that month-end's constituents.

    The method matches factor_research._monthly_ic: holding-period returns use _holding_price_returns,
    and the correlation uses _spearman. Constituents with no prices do not enter this cross section.
    The union of the 36 months is not ranked here.

    Args:
        factors: Factor frames for stocks whose prices were fetched. Keys are Yahoo symbols.
        close: Closes aligned to the SPY calendar.
        windows: Signal months.
        members: Raw constituent symbols for each month.
        yahoo_of: Raw symbol mapped to the Yahoo symbol.

    Returns:
        (monthly IC, cross-section name counts for each factor in each usable month).
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
    Print the same IC table as factor_research, plus a column for the average cross-section count.

    The average cross-section count is the average number of names that entered Spearman in the usable months.

    Args:
        ic: Monthly IC.
        used_counts: Usable-month name counts for each factor.

    Returns:
        Rows sorted by |t| descending, for the t-stat comparison.
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
    print("\n=== Monthly rank IC (point-in-time S&P 500, Spearman, sorted by |t|) ===", flush=True)
    print(
        "t = mean IC / (IC sample standard deviation / sqrt(usable months)). "
        "The average cross-section count is the average number of names that entered Spearman in those usable months.",
        flush=True,
    )
    print(display.to_string(index=False), flush=True)
    return rows


def _group_a_t(*, cache_only: bool) -> dict[str, float]:
    """
    Recompute t-stats for the 34 names in group A with the same method as factor_research.

    That list is a fixed universe. It is for comparison and is not used for the point-in-time IC.

    Args:
        cache_only: When True, group A is also read from the local cache only.

    Returns:
        Factor key mapped to its t-stat.
    """
    tickers = list(define_three_groups()["A"])
    if cache_only:
        fetch_start, fetch_end = _fetch_bounds()
        histories = _load_prices(tickers, fetch_start, fetch_end, cache_only=True)
        spy = _slice_cache(_CALENDAR, fetch_start, fetch_end)
        if spy is None or spy.empty:
            raise RuntimeError("No local SPY prices, so the 34 names cannot be recomputed")
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
    Put t-stats for the fixed 34-name list and the point-in-time S&P 500 in one table.

    Args:
        pit_rows: Rows from the PIT table, already sorted by |t|.
        group_a_t: t-stats for group A.
    """
    print("\n=== t-stat comparison (fixed 34-name list vs point-in-time S&P 500) ===", flush=True)
    print("The 34 names are group A, the same list every month. PIT uses only that month-end's constituents.", flush=True)
    header = f"{'Factor':<16}{'Group A t':>10}{'PIT t':>10}"
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
    Read the local price cache only. No request is sent to yfinance.

    Args:
        symbol: Yahoo symbol.
        start: Slice start (inclusive).
        end: Slice end (inclusive).

    Returns:
        OHLCV inside the range. None when there is no cache or the slice is empty.
    """
    try:
        cached = _read_cache(symbol)
    except Exception as exc:  # noqa: BLE001 — one broken cache does not stop the other symbols
        _log.warning("Failed to read the price cache for %s: %s", symbol, exc)
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
    Fetch prices for these symbols. A single symbol that fails is skipped and left for the gap report.

    Args:
        symbols: Yahoo symbols.
        start: Request start.
        end: Request end.
        cache_only: When True, only the local cache is read and nothing is downloaded to fill gaps.

    Returns:
        OHLCV that was fetched and already prepared.
    """
    panel: dict[str, pd.DataFrame] = {}
    if cache_only:
        for symbol in symbols:
            frame = _slice_cache(symbol, start, end)
            if frame is None:
                _log.warning("%s has no usable local prices", symbol)
                continue
            panel[symbol] = frame
    else:
        try:
            downloaded = build_ohlcv_panel([*symbols, _CALENDAR], start, end)
        except Exception:
            _log.exception("Failed to build the price panel")
            raise
        panel = {symbol: frame for symbol, frame in downloaded.items() if symbol != _CALENDAR}
    loaded: dict[str, pd.DataFrame] = {}
    for symbol, frame in panel.items():
        if frame is None or frame.empty:
            _log.warning("%s has no usable prices", symbol)
            continue
        try:
            loaded[symbol] = _prepare_ohlcv(frame)
        except Exception:
            _log.exception("Failed to prepare prices for %s", symbol)
    return loaded


def run_confirmed(plan: dict[str, Any], *, cache_only: bool) -> int:
    """
    Load prices, run the lookahead check, then print the PIT IC and the 34-name t-stat comparison.

    Args:
        plan: The result of build_plan. Called only after confirmation.
        cache_only: When True, missing prices are not downloaded from yfinance.

    Returns:
        0 when the check passed and the tables were printed; 1 when the run stops.
    """
    symbols: list[str] = plan["symbols"]
    if cache_only:
        print(f"\nUsing the local price cache. Nothing will be downloaded again. {len(symbols)} names to read.", flush=True)
    else:
        print(
            f"\nStarting the download. {len(symbols)} names to process, of which {len(plan['pending'])} are not in a usable cache.",
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
        print(f"Failed to read prices. Research stopped: {exc}", flush=True)
        return 1
    if not histories:
        print("No name returned prices. Research stopped.", flush=True)
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
            _log.exception("Factor calculation failed for %s; this name is left out of the cross section", symbol)
            continue
        factors[symbol] = factor_frame
    if not factors:
        print("Every factor calculation failed. Research stopped.", flush=True)
        return 1
    dropped = sorted(set(histories) - set(factors))
    if dropped:
        print("These symbols have prices, but factor calculation failed, so they are left out of the cross section: " + ", ".join(dropped), flush=True)
    checked = {symbol: histories[symbol] for symbol in factors}

    if not _check_indicator_api(checked, factors):
        print("Subscores do not match indicators.calculate_tech_score. Research stopped.", flush=True)
        return 1

    print(
        f"\nStarting the lookahead check: {len(factors)} names x {len(signals)} signal months. "
        "Factor values truncated to that month-end must match the full history.",
        flush=True,
    )
    try:
        mismatches = verify_no_lookahead(checked, factors, signals)
    except Exception:
        _log.exception("The lookahead check raised an exception")
        print("The lookahead check raised an exception. Research stopped.", flush=True)
        return 1
    if mismatches:
        print(
            f"Lookahead check failed: {len(mismatches)} mismatches. Research stopped. IC was not computed.",
            flush=True,
        )
        for row in mismatches[:12]:
            detail = f", {row['detail']}" if row.get("detail") else ""
            print(
                f"  {row['factor']}  {row['month']}  {row['ticker']}"
                f"  full={row['full']}  truncated={row['truncated']}{detail}",
                flush=True,
            )
        if len(mismatches) > 12:
            print(f"  ... {len(mismatches) - 12} more rows omitted", flush=True)
        return 1
    print(
        f"Lookahead check passed: {len(_FACTORS)} factors x {len(signals)} signal months x {len(factors)} names.",
        flush=True,
    )

    fetch_start, fetch_end = plan["fetch_start"], plan["fetch_end"]
    if cache_only:
        spy = _slice_cache(_CALENDAR, fetch_start, fetch_end)
    else:
        spy = load_ohlcv(_CALENDAR, fetch_start, fetch_end)
    if spy is None or spy.empty:
        print("No SPY prices, so holding-period returns cannot be computed.", flush=True)
        return 1
    master = _prepare_ohlcv(spy).index
    # Factor frames have no Close. Closes come from the same symbols' prices, and only names whose factors were computed are kept.
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
        _log.exception("Recomputing t-stats for the 34 names failed")
        print("Recomputing t-stats for the 34 names failed. The PIT table was printed. The side-by-side comparison was not finished.", flush=True)
        return 1
    _print_t_comparison(pit_rows, group_a_t)
    return 0


def main(argv: list[str] | None = None) -> int:
    """
    Print the estimate first. Download and compute only when --confirm is passed explicitly.

    Args:
        argv: Argument list. sys.argv is used when this is None.

    Returns:
        0 when the estimate was printed or the research finished; 1 when the run stops.
    """
    logging.getLogger("backtest_universe_sp500").setLevel(logging.WARNING)
    arguments = list(sys.argv[1:] if argv is None else argv)
    confirmed = "--confirm" in arguments
    cache_only = "--cache-only" in arguments
    _print_banner()
    try:
        plan = build_plan()
    except Exception as exc:
        _log.exception("The point-in-time list or the estimate failed")
        print(f"The estimate could not be finished. Research stopped: {exc}", flush=True)
        return 1
    _print_estimate(plan)
    if not confirmed:
        return 0
    if not cache_only:
        logging.getLogger("backtest_data").setLevel(logging.INFO)
    return run_confirmed(plan, cache_only=cache_only)


if __name__ == "__main__":
    sys.exit(main())
