# -*- coding: utf-8 -*-
"""
Check whether Group A's monthly baseline is sensitive to the start month, and whether the return is concentrated in a few months.

The rolling test is centered on config.BACKTEST_FIRST_REBALANCE_MONTH, 12 months before and 12 months after,
25 start months in all. The end month is fixed at config.BACKTEST_LAST_MONTH.
Selection, holding, and costs reuse run_group_backtest. This file does not change the rule.

Month-return concentration looks only at the current baseline: compound each of the 36 holding months, then drop the best months.
SPY uses the same holding windows, and the months dropped are the ones where Group A did best.

The same fixed window and the same turnover cost are also used for four comparisons:
equal-weight all 34 names, pick 3 eligible names at random 1000 times,
Spearman rank correlation between the month-end score and next month's return, and the first and last 18 months.

Run from the project root:
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

# Twelve months before and after the center start month, including the center, 25 in all.
_ROLL_RADIUS = 12
_PERCENTILES = (5, 25, 50, 75, 95)
_DROP_BEST = (0, 1, 3, 5)
_DAYS_PER_YEAR = 365.25
_RANDOM_REPS = 1000
_RANDOM_SEED = 42
_HALF_MONTHS = 18


def _parse_year_month(token: str) -> tuple[int, int]:
    """Split YYYY-MM into year and month. A month outside 1-12 is rejected."""
    year_text, month_text = str(token).split("-", 1)
    year, month = int(year_text), int(month_text)
    if month < 1 or month > 12:
        raise ValueError(f"Month must be between 1 and 12, got {token}")
    return year, month


def _shift_month(token: str, offset: int) -> str:
    """Shift YYYY-MM by offset months. A negative offset moves earlier."""
    year, month = _parse_year_month(token)
    index = year * 12 + (month - 1) + offset
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def _month_span(start_month: str, end_month: str) -> int:
    """How many months apart two YYYY-MM values are. The same month is 0."""
    start_year, start = _parse_year_month(start_month)
    end_year, end = _parse_year_month(end_month)
    return (end_year * 12 + end) - (start_year * 12 + start)


def _month_first_day(token: str) -> date:
    """The first day of that month. The backtest uses it as the calendar bound. The rebalance day is still the month-end trading day."""
    year, month = _parse_year_month(token)
    return date(year, month, 1)


def _annualized(cumulative: float, start: date, end: date) -> float:
    """
    Turn a cumulative return into an annualized compound return.

    Years are calendar days in the holding window divided by 365.25, so start months of different lengths can be compared.

    Args:
        cumulative: Cumulative return for this stretch.
        start: First holding day.
        end: Last holding day.

    Returns:
        Annualized return. NaN when the day count is not positive or the cumulative return is not finite.
    """
    days = (end - start).days
    if days <= 0 or not math.isfinite(cumulative) or cumulative <= -1.0:
        return float("nan")
    return float((1.0 + cumulative) ** (_DAYS_PER_YEAR / days) - 1.0)


def _window_return(daily: pd.Series, start: str, end: str) -> float:
    """Compounded return over a holding window. Both endpoints are included. NaN when there is no daily return."""
    window = daily.loc[pd.Timestamp(start) : pd.Timestamp(end)]
    if window.empty or window.isna().all():
        return float("nan")
    return float((1.0 + window.fillna(0.0)).prod() - 1.0)


def _pct(value: float) -> str:
    """Format a ratio as a percent. A non-finite value is an em dash."""
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value * 100:.1f}%"


def _num(value: float) -> str:
    """Format a Sharpe ratio or a percentile to two decimals."""
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:.2f}"


def _percentile_rank(values: list[float], point: float) -> float:
    """
    Empirical percentile of the baseline inside this set of numbers.

    The definition is the share of values that are not above the baseline. 100 means the baseline is the highest in the set.
    About 50 means half of the start months are not above the baseline.

    Args:
        values: Metrics from start months that finished.
        point: The same metric for the current baseline.

    Returns:
        A percentile from 0 to 100. NaN when there is no comparable sample.
    """
    usable = [value for value in values if math.isfinite(value)]
    if not usable or not math.isfinite(point):
        return float("nan")
    return float(100.0 * sum(1 for value in usable if value <= point) / len(usable))


def _start_months() -> list[str]:
    """From 12 months before the center start month through 12 months after, earliest to latest."""
    return [
        _shift_month(BACKTEST_FIRST_REBALANCE_MONTH, offset)
        for offset in range(-_ROLL_RADIUS, _ROLL_RADIUS + 1)
    ]


def _row_from_result(start_month: str, result: dict[str, Any]) -> dict[str, Any]:
    """
    Turn one monthly backtest into a row. A mismatched start or end month is marked insufficient data.

    Args:
        start_month: The first rebalance month requested for this run, YYYY-MM.
        result: Return value of run_group_backtest.

    Returns:
        A dict with annualized return, Sharpe, max drawdown, and month count. ok is False when the data are insufficient.
    """
    months = list(result.get("months") or [])
    expected = _month_span(start_month, BACKTEST_LAST_MONTH)
    reasons: list[str] = []
    if not months:
        reasons.append("no holding months")
    else:
        first_signal = str(months[0]["signal"])[:7]
        last_hold = str(months[-1]["hold_end"])[:7]
        if first_signal != start_month:
            reasons.append(f"first rebalance day is {months[0]['signal']}, not {start_month}")
        if last_hold != BACKTEST_LAST_MONTH:
            reasons.append(f"last holding day is {months[-1]['hold_end']}, not {BACKTEST_LAST_MONTH}")
    if int(result["n_rebalances"]) != expected:
        reasons.append(f"{int(result['n_rebalances'])} rebalances, but this span should be {expected} months")
    hold_start = date.fromisoformat(str(result["start"]))
    hold_end = date.fromisoformat(str(result["end"]))
    cumulative = float(result["cumulative_return_net"])
    row = {
        "start_month": start_month,
        "ok": not reasons,
        "reason": "; ".join(reasons),
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
    Run the Group A monthly baseline once for each of 25 start months. Every run ends at BACKTEST_LAST_MONTH.

    Runs go from earliest to latest. An earlier start month fills the price cache forward, so later months do not refetch the same span.
    A month that fails, or whose window does not match, is marked insufficient data. The other months continue.

    Args:
        tickers: Group A symbols.

    Returns:
        25 rows, in start-month order. The current baseline row also keeps the full backtest result.
    """
    rows: list[dict[str, Any]] = []
    for start_month in _start_months():
        start_day = _month_first_day(start_month)
        _log.info("Rolling start month %s, end month %s", start_month, BACKTEST_LAST_MONTH)
        try:
            result = run_group_backtest(tickers, start_date=start_day, end_date=BACKTEST_END)
        except Exception as exc:  # noqa: BLE001 — a single month's failure stays on the table
            _log.warning("Start month %s has insufficient data: %s", start_month, exc)
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
            _log.warning("Start month %s has insufficient data: %s", start_month, row["reason"])
            row["result"] = None
        else:
            row["result"] = None
        rows.append(row)
    return rows


def _print_rolling(rows: list[dict[str, Any]]) -> None:
    """Print the 25 start months, plus percentiles of annualized return and Sharpe."""
    print(
        f"\n=== Rolling start months (end month fixed at {BACKTEST_LAST_MONTH}, Group A monthly baseline, after costs) ===",
        flush=True,
    )
    print(
        f"Center is {BACKTEST_FIRST_REBALANCE_MONTH}, {_ROLL_RADIUS} months before and after, {len(rows)} start months.",
        flush=True,
    )
    header = (
        f"{'Start':<10}{'Months':>8}{'Cumulative':>12}{'Annualized':>12}"
        f"{'Sharpe':>8}{'Max DD':>10}  Status"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for row in rows:
        mark = "baseline" if row["start_month"] == BACKTEST_FIRST_REBALANCE_MONTH else ""
        status = "insufficient data" if not row["ok"] else mark
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
            print(f"           reason: {row['reason']}", flush=True)

    usable = [row for row in rows if row["ok"]]
    print(
        f"\nPercentiles use only the {len(usable)} start months that finished. Insufficient-data rows stay above and are not in the distribution.",
        flush=True,
    )
    if not usable:
        print("No start month finished, so percentiles cannot be computed.", flush=True)
        return
    baseline = next(
        (row for row in usable if row["start_month"] == BACKTEST_FIRST_REBALANCE_MONTH),
        None,
    )
    for key, title in (("annualized", "Annualized return"), ("sharpe", "Sharpe")):
        values = [float(row[key]) for row in usable]
        print(f"{title} percentiles:", flush=True)
        for pct in _PERCENTILES:
            level = float(np.percentile(values, pct))
            shown = _pct(level) if key == "annualized" else _num(level)
            print(f"  {pct:>2}th  {shown}", flush=True)
        if baseline is None:
            print(f"  The current baseline {BACKTEST_FIRST_REBALANCE_MONTH} has no complete result, so it cannot be ranked.", flush=True)
            continue
        rank = _percentile_rank(values, float(baseline[key]))
        shown = _pct(float(baseline[key])) if key == "annualized" else _num(float(baseline[key]))
        print(
            f"  Current baseline {baseline['start_month']} {title} is {shown}, "
            f"at the {_num(rank)} percentile "
            f"(share of {len(usable)} complete start months that are not above the baseline).",
            flush=True,
        )


def _monthly_returns(result: dict[str, Any], field: str) -> list[dict[str, Any]]:
    """
    Compound each holding month's daily returns into one monthly return.

    Args:
        result: The current baseline backtest. It must include months and daily returns.
        field: net_returns or spy_returns.

    Returns:
        One row per month, with the rebalance month, the holding window, and the monthly return. The monthly return is NaN when the window has no daily returns.
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
    """Compound the months that are not in excluded. An empty excluded set is the full sample."""
    kept = [float(row["return"]) for row in rows if row["signal"] not in excluded]
    if not kept or any(not math.isfinite(value) for value in kept):
        return float("nan")
    return float(math.prod(1.0 + value for value in kept) - 1.0)


def _stock_closes(tickers: list[str]) -> dict[str, pd.Series]:
    """
    Load closes for these symbols over the baseline window.

    Cache before the baseline start is kept so the first holding day's move has a prior close to divide by.
    A symbol with no prices is skipped. The caller marks that name as insufficient data.

    Args:
        tickers: Symbols to load.

    Returns:
        Symbol to close. The index is trading days.
    """
    closes: dict[str, pd.Series] = {}
    for ticker in sorted(set(tickers)):
        try:
            frame = load_ohlcv(ticker, BACKTEST_START, BACKTEST_END)
        except Exception as exc:  # noqa: BLE001 — one name's failure does not stop the others
            _log.warning("%s price load failed: %s", ticker, exc)
            continue
        if frame is None or frame.empty or "Close" not in frame.columns:
            _log.warning("%s has no close, so that month's return is marked insufficient data", ticker)
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
    Price return of one stock over a holding window.

    Daily returns from the first holding day through the last are compounded, including the first day's move versus the prior close.
    This is a price return. Rebalance cost is not included.

    Args:
        close: That name's closes.
        hold_start: First holding day.
        hold_end: Last holding day.

    Returns:
        Price return over the window. NaN when no usable close is available.
    """
    daily = close.pct_change().loc[pd.Timestamp(hold_start) : pd.Timestamp(hold_end)]
    if daily.empty or daily.isna().all():
        return float("nan")
    return float((1.0 + daily.fillna(0.0)).prod() - 1.0)


def _print_top_holdings(ranked: list[dict[str, Any]], n_best: int = 5) -> None:
    """
    Print which 3 names were picked in the best months, and how often those holdings appear.

    Args:
        ranked: Group A monthly returns from high to low. Each row needs picked and the holding window.
        n_best: How many months to expand.
    """
    chosen = ranked[:n_best]
    print(f"\nBest {len(chosen)} months: that month's picks and each name's price return", flush=True)
    counts: dict[str, int] = {}
    closes = _stock_closes([ticker for row in chosen for ticker in row["picked"]])
    for row in chosen:
        print(
            f"{row['signal']}  held {row['period']}  Group A month return {_pct(row['return'])}",
            flush=True,
        )
        if not row["picked"]:
            print("  No stock was picked this month", flush=True)
            continue
        for ticker in row["picked"]:
            counts[ticker] = counts.get(ticker, 0) + 1
            close = closes.get(ticker)
            stock_return = _price_return(close, row["hold_start"], row["hold_end"]) if close is not None else float("nan")
            label = _pct(stock_return) if math.isfinite(stock_return) else "insufficient data"
            print(f"  {ticker:<6}{label}", flush=True)
    print(f"\nHow often these {sum(counts.values())} holdings appear", flush=True)
    if not counts:
        print("No holdings to count.", flush=True)
        return
    for ticker, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {ticker:<6}{count}", flush=True)


def _print_concentration(result: dict[str, Any]) -> None:
    """Print the baseline's 36 months ranked by return, and the cumulative return after the best months are removed."""
    strategy = _monthly_returns(result, "net_returns")
    spy = _monthly_returns(result, "spy_returns")
    print(
        f"\n=== Monthly-return concentration ({BACKTEST_FIRST_REBALANCE_MONTH} to {BACKTEST_LAST_MONTH}, after costs) ===",
        flush=True,
    )
    print(
        "One month is the holding window after one rebalance. Group A drops its own best months. "
        "SPY drops those same windows, not the months when SPY itself did best.",
        flush=True,
    )
    missing = [row["signal"] for row in strategy + spy if not math.isfinite(row["return"])]
    if missing:
        print("These months have no daily return, so the concentration is incomplete: " + ", ".join(missing), flush=True)

    ranked = sorted(strategy, key=lambda row: row["return"], reverse=True)
    print(f"\nGroup A monthly returns, high to low ({len(ranked)} months)", flush=True)
    header = f"{'Rank':>4}  {'Month':<10}{'Holding window':<27}{'Month return':>14}"
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
            f"Compounding the 36 months gives {_pct(product)}, and the full after-cost cumulative is {_pct(reported)}. They do not match.",
            flush=True,
        )

    print("\nAfter dropping Group A's best months (SPY drops the same windows)", flush=True)
    summary = f"{'Scope':<22}{'Group A cumulative':>20}{'SPY cumulative':>16}"
    print(summary, flush=True)
    print("-" * len(summary), flush=True)
    strategy_order = [row["signal"] for row in ranked]
    for drop in _DROP_BEST:
        excluded = set(strategy_order[:drop])
        label = "All months" if drop == 0 else f"Drop best {drop}"
        spy_kept = _compound_except(spy, excluded)
        print(
            f"{label:<22}"
            f"{_pct(_compound_except(strategy, excluded)):>12}"
            f"{_pct(spy_kept):>12}",
            flush=True,
        )
        if drop:
            print(f"  Same windows: {', '.join(strategy_order[:drop])}", flush=True)
            if drop == 5:
                print(f"  SPY cumulative after the correction: {_pct(spy_kept)}", flush=True)
    _print_top_holdings(ranked, n_best=5)


def _load_pool(tickers: list[str]) -> dict[str, Any]:
    """
    Prepare closes, scores, and holding windows with the same warmup and month-ends as the monthly backtest.

    Args:
        tickers: Group A symbols.

    Returns:
        A dict with close, score, ready, holding windows, and each month's eligible names.
    """
    fetch_start = BACKTEST_START - timedelta(days=_WARMUP_DAYS)
    panel = build_ohlcv_panel([*tickers, _BENCHMARK], fetch_start, BACKTEST_END)
    spy = panel.get(_BENCHMARK)
    if spy is None or spy.empty:
        raise RuntimeError("No SPY prices, so the universe comparison cannot run")
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
    Turn many selection paths into after-cost cumulative returns.

    Cost matches the monthly backtest: the one-way rate times the sum of absolute weight changes.
    The first month is bought from cash, so the cost lands on the first holding day. Later rebalance costs land on the signal day.
    A name that did not trade that day gives its weight to the names still held.

    Args:
        close: Closes.
        windows: Each holding month's window and signal day.
        paths: Each path is a sequence of name lists, in the same order as windows.

    Returns:
        Cumulative return for each path.
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
            raise RuntimeError(f"{window['signal_month']} holding days do not line up with closes")
        block = values[hold_pos]  # (days, stocks)
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
        # The first month's cost is on the first holding day. Later costs are on the signal day, which is the last day of the previous stretch.
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
    """Take the symbols actually picked each month from the baseline backtest's rebalance months."""
    by_month = {str(month["signal"])[:7]: list(month["picked"]) for month in baseline["months"]}
    missing = [window["signal_month"] for window in windows if window["signal_month"] not in by_month]
    if missing:
        raise RuntimeError("Baseline picks do not match the holding windows: " + ", ".join(missing))
    return [by_month[window["signal_month"]] for window in windows]


def _print_equal_and_random(
    close: pd.DataFrame,
    windows: list[dict[str, Any]],
    strategy_return: float,
    strategy_picks: list[list[str]],
) -> tuple[pd.Series, list[list[str]]]:
    """
    Print the equal-weight baseline and the distribution of 1000 random draws of 3 names.

    Args:
        close: Closes.
        windows: Holding windows, including each month's eligible names.
        strategy_return: The strategy's after-cost cumulative return. The official backtest is reused. Picks are not recomputed here.
        strategy_picks: The strategy's holdings each month, used to check that the cost math matches the official backtest.

    Returns:
        (equal-weight after-cost daily returns, equal-weight holdings each month). The daily returns are used by the two halves.
    """
    equal_picks = [list(window["eligible"]) for window in windows]
    counts = [len(names) for names in equal_picks]
    print("\n=== Equal-weight baseline (same names, all equal-weight at each month-end, same turnover cost) ===", flush=True)
    print(
        "Eligibility matches the strategy: a technical score that day, a close, and the minimum history. "
        f"Eligible names per month: {min(counts)}-{max(counts)}.",
        flush=True,
    )
    thin = [window["signal_month"] for window, names in zip(windows, equal_picks) if len(names) < len(close.columns)]
    if thin:
        print("These months did not put every symbol into the equal-weight book: " + ", ".join(thin), flush=True)
    checked = float(_simulate_paths(close, windows, [strategy_picks])[0])
    if abs(checked - strategy_return) > 1e-6:
        _log.warning(
            "Recomputing the strategy with the same costs gives %.4f%%, and the official backtest is %.4f%%",
            checked * 100,
            strategy_return * 100,
        )
    equal_return = float(_simulate_paths(close, windows, [equal_picks])[0])
    print(
        f"Strategy cumulative {_pct(strategy_return)}, equal-weight cumulative {_pct(equal_return)}, "
        f"gap {_pct(strategy_return - equal_return)}.",
        flush=True,
    )

    print(
        f"\n=== Random baseline (draw {_TOP_N} eligible names each month, {_RANDOM_REPS} times, seed={_RANDOM_SEED}) ===",
        flush=True,
    )
    short = [window["signal_month"] for window in windows if len(window["eligible"]) < _TOP_N]
    if short:
        print("These months have fewer than 3 eligible names, so the random baseline holds every eligible name: " + ", ".join(short), flush=True)
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
    print("Cumulative-return percentiles:", flush=True)
    for pct in _PERCENTILES:
        print(f"  {pct:>2}th  {_pct(float(np.percentile(random_returns, pct)))}", flush=True)
    rank = _percentile_rank(random_returns.tolist(), strategy_return)
    print(
        f"Strategy {_pct(strategy_return)} is at the {_num(rank)} percentile "
        f"(share of {_RANDOM_REPS} draws whose cumulative return is not above the strategy).",
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
    After-cost daily returns for one path. The method matches _simulate_paths, and the date index is kept.

    Args:
        close: Closes.
        windows: Holding windows.
        picks: The name list for each holding month.

    Returns:
        Daily returns after costs.
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
    # Both implementations are kept to confirm that compounding daily returns and the batch cumulative follow the same path.
    if abs(float(_cumulative(net)) - float(cumulative[0])) > 1e-6:
        _log.warning("Equal-weight daily compounding does not match the batch cumulative")
    return net


def _holding_price_returns(close: pd.DataFrame, hold: pd.DatetimeIndex) -> pd.Series:
    """Price return of each stock over this holding window. A name that did not trade is NaN."""
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
    Spearman correlation between the month-end score and that month's holding-period return.

    Args:
        scores: Technical scores on the signal day.
        returns: Holding-period price returns for the same stocks.

    Returns:
        (correlation, number of paired names, symbols with no pair). The correlation is NaN when fewer than 3 names are paired.
    """
    paired = pd.DataFrame({"score": scores, "ret": returns}).dropna()
    universe = sorted(set(scores.index.astype(str)) | set(returns.index.astype(str)))
    missing = [name for name in universe if name not in set(paired.index.astype(str))]
    if len(paired) < 3:
        return float("nan"), int(len(paired)), missing
    # Pearson correlation of ranks is Spearman. Ties use the average rank.
    ranked = paired.rank(method="average")
    return float(ranked["score"].corr(ranked["ret"])), int(len(paired)), missing


def _print_rank_ic(score: pd.DataFrame, close: pd.DataFrame, windows: list[dict[str, Any]]) -> None:
    """Print each month's rank IC, plus the mean, standard deviation, positive share, and t-statistic."""
    print("\n=== Rank IC (month-end technical score vs that month's holding-period return, Spearman) ===", flush=True)
    print("Uses names that have both a score and a return that month, not only the 3 that were picked.", flush=True)
    header = f"{'Month':<10}{'Names':>6}{'IC':>8}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    correlations: list[float] = []
    for window in windows:
        scores = score.loc[window["signal"]]
        returns = _holding_price_returns(close, window["hold"])
        ic, used, missing = _spearman(scores, returns)
        print(f"{window['signal_month']:<10}{used:>6}{_num(ic):>8}", flush=True)
        if missing:
            print(f"           left out: {', '.join(missing)}", flush=True)
        if math.isfinite(ic):
            correlations.append(ic)
    if len(correlations) < 2:
        print("Fewer than 2 usable months, so the IC standard deviation and t-statistic cannot be computed.", flush=True)
        return
    values = np.array(correlations, dtype=float)
    mean_ic = float(values.mean())
    std_ic = float(values.std(ddof=1))
    stderr = std_ic / math.sqrt(len(values))
    t_stat = mean_ic / stderr if stderr > 0 else float("nan")
    positive = float(np.mean(values > 0))
    print(
        f"Mean IC {_num(mean_ic)}, standard deviation {_num(std_ic)}, "
        f"months with IC > 0 {_pct(positive)} ({int(np.sum(values > 0))}/{len(values)}), "
        f"t-statistic {_num(t_stat)} (mean IC / standard error).",
        flush=True,
    )


def _slice_half(daily: pd.Series, windows: list[dict[str, Any]]) -> pd.Series:
    """Stitch daily returns along holding windows. The windows do not overlap."""
    parts = [daily.loc[pd.Timestamp(window["hold_start"]) : pd.Timestamp(window["hold_end"])] for window in windows]
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts)


def _print_halves(
    baseline: dict[str, Any],
    windows: list[dict[str, Any]],
    equal_daily: pd.Series,
) -> None:
    """Split 36 months into the first 18 and the last 18, and compare the strategy, equal weight, and SPY."""
    print("\n=== First and last 18 months (after costs) ===", flush=True)
    ordered = sorted(windows, key=lambda window: window["hold_start"])
    if len(ordered) != _HALF_MONTHS * 2:
        print(f"{len(ordered)} holding months, not 36. The two halves are still split by time.", flush=True)
    first = ordered[:_HALF_MONTHS]
    second = ordered[_HALF_MONTHS : _HALF_MONTHS * 2]
    strategy = baseline["net_returns"]
    spy = baseline["spy_returns"]
    header = f"{'Segment':<28}{'Strategy':>10}{'Strat Sharpe':>14}{'Equal wt':>10}{'EW Sharpe':>12}{'SPY':>10}{'SPY Sharpe':>12}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for title, group in ((f"{first[0]['hold_start'][:7]} to {first[-1]['hold_end'][:7]}", first), (f"{second[0]['hold_start'][:7]} to {second[-1]['hold_end'][:7]}", second)):
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
    Compare equal weight, random picks, rank IC, and the two halves on the fixed window.

    Args:
        tickers: Group A symbols.
        baseline: The current baseline monthly backtest. Strategy return uses this path. The selection rule is not changed here.
    """
    pool = _load_pool(tickers)
    windows = pool["windows"]
    if len(windows) != len(baseline["months"]):
        _log.warning(
            "Universe holding months %s, baseline backtest %s",
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
    """Run the rolling start months, monthly-return concentration, and the universe comparison."""
    # The backtest logs one pick line per month, and 25 runs would flood the table. Failures are still logged as warnings by this module.
    logging.getLogger("backtest_compare_groups").setLevel(logging.WARNING)
    logging.getLogger("backtest_data").setLevel(logging.WARNING)
    logging.getLogger("backtest_universe_sp500").setLevel(logging.WARNING)
    groups = define_three_groups()
    tickers = groups["A"]
    print(f"Group A ({len(tickers)} names)", flush=True)
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
            f"\nThe current baseline {BACKTEST_FIRST_REBALANCE_MONTH} has no complete result, so monthly-return concentration was not computed.",
            flush=True,
        )
        return
    _print_concentration(baseline)
    run_pool_benchmarks(tickers, baseline)


if __name__ == "__main__":
    main()
