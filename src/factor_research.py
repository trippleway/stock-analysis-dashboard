# -*- coding: utf-8 -*-
"""
Factor research for Group A: monthly rank IC only. No backtest and no stock picking.

The window is the fixed config window (2023-09 through 2026-09). The universe is Group A.
Each month-end factor value is Spearman-correlated with the holding-period return through the next month-end.
All 34 names go into the cross-section. A name is dropped from that month only when it has no factor or no return that day.

The five existing subscores call the scoring functions in indicators.py, each on its own, with no weights.
The five added factors use only close and volume. Windows are trading days, and they look backward only.

The same subscores are then combined five ways, and only the composite rank IC is compared.
That is an ex-post comparison on one sample, used to check the sign of RSI, not to pick a new weight set.

Lookahead check: truncate the data at the signal month-end and recompute. The value must match the full series on that same month-end.
Any mismatch stops the run. IC is not printed.

Run from the project root:
    python src/factor_research.py
"""

from __future__ import annotations

import logging
import math
import sys
import warnings
from datetime import timedelta
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
    _as_timestamp_index,
    _month_end_signals,
    define_three_groups,
)
from backtest_data import build_ohlcv_panel  # noqa: E402
from backtest_robustness import _holding_price_returns, _spearman  # noqa: E402
from config import BACKTEST_END, BACKTEST_START, TECH_SCORE_WEIGHTS  # noqa: E402
from indicators import (  # noqa: E402
    _score_bollinger,
    _score_macd,
    _score_ma_trend,
    _score_rsi,
    _score_volume,
    calculate_indicators,
    calculate_tech_score,
)

_log = logging.getLogger("factor_research")
if not _log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# The calendar matches the existing rank IC: SPY trading days are the month-ends and holding windows.
_CALENDAR = "SPY"
# 252 trading days is about a year. Extra history is fetched so the first signal month's long window can be computed.
# This length is not a knob for the IC.
_WARMUP_DAYS = 450
# Windows from the spec. Each is in that stock's own trading days.
_LOOKBACK_DAYS = 252
_SKIP_DAYS = 21
_VOL_DAYS = 60
_VOLUME_SHORT = 20
_VOLUME_LONG = 60
# Tolerance when a truncated recompute is compared with the full series. A real lookahead is a price tick, far larger than this.
_LOOKAHEAD_ATOL = 1e-8
# calculate_tech_score rounds to two decimals. A one-bucket gap (5 points or more) is still caught.
_API_ATOL = 0.011
_GROUP_SIZE = 34
_MIN_CORR_MONTHS = 3

# Tables and the correlation matrix use this order. The IC table is re-sorted by |t| before printing.
_FACTORS: list[tuple[str, str]] = [
    ("rsi", "RSI subscore"),
    ("macd", "MACD subscore"),
    ("bollinger", "Bollinger subscore"),
    ("ma_trend", "Moving-average subscore"),
    ("volume", "Volume subscore"),
    ("mom_12_1", "12-1 momentum"),
    ("reversal_21", "Short-term reversal"),
    ("low_vol_60", "Low volatility"),
    ("high_52w", "52-week high proximity"),
    ("volume_trend", "Volume trend"),
]
_SHORT = {
    "rsi": "RSI",
    "macd": "MACD",
    "bollinger": "Bollinger",
    "ma_trend": "MA",
    "volume": "Volume",
    "mom_12_1": "12-1",
    "reversal_21": "Reversal",
    "low_vol_60": "Low vol",
    "high_52w": "52w high",
    "volume_trend": "Vol trend",
}
_SUBSCORE_KEYS = ("rsi", "macd", "bollinger", "ma_trend", "volume")


def _prepare_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize to timezone-naive OHLCV with one row per day.

    Args:
        frame: Must include Open, High, Low, Close, and Volume.

    Returns:
        A copy sorted by date. The last row is kept when a day is duplicated.
    """
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Price frame is missing columns: {sorted(missing)}")
    out = frame.loc[:, ["Open", "High", "Low", "Close", "Volume"]].copy()
    out.index = _as_timestamp_index(out.index)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _score_position(ind: pd.DataFrame, i: int) -> tuple[dict[str, float], dict[str, bool]]:
    """
    Score one trading day with the five scoring functions in indicators.py.

    The score looks only at today and the previous trading day, with no weights.
    When an indicator is not ready, the scorer returns a neutral 50. That value is kept,
    and the missing input is recorded so a real 50 can be told apart from a missing indicator.

    Args:
        ind: Sorted result of calculate_indicators.
        i: Row position.

    Returns:
        (five subscores, whether each input is missing).
    """
    row = ind.iloc[i]
    prev = ind.iloc[i - 1] if i > 0 else None

    def cur(col: str) -> float | None:
        if col not in ind.columns:
            return None
        value = row[col]
        if pd.isna(value):
            return None
        return float(value)

    def prv(col: str) -> float | None:
        if prev is None or col not in ind.columns:
            return None
        value = prev[col]
        if pd.isna(value):
            return None
        return float(value)

    rsi = cur("RSI")
    dif, dea, hist = cur("MACD_DIF"), cur("MACD_DEA"), cur("MACD_HIST")
    close = cur("Close")
    upper, mid, lower = cur("BB_UPPER"), cur("BB_MIDDLE"), cur("BB_LOWER")
    ma5, ma20, ma60, ma200 = cur("MA5"), cur("MA20"), cur("MA60"), cur("MA200")
    vol_ratio, open_ = cur("VOL_RATIO"), cur("Open")

    rsi_sc, _ = _score_rsi(rsi, prv("RSI"))
    macd_sc, _ = _score_macd(dif, dea, hist, prv("MACD_DIF"), prv("MACD_DEA"), prv("MACD_HIST"))
    boll_sc, _ = _score_bollinger(close, upper, mid, lower)
    ma_sc, _ = _score_ma_trend(close, ma5, ma20, ma60, ma200)
    vol_sc, _ = _score_volume(vol_ratio, open_, close)

    scores = {
        "rsi": float(rsi_sc),
        "macd": float(macd_sc),
        "bollinger": float(boll_sc),
        "ma_trend": float(ma_sc),
        "volume": float(vol_sc),
    }
    missing = {
        "rsi": rsi is None,
        "macd": dif is None or dea is None or hist is None,
        "bollinger": close is None or upper is None or mid is None or lower is None,
        "ma_trend": close is None or ma20 is None or ma60 is None,
        "volume": vol_ratio is None or open_ is None or close is None,
    }
    return scores, missing


def _price_factors(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the five new factors from close and volume only. rolling and shift look only at today and earlier.

    Args:
        frame: Prepared OHLCV. The index is that stock's own trading days.

    Returns:
        Five columns on the same index as frame. NaN when the window is short or a divisor is 0.
    """
    close = frame["Close"].astype(float)
    volume = frame["Volume"].astype(float)
    # 12-1: hold from 252 trading days ago through 21 trading days ago. The most recent 21 days are not in this return.
    momentum = close.shift(_SKIP_DAYS) / close.shift(_LOOKBACK_DAYS) - 1.0
    # Simple return over the past 21 days. The expected IC is negative. The sign is not flipped and the absolute value is not taken.
    reversal = close / close.shift(_SKIP_DAYS) - 1.0
    # A larger number means higher volatility. This is the standard deviation itself, not the inverse of "low volatility".
    low_vol = close.pct_change().rolling(_VOL_DAYS, min_periods=_VOL_DAYS).std(ddof=1)
    # Includes today. On a new high the ratio is 1.
    near_high = close / close.rolling(_LOOKBACK_DAYS, min_periods=_LOOKBACK_DAYS).max()
    vol_trend = (
        volume.rolling(_VOLUME_SHORT, min_periods=_VOLUME_SHORT).mean()
        / volume.rolling(_VOLUME_LONG, min_periods=_VOLUME_LONG).mean()
    )
    out = pd.DataFrame(
        {
            "mom_12_1": momentum,
            "reversal_21": reversal,
            "low_vol_60": low_vol,
            "high_52w": near_high,
            "volume_trend": vol_trend,
        },
        index=frame.index,
    )
    return out.replace([np.inf, -np.inf], np.nan)


def compute_factors(
    frame: pd.DataFrame,
    score_dates: pd.DatetimeIndex | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute every factor for one stock.

    Args:
        frame: Raw or prepared OHLCV. Only data on or before each index date is used.
        score_dates: Dates on which to compute subscores. None means every trading day.
            Price factors are always recomputed on the full slice so the truncation check can match that day.

    Returns:
        (factor table, whether each subscore input is missing). Factor columns follow _FACTORS.
        The missing table has only the five subscore columns. True means the scorer received an empty input and returned a neutral 50.
    """
    prepared = _prepare_ohlcv(frame)
    prices = _price_factors(prepared)
    ind = calculate_indicators(prepared, use_ta=False).dropna(how="all")
    if score_dates is None:
        positions = range(len(ind))
    else:
        wanted = set(_as_timestamp_index(pd.DatetimeIndex(score_dates)))
        positions = [i for i, ts in enumerate(ind.index) if ts in wanted]

    score_rows: dict[pd.Timestamp, dict[str, float]] = {}
    missing_rows: dict[pd.Timestamp, dict[str, bool]] = {}
    for i in positions:
        scores, missing = _score_position(ind, i)
        stamp = pd.Timestamp(ind.index[i])
        score_rows[stamp] = scores
        missing_rows[stamp] = missing

    scores = pd.DataFrame.from_dict(score_rows, orient="index")
    missing = pd.DataFrame.from_dict(missing_rows, orient="index")
    out = prices.join(scores, how="left")
    for key in _SUBSCORE_KEYS:
        if key not in out.columns:
            out[key] = np.nan
        if key not in missing.columns:
            missing[key] = False
    columns = [key for key, _name in _FACTORS]
    out = out.loc[:, columns].replace([np.inf, -np.inf], np.nan)
    missing = missing.reindex(columns=list(_SUBSCORE_KEYS)).fillna(False)
    return out, missing.astype(bool)


def _values_equal(left: float, right: float) -> bool:
    """Whether two factor values match. NaN on both sides counts as a match."""
    left_nan = pd.isna(left)
    right_nan = pd.isna(right)
    if left_nan and right_nan:
        return True
    if left_nan or right_nan:
        return False
    return bool(np.isclose(float(left), float(right), rtol=0.0, atol=_LOOKAHEAD_ATOL))


def _value_at(frame: pd.DataFrame, stamp: pd.Timestamp, column: str) -> float:
    """Read one factor on one trading day. Returns NaN when that day has no row or the value is not finite."""
    if stamp not in frame.index:
        return float("nan")
    value = frame.at[stamp, column]
    if pd.isna(value) or not math.isfinite(float(value)):
        return float("nan")
    return float(value)


def _api_inputs_complete(ind: pd.DataFrame) -> bool:
    """Whether the last row's indicators are complete. When they are, calculate_tech_score does not reach back for an earlier valid value."""
    needed = (
        "RSI",
        "MACD_DIF",
        "MACD_DEA",
        "MACD_HIST",
        "BB_UPPER",
        "BB_MIDDLE",
        "BB_LOWER",
        "MA5",
        "MA20",
        "MA60",
        "MA200",
        "VOL_RATIO",
        "Open",
        "Close",
    )
    if ind.empty:
        return False
    last = ind.iloc[-1]
    return all(col in ind.columns and pd.notna(last[col]) for col in needed)


def _check_indicator_api(
    histories: dict[str, pd.DataFrame],
    factors: dict[str, pd.DataFrame],
) -> bool:
    """
    Confirm the subscores really are the five pieces inside indicators.calculate_tech_score.

    Only stocks whose last row has a complete indicator set are checked. For those names
    _last_scalar is the last row, so an earlier value is not mixed in. Any mismatch is treated as a wiring error.

    Args:
        histories: Prepared prices.
        factors: Factor tables from compute_factors. Each name's last trading day must already be included.

    Returns:
        True when at least one name was checked and every name that could be checked matched.
    """
    checked = 0
    for ticker, history in histories.items():
        try:
            ind = calculate_indicators(history, use_ta=False)
            if not _api_inputs_complete(ind):
                _log.warning("%s last row is missing indicators, so the calculate_tech_score check is skipped", ticker)
                continue
            api = calculate_tech_score(ind)
            last_day = history.index[-1]
        except Exception:
            _log.exception("%s raised while being compared with calculate_tech_score", ticker)
            return False
        checked += 1
        for key in _SUBSCORE_KEYS:
            got = _value_at(factors[ticker], pd.Timestamp(last_day), key)
            expected = float(api["sub_scores"][key])
            if not _values_equal(got, expected) and not (
                math.isfinite(got) and math.isfinite(expected) and abs(got - expected) <= _API_ATOL
            ):
                _log.warning(
                    "%s %s does not match calculate_tech_score: research=%s, existing API=%s",
                    ticker,
                    key,
                    got,
                    expected,
                )
                return False
    if checked == 0:
        _log.warning("No name has a complete last-row indicator set, so the subscores cannot be confirmed against indicators.py")
        return False
    _log.info("Subscores match calculate_tech_score: checked %s names", checked)
    return True


def verify_no_lookahead(
    histories: dict[str, pd.DataFrame],
    factors: dict[str, pd.DataFrame],
    signals: list[pd.Timestamp],
) -> list[dict[str, Any]]:
    """
    Lookahead check. Every factor, every signal month, and every stock is compared once.

    The month-end value for month t from the full series must equal the value recomputed on data kept only through that month-end.
    NaN on both sides passes (the window is too short, and neither side should invent a number).

    Args:
        histories: Prepared prices, including warmup and the research window.
        factors: Factor tables computed on the full series.
        signals: Signal days, earliest to latest.

    Returns:
        The list of mismatches. An empty list means the check passed.
    """
    mismatches: list[dict[str, Any]] = []
    for pos, signal in enumerate(signals, start=1):
        stamp = pd.Timestamp(signal)
        _log.info("Lookahead check %s / %s: %s", pos, len(signals), stamp.date().isoformat())
        for ticker, history in histories.items():
            truncated = history.loc[history.index <= stamp]
            if not truncated.empty and truncated.index.max() > stamp:
                raise RuntimeError(f"{ticker} still has data after the truncation at {stamp.date()}")
            try:
                recomputed, _missing = compute_factors(truncated, score_dates=pd.DatetimeIndex([stamp]))
            except Exception as exc:
                mismatches.append(
                    {
                        "ticker": ticker,
                        "month": stamp.strftime("%Y-%m"),
                        "factor": "(truncated recompute failed)",
                        "full": float("nan"),
                        "truncated": float("nan"),
                        "detail": str(exc),
                    }
                )
                continue
            full = factors[ticker]
            for key, label in _FACTORS:
                full_value = _value_at(full, stamp, key)
                trunc_value = _value_at(recomputed, stamp, key)
                if _values_equal(full_value, trunc_value):
                    continue
                mismatches.append(
                    {
                        "ticker": ticker,
                        "month": stamp.strftime("%Y-%m"),
                        "factor": label,
                        "full": full_value,
                        "truncated": trunc_value,
                        "detail": "",
                    }
                )
    return mismatches


def _holding_windows(index: pd.DatetimeIndex) -> list[dict[str, Any]]:
    """
    Reuse the signal days and holding windows from the monthly rank IC.

    A signal day is the last trading day of each month inside the performance window. The final trading day itself is excluded.
    The hold starts the trading day after the signal and runs through the next signal (or through the last day of the window for the last stretch).

    Args:
        index: Master calendar, here SPY's trading days.

    Returns:
        One row per month, with signal, signal_month, and hold.
    """
    signals = _month_end_signals(index, BACKTEST_START, BACKTEST_END)
    windows: list[dict[str, Any]] = []
    for pos, signal_day in enumerate(signals):
        stamp = pd.Timestamp(signal_day)
        if stamp not in index:
            continue
        nxt = signals[pos + 1] if pos + 1 < len(signals) else index[-1]
        hold = index[(index > stamp) & (index <= pd.Timestamp(nxt))]
        if len(hold) == 0:
            continue
        windows.append(
            {
                "signal": stamp,
                "signal_month": stamp.strftime("%Y-%m"),
                "hold": hold,
            }
        )
    return windows


def _cross_section(
    factors: dict[str, pd.DataFrame],
    tickers: list[str],
    stamp: pd.Timestamp,
    column: str,
) -> pd.Series:
    """One day's cross-section for one factor. The index is every stock, and missing values are NaN."""
    values = {ticker: _value_at(factors[ticker], stamp, column) for ticker in tickers}
    return pd.Series(values, dtype=float)


def _monthly_ic(
    factors: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    windows: list[dict[str, Any]],
    tickers: list[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Spearman rank IC for every factor and every signal month.

    The algorithm matches backtest_robustness._spearman: average ranks, then Pearson.
    Holding-period returns match backtest_robustness._holding_price_returns.

    Args:
        factors: Factor table for each name.
        close: Closes aligned to the master calendar.
        windows: Signal days and holding windows.
        tickers: Every Group A symbol.

    Returns:
        (IC table, rows are signal months and columns are factor ids, plus months that did not use all 34 names or whose IC is undefined).
    """
    ic = pd.DataFrame(
        index=[window["signal_month"] for window in windows],
        columns=[key for key, _name in _FACTORS],
        dtype=float,
    )
    shortfalls: list[dict[str, Any]] = []
    for window in windows:
        returns = _holding_price_returns(close, window["hold"]).reindex(tickers)
        for key, _label in _FACTORS:
            scores = _cross_section(factors, tickers, window["signal"], key)
            # When the cross-section is constant, Spearman's denominator is 0 and the correlation is undefined.
            # The numpy warning is suppressed. The value stays NaN. No substitute IC is filled in.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                value, used, missing = _spearman(scores, returns)
            ic.loc[window["signal_month"], key] = value
            reason = ""
            if not math.isfinite(value):
                paired = pd.DataFrame({"score": scores, "ret": returns}).dropna()
                if len(paired) < 3:
                    reason = "fewer than 3 usable names"
                elif paired["score"].nunique(dropna=True) < 2 or paired["ret"].nunique(dropna=True) < 2:
                    reason = "no cross-sectional variation"
                else:
                    reason = "correlation is undefined"
            if used != len(tickers) or reason:
                shortfalls.append(
                    {
                        "factor": key,
                        "month": window["signal_month"],
                        "used": used,
                        "missing": missing,
                        "reason": reason,
                    }
                )
    ic.index.name = "signal_month"
    return ic, shortfalls


def _ic_stats(series: pd.Series) -> dict[str, float | int]:
    """
    Mean, sample standard deviation, t-statistic, and positive share of monthly IC.

    t = mean IC / (standard deviation / sqrt(usable months)). A usable month is one whose IC is finite.
    With fewer than two months, the standard deviation and t are NaN.

    Args:
        series: Monthly IC for one factor.

    Returns:
        mean, std, t, positive_ratio, n, and n_positive.
    """
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    n = int(len(values))
    if n == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "t": float("nan"),
            "positive_ratio": float("nan"),
            "n": 0,
            "n_positive": 0,
        }
    mean_ic = float(values.mean())
    positive = float(np.mean(values > 0))
    n_positive = int(np.sum(values > 0))
    if n < 2:
        return {
            "mean": mean_ic,
            "std": float("nan"),
            "t": float("nan"),
            "positive_ratio": positive,
            "n": n,
            "n_positive": n_positive,
        }
    std_ic = float(values.std(ddof=1))
    stderr = std_ic / math.sqrt(n)
    t_stat = mean_ic / stderr if stderr > 0 else float("nan")
    return {
        "mean": mean_ic,
        "std": std_ic,
        "t": t_stat,
        "positive_ratio": positive,
        "n": n,
        "n_positive": n_positive,
    }


def _fmt_num(value: float, digits: int) -> str:
    """Format a finite value to a fixed number of decimals. A non-finite value is an em dash."""
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _fmt_ratio(ratio: float, n_positive: int, n: int) -> str:
    """Share of months with IC > 0, with the numerator and denominator."""
    if n <= 0 or not math.isfinite(ratio):
        return "—"
    return f"{ratio * 100:.1f}% ({n_positive}/{n})"


def _print_definitions() -> None:
    """Print the factor definitions. Signs are shown as specified, not flipped to make IC positive."""
    print("Factor definitions (trading-day windows, using only data through today)", flush=True)
    lines = [
        "RSI / MACD / Bollinger / moving average / volume: subscores from indicators.py, each on its own, unweighted",
        "12-1 momentum: close[t-21] / close[t-252] - 1",
        "Short-term reversal: close[t] / close[t-21] - 1 (no absolute value, no sign flip)",
        "Low volatility: sample standard deviation of simple returns over the past 60 trading days (larger means more volatile; not inverted)",
        "52-week high proximity: today's close / highest close over the past 252 trading days (including today)",
        "Volume trend: 20-day average volume / 60-day average volume",
    ]
    for line in lines:
        print(f"  {line}", flush=True)


def _print_ic_table(ic: pd.DataFrame) -> pd.DataFrame:
    """
    Print one IC summary row per factor, sorted by the absolute t-statistic, largest first.

    Args:
        ic: Monthly IC. Rows are signal months and columns are factor ids.

    Returns:
        The sorted statistics table. Values stay float so the caller can keep using them.
    """
    rows: list[dict[str, Any]] = []
    for key, label in _FACTORS:
        stats = _ic_stats(ic[key])
        t_value = float(stats["t"])
        rows.append(
            {
                "key": key,
                "Factor": label,
                "Mean IC": float(stats["mean"]),
                "IC Std Dev": float(stats["std"]),
                "t-stat": t_value,
                "abs_t": abs(t_value) if math.isfinite(t_value) else -1.0,
                "% Months IC>0": _fmt_ratio(
                    float(stats["positive_ratio"]),
                    int(stats["n_positive"]),
                    int(stats["n"]),
                ),
                "Valid Months": int(stats["n"]),
            }
        )
    rows.sort(key=lambda row: (-float(row["abs_t"]), str(row["Factor"])))
    display = pd.DataFrame(
        {
            "Factor": [row["Factor"] for row in rows],
            "Mean IC": [_fmt_num(row["Mean IC"], 3) for row in rows],
            "IC Std Dev": [_fmt_num(row["IC Std Dev"], 3) for row in rows],
            "t-stat": [_fmt_num(row["t-stat"], 2) for row in rows],
            "% Months IC>0": [row["% Months IC>0"] for row in rows],
            "Valid Months": [row["Valid Months"] for row in rows],
        }
    )
    print("\n=== Monthly rank IC (Spearman, sorted by |t|) ===", flush=True)
    print(
        "t = mean IC / (IC sample standard deviation / sqrt(usable months)). The IC>0 denominator is the usable-month count.",
        flush=True,
    )
    print(display.to_string(index=False), flush=True)
    return pd.DataFrame(rows)


def _print_correlation(ic: pd.DataFrame) -> None:
    """
    Pearson correlation between factors, using each factor's monthly IC series.

    The matrix shows which factors' predictive power shows up together, not the cross-sectional correlation of the factor values themselves.
    Each pair uses only months where both ICs exist. Fewer than 3 overlapping months is left blank.

    Args:
        ic: Monthly IC.
    """
    corr = ic.astype(float).corr(method="pearson", min_periods=_MIN_CORR_MONTHS)
    shown = corr.rename(index=_SHORT, columns=_SHORT)
    print("\n=== Pearson correlation of monthly factor IC series ===", flush=True)
    print("Rows and columns are monthly factor IC, not the stocks' factor values. The diagonal is 1.", flush=True)
    print(shown.to_string(float_format=lambda value: f"{value:6.2f}", na_rep="   —"), flush=True)

    pairs: list[tuple[float, float, str, str]] = []
    keys = [key for key, _label in _FACTORS]
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            value = corr.loc[left, right]
            if pd.isna(value) or not math.isfinite(float(value)):
                continue
            pairs.append((abs(float(value)), float(value), _SHORT[left], _SHORT[right]))
    pairs.sort(key=lambda item: (-item[0], item[2], item[3]))
    print("\nFactor pairs with the highest absolute correlation", flush=True)
    if not pairs:
        print("Not enough overlapping months to compute a correlation.", flush=True)
        return
    for _abs_value, value, left, right in pairs[:5]:
        print(f"  {left} vs {right}  {value:+.2f}", flush=True)


def _print_coverage(shortfalls: list[dict[str, Any]], n_months: int) -> None:
    """
    Say which factors did not include all 34 names in some month, and which months have an undefined IC.

    Args:
        shortfalls: Months that were short of the full sample, or whose correlation is undefined.
        n_months: Number of signal months.
    """
    print("\n=== Cross-section coverage ===", flush=True)
    if not shortfalls:
        print(f"Every factor included all {_GROUP_SIZE} names in all {n_months} months.", flush=True)
        return
    print(
        f"The target is {_GROUP_SIZE} names each month. The months below are missing a factor value or a holding-period return, so that month's IC uses only the names that are complete.",
        flush=True,
    )
    by_factor: dict[str, list[dict[str, Any]]] = {key: [] for key, _label in _FACTORS}
    for row in shortfalls:
        by_factor[str(row["factor"])].append(row)
    labels = dict(_FACTORS)
    for key, _label in _FACTORS:
        rows = by_factor[key]
        thin = [row for row in rows if int(row["used"]) != _GROUP_SIZE]
        undefined = [row for row in rows if row.get("reason")]
        if not thin and not undefined:
            print(f"{labels[key]}: all {n_months} months have {_GROUP_SIZE} names", flush=True)
            continue
        if thin:
            absent: dict[str, int] = {}
            for row in thin:
                for ticker in row["missing"]:
                    absent[str(ticker)] = absent.get(str(ticker), 0) + 1
            detail = ", ".join(f"{ticker} missing {count} months" for ticker, count in sorted(absent.items()))
            print(
                f"{labels[key]}: {len(thin)} months below {_GROUP_SIZE} names ({detail})",
                flush=True,
            )
        else:
            print(f"{labels[key]}: all {n_months} months have {_GROUP_SIZE} names", flush=True)
        for row in undefined:
            print(
                f"  {row['month']} IC is undefined ({row['reason']}, {row['used']} names that month)",
                flush=True,
            )


def _base_tech_weights() -> dict[str, float]:
    """
    Read config.TECH_SCORE_WEIGHTS, in the fixed order of the five subscores.

    Returns:
        Subscore id to weight. A missing key is rejected so a term is not silently dropped.
    """
    missing = [key for key in _SUBSCORE_KEYS if key not in TECH_SCORE_WEIGHTS]
    if missing:
        raise KeyError("TECH_SCORE_WEIGHTS is missing: " + ", ".join(missing))
    return {key: float(TECH_SCORE_WEIGHTS[key]) for key in _SUBSCORE_KEYS}


def _renormalize(base: dict[str, float], keep: tuple[str, ...]) -> dict[str, float]:
    """
    Keep the named subscores and rescale their original relative weights so they sum to 1.

    Args:
        base: Original weights from config.
        keep: Subscore ids to keep.

    Returns:
        A new weight dict that still has all five keys. Dropped keys are 0.
    """
    total = sum(base[key] for key in keep)
    if total == 0:
        raise ValueError("The weights being kept sum to 0, so they cannot be normalized")
    return {key: (base[key] / total if key in keep else 0.0) for key in _SUBSCORE_KEYS}


def _composite_schemes() -> list[dict[str, Any]]:
    """
    Weights for the five composite scores. The scoring rule does not change. Only the weights do.

    The RSI-flipped case sets the RSI weight negative and leaves the other four at the config values, without renormalizing.
    Bollinger-and-moving-average only is the specified 0.50 each, not a rescaling of the original shares.

    Returns:
        One row per scheme, with the name, the weights, and how those weights were built.
        The looked-up names stay Baseline, Drop RSI, and Invert RSI.
    """
    base = _base_tech_weights()
    flipped = dict(base)
    flipped["rsi"] = -base["rsi"]
    trend_only = {key: 0.0 for key in _SUBSCORE_KEYS}
    trend_only["bollinger"] = 0.5
    trend_only["ma_trend"] = 0.5
    return [
        {"name": "Baseline", "weights": base, "note": "config.TECH_SCORE_WEIGHTS"},
        {
            "name": "Drop RSI",
            "weights": _renormalize(base, ("macd", "bollinger", "ma_trend", "volume")),
            "note": "the other four renormalized to their original shares",
        },
        {"name": "Invert RSI", "weights": flipped, "note": "RSI weight flipped negative, the rest unchanged"},
        {
            "name": "Drop RSI and MACD",
            "weights": _renormalize(base, ("bollinger", "ma_trend", "volume")),
            "note": "the other three renormalized to their original shares",
        },
        {"name": "Bollinger and moving average only", "weights": trend_only, "note": "0.50 each"},
    ]


def _fmt_weight(value: float) -> str:
    """Signed weight. Extra zeros are stripped, and at least two decimals are kept."""
    text = f"{value:+.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".00"
    else:
        decimals = text.split(".", 1)[1]
        if len(decimals) < 2:
            text += "0" * (2 - len(decimals))
    return text


def _fmt_weight_set(weights: dict[str, float]) -> str:
    """Print the five weights on one line. A zero weight is still shown so it does not look omitted."""
    labels = {"rsi": "RSI", "macd": "MACD", "bollinger": "Bollinger", "ma_trend": "MA", "volume": "Volume"}
    return ", ".join(f"{labels[key]} {_fmt_weight(weights[key])}" for key in _SUBSCORE_KEYS)


def _composite_cross_section(
    factors: dict[str, pd.DataFrame],
    tickers: list[str],
    stamp: pd.Timestamp,
    weights: dict[str, float],
) -> pd.Series:
    """
    One day's cross-section of the composite score.

    If any subscore with a non-zero weight is missing, that name is NaN. A neutral score is not filled in.
    A subscore that is already the scorer's returned 50 is used as-is and is not rewritten.

    Args:
        factors: Factor table for each name.
        tickers: Every Group A symbol.
        stamp: Signal day.
        weights: Weights for the five subscores. A weight may be negative.

    Returns:
        Composite score indexed by every stock. Missing values are NaN.
    """
    values: dict[str, float] = {}
    for ticker in tickers:
        total = 0.0
        complete = True
        for key, weight in weights.items():
            if weight == 0.0:
                continue
            value = _value_at(factors[ticker], stamp, key)
            if not math.isfinite(value):
                complete = False
                break
            total += weight * value
        values[ticker] = total if complete else float("nan")
    return pd.Series(values, dtype=float)


def _composite_monthly_ic(
    factors: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    windows: list[dict[str, Any]],
    tickers: list[str],
    weights: dict[str, float],
) -> pd.Series:
    """
    Monthly rank IC for one composite score. The method matches a single factor.

    Args:
        factors: Factor table for each name.
        close: Closes aligned to the master calendar.
        windows: Signal days and holding windows.
        tickers: Every Group A symbol.
        weights: Weights for this composite.

    Returns:
        IC indexed by signal month. A month with no cross-sectional variation is NaN.
    """
    values: dict[str, float] = {}
    for window in windows:
        returns = _holding_price_returns(close, window["hold"]).reindex(tickers)
        scores = _composite_cross_section(factors, tickers, window["signal"], weights)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ic, _used, _missing = _spearman(scores, returns)
        values[window["signal_month"]] = ic
    return pd.Series(values, dtype=float, name="ic")


def _hypothesis_sentence(rows: list[dict[str, Any]]) -> str:
    """
    Describe whether this comparison supports the hypothesis, using the direction of mean IC versus Baseline.

    The hypothesis is that RSI, with its current sign, is dragging the composite down.
    Support means dropping RSI, and flipping the RSI weight negative, both raise mean IC above Baseline.
    Only the direction is described.

    Args:
        rows: Statistics for the five schemes. The first row must be Baseline.

    Returns:
        One sentence.
    """
    current = float(rows[0]["mean"])
    by_name = {str(row["name"]): row for row in rows}
    drop = by_name["Drop RSI"]
    flip = by_name["Invert RSI"]
    drop_higher = float(drop["mean"]) > current
    flip_higher = float(flip["mean"]) > current
    if drop_higher and flip_higher:
        return "Dropping RSI, and flipping the RSI weight negative, both raise mean IC above Baseline. The direction matches the idea that the current RSI subscore is dragging the composite down."
    if not drop_higher and not flip_higher:
        return "Dropping RSI, and flipping the RSI weight negative, both fail to raise mean IC above Baseline. The direction does not match that hypothesis."
    if drop_higher:
        return "Dropping RSI raises mean IC above Baseline, but flipping the RSI weight negative does not. The direction only partly matches the hypothesis."
    return "Flipping the RSI weight negative raises mean IC above Baseline, but dropping RSI does not. The direction only partly matches the hypothesis."


def _print_composite_comparison(
    factors: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    windows: list[dict[str, Any]],
    tickers: list[str],
) -> None:
    """
    Print rank IC for the five composites. The order is fixed and is not re-sorted by IC.

    Args:
        factors: Factor table for each name. Subscores have already passed the lookahead check.
        close: Closes aligned to the master calendar.
        windows: Signal days and holding windows.
        tickers: Every Group A symbol.
    """
    schemes = _composite_schemes()
    print("\n=== Composite-weight comparison (same subscores, rank IC only) ===", flush=True)
    print(
        "The hypothesis is that the RSI subscore has the wrong sign and is dragging the composite down. The table shows the direction of the change versus Baseline.",
        flush=True,
    )
    print(f"config.TECH_SCORE_WEIGHTS: {_fmt_weight_set(schemes[0]['weights'])}", flush=True)
    print("The composite is a weighted sum of subscores. The scoring rule was not changed.", flush=True)
    for scheme in schemes:
        print(f"  {scheme['name']} ({scheme['note']}): {_fmt_weight_set(scheme['weights'])}", flush=True)

    rows: list[dict[str, Any]] = []
    for scheme in schemes:
        series = _composite_monthly_ic(factors, close, windows, tickers, scheme["weights"])
        stats = _ic_stats(series)
        rows.append({"name": scheme["name"], "mean": float(stats["mean"]), **stats})
    current_mean = float(rows[0]["mean"])
    display = pd.DataFrame(
        {
            "Variant": [row["name"] for row in rows],
            "Mean IC": [_fmt_num(float(row["mean"]), 3) for row in rows],
            "IC Std Dev": [_fmt_num(float(row["std"]), 3) for row in rows],
            "t-stat": [_fmt_num(float(row["t"]), 2) for row in rows],
            "% Months IC>0": [
                _fmt_ratio(float(row["positive_ratio"]), int(row["n_positive"]), int(row["n"]))
                for row in rows
            ],
            "Mean IC − Baseline": [
                "—" if row["name"] == "Baseline" else _fmt_num(float(row["mean"]) - current_mean, 3)
                for row in rows
            ],
        }
    )
    print(display.to_string(index=False), flush=True)
    print(_hypothesis_sentence(rows), flush=True)
    print(
        "These 5 schemes are an ex-post comparison on the same sample. A higher IC is not evidence that the weight is valid.",
        flush=True,
    )


def _print_placeholder_scores(
    missing_inputs: dict[str, pd.DataFrame],
    windows: list[dict[str, Any]],
) -> None:
    """
    A subscore is a neutral 50 when its indicator is not ready yet. List those stock-months.

    Args:
        missing_inputs: Missing-input table for each name.
        windows: Signal months. Only these month-ends are counted, not the warmup.
    """
    counts: dict[str, dict[str, int]] = {key: {} for key in _SUBSCORE_KEYS}
    for ticker, frame in missing_inputs.items():
        if frame.empty:
            continue
        for window in windows:
            stamp = window["signal"]
            if stamp not in frame.index:
                continue
            for key in _SUBSCORE_KEYS:
                if bool(frame.at[stamp, key]):
                    counts[key][ticker] = counts[key].get(ticker, 0) + 1
    labels = dict(_FACTORS)
    print("\n=== Missing subscore inputs (the scorer returned a neutral 50) ===", flush=True)
    if not any(counts.values()):
        print("No signal month has this case.", flush=True)
        return
    for key in _SUBSCORE_KEYS:
        if not counts[key]:
            continue
        detail = ", ".join(f"{ticker} {count} months" for ticker, count in sorted(counts[key].items()))
        total = sum(counts[key].values())
        print(f"  {labels[key]}: {total} stock-months ({detail})", flush=True)


def _load_histories(tickers: list[str]) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex, pd.DataFrame]:
    """
    Load Group A and SPY prices and build the same close panel the existing rank IC uses.

    Args:
        tickers: Group A symbols.

    Returns:
        (prepared OHLCV for each name, the SPY calendar, closes aligned to that calendar).
        Raises RuntimeError when any name is missing or SPY is missing.
    """
    fetch_start = BACKTEST_START - timedelta(days=_WARMUP_DAYS)
    panel = build_ohlcv_panel([*tickers, _CALENDAR], fetch_start, BACKTEST_END)
    spy = panel.get(_CALENDAR)
    if spy is None or spy.empty:
        raise RuntimeError("No SPY prices, so the existing month-end calendar cannot be reused")
    missing = [ticker for ticker in tickers if ticker not in panel or panel[ticker] is None or panel[ticker].empty]
    if missing:
        raise RuntimeError("These symbols have no prices, so all 34 names cannot be included: " + ", ".join(missing))

    master = _as_timestamp_index(spy.index)
    master = master[(master >= pd.Timestamp(fetch_start)) & (master <= pd.Timestamp(BACKTEST_END))]
    histories: dict[str, pd.DataFrame] = {}
    closes: dict[str, pd.Series] = {}
    for ticker in tickers:
        history = _prepare_ohlcv(panel[ticker])
        histories[ticker] = history
        closes[ticker] = history["Close"].reindex(master)
    close = pd.DataFrame(closes, index=master)
    return histories, master, close


def run_factor_research() -> int:
    """
    Run Group A's factor IC, after the lookahead check.

    Returns:
        0 when the check passed and the tables were printed. 1 when the run stopped.
    """
    logging.getLogger("backtest_data").setLevel(logging.WARNING)
    logging.getLogger("backtest_universe_sp500").setLevel(logging.WARNING)
    logging.getLogger("backtest_compare_groups").setLevel(logging.WARNING)

    try:
        groups = define_three_groups()
        tickers = list(groups["A"])
    except Exception:
        _log.exception("Group A list failed to build")
        print("Group A list failed to build. Factor research stopped.", flush=True)
        return 1

    print(f"Group A ({len(tickers)} names): {', '.join(tickers)}", flush=True)
    if len(tickers) != _GROUP_SIZE:
        print(f"Group A is not {_GROUP_SIZE} names. Factor research stopped.", flush=True)
        return 1

    try:
        histories, master, close = _load_histories(tickers)
    except Exception as exc:
        _log.exception("Price load failed")
        print(f"Price load failed. Factor research stopped: {exc}", flush=True)
        return 1

    windows = _holding_windows(master)
    if not windows:
        print("There are no signal months. Factor research stopped.", flush=True)
        return 1
    signals = [window["signal"] for window in windows]
    # The subscore check uses each name's last trading day, so that day is added to the score dates.
    score_dates = pd.DatetimeIndex([*signals, *(history.index[-1] for history in histories.values())])

    print(
        f"Fixed window {BACKTEST_START.isoformat()} to {BACKTEST_END.isoformat()}. "
        f"Signal months {windows[0]['signal_month']} to {windows[-1]['signal_month']}, {len(windows)} months.",
        flush=True,
    )
    print(
        f"First hold {windows[0]['hold'][0].date().isoformat()} to {windows[0]['hold'][-1].date().isoformat()}; "
        f"last hold {windows[-1]['hold'][0].date().isoformat()} to {windows[-1]['hold'][-1].date().isoformat()}.",
        flush=True,
    )
    _print_definitions()

    factors: dict[str, pd.DataFrame] = {}
    missing_inputs: dict[str, pd.DataFrame] = {}
    for ticker, history in histories.items():
        try:
            factor_frame, missing = compute_factors(history, score_dates=score_dates)
        except Exception:
            _log.exception("%s factor calculation failed", ticker)
            print(f"{ticker} factor calculation failed. Factor research stopped.", flush=True)
            return 1
        factors[ticker] = factor_frame
        missing_inputs[ticker] = missing

    if not _check_indicator_api(histories, factors):
        print("Subscores do not match indicators.calculate_tech_score. Factor research stopped.", flush=True)
        return 1

    print("\nStarting the lookahead check: each signal month is recomputed on data kept only through that month-end.", flush=True)
    try:
        mismatches = verify_no_lookahead(histories, factors, signals)
    except Exception:
        _log.exception("The lookahead check raised an exception")
        print("The lookahead check raised an exception. Factor research stopped.", flush=True)
        return 1
    if mismatches:
        print(
            f"Lookahead check failed: {len(mismatches)} mismatches. Factor research stopped. IC is not computed.",
            flush=True,
        )
        for row in mismatches[:12]:
            detail = f"，{row['detail']}" if row["detail"] else ""
            print(
                f"  {row['factor']}  {row['month']}  {row['ticker']}"
                f"  full={row['full']}  truncated to that month={row['truncated']}{detail}",
                flush=True,
            )
        if len(mismatches) > 12:
            print(f"  ...{len(mismatches) - 12} more rows omitted", flush=True)
        return 1

    finite_counts = {
        key: int(sum(math.isfinite(_value_at(frame, signal, key)) for frame in factors.values() for signal in signals))
        for key, _label in _FACTORS
    }
    empty = [label for key, label in _FACTORS if finite_counts[key] == 0]
    if empty:
        print("These factors have no value in any signal month. Factor research stopped: " + ", ".join(empty), flush=True)
        return 1
    print(
        f"Lookahead check passed: {len(_FACTORS)} factors x {len(signals)} signal months x {len(tickers)} names, "
        "truncated-to-month matches the full series.",
        flush=True,
    )

    ic, shortfalls = _monthly_ic(factors, close, windows, tickers)
    _print_ic_table(ic)
    _print_correlation(ic)
    _print_coverage(shortfalls, len(windows))
    _print_placeholder_scores(missing_inputs, windows)
    try:
        _print_composite_comparison(factors, close, windows, tickers)
    except Exception:
        _log.exception("Composite-weight comparison failed")
        print("Composite-weight comparison failed. Single-factor IC was already printed. This comparison stopped.", flush=True)
        return 1
    return 0


def main() -> None:
    """Run the factor research. Exit code 1 when a check fails."""
    sys.exit(run_factor_research())


if __name__ == "__main__":
    main()
