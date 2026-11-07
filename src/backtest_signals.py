# -*- coding: utf-8 -*-
"""
Technical-only signals for backtests: a 0-100 technical total for every trading day.

This module uses only price and volume indicators (RSI, MACD, Bollinger, moving averages, volume).
It does not touch fundamentals, news, or valuation, so a historical backtest does not pick up lookahead bias.

The score on day t uses only data through day t:
- The indicators come from pandas rolling and ewm (causal; they do not include the future).
- The score rules follow the if-else logic in indicators.py, rewritten as a vectorized calculation over the whole series.
- A MACD golden cross or death cross looks only at the previous day via shift(1). shift(-1) is never used.

Run from the project root:
    python src/backtest_signals.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_SRC = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import TECH_SCORE_WEIGHTS  # noqa: E402
from indicators import calculate_indicators, calculate_tech_score  # noqa: E402

_log = logging.getLogger("backtest_signals")
if not _log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# Backtests always use the pandas indicators, so a short series does not flip between ta and the manual path and create a false positive.
_USE_TA_FOR_BACKTEST = False

# Tolerance when comparing scores. round(2) after the weights can still leave a tiny numeric error.
_SCORE_ATOL = 1e-6


def _as_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return an indicator column as a float Series. A missing column becomes all NaN."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").astype(float)


def _score_rsi_series(rsi: pd.Series) -> pd.Series:
    """
    Vectorized RSI subscore (0-100). The rules match indicators._score_rsi.

    Args:
        rsi: The RSI series for each day.

    Returns:
        A score series on the same index as rsi. A missing RSI gets the neutral score of 50.
    """
    # np.select takes the first condition that is true. Comparisons with NaN are False, and valid overwrites those rows at the end.
    chosen = np.select(
        [
            rsi < 30,
            rsi < 40,
            rsi <= 60,
            rsi <= 70,
        ],
        [85.0, 75.0, 70.0, 60.0],
        default=40.0,
    )
    return pd.Series(np.where(rsi.notna(), chosen, 50.0), index=rsi.index, dtype=float)


def _score_macd_series(
    dif: pd.Series,
    dea: pd.Series,
    hist: pd.Series,
) -> pd.Series:
    """
    Vectorized MACD subscore. The rules, including priority, match indicators._score_macd.

    A golden cross, a death cross, and an expanding histogram use only the previous day (shift 1). They do not read the future.

    Args:
        dif: MACD DIF.
        dea: MACD DEA (the signal line).
        hist: MACD histogram.

    Returns:
        A score series on the same index as dif. If DIF, DEA, or HIST is NaN, the score is 50.
    """
    dif_prev = dif.shift(1)
    dea_prev = dea.shift(1)
    hist_prev = hist.shift(1)

    valid = dif.notna() & dea.notna() & hist.notna()
    prev_ok = dif_prev.notna() & dea_prev.notna()
    hist_prev_ok = hist_prev.notna()

    golden = prev_ok & (dif_prev <= dea_prev) & (dif > dea)
    death = prev_ok & (dif_prev >= dea_prev) & (dif < dea)
    bull = (dif > dea) & (hist > 0)
    bear = (dif < dea) & (hist < 0)
    hist_expand = hist_prev_ok & (hist > hist_prev)
    above_zero_hist = (dif > 0) & hist_expand
    hist_rising = hist_expand

    chosen = np.select(
        [
            golden,
            death,
            bull,
            above_zero_hist,
            bear,
            (dif > 0) & hist_rising,
            (dif > 0) & ~hist_rising,
            (dif < 0) & hist_rising,
        ],
        [90.0, 30.0, 75.0, 70.0, 40.0, 65.0, 55.0, 50.0],
        default=45.0,
    )
    return pd.Series(np.where(valid, chosen, 50.0), index=dif.index, dtype=float)


def _score_bollinger_series(
    close: pd.Series,
    upper: pd.Series,
    mid: pd.Series,
    lower: pd.Series,
) -> pd.Series:
    """
    Vectorized Bollinger subscore. The rules match indicators._score_bollinger.

    Args:
        close: Close.
        upper: Upper Bollinger band.
        mid: Middle Bollinger band.
        lower: Lower Bollinger band.

    Returns:
        A score series on the same index as close. If any input is NaN, the score is 50.
    """
    valid = close.notna() & upper.notna() & mid.notna() & lower.notna()
    chosen = np.select(
        [
            close <= lower,
            close > upper,
            (close > mid) & (close < upper),
        ],
        [85.0, 50.0, 70.0],
        default=45.0,
    )
    return pd.Series(np.where(valid, chosen, 50.0), index=close.index, dtype=float)


def _score_ma_trend_series(
    close: pd.Series,
    ma5: pd.Series,
    ma20: pd.Series,
    ma60: pd.Series,
    ma200: pd.Series,
) -> pd.Series:
    """
    Vectorized moving-average stack subscore. The rules match indicators._score_ma_trend.

    Args:
        close: Close.
        ma5 / ma20 / ma60 / ma200: Moving averages for each window.

    Returns:
        A score series on the same index as close. Missing close, MA20, or MA60 gets a score of 50.
    """
    base_ok = close.notna() & ma20.notna() & ma60.notna()
    ma5_ok = ma5.notna()
    ma200_ok = ma200.notna()

    full_bull = ma5_ok & ma200_ok & (ma5 > ma20) & (ma20 > ma60) & (ma60 > ma200)
    partial_bull = ma200_ok & (ma20 > ma60) & (ma60 > ma200)
    full_bear = ma5_ok & ma200_ok & (ma5 < ma20) & (ma20 < ma60) & (ma60 < ma200)
    above_20_60 = (close > ma20) & (close > ma60)
    below_20_above_60 = (close < ma20) & (close > ma60)

    chosen = np.select(
        [
            full_bull,
            partial_bull,
            full_bear,
            above_20_60,
            below_20_above_60,
        ],
        [90.0, 75.0, 30.0, 70.0, 50.0],
        default=55.0,
    )
    return pd.Series(np.where(base_ok, chosen, 50.0), index=close.index, dtype=float)


def _score_volume_series(
    vol_ratio: pd.Series,
    open_: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """
    Vectorized volume subscore. The rules match indicators._score_volume.

    The same day's Open and Close decide a green or red candle. That is information available on day t, not lookahead.

    Args:
        vol_ratio: That day's volume divided by the 20-day average volume.
        open_: Open.
        close: Close.

    Returns:
        A score series on the same index as close. If any input is NaN, the score is 50.
    """
    valid = vol_ratio.notna() & open_.notna() & close.notna()
    red = close > open_
    black = close < open_
    chosen = np.select(
        [
            (vol_ratio > 2.0) & black,
            (vol_ratio > 1.5) & red,
            vol_ratio < 0.5,
            (vol_ratio >= 0.8) & (vol_ratio <= 1.5),
        ],
        [35.0, 85.0, 50.0, 65.0],
        default=60.0,
    )
    return pd.Series(np.where(valid, chosen, 50.0), index=close.index, dtype=float)


def compute_tech_score_series(price_history: pd.DataFrame) -> pd.Series:
    """
    Compute that day's technical total (0-100) for every trading day and return one time series.

    The score on day t uses prices through day t inclusive. Causal indicators are computed once for the whole series,
    then the RSI, MACD, Bollinger, moving-average, and volume rules are applied as a weighted vectorized score.

    Args:
        price_history: Must include Open, High, Low, Close, and Volume. A date index is recommended.

    Returns:
        A Series on the same index as price_history, named tech_score. An empty Series when it cannot be computed.
    """
    if price_history is None or price_history.empty:
        _log.warning("price_history is empty, so the technical score series cannot be computed")
        return pd.Series(dtype=float, name="tech_score")

    try:
        df = calculate_indicators(price_history, use_ta=_USE_TA_FOR_BACKTEST)
    except Exception:
        _log.exception("Failed to compute technical indicators; returning an empty series")
        return pd.Series(dtype=float, name="tech_score")

    close = _as_float_series(df, "Close")
    open_ = _as_float_series(df, "Open")

    rsi_sc = _score_rsi_series(_as_float_series(df, "RSI"))
    macd_sc = _score_macd_series(
        _as_float_series(df, "MACD_DIF"),
        _as_float_series(df, "MACD_DEA"),
        _as_float_series(df, "MACD_HIST"),
    )
    boll_sc = _score_bollinger_series(
        close,
        _as_float_series(df, "BB_UPPER"),
        _as_float_series(df, "BB_MIDDLE"),
        _as_float_series(df, "BB_LOWER"),
    )
    ma_sc = _score_ma_trend_series(
        close,
        _as_float_series(df, "MA5"),
        _as_float_series(df, "MA20"),
        _as_float_series(df, "MA60"),
        _as_float_series(df, "MA200"),
    )
    vol_sc = _score_volume_series(_as_float_series(df, "VOL_RATIO"), open_, close)

    w = TECH_SCORE_WEIGHTS
    total = (
        rsi_sc * w["rsi"]
        + macd_sc * w["macd"]
        + boll_sc * w["bollinger"]
        + ma_sc * w["ma_trend"]
        + vol_sc * w["volume"]
    )
    total = total.round(2)
    total.name = "tech_score"
    return total


def _values_equal(a: float, b: float, atol: float) -> bool:
    """Whether two scores are equal. Both NaN counts as equal."""
    a_nan = pd.isna(a)
    b_nan = pd.isna(b)
    if a_nan and b_nan:
        return True
    if a_nan or b_nan:
        return False
    return bool(np.isclose(float(a), float(b), atol=atol, rtol=0.0))


def _format_ts(value: object) -> str:
    """Format an index value as a readable date string."""
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return str(value)
    return ts.strftime("%Y-%m-%d")


def verify_no_lookahead(
    price_history: pd.DataFrame,
    *,
    atol: float = _SCORE_ATOL,
    max_report: int = 10,
) -> bool:
    """
    Lookahead self-check: the day-t score from the full history must equal
    the last-day score when the history is truncated to day t and recomputed.

    A mismatch means day t used data from after t (lookahead bias).

    Args:
        price_history: Raw OHLCV.
        atol: Allowed absolute error on the score.
        max_report: How many mismatch locations to print at most.

    Returns:
        True when it passes (no lookahead). False when a mismatch is found or the data is insufficient.
    """
    if price_history is None or price_history.empty:
        _log.warning("Lookahead check failed: price_history is empty")
        return False

    try:
        df = price_history.sort_index()
        full_scores = compute_tech_score_series(df)
    except Exception:
        _log.exception("Lookahead check failed: scores could not be computed from the full history")
        return False

    if full_scores.empty or len(full_scores) != len(df):
        _log.warning("Lookahead check failed: the full score series has an unexpected length")
        return False

    mismatches: list[tuple[str, float, float, float]] = []
    n = len(df)

    for t in range(n):
        truncated = df.iloc[: t + 1]
        try:
            trunc_scores = compute_tech_score_series(truncated)
        except Exception as exc:
            label = _format_ts(df.index[t])
            _log.warning("Lookahead check failed: computing the history truncated to %s raised: %s", label, exc)
            return False

        if trunc_scores.empty:
            label = _format_ts(df.index[t])
            mismatches.append((label, float(full_scores.iloc[t]), float("nan"), float("nan")))
            continue

        full_val = full_scores.iloc[t]
        trunc_val = trunc_scores.iloc[-1]
        if not _values_equal(full_val, trunc_val, atol):
            f_num = float(full_val) if pd.notna(full_val) else float("nan")
            t_num = float(trunc_val) if pd.notna(trunc_val) else float("nan")
            diff = f_num - t_num
            mismatches.append((_format_ts(df.index[t]), f_num, t_num, diff))

    if not mismatches:
        _log.info("Lookahead check passed: scores match on all %s trading days", n)
        return True

    _log.warning(
        "Lookahead check failed: day-t scores disagree in %s of %s places (full history vs truncated to t)",
        len(mismatches),
        n,
    )
    for label, full_val, trunc_val, diff in mismatches[:max_report]:
        _log.warning(
            "  date %s: full=%.4f, truncated=%.4f, diff=%.4f",
            label,
            full_val,
            trunc_val,
            diff,
        )
    if len(mismatches) > max_report:
        _log.warning("  ... %s more rows omitted", len(mismatches) - max_report)
    return False


def _print_last_scores(scores: pd.Series, n: int = 10) -> None:
    """Print the technical score for the last n days."""
    tail = scores.tail(n)
    print("=== Technical score for the last 10 trading days ===")
    print(f"{'Date':<12} {'tech_score':>10}")
    for idx, val in tail.items():
        print(f"{_format_ts(idx):<12} {float(val):>10.2f}")


def _check_matches_last_day_api(price_history: pd.DataFrame, scores: pd.Series) -> bool:
    """
    Extra check: the last day of the series should match calculate_tech_score on the same manual indicators.

    This is not a lookahead check. It confirms the vectorized score did not drift from the existing rules.
    """
    try:
        df_ind = calculate_indicators(price_history, use_ta=_USE_TA_FOR_BACKTEST)
        last = calculate_tech_score(df_ind)
        expected = float(last["total_score"])
        got = float(scores.iloc[-1])
    except Exception:
        _log.exception("An exception occurred while comparing with calculate_tech_score")
        return False

    ok = _values_equal(expected, got, atol=0.01)
    if ok:
        _log.info("Matches calculate_tech_score on the last day: %.2f", got)
    else:
        _log.warning(
            "Does not match calculate_tech_score on the last day: series=%.2f, existing API=%.2f",
            got,
            expected,
        )
    return ok


if __name__ == "__main__":
    from data_collector import fetch_stock_data

    raw = fetch_stock_data("NVDA", force_refresh=False)
    if not raw:
        print("Failed to fetch NVDA data. Check the network or try again later.")
        sys.exit(1)

    ph = raw["price_history"]
    print(
        f"NVDA price rows: {len(ph)} ({ _format_ts(ph.index[0]) } to { _format_ts(ph.index[-1]) })",
        flush=True,
    )

    scores = compute_tech_score_series(ph)
    if scores.empty:
        print("The technical score series is empty. Test stopped.", flush=True)
        sys.exit(1)

    _print_last_scores(scores, n=10)
    print(
        f"Score stats: min={float(scores.min()):.2f}  "
        f"median={float(scores.median()):.2f}  "
        f"max={float(scores.max()):.2f}",
        flush=True,
    )

    last_ok = _check_matches_last_day_api(ph, scores)
    print("Compare with the existing calculate_tech_score:", "passed" if last_ok else "failed", flush=True)

    print("Starting the lookahead check (every trading day is truncated and recomputed; this can take a few seconds)...", flush=True)
    passed = verify_no_lookahead(ph)
    print("verify_no_lookahead:", "passed" if passed else "failed", flush=True)

    sys.exit(0 if passed and last_ok else 1)
