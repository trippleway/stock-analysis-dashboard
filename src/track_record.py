# -*- coding: utf-8 -*-
"""
Strategy performance tracking: record each day's top picks, fill in realized returns, and summarize performance versus SPY.

Run from the project root:
    python src/track_record.py
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from config import CACHE_DIR, DATA_DIR, TRACK_RECORD_DIR  # noqa: E402
except Exception:  # noqa: BLE001
    DATA_DIR = _ROOT / "data"  # type: ignore[misc, assignment]
    CACHE_DIR = Path(DATA_DIR) / "cache"  # type: ignore[misc, assignment]
    TRACK_RECORD_DIR = Path(DATA_DIR) / "track_record"  # type: ignore[misc, assignment]

TRACK_DIR = Path(TRACK_RECORD_DIR)
TRACK_DIR.mkdir(parents=True, exist_ok=True)

RECOMMENDATIONS_CSV = TRACK_DIR / "recommendations.csv"
PERFORMANCE_JSON = TRACK_DIR / "performance.json"

# Cumulative-return sanity cap: an absolute value above 100% is suspicious (usually double compounding or bad data)
CUMULATIVE_SANITY_LIMIT = 1.0
# Benchmark sanity cap: a SPY move of more than 25% in 90 trading days is extremely rare
BENCHMARK_SANITY_LIMIT = 0.25
BENCHMARK_SANITY_DAYS = 90

# Holding period in calendar days for each recommendation, matching the definition of return_1m
HOLDING_DAYS = 30

# How many days a cache may lag the requested end (absorbs weekends and holidays)
CACHE_COVERAGE_TOLERANCE_DAYS = 4
# Seconds to wait between yfinance calls
YF_CALL_SLEEP_SEC = 0.6
# A single name is treated as over-concentrated when its contribution exceeds this share of total excess return
CONCENTRATION_WARN_SHARE = 0.5

_log = logging.getLogger("track_record")
if not _log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

CSV_COLUMNS: list[str] = [
    "date",
    "ticker",
    "sector",
    "top_rank",
    "entry_price",
    "spy_price_entry",
    "composite_score",
    "tech_score",
    "fund_score",
    "val_score",
    "pred_score",
    "news_score",
    "target_price_30d",
    "upside_pct_expected",
    "fair_value",
    "valuation_signal",
    "actual_price_1d",
    "actual_price_1w",
    "actual_price_2w",
    "actual_price_1m",
    "return_1d",
    "return_1w",
    "return_2w",
    "return_1m",
    "spy_price_1m",
    "spy_return_1m",
    "alpha_1m",
    "status",
]

# Holding-day threshold -> (price column, return column)
_HORIZON_SPECS: list[tuple[int, str, str]] = [
    (1, "actual_price_1d", "return_1d"),
    (7, "actual_price_1w", "return_1w"),
    (14, "actual_price_2w", "return_2w"),
    (30, "actual_price_1m", "return_1m"),
]


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_date(value: Any) -> date | None:
    """Parse several date formats and return a timezone-naive date."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "nat"):
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s[:19] if "T" in s or " " in s else s[:10], fmt).date()
        except ValueError:
            continue
    try:
        ts = pd.to_datetime(s, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.date()
    except Exception:  # noqa: BLE001
        return None


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def _ensure_csv_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing columns and enforce the column order."""
    out = df.copy() if df is not None else pd.DataFrame(columns=CSV_COLUMNS)
    for col in CSV_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[CSV_COLUMNS]


def _load_recommendations() -> pd.DataFrame:
    if not RECOMMENDATIONS_CSV.exists():
        return pd.DataFrame(columns=CSV_COLUMNS)
    try:
        df = pd.read_csv(RECOMMENDATIONS_CSV)
        return _ensure_csv_schema(df)
    except Exception as e:  # noqa: BLE001
        _log.warning("Failed to read recommendations.csv: %s", e)
        return pd.DataFrame(columns=CSV_COLUMNS)


def _save_recommendations(df: pd.DataFrame) -> None:
    try:
        TRACK_DIR.mkdir(parents=True, exist_ok=True)
        _ensure_csv_schema(df).to_csv(RECOMMENDATIONS_CSV, index=False, encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        _log.warning("Failed to write recommendations.csv: %s", e)


def _index_as_dates(index: Any) -> list[date | None]:
    """Turn a DatetimeIndex (including timezone-aware) into a list of plain dates."""
    out: list[date | None] = []
    for d in index:
        try:
            ts = pd.Timestamp(d)
            out.append(ts.date())
        except Exception:  # noqa: BLE001
            out.append(_parse_date(d))
    return out


def _history_close_on_or_after(hist: pd.DataFrame, target: date) -> float | None:
    """Close on target or the next trading day. Earlier days are not used, so a forward return is not filled by mistake."""
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None
    try:
        closes = pd.to_numeric(hist["Close"], errors="coerce").dropna()
        if closes.empty:
            return None
        dates = _index_as_dates(closes.index)
        for i, d0 in enumerate(dates):
            if d0 is not None and d0 >= target:
                return float(closes.iloc[i])
        return None
    except Exception as e:  # noqa: BLE001
        _log.debug("Failed to parse close target=%s: %s", target, e)
        return None


def _history_close_on_or_before(hist: pd.DataFrame, target: date) -> float | None:
    """Close on target or the previous trading day. Used for the SPY comparison on the entry day."""
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None
    try:
        closes = pd.to_numeric(hist["Close"], errors="coerce").dropna()
        if closes.empty:
            return None
        dates = _index_as_dates(closes.index)
        last_i = None
        for i, d0 in enumerate(dates):
            if d0 is not None and d0 <= target:
                last_i = i
            elif d0 is not None and d0 > target:
                break
        if last_i is not None:
            return float(closes.iloc[last_i])
        return None
    except Exception as e:  # noqa: BLE001
        _log.debug("Failed to parse close (on_or_before) target=%s: %s", target, e)
        return None


def _load_local_cache_history(ticker: str) -> pd.DataFrame:
    """
    Load the newest price cache from data/cache/YYYY-MM-DD/{TICKER}.parquet.
    It does not depend on today's folder, so offline backfill still works.
    """
    sym = str(ticker).strip().upper()
    cache_root = Path(CACHE_DIR)
    if not cache_root.is_dir():
        return pd.DataFrame()
    candidates: list[Path] = []
    try:
        for day_dir in cache_root.iterdir():
            if not day_dir.is_dir():
                continue
            pq = day_dir / f"{sym}.parquet"
            if pq.is_file():
                candidates.append(pq)
    except Exception as e:  # noqa: BLE001
        _log.debug("Failed to scan the local cache %s: %s", sym, e)
        return pd.DataFrame()
    if not candidates:
        return pd.DataFrame()
    candidates.sort(key=lambda p: p.parent.name)
    latest = candidates[-1]
    try:
        df = pd.read_parquet(latest)
        if df is None or df.empty:
            return pd.DataFrame()
        _log.debug("Using local cache %s <- %s", sym, latest)
        return df
    except Exception as e:  # noqa: BLE001
        _log.warning("Failed to read local parquet %s: %s", latest, e)
        return pd.DataFrame()


def _fetch_yfinance_history(ticker: str, start: date, end: date) -> pd.DataFrame:
    """
    Fetch daily bars for [start, end] with yfinance (end is plus one day so the end date is included). Retry 3 times on failure.

    Args:
        ticker: Symbol.
        start: First date, inclusive.
        end: Last date, inclusive.
    Returns:
        A daily DataFrame. An empty DataFrame when every attempt fails.
    """
    sym = str(ticker).strip().upper()
    end_excl = end + timedelta(days=1)
    delays = (1.0, 2.0, 4.0)
    for attempt in range(3):
        try:
            time.sleep(YF_CALL_SLEEP_SEC)  # Pause so back-to-back calls do not trigger 429.
            t = yf.Ticker(sym)
            hist = t.history(start=start.isoformat(), end=end_excl.isoformat(), auto_adjust=True)
            if hist is not None and not hist.empty:
                return hist
            _log.debug("yfinance history has no data %s [%s ~ %s] (attempt %s)", sym, start, end, attempt + 1)
        except Exception as e:  # noqa: BLE001
            _log.warning(
                "yfinance history failed %s [%s ~ %s] (attempt %s): %s", sym, start, end, attempt + 1, e
            )
        if attempt < 2:
            time.sleep(delays[attempt])
    return pd.DataFrame()


def _history_max_date(hist: pd.DataFrame) -> date | None:
    """Latest date in a history frame. None when it cannot be parsed."""
    if hist is None or hist.empty:
        return None
    dates = [d0 for d0 in _index_as_dates(hist.index) if d0 is not None]
    return max(dates) if dates else None


def _cache_covers(cache_max: date, end: date) -> bool:
    """
    Whether the cache already covers the requested end date.

    When end falls on a weekend or holiday there is no new trading-day bar, so the check steps back to the latest weekday
    and allows CACHE_COVERAGE_TOLERANCE_DAYS of lag, so the cache is not treated as stale every time.

    Args:
        cache_max: Date of the newest row in the cache.
        end: Date the request needs to cover, inclusive.
    Returns:
        True when the cache is new enough to use as-is.
    """
    target = min(end, date.today())
    while target.weekday() >= 5:  # Saturday=5, Sunday=6
        target -= timedelta(days=1)
    return cache_max >= target - timedelta(days=CACHE_COVERAGE_TOLERANCE_DAYS)


def _update_cache_history(ticker: str, hist: pd.DataFrame) -> None:
    """
    Write newly fetched daily bars into today's cache directory so later calls do not refetch them.

    Only {TICKER}.parquet is written. data_collector also requires _info.json before it treats a cache as valid,
    so this write does not change that module's own cache decision.
    """
    sym = str(ticker).strip().upper()
    try:
        day_dir = Path(CACHE_DIR) / date.today().isoformat()
        day_dir.mkdir(parents=True, exist_ok=True)
        hist.to_parquet(day_dir / f"{sym}.parquet", index=True)
        _log.debug("Updated price cache %s -> %s", sym, day_dir)
    except Exception as e:  # noqa: BLE001
        _log.warning("Failed to update the price cache %s: %s", sym, e)


def _fetch_history(ticker: str, start: date, end: date) -> pd.DataFrame:
    """
    Daily bars for [start, end]. Use the cache when it covers the request. Otherwise fetch from yfinance.

    If the cache only reaches an earlier date (for example the last collection was a month ago), reusing it
    would leave new recommendations without a full 30 days of prices. Coverage is checked, then the cache is refetched and updated.

    Args:
        ticker: Symbol.
        start: First date, inclusive.
        end: Date that must be covered, inclusive.
    Returns:
        A daily DataFrame. An empty DataFrame when both sources fail.
    """
    sym = str(ticker).strip().upper()
    local = _load_local_cache_history(sym)
    local_max = _history_max_date(local)

    if local_max is not None and _cache_covers(local_max, end):
        return local

    if local_max is not None:
        _log.info("Cache %s only reaches %s and does not cover %s, fetching yfinance", sym, local_max, end)
    fresh = _fetch_yfinance_history(sym, start, end)
    if not fresh.empty:
        _update_cache_history(sym, fresh)
        return fresh

    if not local.empty:
        _log.warning(
            "yfinance fetch failed, using the truncated local cache %s: data only through %s (requested through %s). "
            "Returns after that date cannot be filled. The next run will try again",
            sym,
            local_max,
            end,
        )
    return local


def record_daily_picks(picks_result: dict[str, Any]) -> None:
    """Append today's top picks to recommendations.csv. The same day and ticker are not written twice."""
    if not isinstance(picks_result, dict):
        _log.warning("record_daily_picks: picks_result is not a dict, skipped")
        return

    day = _parse_date(picks_result.get("date")) or date.today()
    day_str = day.isoformat()
    market = picks_result.get("market_overview") if isinstance(picks_result.get("market_overview"), dict) else {}
    spy_entry = _to_float(market.get("spy_price"))
    picks = picks_result.get("picks") if isinstance(picks_result.get("picks"), list) else []

    existing = _load_recommendations()
    existing_keys: set[tuple[str, str]] = set()
    if not existing.empty:
        for _, row in existing.iterrows():
            d0 = _parse_date(row.get("date"))
            t0 = str(row.get("ticker") or "").strip().upper()
            if d0 and t0:
                existing_keys.add((d0.isoformat(), t0))

    new_rows: list[dict[str, Any]] = []
    for i, p in enumerate(picks, start=1):
        if not isinstance(p, dict):
            continue
        ticker = str(p.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        if (day_str, ticker) in existing_keys:
            _log.info("Record already exists, skipped: %s %s", day_str, ticker)
            continue

        tech = p.get("technical") if isinstance(p.get("technical"), dict) else {}
        fund = p.get("fundamental") if isinstance(p.get("fundamental"), dict) else {}
        val = p.get("valuation") if isinstance(p.get("valuation"), dict) else {}
        pred = p.get("prediction") if isinstance(p.get("prediction"), dict) else {}
        news = p.get("news") if isinstance(p.get("news"), dict) else {}

        entry_px = _to_float(p.get("current_price"))
        row = {
            "date": day_str,
            "ticker": ticker,
            "sector": p.get("sector") or "",
            "top_rank": int(i),
            "entry_price": entry_px if entry_px is not None else pd.NA,
            "spy_price_entry": spy_entry if spy_entry is not None else pd.NA,
            "composite_score": _to_float(p.get("composite_score")),
            "tech_score": _to_float(tech.get("total_score")),
            "fund_score": _to_float(fund.get("total_score")),
            "val_score": _to_float(val.get("score")),
            "pred_score": _to_float(pred.get("score")),
            "news_score": _to_float(news.get("score")),
            "target_price_30d": _to_float(pred.get("predicted_30d")),
            "upside_pct_expected": _to_float(pred.get("expected_return_pct")),
            "fair_value": _to_float(val.get("fair_value")),
            "valuation_signal": val.get("valuation_signal") or "",
            "actual_price_1d": pd.NA,
            "actual_price_1w": pd.NA,
            "actual_price_2w": pd.NA,
            "actual_price_1m": pd.NA,
            "return_1d": pd.NA,
            "return_1w": pd.NA,
            "return_2w": pd.NA,
            "return_1m": pd.NA,
            "spy_price_1m": pd.NA,
            "spy_return_1m": pd.NA,
            "alpha_1m": pd.NA,
            "status": "open",
        }
        new_rows.append(row)
        existing_keys.add((day_str, ticker))

    if not new_rows:
        _log.info("record_daily_picks: no new rows to write (%s)", day_str)
        return

    add_df = _ensure_csv_schema(pd.DataFrame(new_rows))
    if existing.empty:
        merged = add_df
    else:
        merged = pd.concat([_ensure_csv_schema(existing), add_df], ignore_index=True)
    _save_recommendations(merged)
    _log.info("record_daily_picks: added %s rows (%s) -> %s", len(new_rows), day_str, RECOMMENDATIONS_CSV)


def update_actuals(days_lookback: int = 45) -> dict[str, Any]:
    """Fill in realized closes and returns for recommendations inside the last days_lookback days."""
    stats: dict[str, Any] = {
        "total_records": 0,
        "updated_1d": 0,
        "updated_1w": 0,
        "updated_2w": 0,
        "updated_1m": 0,
        "errors": [],
    }
    df = _load_recommendations()
    if df.empty:
        _log.info("update_actuals: no recommendations yet")
        return stats

    today = date.today()
    lookback = max(1, int(days_lookback))
    cutoff = today - timedelta(days=lookback)
    hard_floor = today - timedelta(days=max(lookback, 400))
    stats["total_records"] = int(len(df))

    # Prefetch each ticker and SPY once, over a window long enough for every row.
    hist_cache: dict[str, pd.DataFrame] = {}
    fetch_start_global = hard_floor - timedelta(days=10)

    def get_hist(sym: str) -> pd.DataFrame:
        key = str(sym).upper()
        if key in hist_cache:
            return hist_cache[key]
        h = _fetch_history(key, fetch_start_global, today)
        hist_cache[key] = h
        return h

    for idx, row in df.iterrows():
        try:
            entry_d = _parse_date(row.get("date"))
            if entry_d is None:
                continue
            incomplete = _is_empty(row.get("return_1m"))
            # Always process rows inside the lookback. Outside it, only fill rows that are not yet completed, and not before hard_floor.
            if entry_d < cutoff and (not incomplete or entry_d < hard_floor):
                continue

            ticker = str(row.get("ticker") or "").strip().upper()
            entry_px = _to_float(row.get("entry_price"))
            if not ticker or entry_px is None or entry_px <= 0:
                continue

            elapsed = (today - entry_d).days
            if elapsed < 1:
                continue

            hist = get_hist(ticker)
            if hist.empty:
                stats["errors"].append(f"{ticker}@{entry_d}: no price history")
                continue

            for days_n, price_col, ret_col in _HORIZON_SPECS:
                if elapsed < days_n:
                    continue
                if not _is_empty(df.at[idx, price_col]):
                    continue
                target_d = entry_d + timedelta(days=days_n)
                px = _history_close_on_or_after(hist, target_d)
                if px is None:
                    continue
                df.at[idx, price_col] = round(px, 4)
                df.at[idx, ret_col] = round((px / entry_px) - 1.0, 6)
                if days_n == 1:
                    stats["updated_1d"] += 1
                elif days_n == 7:
                    stats["updated_1w"] += 1
                elif days_n == 14:
                    stats["updated_2w"] += 1
                elif days_n == 30:
                    stats["updated_1m"] += 1

            # 1M: fill SPY and alpha, and backfill a missing spy_price_entry when needed.
            if elapsed >= 30 and (
                _is_empty(df.at[idx, "spy_price_1m"]) or _is_empty(df.at[idx, "alpha_1m"])
            ):
                spy_entry = _to_float(row.get("spy_price_entry"))
                spy_hist = get_hist("SPY")
                if spy_entry is None or spy_entry <= 0:
                    spy_entry = _history_close_on_or_before(spy_hist, entry_d)
                    if spy_entry is None:
                        spy_entry = _history_close_on_or_after(spy_hist, entry_d)
                    if spy_entry is not None:
                        df.at[idx, "spy_price_entry"] = round(spy_entry, 4)
                spy_1m = _history_close_on_or_after(spy_hist, entry_d + timedelta(days=30))
                if spy_1m is not None:
                    df.at[idx, "spy_price_1m"] = round(spy_1m, 4)
                    if spy_entry is not None and spy_entry > 0:
                        spy_ret = (spy_1m / spy_entry) - 1.0
                        df.at[idx, "spy_return_1m"] = round(spy_ret, 6)
                        r1m = _to_float(df.at[idx, "return_1m"])
                        if r1m is not None:
                            df.at[idx, "alpha_1m"] = round(r1m - spy_ret, 6)

            # status
            if not _is_empty(df.at[idx, "return_1m"]):
                df.at[idx, "status"] = "completed_1m"
            elif not _is_empty(df.at[idx, "return_2w"]):
                df.at[idx, "status"] = "partial_2w"
            elif not _is_empty(df.at[idx, "return_1w"]):
                df.at[idx, "status"] = "partial_1w"
            elif not _is_empty(df.at[idx, "return_1d"]):
                df.at[idx, "status"] = "partial_1d"
            else:
                df.at[idx, "status"] = "open"

            _log.debug("Checked %s %s", entry_d, ticker)

        except Exception as e:  # noqa: BLE001
            msg = f"row {idx} update failed: {e}"
            _log.warning(msg)
            stats["errors"].append(msg)

    _save_recommendations(df)
    _log.info(
        "update_actuals done: 1d=%s 1w=%s 2w=%s 1m=%s errors=%s",
        stats["updated_1d"],
        stats["updated_1w"],
        stats["updated_2w"],
        stats["updated_1m"],
        len(stats["errors"]),
    )
    return stats


def _safe_mean(series: pd.Series) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.mean())


def _safe_median(series: pd.Series) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.median())


def _safe_std(series: pd.Series) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return None
    return float(s.std(ddof=1))


def _warn_if_implausible(label: str, cum_return: float | None, trading_days: int) -> None:
    """
    Sanity check on cumulative return. Log a warning when the absolute value exceeds CUMULATIVE_SANITY_LIMIT.

    Args:
        label: Label used in the message, for example "Strategy" or "SPY".
        cum_return: Cumulative return as a fraction, for example 0.35 = +35%.
        trading_days: Number of trading days covered by the equity curve.
    Returns:
        None. A warning is logged when the value looks implausible.
    """
    if cum_return is None:
        return
    if abs(cum_return) > CUMULATIVE_SANITY_LIMIT:
        _log.warning(
            "%s cumulative return %+.2f%% over %s trading days has an absolute value above %.0f%%, which may be a calculation error: "
            "check whether overlapping holding periods were compounded as if they were sequential, or whether the raw prices are bad",
            label,
            cum_return * 100,
            trading_days,
            CUMULATIVE_SANITY_LIMIT * 100,
        )


def _warn_if_benchmark_implausible(spy_cum: float | None, trading_days: int) -> None:
    """
    Sanity check on the SPY benchmark. A very large move over a short window means the window length or the prices are wrong.

    Args:
        spy_cum: SPY cumulative return as a fraction.
        trading_days: Number of trading days covered by the equity curve.
    Returns:
        None
    """
    if spy_cum is None:
        return
    if abs(spy_cum) > BENCHMARK_SANITY_LIMIT and trading_days < BENCHMARK_SANITY_DAYS:
        _log.warning(
            "SPY cumulative return %+.2f%% was reached in only %s trading days, above the %.0f%% bar: "
            "a SPY move of more than %.0f%% inside three months is extremely rare. Check whether the window length was stretched by mistake",
            spy_cum * 100,
            trading_days,
            BENCHMARK_SANITY_LIMIT * 100,
            BENCHMARK_SANITY_LIMIT * 100,
        )


def _daily_close_series(ticker: str, start: date, end: date) -> pd.Series | None:
    """
    Daily close series for one symbol over [start, end].

    Args:
        ticker: Symbol.
        start: First date, inclusive.
        end: Last date, inclusive.
    Returns:
        A Series indexed by date, values are closes, sorted and de-duplicated. None when there is no data.
    """
    hist = _fetch_history(ticker, start, end)
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None
    try:
        closes = pd.to_numeric(hist["Close"], errors="coerce").dropna()
        if closes.empty:
            return None
        s = pd.Series(closes.to_numpy(dtype=float), index=_index_as_dates(closes.index))
        s = s[[d0 is not None for d0 in s.index]]
        # The local cache is a fixed one-year window, so it is sliced to the requested period.
        s = s[[start <= d0 <= end for d0 in s.index]]
        if s.empty:
            return None
        s = s[~s.index.duplicated(keep="last")].sort_index()
        return s
    except Exception as e:  # noqa: BLE001
        _log.warning("Failed to build the daily close series %s [%s ~ %s]: %s", ticker, start, end, e)
        return None


def _position_daily_returns(closes: pd.Series, entry_d: date) -> dict[date, float]:
    """
    Daily returns for one recommendation over its holding window.

    The entry price is the close on the entry day, or the nearest prior trading day. Returns therefore start
    on the first trading day after the entry day. The holding window is (entry_d, entry_d + HOLDING_DAYS].

    Args:
        closes: That symbol's daily closes, indexed by date.
        entry_d: Entry date.
    Returns:
        {trading day: that day's return}. An empty dict when the data are insufficient.
    """
    exit_d = entry_d + timedelta(days=HOLDING_DAYS)
    window = closes[[entry_d <= d0 <= exit_d for d0 in closes.index]]
    if len(window) < 2:
        # When the entry day is not a trading day, walk back to the nearest trading day for the base price.
        base = closes[[d0 <= entry_d for d0 in closes.index]]
        if base.empty:
            return {}
        window = pd.concat([base.iloc[[-1]], closes[[entry_d < d0 <= exit_d for d0 in closes.index]]])
        if len(window) < 2:
            return {}
    rets = window.pct_change().dropna()
    return {d0: float(v) for d0, v in rets.items()}


def _prepare_curve_inputs(df: pd.DataFrame) -> dict[str, Any]:
    """
    Fetch daily closes and expand them into the inputs the equity curve needs. Fetched once and shared by the curve and the attribution.

    Args:
        df: The recommendations table.
    Returns:
        {"positions": [(symbol, entry date, {trading day: that day's return})], "spy_closes": Series|None,
         "anchor": base date, "spine": [trading days]}. An empty dict when the data are insufficient.
    """
    # Collect valid positions. A date and a symbol are required. Returns are recomputed from daily prices, not from return_1m.
    raw: list[tuple[str, date]] = []
    for _, r in df.iterrows():
        entry_d = _parse_date(r.get("date"))
        ticker = str(r.get("ticker") or "").strip().upper()
        if entry_d is not None and ticker:
            raw.append((ticker, entry_d))
    if not raw:
        return {}

    first_entry = min(d0 for _, d0 in raw)
    last_exit = max(d0 for _, d0 in raw) + timedelta(days=HOLDING_DAYS)
    # Fetch 10 extra days so a holiday entry day still has a base close.
    fetch_start = first_entry - timedelta(days=10)
    fetch_end = min(last_exit, date.today())

    # Each symbol is fetched once, covering all of its holding windows.
    closes_by_ticker: dict[str, pd.Series | None] = {}
    for ticker, _ in raw:
        if ticker not in closes_by_ticker:
            closes_by_ticker[ticker] = _daily_close_series(ticker, fetch_start, fetch_end)

    # Expand each position into {trading day: that day's return}.
    positions: list[tuple[str, date, dict[date, float]]] = []
    missing: list[str] = []
    for ticker, entry_d in raw:
        closes = closes_by_ticker.get(ticker)
        rets = _position_daily_returns(closes, entry_d) if closes is not None and not closes.empty else {}
        if rets:
            positions.append((ticker, entry_d, rets))
        else:
            missing.append(f"{ticker}@{entry_d}")
    if missing:
        _log.warning("Equity curve: %s positions have no daily prices and were skipped (%s)", len(missing), ", ".join(missing[:5]))
    if not positions:
        return {}

    # Prefer the SPY calendar as the trading-day spine and as the benchmark price source. Fall back to the union of position dates.
    spy_closes = _daily_close_series("SPY", fetch_start, fetch_end)
    curve_end = max(max(r.keys()) for _, _, r in positions)
    if spy_closes is not None and not spy_closes.empty:
        base_days = [d0 for d0 in spy_closes.index if d0 <= first_entry]
        anchor = base_days[-1] if base_days else first_entry
        spine = [d0 for d0 in spy_closes.index if anchor < d0 <= curve_end]
    else:
        anchor = first_entry
        spine = sorted({d0 for _, _, r in positions for d0 in r if d0 > anchor})

    if not spine:
        return {}
    return {"positions": positions, "spy_closes": spy_closes, "anchor": anchor, "spine": spine}


def _equity_curve(
    positions: list[tuple[str, date, dict[date, float]]],
    spine: list[date],
    exclude: str | list[str] | set[str] | None = None,
) -> tuple[list[float], float]:
    """
    Build an equity curve from daily returns. Each trading day is the equal-weight average of names held that day, then compounded.

    Args:
        positions: [(symbol, entry date, {trading day: that day's return})]
        spine: Trading-day axis.
        exclude: Symbols to drop (one string, a list, or a set). Used by contribution attribution and
                 performance_excluding.
    Returns:
        (equity-curve cumulative returns, final cumulative return)
    """
    if exclude is None:
        exclude_set: set[str] = set()
    elif isinstance(exclude, str):
        exclude_set = {exclude.strip().upper()}
    else:
        exclude_set = {str(t).strip().upper() for t in exclude if str(t).strip()}
    active = [r for t, _d, r in positions if t not in exclude_set]
    equity = 1.0
    curve: list[float] = [0.0]  # Starts on the base date, return 0.
    for d0 in spine:
        todays = [r[d0] for r in active if d0 in r]
        port_ret = sum(todays) / len(todays) if todays else 0.0  # A day with no positions does not move.
        equity *= 1.0 + port_ret
        curve.append(round(equity - 1.0, 6))
    return curve, equity - 1.0


def _build_daily_equity_curve(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Build the equity curve in trading-day units, so overlapping holding periods are handled correctly.

    Each day's portfolio return is the equal-weight average of names held that day
    (entry_d < day <= entry_d + HOLDING_DAYS). A day with no positions has return 0.
    The SPY benchmark is a simple buy-and-hold over the same calendar window, so overlapping holds are not compounded twice.

    Args:
        inputs: Output of _prepare_curve_inputs().
    Returns:
        The cumulative result dict. An empty dict when the data are insufficient.
    """
    if not inputs:
        return {}
    positions = inputs["positions"]
    spine = inputs["spine"]
    anchor = inputs["anchor"]
    spy_closes = inputs["spy_closes"]

    strat_curve, strategy_cum = _equity_curve(positions, spine)
    curve_dates = [anchor.isoformat()] + [d0.isoformat() for d0 in spine]
    trading_days = len(curve_dates)

    # SPY: simple buy-and-hold over the same window (buy the anchor close, sell the close on the last curve day).
    spy_cum: float | None = None
    spy_curve: list[float] = []
    if spy_closes is not None and anchor in spy_closes.index:
        spy_base = float(spy_closes.loc[anchor])
        if spy_base > 0:
            spy_curve = [0.0]
            for d0 in spine:
                px = float(spy_closes.loc[d0]) if d0 in spy_closes.index else None
                prev = spy_curve[-1]
                spy_curve.append(round(px / spy_base - 1.0, 6) if px else prev)
            spy_cum = spy_curve[-1]

    _warn_if_implausible("Strategy", strategy_cum, trading_days)
    _warn_if_implausible("SPY", spy_cum, trading_days)
    _warn_if_benchmark_implausible(spy_cum, trading_days)

    return {
        "strategy_cumulative_return": round(strategy_cum, 6),
        "spy_cumulative_return": round(spy_cum, 6) if spy_cum is not None else None,
        "outperformance": round(strategy_cum - spy_cum, 6) if spy_cum is not None else None,
        "trading_days": trading_days,
        "date_range": {"start": curve_dates[0], "end": curve_dates[-1]},
        "strategy_equity_curve": strat_curve,
        "spy_equity_curve": spy_curve,
        "dates": curve_dates,
    }


def _build_concentration(
    df: pd.DataFrame, inputs: dict[str, Any], cumulative: dict[str, Any] | str
) -> dict[str, Any]:
    """
    Answer which names the excess return is concentrated in, using leave-one-out contribution.

    Contribution = full equity-curve cumulative return minus the cumulative return recomputed without that name.
    Under equal weight, dropping one name raises the weight of the others, so the contributions do not sum exactly to the total return.
    That is inherent to leave-one-out attribution. It is used to judge concentration, not to split the return into exact pieces.

    Args:
        df: The recommendations table.
        inputs: Output of _prepare_curve_inputs().
        cumulative: Result of _build_daily_equity_curve().
    Returns:
        The concentration result dict. An empty dict when the data are insufficient.
    """
    if not inputs or not isinstance(cumulative, dict):
        return {}
    positions = inputs["positions"]
    spine = inputs["spine"]
    total_cum = float(cumulative["strategy_cumulative_return"])
    outperf = cumulative.get("outperformance")

    # Pick count and average return_1m for each name, from the full table.
    tickers = sorted({t for t, _d, _r in positions})
    by_ticker: dict[str, Any] = {}
    for ticker in tickers:
        rows = df[df["ticker"].astype(str).str.strip().str.upper() == ticker]
        _, cum_without = _equity_curve(positions, spine, exclude=ticker)
        by_ticker[ticker] = {
            "picks": int(len(rows)),
            "avg_return_1m": _safe_mean(rows["return_1m"]),
            "contribution": round(total_cum - cum_without, 6),
        }

    # Sort by contribution, highest first, so a report can print the table as-is.
    by_ticker = dict(sorted(by_ticker.items(), key=lambda kv: kv[1]["contribution"], reverse=True))

    top_ticker, top_stats = next(iter(by_ticker.items()))
    top_contrib = float(top_stats["contribution"])
    # Share uses total excess return as the denominator. When the SPY benchmark is missing, fall back to total cumulative return.
    denom = float(outperf) if outperf not in (None, 0) else total_cum
    share = (top_contrib / denom) if denom > 0 else None

    warning = None
    if share is not None and share > CONCENTRATION_WARN_SHARE:
        warning = "Performance is highly concentrated in one name, so the sample is a weak representation"
        _log.warning(
            "%s contributed %+.2f%%, %.1f%% of total excess return (>%.0f%%): %s",
            top_ticker,
            top_contrib * 100,
            share * 100,
            CONCENTRATION_WARN_SHARE * 100,
            warning,
        )

    return {
        "by_ticker": by_ticker,
        "top_contributor": {
            "ticker": top_ticker,
            "contribution": round(top_contrib, 6),
            "share_of_outperformance": round(share, 4) if share is not None else None,
            "picks": int(top_stats["picks"]),
        },
        "concentration_warning": warning,
    }


def performance_excluding(tickers: list[str]) -> dict[str, Any]:
    """
    Strategy performance after the named symbols are removed.

    Answers whether the strategy still has alpha after the largest contributor is removed.
    It recomputes with the same daily equal-weight compounding as the full curve, but drops every position in the named symbols.
    The SPY benchmark stays a simple buy-and-hold over the same window and is not affected by the exclusion.

    Args:
        tickers: Symbols to exclude. Case is ignored.
    Returns:
        {
            excluded: symbols actually excluded (uppercase, de-duplicated, sorted),
            strategy_return: strategy cumulative return after the exclusion,
            spy_return: SPY buy-and-hold over the same window,
            outperformance: strategy minus SPY,
            still_beats_spy: whether the strategy is still above SPY,
            remaining_picks_count: recommendation count left after the exclusion,
        }
        When the data are insufficient, strategy_return and spy_return are None and still_beats_spy is False.
    """
    excluded = sorted({str(t).strip().upper() for t in (tickers or []) if str(t).strip()})
    empty: dict[str, Any] = {
        "excluded": excluded,
        "strategy_return": None,
        "spy_return": None,
        "outperformance": None,
        "still_beats_spy": False,
        "remaining_picks_count": 0,
    }

    df = _load_recommendations()
    if df.empty:
        return empty

    ticker_col = df["ticker"].astype(str).str.strip().str.upper()
    remaining = int((~ticker_col.isin(set(excluded))).sum()) if excluded else int(len(df))

    inputs = _prepare_curve_inputs(df)
    if not inputs:
        empty["remaining_picks_count"] = remaining
        return empty

    # SPY: simple buy-and-hold over the same window, same as the full curve, unaffected by the exclusion.
    spy_closes = inputs["spy_closes"]
    anchor = inputs["anchor"]
    spine = inputs["spine"]
    spy_ret: float | None = None
    if spy_closes is not None and anchor in spy_closes.index:
        spy_base = float(spy_closes.loc[anchor])
        if spy_base > 0 and spine:
            last_px: float | None = None
            for d0 in reversed(spine):
                if d0 in spy_closes.index:
                    last_px = float(spy_closes.loc[d0])
                    break
            if last_px is not None:
                spy_ret = last_px / spy_base - 1.0

    _, strategy_ret = _equity_curve(inputs["positions"], spine, exclude=excluded)

    outperf = (strategy_ret - spy_ret) if spy_ret is not None else None
    still_beats = bool(spy_ret is not None and strategy_ret > spy_ret)

    return {
        "excluded": excluded,
        "strategy_return": round(strategy_ret, 6),
        "spy_return": round(spy_ret, 6) if spy_ret is not None else None,
        "outperformance": round(outperf, 6) if outperf is not None else None,
        "still_beats_spy": still_beats,
        "remaining_picks_count": remaining,
    }


def calculate_performance() -> dict[str, Any]:
    """Build a performance snapshot from recommendations.csv."""
    df = _load_recommendations()
    empty_summary = {
        "total_recommendations": 0,
        "completed_1m": 0,
        "date_range": {"first": None, "last": None},
    }
    if df.empty:
        return {
            "summary": empty_summary,
            "aggregate": "insufficient data",
            "by_sector": "insufficient data",
            "by_month": "insufficient data",
            "cumulative": "insufficient data",
            "concentration": "insufficient data",
        }

    dates = [_parse_date(x) for x in df["date"].tolist()]
    dates_valid = sorted([d for d in dates if d is not None])
    completed = df[pd.to_numeric(df["return_1m"], errors="coerce").notna()].copy()
    summary = {
        "total_recommendations": int(len(df)),
        "completed_1m": int(len(completed)),
        "date_range": {
            "first": dates_valid[0].isoformat() if dates_valid else None,
            "last": dates_valid[-1].isoformat() if dates_valid else None,
        },
    }

    if len(completed) < 5:
        return {
            "summary": summary,
            "aggregate": "insufficient data",
            "by_sector": "insufficient data",
            "by_month": "insufficient data",
            "cumulative": "insufficient data",
            "concentration": "insufficient data",
            "note": "completed_1m < 5, so more tracked records are needed before statistics are meaningful",
        }

    r1m = pd.to_numeric(completed["return_1m"], errors="coerce")
    alpha = pd.to_numeric(completed["alpha_1m"], errors="coerce")
    spy_r = pd.to_numeric(completed["spy_return_1m"], errors="coerce")

    avg_1m = _safe_mean(r1m) or 0.0
    std_1m = _safe_std(r1m)
    # Sharpe treats each recommendation's 1M return as a monthly return, annualized with sqrt(12). rf=0.
    if std_1m is not None and std_1m > 1e-12:
        sharpe = (avg_1m / std_1m) * math.sqrt(12)
    else:
        sharpe = 0.0

    win_rate = float((r1m > 0).mean()) if len(r1m.dropna()) else 0.0
    beat = completed.dropna(subset=["return_1m", "spy_return_1m"])
    if not beat.empty:
        beat_rate = float((pd.to_numeric(beat["return_1m"], errors="coerce") > pd.to_numeric(beat["spy_return_1m"], errors="coerce")).mean())
    else:
        beat_rate = 0.0

    aggregate = {
        "avg_return_1d": _safe_mean(pd.to_numeric(df["return_1d"], errors="coerce")),
        "avg_return_1w": _safe_mean(pd.to_numeric(df["return_1w"], errors="coerce")),
        "avg_return_2w": _safe_mean(pd.to_numeric(df["return_2w"], errors="coerce")),
        "avg_return_1m": avg_1m,
        "median_return_1m": _safe_median(r1m),
        "win_rate_1m": win_rate,
        "beat_spy_rate_1m": beat_rate,
        "avg_alpha_1m": _safe_mean(alpha) or 0.0,
        "std_return_1m": std_1m,
        "sharpe_ratio_1m": round(float(sharpe), 4),
        "max_return_1m": float(r1m.max()) if len(r1m.dropna()) else None,
        "max_loss_1m": float(r1m.min()) if len(r1m.dropna()) else None,
    }

    # by_sector
    by_sector: dict[str, Any] = {}
    for sector, g in completed.groupby(completed["sector"].fillna("unknown").astype(str)):
        rr = pd.to_numeric(g["return_1m"], errors="coerce").dropna()
        by_sector[str(sector)] = {
            "count": int(len(g)),
            "avg_return_1m": float(rr.mean()) if len(rr) else None,
            "win_rate_1m": float((rr > 0).mean()) if len(rr) else None,
        }

    # by_month: grouped by the recommendation month.
    completed = completed.copy()
    completed["_month"] = completed["date"].map(lambda x: (_parse_date(x).strftime("%Y-%m") if _parse_date(x) else "unknown"))
    by_month: list[dict[str, Any]] = []
    for month, g in completed.groupby("_month"):
        rr = pd.to_numeric(g["return_1m"], errors="coerce").dropna()
        sr = pd.to_numeric(g["spy_return_1m"], errors="coerce").dropna()
        avg_r = float(rr.mean()) if len(rr) else None
        avg_s = float(sr.mean()) if len(sr) else None
        by_month.append(
            {
                "month": str(month),
                "picks": int(len(g)),
                "avg_return_1m": avg_r,
                "spy_return_1m": avg_s,
                "alpha": (avg_r - avg_s) if avg_r is not None and avg_s is not None else None,
            }
        )
    by_month.sort(key=lambda x: x["month"])

    # cumulative: an equity curve in trading-day units. Overlapping holds must not be compounded pick by pick.
    curve_inputs = _prepare_curve_inputs(df)
    cumulative: dict[str, Any] | str = _build_daily_equity_curve(curve_inputs) or "insufficient data"
    concentration: dict[str, Any] | str = _build_concentration(df, curve_inputs, cumulative) or "insufficient data"

    return {
        "summary": summary,
        "aggregate": aggregate,
        "by_sector": by_sector,
        "by_month": by_month,
        "cumulative": cumulative,
        "concentration": concentration,
    }


def save_performance_snapshot(perf: dict[str, Any] | None = None) -> Path:
    """
    Compute performance and write performance.json. Returns the path.

    Args:
        perf: An already computed performance dict. None calls calculate_performance().
              Passing it avoids a second price fetch and a second warning.
    Returns:
        Path to performance.json.
    """
    if perf is None:
        perf = calculate_performance()
    payload = {
        "snapshot_time": datetime.now().isoformat(timespec="seconds"),
        **perf,
    }
    TRACK_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(PERFORMANCE_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        _log.info("Saved performance snapshot: %s", PERFORMANCE_JSON)
    except Exception as e:  # noqa: BLE001
        _log.warning("Failed to write performance.json: %s", e)
    return PERFORMANCE_JSON


def backfill_from_daily_picks() -> int:
    """Scan data/daily_picks/*.json and write recommendations that are not stored yet."""
    picks_dir = Path(DATA_DIR) / "daily_picks"
    if not picks_dir.exists():
        return 0
    n = 0
    for path in sorted(picks_dir.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            before = len(_load_recommendations())
            record_daily_picks(data)
            after = len(_load_recommendations())
            n += max(0, after - before)
        except Exception as e:  # noqa: BLE001
            _log.warning("backfill %s failed: %s", path.name, e)
    return n


if __name__ == "__main__":
    print("Scanning existing daily_picks and writing the track record...")
    added = backfill_from_daily_picks()
    print(f"New records: {added}")

    print("Running update_actuals...")
    stats = update_actuals()
    print(f"Update stats: {stats}")

    # Compute and print current performance.
    perf = calculate_performance()
    print("\n=== Strategy performance ===")
    summary = perf["summary"]
    print(f"Total recommendations: {summary['total_recommendations']}")
    print(f"Completed 1M tracks: {summary['completed_1m']}")
    print(f"Date range: {summary['date_range']['first']} ~ {summary['date_range']['last']}")

    if summary["completed_1m"] >= 5:
        agg = perf["aggregate"]
        assert isinstance(agg, dict)
        print(f"\n1M average return: {float(agg['avg_return_1m'] or 0):+.2%}")
        print(f"1M win rate: {float(agg['win_rate_1m'] or 0):.1%}")
        print(f"1M beat-SPY rate: {float(agg['beat_spy_rate_1m'] or 0):.1%}")
        print(f"1M average alpha: {float(agg['avg_alpha_1m'] or 0):+.2%}")
        print(f"Sharpe Ratio：{float(agg['sharpe_ratio_1m'] or 0):.2f}")
        cum = perf["cumulative"]
        if isinstance(cum, dict):
            dr = cum["date_range"]
            print(f"\nEquity-curve window: {dr['start']} ~ {dr['end']} ({cum['trading_days']} trading days)")
            print(f"Cumulative strategy return: {float(cum['strategy_cumulative_return'] or 0):+.2%}")
            print(f"Cumulative SPY return (buy and hold): {float(cum['spy_cumulative_return'] or 0):+.2%}")
            print(f"Excess return: {float(cum['outperformance'] or 0):+.2%}")

            # Comparison after dropping the largest contributor: does alpha remain without that name?
            print("\n=== Performance after dropping the largest contributor ===")
            full_s = float(cum["strategy_cumulative_return"] or 0)
            full_spy = float(cum["spy_cumulative_return"] or 0)
            print(f"Full strategy: {full_s:+.2%} vs SPY {full_spy:+.2%}")
            excl_ticker = "MU"
            if isinstance(conc := perf.get("concentration"), dict):
                top = conc.get("top_contributor") if isinstance(conc.get("top_contributor"), dict) else {}
                if top.get("ticker"):
                    excl_ticker = str(top["ticker"])
            excl = performance_excluding([excl_ticker])
            excl_s = excl.get("strategy_return")
            excl_spy = excl.get("spy_return")
            if excl_s is not None and excl_spy is not None:
                verdict = "still beats SPY" if excl["still_beats_spy"] else "does not beat SPY"
                print(
                    f"Excluding {excl_ticker} ({excl['remaining_picks_count']} picks left): "
                    f"{excl_s:+.2%} vs SPY {excl_spy:+.2%} -> {verdict}"
                    f" (excess {float(excl['outperformance'] or 0):+.2%})"
                )
            else:
                print(f"Excluding {excl_ticker}: insufficient data, cannot compute")
        else:
            print("\nEquity curve: insufficient data")

        conc = perf["concentration"]
        if isinstance(conc, dict):
            print("\n=== Return concentration (leave-one-out) ===")
            print(f"{'Name':<8}{'Picks':>6}{'Avg 1M':>10}{'Contrib':>11}")
            for tk, st in conc["by_ticker"].items():
                avg = st["avg_return_1m"]
                avg_s = f"{avg:+.2%}" if avg is not None else "—"
                print(f"{tk:<8}{st['picks']:>5}{avg_s:>10}{st['contribution']:>+11.2%}")
            top = conc["top_contributor"]
            share = top["share_of_outperformance"]
            share_s = f", {share:.1%} of total excess return" if share is not None else ""
            print(f"\nLargest contributor: {top['ticker']} ({top['contribution']:+.2%}{share_s})")
            if conc["concentration_warning"]:
                print(f"⚠️  {conc['concentration_warning']}")
        else:
            print("\nReturn concentration: insufficient data")
    else:
        print("\nFewer than 5 completed tracks. More recommendations are needed before statistics are meaningful")

    path = save_performance_snapshot(perf)
    print(f"\nSnapshot written: {path}")
