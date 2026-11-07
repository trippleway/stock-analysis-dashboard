# -*- coding: utf-8 -*-
"""
Price panel for backtests: fetch OHLCV for many names into a daily panel that can be aligned, with a local cache.

Technical scores and returns both use the same adjusted prices (auto_adjust=True).
An unadjusted close looks like a crash on a split, and backtest returns would be wrong.

Cache directory: data/backtest_cache/ohlcv/{TICKER}.parquet
A cache that already covers the requested range is not sent to yfinance again.

Run from the project root:
    python src/backtest_data.py
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

_ROOT = Path(__file__).resolve().parent.parent
_SRC = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import DATA_DIR  # noqa: E402

_log = logging.getLogger("backtest_data")
if not _log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

_OHLCV_DIR = Path(DATA_DIR) / "backtest_cache" / "ohlcv"
_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
_YF_CALL_SLEEP_SEC = 0.7
_YF_RETRY_DELAYS = (1.0, 2.0, 4.0)
# If the cache ends within this many calendar days of the requested end, it counts as covering the range (weekends and a session that has not closed yet).
_CACHE_END_TOLERANCE_DAYS = 7


def _coerce_date(value: date | datetime | str, name: str) -> date:
    """
    Turn a date, datetime, or ISO string into a date.

    Args:
        value: Date input.
        name: Parameter name, used in the error message.

    Returns:
        datetime.date.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        try:
            return date.fromisoformat(text[:10])
        except ValueError as exc:
            raise ValueError(f"{name} is not a valid date: {value!r}. Use YYYY-MM-DD") from exc
    raise TypeError(f"{name} must be a date, got {type(value).__name__}")


def _normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a yfinance daily frame: drop the timezone, keep only OHLCV, and keep the last row when a day is duplicated.

    Args:
        raw: The result of yfinance history.

    Returns:
        OHLCV indexed by a timezone-naive date. An empty frame means this data cannot be used.
    """
    if raw is None or raw.empty:
        return pd.DataFrame(columns=list(_OHLCV_COLUMNS))

    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    missing = [col for col in _OHLCV_COLUMNS if col not in frame.columns]
    if missing:
        return pd.DataFrame(columns=list(_OHLCV_COLUMNS))

    out = frame.loc[:, list(_OHLCV_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["Close"])
    out.index.name = "date"
    return out


def _cache_path(ticker: str) -> Path:
    """Parquet path for one symbol."""
    return _OHLCV_DIR / f"{ticker.upper()}.parquet"


def _read_cache(ticker: str) -> pd.DataFrame | None:
    """Read local daily bars. A bad file returns None and does not stop the other symbols."""
    path = _cache_path(ticker)
    if not path.is_file():
        return None
    try:
        cached = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        _log.warning("Failed to read the price cache for %s; it will be downloaded again: %s", ticker, exc)
        return None
    cached = _normalize_ohlcv(cached)
    if cached.empty:
        return None
    return cached


def _cache_covers(cached: pd.DataFrame, start: date, end: date) -> bool:
    """Whether the cache dates cover the requested range. The end may be a few calendar days late."""
    first = cached.index.min().date()
    last = cached.index.max().date()
    return first <= start and last >= end - timedelta(days=_CACHE_END_TOLERANCE_DAYS)


def _download_ohlcv(ticker: str, start: date, end: date) -> pd.DataFrame | None:
    """
    Fetch daily bars for one symbol from yfinance. Retry 3 times on failure.

    Args:
        ticker: Ticker symbol.
        start: Start date (inclusive).
        end: End date (inclusive). yfinance end is exclusive, so one day is added here.

    Returns:
        Cleaned OHLCV. None when all three attempts fail or there is no data.
    """
    last_err: Exception | None = None
    end_exclusive = end + timedelta(days=1)
    for attempt in range(3):
        try:
            time.sleep(_YF_CALL_SLEEP_SEC)
            raw = yf.Ticker(ticker).history(
                start=start.isoformat(),
                end=end_exclusive.isoformat(),
                interval="1d",
                auto_adjust=True,
                actions=False,
            )
            frame = _normalize_ohlcv(raw)
            if frame.empty:
                last_err = ValueError("no usable close")
            else:
                return frame
        except Exception as exc:  # noqa: BLE001 — a single-name failure is retried
            last_err = exc
        if attempt < 2:
            time.sleep(_YF_RETRY_DELAYS[attempt])
    _log.warning("Failed to fetch prices for %s after 3 attempts: %s", ticker, last_err)
    return None


def load_ohlcv(ticker: str, start: date, end: date) -> pd.DataFrame | None:
    """
    Load OHLCV for one symbol inside the range. A local cache that is enough is not downloaded again.

    Args:
        ticker: Ticker symbol.
        start: Start date (inclusive).
        end: End date (inclusive).

    Returns:
        OHLCV inside the range. None when there is no data at all.
    """
    symbol = ticker.upper().strip()
    start_d = _coerce_date(start, "start")
    end_d = _coerce_date(end, "end")
    if start_d > end_d:
        raise ValueError(f"start ({start_d}) is after end ({end_d}) for {symbol}")

    cached = _read_cache(symbol)
    if cached is not None and _cache_covers(cached, start_d, end_d):
        sliced = cached.loc[pd.Timestamp(start_d) : pd.Timestamp(end_d)]
        _log.debug("Using the price cache for %s (%s rows)", symbol, len(sliced))
        return sliced if not sliced.empty else None

    _OHLCV_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = _download_ohlcv(symbol, start_d, end_d)
    if downloaded is None:
        if cached is not None:
            _log.warning("Redownload failed for %s; the older cache is reused", symbol)
            sliced = cached.loc[pd.Timestamp(start_d) : pd.Timestamp(end_d)]
            return sliced if not sliced.empty else None
        return None

    if cached is not None:
        merged = pd.concat([cached, downloaded])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = downloaded
    try:
        merged.to_parquet(_cache_path(symbol), index=True)
    except Exception as exc:  # noqa: BLE001
        _log.warning("Failed to write the price cache for %s; the in-memory data is still used: %s", symbol, exc)
    sliced = merged.loc[pd.Timestamp(start_d) : pd.Timestamp(end_d)]
    return sliced if not sliced.empty else None


def build_ohlcv_panel(
    tickers: list[str],
    start: date,
    end: date,
) -> dict[str, pd.DataFrame]:
    """
    Build a multi-name OHLCV panel shared by technical scores and returns.

    A name that cannot be fetched is skipped with a warning. It does not stop the whole backtest.
    If the caller needs moving-average warmup, it should move start earlier itself.

    Args:
        tickers: Ticker symbols. Duplicates and blanks are dropped.
        start: Start date (inclusive).
        end: End date (inclusive).

    Returns:
        {symbol: OHLCV DataFrame}. A failed symbol does not appear in the dict.
    """
    start_d = _coerce_date(start, "start")
    end_d = _coerce_date(end, "end")
    symbols: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        symbol = str(ticker).upper().strip()
        if symbol and symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)

    panel: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            frame = load_ohlcv(symbol, start_d, end_d)
        except Exception as exc:  # noqa: BLE001
            _log.warning("Failed to build prices for %s: %s", symbol, exc)
            frame = None
        if frame is None or frame.empty:
            _log.warning("%s has no usable prices from %s to %s, so this name will not enter the backtest", symbol, start_d, end_d)
            continue
        panel[symbol] = frame
        _log.info(
            "%s prices: %s rows (%s to %s)",
            symbol,
            len(frame),
            frame.index.min().date().isoformat(),
            frame.index.max().date().isoformat(),
        )
    _log.info("Price panel finished: requested %s names, succeeded for %s", len(symbols), len(panel))
    return panel


if __name__ == "__main__":
    today = date.today()
    start = today - timedelta(days=400)
    panel = build_ohlcv_panel(["SPY", "NVDA"], start, today)
    for symbol, frame in panel.items():
        print(
            f"{symbol}: {len(frame)} rows, "
            f"{frame.index.min().date()} to {frame.index.max().date()}, "
            f"last close {float(frame['Close'].iloc[-1]):.2f}",
            flush=True,
        )
    missing = [symbol for symbol in ("SPY", "NVDA") if symbol not in panel]
    if missing:
        print("Missing:", ", ".join(missing), flush=True)
        sys.exit(1)
