# -*- coding: utf-8 -*-
"""
Stock data collector: OHLCV and fundamental fields from yfinance, with a daily Parquet/JSON cache.

Run from the project root:
    python src/data_collector.py
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf
from tqdm import tqdm

# Load config from the project root (sys.path defaults to src/ when this file is run directly).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import ALL_STOCKS, CACHE_DIR, LOG_DIR, TICKER_LIST  # noqa: E402

# Fields to pull from yfinance info (they become the info keys of the returned dict).
_INFO_KEYS = (
    "sector",
    "industry",
    "marketCap",
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "returnOnEquity",
    "profitMargins",
    "grossMargins",
    "revenueGrowth",
    "earningsGrowth",
    "debtToEquity",
    "currentRatio",
    "targetMeanPrice",
    "recommendationMean",
    "fiftyTwoWeekHigh",
    "fiftyTwoWeekLow",
)

_logger: logging.Logger | None = None


def _get_logger() -> logging.Logger:
    """Return the logger that writes to a file and the terminal (singleton)."""
    global _logger
    if _logger is not None:
        return _logger

    log = logging.getLogger("stock_collector")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()

    today_str = datetime.now().strftime("%Y-%m-%d")
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"collector_{today_str}.log"

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    log.addHandler(fh)
    log.addHandler(ch)
    _logger = log
    return log


def _today_cache_dir() -> Path:
    """Today's cache directory, CACHE_DIR / YYYY-MM-DD."""
    d = datetime.now().strftime("%Y-%m-%d")
    p = Path(CACHE_DIR) / d
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_paths(ticker: str, date_str: str | None = None) -> tuple[Path, Path]:
    """
    Paths to the parquet and info json under the folder for a given date.

    Args:
        ticker: Ticker symbol.
        date_str: Date string YYYY-MM-DD. None means today.

    Returns:
        (price_parquet_path, info_json_path)
    """
    d = date_str or datetime.now().strftime("%Y-%m-%d")
    base = Path(CACHE_DIR) / d
    sym = ticker.upper()
    return base / f"{sym}.parquet", base / f"{sym}_info.json"


def _extract_info(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Pull the fixed fields from a yfinance info dict. A missing field is None."""
    if not raw:
        return {k: None for k in _INFO_KEYS}
    return {k: raw.get(k) for k in _INFO_KEYS}


def _pick_name(ticker: str, info: dict[str, Any]) -> str:
    """Display name: yfinance longName or shortName first, then config.ALL_STOCKS."""
    for key in ("longName", "shortName", "displayName"):
        v = info.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ALL_STOCKS.get(ticker.upper(), ticker.upper())


def _pick_current_price(info: dict[str, Any], hist: pd.DataFrame) -> float | None:
    """Current price: regularMarketPrice or currentPrice first, otherwise the last Close in history."""
    for key in ("regularMarketPrice", "currentPrice", "postMarketPrice", "preMarketPrice"):
        v = info.get(key)
        if v is not None and isinstance(v, (int, float)) and not (isinstance(v, float) and v != v):
            return float(v)
    if hist is not None and not hist.empty and "Close" in hist.columns:
        last = hist["Close"].iloc[-1]
        if pd.notna(last):
            return float(last)
    return None


def _trim_ohlcv(hist: pd.DataFrame) -> pd.DataFrame:
    """Keep only the OHLCV columns. Adj Close is dropped when it is present."""
    cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in hist.columns]
    if not cols:
        return hist.copy()
    out = hist[cols].copy()
    return out


def _load_cache_if_valid(
    ticker: str, force_refresh: bool, log: logging.Logger
) -> dict[str, Any] | None:
    """
    If today's cache exists and a refresh was not forced, load it and return the full result dict. Otherwise return None.

    The cache is valid only when both the parquet and the info json exist and the date folder is today.
    """
    if force_refresh:
        return None
    today = datetime.now().strftime("%Y-%m-%d")
    pq_path, js_path = _cache_paths(ticker, today)
    if not pq_path.is_file() or not js_path.is_file():
        return None
    try:
        price_history = pd.read_parquet(pq_path)
        with open(js_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("cache_date") != today:
            return None
        sym = ticker.upper()
        name = meta.get("name") or _pick_name(sym, meta.get("info_full") or {})
        info_subset = meta.get("info") or _extract_info(meta.get("info_full") or {})
        current_price = meta.get("current_price")
        if current_price is None:
            current_price = _pick_current_price(meta.get("info_full") or {}, price_history)
        fetched_at = meta.get("fetched_at") or datetime.now(timezone.utc).isoformat()
        log.debug("Using today's cache: %s", sym)
        return {
            "ticker": sym,
            "name": name,
            "price_history": price_history,
            "current_price": current_price,
            "info": info_subset,
            "fetched_at": fetched_at,
        }
    except Exception as e:
        log.warning("Failed to read the cache for %s: %s. It will be fetched again", ticker, e)
        return None


def _save_cache(
    ticker: str,
    name: str,
    price_history: pd.DataFrame,
    current_price: float | None,
    info_subset: dict[str, Any],
    info_full: dict[str, Any],
    fetched_at: str,
    log: logging.Logger,
) -> None:
    """Write today's parquet and _info.json."""
    today = datetime.now().strftime("%Y-%m-%d")
    base = Path(CACHE_DIR) / today
    base.mkdir(parents=True, exist_ok=True)
    sym = ticker.upper()
    pq_path = base / f"{sym}.parquet"
    js_path = base / f"{sym}_info.json"
    try:
        price_history.to_parquet(pq_path, index=True)
    except Exception as e:
        log.error("Failed to write Parquet for %s: %s", sym, e)
        raise
    meta = {
        "cache_date": today,
        "ticker": sym,
        "name": name,
        "current_price": current_price,
        "info": info_subset,
        "info_full": info_full,
        "fetched_at": fetched_at,
    }
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)


def _fetch_yfinance_once(ticker: str) -> tuple[pd.DataFrame, dict[str, Any], str, float | None]:
    """
    One yfinance call, with no retry inside this function.

    Returns:
        (price_history OHLCV, full info dict, company name, current price)
    """
    sym = ticker.upper()
    t = yf.Ticker(sym)
    # One year of daily bars. Keep the unadjusted OHLC so the columns match the OHLCV description.
    hist = t.history(period="1y", interval="1d", auto_adjust=False)
    if hist.empty:
        raise ValueError(f"{sym}: no historical prices")
    hist = _trim_ohlcv(hist)
    info_full = t.info or {}
    if not info_full:
        # Some environments return an empty info. Prices are still enough to continue.
        info_full = {}
    name = _pick_name(sym, info_full)
    current_price = _pick_current_price(info_full, hist)
    return hist, info_full, name, current_price


def _fetch_with_retries(ticker: str, log: logging.Logger) -> tuple[pd.DataFrame, dict[str, Any], str, float | None] | None:
    """
    yfinance fetch with retries: up to 4 attempts (waits of 5, 10, and 20 seconds after a failure).

    Returns:
        (hist, info_full, name, current_price) on success. None when every attempt fails.
    """
    delays = (5, 10, 20)
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            return _fetch_yfinance_once(ticker)
        except Exception as e:
            last_err = e
            log.warning(
                "Fetch failed for %s, attempt %s: %s",
                ticker.upper(),
                attempt + 1,
                e,
            )
            if attempt < 3:
                time.sleep(delays[attempt])
    log.error("Gave up on %s after the maximum number of retries: %s", ticker.upper(), last_err)
    return None


def fetch_stock_data(ticker: str, force_refresh: bool = False) -> dict[str, Any] | None:
    """
    Fetch prices and a short fundamental snapshot for one stock, using the daily local cache.

    Args:
        ticker: Ticker symbol (case insensitive).
        force_refresh: When True, ignore the cache and fetch from yfinance again.

    Returns:
        On success, a dict with ticker, name, price_history, current_price, info, and fetched_at.
        None on failure. This function does not raise.
    """
    log = _get_logger()
    sym = ticker.upper()

    cached = _load_cache_if_valid(sym, force_refresh, log)
    if cached is not None:
        return cached

    got = _fetch_with_retries(sym, log)
    if got is None:
        return None

    hist, info_full, name, current_price = got
    info_subset = _extract_info(info_full)
    fetched_at = datetime.now(timezone.utc).isoformat()

    result = {
        "ticker": sym,
        "name": name,
        "price_history": hist,
        "current_price": current_price,
        "info": info_subset,
        "fetched_at": fetched_at,
    }

    try:
        _save_cache(sym, name, hist, current_price, info_subset, info_full, fetched_at, log)
    except Exception:
        # A cache write failure still returns the in-memory result.
        log.exception("Cache write failed; the in-memory result is still returned: %s", sym)

    return result


def fetch_spy_quote(force_refresh: bool = False) -> tuple[float | None, float | None]:
    """Latest SPY close and the percent change from the previous close.

    SPY is not part of the stock universe, so the daily batch does not include it.
    This reads the same daily cache as the other tickers. A missing or unusable
    history returns no quote instead of raising.

    Args:
        force_refresh: When True, ignore today's cache and fetch again.

    Returns:
        (price, change_pct). Both are None when the quote is unavailable.
    """
    log = _get_logger()
    try:
        data = fetch_stock_data("SPY", force_refresh=force_refresh)
    except Exception as e:  # noqa: BLE001
        log.warning("SPY quote fetch failed: %s", e)
        return None, None
    if not isinstance(data, dict):
        log.warning("SPY quote is unavailable")
        return None, None
    hist = data.get("price_history")
    if not isinstance(hist, pd.DataFrame) or "Close" not in hist.columns:
        log.warning("SPY quote has no Close column")
        return None, None
    closes = pd.to_numeric(hist["Close"], errors="coerce").dropna()
    if len(closes) < 2:
        log.warning("SPY quote has fewer than two closes")
        return None, None
    price = float(closes.iloc[-1])
    previous = float(closes.iloc[-2])
    if previous == 0 or not math.isfinite(price) or not math.isfinite(previous):
        log.warning("SPY quote closes are not usable")
        return None, None
    change_pct = (price / previous - 1.0) * 100.0
    if not math.isfinite(change_pct):
        return None, None
    return price, change_pct


def collect_all_stocks(
    tickers: list[str] | None = None,
    force_refresh: bool = False,
) -> dict[str, dict[str, Any] | None]:
    """
    Fetch many stocks. config.TICKER_LIST is the default.

    Args:
        tickers: Symbol list. None uses TICKER_LIST.
        force_refresh: Whether to skip the cache.

    Returns:
        A dict keyed by uppercase symbol. The value is the fetch_stock_data result, or None on failure.
    """
    log = _get_logger()
    symbols = list(tickers) if tickers is not None else list(TICKER_LIST)
    symbols = [s.upper() for s in symbols]
    out: dict[str, dict[str, Any] | None] = {}

    for i, sym in enumerate(tqdm(symbols, desc="Collecting", unit="name")):
        try:
            data = fetch_stock_data(sym, force_refresh=force_refresh)
            out[sym] = data
            if data is None:
                log.error("Failed during the batch: %s", sym)
        except Exception as e:
            log.exception("Unexpected error during the batch for %s: %s", sym, e)
            out[sym] = None
        if i < len(symbols) - 1:
            time.sleep(1)

    ok = sum(1 for v in out.values() if v is not None)
    bad = len(out) - ok
    log.info("Batch finished: %s succeeded, %s failed", ok, bad)
    print(f"[collect_all_stocks] {ok} succeeded, {bad} failed")
    return out


def _print_summary(data: dict[str, Any]) -> None:
    """Print a one-name summary (used by the __main__ test)."""
    sym = data["ticker"]
    name = data["name"]
    price = data["current_price"]
    rows = len(data["price_history"])
    print(f"  {sym} | {name} | price: {price} | history rows: {rows} | fetched_at: {data['fetched_at']}")


if __name__ == "__main__":
    _get_logger().info("Starting a test fetch of NVDA, AAPL, and TSM")
    test_list = ["NVDA", "AAPL", "TSM"]
    for t in test_list:
        r = fetch_stock_data(t, force_refresh=False)
        if r:
            _print_summary(r)
        else:
            print(f"  {t}: fetch failed")