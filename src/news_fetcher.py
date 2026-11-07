# -*- coding: utf-8 -*-
"""
Stock news fetcher: parses old and new yfinance news shapes and caches them locally each day.

Run from the project root:
    python src/news_fetcher.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yfinance as yf
from tqdm import tqdm

# Load config from the project root (sys.path defaults to src/ when this file is run directly).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import CACHE_DIR, LOG_DIR  # noqa: E402

_logger: logging.Logger | None = None


def _get_logger() -> logging.Logger:
    """Return the news-module logger (console and the logs directory)."""
    global _logger
    if _logger is not None:
        return _logger

    log = logging.getLogger("news_fetcher")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()

    today_str = datetime.now().strftime("%Y-%m-%d")
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"news_fetcher_{today_str}.log"

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


def _news_cache_path(ticker: str) -> Path:
    """Today's news cache path: CACHE_DIR/YYYY-MM-DD/news/{ticker}.json."""
    today = datetime.now().strftime("%Y-%m-%d")
    p = Path(CACHE_DIR) / today / "news"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{ticker.upper()}.json"


def _safe_parse_datetime(value: Any) -> datetime | None:
    """Turn as many time formats as possible into a UTC-aware datetime."""
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, str):
            v = value.strip()
            if not v:
                return None
            if v.endswith("Z"):
                v = v[:-1] + "+00:00"
            dt = datetime.fromisoformat(v)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
    except Exception:  # noqa: BLE001
        return None
    return None


def _clean_text(text: str) -> str:
    """
    Clean text:
    - strip HTML tags
    - collapse extra spaces and newlines
    - truncate to 500 characters
    """
    s = "" if text is None else str(text)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > 500:
        s = s[:500].rstrip() + "..."
    return s


def _parse_news_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """Parse one news item. Compatible with the old and new yfinance shapes."""
    if not isinstance(item, dict):
        return None

    title = ""
    publisher = ""
    link = ""
    summary = ""
    related: list[str] = []
    published_dt: datetime | None = None

    # Shape A (older yfinance).
    if "title" in item or "providerPublishTime" in item:
        try:
            title = _clean_text(str(item.get("title", "")))
        except Exception:  # noqa: BLE001
            title = ""
        try:
            publisher = _clean_text(str(item.get("publisher", "")))
        except Exception:  # noqa: BLE001
            publisher = ""
        try:
            link = str(item.get("link", "") or "").strip()
        except Exception:  # noqa: BLE001
            link = ""
        try:
            summary = _clean_text(str(item.get("summary", "")))
        except Exception:  # noqa: BLE001
            summary = ""
        try:
            rt = item.get("relatedTickers", [])
            if isinstance(rt, list):
                related = [str(x).upper() for x in rt if x]
        except Exception:  # noqa: BLE001
            related = []
        published_dt = _safe_parse_datetime(item.get("providerPublishTime"))

    # Shape B (yfinance 1.3.0).
    content = item.get("content")
    if isinstance(content, dict):
        try:
            title = _clean_text(str(content.get("title", title or "")))
        except Exception:  # noqa: BLE001
            pass
        try:
            summary = _clean_text(str(content.get("summary", summary or "")))
        except Exception:  # noqa: BLE001
            pass
        try:
            provider = content.get("provider", {})
            if isinstance(provider, dict):
                publisher = _clean_text(str(provider.get("displayName", publisher or "")))
        except Exception:  # noqa: BLE001
            pass
        try:
            cu = content.get("canonicalUrl", {})
            if isinstance(cu, dict):
                link = str(cu.get("url", link or "") or "").strip()
        except Exception:  # noqa: BLE001
            pass
        try:
            finance = content.get("finance", {})
            stock_tickers = finance.get("stockTickers", []) if isinstance(finance, dict) else []
            if isinstance(stock_tickers, list):
                tmp: list[str] = []
                for x in stock_tickers:
                    if isinstance(x, dict) and x.get("symbol"):
                        tmp.append(str(x["symbol"]).upper())
                if tmp:
                    related = tmp
        except Exception:  # noqa: BLE001
            pass
        published_dt = _safe_parse_datetime(content.get("pubDate")) or published_dt

    if not title:
        return None
    if published_dt is None:
        # No timestamp means it is treated as old news and dropped, so sorting and the day window stay reliable.
        return None

    published_at = published_dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "title": title,
        "publisher": publisher,
        "link": link,
        "published_at": published_at,
        "summary": summary,
        "tickers_related": related,
        "_published_dt": published_dt,  # Used only for sorting. Removed before the file is written.
    }


def _load_cache(path: Path) -> list[dict[str, Any]] | None:
    """Read today's cached JSON."""
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:  # noqa: BLE001
        return None
    return None


def _save_cache(path: Path, news: list[dict[str, Any]]) -> None:
    """Write today's cached JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(news, f, ensure_ascii=False, indent=2)


def fetch_stock_news(ticker: str, max_items: int = 10, force_refresh: bool = False) -> list[dict[str, Any]]:
    """
    Fetch news for one stock from yfinance and return a common shape.

    Args:
        ticker: Ticker symbol.
        max_items: Maximum number of items to return.
        force_refresh: When True, skip today's cache and fetch again.

    Returns:
        list[dict] with title, publisher, link, published_at, summary, and tickers_related.
    """
    log = _get_logger()
    sym = ticker.upper()
    cache_path = _news_cache_path(sym)
    max_n = max(0, int(max_items))

    if not force_refresh:
        cached = _load_cache(cache_path)
        if cached is not None:
            log.info("Using the news cache: %s (%s items)", sym, len(cached))
            return cached[:max_n] if max_n else []

    try:
        raw_news = yf.Ticker(sym).news or []
    except Exception as e:  # noqa: BLE001
        log.error("Failed to fetch news for %s: %s", sym, e)
        return []

    if not isinstance(raw_news, list) or not raw_news:
        log.info("%s has no news", sym)
        _save_cache(cache_path, [])
        return []

    parsed: list[dict[str, Any]] = []
    seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
    for item in raw_news:
        one = _parse_news_item(item)
        if not one:
            continue
        dt = one.get("_published_dt")
        if not isinstance(dt, datetime):
            continue
        if dt < seven_days_ago:
            continue
        parsed.append(one)

    parsed.sort(key=lambda x: x["_published_dt"], reverse=True)
    cleaned: list[dict[str, Any]] = []
    for n in parsed[:max_n]:
        n.pop("_published_dt", None)
        cleaned.append(n)

    _save_cache(cache_path, cleaned)
    log.info("Fetched %s news items for %s", len(cleaned), sym)
    return cleaned


def fetch_news_batch(
    tickers: list[str],
    max_per_ticker: int = 8,
    force_refresh: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """
    Fetch news for many symbols.

    - Sleep 0.5 seconds after each request to avoid rate limits
    - A failure returns [] for that symbol
    """
    log = _get_logger()
    out: dict[str, list[dict[str, Any]]] = {}
    symbols = [str(t).upper() for t in tickers]
    for i, sym in enumerate(tqdm(symbols, desc="News", unit="name")):
        try:
            out[sym] = fetch_stock_news(sym, max_items=max_per_ticker, force_refresh=force_refresh)
        except Exception as e:  # noqa: BLE001
            log.error("Batch fetch failed for %s: %s", sym, e)
            out[sym] = []
        if i < len(symbols) - 1:
            time.sleep(0.5)
    return out


if __name__ == "__main__":
    # A short test
    sample = ["NVDA", "AAPL", "TSM"]
    res = fetch_news_batch(sample, max_per_ticker=5, force_refresh=False)
    for t in sample:
        print(f"{t}: {len(res.get(t, []))} items")
