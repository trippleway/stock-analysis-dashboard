# -*- coding: utf-8 -*-
"""
股票新聞抓取器：支援 yfinance 新舊 news 結構解析與每日本地快取。

執行方式（於專案根目錄）：
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

# 從專案根目錄載入 config（直接執行本檔時 sys.path 預設為 src/）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import CACHE_DIR, LOG_DIR  # noqa: E402

_logger: logging.Logger | None = None


def _get_logger() -> logging.Logger:
    """取得新聞模組 logger（同時輸出到 console 與 logs）。"""
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
    """回傳當日新聞快取路徑：CACHE_DIR/YYYY-MM-DD/news/{ticker}.json。"""
    today = datetime.now().strftime("%Y-%m-%d")
    p = Path(CACHE_DIR) / today / "news"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{ticker.upper()}.json"


def _safe_parse_datetime(value: Any) -> datetime | None:
    """盡量把多種格式時間轉成 UTC aware datetime。"""
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
    清理文字內容：
    - 去除 HTML 標籤
    - 合併多餘空白與換行
    - 長度截斷 500 字元
    """
    s = "" if text is None else str(text)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > 500:
        s = s[:500].rstrip() + "..."
    return s


def _parse_news_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """解析單筆新聞，兼容 yfinance 舊版/新版結構。"""
    if not isinstance(item, dict):
        return None

    title = ""
    publisher = ""
    link = ""
    summary = ""
    related: list[str] = []
    published_dt: datetime | None = None

    # 結構 A（舊版）
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

    # 結構 B（新版 1.3.0）
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
        # 若沒有時間，視為舊新聞過濾掉（避免排序與天數判斷不準）
        return None

    published_at = published_dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "title": title,
        "publisher": publisher,
        "link": link,
        "published_at": published_at,
        "summary": summary,
        "tickers_related": related,
        "_published_dt": published_dt,  # 內部排序用，寫入前會移除
    }


def _load_cache(path: Path) -> list[dict[str, Any]] | None:
    """讀取當日快取 JSON。"""
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
    """寫入當日快取 JSON。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(news, f, ensure_ascii=False, indent=2)


def fetch_stock_news(ticker: str, max_items: int = 10, force_refresh: bool = False) -> list[dict[str, Any]]:
    """
    抓取單一股票新聞（yfinance），並回傳統一格式。

    參數:
        ticker: 股票代號。
        max_items: 最多回傳筆數。
        force_refresh: True 時略過當日快取重抓。

    回傳:
        list[dict]，欄位為 title/publisher/link/published_at/summary/tickers_related。
    """
    log = _get_logger()
    sym = ticker.upper()
    cache_path = _news_cache_path(sym)
    max_n = max(0, int(max_items))

    if not force_refresh:
        cached = _load_cache(cache_path)
        if cached is not None:
            log.info("使用新聞快取：%s (%s 筆)", sym, len(cached))
            return cached[:max_n] if max_n else []

    try:
        raw_news = yf.Ticker(sym).news or []
    except Exception as e:  # noqa: BLE001
        log.error("抓取 %s 新聞失敗：%s", sym, e)
        return []

    if not isinstance(raw_news, list) or not raw_news:
        log.info("%s 無新聞資料", sym)
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
    log.info("成功抓取 %s 新聞 %s 筆", sym, len(cleaned))
    return cleaned


def fetch_news_batch(
    tickers: list[str],
    max_per_ticker: int = 8,
    force_refresh: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """
    批量抓取多檔新聞。

    - 每檔請求後 sleep 0.5 秒避免限流
    - 失敗時該檔回傳 []
    """
    log = _get_logger()
    out: dict[str, list[dict[str, Any]]] = {}
    symbols = [str(t).upper() for t in tickers]
    for i, sym in enumerate(tqdm(symbols, desc="抓取新聞", unit="檔")):
        try:
            out[sym] = fetch_stock_news(sym, max_items=max_per_ticker, force_refresh=force_refresh)
        except Exception as e:  # noqa: BLE001
            log.error("批量抓取 %s 失敗：%s", sym, e)
            out[sym] = []
        if i < len(symbols) - 1:
            time.sleep(0.5)
    return out


if __name__ == "__main__":
    # 簡單測試
    sample = ["NVDA", "AAPL", "TSM"]
    res = fetch_news_batch(sample, max_per_ticker=5, force_refresh=False)
    for t in sample:
        print(f"{t}: {len(res.get(t, []))} 則")
