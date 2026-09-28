# -*- coding: utf-8 -*-
"""
股票數據收集器：透過 yfinance 取得 OHLCV 與基本面欄位，並支援每日 Parquet/JSON 快取。

執行方式（於專案根目錄）：
    python src/data_collector.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf
from tqdm import tqdm

# 從專案根目錄載入 config（直接執行本檔時 sys.path 預設為 src/）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import ALL_STOCKS, CACHE_DIR, LOG_DIR, TICKER_LIST  # noqa: E402

# yfinance info 中要擷取的欄位（對應回傳 dict 的 info 鍵）
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
    """取得已設定檔案與終端機輸出的 logger（單例）。"""
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
    """當日快取目錄 CACHE_DIR / YYYY-MM-DD。"""
    d = datetime.now().strftime("%Y-%m-%d")
    p = Path(CACHE_DIR) / d
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_paths(ticker: str, date_str: str | None = None) -> tuple[Path, Path]:
    """
    取得指定日期資料夾下的 parquet 與 info json 路徑。

    參數:
        ticker: 股票代號。
        date_str: 日期字串 YYYY-MM-DD；None 表示今天。

    回傳:
        (price_parquet_path, info_json_path)
    """
    d = date_str or datetime.now().strftime("%Y-%m-%d")
    base = Path(CACHE_DIR) / d
    sym = ticker.upper()
    return base / f"{sym}.parquet", base / f"{sym}_info.json"


def _extract_info(raw: dict[str, Any] | None) -> dict[str, Any]:
    """從 yfinance 的 info 字典擷取固定欄位，缺漏則為 None。"""
    if not raw:
        return {k: None for k in _INFO_KEYS}
    return {k: raw.get(k) for k in _INFO_KEYS}


def _pick_name(ticker: str, info: dict[str, Any]) -> str:
    """公司顯示名稱：優先 yfinance longName / shortName，其次 config.ALL_STOCKS。"""
    for key in ("longName", "shortName", "displayName"):
        v = info.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ALL_STOCKS.get(ticker.upper(), ticker.upper())


def _pick_current_price(info: dict[str, Any], hist: pd.DataFrame) -> float | None:
    """當前股價：優先 regularMarketPrice / currentPrice，否則用歷史最後一日 Close。"""
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
    """僅保留 OHLCV 欄位（若有 Adj Close 則捨棄）。"""
    cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in hist.columns]
    if not cols:
        return hist.copy()
    out = hist[cols].copy()
    return out


def _load_cache_if_valid(
    ticker: str, force_refresh: bool, log: logging.Logger
) -> dict[str, Any] | None:
    """
    若存在「今天」的快取且未強制重抓，則載入並回傳完整結果 dict；否則回傳 None。

    僅當 parquet 與 info json 皆存在且日期資料夾為今日時視為有效。
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
        log.debug("使用本日快取: %s", sym)
        return {
            "ticker": sym,
            "name": name,
            "price_history": price_history,
            "current_price": current_price,
            "info": info_subset,
            "fetched_at": fetched_at,
        }
    except Exception as e:
        log.warning("讀取快取失敗 %s: %s，將重新抓取", ticker, e)
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
    """寫入當日 parquet 與 _info.json。"""
    today = datetime.now().strftime("%Y-%m-%d")
    base = Path(CACHE_DIR) / today
    base.mkdir(parents=True, exist_ok=True)
    sym = ticker.upper()
    pq_path = base / f"{sym}.parquet"
    js_path = base / f"{sym}_info.json"
    try:
        price_history.to_parquet(pq_path, index=True)
    except Exception as e:
        log.error("寫入 Parquet 失敗 %s: %s", sym, e)
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
    單次呼叫 yfinance（不內建重試）。

    回傳:
        (price_history OHLCV, info 完整 dict, 公司名稱, 當前股價)
    """
    sym = ticker.upper()
    t = yf.Ticker(sym)
    # 1 年日線；保留傳統 OHLC（非還原權）以符合 OHLCV 描述
    hist = t.history(period="1y", interval="1d", auto_adjust=False)
    if hist.empty:
        raise ValueError(f"{sym}: 無歷史價量資料")
    hist = _trim_ohlcv(hist)
    info_full = t.info or {}
    if not info_full:
        # 部分環境 info 可能空，仍允許用價量繼續
        info_full = {}
    name = _pick_name(sym, info_full)
    current_price = _pick_current_price(info_full, hist)
    return hist, info_full, name, current_price


def _fetch_with_retries(ticker: str, log: logging.Logger) -> tuple[pd.DataFrame, dict[str, Any], str, float | None] | None:
    """
    帶重試的 yfinance 抓取：最多 4 次嘗試（失敗後間隔 5、10、20 秒）。

    回傳:
        成功為 (hist, info_full, name, current_price)；全失敗為 None。
    """
    delays = (5, 10, 20)
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            return _fetch_yfinance_once(ticker)
        except Exception as e:
            last_err = e
            log.warning(
                "抓取 %s 第 %s 次失敗: %s",
                ticker.upper(),
                attempt + 1,
                e,
            )
            if attempt < 3:
                time.sleep(delays[attempt])
    log.error("抓取 %s 已達最大重試次數，放棄: %s", ticker.upper(), last_err)
    return None


def fetch_stock_data(ticker: str, force_refresh: bool = False) -> dict[str, Any] | None:
    """
    取得單一股票的價量與精簡基本面資訊，並使用每日本機快取。

    參數:
        ticker: 股票代號（不分大小寫）。
        force_refresh: True 時忽略快取強制向 yfinance 重新抓取。

    回傳:
        成功時為包含 ticker、name、price_history、current_price、info、fetched_at 的 dict；
        失敗時為 None（不拋出例外）。
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
        # 快取失敗仍回傳記憶體結果
        log.exception("快取寫入失敗，仍回傳記憶體資料: %s", sym)

    return result


def collect_all_stocks(
    tickers: list[str] | None = None,
    force_refresh: bool = False,
) -> dict[str, dict[str, Any] | None]:
    """
    批量抓取多檔股票；預設使用 config.TICKER_LIST。

    參數:
        tickers: 代號列表；None 時使用 TICKER_LIST。
        force_refresh: 是否強制略過快取。

    回傳:
        字典 key 為代號（大寫），value 為 fetch_stock_data 的結果或失敗時的 None。
    """
    log = _get_logger()
    symbols = list(tickers) if tickers is not None else list(TICKER_LIST)
    symbols = [s.upper() for s in symbols]
    out: dict[str, dict[str, Any] | None] = {}

    for i, sym in enumerate(tqdm(symbols, desc="收集股票", unit="檔")):
        try:
            data = fetch_stock_data(sym, force_refresh=force_refresh)
            out[sym] = data
            if data is None:
                log.error("批量收集中失敗: %s", sym)
        except Exception as e:
            log.exception("批量收集中未預期錯誤 %s: %s", sym, e)
            out[sym] = None
        if i < len(symbols) - 1:
            time.sleep(1)

    ok = sum(1 for v in out.values() if v is not None)
    bad = len(out) - ok
    log.info("批量收集完成：成功 %s 檔、失敗 %s 檔", ok, bad)
    print(f"[collect_all_stocks] 成功 {ok} 檔、失敗 {bad} 檔")
    return out


def _print_summary(data: dict[str, Any]) -> None:
    """印出單檔結果摘要（供 __main__ 測試）。"""
    sym = data["ticker"]
    name = data["name"]
    price = data["current_price"]
    rows = len(data["price_history"])
    print(f"  {sym} | {name} | 當前價: {price} | 歷史筆數: {rows} | fetched_at: {data['fetched_at']}")


if __name__ == "__main__":
    _get_logger().info("開始測試抓取 NVDA、AAPL、TSM")
    test_list = ["NVDA", "AAPL", "TSM"]
    for t in test_list:
        r = fetch_stock_data(t, force_refresh=False)
        if r:
            _print_summary(r)
        else:
            print(f"  {t}: 抓取失敗")