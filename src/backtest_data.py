# -*- coding: utf-8 -*-
"""
回測用價量矩陣：把多檔 OHLCV 抓成可對齊的日線面板，並做本地快取。

技術面分數與報酬都用同一份還原價（auto_adjust=True）。
未還原的收盤價遇到分割會像崩盤，回測報酬會失真。

快取目錄：data/backtest_cache/ohlcv/{TICKER}.parquet
已涵蓋請求區間就不再打 yfinance。

執行方式（於專案根目錄）：
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
# 快取最後一天比請求迄日早這麼多天以內，視為已涵蓋（吸收週末與尚未收盤）
_CACHE_END_TOLERANCE_DAYS = 7


def _coerce_date(value: date | datetime | str, name: str) -> date:
    """
    把 date / datetime / ISO 字串轉成 date。

    參數:
        value: 日期輸入。
        name: 參數名稱，用於錯誤訊息。

    回傳:
        datetime.date。
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
            raise ValueError(f"{name} 不是合法日期：{value!r}，請用 YYYY-MM-DD") from exc
    raise TypeError(f"{name} 須為 date，收到 {type(value).__name__}")


def _normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """
    整理 yfinance 日線：去掉時區、只留 OHLCV、同一天留最後一筆。

    參數:
        raw: yfinance history 的結果。

    回傳:
        索引為無時區日期的 OHLCV。空表表示這份資料不能用。
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
    """單一代號的 parquet 路徑。"""
    return _OHLCV_DIR / f"{ticker.upper()}.parquet"


def _read_cache(ticker: str) -> pd.DataFrame | None:
    """讀取本地日線。壞檔回傳 None，不中斷其他代號。"""
    path = _cache_path(ticker)
    if not path.is_file():
        return None
    try:
        cached = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        _log.warning("讀取 %s 價量快取失敗，將重新下載: %s", ticker, exc)
        return None
    cached = _normalize_ohlcv(cached)
    if cached.empty:
        return None
    return cached


def _cache_covers(cached: pd.DataFrame, start: date, end: date) -> bool:
    """快取的日期是否包住請求區間。迄日允許晚幾個日曆天。"""
    first = cached.index.min().date()
    last = cached.index.max().date()
    return first <= start and last >= end - timedelta(days=_CACHE_END_TOLERANCE_DAYS)


def _download_ohlcv(ticker: str, start: date, end: date) -> pd.DataFrame | None:
    """
    向 yfinance 抓單一代號日線，失敗重試 3 次。

    參數:
        ticker: 股票代號。
        start: 起始日（含）。
        end: 結束日（含）。yfinance 的 end 是不含，所以這裡加一天。

    回傳:
        整理後的 OHLCV；三次都失敗或沒有資料則為 None。
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
                last_err = ValueError("沒有可用的收盤價")
            else:
                return frame
        except Exception as exc:  # noqa: BLE001 — 單檔失敗改重試
            last_err = exc
        if attempt < 2:
            time.sleep(_YF_RETRY_DELAYS[attempt])
    _log.warning("抓取 %s 價量失敗（已重試 3 次）: %s", ticker, last_err)
    return None


def load_ohlcv(ticker: str, start: date, end: date) -> pd.DataFrame | None:
    """
    取得單一代號在區間內的 OHLCV。本地快取夠用就不再下載。

    參數:
        ticker: 股票代號。
        start: 起始日（含）。
        end: 結束日（含）。

    回傳:
        區間內的 OHLCV；完全沒有資料則為 None。
    """
    symbol = ticker.upper().strip()
    start_d = _coerce_date(start, "start")
    end_d = _coerce_date(end, "end")
    if start_d > end_d:
        raise ValueError(f"{symbol} 的 start ({start_d}) 晚於 end ({end_d})")

    cached = _read_cache(symbol)
    if cached is not None and _cache_covers(cached, start_d, end_d):
        sliced = cached.loc[pd.Timestamp(start_d) : pd.Timestamp(end_d)]
        _log.debug("使用價量快取 %s（%s 列）", symbol, len(sliced))
        return sliced if not sliced.empty else None

    _OHLCV_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = _download_ohlcv(symbol, start_d, end_d)
    if downloaded is None:
        if cached is not None:
            _log.warning("%s 重新下載失敗，沿用較舊的快取", symbol)
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
        _log.warning("寫入 %s 價量快取失敗，仍使用記憶體資料: %s", symbol, exc)
    sliced = merged.loc[pd.Timestamp(start_d) : pd.Timestamp(end_d)]
    return sliced if not sliced.empty else None


def build_ohlcv_panel(
    tickers: list[str],
    start: date,
    end: date,
) -> dict[str, pd.DataFrame]:
    """
    建立多檔 OHLCV 面板，供技術面分數與報酬共用。

    某一檔抓不到就略過並記警告，不讓整組回測停掉。
    呼叫端若需要均線暖機，應自行把 start 往前移。

    參數:
        tickers: 股票代號。重複與空白會去掉。
        start: 起始日（含）。
        end: 結束日（含）。

    回傳:
        {代號: OHLCV DataFrame}。失敗的代號不會出現在字典裡。
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
            _log.warning("建立 %s 價量失敗: %s", symbol, exc)
            frame = None
        if frame is None or frame.empty:
            _log.warning("%s 在 %s ~ %s 沒有可用價量，這檔不會進入回測", symbol, start_d, end_d)
            continue
        panel[symbol] = frame
        _log.info(
            "%s 價量 %s 列（%s ~ %s）",
            symbol,
            len(frame),
            frame.index.min().date().isoformat(),
            frame.index.max().date().isoformat(),
        )
    _log.info("價量面板完成：請求 %s 檔，成功 %s 檔", len(symbols), len(panel))
    return panel


if __name__ == "__main__":
    today = date.today()
    start = today - timedelta(days=400)
    panel = build_ohlcv_panel(["SPY", "NVDA"], start, today)
    for symbol, frame in panel.items():
        print(
            f"{symbol}: {len(frame)} 列, "
            f"{frame.index.min().date()} ~ {frame.index.max().date()}, "
            f"最後收盤 {float(frame['Close'].iloc[-1]):.2f}",
            flush=True,
        )
    missing = [symbol for symbol in ("SPY", "NVDA") if symbol not in panel]
    if missing:
        print("缺少:", ", ".join(missing), flush=True)
        sys.exit(1)
