# -*- coding: utf-8 -*-
"""
回測用 S&P 500 時點股票池。涵蓋 NYSE 與 NASDAQ，對應「美國主流大型股」母體。

與 backtest_universe.py（NASDAQ-100）平行，不取代那份模組。
今天手動挑的 60 檔有不少根本不在 NASDAQ 上市，NASDAQ-100 從一開始就不是合適基準。

資料來源
--------
GitHub fja05680/sp500（MIT）。執行時先列 repo 檔案，優先用檔名帶日期、
且日期最新的「Historical Components」CSV；目前 repo 沒有這種檔名，
因此使用 README 指定的最新歷史檔：

    S&P 500 Historical Components & Changes (Updated).csv

原始檔是逐行 ``date,tickers``，tickers 為當天全部成分股、逗號分隔、整欄加引號。
不是每個交易日都有一列，而是指數調整發生時才記一筆完整名單。

已知限制（作者 README 自己寫的，這不是無誤差的權威資料）
--------------------------------------------------------
- 底稿來自 Andreas Clenow《Trading Evolved》附檔（1996–2019），其後由作者
  對照 Wikipedia「Selected Changes」手工補上。Wikipedia 那一節不是完整變更清單，
  日期與事件可能缺漏。
- 1996 年起前幾年一列往往少於 500 檔（最早約 487）。作者認為大約 2001 年以前
  可能有缺符號，無法獨立核對。
- 新增與剔除的空窗會讓檔數不是剛好 500，有時略多（例如 505）、有時略少。
- 代號是當天實際交易代號。改名（FB 與 META）不會自動視為同一家公司。
- 檔案最後一列之後的調整還沒被收進來。查詢日晚於最後一列時，只能沿用該列，
  不能假裝已經涵蓋到今天。

執行方式（於專案根目錄）：
    python src/backtest_universe_sp500.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
import yfinance as yf

_ROOT = Path(__file__).resolve().parent.parent
_SRC = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import DATA_DIR  # noqa: E402

_log = logging.getLogger("backtest_universe_sp500")
if not _log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

_REPO_API = "https://api.github.com/repos/fja05680/sp500/contents/"
_RAW_BASE = "https://raw.githubusercontent.com/fja05680/sp500/master/"
# README 指定的維護中歷史檔。repo 沒有「檔名帶更新日期」時用這份。
_UPDATED_FILENAME = "S&P 500 Historical Components & Changes (Updated).csv"
_CACHE_DIR = Path(DATA_DIR) / "backtest_cache"
_CACHE_CSV = _CACHE_DIR / "sp500_historical.csv"
_CACHE_META = _CACHE_DIR / "sp500_historical.meta.json"
_CACHE_MAX_AGE = timedelta(days=7)
_HTTP_TIMEOUT_SEC = 60
_HTTP_RETRY_DELAYS = (1.0, 2.0, 4.0)
_YF_CALL_SLEEP_SEC = 0.7
_YF_RETRY_DELAYS = (1.0, 2.0, 4.0)

# 季度取樣對齊季初，與 NASDAQ-100 模組同一套時點。
_FREQ_TO_PANDAS: dict[str, str] = {
    "quarterly": "QS",
    "monthly": "MS",
    "yearly": "YS",
    "daily": "D",
}

# yfinance info["country"] 視為美國公司的寫法。
_US_COUNTRIES = frozenset({"UNITED STATES", "USA", "US", "U.S.", "U.S.A."})
# 檔名裡的 (MM-DD-YYYY)，例如 Changes(08-01-2026).csv。
_FILENAME_DATE = re.compile(r"\((\d{2})-(\d{2})-(\d{4})\)")

_HISTORY: pd.DataFrame | None = None
_STALE_WARNED = False


class UniverseCoverageError(RuntimeError):
    """查詢日早於 S&P 500 歷史檔起點，或該筆快照是空的。"""


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


def _validate_range(start: date, end: date) -> None:
    """起始日不可晚於結束日。"""
    if start > end:
        raise ValueError(
            f"start_date ({start.isoformat()}) 晚於 end_date ({end.isoformat()})"
        )


def _http_get(url: str, *, as_json: bool = False) -> bytes | list | dict:
    """
    GET 一個 URL，失敗重試 3 次。

    參數:
        url: 完整網址。
        as_json: True 時回傳解析後的 JSON。

    回傳:
        回應本文，或 JSON 物件。
    """
    last_err: Exception | None = None
    headers = {"User-Agent": "stock-analysis-backtest", "Accept": "application/vnd.github+json"}
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=_HTTP_TIMEOUT_SEC)
            resp.raise_for_status()
            if as_json:
                return resp.json()
            return resp.content
        except Exception as exc:  # noqa: BLE001 — 網路失敗改重試
            last_err = exc
            _log.warning("下載失敗第 %s 次 %s: %s", attempt + 1, url, exc)
            if attempt < 2:
                time.sleep(_HTTP_RETRY_DELAYS[attempt])
    raise RuntimeError(f"下載失敗（已重試 3 次）：{url}。{last_err}") from last_err


def _filename_stamp(name: str) -> date | None:
    """
    從檔名取出 (MM-DD-YYYY)。沒有日期則回傳 None。

    參數:
        name: repo 裡的檔名。

    回傳:
        檔名上的日期，或 None。
    """
    match = _FILENAME_DATE.search(name)
    if match is None:
        return None
    month, day, year = (int(part) for part in match.groups())
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _raw_url(filename: str) -> str:
    """把 repo 檔名轉成 raw.githubusercontent.com 網址。"""
    return _RAW_BASE + quote(filename)


def _resolve_remote_file() -> tuple[str, str]:
    """
    列出 fja05680/sp500 的檔案，決定要下載哪一份歷史成分股 CSV。

    優先順序：檔名帶日期、且日期最新的 Historical Components CSV；
    沒有這種檔時，用 README 指定的 ``(Updated)`` 檔。
    列出檔案失敗會直接拋錯，不靜默改抓過期網址。

    回傳:
        (檔名, raw URL)。
    """
    payload = _http_get(_REPO_API, as_json=True)
    if not isinstance(payload, list):
        raise RuntimeError(f"GitHub 檔案列表格式異常：{type(payload).__name__}")

    names = [str(item.get("name") or "") for item in payload if isinstance(item, dict)]
    historical = [
        name
        for name in names
        if name.lower().endswith(".csv") and "historical components" in name.lower()
    ]
    dated = [(stamp, name) for name in historical if (stamp := _filename_stamp(name)) is not None]
    if dated:
        dated.sort()
        filename = dated[-1][1]
        _log.info("使用檔名日期最新的歷史成分股檔：%s", filename)
        return filename, _raw_url(filename)

    if _UPDATED_FILENAME in names:
        _log.info(
            "repo 目前沒有檔名帶日期的 Historical Components CSV，"
            "改用 README 指定的最新歷史檔：%s",
            _UPDATED_FILENAME,
        )
        return _UPDATED_FILENAME, _raw_url(_UPDATED_FILENAME)

    raise RuntimeError(
        "在 fja05680/sp500 找不到 S&P 500 歷史成分股 CSV。"
        f"看到的檔案：{', '.join(names) or '（空）'}"
    )


def _parse_ticker_cell(value: object) -> set[str]:
    """把一格逗號分隔的成分股字串拆成大寫代號集合。"""
    if not isinstance(value, str) or not value.strip():
        return set()
    return {part.strip().upper() for part in value.split(",") if part.strip()}


def _parse_history(path: Path) -> pd.DataFrame:
    """
    讀取快取 CSV，解析成 date 與 tickers 集合。

    參數:
        path: 本地 CSV 路徑。

    回傳:
        依日期排序的 DataFrame，欄位為 date（Timestamp）與 tickers（set[str]）。
    """
    raw = pd.read_csv(path)
    if "date" not in raw.columns or "tickers" not in raw.columns:
        raise ValueError(f"CSV 缺少 date/tickers 欄位，實際欄位：{list(raw.columns)}")

    parsed = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["date"], errors="coerce").dt.normalize(),
            "tickers": raw["tickers"].map(_parse_ticker_cell),
        }
    )
    parsed = parsed.dropna(subset=["date"])
    parsed = parsed.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if parsed.empty:
        raise ValueError(f"{path} 沒有可用的日期列")

    sizes = parsed["tickers"].map(len)
    median = float(sizes.median())
    if median < 450:
        raise ValueError(
            f"成分股數量中位數只有 {median:.0f}，CSV 的 tickers 欄可能沒被當成單一欄位解析"
        )
    empty_rows = int((sizes == 0).sum())
    if empty_rows:
        _log.warning("有 %s 列的成分股是空的，這些日期不能當作有效快照", empty_rows)
    return parsed


def _read_fetched_at() -> datetime | None:
    """讀取快取旁的抓取時間。沒有 meta 時改看檔案修改時間。"""
    if _CACHE_META.is_file():
        try:
            meta = json.loads(_CACHE_META.read_text(encoding="utf-8"))
            text = str(meta.get("fetched_at") or "")
            if text:
                return datetime.fromisoformat(text)
        except Exception as exc:  # noqa: BLE001
            _log.warning("讀取快取時間失敗，改看檔案修改時間: %s", exc)
    if _CACHE_CSV.is_file():
        return datetime.fromtimestamp(_CACHE_CSV.stat().st_mtime, tz=timezone.utc)
    return None


def _cache_is_fresh() -> bool:
    """本地 CSV 存在，且抓取時間在 7 天以內。"""
    if not _CACHE_CSV.is_file():
        return False
    fetched = _read_fetched_at()
    if fetched is None:
        return False
    if fetched.tzinfo is None:
        fetched = fetched.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - fetched
    return age <= _CACHE_MAX_AGE


def _write_meta(filename: str, url: str) -> None:
    """寫下這次抓取的來源與時間，供 7 日快取判斷。"""
    meta = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source_filename": filename,
        "source_url": url,
    }
    _CACHE_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def download_sp500_history() -> pd.DataFrame:
    """
    下載 S&P 500 歷史成分股，並快取到 data/backtest_cache/sp500_historical.csv。

    本地已有、且抓取日期在 7 天以內，就直接讀快取、不再打 GitHub。
    快取損壞時會重新下載，不沿用解析失敗的檔案。

    回傳:
        DataFrame，欄位 date、tickers（set[str]），一列是一次指數調整後的完整名單。
    """
    global _HISTORY
    if _HISTORY is not None:
        return _HISTORY

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if _cache_is_fresh():
        try:
            parsed = _parse_history(_CACHE_CSV)
        except Exception as exc:  # noqa: BLE001
            _log.warning("本地快取無法解析，改為重新下載: %s", exc)
        else:
            _HISTORY = parsed
            first = parsed["date"].iloc[0].date()
            last = parsed["date"].iloc[-1].date()
            _log.info(
                "使用 7 日內快取 %s（%s 列，%s ~ %s）",
                _CACHE_CSV.name,
                len(parsed),
                first.isoformat(),
                last.isoformat(),
            )
            return parsed

    filename, url = _resolve_remote_file()
    content = _http_get(url)
    if not isinstance(content, (bytes, bytearray)) or not content:
        raise RuntimeError(f"下載到的內容是空的：{url}")
    _CACHE_CSV.write_bytes(content)
    _write_meta(filename, url)
    parsed = _parse_history(_CACHE_CSV)
    _HISTORY = parsed
    first = parsed["date"].iloc[0].date()
    last = parsed["date"].iloc[-1].date()
    _log.info(
        "已下載 %s → %s（%s 列，%s ~ %s）",
        filename,
        _CACHE_CSV,
        len(parsed),
        first.isoformat(),
        last.isoformat(),
    )
    return parsed


def _history_bounds(history: pd.DataFrame) -> tuple[date, date]:
    """回傳歷史檔第一列與最後一列的日期。"""
    return history["date"].iloc[0].date(), history["date"].iloc[-1].date()


def _warn_if_past_coverage(history: pd.DataFrame, end: date) -> None:
    """查詢迄日晚於最後一列時提出警告。之後的調整不在檔案裡。同一次執行只警告一次。"""
    global _STALE_WARNED
    _, last = _history_bounds(history)
    if end > last and not _STALE_WARNED:
        _STALE_WARNED = True
        _log.warning(
            "S&P 500 來源最後一列是 %s，查詢迄日是 %s。"
            "這段空窗的成分股異動還沒進檔案，只能沿用最後一筆有效名單。",
            last.isoformat(),
            end.isoformat(),
        )


def _snapshot_as_of(history: pd.DataFrame, as_of: date) -> set[str]:
    """
    取「小於等於 as_of 的最近一列」成分股。

    原始資料是事件觸發，不是每天一列。兩次調整之間，沿用前一筆完整名單。

    參數:
        history: download_sp500_history 的結果。
        as_of: 查詢日。

    回傳:
        該日有效的成分股集合。
    """
    position = int(history["date"].searchsorted(pd.Timestamp(as_of), side="right")) - 1
    if position < 0:
        first, _last = _history_bounds(history)
        raise UniverseCoverageError(
            f"無法取得 {as_of.isoformat()} 的 S&P 500 成分股："
            f"早於資料起點 {first.isoformat()}。"
        )
    symbols = history["tickers"].iloc[position]
    row_date = history["date"].iloc[position].date()
    if not symbols:
        raise UniverseCoverageError(
            f"無法取得 {as_of.isoformat()} 的 S&P 500 成分股："
            f"最近一筆有效列 {row_date.isoformat()} 是空名單。"
        )
    return set(symbols)


def _sample_dates(start: date, end: date, freq: str) -> list[date]:
    """
    在區間內依頻率產生取樣日。回測起點若不是頻率邊界，仍會納入。

    參數:
        start: 區間起日（含）。
        end: 區間迄日（含）。
        freq: quarterly / monthly / yearly / daily。

    回傳:
        由早到晚的 date 列表。
    """
    key = freq.strip().lower()
    pandas_freq = _FREQ_TO_PANDAS.get(key)
    if pandas_freq is None:
        allowed = ", ".join(_FREQ_TO_PANDAS)
        raise ValueError(f"不支援的取樣頻率 {freq!r}，請用 {allowed}")

    stamps = pd.date_range(start=start, end=end, freq=pandas_freq)
    dates = [ts.date() for ts in stamps]
    if not dates or dates[0] > start:
        dates.insert(0, start)

    out: list[date] = []
    seen: set[date] = set()
    for sample in dates:
        if start <= sample <= end and sample not in seen:
            seen.add(sample)
            out.append(sample)
    if not out:
        raise ValueError(
            f"在 {start.isoformat()} 到 {end.isoformat()} 之間沒有取樣日（freq={freq}）"
        )
    return out


def get_point_in_time_sp500(
    start_date: date,
    end_date: date,
    freq: str = "quarterly",
) -> dict[date, set[str]]:
    """
    依頻率取出每個時點當時的 S&P 500 成分股。

    每個取樣日使用 CSV 裡小於等於該日的最近一筆完整名單。
    quarterly 對齊季初，與 NASDAQ-100 模組同一組時點。

    參數:
        start_date: 回測起始日（含）。
        end_date: 回測結束日（含）。
        freq: 取樣頻率，預設 quarterly。

    回傳:
        {取樣日: 當天有效成分股集合}，依日期由早到晚。
    """
    start = _coerce_date(start_date, "start_date")
    end = _coerce_date(end_date, "end_date")
    _validate_range(start, end)
    history = download_sp500_history()
    _warn_if_past_coverage(history, end)

    universe: dict[date, set[str]] = {}
    for sample in _sample_dates(start, end, freq):
        universe[sample] = _snapshot_as_of(history, sample)

    _log.info(
        "S&P 500 時點股票池 %s ~ %s（%s）：%s 個時點",
        start.isoformat(),
        end.isoformat(),
        freq,
        len(universe),
    )
    return universe


def get_union_universe_sp500(start_date: date, end_date: date) -> set[str]:
    """
    回傳整段期間內曾經出現在 S&P 500 的成分股聯集。

    從「起始日當時有效的那一列」開始，併入迄日以前的每一筆事件快照。
    只併季度快照會漏掉季中納入、季末前又被剔除的股票。

    參數:
        start_date: 區間起日（含）。
        end_date: 區間迄日（含）。

    回傳:
        期間內至少在某一筆有效快照裡出現過的代號集合。
    """
    start = _coerce_date(start_date, "start_date")
    end = _coerce_date(end_date, "end_date")
    _validate_range(start, end)
    history = download_sp500_history()
    _warn_if_past_coverage(history, end)

    start_pos = int(history["date"].searchsorted(pd.Timestamp(start), side="right")) - 1
    if start_pos < 0:
        first, _last = _history_bounds(history)
        raise UniverseCoverageError(
            f"無法建立 {start.isoformat()} 起的 S&P 500 聯集：早於資料起點 {first.isoformat()}。"
        )
    end_pos = int(history["date"].searchsorted(pd.Timestamp(end), side="right"))
    window = history.iloc[start_pos:end_pos]
    if window.empty:
        raise UniverseCoverageError(
            f"在 {start.isoformat()} 到 {end.isoformat()} 之間沒有可用的 S&P 500 快照。"
        )

    union: set[str] = set()
    for symbols in window["tickers"]:
        union.update(symbols)
    if not union:
        raise UniverseCoverageError(
            f"{start.isoformat()} 到 {end.isoformat()} 的 S&P 500 聯集是空的。"
        )

    _log.info(
        "S&P 500 歷史成分股聯集 %s ~ %s：%s 檔（%s 筆事件快照）",
        start.isoformat(),
        end.isoformat(),
        len(union),
        len(window),
    )
    return union


def _quote_missing(ticker: str) -> bool:
    """
    近端日線是否完全沒有資料。已下市代號的 info 常常沒有 country。

    參數:
        ticker: 股票代號。

    回傳:
        True 表示沒有可用報價。
    """
    try:
        hist = yf.Ticker(ticker).history(period="5d")
    except Exception as exc:  # noqa: BLE001
        _log.warning("讀取 %s 近期報價失敗: %s", ticker, exc)
        return False
    return hist is None or hist.empty


def _fetch_listing_info(ticker: str) -> tuple[dict[str, str] | None, str]:
    """
    讀取 yfinance info 的 country、exchange 與公司名。暫時性失敗重試 3 次。

    exchange 只代表掛牌場所。TSM 與 DIS 都是 NYQ，單看 exchange 會把外國 ADR
    誤判成美國公司，所以國籍以 country 為準，exchange 只一併記下來。

    參數:
        ticker: 股票代號。

    回傳:
        (欄位 dict, 失敗原因)。成功時失敗原因為空字串。
    """
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            time.sleep(_YF_CALL_SLEEP_SEC)
            info = yf.Ticker(ticker).info or {}
            if not isinstance(info, dict):
                last_err = ValueError("info 不是 dict")
            else:
                country = str(info.get("country") or "").strip()
                exchange = str(info.get("exchange") or "").strip()
                name = str(info.get("longName") or info.get("shortName") or "").strip()
                if country or exchange:
                    return {"country": country, "exchange": exchange, "name": name}, ""
                if info and _quote_missing(ticker):
                    reason = "Yahoo 無報價（可能已下市），info 沒有 country / exchange"
                    _log.warning("無法判斷 %s 是否為美國公司：%s", ticker, reason)
                    return None, reason
                last_err = ValueError("info 沒有 country / exchange")
        except Exception as exc:  # noqa: BLE001 — 單檔失敗改走重試
            last_err = exc
        if attempt < 2:
            time.sleep(_YF_RETRY_DELAYS[attempt])
    reason = f"已重試 3 次仍失敗：{last_err}"
    _log.warning("無法判斷 %s 是否為美國公司（%s）", ticker, reason)
    return None, reason


def _is_foreign_issuer(country: str, name: str) -> bool | None:
    """
    判斷是不是 S&P 500 通常不收的外國發行人 / ADR。

    參數:
        country: yfinance info['country']。
        name: 公司名稱，用來抓名稱裡就寫著 ADR 的情況。

    回傳:
        True 表示外國或 ADR；False 表示美國公司；None 表示資料不夠、不猜。
    """
    upper_name = name.upper()
    if "ADR" in upper_name or "DEPOSITARY" in upper_name:
        return True
    if not country:
        return None
    return country.strip().upper() not in _US_COUNTRIES


def _split_only_in_current(
    tickers: list[str],
) -> tuple[list[str], list[str], list[str], dict[str, str]]:
    """
    把「只在目前清單」的股票分成不合格的外國發行人，以及沒進過指數的美國公司。

    參數:
        tickers: only_in_current 代號。

    回傳:
        (not_sp500_listed, sp500_but_excluded, eligibility_unknown, unknown_reasons)。
    """
    foreign: list[str] = []
    excluded: list[str] = []
    unknown: list[str] = []
    unknown_reasons: dict[str, str] = {}
    for ticker in tickers:
        info, reason = _fetch_listing_info(ticker)
        if info is None:
            unknown.append(ticker)
            unknown_reasons[ticker] = reason or "沒有 country / exchange"
            continue
        verdict = _is_foreign_issuer(info["country"], info["name"])
        label = (
            f"country={info['country'] or '（空）'} exchange={info['exchange'] or '（空）'}"
        )
        if verdict is None:
            unknown.append(ticker)
            unknown_reasons[ticker] = f"有掛牌場所但沒有 country，無法判斷是否為美國公司（{label}）"
            _log.warning("無法判斷 %s 是否為美國公司：%s", ticker, unknown_reasons[ticker])
            continue
        if verdict:
            foreign.append(ticker)
            _log.info("%s %s → 外國發行人/ADR，S&P 500 通常不合格", ticker, label)
        else:
            excluded.append(ticker)
            _log.info("%s %s → 美國公司，但這段期間沒進過 S&P 500", ticker, label)
    return sorted(foreign), sorted(excluded), sorted(unknown), unknown_reasons


def compare_universes_sp500(
    current_list: list[str],
    point_in_time_union: set[str],
) -> dict[str, list[str] | float | dict[str, str]]:
    """
    對比「今天手動清單」與 S&P 500 歷史成分股聯集。

    overlap_pct 是目前清單中、曾經出現在這段 S&P 500 歷史的比例（0-100）。
    分母是目前清單檔數。代號依字面比對。

    only_in_current 再拆成：
    - not_sp500_listed：yfinance info['country'] 不是美國，或名稱標明 ADR。
      這類公司通常不符合 S&P 500「美國公司」資格，不是 survivorship bias。
      exchange 只記錄掛牌場所，不用來判斷國籍：TSM 與 DIS 的 exchange 都是 NYQ。
    - sp500_but_excluded：美國公司，但這段期間沒進過 S&P 500。
      這才是今天篩、過去不在指數裡的候選。
    查不到 country 的放在 eligibility_unknown，不併入上面兩類。

    參數:
        current_list: 目前手動選股清單。
        point_in_time_union: get_union_universe_sp500 的結果。

    回傳:
        only_in_current、only_in_historical、overlap、overlap_pct、
        not_sp500_listed、sp500_but_excluded、eligibility_unknown、
        eligibility_unknown_reasons。
    """
    current = {str(ticker).upper().strip() for ticker in current_list if str(ticker).strip()}
    historical = {
        str(ticker).upper().strip() for ticker in point_in_time_union if str(ticker).strip()
    }
    overlap = current & historical
    only_current = sorted(current - historical)
    only_historical = historical - current
    overlap_pct = round(100.0 * len(overlap) / len(current), 2) if current else 0.0
    foreign, excluded, unknown, unknown_reasons = _split_only_in_current(only_current)

    return {
        "only_in_current": only_current,
        "only_in_historical": sorted(only_historical),
        "overlap": sorted(overlap),
        "overlap_pct": overlap_pct,
        "not_sp500_listed": foreign,
        "sp500_but_excluded": excluded,
        "eligibility_unknown": unknown,
        "eligibility_unknown_reasons": unknown_reasons,
    }


def _print_ticker_block(title: str, tickers: list[str]) -> None:
    """印出一組代號：先給數量，再逐列輸出。"""
    print(f"{title}（{len(tickers)} 檔）", flush=True)
    if not tickers:
        print("  （無）", flush=True)
        return
    line: list[str] = []
    for ticker in tickers:
        line.append(ticker)
        if len(line) == 12:
            print("  " + ", ".join(line), flush=True)
            line = []
    if line:
        print("  " + ", ".join(line), flush=True)


if __name__ == "__main__":
    from backtest_universe import get_union_universe
    from config import TICKER_LIST

    start = date(2022, 1, 1)
    end = date.today()
    history = download_sp500_history()
    first, last = _history_bounds(history)
    print(
        f"S&P 500 時點股票池：{start.isoformat()} ~ {end.isoformat()}，每季一個時點",
        flush=True,
    )
    print(
        f"來源涵蓋 {first.isoformat()} ~ {last.isoformat()}（{len(history)} 筆事件列）。"
        "這份檔來自 Wikipedia 修訂整理，作者說明變更清單可能有缺漏。",
        flush=True,
    )

    universes = get_point_in_time_sp500(start, end, freq="quarterly")
    print("=== 每季成分股數量 ===", flush=True)
    for sample, members in universes.items():
        flag = ""
        if not 490 <= len(members) <= 520:
            flag = "  ← 偏離 500 較多，請對一下來源列"
            _log.warning("%s 的成分股數是 %s，偏離約 500", sample.isoformat(), len(members))
        print(f"  {sample.isoformat()}  {len(members):3d} 檔{flag}", flush=True)

    union = get_union_universe_sp500(start, end)
    quarter_sizes = [len(members) for members in universes.values()]
    print(f"期間聯集大小: {len(union)} 檔", flush=True)
    print(
        f"單季規模: min={min(quarter_sizes)}  max={max(quarter_sizes)}"
        "（指數名義 500 檔，新增剔除空窗會有個位數出入）",
        flush=True,
    )
    if len(union) < max(quarter_sizes):
        raise RuntimeError("聯集小於單季成分股數，時點名單計算有誤")

    report = compare_universes_sp500(TICKER_LIST, union)
    overlap_pct = float(report["overlap_pct"])
    only_current = list(report["only_in_current"])
    only_historical = list(report["only_in_historical"])
    overlap = list(report["overlap"])
    not_listed = list(report["not_sp500_listed"])
    excluded = list(report["sp500_but_excluded"])
    unknown = list(report["eligibility_unknown"])
    classified = len(not_listed) + len(excluded) + len(unknown)
    if classified != len(only_current):
        raise RuntimeError(
            f"only_in_current 拆分數量不符：{classified} != {len(only_current)}"
        )

    print("=== 目前 60 檔 vs 2022 至今 S&P 500 聯集 ===", flush=True)
    print(f"目前清單: {len(TICKER_LIST)} 檔", flush=True)
    print(f"重疊: {len(overlap)} 檔", flush=True)
    print(
        f"overlap_pct: {overlap_pct:.2f}%（目前清單中，曾經進入這段歷史 S&P 500 的比例）",
        flush=True,
    )
    _print_ticker_block("只在目前清單、這段歷史從沒進過 S&P 500", only_current)
    _print_ticker_block("重疊", overlap)
    _print_ticker_block("只在歷史 S&P 500、目前清單沒有", only_historical)

    print("=== only_in_current 依是否像 S&P 500 合格公司拆分 ===", flush=True)
    print(
        "not_sp500_listed：country 不是美國，或名稱標明 ADR。"
        "S&P 500 通常只收美國公司，這不是 survivorship bias。",
        flush=True,
    )
    _print_ticker_block("not_sp500_listed", not_listed)
    print(
        "sp500_but_excluded：美國公司，但這段期間沒進過 S&P 500。"
        "這才是今天篩、過去不在指數裡。",
        flush=True,
    )
    _print_ticker_block("sp500_but_excluded", excluded)
    if unknown:
        reasons = report["eligibility_unknown_reasons"]
        reason_map = reasons if isinstance(reasons, dict) else {}
        print(
            "eligibility_unknown：yfinance 沒有足夠的 country，未歸入上面兩類。",
            flush=True,
        )
        print(f"eligibility_unknown（{len(unknown)} 檔）", flush=True)
        for ticker in unknown:
            print(f"  {ticker}（{reason_map.get(ticker, '沒有 country')}）", flush=True)

    current = {str(ticker).upper() for ticker in TICKER_LIST}
    nasdaq_union = get_union_universe(start, end)
    nasdaq_overlap = current & nasdaq_union
    nasdaq_pct = round(100.0 * len(nasdaq_overlap) / len(current), 2) if current else 0.0
    print("=== overlap_pct 對照（目前清單中，曾經進入該指數的比例）===", flush=True)
    print(f"  NASDAQ-100   {nasdaq_pct:6.2f}%   （重疊 {len(nasdaq_overlap)} / {len(current)}）", flush=True)
    print(f"  S&P 500      {overlap_pct:6.2f}%   （重疊 {len(overlap)} / {len(current)}）", flush=True)
    print(
        "代號依字面比對。exchange 不能區分 ADR：TSM 與 DIS 都是 NYQ，國籍看的是 country。",
        flush=True,
    )
