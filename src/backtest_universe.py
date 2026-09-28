# -*- coding: utf-8 -*-
"""
回測用時點股票池：用「該日期當時」的 NASDAQ-100 成分股，避開 survivorship bias。

今天手動挑的 60 檔，是被這幾年漲幅篩過的贏家。拿這份名單回測 3 年前，
等於偷看了未來。本模組改問 nasdaq-100-ticker-history：
某個歷史日期當天，指數裡實際有哪些股票。

依賴（PyPI 目前沒有發佈版，需從 GitHub 安裝）：
    pip install "nasdaq-100-ticker-history @ git+https://github.com/jmccarrell/n100tickers.git"

執行方式（於專案根目錄）：
    python src/backtest_universe.py
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

try:
    from nasdaq_100_ticker_history import tickers_as_of
except ImportError as exc:  # pragma: no cover - 環境缺套件時給安裝指令
    raise ImportError(
        "找不到 nasdaq-100-ticker-history。請安裝："
        ' pip install "nasdaq-100-ticker-history @ git+https://github.com/jmccarrell/n100tickers.git"'
    ) from exc

_log = logging.getLogger("backtest_universe")
if not _log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# 取樣頻率對應 pandas offset。季度用季初（1/1、4/1、7/1、10/1），
# 讓「2023 Q1」對上 2023-01-01 當天的真實成分股，而不是季末才看。
_FREQ_TO_PANDAS: dict[str, str] = {
    "quarterly": "QS",
    "monthly": "MS",
    "yearly": "YS",
    "daily": "D",
}

# yfinance info["exchange"] 的 NASDAQ 掛牌代碼。
# NMS / NGM / NCM 分別是 Nasdaq Global Select / Global / Capital。
_NASDAQ_EXCHANGE_CODES = frozenset({"NMS", "NGM", "NCM", "NAS", "NASDAQ"})
_YF_CALL_SLEEP_SEC = 0.7
_YF_RETRY_DELAYS = (1.0, 2.0, 4.0)


class UniverseCoverageError(RuntimeError):
    """查詢日超出 nasdaq-100-ticker-history 涵蓋範圍，或套件回傳空名單。"""


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


def _membership_as_of(as_of: date) -> set[str]:
    """
    取得單一日期的 NASDAQ-100 成分股。

    套件缺該年資料時會丟 NotImplementedError；這裡改成
    UniverseCoverageError，訊息寫明是哪個日期超出涵蓋範圍。
    回傳空集合同樣視為失敗，避免回測靜默用到空股票池。

    參數:
        as_of: 查詢日（含當天生效的指數調整）。

    回傳:
        當天成分股代號集合（大寫）。
    """
    label = as_of.isoformat()
    try:
        raw = tickers_as_of(as_of.year, as_of.month, as_of.day)
    except NotImplementedError as exc:
        message = (
            f"無法取得 {label} 的 NASDAQ-100 成分股："
            f"該日期超出 nasdaq-100-ticker-history 涵蓋範圍。原始錯誤：{exc}"
        )
        _log.error(message)
        raise UniverseCoverageError(message) from exc
    except Exception as exc:
        message = f"無法取得 {label} 的 NASDAQ-100 成分股：{exc}"
        _log.error(message)
        raise UniverseCoverageError(message) from exc

    symbols = {str(ticker).upper().strip() for ticker in raw if str(ticker).strip()}
    if not symbols:
        message = (
            f"無法取得 {label} 的 NASDAQ-100 成分股："
            "套件回傳空集合，視為超出涵蓋範圍或資料異常，拒絕靜默繼續。"
        )
        _log.error(message)
        raise UniverseCoverageError(message)
    return symbols


def _sample_dates(start: date, end: date, freq: str) -> list[date]:
    """
    在區間內依頻率產生取樣日。回測起點若不是頻率邊界，仍會納入，
    否則起點到第一個取樣日之間沒有股票池。

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


def get_point_in_time_universe(
    start_date: date,
    end_date: date,
    freq: str = "quarterly",
) -> dict[date, set[str]]:
    """
    依頻率取出每個時點「當天」的 NASDAQ-100 成分股。

    quarterly 對齊季初：2023 Q1 用 2023-01-01 的名單，2024 Q1 用 2024-01-01，
    全程不共用同一份清單。任一取樣日超出套件涵蓋範圍會直接拋錯。

    參數:
        start_date: 回測起始日（含）。
        end_date: 回測結束日（含）。
        freq: 取樣頻率，預設 quarterly。

    回傳:
        {取樣日: 當天成分股集合}，依日期由早到晚。
    """
    start = _coerce_date(start_date, "start_date")
    end = _coerce_date(end_date, "end_date")
    _validate_range(start, end)

    samples = _sample_dates(start, end, freq)
    universe: dict[date, set[str]] = {}
    for sample in samples:
        universe[sample] = _membership_as_of(sample)

    _log.info(
        "時點股票池 %s ~ %s（%s）：%s 個時點",
        start.isoformat(),
        end.isoformat(),
        freq,
        len(universe),
    )
    return universe


def get_union_universe(start_date: date, end_date: date) -> set[str]:
    """
    回傳整段期間內曾經進過 NASDAQ-100 的成分股聯集。

    用逐日成分股取聯集，而不是只併每季快照。季中才納入、季末前又被剔除的
    股票，季度快照會漏掉；抓歷史價格時範圍會不夠。

    參數:
        start_date: 區間起日（含）。
        end_date: 區間迄日（含）。

    回傳:
        期間內至少出現過一天的代號集合。
    """
    start = _coerce_date(start_date, "start_date")
    end = _coerce_date(end_date, "end_date")
    _validate_range(start, end)

    union: set[str] = set()
    cursor = start
    step = timedelta(days=1)
    while cursor <= end:
        union.update(_membership_as_of(cursor))
        cursor += step

    _log.info(
        "歷史成分股聯集 %s ~ %s：%s 檔",
        start.isoformat(),
        end.isoformat(),
        len(union),
    )
    return union


def _is_nasdaq_listed(exchange: str) -> bool:
    """
    依 yfinance info['exchange'] 判斷是否在 NASDAQ 掛牌。

    參數:
        exchange: Yahoo 交易所代碼，例如 NMS、NYQ。

    回傳:
        True 表示 NASDAQ 掛牌。
    """
    code = exchange.strip().upper()
    if code in _NASDAQ_EXCHANGE_CODES:
        return True
    # 少數環境會回傳 NasdaqGS / NasdaqGM，而不是 NMS / NGM
    return "NASDAQ" in code


def _quote_missing(ticker: str) -> bool:
    """
    看近端日線是否完全沒有資料。Yahoo 對已下市代號常回 404，info 也不會有 exchange。

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


def _fetch_exchange(ticker: str) -> tuple[str | None, str]:
    """
    讀取 yfinance info['exchange']。暫時性失敗重試 3 次，呼叫之間會停頓。

    參數:
        ticker: 股票代號。

    回傳:
        (交易所代碼, 失敗原因)。成功時失敗原因為空字串。
        已下市、報價 404 不會重試，原因會寫明 info 沒有 exchange。
    """
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            time.sleep(_YF_CALL_SLEEP_SEC)
            info = yf.Ticker(ticker).info or {}
            exchange = info.get("exchange") if isinstance(info, dict) else None
            if isinstance(exchange, str) and exchange.strip():
                return exchange.strip(), ""
            # 有回應但沒有 exchange：下市代號重試也不會補上這個欄位
            if isinstance(info, dict) and info and _quote_missing(ticker):
                reason = "Yahoo 無報價（可能已下市），info.exchange 為空"
                _log.warning("無法判斷 %s 的交易所：%s", ticker, reason)
                return None, reason
            last_err = ValueError("info.exchange 為空")
        except Exception as exc:  # noqa: BLE001 — 單檔失敗改走重試，不中斷整批分類
            last_err = exc
        if attempt < 2:
            time.sleep(_YF_RETRY_DELAYS[attempt])
    reason = f"已重試 3 次仍失敗：{last_err}"
    _log.warning("無法判斷 %s 的交易所（%s）", ticker, reason)
    return None, reason


def _split_only_in_current(
    tickers: list[str],
) -> tuple[list[str], list[str], list[str], dict[str, str]]:
    """
    把「只在目前清單」的股票分成兩類。

    參數:
        tickers: only_in_current 代號。

    回傳:
        (not_nasdaq_listed, nasdaq_but_excluded, exchange_unknown, unknown_reasons)。
        前兩類分別是「交易所不是 NASDAQ」與「NASDAQ 掛牌但沒進過指數」。
        查不到 exchange 的放在第三類，不猜邊。
    """
    not_nasdaq: list[str] = []
    nasdaq_excluded: list[str] = []
    unknown: list[str] = []
    unknown_reasons: dict[str, str] = {}
    for ticker in tickers:
        exchange, reason = _fetch_exchange(ticker)
        if exchange is None:
            unknown.append(ticker)
            unknown_reasons[ticker] = reason or "info.exchange 為空"
            continue
        if _is_nasdaq_listed(exchange):
            nasdaq_excluded.append(ticker)
            _log.info("%s 交易所=%s → NASDAQ 掛牌但未進指數", ticker, exchange)
        else:
            not_nasdaq.append(ticker)
            _log.info("%s 交易所=%s → 非 NASDAQ 掛牌", ticker, exchange)
    return sorted(not_nasdaq), sorted(nasdaq_excluded), sorted(unknown), unknown_reasons


def compare_universes(
    current_list: list[str],
    point_in_time_union: set[str],
) -> dict[str, list[str] | float | dict[str, str]]:
    """
    對比「今天手動清單」與「歷史時點成分股聯集」。

    overlap_pct 是目前清單中、曾經出現在歷史 NASDAQ-100 的比例（0-100）。
    分母是目前清單檔數，不是歷史聯集：問題是今天挑的股票有多少真的
    在當時指數裡，而不是兩份名單的 Jaccard 相似度。
    代號依字面比對（FB 與 META 視為不同代號，因為指數當時用的就是舊代號）。

    only_in_current 會再用 yfinance info['exchange'] 拆成：
    - not_nasdaq_listed：交易所不是 NASDAQ，指數範圍本來就不涵蓋
    - nasdaq_but_excluded：NASDAQ 掛牌，但這段歷史沒進過 NASDAQ-100
    查不到交易所的放在 exchange_unknown，不併入上面兩類。
    exchange_unknown_reasons 記錄每一檔沒歸類的原因。

    參數:
        current_list: 目前手動選股清單。
        point_in_time_union: get_union_universe 的結果。

    回傳:
        only_in_current、only_in_historical、overlap、overlap_pct，
        以及 not_nasdaq_listed、nasdaq_but_excluded、exchange_unknown、
        exchange_unknown_reasons。
    """
    current = {str(ticker).upper().strip() for ticker in current_list if str(ticker).strip()}
    historical = {
        str(ticker).upper().strip() for ticker in point_in_time_union if str(ticker).strip()
    }
    overlap = current & historical
    only_current = sorted(current - historical)
    only_historical = historical - current
    overlap_pct = round(100.0 * len(overlap) / len(current), 2) if current else 0.0
    not_nasdaq, nasdaq_excluded, unknown, unknown_reasons = _split_only_in_current(only_current)

    return {
        "only_in_current": only_current,
        "only_in_historical": sorted(only_historical),
        "overlap": sorted(overlap),
        "overlap_pct": overlap_pct,
        "not_nasdaq_listed": not_nasdaq,
        "nasdaq_but_excluded": nasdaq_excluded,
        "exchange_unknown": unknown,
        "exchange_unknown_reasons": unknown_reasons,
    }


def _print_ticker_block(title: str, tickers: list[str]) -> None:
    """印出一組代號：先給數量，再逐列輸出，避免一行過長。"""
    print(f"{title}（{len(tickers)} 檔）")
    if not tickers:
        print("  （無）")
        return
    line: list[str] = []
    for ticker in tickers:
        line.append(ticker)
        if len(line) == 12:
            print("  " + ", ".join(line))
            line = []
    if line:
        print("  " + ", ".join(line))


if __name__ == "__main__":
    from config import TICKER_LIST

    start = date(2022, 1, 1)
    end = date.today()
    print(
        f"NASDAQ-100 時點股票池：{start.isoformat()} ~ {end.isoformat()}，每季一個時點",
        flush=True,
    )

    universes = get_point_in_time_universe(start, end, freq="quarterly")
    print("=== 每季成分股數量 ===", flush=True)
    for sample, members in universes.items():
        print(f"  {sample.isoformat()}  {len(members):3d} 檔", flush=True)

    union = get_union_universe(start, end)
    quarter_sizes = [len(members) for members in universes.values()]
    print(f"期間聯集大小: {len(union)} 檔", flush=True)
    print(
        f"單季規模: min={min(quarter_sizes)}  max={max(quarter_sizes)}（指數約 100 檔，季中調整會有個位數出入）",
        flush=True,
    )
    if len(union) < max(quarter_sizes):
        raise RuntimeError("聯集小於單季成分股數，時點名單計算有誤")

    report = compare_universes(TICKER_LIST, union)
    overlap_pct = float(report["overlap_pct"])
    only_current = list(report["only_in_current"])
    only_historical = list(report["only_in_historical"])
    overlap = list(report["overlap"])

    print("=== 目前 60 檔 vs 2022 至今 NASDAQ-100 聯集 ===", flush=True)
    print(f"目前清單: {len(TICKER_LIST)} 檔", flush=True)
    print(f"重疊: {len(overlap)} 檔", flush=True)
    print(
        f"overlap_pct: {overlap_pct:.2f}%（目前清單中，曾經進入這段歷史 NASDAQ-100 的比例）",
        flush=True,
    )
    _print_ticker_block("只在目前清單、這段歷史從沒進過 NASDAQ-100", only_current)
    _print_ticker_block("重疊", overlap)
    _print_ticker_block("只在歷史 NASDAQ-100、目前清單沒有", only_historical)
    print(
        "代號依字面比對：例如 2022 上半年指數用的是 FB，不是 META。",
        flush=True,
    )

    not_nasdaq = list(report["not_nasdaq_listed"])
    nasdaq_excluded = list(report["nasdaq_but_excluded"])
    unknown = list(report["exchange_unknown"])
    classified = len(not_nasdaq) + len(nasdaq_excluded) + len(unknown)
    if classified != len(only_current):
        raise RuntimeError(
            f"only_in_current 拆分數量不符：{classified} != {len(only_current)}"
        )

    print("=== only_in_current 依交易所拆分 ===", flush=True)
    print(
        "not_nasdaq_listed：交易所不是 NASDAQ。指數範圍本來就不涵蓋，不是 survivorship bias。",
        flush=True,
    )
    _print_ticker_block("not_nasdaq_listed", not_nasdaq)
    print(
        "nasdaq_but_excluded：NASDAQ 掛牌，但這段歷史沒進過 NASDAQ-100。"
        "這才是今天篩、過去不在成分股裡。",
        flush=True,
    )
    _print_ticker_block("nasdaq_but_excluded", nasdaq_excluded)
    if unknown:
        reasons = report["exchange_unknown_reasons"]
        print(
            "exchange_unknown：yfinance info 沒有 exchange，未歸入上面兩類。",
            flush=True,
        )
        print(f"exchange_unknown（{len(unknown)} 檔）", flush=True)
        reason_map = reasons if isinstance(reasons, dict) else {}
        for ticker in unknown:
            reason = reason_map.get(ticker, "info.exchange 為空")
            print(f"  {ticker}（{reason}）", flush=True)
