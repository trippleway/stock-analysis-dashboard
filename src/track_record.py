# -*- coding: utf-8 -*-
"""
策略績效追蹤：記錄每日 Top 推薦、回填實際報酬、彙總 vs SPY 績效。

執行方式（於專案根目錄）：
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

# 累積報酬合理性上限：絕對值超過 100% 視為可疑（多半是重複複利或資料異常）
CUMULATIVE_SANITY_LIMIT = 1.0
# 基準合理性上限：SPY 在 90 個交易日內漲跌超過 25% 極罕見
BENCHMARK_SANITY_LIMIT = 0.25
BENCHMARK_SANITY_DAYS = 90

# 每筆推薦的持有期（日曆天），與 return_1m 的定義一致
HOLDING_DAYS = 30

# 快取涵蓋範圍容許的落後天數（吸收週末與假日）
CACHE_COVERAGE_TOLERANCE_DAYS = 4
# yfinance 呼叫間的節流秒數
YF_CALL_SLEEP_SEC = 0.6
# 單一標的貢獻超過總超額報酬的此比例即視為績效過度集中
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

# 持有天數門檻 → (價格欄, 報酬欄)
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
    """支援多種日期格式，回傳 date（無時區）。"""
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
    """補齊缺少欄位、統一欄位順序。"""
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
        _log.warning("讀取 recommendations.csv 失敗：%s", e)
        return pd.DataFrame(columns=CSV_COLUMNS)


def _save_recommendations(df: pd.DataFrame) -> None:
    try:
        TRACK_DIR.mkdir(parents=True, exist_ok=True)
        _ensure_csv_schema(df).to_csv(RECOMMENDATIONS_CSV, index=False, encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        _log.warning("寫入 recommendations.csv 失敗：%s", e)


def _index_as_dates(index: Any) -> list[date | None]:
    """將 DatetimeIndex（含 tz）轉成純 date 列表。"""
    out: list[date | None] = []
    for d in index:
        try:
            ts = pd.Timestamp(d)
            out.append(ts.date())
        except Exception:  # noqa: BLE001
            out.append(_parse_date(d))
    return out


def _history_close_on_or_after(hist: pd.DataFrame, target: date) -> float | None:
    """取 target 當天或之後最近交易日收盤價（不含更早日，避免誤填前瞻報酬）。"""
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
        _log.debug("解析收盤價失敗 target=%s: %s", target, e)
        return None


def _history_close_on_or_before(hist: pd.DataFrame, target: date) -> float | None:
    """取 target 當天或之前最近交易日收盤價（用於進場日 SPY 對照）。"""
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
        _log.debug("解析收盤價(on_or_before)失敗 target=%s: %s", target, e)
        return None


def _load_local_cache_history(ticker: str) -> pd.DataFrame:
    """
    從 data/cache/YYYY-MM-DD/{TICKER}.parquet 載入最新一版價量快取。
    （不依賴「今天」資料夾，方便離線回填。）
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
        _log.debug("掃描本地快取失敗 %s: %s", sym, e)
        return pd.DataFrame()
    if not candidates:
        return pd.DataFrame()
    candidates.sort(key=lambda p: p.parent.name)
    latest = candidates[-1]
    try:
        df = pd.read_parquet(latest)
        if df is None or df.empty:
            return pd.DataFrame()
        _log.debug("使用本地快取 %s ← %s", sym, latest)
        return df
    except Exception as e:  # noqa: BLE001
        _log.warning("讀取本地 parquet 失敗 %s: %s", latest, e)
        return pd.DataFrame()


def _fetch_yfinance_history(ticker: str, start: date, end: date) -> pd.DataFrame:
    """
    用 yfinance 抓 [start, end] 區間日線（end 會 +1 天以含當日），失敗重試 3 次。

    參數：
        ticker: 股票代號
        start: 起始日期（含）
        end: 結束日期（含）
    回傳：
        日線 DataFrame；全部失敗時回傳空 DataFrame
    """
    sym = str(ticker).strip().upper()
    end_excl = end + timedelta(days=1)
    delays = (1.0, 2.0, 4.0)
    for attempt in range(3):
        try:
            time.sleep(YF_CALL_SLEEP_SEC)  # 節流，避免連續呼叫觸發 429
            t = yf.Ticker(sym)
            hist = t.history(start=start.isoformat(), end=end_excl.isoformat(), auto_adjust=True)
            if hist is not None and not hist.empty:
                return hist
            _log.debug("yfinance history 無資料 %s [%s ~ %s]（第 %s 次）", sym, start, end, attempt + 1)
        except Exception as e:  # noqa: BLE001
            _log.warning(
                "yfinance history 失敗 %s [%s ~ %s]（第 %s 次）: %s", sym, start, end, attempt + 1, e
            )
        if attempt < 2:
            time.sleep(delays[attempt])
    return pd.DataFrame()


def _history_max_date(hist: pd.DataFrame) -> date | None:
    """取歷史資料中最新一筆的日期（無法解析時回傳 None）。"""
    if hist is None or hist.empty:
        return None
    dates = [d0 for d0 in _index_as_dates(hist.index) if d0 is not None]
    return max(dates) if dates else None


def _cache_covers(cache_max: date, end: date) -> bool:
    """
    判斷快取是否已涵蓋請求的 end 日期。

    end 落在週末或假日時不會有新交易日資料，因此先退回最近的工作日，
    並容許 CACHE_COVERAGE_TOLERANCE_DAYS 天落差，避免每次都誤判快取過期。

    參數：
        cache_max: 快取中最新一筆資料的日期
        end: 請求涵蓋到的日期
    回傳：
        True 表示快取足夠新，可直接使用
    """
    target = min(end, date.today())
    while target.weekday() >= 5:  # 週六=5、週日=6
        target -= timedelta(days=1)
    return cache_max >= target - timedelta(days=CACHE_COVERAGE_TOLERANCE_DAYS)


def _update_cache_history(ticker: str, hist: pd.DataFrame) -> None:
    """
    把新抓到的日線寫入當日快取目錄，讓後續呼叫不必重抓。

    只寫 {TICKER}.parquet；data_collector 另需 _info.json 才視為有效快取，
    因此不會影響它自己的快取判斷。
    """
    sym = str(ticker).strip().upper()
    try:
        day_dir = Path(CACHE_DIR) / date.today().isoformat()
        day_dir.mkdir(parents=True, exist_ok=True)
        hist.to_parquet(day_dir / f"{sym}.parquet", index=True)
        _log.debug("已更新價量快取 %s → %s", sym, day_dir)
    except Exception as e:  # noqa: BLE001
        _log.warning("更新價量快取失敗 %s: %s", sym, e)


def _fetch_history(ticker: str, start: date, end: date) -> pd.DataFrame:
    """
    取得 [start, end] 期間日線：快取涵蓋請求區間時優先用快取，否則改抓 yfinance。

    快取若只到較早日期（例如最後一次收集是一個月前），直接沿用會導致新推薦
    永遠回填不到滿 30 天的價格，因此必須偵測涵蓋範圍後重抓並更新快取。

    參數：
        ticker: 股票代號
        start: 起始日期（含）
        end: 需要涵蓋到的日期（含）
    回傳：
        日線 DataFrame；皆失敗時回傳空 DataFrame
    """
    sym = str(ticker).strip().upper()
    local = _load_local_cache_history(sym)
    local_max = _history_max_date(local)

    if local_max is not None and _cache_covers(local_max, end):
        return local

    if local_max is not None:
        _log.info("快取 %s 僅到 %s，未涵蓋 %s，改抓 yfinance", sym, local_max, end)
    fresh = _fetch_yfinance_history(sym, start, end)
    if not fresh.empty:
        _update_cache_history(sym, fresh)
        return fresh

    if not local.empty:
        _log.warning(
            "yfinance 抓取失敗，改用截斷的本地快取 %s：資料僅到 %s（請求至 %s），"
            "該日之後的報酬無法回填，下次執行會再試",
            sym,
            local_max,
            end,
        )
    return local


def record_daily_picks(picks_result: dict[str, Any]) -> None:
    """
    將當日 Top picks append 到 recommendations.csv（同日同 ticker 不重複）。
    """
    if not isinstance(picks_result, dict):
        _log.warning("record_daily_picks：picks_result 不是 dict，略過")
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
            _log.info("已存在紀錄，略過：%s %s", day_str, ticker)
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
        _log.info("record_daily_picks：無新紀錄可寫入（%s）", day_str)
        return

    add_df = _ensure_csv_schema(pd.DataFrame(new_rows))
    if existing.empty:
        merged = add_df
    else:
        merged = pd.concat([_ensure_csv_schema(existing), add_df], ignore_index=True)
    _save_recommendations(merged)
    _log.info("record_daily_picks：新增 %s 筆（%s）→ %s", len(new_rows), day_str, RECOMMENDATIONS_CSV)


def update_actuals(days_lookback: int = 45) -> dict[str, Any]:
    """
    回填近 days_lookback 天內推薦的實際收盤與報酬。
    """
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
        _log.info("update_actuals：尚無推薦紀錄")
        return stats

    today = date.today()
    lookback = max(1, int(days_lookback))
    cutoff = today - timedelta(days=lookback)
    hard_floor = today - timedelta(days=max(lookback, 400))
    stats["total_records"] = int(len(df))

    # 預先快取各 ticker / SPY 歷史（一次抓足夠長區間）
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
            # lookback 內一律處理；之外僅補尚未 completed，且不早於 hard_floor
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
                stats["errors"].append(f"{ticker}@{entry_d}: 無歷史股價")
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

            # 1M：補 SPY 與 alpha（必要時回填缺漏的 spy_price_entry）
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

            _log.debug("已檢查 %s %s", entry_d, ticker)

        except Exception as e:  # noqa: BLE001
            msg = f"列 {idx} 更新失敗: {e}"
            _log.warning(msg)
            stats["errors"].append(msg)

    _save_recommendations(df)
    _log.info(
        "update_actuals 完成：1d=%s 1w=%s 2w=%s 1m=%s errors=%s",
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
    累積報酬合理性檢查：絕對值超過 CUMULATIVE_SANITY_LIMIT 時 log warning。

    參數：
        label: 標籤（例如「策略」、「SPY」），用於訊息辨識
        cum_return: 累積報酬（小數，例如 0.35 = +35%）
        trading_days: 資金曲線涵蓋的交易日數
    回傳：
        None
    """
    if cum_return is None:
        return
    if abs(cum_return) > CUMULATIVE_SANITY_LIMIT:
        _log.warning(
            "%s 累積報酬 %+.2f%%（%s 個交易日）絕對值超過 %.0f%%，可能為計算錯誤："
            "請檢查是否把重疊的持有期當成接續操作連乘，或原始價格資料異常",
            label,
            cum_return * 100,
            trading_days,
            CUMULATIVE_SANITY_LIMIT * 100,
        )


def _warn_if_benchmark_implausible(spy_cum: float | None, trading_days: int) -> None:
    """
    基準（SPY）合理性檢查：短期間內漲跌幅過大代表期間長度或價格資料有問題。

    參數：
        spy_cum: SPY 累積報酬（小數）
        trading_days: 資金曲線涵蓋的交易日數
    回傳：
        None
    """
    if spy_cum is None:
        return
    if abs(spy_cum) > BENCHMARK_SANITY_LIMIT and trading_days < BENCHMARK_SANITY_DAYS:
        _log.warning(
            "SPY 累積報酬 %+.2f%% 僅用 %s 個交易日達成，超過 %.0f%% 門檻："
            "SPY 在三個月內漲跌超過 %.0f%% 極為罕見，請檢查期間長度是否被錯誤放大",
            spy_cum * 100,
            trading_days,
            BENCHMARK_SANITY_LIMIT * 100,
            BENCHMARK_SANITY_LIMIT * 100,
        )


def _daily_close_series(ticker: str, start: date, end: date) -> pd.Series | None:
    """
    取單一標的在 [start, end] 期間的每日收盤價序列。

    參數：
        ticker: 股票代號
        start: 起始日期（含）
        end: 結束日期（含）
    回傳：
        index 為 date、值為收盤價的 Series（已排序去重）；無資料時回傳 None
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
        # 本地快取為固定 1 年區間，需自行裁切到需要的期間
        s = s[[start <= d0 <= end for d0 in s.index]]
        if s.empty:
            return None
        s = s[~s.index.duplicated(keep="last")].sort_index()
        return s
    except Exception as e:  # noqa: BLE001
        _log.warning("建立每日收盤序列失敗 %s [%s ~ %s]: %s", ticker, start, end, e)
        return None


def _position_daily_returns(closes: pd.Series, entry_d: date) -> dict[date, float]:
    """
    計算單筆推薦在持有期內的「每日報酬」。

    進場價為進場日（或之前最近交易日）收盤，因此報酬自進場日之後的第一個
    交易日起算，持有區間為 (entry_d, entry_d + HOLDING_DAYS]。

    參數：
        closes: 該標的的每日收盤序列（index 為 date）
        entry_d: 進場日期
    回傳：
        {交易日: 當日報酬} 字典；資料不足時回傳空 dict
    """
    exit_d = entry_d + timedelta(days=HOLDING_DAYS)
    window = closes[[entry_d <= d0 <= exit_d for d0 in closes.index]]
    if len(window) < 2:
        # 進場日非交易日時，往前找最近一個交易日當基準價
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
    抓取每日收盤價並展開成資金曲線所需的原料（只抓一次，供曲線與歸因共用）。

    參數：
        df: recommendations 資料表
    回傳：
        {"positions": [(代號, 進場日, {交易日: 當日報酬})], "spy_closes": Series|None,
         "anchor": 基準日, "spine": [交易日]}；資料不足時回傳 {}
    """
    # 蒐集有效持倉（需有日期與代號；報酬由每日價格重算，不依賴 return_1m）
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
    # 往前多取 10 天，確保進場日遇假日時仍找得到基準收盤價
    fetch_start = first_entry - timedelta(days=10)
    fetch_end = min(last_exit, date.today())

    # 每個代號只抓一次，涵蓋其所有持倉區間
    closes_by_ticker: dict[str, pd.Series | None] = {}
    for ticker, _ in raw:
        if ticker not in closes_by_ticker:
            closes_by_ticker[ticker] = _daily_close_series(ticker, fetch_start, fetch_end)

    # 逐筆展開為 {交易日: 當日報酬}
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
        _log.warning("資金曲線：%s 筆持倉缺少每日價格，已略過（%s）", len(missing), ", ".join(missing[:5]))
    if not positions:
        return {}

    # 交易日主軸優先用 SPY 日曆（同時作為基準價來源），失敗則用持倉日期聯集
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
    由每日報酬合成資金曲線：每個交易日取在倉標的當日報酬的等權平均後複利。

    參數：
        positions: [(代號, 進場日, {交易日: 當日報酬})]
        spine: 交易日主軸
        exclude: 要排除的代號（單一字串、清單或集合；用於貢獻度歸因與
                 performance_excluding）
    回傳：
        (資金曲線累積報酬列表, 最終累積報酬)
    """
    if exclude is None:
        exclude_set: set[str] = set()
    elif isinstance(exclude, str):
        exclude_set = {exclude.strip().upper()}
    else:
        exclude_set = {str(t).strip().upper() for t in exclude if str(t).strip()}
    active = [r for t, _d, r in positions if t not in exclude_set]
    equity = 1.0
    curve: list[float] = [0.0]  # 起點為基準日、報酬 0
    for d0 in spine:
        todays = [r[d0] for r in active if d0 in r]
        port_ret = sum(todays) / len(todays) if todays else 0.0  # 無持倉當日不動
        equity *= 1.0 + port_ret
        curve.append(round(equity - 1.0, 6))
    return curve, equity - 1.0


def _build_daily_equity_curve(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    以「交易日」為單位建立資金曲線，正確處理重疊的持有期。

    每個交易日的組合報酬 = 當日所有在倉標的（entry_d < 該日 <= entry_d+HOLDING_DAYS）
    當日報酬的等權平均；無持倉當日報酬視為 0。SPY 基準改為同一段日曆期間的
    單純買進持有，避免把重疊持有期的報酬重複複利。

    參數：
        inputs: _prepare_curve_inputs() 的輸出
    回傳：
        cumulative 結果 dict；資料不足時回傳 {}
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

    # SPY：同期間單純買進持有（anchor 收盤買、曲線最後一日收盤賣）
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

    _warn_if_implausible("策略", strategy_cum, trading_days)
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
    回答「超額報酬集中在哪幾檔」：以留一法（leave-one-out）計算每檔的貢獻度。

    貢獻度 = 完整資金曲線累積報酬 − 移除該標的後重算的累積報酬。等權配置下
    移除一檔會使其餘標的權重上升，因此各檔貢獻度之和不會恰等於總報酬，
    這是留一法歸因的固有性質，用於判斷「集中度」而非精確拆帳。

    參數：
        df: recommendations 資料表
        inputs: _prepare_curve_inputs() 的輸出
        cumulative: _build_daily_equity_curve() 的結果
    回傳：
        concentration 結果 dict；資料不足時回傳 {}
    """
    if not inputs or not isinstance(cumulative, dict):
        return {}
    positions = inputs["positions"]
    spine = inputs["spine"]
    total_cum = float(cumulative["strategy_cumulative_return"])
    outperf = cumulative.get("outperformance")

    # 各檔入選次數與平均 return_1m（以完整資料表為準）
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

    # 依貢獻度由高到低排序，方便報告直接呈現
    by_ticker = dict(sorted(by_ticker.items(), key=lambda kv: kv[1]["contribution"], reverse=True))

    top_ticker, top_stats = next(iter(by_ticker.items()))
    top_contrib = float(top_stats["contribution"])
    # 佔比以總超額報酬為分母（缺 SPY 基準時退回總累積報酬）
    denom = float(outperf) if outperf not in (None, 0) else total_cum
    share = (top_contrib / denom) if denom > 0 else None

    warning = None
    if share is not None and share > CONCENTRATION_WARN_SHARE:
        warning = "績效高度集中於單一標的，樣本代表性有限"
        _log.warning(
            "%s 貢獻 %+.2f%%，佔總超額報酬 %.1f%%（>%.0f%%）：%s",
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
    計算排除指定標的後的策略績效。

    回答「扣掉最大貢獻者後，策略還有沒有 alpha」：用與完整資金曲線相同的
    每日等權複利邏輯重算，但把指定標的的所有持倉移除；SPY 基準維持同期間
    單純買進持有（不受排除影響）。

    參數：
        tickers: 要排除的股票代號清單（不分大小寫）
    回傳：
        {
            excluded: 實際排除的代號（大寫、去重、排序）,
            strategy_return: 排除後策略累積報酬,
            spy_return: 同期間 SPY 買進持有報酬,
            outperformance: 策略 − SPY,
            still_beats_spy: 策略是否仍高於 SPY,
            remaining_picks_count: 排除後剩餘推薦筆數,
        }
        資料不足時 strategy_return / spy_return 為 None，still_beats_spy 為 False
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

    # SPY：同期間單純買進持有（與完整曲線相同，不受排除影響）
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
    """從 recommendations.csv 計算績效快照。"""
    df = _load_recommendations()
    empty_summary = {
        "total_recommendations": 0,
        "completed_1m": 0,
        "date_range": {"first": None, "last": None},
    }
    if df.empty:
        return {
            "summary": empty_summary,
            "aggregate": "資料不足",
            "by_sector": "資料不足",
            "by_month": "資料不足",
            "cumulative": "資料不足",
            "concentration": "資料不足",
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
            "aggregate": "資料不足",
            "by_sector": "資料不足",
            "by_month": "資料不足",
            "cumulative": "資料不足",
            "concentration": "資料不足",
            "note": "completed_1m < 5，需更多追蹤紀錄才能產生統計",
        }

    r1m = pd.to_numeric(completed["return_1m"], errors="coerce")
    alpha = pd.to_numeric(completed["alpha_1m"], errors="coerce")
    spy_r = pd.to_numeric(completed["spy_return_1m"], errors="coerce")

    avg_1m = _safe_mean(r1m) or 0.0
    std_1m = _safe_std(r1m)
    # Sharpe：以「每次推薦的 1M 報酬」近似月報酬，年化 × sqrt(12)；rf=0
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

    # by_month：依推薦月份分組
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

    # cumulative：以交易日為單位的資金曲線（持有期重疊時不可用各筆報酬連乘）
    curve_inputs = _prepare_curve_inputs(df)
    cumulative: dict[str, Any] | str = _build_daily_equity_curve(curve_inputs) or "資料不足"
    concentration: dict[str, Any] | str = _build_concentration(df, curve_inputs, cumulative) or "資料不足"

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
    計算績效並寫入 performance.json，回傳路徑。

    參數：
        perf: 已算好的績效 dict；None 時自行呼叫 calculate_performance()
              （傳入可避免重複抓價與重複 log 警告）
    回傳：
        performance.json 路徑
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
        _log.info("已儲存績效快照：%s", PERFORMANCE_JSON)
    except Exception as e:  # noqa: BLE001
        _log.warning("寫入 performance.json 失敗：%s", e)
    return PERFORMANCE_JSON


def backfill_from_daily_picks() -> int:
    """
    （輔助）掃描 data/daily_picks/*.json，補寫尚未入庫的推薦。
    """
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
            _log.warning("backfill %s 失敗：%s", path.name, e)
    return n


if __name__ == "__main__":
    print("掃描既有 daily_picks 並寫入 track record…")
    added = backfill_from_daily_picks()
    print(f"新增紀錄：{added} 筆")

    print("執行 update_actuals...")
    stats = update_actuals()
    print(f"更新統計：{stats}")

    # 計算並印出當前績效
    perf = calculate_performance()
    print("\n=== 策略績效 ===")
    summary = perf["summary"]
    print(f"總推薦數：{summary['total_recommendations']}")
    print(f"已完成 1M 追蹤：{summary['completed_1m']}")
    print(f"日期範圍：{summary['date_range']['first']} ~ {summary['date_range']['last']}")

    if summary["completed_1m"] >= 5:
        agg = perf["aggregate"]
        assert isinstance(agg, dict)
        print(f"\n1M 平均報酬：{float(agg['avg_return_1m'] or 0):+.2%}")
        print(f"1M 勝率：{float(agg['win_rate_1m'] or 0):.1%}")
        print(f"1M 跑贏 SPY 比例：{float(agg['beat_spy_rate_1m'] or 0):.1%}")
        print(f"1M 平均 Alpha：{float(agg['avg_alpha_1m'] or 0):+.2%}")
        print(f"Sharpe Ratio：{float(agg['sharpe_ratio_1m'] or 0):.2f}")
        cum = perf["cumulative"]
        if isinstance(cum, dict):
            dr = cum["date_range"]
            print(f"\n資金曲線期間：{dr['start']} ~ {dr['end']}（{cum['trading_days']} 個交易日）")
            print(f"累積策略報酬：{float(cum['strategy_cumulative_return'] or 0):+.2%}")
            print(f"累積 SPY 報酬（買進持有）：{float(cum['spy_cumulative_return'] or 0):+.2%}")
            print(f"超額報酬：{float(cum['outperformance'] or 0):+.2%}")

            # 排除最大貢獻者後的對照：回答「扣掉這檔後還有沒有 alpha」
            print("\n=== 排除最大貢獻者後的績效對照 ===")
            full_s = float(cum["strategy_cumulative_return"] or 0)
            full_spy = float(cum["spy_cumulative_return"] or 0)
            print(f"完整策略：{full_s:+.2%} vs SPY {full_spy:+.2%}")
            excl_ticker = "MU"
            if isinstance(conc := perf.get("concentration"), dict):
                top = conc.get("top_contributor") if isinstance(conc.get("top_contributor"), dict) else {}
                if top.get("ticker"):
                    excl_ticker = str(top["ticker"])
            excl = performance_excluding([excl_ticker])
            excl_s = excl.get("strategy_return")
            excl_spy = excl.get("spy_return")
            if excl_s is not None and excl_spy is not None:
                verdict = "仍跑贏 SPY" if excl["still_beats_spy"] else "無法跑贏 SPY"
                print(
                    f"排除 {excl_ticker}（剩 {excl['remaining_picks_count']} 筆）："
                    f"{excl_s:+.2%} vs SPY {excl_spy:+.2%} → {verdict}"
                    f"（超額 {float(excl['outperformance'] or 0):+.2%}）"
                )
            else:
                print(f"排除 {excl_ticker}：資料不足，無法計算")
        else:
            print("\n資金曲線：資料不足")

        conc = perf["concentration"]
        if isinstance(conc, dict):
            print("\n=== 報酬集中度（留一法歸因）===")
            print(f"{'標的':<8}{'入選':>5}{'平均1M':>10}{'貢獻度':>11}")
            for tk, st in conc["by_ticker"].items():
                avg = st["avg_return_1m"]
                avg_s = f"{avg:+.2%}" if avg is not None else "—"
                print(f"{tk:<8}{st['picks']:>5}{avg_s:>10}{st['contribution']:>+11.2%}")
            top = conc["top_contributor"]
            share = top["share_of_outperformance"]
            share_s = f"，佔總超額報酬 {share:.1%}" if share is not None else ""
            print(f"\n最大貢獻者：{top['ticker']}（{top['contribution']:+.2%}{share_s}）")
            if conc["concentration_warning"]:
                print(f"⚠️  {conc['concentration_warning']}")
        else:
            print("\n報酬集中度：資料不足")
    else:
        print("\n完成追蹤紀錄不足（<5 筆），需要更多推薦累積才能產生統計")

    path = save_performance_snapshot(perf)
    print(f"\n已寫入快照：{path}")
