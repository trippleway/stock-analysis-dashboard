# -*- coding: utf-8 -*-
"""
組 A 的因子研究：只算月度 rank IC，不做回測、也不選股。

區間沿用 config 的固定窗口（2023-09 至 2026-09）。母體是組 A。
每個月底的因子值，與該月持有期（下一個月底為止）報酬做 Spearman 相關。
34 檔都放進橫斷面；某一檔當天沒有因子或沒有報酬，才從該月剔除。

既有五項子分數直接呼叫 indicators.py 的評分函式，各自獨立，不加權。
新增的五個因子只用收盤價與成交量，窗口都是交易日，而且只往回看。

最後用同一批子分數做五種加權，只比較綜合分的 rank IC。
這是同一段資料上的事後對照，用來檢查 RSI 的方向，不挑選新權重。

前視偏差檢查：把資料截斷到訊號月月底再重算，必須與完整資料在同一個月底的值相同。
有任何不一致就中止，不印 IC。

執行方式（於專案根目錄）：
    python src/factor_research.py
"""

from __future__ import annotations

import logging
import math
import sys
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_SRC = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from backtest_compare_groups import (  # noqa: E402
    _as_timestamp_index,
    _month_end_signals,
    define_three_groups,
)
from backtest_data import build_ohlcv_panel  # noqa: E402
from backtest_robustness import _holding_price_returns, _spearman  # noqa: E402
from config import BACKTEST_END, BACKTEST_START, TECH_SCORE_WEIGHTS  # noqa: E402
from indicators import (  # noqa: E402
    _score_bollinger,
    _score_macd,
    _score_ma_trend,
    _score_rsi,
    _score_volume,
    calculate_indicators,
    calculate_tech_score,
)

_log = logging.getLogger("factor_research")
if not _log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# 日曆與現有 rank IC 相同，用 SPY 的交易日當月底與持有區間。
_CALENDAR = "SPY"
# 252 個交易日大約要一年。固定往前多取，讓第一個訊號月的長窗口有機會算得出來。
# 這個長度不是拿來調整 IC 的參數。
_WARMUP_DAYS = 450
# 規格裡的窗口，單位都是該股票自己的交易日。
_LOOKBACK_DAYS = 252
_SKIP_DAYS = 21
_VOL_DAYS = 60
_VOLUME_SHORT = 20
_VOLUME_LONG = 60
# 截斷重算與完整資料的容許誤差。真正用到未來會差一個價格跳動，遠大於這個值。
_LOOKAHEAD_ATOL = 1e-8
# calculate_tech_score 會 round 到小數兩位。差一個評分檔（5 分以上）仍會被抓到。
_API_ATOL = 0.011
_GROUP_SIZE = 34
_MIN_CORR_MONTHS = 3

# 表與相關矩陣都用這個順序。IC 表印出前再依 |t| 排序。
_FACTORS: list[tuple[str, str]] = [
    ("rsi", "RSI 子分數"),
    ("macd", "MACD 子分數"),
    ("bollinger", "布林子分數"),
    ("ma_trend", "均線子分數"),
    ("volume", "成交量子分數"),
    ("mom_12_1", "12-1 動能"),
    ("reversal_21", "短期反轉"),
    ("low_vol_60", "低波動"),
    ("high_52w", "52 週高點接近度"),
    ("volume_trend", "量能趨勢"),
]
_SHORT = {
    "rsi": "RSI",
    "macd": "MACD",
    "bollinger": "布林",
    "ma_trend": "均線",
    "volume": "成交量",
    "mom_12_1": "12-1",
    "reversal_21": "反轉",
    "low_vol_60": "低波動",
    "high_52w": "52週高",
    "volume_trend": "量能",
}
_SUBSCORE_KEYS = ("rsi", "macd", "bollinger", "ma_trend", "volume")


def _prepare_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """
    整理成無時區、一天一列的 OHLCV。

    參數:
        frame: 至少要有 Open、High、Low、Close、Volume。

    回傳:
        依日期排序的複本。同一天留最後一筆。
    """
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"價量缺少欄位: {sorted(missing)}")
    out = frame.loc[:, ["Open", "High", "Low", "Close", "Volume"]].copy()
    out.index = _as_timestamp_index(out.index)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _score_position(ind: pd.DataFrame, i: int) -> tuple[dict[str, float], dict[str, bool]]:
    """
    用 indicators.py 的五個評分函式，給單一交易日的子分數。

    分數只看當天與前一個交易日，不加權。指標還沒算出來時，評分函式回傳中性 50，
    這裡照單全收，同時記下輸入是不是缺的，方便事後分辨「真的 50 分」和「還沒有指標」。

    參數:
        ind: calculate_indicators 的結果，已排序。
        i: 列位置。

    回傳:
        (五項子分數, 該項輸入是否缺失)。
    """
    row = ind.iloc[i]
    prev = ind.iloc[i - 1] if i > 0 else None

    def cur(col: str) -> float | None:
        if col not in ind.columns:
            return None
        value = row[col]
        if pd.isna(value):
            return None
        return float(value)

    def prv(col: str) -> float | None:
        if prev is None or col not in ind.columns:
            return None
        value = prev[col]
        if pd.isna(value):
            return None
        return float(value)

    rsi = cur("RSI")
    dif, dea, hist = cur("MACD_DIF"), cur("MACD_DEA"), cur("MACD_HIST")
    close = cur("Close")
    upper, mid, lower = cur("BB_UPPER"), cur("BB_MIDDLE"), cur("BB_LOWER")
    ma5, ma20, ma60, ma200 = cur("MA5"), cur("MA20"), cur("MA60"), cur("MA200")
    vol_ratio, open_ = cur("VOL_RATIO"), cur("Open")

    rsi_sc, _ = _score_rsi(rsi, prv("RSI"))
    macd_sc, _ = _score_macd(dif, dea, hist, prv("MACD_DIF"), prv("MACD_DEA"), prv("MACD_HIST"))
    boll_sc, _ = _score_bollinger(close, upper, mid, lower)
    ma_sc, _ = _score_ma_trend(close, ma5, ma20, ma60, ma200)
    vol_sc, _ = _score_volume(vol_ratio, open_, close)

    scores = {
        "rsi": float(rsi_sc),
        "macd": float(macd_sc),
        "bollinger": float(boll_sc),
        "ma_trend": float(ma_sc),
        "volume": float(vol_sc),
    }
    missing = {
        "rsi": rsi is None,
        "macd": dif is None or dea is None or hist is None,
        "bollinger": close is None or upper is None or mid is None or lower is None,
        "ma_trend": close is None or ma20 is None or ma60 is None,
        "volume": vol_ratio is None or open_ is None or close is None,
    }
    return scores, missing


def _price_factors(frame: pd.DataFrame) -> pd.DataFrame:
    """
    只用收盤價與成交量計算五個新因子。rolling 與 shift 都只看當天以前。

    參數:
        frame: 已整理的 OHLCV，索引是該股票自己的交易日。

    回傳:
        與 frame 同索引的五欄。窗口不夠或除以 0 時為 NaN。
    """
    close = frame["Close"].astype(float)
    volume = frame["Volume"].astype(float)
    # 12-1：從 252 個交易日前持有到 21 個交易日前。最近 21 日不放進這個報酬。
    momentum = close.shift(_SKIP_DAYS) / close.shift(_LOOKBACK_DAYS) - 1.0
    # 過去 21 日簡單報酬。預期 IC 為負，這裡不反號、也不取絕對值。
    reversal = close / close.shift(_SKIP_DAYS) - 1.0
    # 數值愈大波動愈高。這是標準差本身，不是「低波動」的倒數。
    low_vol = close.pct_change().rolling(_VOL_DAYS, min_periods=_VOL_DAYS).std(ddof=1)
    # 含當天。創高當天比值是 1。
    near_high = close / close.rolling(_LOOKBACK_DAYS, min_periods=_LOOKBACK_DAYS).max()
    vol_trend = (
        volume.rolling(_VOLUME_SHORT, min_periods=_VOLUME_SHORT).mean()
        / volume.rolling(_VOLUME_LONG, min_periods=_VOLUME_LONG).mean()
    )
    out = pd.DataFrame(
        {
            "mom_12_1": momentum,
            "reversal_21": reversal,
            "low_vol_60": low_vol,
            "high_52w": near_high,
            "volume_trend": vol_trend,
        },
        index=frame.index,
    )
    return out.replace([np.inf, -np.inf], np.nan)


def compute_factors(
    frame: pd.DataFrame,
    score_dates: pd.DatetimeIndex | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    計算一檔股票的全部因子。

    參數:
        frame: 原始或已整理的 OHLCV。只會使用索引上該日以前的資料。
        score_dates: 要算子分數的日期。None 表示每個交易日都算。
            價量因子一律整段重算，截斷檢查才對得到當天。

    回傳:
        (因子表, 子分數輸入是否缺失)。因子表欄位順序與 _FACTORS 相同。
        缺失表只含子分數五欄，True 代表評分函式拿到的是空值、回傳了中性 50。
    """
    prepared = _prepare_ohlcv(frame)
    prices = _price_factors(prepared)
    ind = calculate_indicators(prepared, use_ta=False).dropna(how="all")
    if score_dates is None:
        positions = range(len(ind))
    else:
        wanted = set(_as_timestamp_index(pd.DatetimeIndex(score_dates)))
        positions = [i for i, ts in enumerate(ind.index) if ts in wanted]

    score_rows: dict[pd.Timestamp, dict[str, float]] = {}
    missing_rows: dict[pd.Timestamp, dict[str, bool]] = {}
    for i in positions:
        scores, missing = _score_position(ind, i)
        stamp = pd.Timestamp(ind.index[i])
        score_rows[stamp] = scores
        missing_rows[stamp] = missing

    scores = pd.DataFrame.from_dict(score_rows, orient="index")
    missing = pd.DataFrame.from_dict(missing_rows, orient="index")
    out = prices.join(scores, how="left")
    for key in _SUBSCORE_KEYS:
        if key not in out.columns:
            out[key] = np.nan
        if key not in missing.columns:
            missing[key] = False
    columns = [key for key, _name in _FACTORS]
    out = out.loc[:, columns].replace([np.inf, -np.inf], np.nan)
    missing = missing.reindex(columns=list(_SUBSCORE_KEYS)).fillna(False)
    return out, missing.astype(bool)


def _values_equal(left: float, right: float) -> bool:
    """兩個因子值是否相同。兩邊都是 NaN 視為相同。"""
    left_nan = pd.isna(left)
    right_nan = pd.isna(right)
    if left_nan and right_nan:
        return True
    if left_nan or right_nan:
        return False
    return bool(np.isclose(float(left), float(right), rtol=0.0, atol=_LOOKAHEAD_ATOL))


def _value_at(frame: pd.DataFrame, stamp: pd.Timestamp, column: str) -> float:
    """讀取某個交易日的因子。該日沒有列，或值不是有限數，回傳 NaN。"""
    if stamp not in frame.index:
        return float("nan")
    value = frame.at[stamp, column]
    if pd.isna(value) or not math.isfinite(float(value)):
        return float("nan")
    return float(value)


def _api_inputs_complete(ind: pd.DataFrame) -> bool:
    """最後一列的指標是否齊全。齊全時 calculate_tech_score 不會去抓更早的有效值。"""
    needed = (
        "RSI",
        "MACD_DIF",
        "MACD_DEA",
        "MACD_HIST",
        "BB_UPPER",
        "BB_MIDDLE",
        "BB_LOWER",
        "MA5",
        "MA20",
        "MA60",
        "MA200",
        "VOL_RATIO",
        "Open",
        "Close",
    )
    if ind.empty:
        return False
    last = ind.iloc[-1]
    return all(col in ind.columns and pd.notna(last[col]) for col in needed)


def _check_indicator_api(
    histories: dict[str, pd.DataFrame],
    factors: dict[str, pd.DataFrame],
) -> bool:
    """
    確認子分數真的是 indicators.calculate_tech_score 的那五項。

    只核對最後一列指標齊全的股票。那些股票的 _last_scalar 就是最後一列，
    不會混進更早的數值。有任一檔對不上就視為接線錯誤。

    參數:
        histories: 已整理的價量。
        factors: compute_factors 的因子表，必須已包含每檔最後一個交易日。

    回傳:
        True 表示至少一檔核對通過，且所有可核對的股票都一致。
    """
    checked = 0
    for ticker, history in histories.items():
        try:
            ind = calculate_indicators(history, use_ta=False)
            if not _api_inputs_complete(ind):
                _log.warning("%s 最後一列指標不齊，略過與 calculate_tech_score 的對照", ticker)
                continue
            api = calculate_tech_score(ind)
            last_day = history.index[-1]
        except Exception:
            _log.exception("%s 與 calculate_tech_score 對照時發生例外", ticker)
            return False
        checked += 1
        for key in _SUBSCORE_KEYS:
            got = _value_at(factors[ticker], pd.Timestamp(last_day), key)
            expected = float(api["sub_scores"][key])
            if not _values_equal(got, expected) and not (
                math.isfinite(got) and math.isfinite(expected) and abs(got - expected) <= _API_ATOL
            ):
                _log.warning(
                    "%s 的 %s 與 calculate_tech_score 不一致：研究值=%s，既有 API=%s",
                    ticker,
                    key,
                    got,
                    expected,
                )
                return False
    if checked == 0:
        _log.warning("沒有任何一檔的最後一列指標齊全，無法確認子分數來自 indicators.py")
        return False
    _log.info("子分數與 calculate_tech_score 一致：核對 %s 檔", checked)
    return True


def verify_no_lookahead(
    histories: dict[str, pd.DataFrame],
    factors: dict[str, pd.DataFrame],
    signals: list[pd.Timestamp],
) -> list[dict[str, Any]]:
    """
    前視偏差檢查。每個因子、每個訊號月、每檔股票都比一次。

    完整資料算出的第 t 月月底值，必須等於只保留到該月底再重算的值。
    兩端都是 NaN 算通過（窗口不夠，兩邊都不該冒出一個數）。

    參數:
        histories: 已整理、含暖機與研究區間的價量。
        factors: 用完整資料算好的因子表。
        signals: 訊號日，由早到晚。

    回傳:
        不一致的清單。空清單表示通過。
    """
    mismatches: list[dict[str, Any]] = []
    for pos, signal in enumerate(signals, start=1):
        stamp = pd.Timestamp(signal)
        _log.info("前視偏差檢查 %s / %s：%s", pos, len(signals), stamp.date().isoformat())
        for ticker, history in histories.items():
            truncated = history.loc[history.index <= stamp]
            if not truncated.empty and truncated.index.max() > stamp:
                raise RuntimeError(f"{ticker} 截斷到 {stamp.date()} 之後仍有資料")
            try:
                recomputed, _missing = compute_factors(truncated, score_dates=pd.DatetimeIndex([stamp]))
            except Exception as exc:
                mismatches.append(
                    {
                        "ticker": ticker,
                        "month": stamp.strftime("%Y-%m"),
                        "factor": "（截斷重算失敗）",
                        "full": float("nan"),
                        "truncated": float("nan"),
                        "detail": str(exc),
                    }
                )
                continue
            full = factors[ticker]
            for key, label in _FACTORS:
                full_value = _value_at(full, stamp, key)
                trunc_value = _value_at(recomputed, stamp, key)
                if _values_equal(full_value, trunc_value):
                    continue
                mismatches.append(
                    {
                        "ticker": ticker,
                        "month": stamp.strftime("%Y-%m"),
                        "factor": label,
                        "full": full_value,
                        "truncated": trunc_value,
                        "detail": "",
                    }
                )
    return mismatches


def _holding_windows(index: pd.DatetimeIndex) -> list[dict[str, Any]]:
    """
    沿用月度 rank IC 的訊號日與持有區間。

    訊號日是績效區間內每個月最後一個交易日，最後一個交易日本身排除。
    持有從訊號日的下一個交易日開始，到下一個訊號日（最後一段則到區間最後一天）。

    參數:
        index: 主日曆，這裡是 SPY 的交易日。

    回傳:
        每個月一列，含 signal、signal_month、hold。
    """
    signals = _month_end_signals(index, BACKTEST_START, BACKTEST_END)
    windows: list[dict[str, Any]] = []
    for pos, signal_day in enumerate(signals):
        stamp = pd.Timestamp(signal_day)
        if stamp not in index:
            continue
        nxt = signals[pos + 1] if pos + 1 < len(signals) else index[-1]
        hold = index[(index > stamp) & (index <= pd.Timestamp(nxt))]
        if len(hold) == 0:
            continue
        windows.append(
            {
                "signal": stamp,
                "signal_month": stamp.strftime("%Y-%m"),
                "hold": hold,
            }
        )
    return windows


def _cross_section(
    factors: dict[str, pd.DataFrame],
    tickers: list[str],
    stamp: pd.Timestamp,
    column: str,
) -> pd.Series:
    """某一天、某一因子的橫斷面。索引是全部股票，沒有值的位置是 NaN。"""
    values = {ticker: _value_at(factors[ticker], stamp, column) for ticker in tickers}
    return pd.Series(values, dtype=float)


def _monthly_ic(
    factors: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    windows: list[dict[str, Any]],
    tickers: list[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    每個因子、每個訊號月的 Spearman rank IC。

    演算法與 backtest_robustness._spearman 相同：平均名次後再算 Pearson。
    持有期報酬與 backtest_robustness._holding_price_returns 相同。

    參數:
        factors: 每檔的因子表。
        close: 對齊到主日曆的收盤價。
        windows: 訊號日與持有區間。
        tickers: 組 A 全部代號。

    回傳:
        (IC 表，列是訊號月、欄是因子代號, 未滿 34 檔或 IC 無法定義的月份明細)。
    """
    ic = pd.DataFrame(
        index=[window["signal_month"] for window in windows],
        columns=[key for key, _name in _FACTORS],
        dtype=float,
    )
    shortfalls: list[dict[str, Any]] = []
    for window in windows:
        returns = _holding_price_returns(close, window["hold"]).reindex(tickers)
        for key, _label in _FACTORS:
            scores = _cross_section(factors, tickers, window["signal"], key)
            # 橫斷面完全相同時，Spearman 的分母是 0，相關係數本來就無法定義。
            # 壓掉 numpy 的警告，數值仍是 NaN，不另給一個 IC。
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                value, used, missing = _spearman(scores, returns)
            ic.loc[window["signal_month"], key] = value
            reason = ""
            if not math.isfinite(value):
                paired = pd.DataFrame({"score": scores, "ret": returns}).dropna()
                if len(paired) < 3:
                    reason = "有效樣本少於 3 檔"
                elif paired["score"].nunique(dropna=True) < 2 or paired["ret"].nunique(dropna=True) < 2:
                    reason = "橫斷面沒有差異"
                else:
                    reason = "相關係數無法定義"
            if used != len(tickers) or reason:
                shortfalls.append(
                    {
                        "factor": key,
                        "month": window["signal_month"],
                        "used": used,
                        "missing": missing,
                        "reason": reason,
                    }
                )
    ic.index.name = "signal_month"
    return ic, shortfalls


def _ic_stats(series: pd.Series) -> dict[str, float | int]:
    """
    月度 IC 的平均、樣本標準差、t 統計量與正值比例。

    t = 平均 IC / (標準差 / sqrt(有效月數))。有效月是 IC 為有限值的月份。
    少於兩個月時，標準差與 t 為 NaN。

    參數:
        series: 單一因子的月度 IC。

    回傳:
        mean、std、t、positive_ratio、n、n_positive。
    """
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    n = int(len(values))
    if n == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "t": float("nan"),
            "positive_ratio": float("nan"),
            "n": 0,
            "n_positive": 0,
        }
    mean_ic = float(values.mean())
    positive = float(np.mean(values > 0))
    n_positive = int(np.sum(values > 0))
    if n < 2:
        return {
            "mean": mean_ic,
            "std": float("nan"),
            "t": float("nan"),
            "positive_ratio": positive,
            "n": n,
            "n_positive": n_positive,
        }
    std_ic = float(values.std(ddof=1))
    stderr = std_ic / math.sqrt(n)
    t_stat = mean_ic / stderr if stderr > 0 else float("nan")
    return {
        "mean": mean_ic,
        "std": std_ic,
        "t": t_stat,
        "positive_ratio": positive,
        "n": n,
        "n_positive": n_positive,
    }


def _fmt_num(value: float, digits: int) -> str:
    """有限值格式化成固定小數；無效值顯示破折號。"""
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _fmt_ratio(ratio: float, n_positive: int, n: int) -> str:
    """IC>0 的月份比例，附上分子分母。"""
    if n <= 0 or not math.isfinite(ratio):
        return "—"
    return f"{ratio * 100:.1f}%（{n_positive}/{n}）"


def _print_definitions() -> None:
    """印出因子定義。符號如實呈現，沒有為了讓 IC 變正而改方向。"""
    print("因子定義（交易日窗口，只用當天以前的資料）", flush=True)
    lines = [
        "RSI / MACD / 布林 / 均線 / 成交量：indicators.py 的子分數，各自獨立，不加權",
        "12-1 動能：收盤[t-21] / 收盤[t-252] - 1",
        "短期反轉：收盤[t] / 收盤[t-21] - 1（不取絕對值、不反號）",
        "低波動：過去 60 個交易日簡單報酬的樣本標準差（數值愈大波動愈高，不取倒數）",
        "52 週高點接近度：當日收盤 / 過去 252 個交易日最高收盤（含當日）",
        "量能趨勢：近 20 日均量 / 近 60 日均量",
    ]
    for line in lines:
        print(f"  {line}", flush=True)


def _print_ic_table(ic: pd.DataFrame) -> pd.DataFrame:
    """
    印出每個因子一列的 IC 摘要，依 t 統計量絕對值由大到小排。

    參數:
        ic: 月度 IC。列是訊號月，欄是因子代號。

    回傳:
        排序後的統計表，數值仍是 float，方便呼叫端接著使用。
    """
    rows: list[dict[str, Any]] = []
    for key, label in _FACTORS:
        stats = _ic_stats(ic[key])
        t_value = float(stats["t"])
        rows.append(
            {
                "key": key,
                "因子": label,
                "平均 IC": float(stats["mean"]),
                "IC 標準差": float(stats["std"]),
                "t 統計量": t_value,
                "abs_t": abs(t_value) if math.isfinite(t_value) else -1.0,
                "IC>0 月份比例": _fmt_ratio(
                    float(stats["positive_ratio"]),
                    int(stats["n_positive"]),
                    int(stats["n"]),
                ),
                "有效月數": int(stats["n"]),
            }
        )
    rows.sort(key=lambda row: (-float(row["abs_t"]), str(row["因子"])))
    display = pd.DataFrame(
        {
            "因子": [row["因子"] for row in rows],
            "平均 IC": [_fmt_num(row["平均 IC"], 3) for row in rows],
            "IC 標準差": [_fmt_num(row["IC 標準差"], 3) for row in rows],
            "t 統計量": [_fmt_num(row["t 統計量"], 2) for row in rows],
            "IC>0 月份比例": [row["IC>0 月份比例"] for row in rows],
            "有效月數": [row["有效月數"] for row in rows],
        }
    )
    print("\n=== 月度 rank IC（Spearman，依 |t| 由大到小）===", flush=True)
    print(
        "t = 平均 IC / (IC 樣本標準差 / sqrt(有效月數))。IC>0 的分母是有效月數。",
        flush=True,
    )
    print(display.to_string(index=False), flush=True)
    return pd.DataFrame(rows)


def _print_correlation(ic: pd.DataFrame) -> None:
    """
    用每月 IC 序列算因子之間的皮爾森相關。

    這張矩陣看的是「哪些因子的預測力一起出現」，不是因子值本身的橫斷面相關。
    兩兩只用同時有 IC 的月份；重疊少於 3 個月則留空。

    參數:
        ic: 月度 IC。
    """
    corr = ic.astype(float).corr(method="pearson", min_periods=_MIN_CORR_MONTHS)
    shown = corr.rename(index=_SHORT, columns=_SHORT)
    print("\n=== 因子月度 IC 序列的皮爾森相關 ===", flush=True)
    print("列與欄是因子的月 IC，不是股票的因子值。對角是 1。", flush=True)
    print(shown.to_string(float_format=lambda value: f"{value:6.2f}", na_rep="   —"), flush=True)

    pairs: list[tuple[float, float, str, str]] = []
    keys = [key for key, _label in _FACTORS]
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            value = corr.loc[left, right]
            if pd.isna(value) or not math.isfinite(float(value)):
                continue
            pairs.append((abs(float(value)), float(value), _SHORT[left], _SHORT[right]))
    pairs.sort(key=lambda item: (-item[0], item[2], item[3]))
    print("\n絕對相關最高的因子對", flush=True)
    if not pairs:
        print("沒有足夠重疊的月份可以算相關。", flush=True)
        return
    for _abs_value, value, left, right in pairs[:5]:
        print(f"  {left} vs {right}  {value:+.2f}", flush=True)


def _print_coverage(shortfalls: list[dict[str, Any]], n_months: int) -> None:
    """
    說明哪些因子沒有把 34 檔都納進某個月，以及哪些月份的 IC 無法定義。

    參數:
        shortfalls: 未滿全樣本，或相關係數無法定義的月份。
        n_months: 訊號月總數。
    """
    print("\n=== 橫斷面涵蓋 ===", flush=True)
    if not shortfalls:
        print(f"每個因子的 {n_months} 個月份都納入 {_GROUP_SIZE} 檔。", flush=True)
        return
    print(
        f"目標是每個月 {_GROUP_SIZE} 檔。下面這些月份少了因子值或持有期報酬，該月 IC 只用得齊的股票。",
        flush=True,
    )
    by_factor: dict[str, list[dict[str, Any]]] = {key: [] for key, _label in _FACTORS}
    for row in shortfalls:
        by_factor[str(row["factor"])].append(row)
    labels = dict(_FACTORS)
    for key, _label in _FACTORS:
        rows = by_factor[key]
        thin = [row for row in rows if int(row["used"]) != _GROUP_SIZE]
        undefined = [row for row in rows if row.get("reason")]
        if not thin and not undefined:
            print(f"{labels[key]}：{n_months} 個月都是 {_GROUP_SIZE} 檔", flush=True)
            continue
        if thin:
            absent: dict[str, int] = {}
            for row in thin:
                for ticker in row["missing"]:
                    absent[str(ticker)] = absent.get(str(ticker), 0) + 1
            detail = "，".join(f"{ticker} 缺 {count} 個月" for ticker, count in sorted(absent.items()))
            print(
                f"{labels[key]}：{len(thin)} 個月未滿 {_GROUP_SIZE} 檔（{detail}）",
                flush=True,
            )
        else:
            print(f"{labels[key]}：{n_months} 個月都是 {_GROUP_SIZE} 檔", flush=True)
        for row in undefined:
            print(
                f"  {row['month']} 的 IC 無法定義（{row['reason']}，該月有 {row['used']} 檔）",
                flush=True,
            )


def _base_tech_weights() -> dict[str, float]:
    """
    讀取 config.TECH_SCORE_WEIGHTS，順序固定為五項子分數。

    回傳:
        子分數代號對到權重。缺任何一項就拒絕，避免默默少算一項。
    """
    missing = [key for key in _SUBSCORE_KEYS if key not in TECH_SCORE_WEIGHTS]
    if missing:
        raise KeyError("TECH_SCORE_WEIGHTS 缺少：" + "、".join(missing))
    return {key: float(TECH_SCORE_WEIGHTS[key]) for key in _SUBSCORE_KEYS}


def _renormalize(base: dict[str, float], keep: tuple[str, ...]) -> dict[str, float]:
    """
    保留指定子分數，並按原本的相對比例把權重總和調成 1。

    參數:
        base: config 的原始權重。
        keep: 要留下的子分數代號。

    回傳:
        五項都在的新權重。被拿掉的項目是 0。
    """
    total = sum(base[key] for key in keep)
    if total == 0:
        raise ValueError("要留下的權重總和是 0，無法歸一化")
    return {key: (base[key] / total if key in keep else 0.0) for key in _SUBSCORE_KEYS}


def _composite_schemes() -> list[dict[str, Any]]:
    """
    五種綜合分的權重。評分規則不變，只改加權。

    RSI 反向是把 RSI 的權重改成負數，其餘四項維持 config 的原值，不重新歸一。
    只保留布林與均線是題目指定的各 0.50，不從原比例換算。

    回傳:
        每種一列，含名稱、權重、這組權重怎麼來的。
    """
    base = _base_tech_weights()
    flipped = dict(base)
    flipped["rsi"] = -base["rsi"]
    trend_only = {key: 0.0 for key in _SUBSCORE_KEYS}
    trend_only["bollinger"] = 0.5
    trend_only["ma_trend"] = 0.5
    return [
        {"name": "現況", "weights": base, "note": "config.TECH_SCORE_WEIGHTS"},
        {
            "name": "移除 RSI",
            "weights": _renormalize(base, ("macd", "bollinger", "ma_trend", "volume")),
            "note": "其餘四項按原比例歸一",
        },
        {"name": "RSI 反向", "weights": flipped, "note": "RSI 權重取負，其餘不變"},
        {
            "name": "移除 RSI 與 MACD",
            "weights": _renormalize(base, ("bollinger", "ma_trend", "volume")),
            "note": "其餘三項按原比例歸一",
        },
        {"name": "只保留布林與均線", "weights": trend_only, "note": "各 0.50"},
    ]


def _fmt_weight(value: float) -> str:
    """權重帶正負號。去掉多餘的 0，至少留兩位小數。"""
    text = f"{value:+.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".00"
    else:
        decimals = text.split(".", 1)[1]
        if len(decimals) < 2:
            text += "0" * (2 - len(decimals))
    return text


def _fmt_weight_set(weights: dict[str, float]) -> str:
    """把五項權重印成一行。權重為 0 的項目也留下，避免看成被漏掉。"""
    labels = {"rsi": "RSI", "macd": "MACD", "bollinger": "布林", "ma_trend": "均線", "volume": "成交量"}
    return "、".join(f"{labels[key]} {_fmt_weight(weights[key])}" for key in _SUBSCORE_KEYS)


def _composite_cross_section(
    factors: dict[str, pd.DataFrame],
    tickers: list[str],
    stamp: pd.Timestamp,
    weights: dict[str, float],
) -> pd.Series:
    """
    某一天的綜合分橫斷面。

    權重不是 0 的子分數只要有一個缺失，這檔就是 NaN，不用中性分數補上。
    子分數本身若已是評分函式回傳的 50，這裡照用，不另行改寫。

    參數:
        factors: 每檔的因子表。
        tickers: 組 A 全部代號。
        stamp: 訊號日。
        weights: 五項子分數的權重，可以是負的。

    回傳:
        索引為全部股票的綜合分。缺失是 NaN。
    """
    values: dict[str, float] = {}
    for ticker in tickers:
        total = 0.0
        complete = True
        for key, weight in weights.items():
            if weight == 0.0:
                continue
            value = _value_at(factors[ticker], stamp, key)
            if not math.isfinite(value):
                complete = False
                break
            total += weight * value
        values[ticker] = total if complete else float("nan")
    return pd.Series(values, dtype=float)


def _composite_monthly_ic(
    factors: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    windows: list[dict[str, Any]],
    tickers: list[str],
    weights: dict[str, float],
) -> pd.Series:
    """
    一種綜合分的月度 rank IC。算法與單一因子相同。

    參數:
        factors: 每檔的因子表。
        close: 對齊到主日曆的收盤價。
        windows: 訊號日與持有區間。
        tickers: 組 A 全部代號。
        weights: 這種綜合分的權重。

    回傳:
        索引為訊號月的 IC。橫斷面沒有差異時該月是 NaN。
    """
    values: dict[str, float] = {}
    for window in windows:
        returns = _holding_price_returns(close, window["hold"]).reindex(tickers)
        scores = _composite_cross_section(factors, tickers, window["signal"], weights)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ic, _used, _missing = _spearman(scores, returns)
        values[window["signal_month"]] = ic
    return pd.Series(values, dtype=float, name="ic")


def _hypothesis_sentence(rows: list[dict[str, Any]]) -> str:
    """
    用平均 IC 相對現況的方向，描述假設有沒有被這次比較支持。

    假設是「目前符號的 RSI 在拖累綜合分」。支持的方向是：拿掉 RSI，
    以及把 RSI 權重取負，平均 IC 都高於現況。這裡只描述方向。

    參數:
        rows: 五種方案的統計，第一項必須是現況。

    回傳:
        一句話。
    """
    current = float(rows[0]["mean"])
    by_name = {str(row["name"]): row for row in rows}
    drop = by_name["移除 RSI"]
    flip = by_name["RSI 反向"]
    drop_higher = float(drop["mean"]) > current
    flip_higher = float(flip["mean"]) > current
    if drop_higher and flip_higher:
        return "移除 RSI，以及把 RSI 權重取負，平均 IC 都高於現況。變化方向與「目前的 RSI 子分數在拖累綜合分」一致。"
    if not drop_higher and not flip_higher:
        return "移除 RSI，以及把 RSI 權重取負，平均 IC 都沒有高於現況。變化方向與這個假設不一致。"
    if drop_higher:
        return "移除 RSI 的平均 IC 高於現況，但 RSI 權重取負沒有。變化方向只部分符合假設。"
    return "RSI 權重取負的平均 IC 高於現況，但移除 RSI 沒有。變化方向只部分符合假設。"


def _print_composite_comparison(
    factors: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    windows: list[dict[str, Any]],
    tickers: list[str],
) -> None:
    """
    印出五種綜合分的 rank IC。順序固定，不依 IC 重排。

    參數:
        factors: 每檔的因子表。子分數已經做過前視偏差檢查。
        close: 對齊到主日曆的收盤價。
        windows: 訊號日與持有區間。
        tickers: 組 A 全部代號。
    """
    schemes = _composite_schemes()
    print("\n=== 綜合分權重比較（同一批子分數，只算 rank IC）===", flush=True)
    print(
        "假設是 RSI 子分數的方向錯誤，拖累了綜合分。下面看的是相對現況的變化方向。",
        flush=True,
    )
    print(f"config.TECH_SCORE_WEIGHTS：{_fmt_weight_set(schemes[0]['weights'])}", flush=True)
    print("綜合分是子分數的加權和。評分規則沒有改。", flush=True)
    for scheme in schemes:
        print(f"  {scheme['name']}（{scheme['note']}）：{_fmt_weight_set(scheme['weights'])}", flush=True)

    rows: list[dict[str, Any]] = []
    for scheme in schemes:
        series = _composite_monthly_ic(factors, close, windows, tickers, scheme["weights"])
        stats = _ic_stats(series)
        rows.append({"name": scheme["name"], "mean": float(stats["mean"]), **stats})
    current_mean = float(rows[0]["mean"])
    display = pd.DataFrame(
        {
            "方案": [row["name"] for row in rows],
            "平均 IC": [_fmt_num(float(row["mean"]), 3) for row in rows],
            "IC 標準差": [_fmt_num(float(row["std"]), 3) for row in rows],
            "t 統計量": [_fmt_num(float(row["t"]), 2) for row in rows],
            "IC>0 月份比例": [
                _fmt_ratio(float(row["positive_ratio"]), int(row["n_positive"]), int(row["n"]))
                for row in rows
            ],
            "平均 IC − 現況": [
                "—" if row["name"] == "現況" else _fmt_num(float(row["mean"]) - current_mean, 3)
                for row in rows
            ],
        }
    )
    print(display.to_string(index=False), flush=True)
    print(_hypothesis_sentence(rows), flush=True)
    print(
        "這 5 種是同一批資料上的事後比較，即使某一種 IC 較高，也不構成「這個權重有效」的證據。",
        flush=True,
    )


def _print_placeholder_scores(
    missing_inputs: dict[str, pd.DataFrame],
    windows: list[dict[str, Any]],
) -> None:
    """
    子分數在指標還沒算出來時會得到中性 50。把這種股票月數列出來。

    參數:
        missing_inputs: 每檔的缺失表。
        windows: 訊號月。只統計這些月底，不統計暖機期。
    """
    counts: dict[str, dict[str, int]] = {key: {} for key in _SUBSCORE_KEYS}
    for ticker, frame in missing_inputs.items():
        if frame.empty:
            continue
        for window in windows:
            stamp = window["signal"]
            if stamp not in frame.index:
                continue
            for key in _SUBSCORE_KEYS:
                if bool(frame.at[stamp, key]):
                    counts[key][ticker] = counts[key].get(ticker, 0) + 1
    labels = dict(_FACTORS)
    print("\n=== 子分數輸入缺失（評分函式回傳中性 50）===", flush=True)
    if not any(counts.values()):
        print("訊號月上沒有這種情形。", flush=True)
        return
    for key in _SUBSCORE_KEYS:
        if not counts[key]:
            continue
        detail = "，".join(f"{ticker} {count} 個月" for ticker, count in sorted(counts[key].items()))
        total = sum(counts[key].values())
        print(f"  {labels[key]}：{total} 個股票月（{detail}）", flush=True)


def _load_histories(tickers: list[str]) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex, pd.DataFrame]:
    """
    讀取組 A 與 SPY 的價量，並做成與現有 rank IC 相同的收盤價面板。

    參數:
        tickers: 組 A 代號。

    回傳:
        (每檔已整理的 OHLCV, SPY 日曆, 對齊日曆的收盤價)。
        缺任何一檔或沒有 SPY 時拋出 RuntimeError。
    """
    fetch_start = BACKTEST_START - timedelta(days=_WARMUP_DAYS)
    panel = build_ohlcv_panel([*tickers, _CALENDAR], fetch_start, BACKTEST_END)
    spy = panel.get(_CALENDAR)
    if spy is None or spy.empty:
        raise RuntimeError("沒有 SPY 價量，無法沿用現有的月底日曆")
    missing = [ticker for ticker in tickers if ticker not in panel or panel[ticker] is None or panel[ticker].empty]
    if missing:
        raise RuntimeError("這些代號沒有價量，無法把 34 檔都納入：" + ", ".join(missing))

    master = _as_timestamp_index(spy.index)
    master = master[(master >= pd.Timestamp(fetch_start)) & (master <= pd.Timestamp(BACKTEST_END))]
    histories: dict[str, pd.DataFrame] = {}
    closes: dict[str, pd.Series] = {}
    for ticker in tickers:
        history = _prepare_ohlcv(panel[ticker])
        histories[ticker] = history
        closes[ticker] = history["Close"].reindex(master)
    close = pd.DataFrame(closes, index=master)
    return histories, master, close


def run_factor_research() -> int:
    """
    跑組 A 的因子 IC，並先做前視偏差檢查。

    回傳:
        0 表示檢查通過且表已印出；1 表示中止。
    """
    logging.getLogger("backtest_data").setLevel(logging.WARNING)
    logging.getLogger("backtest_universe_sp500").setLevel(logging.WARNING)
    logging.getLogger("backtest_compare_groups").setLevel(logging.WARNING)

    try:
        groups = define_three_groups()
        tickers = list(groups["A"])
    except Exception:
        _log.exception("組 A 名單建立失敗")
        print("組 A 名單建立失敗，因子研究中止。", flush=True)
        return 1

    print(f"組 A（{len(tickers)} 檔）：{', '.join(tickers)}", flush=True)
    if len(tickers) != _GROUP_SIZE:
        print(f"組 A 不是 {_GROUP_SIZE} 檔，因子研究中止。", flush=True)
        return 1

    try:
        histories, master, close = _load_histories(tickers)
    except Exception as exc:
        _log.exception("價量讀取失敗")
        print(f"價量讀取失敗，因子研究中止：{exc}", flush=True)
        return 1

    windows = _holding_windows(master)
    if not windows:
        print("沒有任何訊號月，因子研究中止。", flush=True)
        return 1
    signals = [window["signal"] for window in windows]
    # 子分數對照要用每檔最後一個交易日，所以評分日多加這一天。
    score_dates = pd.DatetimeIndex([*signals, *(history.index[-1] for history in histories.values())])

    print(
        f"固定區間 {BACKTEST_START.isoformat()} 至 {BACKTEST_END.isoformat()}。"
        f"訊號月 {windows[0]['signal_month']} 至 {windows[-1]['signal_month']}，共 {len(windows)} 個。",
        flush=True,
    )
    print(
        f"第一段持有 {windows[0]['hold'][0].date().isoformat()} 至 {windows[0]['hold'][-1].date().isoformat()}；"
        f"最後一段持有 {windows[-1]['hold'][0].date().isoformat()} 至 {windows[-1]['hold'][-1].date().isoformat()}。",
        flush=True,
    )
    _print_definitions()

    factors: dict[str, pd.DataFrame] = {}
    missing_inputs: dict[str, pd.DataFrame] = {}
    for ticker, history in histories.items():
        try:
            factor_frame, missing = compute_factors(history, score_dates=score_dates)
        except Exception:
            _log.exception("%s 因子計算失敗", ticker)
            print(f"{ticker} 因子計算失敗，因子研究中止。", flush=True)
            return 1
        factors[ticker] = factor_frame
        missing_inputs[ticker] = missing

    if not _check_indicator_api(histories, factors):
        print("子分數與 indicators.calculate_tech_score 不一致，因子研究中止。", flush=True)
        return 1

    print("\n開始前視偏差檢查：每個訊號月只保留到該月底再重算。", flush=True)
    try:
        mismatches = verify_no_lookahead(histories, factors, signals)
    except Exception:
        _log.exception("前視偏差檢查發生例外")
        print("前視偏差檢查發生例外，因子研究中止。", flush=True)
        return 1
    if mismatches:
        print(
            f"前視偏差檢查失敗：{len(mismatches)} 處不一致。因子研究中止，不計算 IC。",
            flush=True,
        )
        for row in mismatches[:12]:
            detail = f"，{row['detail']}" if row["detail"] else ""
            print(
                f"  {row['factor']}  {row['month']}  {row['ticker']}"
                f"  完整資料={row['full']}  截斷至該月={row['truncated']}{detail}",
                flush=True,
            )
        if len(mismatches) > 12:
            print(f"  …其餘 {len(mismatches) - 12} 筆省略", flush=True)
        return 1

    finite_counts = {
        key: int(sum(math.isfinite(_value_at(frame, signal, key)) for frame in factors.values() for signal in signals))
        for key, _label in _FACTORS
    }
    empty = [label for key, label in _FACTORS if finite_counts[key] == 0]
    if empty:
        print("這些因子在所有訊號月都沒有數值，因子研究中止：" + "、".join(empty), flush=True)
        return 1
    print(
        f"前視偏差檢查通過：{len(_FACTORS)} 個因子 × {len(signals)} 個訊號月 × {len(tickers)} 檔，"
        "截斷至該月與完整資料一致。",
        flush=True,
    )

    ic, shortfalls = _monthly_ic(factors, close, windows, tickers)
    _print_ic_table(ic)
    _print_correlation(ic)
    _print_coverage(shortfalls, len(windows))
    _print_placeholder_scores(missing_inputs, windows)
    try:
        _print_composite_comparison(factors, close, windows, tickers)
    except Exception:
        _log.exception("綜合分權重比較失敗")
        print("綜合分權重比較失敗。單一因子的 IC 已印出，這段比較中止。", flush=True)
        return 1
    return 0


def main() -> None:
    """執行因子研究。檢查失敗時以離開碼 1 結束。"""
    sys.exit(run_factor_research())


if __name__ == "__main__":
    main()
