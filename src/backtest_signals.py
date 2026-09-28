# -*- coding: utf-8 -*-
"""
回測用純技術面訊號：對每一個交易日產出 0-100 技術面總分時間序列。

本模組只使用價量技術指標（RSI / MACD / 布林 / 均線 / 成交量），
不碰基本面、新聞、估值，以避免歷史回測的前視偏差（lookahead bias）。

第 t 天的分數只使用第 t 天（含）以前的資料：
- 指標本身來自 pandas rolling / ewm（因果運算，不含未來）
- 評分規則沿用 indicators.py 的 if-else 邏輯，改寫成整條序列的向量化計算
- MACD 金叉/死叉只看 shift(1) 的「前一日」，絕不使用 shift(-1)

執行方式（於專案根目錄）：
    python src/backtest_signals.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_SRC = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import TECH_SCORE_WEIGHTS  # noqa: E402
from indicators import calculate_indicators, calculate_tech_score  # noqa: E402

_log = logging.getLogger("backtest_signals")
if not _log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# 回測一律走 pandas 手動指標，避免短序列時 ta 套件與手動路徑切換造成假陽性
_USE_TA_FOR_BACKTEST = False

# 分數比對容差（加權後 round(2) 仍可能有極小數值誤差）
_SCORE_ATOL = 1e-6


def _as_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    """取出指標欄位為 float Series；欄位不存在則回傳全 NaN。"""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").astype(float)


def _score_rsi_series(rsi: pd.Series) -> pd.Series:
    """
    向量化 RSI 子分數（0-100），規則與 indicators._score_rsi 一致。

    參數:
        rsi: 當日 RSI 序列。

    回傳:
        與 rsi 同索引的分數序列；RSI 為 NaN 時給中性 50 分。
    """
    # np.select 依序取第一個成立的條件；NaN 比較皆為 False，最後用 valid 覆蓋
    chosen = np.select(
        [
            rsi < 30,
            rsi < 40,
            rsi <= 60,
            rsi <= 70,
        ],
        [85.0, 75.0, 70.0, 60.0],
        default=40.0,
    )
    return pd.Series(np.where(rsi.notna(), chosen, 50.0), index=rsi.index, dtype=float)


def _score_macd_series(
    dif: pd.Series,
    dea: pd.Series,
    hist: pd.Series,
) -> pd.Series:
    """
    向量化 MACD 子分數，規則與 indicators._score_macd 一致（含優先順序）。

    金叉/死叉與柱狀放大只使用前一日（shift 1），不會讀到未來。

    參數:
        dif: MACD DIF。
        dea: MACD DEA（訊號線）。
        hist: MACD 柱狀圖。

    回傳:
        與 dif 同索引的分數序列；DIF/DEA/HIST 任一為 NaN 時給 50 分。
    """
    dif_prev = dif.shift(1)
    dea_prev = dea.shift(1)
    hist_prev = hist.shift(1)

    valid = dif.notna() & dea.notna() & hist.notna()
    prev_ok = dif_prev.notna() & dea_prev.notna()
    hist_prev_ok = hist_prev.notna()

    golden = prev_ok & (dif_prev <= dea_prev) & (dif > dea)
    death = prev_ok & (dif_prev >= dea_prev) & (dif < dea)
    bull = (dif > dea) & (hist > 0)
    bear = (dif < dea) & (hist < 0)
    hist_expand = hist_prev_ok & (hist > hist_prev)
    above_zero_hist = (dif > 0) & hist_expand
    hist_rising = hist_expand

    chosen = np.select(
        [
            golden,
            death,
            bull,
            above_zero_hist,
            bear,
            (dif > 0) & hist_rising,
            (dif > 0) & ~hist_rising,
            (dif < 0) & hist_rising,
        ],
        [90.0, 30.0, 75.0, 70.0, 40.0, 65.0, 55.0, 50.0],
        default=45.0,
    )
    return pd.Series(np.where(valid, chosen, 50.0), index=dif.index, dtype=float)


def _score_bollinger_series(
    close: pd.Series,
    upper: pd.Series,
    mid: pd.Series,
    lower: pd.Series,
) -> pd.Series:
    """
    向量化布林帶子分數，規則與 indicators._score_bollinger 一致。

    參數:
        close: 收盤價。
        upper: 布林上軌。
        mid: 布林中軌。
        lower: 布林下軌。

    回傳:
        與 close 同索引的分數序列；任一輸入為 NaN 時給 50 分。
    """
    valid = close.notna() & upper.notna() & mid.notna() & lower.notna()
    chosen = np.select(
        [
            close <= lower,
            close > upper,
            (close > mid) & (close < upper),
        ],
        [85.0, 50.0, 70.0],
        default=45.0,
    )
    return pd.Series(np.where(valid, chosen, 50.0), index=close.index, dtype=float)


def _score_ma_trend_series(
    close: pd.Series,
    ma5: pd.Series,
    ma20: pd.Series,
    ma60: pd.Series,
    ma200: pd.Series,
) -> pd.Series:
    """
    向量化均線排列子分數，規則與 indicators._score_ma_trend 一致。

    參數:
        close: 收盤價。
        ma5 / ma20 / ma60 / ma200: 各期移動平均。

    回傳:
        與 close 同索引的分數序列；close/MA20/MA60 缺值時給 50 分。
    """
    base_ok = close.notna() & ma20.notna() & ma60.notna()
    ma5_ok = ma5.notna()
    ma200_ok = ma200.notna()

    full_bull = ma5_ok & ma200_ok & (ma5 > ma20) & (ma20 > ma60) & (ma60 > ma200)
    partial_bull = ma200_ok & (ma20 > ma60) & (ma60 > ma200)
    full_bear = ma5_ok & ma200_ok & (ma5 < ma20) & (ma20 < ma60) & (ma60 < ma200)
    above_20_60 = (close > ma20) & (close > ma60)
    below_20_above_60 = (close < ma20) & (close > ma60)

    chosen = np.select(
        [
            full_bull,
            partial_bull,
            full_bear,
            above_20_60,
            below_20_above_60,
        ],
        [90.0, 75.0, 30.0, 70.0, 50.0],
        default=55.0,
    )
    return pd.Series(np.where(base_ok, chosen, 50.0), index=close.index, dtype=float)


def _score_volume_series(
    vol_ratio: pd.Series,
    open_: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """
    向量化成交量子分數，規則與 indicators._score_volume 一致。

    使用當日 Open/Close 判斷紅黑 K，屬「第 t 天可得資訊」，不是前視。

    參數:
        vol_ratio: 當日量 / 20 日均量。
        open_: 開盤價。
        close: 收盤價。

    回傳:
        與 close 同索引的分數序列；任一輸入為 NaN 時給 50 分。
    """
    valid = vol_ratio.notna() & open_.notna() & close.notna()
    red = close > open_
    black = close < open_
    chosen = np.select(
        [
            (vol_ratio > 2.0) & black,
            (vol_ratio > 1.5) & red,
            vol_ratio < 0.5,
            (vol_ratio >= 0.8) & (vol_ratio <= 1.5),
        ],
        [35.0, 85.0, 50.0, 65.0],
        default=60.0,
    )
    return pd.Series(np.where(valid, chosen, 50.0), index=close.index, dtype=float)


def compute_tech_score_series(price_history: pd.DataFrame) -> pd.Series:
    """
    對每一個交易日計算當天的技術面總分（0-100），回傳一條時間序列。

    第 t 天分數只使用第 t 天（含）為止的價量：先一次算出整條因果指標，
    再把 RSI/MACD/布林/均線/成交量評分規則向量化加權。

    參數:
        price_history: 需包含 Open、High、Low、Close、Volume；索引建議為日期。

    回傳:
        與 price_history 同索引的 Series，名稱為 tech_score；無法計算時為空 Series。
    """
    if price_history is None or price_history.empty:
        _log.warning("price_history 為空，無法計算技術面分數序列")
        return pd.Series(dtype=float, name="tech_score")

    try:
        df = calculate_indicators(price_history, use_ta=_USE_TA_FOR_BACKTEST)
    except Exception:
        _log.exception("計算技術指標失敗，回傳空序列")
        return pd.Series(dtype=float, name="tech_score")

    close = _as_float_series(df, "Close")
    open_ = _as_float_series(df, "Open")

    rsi_sc = _score_rsi_series(_as_float_series(df, "RSI"))
    macd_sc = _score_macd_series(
        _as_float_series(df, "MACD_DIF"),
        _as_float_series(df, "MACD_DEA"),
        _as_float_series(df, "MACD_HIST"),
    )
    boll_sc = _score_bollinger_series(
        close,
        _as_float_series(df, "BB_UPPER"),
        _as_float_series(df, "BB_MIDDLE"),
        _as_float_series(df, "BB_LOWER"),
    )
    ma_sc = _score_ma_trend_series(
        close,
        _as_float_series(df, "MA5"),
        _as_float_series(df, "MA20"),
        _as_float_series(df, "MA60"),
        _as_float_series(df, "MA200"),
    )
    vol_sc = _score_volume_series(_as_float_series(df, "VOL_RATIO"), open_, close)

    w = TECH_SCORE_WEIGHTS
    total = (
        rsi_sc * w["rsi"]
        + macd_sc * w["macd"]
        + boll_sc * w["bollinger"]
        + ma_sc * w["ma_trend"]
        + vol_sc * w["volume"]
    )
    total = total.round(2)
    total.name = "tech_score"
    return total


def _values_equal(a: float, b: float, atol: float) -> bool:
    """兩個分數相等（兩邊皆 NaN 視為相等）。"""
    a_nan = pd.isna(a)
    b_nan = pd.isna(b)
    if a_nan and b_nan:
        return True
    if a_nan or b_nan:
        return False
    return bool(np.isclose(float(a), float(b), atol=atol, rtol=0.0))


def _format_ts(value: object) -> str:
    """把索引值格式化成可讀日期字串。"""
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return str(value)
    return ts.strftime("%Y-%m-%d")


def verify_no_lookahead(
    price_history: pd.DataFrame,
    *,
    atol: float = _SCORE_ATOL,
    max_report: int = 10,
) -> bool:
    """
    前視偏差自我檢查：完整資料算出的第 t 天分數，必須等於
    「只截斷到第 t 天」再重算的最後一日分數。

    若不相等，代表第 t 天用到了 t+1 之後的資料（lookahead bias）。

    參數:
        price_history: 原始 OHLCV。
        atol: 分數容許絕對誤差。
        max_report: 最多印出幾筆差異位置。

    回傳:
        True 表示通過（無前視偏差）；False 表示發現不一致或資料不足。
    """
    if price_history is None or price_history.empty:
        _log.warning("前視偏差檢查失敗：price_history 為空")
        return False

    try:
        df = price_history.sort_index()
        full_scores = compute_tech_score_series(df)
    except Exception:
        _log.exception("前視偏差檢查失敗：完整資料無法計算分數")
        return False

    if full_scores.empty or len(full_scores) != len(df):
        _log.warning("前視偏差檢查失敗：完整分數序列長度異常")
        return False

    mismatches: list[tuple[str, float, float, float]] = []
    n = len(df)

    for t in range(n):
        truncated = df.iloc[: t + 1]
        try:
            trunc_scores = compute_tech_score_series(truncated)
        except Exception as exc:
            label = _format_ts(df.index[t])
            _log.warning("前視偏差檢查失敗：截斷至 %s 時計算例外: %s", label, exc)
            return False

        if trunc_scores.empty:
            label = _format_ts(df.index[t])
            mismatches.append((label, float(full_scores.iloc[t]), float("nan"), float("nan")))
            continue

        full_val = full_scores.iloc[t]
        trunc_val = trunc_scores.iloc[-1]
        if not _values_equal(full_val, trunc_val, atol):
            f_num = float(full_val) if pd.notna(full_val) else float("nan")
            t_num = float(trunc_val) if pd.notna(trunc_val) else float("nan")
            diff = f_num - t_num
            mismatches.append((_format_ts(df.index[t]), f_num, t_num, diff))

    if not mismatches:
        _log.info("前視偏差檢查通過：%s 個交易日分數完全一致", n)
        return True

    _log.warning(
        "前視偏差檢查失敗：共 %s / %s 處第 t 天分數不一致（完整資料 vs 截斷至 t）",
        len(mismatches),
        n,
    )
    for label, full_val, trunc_val, diff in mismatches[:max_report]:
        _log.warning(
            "  日期 %s: 完整資料=%.4f, 截斷資料=%.4f, 差=%.4f",
            label,
            full_val,
            trunc_val,
            diff,
        )
    if len(mismatches) > max_report:
        _log.warning("  …其餘 %s 筆省略", len(mismatches) - max_report)
    return False


def _print_last_scores(scores: pd.Series, n: int = 10) -> None:
    """在終端機印出最後 n 天的技術面分數。"""
    tail = scores.tail(n)
    print("=== 最後 10 個交易日技術面分數 ===")
    print(f"{'日期':<12} {'tech_score':>10}")
    for idx, val in tail.items():
        print(f"{_format_ts(idx):<12} {float(val):>10.2f}")


def _check_matches_last_day_api(price_history: pd.DataFrame, scores: pd.Series) -> bool:
    """
    額外核對：序列最後一日應與 calculate_tech_score（同一套手動指標）一致。

    這不是前視檢查，而是確認向量化評分沒有寫歪既有規則。
    """
    try:
        df_ind = calculate_indicators(price_history, use_ta=_USE_TA_FOR_BACKTEST)
        last = calculate_tech_score(df_ind)
        expected = float(last["total_score"])
        got = float(scores.iloc[-1])
    except Exception:
        _log.exception("與 calculate_tech_score 對照時發生例外")
        return False

    ok = _values_equal(expected, got, atol=0.01)
    if ok:
        _log.info("與 calculate_tech_score 最後一日一致: %.2f", got)
    else:
        _log.warning(
            "與 calculate_tech_score 最後一日不一致: 序列=%.2f, 既有 API=%.2f",
            got,
            expected,
        )
    return ok


if __name__ == "__main__":
    from data_collector import fetch_stock_data

    raw = fetch_stock_data("NVDA", force_refresh=False)
    if not raw:
        print("NVDA 資料抓取失敗，請檢查網路或稍後再試。")
        sys.exit(1)

    ph = raw["price_history"]
    print(
        f"NVDA 價量筆數: {len(ph)}（{ _format_ts(ph.index[0]) } ~ { _format_ts(ph.index[-1]) }）",
        flush=True,
    )

    scores = compute_tech_score_series(ph)
    if scores.empty:
        print("技術面分數序列為空，測試中止。", flush=True)
        sys.exit(1)

    _print_last_scores(scores, n=10)
    print(
        f"分數統計: min={float(scores.min()):.2f}  "
        f"median={float(scores.median()):.2f}  "
        f"max={float(scores.max()):.2f}",
        flush=True,
    )

    last_ok = _check_matches_last_day_api(ph, scores)
    print("與既有 calculate_tech_score 對照:", "通過" if last_ok else "失敗", flush=True)

    print("開始前視偏差檢查（每個交易日截斷重算，可能需要數秒）...", flush=True)
    passed = verify_no_lookahead(ph)
    print("verify_no_lookahead:", "通過" if passed else "失敗", flush=True)

    sys.exit(0 if passed and last_ok else 1)
