# -*- coding: utf-8 -*-
"""
技術指標計算與綜合技術評分。

依賴 config.TECH_PARAMS、TECH_SCORE_WEIGHTS；價量資料建議來自 data_collector.fetch_stock_data。

執行方式（於專案根目錄）：
    python src/indicators.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# 從專案根目錄載入 config（直接執行本檔時 sys.path 預設為 src/）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import TECH_PARAMS, TECH_SCORE_WEIGHTS  # noqa: E402

# 嘗試載入 ta；失敗則全程使用手動計算
try:
    from ta.momentum import ROCIndicator, RSIIndicator, StochasticOscillator
    from ta.trend import ADXIndicator, MACD, SMAIndicator
    from ta.volatility import AverageTrueRange, BollingerBands
    from ta.volume import OnBalanceVolumeIndicator

    _TA_AVAILABLE = True
except Exception:  # noqa: BLE001 — 允許任何匯入錯誤並改用手動實作
    _TA_AVAILABLE = False


def _ma_periods() -> tuple[int, int, int, int]:
    """由 TECH_PARAMS 取得 MA5/MA20/MA60/MA200 週期。"""
    p = TECH_PARAMS.get("ma_periods", [5, 20, 60, 200])
    if len(p) != 4:
        return 5, 20, 60, 200
    return int(p[0]), int(p[1]), int(p[2]), int(p[3])


def _wilder_ema(s: pd.Series, length: int) -> pd.Series:
    """Wilder 平滑（與 ta 內部常用的 ewm(alpha=1/length) 一致）。"""
    return s.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()


def _adx_manual(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """
    手動計算 ADX（平均趨向指標）：衡量趨勢強度，與方向無關。
    使用 +DI/-DI 與 DX 的 Wilder 平滑近似實作。
    """
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)

    atr = _wilder_ema(tr, length)
    plus_di = 100.0 * (_wilder_ema(plus_dm, length) / (atr + 1e-12))
    minus_di = 100.0 * (_wilder_ema(minus_dm, length) / (atr + 1e-12))
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    adx = _wilder_ema(dx, length)
    return adx


def _macd_manual(close: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """手動 MACD：DIF=快EMA-慢EMA，DEA=DIF 的 EMA，HIST=DIF-DEA。"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = dif - dea
    return dif, dea, hist


def _rsi_manual(close: pd.Series, length: int) -> pd.Series:
    """手動 RSI（Wilder）：相對強弱指標，衡量漲跌動能。"""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _stoch_manual(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
    smooth: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """手動隨機指標：%K 為收盤在區間位置，%D 為 %K 平滑。"""
    lowest_low = low.rolling(window, min_periods=window).min()
    highest_high = high.rolling(window, min_periods=window).max()
    k = 100.0 * (close - lowest_low) / (highest_high - lowest_low + 1e-12)
    d = k.rolling(smooth, min_periods=smooth).mean()
    return k, d


def _bollinger_manual(close: pd.Series, window: int, n_std: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    """手動布林帶：中軌為均線，上下軌為均線加減 n 倍標準差。"""
    mid = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    return upper, mid, lower


def _atr_manual(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """手動 ATR：真實波幅的平均，衡量波動程度。"""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return _wilder_ema(tr, length)


def _obv_manual(close: pd.Series, volume: pd.Series) -> pd.Series:
    """手動 OBV：依漲跌決定量能加減的累積指標。"""
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume.astype(float)).fillna(0.0).cumsum()


def _hv20_manual(close: pd.Series, window: int = 20) -> pd.Series:
    """20 日歷史波動率年化：對數報酬標準差 × sqrt(252)。"""
    lr = np.log(close / close.shift(1))
    return lr.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(252.0)


def _assign_indicators(
    out: pd.DataFrame,
    *,
    use_ta: bool,
) -> None:
    """
    將所有指標欄位寫入 out（原地修改）。

    參數:
        out: 已含 OHLCV 的 DataFrame。
        use_ta: True 時優先呼叫 ta，失敗則改為手動。
    """
    c = out["Close"].astype(float)
    h = out["High"].astype(float)
    lo = out["Low"].astype(float)
    v = out["Volume"].astype(float)

    p_rsi = int(TECH_PARAMS["rsi_period"])
    p_macd_f = int(TECH_PARAMS["macd_fast"])
    p_macd_s = int(TECH_PARAMS["macd_slow"])
    p_macd_sig = int(TECH_PARAMS["macd_signal"])
    p_bb = int(TECH_PARAMS["bb_period"])
    p_bb_std = float(TECH_PARAMS["bb_std"])
    p_atr = int(TECH_PARAMS["atr_period"])
    p_kd = int(TECH_PARAMS["kd_period"])
    p_vol_ma = int(TECH_PARAMS["volume_ma_period"])
    p_roc = int(TECH_PARAMS["roc_period"])
    p_adx = int(TECH_PARAMS["adx_period"])
    ma5_w, ma20_w, ma60_w, ma200_w = _ma_periods()

    done_ta = False
    if use_ta and _TA_AVAILABLE:
        try:
            # ── 移動平均：平滑價格以觀察趨勢方向 ──
            out["MA5"] = SMAIndicator(c, window=ma5_w).sma_indicator()
            out["MA20"] = SMAIndicator(c, window=ma20_w).sma_indicator()
            out["MA60"] = SMAIndicator(c, window=ma60_w).sma_indicator()
            out["MA200"] = SMAIndicator(c, window=ma200_w).sma_indicator()

            # ── MACD：快慢線差離與訊號線，用於判斷多空與交叉 ──
            macd = MACD(c, window_slow=p_macd_s, window_fast=p_macd_f, window_sign=p_macd_sig)
            out["MACD_DIF"] = macd.macd()
            out["MACD_DEA"] = macd.macd_signal()
            out["MACD_HIST"] = macd.macd_diff()

            # ── ADX：趨勢強度（不論漲跌） ──
            out["ADX"] = ADXIndicator(h, lo, c, window=p_adx).adx()

            # ── RSI：14 期相對強弱，辨識超買超賣 ──
            out["RSI"] = RSIIndicator(c, window=p_rsi).rsi()

            # ── KD：隨機指標，衡量收盤在區間的相對位置 ──
            st = StochasticOscillator(h, lo, c, window=p_kd, smooth_window=3)
            out["KD_K"] = st.stoch()
            out["KD_D"] = st.stoch_signal()

            # ── ROC：變動率，衡量一段期間漲跌幅 ──
            out["ROC"] = ROCIndicator(c, window=p_roc).roc()

            # ── 布林帶：均線加減標準差，觀察價格通道 ──
            bb = BollingerBands(c, window=p_bb, window_dev=p_bb_std)
            out["BB_UPPER"] = bb.bollinger_hband()
            out["BB_MIDDLE"] = bb.bollinger_mavg()
            out["BB_LOWER"] = bb.bollinger_lband()

            # ── ATR：真實波幅，衡量波動性 ──
            out["ATR"] = AverageTrueRange(h, lo, c, window=p_atr).average_true_range()

            # ── 20 日歷史波動率年化 ──
            out["HV20"] = _hv20_manual(c, window=20)

            # ── 成交量：均量、OBV、量比比率 ──
            out["VOL_MA20"] = v.rolling(p_vol_ma, min_periods=p_vol_ma).mean()
            out["OBV"] = OnBalanceVolumeIndicator(c, v).on_balance_volume()
            out["VOL_RATIO"] = v / (out["VOL_MA20"] + 1e-12)

            done_ta = True
        except Exception:
            done_ta = False

    if not done_ta:
        # ta 不可用或計算失敗時，改用手動實作（同上註解對應之指標意義）
        out["MA5"] = c.rolling(ma5_w, min_periods=ma5_w).mean()
        out["MA20"] = c.rolling(ma20_w, min_periods=ma20_w).mean()
        out["MA60"] = c.rolling(ma60_w, min_periods=ma60_w).mean()
        out["MA200"] = c.rolling(ma200_w, min_periods=ma200_w).mean()

        dif, dea, hist = _macd_manual(c, p_macd_f, p_macd_s, p_macd_sig)
        out["MACD_DIF"] = dif
        out["MACD_DEA"] = dea
        out["MACD_HIST"] = hist

        out["ADX"] = _adx_manual(h, lo, c, p_adx)

        out["RSI"] = _rsi_manual(c, p_rsi)

        kd_k, kd_d = _stoch_manual(h, lo, c, p_kd, smooth=3)
        out["KD_K"] = kd_k
        out["KD_D"] = kd_d

        out["ROC"] = (c - c.shift(p_roc)) / (c.shift(p_roc) + 1e-12) * 100.0

        bu, bm, bl = _bollinger_manual(c, p_bb, p_bb_std)
        out["BB_UPPER"] = bu
        out["BB_MIDDLE"] = bm
        out["BB_LOWER"] = bl

        out["ATR"] = _atr_manual(h, lo, c, p_atr)
        out["HV20"] = _hv20_manual(c, window=20)

        out["VOL_MA20"] = v.rolling(p_vol_ma, min_periods=p_vol_ma).mean()
        out["OBV"] = _obv_manual(c, v)
        out["VOL_RATIO"] = v / (out["VOL_MA20"] + 1e-12)


def calculate_indicators(price_history: pd.DataFrame, *, use_ta: bool = True) -> pd.DataFrame:
    """
    由原始 OHLCV 計算並附加所有技術指標欄位。

    參數:
        price_history: 需包含 Open、High、Low、Close、Volume；索引建議為日期。
        use_ta: True 時優先用 ta 套件；False 時強制走 pandas 手動實作
            （rolling / ewm 僅使用當日含以前資料，適合回測）。

    回傳:
        複本 DataFrame，含 MA、MACD、ADX、RSI、KD、ROC、布林、ATR、HV20、量能相關欄位。
    """
    required = {"Open", "High", "Low", "Close", "Volume"}
    miss = required - set(price_history.columns)
    if miss:
        raise ValueError(f"price_history 缺少欄位: {sorted(miss)}")

    out = price_history.copy()
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.sort_index()

    try:
        _assign_indicators(out, use_ta=use_ta)
    except Exception:
        _assign_indicators(out, use_ta=False)

    return out


def _last_scalar(s: pd.Series) -> float | None:
    """取序列最後一個有效數值；無則 None。"""
    x = s.dropna()
    if x.empty:
        return None
    v = float(x.iloc[-1])
    if np.isnan(v):
        return None
    return v


def _score_rsi(curr: float | None, prev: float | None) -> tuple[float, str]:
    """
    RSI 子分數與中文信號。

    評分規則對照：
    - RSI < 30：85 分（超賣反彈機會）
    - 30 <= RSI < 40：75 分
    - 40 <= RSI <= 60：70 分（健康區間）
    - 60 < RSI <= 70：60 分
    - RSI > 70：40 分（超買警示）
    """
    if curr is None:
        return 50.0, "中性"

    # 信號（含超買/超賣/中性/底部反彈；優先判斷極端區再判斷反彈）
    if curr > 70:
        sig = "超買"
    elif curr < 30:
        sig = "超賣"
    elif prev is not None and curr > prev and prev < 35:
        sig = "金叉反彈"
    else:
        sig = "中性"

    # 分數
    if curr < 30:
        sc = 85.0
    elif curr < 40:
        sc = 75.0
    elif curr <= 60:
        sc = 70.0
    elif curr <= 70:
        sc = 60.0
    else:
        sc = 40.0
    return sc, sig


def _score_macd(
    dif: float | None,
    dea: float | None,
    hist: float | None,
    dif_prev: float | None,
    dea_prev: float | None,
    hist_prev: float | None,
) -> tuple[float, str]:
    """
    MACD 子分數與中文信號。

    評分規則對照（依優先順序）：
    - 黃金交叉當天（DIF 剛上穿 DEA）：90 分
    - 持續多頭（DIF > DEA 且 MACD_HIST > 0）：75 分
    - 零軸之上且柱狀放大：70 分
    - 死亡交叉：30 分
    - 持續空頭：40 分
    """
    if dif is None or dea is None or hist is None:
        return 50.0, "持續多頭"

    gc = (
        dif_prev is not None
        and dea_prev is not None
        and dif_prev <= dea_prev
        and dif > dea
    )
    dc = (
        dif_prev is not None
        and dea_prev is not None
        and dif_prev >= dea_prev
        and dif < dea
    )
    bull = dif > dea and hist > 0
    bear = dif < dea and hist < 0
    hist_expand = hist_prev is not None and hist > hist_prev
    above_zero_hist = dif > 0 and hist_expand

    if gc:
        sc, sig = 90.0, "金叉"
    elif dc:
        sc, sig = 30.0, "死叉"
    elif bull:
        sc, sig = 75.0, "持續多頭"
    elif above_zero_hist:
        sc, sig = 70.0, "持續多頭"
    elif bear:
        sc, sig = 40.0, "持續空頭"
    else:
        # 邊緣狀態：趨勢剛反轉或在零軸附近震盪
        hist_rising = hist_prev is not None and hist > hist_prev

        if dif > 0 and hist_rising:
            # 零軸上方、柱狀開始轉強 → 多頭初期
            sc, sig = 65.0, "趨勢轉強"
        elif dif > 0 and not hist_rising:
            # 零軸上方但動能轉弱 → 多頭末段
            sc, sig = 55.0, "動能減弱"
        elif dif < 0 and hist_rising:
            # 零軸下方但柱狀縮短 → 跌勢趨緩
            sc, sig = 50.0, "跌勢趨緩"
        else:
            # 零軸下方且柱狀仍在擴大 → 弱勢
            sc, sig = 45.0, "弱勢整理"
    return sc, sig


def _score_bollinger(
    close: float | None,
    up: float | None,
    mid: float | None,
    low: float | None,
) -> tuple[float, str]:
    """
    布林帶子分數與信號。

    評分規則對照（含觸軌、過熱、站穩中軌等）：
    - 股價觸及下軌（超賣）：85 分
    - 股價突破上軌：50 分（過熱）
    - 股價站上中軌且未觸上軌：70 分
    - 股價在中軌下方：45 分
    """
    if close is None or up is None or mid is None or low is None:
        return 50.0, "中軌之下"

    # 觸及或跌破下軌（超賣區）— 對應規則「觸及下軌」給分，信號欄用表格用語
    if close <= low:
        return 85.0, "跌破下軌"
    if close > up:
        return 50.0, "突破上軌"
    if close > mid and close < up:
        return 70.0, "中軌之上"
    return 45.0, "中軌之下"


def _score_ma_trend(
    close: float | None,
    ma5: float | None,
    ma20: float | None,
    ma60: float | None,
    ma200: float | None,
) -> tuple[float, str]:
    """
    均線排列與股價位置評分。

    評分規則對照：
    - 多頭排列（MA5>MA20>MA60>MA200）：90 分
    - 部分多頭（MA20>MA60>MA200）：75 分
    - 股價站上 MA20 與 MA60：70 分
    - 股價跌破 MA20 但在 MA60 上：50 分
    - 空頭排列：30 分
    """
    if close is None or ma20 is None or ma60 is None:
        return 50.0, "整理"

    full_bull = (
        ma5 is not None
        and ma200 is not None
        and ma5 > ma20 > ma60 > ma200
    )
    # 部分多頭：MA20>MA60>MA200（未要求 MA5 亦多頭排列）
    partial_bull = ma200 is not None and ma20 > ma60 > ma200
    full_bear = (
        ma5 is not None
        and ma200 is not None
        and ma5 < ma20 < ma60 < ma200
    )
    above_20_60 = close > ma20 and close > ma60
    below_20_above_60 = close < ma20 and close > ma60

    if full_bull:
        return 90.0, "多頭排列"
    # 部分多頭：MA20>MA60>MA200 但未達完整多頭排列，信號歸類為「整理」
    if partial_bull:
        return 75.0, "整理"
    if full_bear:
        return 30.0, "空頭排列"
    if above_20_60:
        return 70.0, "整理"
    if below_20_above_60:
        return 50.0, "整理"
    return 55.0, "整理"


def _score_volume(vol_ratio: float | None, open_: float | None, close: float | None) -> tuple[float, str]:
    """
    成交量子分數與信號。

    評分規則對照：
    - 放量上漲（VOL_RATIO > 1.5 且當日紅 K）：85 分
    - 正常量能（0.8-1.5 倍）：65 分
    - 縮量（< 0.5 倍）：50 分
    - 異常放量下跌（>2 倍且黑 K）：35 分
    """
    if vol_ratio is None or open_ is None or close is None:
        return 50.0, "正常"

    red = close > open_
    black = close < open_

    # 異常放量下跌（>2 倍且黑 K）
    if vol_ratio > 2.0 and black:
        return 35.0, "異常放量"
    # 放量上漲（> 1.5 且紅 K）
    if vol_ratio > 1.5 and red:
        return 85.0, "放量上漲"
    # 縮量（< 0.5 倍）
    if vol_ratio < 0.5:
        return 50.0, "縮量"
    # 正常量能（0.8 - 1.5 倍）
    if 0.8 <= vol_ratio <= 1.5:
        return 65.0, "正常"
    return 60.0, "正常"


def calculate_tech_score(df_with_indicators: pd.DataFrame) -> dict[str, Any]:
    """
    依最後一個交易日資料計算加權技術總分與子分數、信號、關鍵數值。

    參數:
        df_with_indicators: 已由 calculate_indicators 附加指標欄位之 DataFrame。

    回傳:
        包含 total_score、sub_scores、signals、current_values 的 dict（分數區間 0-100）。
    """
    df = df_with_indicators.dropna(how="all")
    if df.empty:
        raise ValueError("df_with_indicators 為空")

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else None

    def gv(col: str) -> float | None:
        if col not in df.columns:
            return None
        return _last_scalar(df[col])

    rsi = gv("RSI")
    rsi_prev = float(prev["RSI"]) if prev is not None and "RSI" in prev.index and pd.notna(prev["RSI"]) else None

    dif = gv("MACD_DIF")
    dea = gv("MACD_DEA")
    hist = gv("MACD_HIST")
    dif_prev = float(prev["MACD_DIF"]) if prev is not None and pd.notna(prev.get("MACD_DIF")) else None
    dea_prev = float(prev["MACD_DEA"]) if prev is not None and pd.notna(prev.get("MACD_DEA")) else None
    hist_prev = float(prev["MACD_HIST"]) if prev is not None and pd.notna(prev.get("MACD_HIST")) else None

    close = gv("Close")
    op = gv("Open")
    ma20 = gv("MA20")
    ma5 = gv("MA5")
    ma60 = gv("MA60")
    ma200 = gv("MA200")

    rsi_sc, rsi_sig = _score_rsi(rsi, rsi_prev)
    macd_sc, macd_sig = _score_macd(dif, dea, hist, dif_prev, dea_prev, hist_prev)
    boll_sc, boll_sig = _score_bollinger(
        close,
        gv("BB_UPPER"),
        gv("BB_MIDDLE"),
        gv("BB_LOWER"),
    )
    ma_sc, ma_sig = _score_ma_trend(close, ma5, ma20, ma60, ma200)
    vol_sc, vol_sig = _score_volume(gv("VOL_RATIO"), op, close)

    sub_scores = {
        "rsi": round(rsi_sc, 2),
        "macd": round(macd_sc, 2),
        "bollinger": round(boll_sc, 2),
        "ma_trend": round(ma_sc, 2),
        "volume": round(vol_sc, 2),
    }

    w = TECH_SCORE_WEIGHTS
    total = (
        sub_scores["rsi"] * w["rsi"]
        + sub_scores["macd"] * w["macd"]
        + sub_scores["bollinger"] * w["bollinger"]
        + sub_scores["ma_trend"] * w["ma_trend"]
        + sub_scores["volume"] * w["volume"]
    )

    price_vs_ma20 = None
    if close is not None and ma20 is not None and ma20 != 0:
        price_vs_ma20 = (close / ma20 - 1.0) * 100.0

    current_values: dict[str, Any] = {
        "rsi": rsi,
        "macd_dif": dif,
        "macd_dea": dea,
        "macd_hist": hist,
        "price_vs_ma20": price_vs_ma20,
        "close": close,
        "vol_ratio": gv("VOL_RATIO"),
        "adx": gv("ADX"),
        "kd_k": gv("KD_K"),
        "kd_d": gv("KD_D"),
        "ma5": ma5,
        "ma20": ma20,
        "ma60": ma60,
        "ma200": ma200,
        "bb_upper": gv("BB_UPPER"),
        "bb_middle": gv("BB_MIDDLE"),
        "bb_lower": gv("BB_LOWER"),
        "roc": gv("ROC"),
        "atr": gv("ATR"),
        "hv20": gv("HV20"),
    }

    return {
        "total_score": round(float(total), 2),
        "sub_scores": sub_scores,
        "signals": {
            "rsi": rsi_sig,
            "macd": macd_sig,
            "bollinger": boll_sig,
            "ma_trend": ma_sig,
            "volume": vol_sig,
        },
        "current_values": current_values,
    }


if __name__ == "__main__":
    _SRC_DIR = Path(__file__).resolve().parent
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))
    from data_collector import fetch_stock_data

    raw = fetch_stock_data("NVDA", force_refresh=False)
    if not raw:
        print("NVDA 資料抓取失敗，請檢查網路或稍後再試。")
    else:
        ph = raw["price_history"]
        df_ind = calculate_indicators(ph)
        score = calculate_tech_score(df_ind)
        print("=== NVDA 技術評分摘要 ===")
        print("total_score:", score["total_score"])
        print("sub_scores:", score["sub_scores"])
        print("signals:", score["signals"])
        print("current_values:")
        for k, v in score["current_values"].items():
            print(f"  {k}: {v}")
