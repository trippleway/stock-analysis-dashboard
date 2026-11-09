# -*- coding: utf-8 -*-
"""
Technical indicators and the combined technical score.

Depends on config.TECH_PARAMS and TECH_SCORE_WEIGHTS. Price history should come from data_collector.fetch_stock_data.

Run from the project root:
    python src/indicators.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Load config from the project root (sys.path defaults to src/ when this file is run directly).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import TECH_PARAMS, TECH_SCORE_WEIGHTS  # noqa: E402

# Try to import ta. If that fails, the manual calculation is used throughout.
try:
    from ta.momentum import ROCIndicator, RSIIndicator, StochasticOscillator
    from ta.trend import ADXIndicator, MACD, SMAIndicator
    from ta.volatility import AverageTrueRange, BollingerBands
    from ta.volume import OnBalanceVolumeIndicator

    _TA_AVAILABLE = True
except Exception:  # noqa: BLE001 — any import error falls back to the manual implementation
    _TA_AVAILABLE = False


def _ma_periods() -> tuple[int, int, int, int]:
    """Read the MA5/MA20/MA60/MA200 windows from TECH_PARAMS."""
    p = TECH_PARAMS.get("ma_periods", [5, 20, 60, 200])
    if len(p) != 4:
        return 5, 20, 60, 200
    return int(p[0]), int(p[1]), int(p[2]), int(p[3])


def _wilder_ema(s: pd.Series, length: int) -> pd.Series:
    """Wilder smoothing (the same ewm(alpha=1/length) that ta commonly uses)."""
    return s.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()


def _adx_manual(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """
    Manual ADX (average directional index): trend strength, independent of direction.
    Uses a Wilder-smoothed approximation of +DI, -DI, and DX.
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
    """Manual MACD: DIF is the fast EMA minus the slow EMA, DEA is the EMA of DIF, and HIST is DIF minus DEA."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = dif - dea
    return dif, dea, hist


def _rsi_manual(close: pd.Series, length: int) -> pd.Series:
    """Manual RSI (Wilder): relative strength, a measure of up and down momentum."""
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
    """Manual stochastic: %K is where the close sits in the range, and %D is the smoothed %K."""
    lowest_low = low.rolling(window, min_periods=window).min()
    highest_high = high.rolling(window, min_periods=window).max()
    k = 100.0 * (close - lowest_low) / (highest_high - lowest_low + 1e-12)
    d = k.rolling(smooth, min_periods=smooth).mean()
    return k, d


def _bollinger_manual(close: pd.Series, window: int, n_std: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Manual Bollinger bands: the middle band is the moving average, and the outer bands are that average plus or minus n standard deviations."""
    mid = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    return upper, mid, lower


def _atr_manual(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """Manual ATR: the average true range, a measure of volatility."""
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
    """Manual OBV: a cumulative volume indicator that adds or subtracts volume based on the price move."""
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume.astype(float)).fillna(0.0).cumsum()


def _hv20_manual(close: pd.Series, window: int = 20) -> pd.Series:
    """20-day historical volatility, annualized: the standard deviation of log returns times sqrt(252)."""
    lr = np.log(close / close.shift(1))
    return lr.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(252.0)


def _assign_indicators(
    out: pd.DataFrame,
    *,
    use_ta: bool,
) -> None:
    """
    Write every indicator column into out (modified in place).

    Args:
        out: A DataFrame that already has OHLCV.
        use_ta: When True, call ta first and fall back to the manual calculation on failure.
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
            # Moving averages: smooth the price to see the trend direction.
            out["MA5"] = SMAIndicator(c, window=ma5_w).sma_indicator()
            out["MA20"] = SMAIndicator(c, window=ma20_w).sma_indicator()
            out["MA60"] = SMAIndicator(c, window=ma60_w).sma_indicator()
            out["MA200"] = SMAIndicator(c, window=ma200_w).sma_indicator()

            # MACD: the fast/slow difference and the signal line, used for trend and crosses.
            macd = MACD(c, window_slow=p_macd_s, window_fast=p_macd_f, window_sign=p_macd_sig)
            out["MACD_DIF"] = macd.macd()
            out["MACD_DEA"] = macd.macd_signal()
            out["MACD_HIST"] = macd.macd_diff()

            # ADX: trend strength, regardless of up or down.
            out["ADX"] = ADXIndicator(h, lo, c, window=p_adx).adx()

            # RSI: 14-period relative strength, used to spot overbought and oversold.
            out["RSI"] = RSIIndicator(c, window=p_rsi).rsi()

            # Stochastic: where the close sits inside the recent range.
            st = StochasticOscillator(h, lo, c, window=p_kd, smooth_window=3)
            out["KD_K"] = st.stoch()
            out["KD_D"] = st.stoch_signal()

            # ROC: rate of change over a window.
            out["ROC"] = ROCIndicator(c, window=p_roc).roc()

            # Bollinger bands: a moving average plus or minus standard deviations, a price channel.
            bb = BollingerBands(c, window=p_bb, window_dev=p_bb_std)
            out["BB_UPPER"] = bb.bollinger_hband()
            out["BB_MIDDLE"] = bb.bollinger_mavg()
            out["BB_LOWER"] = bb.bollinger_lband()

            # ATR: true range, a measure of volatility.
            out["ATR"] = AverageTrueRange(h, lo, c, window=p_atr).average_true_range()

            # 20-day historical volatility, annualized.
            out["HV20"] = _hv20_manual(c, window=20)

            # Volume: average volume, OBV, and the volume ratio.
            out["VOL_MA20"] = v.rolling(p_vol_ma, min_periods=p_vol_ma).mean()
            out["OBV"] = OnBalanceVolumeIndicator(c, v).on_balance_volume()
            out["VOL_RATIO"] = v / (out["VOL_MA20"] + 1e-12)

            done_ta = True
        except Exception:
            done_ta = False

    if not done_ta:
        # When ta is unavailable or the calculation fails, use the manual implementation (same indicators as the comments above).
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
    Compute every technical indicator from raw OHLCV and attach the columns.

    Args:
        price_history: Must include Open, High, Low, Close, and Volume. A date index is recommended.
        use_ta: When True, prefer the ta package. When False, force the pandas implementation.
            Rolling and ewm use only data through the current day, which is what a backtest needs.

    Returns:
        A copied DataFrame with MA, MACD, ADX, RSI, stochastic, ROC, Bollinger, ATR, HV20, and volume columns.
    """
    required = {"Open", "High", "Low", "Close", "Volume"}
    miss = required - set(price_history.columns)
    if miss:
        raise ValueError(f"price_history is missing columns: {sorted(miss)}")

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
    """The last valid number in the series, or None."""
    x = s.dropna()
    if x.empty:
        return None
    v = float(x.iloc[-1])
    if np.isnan(v):
        return None
    return v


def _score_rsi(curr: float | None, prev: float | None) -> tuple[float, str]:
    """
    RSI subscore and signal.

    Score rules:
    - RSI < 30: 85 (oversold bounce setup)
    - 30 <= RSI < 40: 75
    - 40 <= RSI <= 60: 70 (healthy range)
    - 60 < RSI <= 70: 60
    - RSI > 70: 40 (overbought warning). The overbought signal token is left as-is because daily_picker compares it.
    """
    if curr is None:
        return 50.0, "Neutral"

    # Signals cover overbought, oversold, neutral, and a bounce. Extremes are checked before the bounce.
    # The overbought signal token is compared in daily_picker and entry_price, so it stays.
    if curr > 70:
        sig = "Overbought"
    elif curr < 30:
        sig = "Oversold"
    elif prev is not None and curr > prev and prev < 35:
        sig = "Oversold bounce"
    else:
        sig = "Neutral"

    # Score
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
    MACD subscore and signal.

    Score rules, in priority order:
    - Golden cross that day (DIF just crossed above DEA): 90. The golden-cross signal token is left as-is because daily_picker compares it.
    - Still bullish (DIF > DEA and MACD_HIST > 0): 75
    - Above zero and the histogram is expanding: 70
    - Death cross: 30. The death-cross signal token is left as-is because daily_picker compares it.
    - Still bearish: 40
    """
    if dif is None or dea is None or hist is None:
        return 50.0, "Still bullish"

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
        sc, sig = 90.0, "Golden Cross"
    elif dc:
        sc, sig = 30.0, "Death Cross"
    elif bull:
        sc, sig = 75.0, "Still bullish"
    elif above_zero_hist:
        sc, sig = 70.0, "Still bullish"
    elif bear:
        sc, sig = 40.0, "Still bearish"
    else:
        # Edge cases: the trend just turned, or price is chopping near zero.
        hist_rising = hist_prev is not None and hist > hist_prev

        if dif > 0 and hist_rising:
            # Above zero and the histogram is starting to strengthen: early bullish.
            sc, sig = 65.0, "Trend strengthening"
        elif dif > 0 and not hist_rising:
            # Above zero but momentum is fading: late bullish.
            sc, sig = 55.0, "Momentum fading"
        elif dif < 0 and hist_rising:
            # Below zero but the histogram is shrinking: the decline is slowing.
            sc, sig = 50.0, "Decline slowing"
        else:
            # Below zero and the histogram is still expanding: weak.
            sc, sig = 45.0, "Weak consolidation"
    return sc, sig


def _score_bollinger(
    close: float | None,
    up: float | None,
    mid: float | None,
    low: float | None,
) -> tuple[float, str]:
    """
    Bollinger subscore and signal.

    Score rules, including a band touch, overheating, and holding the middle band:
    - Price touches the lower band (oversold): 85
    - Price breaks the upper band: 50 (overheated)
    - Price is above the middle band and has not touched the upper band: 70
    - Price is below the middle band: 45
    """
    if close is None or up is None or mid is None or low is None:
        return 50.0, "Below the middle band"

    # At or through the lower band (oversold). The score matches the "touched the lower band" rule.
    if close <= low:
        return 85.0, "Broke the lower band"
    if close > up:
        return 50.0, "Broke the upper band"
    if close > mid and close < up:
        return 70.0, "Above the middle band"
    return 45.0, "Below the middle band"


def _score_ma_trend(
    close: float | None,
    ma5: float | None,
    ma20: float | None,
    ma60: float | None,
    ma200: float | None,
) -> tuple[float, str]:
    """
    Score the moving-average stack and where price sits.

    Score rules:
    - Bullish stack (MA5>MA20>MA60>MA200): 90. The bullish-stack signal token is left as-is because daily_picker compares it.
    - Partial bullish stack (MA20>MA60>MA200): 75
    - Price above MA20 and MA60: 70
    - Price below MA20 but above MA60: 50
    - Bearish stack: 30
    """
    if close is None or ma20 is None or ma60 is None:
        return 50.0, "Consolidating"

    full_bull = (
        ma5 is not None
        and ma200 is not None
        and ma5 > ma20 > ma60 > ma200
    )
    # Partial bullish stack: MA20>MA60>MA200, without also requiring MA5 to complete the bullish stack.
    partial_bull = ma200 is not None and ma20 > ma60 > ma200
    full_bear = (
        ma5 is not None
        and ma200 is not None
        and ma5 < ma20 < ma60 < ma200
    )
    above_20_60 = close > ma20 and close > ma60
    below_20_above_60 = close < ma20 and close > ma60

    if full_bull:
        return 90.0, "Bullish Alignment"
    # Partial bullish stack: MA20>MA60>MA200 but not a full bullish stack. The signal is "Consolidating".
    if partial_bull:
        return 75.0, "Consolidating"
    if full_bear:
        return 30.0, "Bearish stack"
    if above_20_60:
        return 70.0, "Consolidating"
    if below_20_above_60:
        return 50.0, "Consolidating"
    return 55.0, "Consolidating"


def _score_volume(vol_ratio: float | None, open_: float | None, close: float | None) -> tuple[float, str]:
    """
    Volume subscore and signal.

    Score rules:
    - Up on heavy volume (VOL_RATIO > 1.5 and a green candle): 85
    - Normal volume (0.8-1.5 times): 65
    - Light volume (< 0.5 times): 50
    - Down on abnormal volume (>2 times and a red candle): 35
    """
    if vol_ratio is None or open_ is None or close is None:
        return 50.0, "Normal"

    red = close > open_
    black = close < open_

    # Down on abnormal volume (>2 times and a red candle).
    if vol_ratio > 2.0 and black:
        return 35.0, "Abnormal volume"
    # Up on heavy volume (> 1.5 and a green candle).
    if vol_ratio > 1.5 and red:
        return 85.0, "Up on heavy volume"
    # Light volume (< 0.5 times).
    if vol_ratio < 0.5:
        return 50.0, "Light volume"
    # Normal volume (0.8 to 1.5 times).
    if 0.8 <= vol_ratio <= 1.5:
        return 65.0, "Normal"
    return 60.0, "Normal"


def calculate_tech_score(df_with_indicators: pd.DataFrame) -> dict[str, Any]:
    """
    Compute the weighted technical total, subscores, signals, and key values from the last trading day.

    Args:
        df_with_indicators: A DataFrame that already has indicator columns from calculate_indicators.

    Returns:
        A dict with total_score, sub_scores, signals, and current_values. Scores are on 0-100.
    """
    df = df_with_indicators.dropna(how="all")
    if df.empty:
        raise ValueError("df_with_indicators is empty")

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
        print("Failed to fetch NVDA data. Check the network or try again later.")
    else:
        ph = raw["price_history"]
        df_ind = calculate_indicators(ph)
        score = calculate_tech_score(df_ind)
        print("=== NVDA technical score ===")
        print("total_score:", score["total_score"])
        print("sub_scores:", score["sub_scores"])
        print("signals:", score["signals"])
        print("current_values:")
        for k, v in score["current_values"].items():
            print(f"  {k}: {v}")
