# -*- coding: utf-8 -*-
"""
股價預測模組：使用 Prophet 預測未來 30 天價格，失敗時回退線性外推。

執行方式（於專案根目錄）：
    python src/predictor.py
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# 從專案根目錄載入 config（直接執行本檔時 sys.path 預設為 src/）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import PREDICTION_PARAMS  # noqa: E402

try:
    from prophet import Prophet

    _PROPHET_AVAILABLE = True
except Exception:  # noqa: BLE001
    Prophet = None  # type: ignore[assignment]
    _PROPHET_AVAILABLE = False

_LOG = logging.getLogger("predictor")
if not _LOG.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@contextmanager
def _suppress_stdout_stderr():
    """暫時抑制 stdout/stderr（降低 Prophet/cmdstanpy 訓練輸出噪音）。"""
    devnull = open(os.devnull, "w", encoding="utf-8")
    old_out = sys.stdout
    old_err = sys.stderr
    try:
        sys.stdout = devnull
        sys.stderr = devnull
        yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        devnull.close()


def _prediction_confidence(pred: float, low: float, high: float) -> tuple[str, float]:
    """
    依區間寬度回傳信心度。

    區間寬度 = (predicted_high - predicted_low) / predicted_30d
    - < 15%: high
    - 15% ~ 30%: medium
    - > 30%: low
    """
    if pred <= 0:
        return "low", 1.0
    width = max(0.0, (high - low) / pred)
    if width < 0.15:
        return "high", width
    if width <= 0.30:
        return "medium", width
    return "low", width


def _prediction_score(expected_return_pct: float, confidence: str) -> int:
    """
    預測吸引力分數。

    - 預期報酬 > +10% 且 high 可信度：90
    - 預期報酬 > +5%：75
    - 預期報酬 0% ~ +5%：60
    - 預期報酬 -5% ~ 0%：45
    - 預期報酬 < -5%：25
    """
    if expected_return_pct > 10 and confidence == "high":
        return 90
    if expected_return_pct > 5:
        return 75
    if expected_return_pct >= 0:
        return 60
    if expected_return_pct >= -5:
        return 45
    return 25


def _prepare_close_series(price_history: pd.DataFrame) -> pd.Series:
    """取最近 history_days 的 Close，轉成乾淨時間序列。"""
    if "Close" not in price_history.columns:
        raise ValueError("price_history 缺少 Close 欄位")
    history_days = int(PREDICTION_PARAMS.get("history_days", 365))
    s = pd.to_numeric(price_history["Close"], errors="coerce").dropna()
    if s.empty:
        raise ValueError("Close 資料為空")
    s = s.sort_index().tail(history_days)
    return s


def _prophet_predict(close: pd.Series) -> tuple[pd.DataFrame, str]:
    """使用 Prophet 產生未來 forecast_days 個交易日預測。"""
    if not _PROPHET_AVAILABLE:
        raise RuntimeError("Prophet 未安裝或匯入失敗")
    if len(close) < 80:
        raise ValueError("Prophet 訓練資料不足（至少約 80 筆）")

    interval = float(PREDICTION_PARAMS.get("confidence_interval", 0.95))
    cp_prior = float(PREDICTION_PARAMS.get("prophet_changepoint_prior", 0.05))
    forecast_days = int(PREDICTION_PARAMS.get("forecast_days", 30))

    # Prophet 格式：ds / y，且 ds 去除時區
    ds = pd.to_datetime(close.index)
    if isinstance(ds, pd.DatetimeIndex) and ds.tz is not None:
        ds = ds.tz_localize(None)
    train = pd.DataFrame({"ds": ds, "y": close.values})

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    model = Prophet(
        changepoint_prior_scale=cp_prior,
        interval_width=interval,
        weekly_seasonality=False,
        daily_seasonality=False,
        yearly_seasonality=True,
    )
    model.add_country_holidays(country_name="US")

    with _suppress_stdout_stderr():
        model.fit(train)
        # 使用日曆日先往後推，再濾掉週末，取得 30 個交易日
        future = model.make_future_dataframe(periods=forecast_days * 2, freq="D", include_history=False)
        future = future[future["ds"].dt.weekday < 5].head(forecast_days)
        fc = model.predict(future)

    out = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    return out, "prophet"


def _linear_predict(close: pd.Series) -> tuple[pd.DataFrame, str]:
    """線性外推備援法：用最近 60 天做回歸並外推 30 天。"""
    forecast_days = int(PREDICTION_PARAMS.get("forecast_days", 30))
    sample = close.tail(60)
    if len(sample) < 20:
        raise ValueError("線性回歸資料不足（至少 20 筆）")

    y = sample.values.astype(float)
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    future_x = np.arange(len(y), len(y) + forecast_days, dtype=float)
    yhat = slope * future_x + intercept

    sigma = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
    ci = 1.96 * sigma
    lower = yhat - ci
    upper = yhat + ci

    last_dt = pd.to_datetime(sample.index[-1])
    future_dates = pd.bdate_range(start=last_dt + pd.Timedelta(days=1), periods=forecast_days)
    out = pd.DataFrame(
        {
            "ds": future_dates,
            "yhat": yhat,
            "yhat_lower": lower,
            "yhat_upper": upper,
        }
    )
    return out, "linear"


def predict_price(ticker: str, price_history: pd.DataFrame) -> dict[str, Any]:
    """
    預測單一股票未來 30 天價格。

    流程：
    1) 優先使用 Prophet（含 US 假日，週末不預測）
    2) Prophet 失敗則回退線性外推

    參數:
        ticker: 股票代號。
        price_history: OHLCV DataFrame，index 為日期，需含 Close 欄位。

    回傳:
        含當前價、30 日預測價、信賴區間、預測序列、模型來源、信心度與評分。
    """
    sym = ticker.strip().upper()
    close = _prepare_close_series(price_history)
    current_price = float(close.iloc[-1])

    model_used = "linear"
    try:
        pred_df, model_used = _prophet_predict(close)
    except Exception as e:
        _LOG.warning("%s Prophet 失敗，改用線性法：%s", sym, e)
        pred_df, model_used = _linear_predict(close)

    if pred_df.empty:
        raise ValueError(f"{sym} 無可用預測結果")

    pred_df = pred_df.sort_values("ds").reset_index(drop=True)
    last = pred_df.iloc[-1]
    predicted_30d = float(last["yhat"])
    predicted_low = float(last["yhat_lower"])
    predicted_high = float(last["yhat_upper"])

    expected_return_pct = (predicted_30d - current_price) / current_price * 100.0
    confidence, width = _prediction_confidence(predicted_30d, predicted_low, predicted_high)
    score = _prediction_score(expected_return_pct, confidence)

    return {
        "ticker": sym,
        "current_price": round(current_price, 4),
        "predicted_30d": round(predicted_30d, 4),
        "predicted_high": round(predicted_high, 4),
        "predicted_low": round(predicted_low, 4),
        "expected_return_pct": round(expected_return_pct, 2),
        "prediction_dates": [d.strftime("%Y-%m-%d") for d in pd.to_datetime(pred_df["ds"])],
        "prediction_prices": [round(float(x), 4) for x in pred_df["yhat"].tolist()],
        "prediction_upper": [round(float(x), 4) for x in pred_df["yhat_upper"].tolist()],
        "prediction_lower": [round(float(x), 4) for x in pred_df["yhat_lower"].tolist()],
        "model_used": model_used,
        "confidence": confidence,
        "score": int(score),
        "interval_width": round(float(width), 4),
    }


def predict_batch(all_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    批量預測並依 expected_return_pct 由高到低排序。

    參數:
        all_data: collect_all_stocks() 回傳字典，value 需含 price_history。

    回傳:
        預測結果列表；個別失敗會略過並記錄 log。
    """
    rows: list[dict[str, Any]] = []
    for key, payload in all_data.items():
        if not isinstance(payload, dict):
            continue
        sym = str(payload.get("ticker") or key).upper()
        ph = payload.get("price_history")
        if not isinstance(ph, pd.DataFrame):
            continue
        try:
            rows.append(predict_price(sym, ph))
        except Exception as e:
            _LOG.error("%s 預測失敗：%s", sym, e)
    rows.sort(key=lambda x: x.get("expected_return_pct", float("-inf")), reverse=True)
    return rows


def _fmt(x: float) -> str:
    """價格格式化。"""
    return f"{x:.2f}"


def _print_table(rows: list[dict[str, Any]]) -> None:
    """格式化輸出預測摘要。"""
    headers = ("ticker", "current", "pred_30d", "ret%", "model", "conf", "score")
    widths = (8, 10, 10, 8, 10, 8, 6)
    line = " ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))
    print(line)
    print("-" * len(line))
    for r in rows:
        print(
            " ".join(
                [
                    r["ticker"].ljust(widths[0]),
                    _fmt(r["current_price"]).ljust(widths[1]),
                    _fmt(r["predicted_30d"]).ljust(widths[2]),
                    f"{r['expected_return_pct']:.2f}%".ljust(widths[3]),
                    r["model_used"].ljust(widths[4]),
                    r["confidence"].ljust(widths[5]),
                    str(r["score"]).ljust(widths[6]),
                ]
            )
        )


if __name__ == "__main__":
    _SRC_DIR = Path(__file__).resolve().parent
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))

    from data_collector import fetch_stock_data

    tickers = ["NVDA", "TSM", "JPM"]
    results: list[dict[str, Any]] = []
    for t in tickers:
        raw = fetch_stock_data(t, force_refresh=False)
        if not raw or not isinstance(raw.get("price_history"), pd.DataFrame):
            print(f"[{t}] 無可用歷史資料，跳過。")
            continue
        try:
            results.append(predict_price(t, raw["price_history"]))
        except Exception as e:
            print(f"[{t}] 預測失敗：{e}")

    results.sort(key=lambda x: x["expected_return_pct"], reverse=True)
    print("\n=== 未來 30 天股價預測 ===")
    _print_table(results)
    print("\n本預測僅供參考，不構成投資建議")
