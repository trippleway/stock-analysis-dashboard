# -*- coding: utf-8 -*-
"""
理想進場價區間：依 daily_picker.analyze_single_stock 整合結果計算。

可選在 stock_data 內附帶 raw 的 price_history（DataFrame），以計算近 30 日低點支撐。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_log = logging.getLogger("entry_price")


def get_status_emoji(status: str) -> str:
    return {
        "buy_now": "🟢",
        "watch": "🟡",
        "wait": "🟠",
        "avoid": "🔴",
    }.get(status, "⚪")


def get_status_label_cn(status: str) -> str:
    return {
        "buy_now": "立即可進場",
        "watch": "接近進場區",
        "wait": "等待回檔",
        "avoid": "不宜追高",
    }.get(status, "中性")


def _to_float(x: Any, default: float | None = None) -> float | None:
    try:
        if x is None:
            return default
        v = float(x)
        if v != v:  # nan
            return default
        return v
    except (TypeError, ValueError):
        return default


def _last_n_low_min(df: pd.DataFrame, n: int = 30) -> float | None:
    """過去 n 根 K 棒最低價（Low）。"""
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None
        col = "Low" if "Low" in df.columns else ("low" if "low" in df.columns else None)
        if col is None:
            return None
        tail = df.tail(n)
        lo = pd.to_numeric(tail[col], errors="coerce").dropna()
        if lo.empty:
            return None
        return float(lo.min())
    except Exception:  # noqa: BLE001
        return None


def _retracement_pct(composite_score: float) -> float:
    if composite_score >= 80:
        return 0.03
    if composite_score >= 60:
        return 0.05
    if composite_score >= 40:
        return 0.08
    return 0.12


def _ordered_entry_triple(
    conservative: float,
    standard: float,
    aggressive: float,
    current: float,
) -> tuple[float, float, float, bool]:
    """
    確保 conservative ≤ standard ≤ aggressive ≤ current。
    若不成立，改為 current × (0.92, 0.96, 0.98)，並標記 used_fallback。
    """
    c, s, a, cur = conservative, standard, aggressive, current
    ok = c <= s <= a <= cur and cur > 0
    if ok:
        return c, s, a, False
    cur = max(cur, 1e-9)
    return cur * 0.92, cur * 0.96, cur * 0.98, True


def calculate_entry_prices(stock_data: dict[str, Any]) -> dict[str, Any]:
    """
    依 analyze_single_stock 結果計算進場價與狀態。

    可選：stock_data['price_history'] 為 pd.DataFrame（與 fetch_stock_data 相同），
    用於「過去 30 日最低點」支撐；若缺省則僅以布林下軌與 MA20 估算標準進場價。
    """
    err_reasoning = "資料不足，無法完整計算進場價，請確認技術與估值欄位。"
    empty = {
        "ticker": str(stock_data.get("ticker") or "?"),
        "current_price": 0.0,
        "entry_prices": {"conservative": 0.0, "standard": 0.0, "aggressive": 0.0},
        "current_status": "avoid",
        "current_zone": "far_above",
        "distance": {
            "to_fair_value_pct": 0.0,
            "to_conservative_pct": 0.0,
            "to_standard_pct": 0.0,
            "to_aggressive_pct": 0.0,
        },
        "reasoning": err_reasoning,
        "batch_plan": [],
        "risk_reward_ratio": 0.0,
    }

    try:
        if not isinstance(stock_data, dict):
            return empty

        ticker = str(stock_data.get("ticker") or "?").strip().upper()
        current = _to_float(stock_data.get("current_price"))
        if current is None or current <= 0:
            return {**empty, "ticker": ticker, "reasoning": err_reasoning}

        valuation = stock_data.get("valuation") if isinstance(stock_data.get("valuation"), dict) else {}
        technical = stock_data.get("technical") if isinstance(stock_data.get("technical"), dict) else {}
        prediction = stock_data.get("prediction") if isinstance(stock_data.get("prediction"), dict) else {}
        fundamental = stock_data.get("fundamental") if isinstance(stock_data.get("fundamental"), dict) else {}

        fair_raw = _to_float(valuation.get("fair_value"))
        fair_for_pct = fair_raw if fair_raw is not None and fair_raw > 0 else current

        composite_score = float(_to_float(stock_data.get("composite_score"), 50.0) or 50.0)
        valuation_signal = str(valuation.get("valuation_signal") or "").strip().lower()
        rsi_signal = str((technical.get("signals") or {}).get("rsi") or "").strip()

        # (A) 保守
        if fair_raw is not None and fair_raw > 0:
            conservative = fair_raw * 0.75
        else:
            conservative = current * 0.85

        # (B) 標準：三者取最大
        cv = technical.get("current_values") if isinstance(technical.get("current_values"), dict) else {}
        bb_lower = _to_float(cv.get("bb_lower"))
        ma20 = _to_float(cv.get("ma20"))
        candidates: list[float] = []
        if bb_lower is not None and bb_lower > 0:
            candidates.append(bb_lower)
        if ma20 is not None and ma20 > 0:
            candidates.append(ma20 * 0.98)
        ph = stock_data.get("price_history")
        low30 = _last_n_low_min(ph, 30) if isinstance(ph, pd.DataFrame) else None
        if low30 is not None and low30 > 0:
            candidates.append(low30 * 1.02)
        if candidates:
            standard = float(max(candidates))
        else:
            standard = current * 0.94

        # (C) 積極
        retr = _retracement_pct(composite_score)
        aggressive = current * (1.0 - retr)

        conservative = float(conservative)
        standard = float(standard)
        aggressive = float(aggressive)

        c_f, s_f, a_f, used_fb = _ordered_entry_triple(conservative, standard, aggressive, current)
        conservative, standard, aggressive = c_f, s_f, a_f
        if used_fb:
            _log.debug("%s 進場價順序異常，改用 current×(0.92,0.96,0.98)", ticker)

        # 距離百分比（相對 current）
        to_fair = (fair_for_pct - current) / current * 100.0
        to_cons = (conservative - current) / current * 100.0
        to_std = (standard - current) / current * 100.0
        to_agg = (aggressive - current) / current * 100.0

        # current_zone
        if current <= conservative:
            current_zone = "below_conservative"
        elif current <= standard:
            current_zone = "in_conservative_zone"
        elif current <= aggressive:
            current_zone = "in_standard_zone"
        elif current <= aggressive * 1.05:
            current_zone = "above_aggressive"
        else:
            current_zone = "far_above"

        rsi_overbought = rsi_signal == "超買"
        overvalued = valuation_signal == "overvalued"
        avoid_price = current > aggressive * 1.05
        avoid_val = overvalued and rsi_overbought

        if avoid_price or avoid_val:
            current_status = "avoid"
        elif current <= standard:
            current_status = "buy_now"
        elif current <= aggressive:
            current_status = "watch"
        elif current <= aggressive * 1.05:
            current_status = "wait"
        else:
            current_status = "avoid"

        # reasoning
        fund_total = _to_float(fundamental.get("total_score"), 0.0) or 0.0
        suffix = ""
        if current_status == "avoid":
            if fund_total >= 85:
                suffix += "（基本面佳，宜等回檔後再分批布局）"
            if valuation_signal == "undervalued":
                suffix += "（中長期估值仍偏低，但短線已脫離進場帶）"
        else:
            if fund_total >= 85:
                suffix += "（基本面優異，可較大膽分批）"
            if valuation_signal == "undervalued":
                suffix += "（估值低估，下檔有限）"

        pct_to_std = max(0.0, (current - standard) / current * 100.0) if current > 0 else 0.0

        if current_status == "buy_now":
            base = (
                f"當前股價 {current:.2f} 已進入進場區間（標準進場 {standard:.2f}），可分批建倉。"
            )
        elif current_status == "watch":
            base = (
                f"{ticker} 接近進場區，距標準進場價 {standard:.2f} 還需下跌 {pct_to_std:.1f}%，建議耐心等待。"
            )
        elif current_status == "wait":
            base = f"{ticker} 已突破進場區間，建議等回測 {standard:.2f} 附近再進場。"
        elif avoid_val:
            base = f"{ticker} 估值偏高且技術面過熱，不建議追高，等待 -10% 以上回檔。"
        else:
            base = (
                f"{ticker} 股價已明顯高於積極進場區（逾 5%），不建議追高，"
                f"可等待回檔至標準進場 {standard:.2f} 附近再評估。"
            )

        reasoning = (base + suffix).strip()

        # batch_plan
        first_px = current if current_status == "buy_now" else aggressive
        batch_plan = [
            {
                "price": round(float(first_px), 4),
                "allocation_pct": 30,
                "condition": "第 1 批：現價進場（已在買進區）或積極價附近" if current_status == "buy_now" else "第 1 批：積極進場價附近試單",
            },
            {
                "price": round(float(standard), 4),
                "allocation_pct": 35,
                "condition": "第 2 批：標準進場（技術支撐區）",
            },
            {
                "price": round(float(conservative), 4),
                "allocation_pct": 35,
                "condition": "第 3 批：保守進場（安全邊際較大）",
            },
        ]

        pred30 = _to_float(prediction.get("predicted_30d"), current) or current
        denom = current - conservative
        if abs(denom) < 1e-9 or denom <= 0:
            risk_reward = 0.0
        else:
            risk_reward = float((pred30 - current) / denom)

        return {
            "ticker": ticker,
            "current_price": round(float(current), 4),
            "entry_prices": {
                "conservative": round(conservative, 4),
                "standard": round(standard, 4),
                "aggressive": round(aggressive, 4),
            },
            "current_status": current_status,
            "current_zone": current_zone,
            "distance": {
                "to_fair_value_pct": round(float(to_fair), 4),
                "to_conservative_pct": round(float(to_cons), 4),
                "to_standard_pct": round(float(to_std), 4),
                "to_aggressive_pct": round(float(to_agg), 4),
            },
            "reasoning": reasoning,
            "batch_plan": batch_plan,
            "risk_reward_ratio": round(float(risk_reward), 4),
        }
    except Exception as e:  # noqa: BLE001
        _log.warning("calculate_entry_prices 失敗: %s", e)
        return {
            **empty,
            "ticker": str(stock_data.get("ticker") or "?"),
            "reasoning": f"計算過程發生錯誤：{e}",
        }


if __name__ == "__main__":
    try:
        from data_collector import fetch_stock_data  # noqa: E402
        from daily_picker import analyze_single_stock  # noqa: E402
    except ModuleNotFoundError:
        from src.data_collector import fetch_stock_data  # type: ignore  # noqa: E402
        from src.daily_picker import analyze_single_stock  # type: ignore  # noqa: E402

    for sym in ("NVDA", "TSM", "AAPL"):
        print("\n" + "=" * 72)
        print(f"  {sym} 進場價分析")
        print("=" * 72)
        raw = fetch_stock_data(sym, force_refresh=False)
        if not raw:
            print("  無法取得資料，略過")
            continue
        analyzed = analyze_single_stock(sym, raw)
        analyzed["price_history"] = raw.get("price_history")
        out = calculate_entry_prices(analyzed)
        print(f"  現價: {out['current_price']}")
        ep = out["entry_prices"]
        print(f"  進場價 — 保守 {ep['conservative']:.2f} | 標準 {ep['standard']:.2f} | 積極 {ep['aggressive']:.2f}")
        print(f"  狀態: {get_status_emoji(out['current_status'])} {out['current_status']} / {get_status_label_cn(out['current_status'])}")
        print(f"  區間位置: {out['current_zone']}")
        print(f"  風險報酬比: {out['risk_reward_ratio']}")
        print(f"  說明: {out['reasoning']}")
        print("  分批計畫:")
        for b in out.get("batch_plan") or []:
            print(f"    {b.get('allocation_pct')}% @ {b.get('price')} — {b.get('condition')}")
        print(f"  距離%: {out['distance']}")
