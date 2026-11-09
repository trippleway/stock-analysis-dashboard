# -*- coding: utf-8 -*-
"""
Ideal entry-price band, computed from the combined result of daily_picker.analyze_single_stock.

stock_data may include the raw price_history DataFrame so the recent 30-day low can be used as support.
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
        "buy_now": "Buy now",
        "watch": "Near the entry zone",
        "wait": "Wait for a pullback",
        "avoid": "Do not chase",
    }.get(status, "Neutral")


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
    """Lowest Low over the last n bars."""
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
    Keep conservative <= standard <= aggressive <= current.
    If that order fails, use current times (0.92, 0.96, 0.98) and mark used_fallback.
    """
    c, s, a, cur = conservative, standard, aggressive, current
    ok = c <= s <= a <= cur and cur > 0
    if ok:
        return c, s, a, False
    cur = max(cur, 1e-9)
    return cur * 0.92, cur * 0.96, cur * 0.98, True


def calculate_entry_prices(stock_data: dict[str, Any]) -> dict[str, Any]:
    """
    Compute entry prices and status from an analyze_single_stock result.

    Optional: stock_data['price_history'] is a pd.DataFrame (the same shape as fetch_stock_data)
    used for the "lowest close of the last 30 days" support. Without it, the standard entry uses only the lower Bollinger band and MA20.
    """
    err_reasoning = "Not enough data to compute a full entry price. Check the technical and valuation fields."
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
        # An implausible trailing-P/E fair value must not be described as cheap.
        valuation_trusted = valuation.get("valuation_reliable") is not False
        rsi_signal = str((technical.get("signals") or {}).get("rsi") or "").strip()

        # (A) Conservative
        if fair_raw is not None and fair_raw > 0:
            conservative = fair_raw * 0.75
        else:
            conservative = current * 0.85

        # (B) Standard: the maximum of the three candidates
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

        # (C) Aggressive
        retr = _retracement_pct(composite_score)
        aggressive = current * (1.0 - retr)

        conservative = float(conservative)
        standard = float(standard)
        aggressive = float(aggressive)

        c_f, s_f, a_f, used_fb = _ordered_entry_triple(conservative, standard, aggressive, current)
        conservative, standard, aggressive = c_f, s_f, a_f
        if used_fb:
            _log.debug("%s entry prices were out of order; using current x (0.92, 0.96, 0.98)", ticker)

        # Distance percents, relative to current.
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

        rsi_overbought = rsi_signal == "Overbought"
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
                suffix += " (fundamentals are strong; wait for a pullback and then scale in)"
            if valuation_signal == "undervalued" and valuation_trusted:
                suffix += " (the medium-term valuation is still low, but the short-term price has left the entry band)"
        else:
            if fund_total >= 85:
                suffix += " (fundamentals are excellent; scaling in can be a bit more aggressive)"
            if valuation_signal == "undervalued" and valuation_trusted:
                suffix += " (valuation is cheap and downside looks limited)"

        pct_to_std = max(0.0, (current - standard) / current * 100.0) if current > 0 else 0.0

        if current_status == "buy_now":
            base = (
                f"The current price {current:.2f} is inside the entry zone (standard entry {standard:.2f}). Scale in."
            )
        elif current_status == "watch":
            base = (
                f"{ticker} is near the entry zone. It still needs to fall {pct_to_std:.1f}% to the standard entry {standard:.2f}. Wait."
            )
        elif current_status == "wait":
            base = f"{ticker} has broken above the entry zone. Wait for a retest near {standard:.2f} before entering."
        elif avoid_val:
            base = f"{ticker} looks expensive and the technicals are overheated. Do not chase. Wait for a pullback of 10% or more."
        else:
            base = (
                f"{ticker} is clearly above the aggressive entry zone (more than 5%). Do not chase. "
                f"Wait for a pullback near the standard entry {standard:.2f} and reassess."
            )

        reasoning = (base + suffix).strip()

        # batch_plan
        first_px = current if current_status == "buy_now" else aggressive
        batch_plan = [
            {
                "price": round(float(first_px), 4),
                "allocation_pct": 30,
                "condition": "Tranche 1: enter at the current price (already in the buy zone) or near the aggressive price" if current_status == "buy_now" else "Tranche 1: a starter near the aggressive entry",
            },
            {
                "price": round(float(standard), 4),
                "allocation_pct": 35,
                "condition": "Tranche 2: standard entry (technical support)",
            },
            {
                "price": round(float(conservative), 4),
                "allocation_pct": 35,
                "condition": "Tranche 3: conservative entry (a larger margin of safety)",
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
        _log.warning("calculate_entry_prices failed: %s", e)
        return {
            **empty,
            "ticker": str(stock_data.get("ticker") or "?"),
            "reasoning": f"An error occurred during the calculation: {e}",
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
        print(f"  {sym} entry-price analysis")
        print("=" * 72)
        raw = fetch_stock_data(sym, force_refresh=False)
        if not raw:
            print("  Could not load data, skipping")
            continue
        analyzed = analyze_single_stock(sym, raw)
        analyzed["price_history"] = raw.get("price_history")
        out = calculate_entry_prices(analyzed)
        print(f"  Price: {out['current_price']}")
        ep = out["entry_prices"]
        print(f"  Entry — conservative {ep['conservative']:.2f} | standard {ep['standard']:.2f} | aggressive {ep['aggressive']:.2f}")
        print(f"  Status: {get_status_emoji(out['current_status'])} {out['current_status']} / {get_status_label_cn(out['current_status'])}")
        print(f"  Zone: {out['current_zone']}")
        print(f"  Risk/reward: {out['risk_reward_ratio']}")
        print(f"  Note: {out['reasoning']}")
        print("  Scale-in plan:")
        for b in out.get("batch_plan") or []:
            print(f"    {b.get('allocation_pct')}% @ {b.get('price')} — {b.get('condition')}")
        print(f"  Distance %: {out['distance']}")
