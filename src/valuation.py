# -*- coding: utf-8 -*-
"""
Valuation: a fair price, the upside or downside, and a valuation signal.

Run from the project root:
    python src/valuation.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

# Load config from the project root (sys.path defaults to src/ when this file is run directly).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import VALUATION_PARAMS  # noqa: E402
try:  # Run directly (python src/valuation.py)
    from fundamentals import get_stock_sector  # type: ignore
except ModuleNotFoundError:  # Imported as a package (import src.valuation)
    from src.fundamentals import get_stock_sector  # type: ignore


def _to_float_or_none(value: Any) -> float | None:
    """Turn an input into a float. None when it cannot be converted or is NaN or Inf."""
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _normalize_weights(
    pe_price: float | None,
    peg_price: float | None,
    analyst_price: float | None,
) -> tuple[float, float, float]:
    """
    Redistribute weights across the valuation methods that are available.

    Base weights: P/E 40%, PEG 30%, analyst target 30%.
    The weight of an unavailable method (price is None) is spread across the available methods in proportion.
    """
    base = {"pe": 0.40, "peg": 0.30, "analyst": 0.30}
    available = {
        "pe": pe_price is not None,
        "peg": peg_price is not None,
        "analyst": analyst_price is not None,
    }
    total = sum(base[k] for k, ok in available.items() if ok)
    if total <= 0:
        return 0.0, 0.0, 0.0
    return (
        base["pe"] / total if available["pe"] else 0.0,
        base["peg"] / total if available["peg"] else 0.0,
        base["analyst"] / total if available["analyst"] else 0.0,
    )


def _valuation_score(upside_pct: float) -> int:
    """
    Valuation attractiveness score (0-100).

    - upside above +30%: 95
    - +15% to +30%: 85
    - +5% to +15%: 70
    - -5% to +5%: 55
    - -15% to -5%: 40
    - below -15%: 25
    """
    if upside_pct > 30:
        return 95
    if upside_pct >= 15:
        return 85
    if upside_pct >= 5:
        return 70
    if upside_pct >= -5:
        return 55
    if upside_pct >= -15:
        return 40
    return 25


def calculate_fair_value(
    ticker: str,
    info: dict[str, Any],
    current_price: float,
) -> dict[str, Any]:
    """
    Estimate a fair price from P/E, PEG, and the analyst target, then combine them with weights.

    Args:
        ticker: Ticker symbol.
        info: The info dict returned by data_collector.
        current_price: Current price.

    Returns:
        A valuation result with fair_value, upside_pct, valuation_signal, methods, and score.
    """
    sym = ticker.strip().upper()
    safe_info = info if isinstance(info, dict) else {}
    curr = _to_float_or_none(current_price)
    if curr is None or curr <= 0:
        raise ValueError("current_price must be positive")

    # Look up the sector from SECTOR_MAP. Unknown names use consumer as a conservative default.
    sector_key = get_stock_sector(sym)
    if sector_key == "unknown":
        sector_key = "consumer"

    pe_bench = VALUATION_PARAMS.get("pe_benchmarks", {})
    bench = pe_bench.get(sector_key) or pe_bench.get("consumer") or {"low": 15, "mid": 22, "high": 30}
    pe_low = float(bench["low"])
    pe_mid = float(bench["mid"])
    pe_high = float(bench["high"])

    trailing_pe = _to_float_or_none(safe_info.get("trailingPE"))
    forward_pe = _to_float_or_none(safe_info.get("forwardPE"))
    earnings_growth = _to_float_or_none(safe_info.get("earningsGrowth"))
    analyst_target = _to_float_or_none(safe_info.get("targetMeanPrice"))

    # Method A: P/E. Back out EPS, then multiply by the sector's fair P/E band.
    eps = None
    pe_low_price = None
    pe_mid_price = None
    pe_high_price = None
    if trailing_pe is not None and trailing_pe > 0:
        eps = curr / trailing_pe
        pe_low_price = eps * pe_low
        pe_mid_price = eps * pe_mid
        pe_high_price = eps * pe_high

    # If trailingPE is unavailable, forwardPE can estimate EPS so at least one method can return a price.
    if eps is None and forward_pe is not None and forward_pe > 0:
        eps = curr / forward_pe
        pe_low_price = eps * pe_low
        pe_mid_price = eps * pe_mid
        pe_high_price = eps * pe_high

    pe_fair = pe_mid_price

    # Method B: PEG. Drop it when growth is missing or negative.
    peg_fair = None
    peg_growth = None
    peg_target_pe = None
    if eps is not None and earnings_growth is not None and earnings_growth >= 0:
        growth_pct = min(earnings_growth * 100.0, 50.0)  # Cap growth above 50%.
        peg_growth = growth_pct
        peg_target_pe = min(growth_pct, pe_high * 1.2)
        if peg_target_pe > 0:
            peg_fair = eps * peg_target_pe

    # Method C: analyst target price.
    analyst_fair = analyst_target if analyst_target is not None and analyst_target > 0 else None

    w_pe, w_peg, w_ana = _normalize_weights(pe_fair, peg_fair, analyst_fair)
    if w_pe == 0 and w_peg == 0 and w_ana == 0:
        raise ValueError(f"{sym} has no usable valuation method (missing P/E, growth, and analyst target)")

    fair_value = 0.0
    if pe_fair is not None:
        fair_value += pe_fair * w_pe
    if peg_fair is not None:
        fair_value += peg_fair * w_peg
    if analyst_fair is not None:
        fair_value += analyst_fair * w_ana

    upside_pct = (fair_value - curr) / curr * 100.0
    if upside_pct > 15:
        signal = "undervalued"
    elif upside_pct >= -10:
        signal = "fair"
    else:
        signal = "overvalued"

    return {
        "ticker": sym,
        "current_price": round(curr, 4),
        "fair_value": round(float(fair_value), 4),
        "upside_pct": round(float(upside_pct), 2),
        "valuation_signal": signal,
        "methods": {
            "pe_method": {
                "fair_price": round(pe_fair, 4) if pe_fair is not None else None,
                "range_low": round(pe_low_price, 4) if pe_low_price is not None else None,
                "range_mid": round(pe_mid_price, 4) if pe_mid_price is not None else None,
                "range_high": round(pe_high_price, 4) if pe_high_price is not None else None,
                "weight": round(w_pe, 4),
            },
            "peg_method": {
                "fair_price": round(peg_fair, 4) if peg_fair is not None else None,
                "growth_rate": round(peg_growth, 4) if peg_growth is not None else None,
                "target_pe": round(peg_target_pe, 4) if peg_target_pe is not None else None,
                "weight": round(w_peg, 4),
            },
            "analyst_target": {
                "fair_price": round(analyst_fair, 4) if analyst_fair is not None else None,
                "weight": round(w_ana, 4),
            },
        },
        "sector_used": sector_key,
        "score": _valuation_score(upside_pct),
    }


def analyze_valuation_batch(all_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Value a batch and sort by upside_pct, high to low (the most undervalued names first).

    Args:
        all_data: The dict returned by collect_all_stocks().

    Returns:
        A list of valuation results. A single failure is skipped.
    """
    rows: list[dict[str, Any]] = []
    for key, payload in all_data.items():
        if not isinstance(payload, dict):
            continue
        sym = str(payload.get("ticker") or key).upper()
        info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
        curr = _to_float_or_none(payload.get("current_price"))
        if curr is None or curr <= 0:
            continue
        try:
            rows.append(calculate_fair_value(sym, info, curr))
        except Exception:
            continue
    rows.sort(key=lambda x: x.get("upside_pct", float("-inf")), reverse=True)
    return rows


def _fmt_price(x: float | None) -> str:
    """Format a price field."""
    if x is None:
        return "-"
    return f"{x:.2f}"


def _print_table(rows: list[dict[str, Any]]) -> None:
    """Print a short valuation summary table."""
    headers = ("ticker", "current", "fair", "upside%", "signal", "PE low-mid-high")
    widths = (8, 10, 10, 10, 12, 26)
    line = " ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))
    print(line)
    print("-" * len(line))
    for r in rows:
        pe = r["methods"]["pe_method"]
        pe_range = f"{_fmt_price(pe['range_low'])}-{_fmt_price(pe['range_mid'])}-{_fmt_price(pe['range_high'])}"
        print(
            " ".join(
                [
                    r["ticker"].ljust(widths[0]),
                    _fmt_price(r["current_price"]).ljust(widths[1]),
                    _fmt_price(r["fair_value"]).ljust(widths[2]),
                    f"{r['upside_pct']:.2f}%".ljust(widths[3]),
                    r["valuation_signal"].ljust(widths[4]),
                    pe_range.ljust(widths[5]),
                ]
            )
        )


if __name__ == "__main__":
    _SRC_DIR = Path(__file__).resolve().parent
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))

    from data_collector import fetch_stock_data

    # The request originally mentioned KO. The final list uses XOM, which is on the universe.
    tickers = ["NVDA", "AAPL", "TSM", "JPM", "XOM"]
    results: list[dict[str, Any]] = []
    for t in tickers:
        raw = fetch_stock_data(t, force_refresh=False)
        if not raw:
            print(f"[{t}] Fetch failed, skipping.")
            continue
        info = raw.get("info") if isinstance(raw.get("info"), dict) else {}
        curr = _to_float_or_none(raw.get("current_price"))
        if curr is None or curr <= 0:
            print(f"[{t}] current_price is invalid, skipping.")
            continue
        try:
            results.append(calculate_fair_value(t, info, curr))
        except Exception as e:
            print(f"[{t}] Valuation failed: {e}")

    results.sort(key=lambda x: x["upside_pct"], reverse=True)
    print("\n=== Valuation comparison ===")
    _print_table(results)
