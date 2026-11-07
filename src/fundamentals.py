# -*- coding: utf-8 -*-
"""
Fundamental analysis: subscores and a weighted total from yfinance info fields.

Run from the project root:
    python src/fundamentals.py
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

from config import SECTOR_BASE_SCORE, SECTOR_MAP  # noqa: E402

# Weights for the fundamental total.
_WEIGHT_GROWTH = 0.30
_WEIGHT_PROFIT = 0.30
_WEIGHT_HEALTH = 0.20
_WEIGHT_INDUSTRY = 0.20


def _to_float_or_none(value: Any) -> float | None:
    """Turn any input into a float. Invalid values, nan, and inf become None."""
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def get_stock_sector(ticker: str) -> str:
    """
    Look up the sector key for a stock from config.SECTOR_MAP.

    Args:
        ticker: Ticker symbol (case insensitive).

    Returns:
        For example 'semiconductor' or 'finance'. 'unknown' when it is not found.
    """
    sym = ticker.strip().upper()
    for sector_key, tickers in SECTOR_MAP.items():
        if sym in (t.upper() for t in tickers):
            return sector_key
    return "unknown"


def _score_growth(rev: float | None, ear: float | None) -> tuple[float, str]:
    """
    Growth subscore from revenue growth and earnings growth (usually a decimal, so 0.15 means 15%).

    Score rules:
    - Both above 30%: 95. The signal token is left as-is because daily_picker compares it.
    - Both above 15%: 85
    - Both above 5%: 70 (steady growth)
    - One positive and one negative, or both in 0-5%: 55 (low growth)
    - Both below 0: 30 (declining)
    - Missing data (None): a neutral 50
    """
    if rev is None or ear is None:
        return 50.0, "Low growth"

    if rev > 0.30 and ear > 0.30:
        return 95.0, "高速成長"
    if rev > 0.15 and ear > 0.15:
        return 85.0, "Steady growth"
    if rev > 0.05 and ear > 0.05:
        return 70.0, "Steady growth"

    one_pos_one_neg = (rev > 0 > ear) or (ear > 0 > rev)
    both_low_band = 0.0 <= rev <= 0.05 and 0.0 <= ear <= 0.05
    if one_pos_one_neg or both_low_band:
        return 55.0, "Low growth"

    if rev < 0.0 and ear < 0.0:
        return 30.0, "Declining"

    return 55.0, "Low growth"


def _gross_margin_points(g: float | None) -> int:
    """Gross-margin points by band. A missing value gets the middle band, 15 points."""
    if g is None:
        return 15
    if g > 0.50:
        return 25
    if g > 0.40:
        return 20
    if g > 0.30:
        return 15
    if g > 0.20:
        return 10
    return 5


def _profit_margin_points(p: float | None) -> int:
    """Net-margin points. A missing value gets the middle band, 15 points."""
    if p is None:
        return 15
    if p > 0.25:
        return 25
    if p > 0.15:
        return 20
    if p > 0.08:
        return 15
    if p > 0.03:
        return 10
    return 5


def _roe_points(r: float | None) -> int:
    """ROE points. A missing value gets the middle band, 20 points (between 10-15% and 15-25%)."""
    if r is None:
        return 20
    if r > 0.25:
        return 30
    if r > 0.15:
        return 25
    if r > 0.10:
        return 20
    if r > 0.05:
        return 10
    return 5


def _score_profitability(
    gross: float | None,
    profit: float | None,
    roe: float | None,
) -> tuple[float, str]:
    """
    Profitability subscore: the three point bands plus a base of 20, capped at 100.

    Rules:
    - grossMargins adds 25/20/15/10/5 by band
    - profitMargins adds 25/20/15/10/5 by band
    - ROE adds 30/25/20/10/5 by band
    - The three items start from a base of 20
    - Missing data gets the middle band inside each *_points helper
    """
    raw = 20 + _gross_margin_points(gross) + _profit_margin_points(profit) + _roe_points(roe)
    score = float(max(0.0, min(100.0, raw)))

    if score >= 85:
        sig = "Excellent"
    elif score >= 70:
        sig = "Good"
    elif score >= 55:
        sig = "Average"
    else:
        sig = "Weak"
    return score, sig


def _debt_component(dte_raw: float | None) -> float:
    """
    Debt-to-equity component (0-50). The API often returns a percent number, so it is divided by 100 before the thresholds.

    Rules:
    - Below 0.5 (below 50%): 50
    - 0.5-1.0: 40
    - 1.0-2.0: 30
    - Above 2.0: 20
    - Missing: 30
    """
    if dte_raw is None:
        return 30.0
    ratio = dte_raw / 100.0
    if ratio < 0.5:
        return 50.0
    if ratio <= 1.0:
        return 40.0
    if ratio <= 2.0:
        return 30.0
    return 20.0


def _current_ratio_component(cr: float | None) -> float:
    """
    Current-ratio component (0-50).

    Rules:
    - Above 2.0: 50
    - 1.5-2.0: 40
    - 1.0-1.5: 30
    - Below 1.0: 15
    - Missing: 30
    """
    if cr is None:
        return 30.0
    if cr > 2.0:
        return 50.0
    if cr >= 1.5:
        return 40.0
    if cr >= 1.0:
        return 30.0
    return 15.0


def _score_financial_health(dte_raw: float | None, cr: float | None) -> tuple[float, str]:
    """
    Financial-health subscore: debt-to-equity and current ratio each contribute 0-50, for a total of 0-100.

    The signal reads the combined score. A high total means leverage and liquidity look steadier.
    The "needs attention" and "risk" tokens are left as-is because daily_picker compares them.
    """
    d = _debt_component(dte_raw)
    c = _current_ratio_component(cr)
    total = max(0.0, min(100.0, d + c))

    if total >= 80:
        sig = "Healthy"
    elif total >= 60:
        sig = "Acceptable"
    elif total >= 40:
        sig = "需注意"
    else:
        sig = "風險"
    return total, sig


def _score_industry_outlook(
    ticker: str,
    info: dict[str, Any],
) -> tuple[float, str]:
    """
    Industry-outlook subscore: a sector base score plus a forward P/E adjustment.

    Rules:
    - Base = 50 + SECTOR_BASE_SCORE[sector]
    - If forwardPE < trailingPE (earnings are expected to grow): +10
    - Capped at 100
    """
    sector_key = get_stock_sector(ticker)
    base_bonus = SECTOR_BASE_SCORE.get(sector_key, 0)
    score = 50.0 + float(base_bonus)

    fpe = _to_float_or_none(info.get("forwardPE"))
    tpe = _to_float_or_none(info.get("trailingPE"))
    if fpe is not None and tpe is not None and fpe < tpe:
        score += 10.0

    score = max(0.0, min(100.0, score))

    if score >= 70:
        sig = "In favor"
    elif score >= 58:
        sig = "Neutral"
    else:
        sig = "Under pressure"
    return score, sig


def analyze_fundamentals(ticker: str, info: dict[str, Any]) -> dict[str, Any]:
    """
    Compute the weighted fundamental total, subscores, details, and signals from one stock's info dict.

    Args:
        ticker: Ticker symbol.
        info: The info sub-dict returned by data_collector. Fields may be None.

    Returns:
        A dict with ticker, total_score, sub_scores, details, and signals.
    """
    sym = ticker.strip().upper()
    safe_info = info if isinstance(info, dict) else {}

    rev = _to_float_or_none(safe_info.get("revenueGrowth"))
    ear = _to_float_or_none(safe_info.get("earningsGrowth"))
    gross = _to_float_or_none(safe_info.get("grossMargins"))
    profit = _to_float_or_none(safe_info.get("profitMargins"))
    roe = _to_float_or_none(safe_info.get("returnOnEquity"))
    dte_raw = _to_float_or_none(safe_info.get("debtToEquity"))
    cr = _to_float_or_none(safe_info.get("currentRatio"))

    g_score, g_sig = _score_growth(rev, ear)
    p_score, p_sig = _score_profitability(gross, profit, roe)
    h_score, h_sig = _score_financial_health(dte_raw, cr)
    i_score, i_sig = _score_industry_outlook(sym, safe_info)

    total = (
        g_score * _WEIGHT_GROWTH
        + p_score * _WEIGHT_PROFIT
        + h_score * _WEIGHT_HEALTH
        + i_score * _WEIGHT_INDUSTRY
    )

    sector_key = get_stock_sector(sym)
    sector_name = safe_info.get("sector")
    if not isinstance(sector_name, str):
        sector_name = None

    details: dict[str, Any] = {
        "revenue_growth": rev,
        "earnings_growth": ear,
        "gross_margin": gross,
        "profit_margin": profit,
        "roe": roe,
        "debt_to_equity": (dte_raw / 100.0) if dte_raw is not None else None,
        "debt_to_equity_raw": dte_raw,
        "current_ratio": cr,
        "sector": sector_name or "",
        "mapped_sector": sector_key,
        "industry": safe_info.get("industry") if isinstance(safe_info.get("industry"), str) else None,
        "market_cap": _to_float_or_none(safe_info.get("marketCap")),
        "trailing_pe": _to_float_or_none(safe_info.get("trailingPE")),
        "forward_pe": _to_float_or_none(safe_info.get("forwardPE")),
        "price_to_book": _to_float_or_none(safe_info.get("priceToBook")),
    }

    return {
        "ticker": sym,
        "total_score": round(float(total), 2),
        "sub_scores": {
            "growth": round(g_score, 2),
            "profitability": round(p_score, 2),
            "financial_health": round(h_score, 2),
            "industry_outlook": round(i_score, 2),
        },
        "details": details,
        "signals": {
            "growth": g_sig,
            "profitability": p_sig,
            "financial_health": h_sig,
            "industry_outlook": i_sig,
        },
    }


def analyze_all_stocks(all_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Analyze the dict returned by collect_all_stocks() and sort by total_score, high to low.

    Args:
        all_data: Keys are symbols. Values are fetch-result dicts or None.

    Returns:
        One analyze_fundamentals result dict per element.
    """
    rows: list[dict[str, Any]] = []
    for key, payload in all_data.items():
        if not isinstance(payload, dict):
            continue
        info = payload.get("info")
        if not isinstance(info, dict):
            info = {}
        sym = str(payload.get("ticker") or key).upper()
        rows.append(analyze_fundamentals(sym, info))
    rows.sort(key=lambda r: r.get("total_score", 0.0), reverse=True)
    return rows


def _pct(x: float | None, *, digits: int = 1) -> str:
    """Format a decimal ratio as a percent string. None becomes '-'."""
    if x is None:
        return "-"
    return f"{x * 100:.{digits}f}%"


def _print_results_table(results: list[dict[str, Any]]) -> None:
    """Print a fundamental summary table with fixed column widths."""
    headers = (
        "Ticker",
        "Total",
        "Grw",
        "Prf",
        "Hlth",
        "Ind",
        "RevGr",
        "ROE",
        "Gross",
        "D/E",
        "Sector",
    )
    widths = (8, 7, 5, 5, 5, 5, 8, 8, 8, 7, 14)
    line = " ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))
    print(line)
    print("-" * len(line))
    for r in results:
        d = r["details"]
        debt = d.get("debt_to_equity")
        debt_s = f"{debt:.2f}" if debt is not None else "-"
        row = (
            r["ticker"].ljust(widths[0]),
            f"{r['total_score']:.1f}".ljust(widths[1]),
            f"{r['sub_scores']['growth']:.0f}".ljust(widths[2]),
            f"{r['sub_scores']['profitability']:.0f}".ljust(widths[3]),
            f"{r['sub_scores']['financial_health']:.0f}".ljust(widths[4]),
            f"{r['sub_scores']['industry_outlook']:.0f}".ljust(widths[5]),
            _pct(d.get("revenue_growth")).ljust(widths[6]),
            _pct(d.get("roe")).ljust(widths[7]),
            _pct(d.get("gross_margin")).ljust(widths[8]),
            debt_s.ljust(widths[9]),
            (d.get("mapped_sector") or "-")[: widths[10] - 1].ljust(widths[10]),
        )
        print(" ".join(row))


if __name__ == "__main__":
    _SRC_DIR = Path(__file__).resolve().parent
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))
    from data_collector import fetch_stock_data

    tickers = ["NVDA", "AAPL", "TSM", "JPM", "XOM"]
    print("Fetching data...")
    results: list[dict[str, Any]] = []
    for t in tickers:
        raw = fetch_stock_data(t, force_refresh=False)
        if not raw:
            print(f"[{t}] Fetch failed, skipping.")
            continue
        info = raw.get("info") or {}
        results.append(analyze_fundamentals(t, info))

    print("\n=== Fundamental scores (total and subscores) ===\n")
    _print_results_table(results)

    print("\n=== Subscores and signals by name ===\n")
    for r in results:
        print(f"{r['ticker']}: total={r['total_score']} | sub={r['sub_scores']} | signals={r['signals']}")
