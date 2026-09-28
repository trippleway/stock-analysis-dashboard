# -*- coding: utf-8 -*-
"""
基本面分析：依 yfinance info 欄位計算子分數與加權總分。

執行方式（於專案根目錄）：
    python src/fundamentals.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

# 從專案根目錄載入 config（直接執行本檔時 sys.path 預設為 src/）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import SECTOR_BASE_SCORE, SECTOR_MAP  # noqa: E402

# 基本面總分權重
_WEIGHT_GROWTH = 0.30
_WEIGHT_PROFIT = 0.30
_WEIGHT_HEALTH = 0.20
_WEIGHT_INDUSTRY = 0.20


def _to_float_or_none(value: Any) -> float | None:
    """將任意輸入轉成 float；無效或 nan/inf 則 None。"""
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
    依 config.SECTOR_MAP 反查股票所屬產業鍵名。

    參數:
        ticker: 股票代號（不分大小寫）。

    回傳:
        例如 'semiconductor'、'finance'；找不到則 'unknown'。
    """
    sym = ticker.strip().upper()
    for sector_key, tickers in SECTOR_MAP.items():
        if sym in (t.upper() for t in tickers):
            return sector_key
    return "unknown"


def _score_growth(rev: float | None, ear: float | None) -> tuple[float, str]:
    """
    成長性子分數：綜合營收成長與盈餘成長（通常為小數，如 0.15 代表 15%）。

    評分規則對照：
    - 兩者皆 > 30%：95 分（高速成長）
    - 兩者皆 > 15%：85 分
    - 兩者皆 > 5%：70 分（穩健成長）
    - 一正一負或都在 0-5%：55 分（低成長）
    - 兩者皆 < 0：30 分（衰退）
    - 資料缺失（None）：50 分中性值
    """
    if rev is None or ear is None:
        return 50.0, "低成長"

    if rev > 0.30 and ear > 0.30:
        return 95.0, "高速成長"
    if rev > 0.15 and ear > 0.15:
        return 85.0, "穩健成長"
    if rev > 0.05 and ear > 0.05:
        return 70.0, "穩健成長"

    one_pos_one_neg = (rev > 0 > ear) or (ear > 0 > rev)
    both_low_band = 0.0 <= rev <= 0.05 and 0.0 <= ear <= 0.05
    if one_pos_one_neg or both_low_band:
        return 55.0, "低成長"

    if rev < 0.0 and ear < 0.0:
        return 30.0, "衰退"

    return 55.0, "低成長"


def _gross_margin_points(g: float | None) -> int:
    """毛利率加分（對照區間給分）；缺失給中間檔 15 分。"""
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
    """淨利率加分；缺失給中間檔 15 分。"""
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
    """ROE 加分；缺失給中間檔 20 分（介於 10-15% 與 15-25% 之間）。"""
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
    獲利品質子分數：三項加總加分 + 基礎分 20，最高 100。

    規則對照：
    - grossMargins 分檔加 25/20/15/10/5
    - profitMargins 分檔加 25/20/15/10/5
    - ROE 分檔加 30/25/20/10/5
    - 三項基礎分 20 分
    - 資料缺失時各項給中間分（於各 *_points 內處理）
    """
    raw = 20 + _gross_margin_points(gross) + _profit_margin_points(profit) + _roe_points(roe)
    score = float(max(0.0, min(100.0, raw)))

    if score >= 85:
        sig = "優秀"
    elif score >= 70:
        sig = "良好"
    elif score >= 55:
        sig = "一般"
    else:
        sig = "偏弱"
    return score, sig


def _debt_component(dte_raw: float | None) -> float:
    """
    負債權益比分項（0-50）：API 常為「百分比數字」，先除以 100 再與門檻比較。

    規則對照：
    - < 0.5（<50%）：50 分
    - 0.5-1.0：40 分
    - 1.0-2.0：30 分
    - > 2.0：20 分
    - 缺失：30 分
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
    流動比分項（0-50）。

    規則對照：
    - > 2.0：50 分
    - 1.5-2.0：40 分
    - 1.0-1.5：30 分
    - < 1.0：15 分
    - 缺失：30 分
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
    財務健康子分數：負債權益比與流動比各占 0-50，合計 0-100。

    綜合解讀（信號）：
    - 合計高代表槓桿與流動性較穩健
    """
    d = _debt_component(dte_raw)
    c = _current_ratio_component(cr)
    total = max(0.0, min(100.0, d + c))

    if total >= 80:
        sig = "健康"
    elif total >= 60:
        sig = "尚可"
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
    產業前景子分數：產業基礎分 + 本益比前瞻調整。

    規則對照：
    - 基礎分 = 50 + SECTOR_BASE_SCORE[sector]
    - 若 forwardPE < trailingPE（預期盈餘成長）：+10
    - 最高 100 分封頂
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
        sig = "熱門"
    elif score >= 58:
        sig = "中性"
    else:
        sig = "承壓"
    return score, sig


def analyze_fundamentals(ticker: str, info: dict[str, Any]) -> dict[str, Any]:
    """
    依單一股票的 info 字典計算基本面加權總分與子分數、細節與中文信號。

    參數:
        ticker: 股票代號。
        info: data_collector 回傳之 info 子字典（可含 None 欄位）。

    回傳:
        含 ticker、total_score、sub_scores、details、signals 的 dict。
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
    批量分析 collect_all_stocks() 回傳的字典，依 total_score 由高到低排序。

    參數:
        all_data: 鍵為代號、值為 fetch 結果 dict 或 None。

    回傳:
        每個元素為 analyze_fundamentals 的結果 dict。
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
    """將小數比例格式化成百分比字串；None 則 '-'。"""
    if x is None:
        return "-"
    return f"{x * 100:.{digits}f}%"


def _print_results_table(results: list[dict[str, Any]]) -> None:
    """以固定欄寬印出基本面摘要表。"""
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
    print("抓取資料中…")
    results: list[dict[str, Any]] = []
    for t in tickers:
        raw = fetch_stock_data(t, force_refresh=False)
        if not raw:
            print(f"[{t}] 抓取失敗，跳過。")
            continue
        info = raw.get("info") or {}
        results.append(analyze_fundamentals(t, info))

    print("\n=== 基本面評分（總分與子分數）===\n")
    _print_results_table(results)

    print("\n=== 各股子分數與信號 ===\n")
    for r in results:
        print(f"{r['ticker']}: total={r['total_score']} | sub={r['sub_scores']} | signals={r['signals']}")
