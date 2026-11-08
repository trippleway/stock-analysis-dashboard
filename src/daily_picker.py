# -*- coding: utf-8 -*-
"""
Daily stock-picking pipeline. Combines technicals, fundamentals, valuation, and the forecast into top picks.

Run from the project root:
    python src/daily_picker.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

# Load config from the project root (running this file directly puts src/ on sys.path)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import FILTER_THRESHOLDS, PICKS_DIR, SCORE_WEIGHTS, TICKER_LIST  # noqa: E402
try:  # Run as a script (python src/daily_picker.py)
    from data_collector import collect_all_stocks  # type: ignore
    from entry_price import calculate_entry_prices, get_status_emoji, get_status_label_cn  # type: ignore
    from fundamentals import analyze_fundamentals, get_stock_sector  # type: ignore
    from indicators import calculate_indicators, calculate_tech_score  # type: ignore
    from predictor import predict_price  # type: ignore
    from sentiment import analyze_ticker_sentiment  # type: ignore
    from valuation import calculate_fair_value  # type: ignore
except ModuleNotFoundError:  # Imported as a package (import src.daily_picker)
    from src.data_collector import collect_all_stocks  # type: ignore
    from src.entry_price import calculate_entry_prices, get_status_emoji, get_status_label_cn  # type: ignore
    from src.fundamentals import analyze_fundamentals, get_stock_sector  # type: ignore
    from src.indicators import calculate_indicators, calculate_tech_score  # type: ignore
    from src.predictor import predict_price  # type: ignore
    from src.sentiment import analyze_ticker_sentiment  # type: ignore
    from src.valuation import calculate_fair_value  # type: ignore

_LOG = logging.getLogger("daily_picker")
if not _LOG.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _to_float_or_none(value: Any) -> float | None:
    """Convert a value to float, or return None when it is not numeric."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _volume_score(vol_ratio: float | None) -> float:
    """
    Volume score from 0 to 100.

    - >1.5: 80
    - 1.0~1.5: 70
    - 0.5~1.0: 55
    - <0.5: 35
    """
    if vol_ratio is None:
        return 35.0
    if vol_ratio > 1.5:
        return 80.0
    if vol_ratio >= 1.0:
        return 70.0
    if vol_ratio > 0.5:
        return 55.0
    return 35.0


def _build_pick_reasons(
    sector: str,
    technical: dict[str, Any],
    fundamental: dict[str, Any],
    valuation: dict[str, Any],
    prediction: dict[str, Any],
    news: dict[str, Any],
    entry_price: dict[str, Any] | None = None,
) -> list[str]:
    """Build pick reasons from fundamentals, technicals, news, and entry price. At most about six."""
    reasons: list[str] = []

    if fundamental.get("signals", {}).get("growth") == "High Growth":
        reasons.append("Revenue and earnings are both growing quickly")
    if valuation.get("valuation_signal") == "undervalued":
        up = _to_float_or_none(valuation.get("upside_pct")) or 0.0
        reasons.append(f"Valuation looks cheap, upside +{up:.1f}%")
    if technical.get("signals", {}).get("macd") == "Golden Cross":
        reasons.append("MACD golden cross")
    if technical.get("signals", {}).get("ma_trend") == "Bullish Alignment":
        reasons.append("Moving averages are stacked bullishly")
    if (_to_float_or_none(prediction.get("expected_return_pct")) or 0.0) > 10:
        x = _to_float_or_none(prediction.get("expected_return_pct")) or 0.0
        reasons.append(f"Prophet 30-day forecast +{x:.1f}%")
    if (_to_float_or_none(fundamental.get("sub_scores", {}).get("profitability")) or 0.0) > 85:
        reasons.append("Profitability is strong")
    if sector in ("semiconductor", "ai_software"):
        reasons.append("In a peer group with a favorable industry outlook")
    if news.get("sentiment_label") == "Strongly Bullish":
        reasons.append("Recent news is strongly positive")
    if news.get("sentiment_label") == "Bullish" and int(news.get("positive_count", 0) or 0) >= 3:
        reasons.append("Several recent articles are positive")
    if (_to_float_or_none(news.get("sentiment_score")) or 0.0) > 0.3 and int(news.get("news_count", 0) or 0) >= 3:
        reasons.append(f"News sentiment is positive ({int(news.get('positive_count', 0) or 0)} positive articles)")

    if isinstance(entry_price, dict) and entry_price:
        if entry_price.get("current_status") == "buy_now":
            reasons.append("✓ Price is already inside the entry zone")
        dist_fair = _to_float_or_none((entry_price.get("distance") or {}).get("to_fair_value_pct"))
        if dist_fair is not None and dist_fair > 30:
            reasons.append(f"✓ Still {dist_fair:.1f}% below fair value")
        cur_ep = _to_float_or_none(entry_price.get("current_price"))
        batches = entry_price.get("batch_plan") if isinstance(entry_price.get("batch_plan"), list) else []
        if len(batches) >= 3 and cur_ep is not None and cur_ep > 0:
            p3 = _to_float_or_none(batches[2].get("price"))
            if p3 is not None and p3 < cur_ep * 0.85:
                reasons.append("✓ The staged-entry path is clear")

    # Drop duplicates and keep at most six, including the entry-price rules
    uniq = list(dict.fromkeys(reasons))
    return uniq[:6]


def _build_risk_factors(
    technical: dict[str, Any],
    fundamental: dict[str, Any],
    valuation: dict[str, Any],
    prediction: dict[str, Any],
    news: dict[str, Any],
) -> list[str]:
    """Build at most three risk notes from the same rules."""
    risks: list[str] = []

    if technical.get("signals", {}).get("rsi") == "Overbought":
        risks.append("RSI is overbought; a short-term pullback is a risk")
    if valuation.get("valuation_signal") == "overvalued":
        up = _to_float_or_none(valuation.get("upside_pct")) or 0.0
        risks.append(f"Valuation looks rich by {abs(up):.1f}%")
    if fundamental.get("signals", {}).get("financial_health") in ("Caution", "Risk"):
        risks.append("Financial leverage is high")
    if prediction.get("confidence") == "low":
        risks.append("The forecast is uncertain")
    if technical.get("signals", {}).get("macd") == "Death Cross":
        risks.append("MACD death cross")
    if news.get("sentiment_label") == "Strongly Bearish":
        risks.append("Recent news is clearly negative")
    if news.get("sentiment_label") == "Bearish" and int(news.get("negative_count", 0) or 0) >= 2:
        risks.append("Several recent articles are negative")
    if int(news.get("news_count", 0) or 0) == 0:
        risks.append("No recent news to use as context")

    return list(dict.fromkeys(risks))[:3]


def analyze_single_stock(ticker: str, raw_data: dict[str, Any]) -> dict[str, Any]:
    """
    Analyze one stock and return the combined result.

    Args:
        ticker: Ticker symbol.
        raw_data: The dict returned by fetch_stock_data.

    Returns:
        Technicals, fundamentals, valuation, forecast, composite score, filter result, reasons, and risks.
    """
    sym = ticker.strip().upper()
    if not isinstance(raw_data, dict):
        raise ValueError(f"{sym} raw_data has the wrong shape")

    info = raw_data.get("info") if isinstance(raw_data.get("info"), dict) else {}
    price_history = raw_data.get("price_history")
    current_price = _to_float_or_none(raw_data.get("current_price"))
    if not isinstance(price_history, pd.DataFrame):
        raise ValueError(f"{sym} is missing price_history")
    if current_price is None or current_price <= 0:
        raise ValueError(f"{sym} has an invalid current_price")

    name = raw_data.get("name") if isinstance(raw_data.get("name"), str) else sym
    sector = get_stock_sector(sym)

    _LOG.debug("Analyzing %s: technicals", sym)
    df_ind = calculate_indicators(price_history)
    technical = calculate_tech_score(df_ind)

    _LOG.debug("Analyzing %s: fundamentals", sym)
    fundamental = analyze_fundamentals(sym, info)

    _LOG.debug("Analyzing %s: valuation", sym)
    valuation = calculate_fair_value(sym, info, current_price)

    _LOG.debug("Analyzing %s: forecast", sym)
    prediction = predict_price(sym, price_history)

    _LOG.debug("Analyzing %s: news sentiment", sym)
    news_result = analyze_ticker_sentiment(sym)

    vol_ratio = _to_float_or_none(technical.get("current_values", {}).get("vol_ratio"))
    vol_score = _volume_score(vol_ratio)

    # Composite including news. Weights live in config.SCORE_WEIGHTS.
    composite = (
        (_to_float_or_none(technical.get("total_score")) or 0.0) * SCORE_WEIGHTS["technical"]
        + (_to_float_or_none(fundamental.get("total_score")) or 0.0) * SCORE_WEIGHTS["fundamental"]
        + (_to_float_or_none(valuation.get("score")) or 0.0) * SCORE_WEIGHTS["valuation"]
        + (_to_float_or_none(prediction.get("score")) or 0.0) * SCORE_WEIGHTS["prediction"]
        + vol_score * SCORE_WEIGHTS["volume"]
        + (_to_float_or_none(news_result.get("score")) or 50.0) * SCORE_WEIGHTS["news"]
    )

    # Filter thresholds. Every condition has to pass.
    passes_filter = (
        (_to_float_or_none(technical.get("total_score")) or 0.0) > FILTER_THRESHOLDS["min_tech_score"]
        and (_to_float_or_none(fundamental.get("total_score")) or 0.0) > FILTER_THRESHOLDS["min_fund_score"]
        and (_to_float_or_none(prediction.get("expected_return_pct")) or -999.0) > FILTER_THRESHOLDS["min_pred_return"]
        and (vol_ratio is not None and vol_ratio > FILTER_THRESHOLDS["min_volume_ratio"])
    )

    composite_score = round(float(composite), 2)

    entry_result = calculate_entry_prices(
        {
            "ticker": sym,
            "current_price": current_price,
            "price_history": price_history,
            "technical": technical,
            "fundamental": fundamental,
            "valuation": valuation,
            "prediction": prediction,
            "composite_score": composite_score,
        }
    )

    pick_reasons = _build_pick_reasons(
        sector,
        technical,
        fundamental,
        valuation,
        prediction,
        news_result,
        entry_price=entry_result,
    )
    risk_factors = _build_risk_factors(technical, fundamental, valuation, prediction, news_result)

    return {
        "ticker": sym,
        "name": name,
        "current_price": round(current_price, 4),
        "sector": sector,
        "technical": technical,
        "fundamental": fundamental,
        "valuation": valuation,
        "prediction": prediction,
        "news": news_result,
        "composite_score": composite_score,
        "passes_filter": bool(passes_filter),
        "pick_reasons": pick_reasons,
        "risk_factors": risk_factors,
        "entry_price": entry_result,
    }


def pick_top_stocks(all_data: dict[str, Any], top_n: int = 3) -> dict[str, Any]:
    """
    Pick the daily top N from a batch of raw stock payloads.

    Returns:
        date, picks, market_overview, and all_scores.
    """
    analyzed: list[dict[str, Any]] = []
    for key, raw in tqdm(all_data.items(), desc="Combined analysis", unit="name"):
        if not isinstance(raw, dict):
            continue
        sym = str(raw.get("ticker") or key).upper()
        try:
            analyzed.append(analyze_single_stock(sym, raw))
        except Exception as e:
            _LOG.error("%s analysis failed: %s", sym, e)

    analyzed.sort(key=lambda x: x.get("composite_score", 0.0), reverse=True)
    passed = [x for x in analyzed if x.get("passes_filter")]
    picks = passed[:top_n]

    # Market snapshot from SPY
    spy_price = None
    spy_change_pct = None
    spy_raw = all_data.get("SPY")
    if isinstance(spy_raw, dict):
        ph = spy_raw.get("price_history")
        if isinstance(ph, pd.DataFrame) and "Close" in ph.columns and len(ph) >= 2:
            c = pd.to_numeric(ph["Close"], errors="coerce").dropna()
            if len(c) >= 2:
                spy_price = float(c.iloc[-1])
                spy_change_pct = float((c.iloc[-1] / c.iloc[-2] - 1.0) * 100.0)

    all_scores = [
        {
            "ticker": r["ticker"],
            "name": r["name"],
            "composite_score": r["composite_score"],
            "passes_filter": r["passes_filter"],
            "tech_score": r["technical"]["total_score"],
            "fund_score": r["fundamental"]["total_score"],
            "valuation_score": r["valuation"]["score"],
            "prediction_return_pct": r["prediction"]["expected_return_pct"],
            "news_score": r.get("news", {}).get("score"),
        }
        for r in analyzed
    ]

    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "picks": picks,
        "market_overview": {
            "spy_price": round(spy_price, 4) if spy_price is not None else None,
            "spy_change_pct": round(spy_change_pct, 2) if spy_change_pct is not None else None,
            "total_analyzed": len(analyzed),
            "passed_filter": len(passed),
        },
        "all_scores": all_scores,
    }


def _json_safe(obj: Any) -> Any:
    """Recursively convert a value into something JSON can serialize."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in obj]
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:  # noqa: BLE001
            return str(obj)
    return str(obj)


def save_daily_picks(result: dict[str, Any]) -> Path:
    """
    Save the daily picks to PICKS_DIR / YYYY-MM-DD.json.
    """
    date_str = str(result.get("date") or datetime.now().strftime("%Y-%m-%d"))
    out_dir = Path(PICKS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date_str}.json"
    payload = _json_safe(result)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def _print_top3_report(result: dict[str, Any]) -> None:
    """Print the top-picks report."""
    print("\n=== Daily top picks ===")
    print(f"Date: {result.get('date')}")
    m = result.get("market_overview", {})
    print(
        f"Market: SPY={m.get('spy_price')} ({m.get('spy_change_pct')}%), "
        f"analyzed {m.get('total_analyzed')}, passed filter {m.get('passed_filter')}"
    )
    print("-" * 80)
    for i, p in enumerate(result.get("picks", []), start=1):
        print(
            f"{i}. {p['ticker']} {p['name']} | price {p['current_price']:.2f} | "
            f"composite {p['composite_score']:.2f}"
        )
        print(
            f"   technical={p['technical']['total_score']:.1f} "
            f"fundamental={p['fundamental']['total_score']:.1f} "
            f"valuation={p['valuation']['score']:.1f} "
            f"forecast return={p['prediction']['expected_return_pct']:.2f}%"
        )
        news = p.get("news", {}) or {}
        print(
            f"   news: {news.get('sentiment_label', 'No News')} "
            f"({int(news.get('news_count', 0) or 0)} articles: "
            f"positive {int(news.get('positive_count', 0) or 0)} / "
            f"negative {int(news.get('negative_count', 0) or 0)})"
        )
        if p.get("pick_reasons"):
            print("   reasons: " + "; ".join(p["pick_reasons"]))
        if p.get("risk_factors"):
            print("   risks: " + "; ".join(p["risk_factors"]))
        ep = p.get("entry_price") if isinstance(p.get("entry_price"), dict) else {}
        if ep:
            est = str(ep.get("current_status") or "")
            emoji = get_status_emoji(est)
            label = get_status_label_cn(est)
            ev = ep.get("entry_prices") or {}
            cons = _to_float_or_none(ev.get("conservative")) or 0.0
            std = _to_float_or_none(ev.get("standard")) or 0.0
            agg = _to_float_or_none(ev.get("aggressive")) or 0.0
            reasoning = str(ep.get("reasoning") or "").strip()
            print(
                f"   Entry: {emoji} {label}\n"
                f"      conservative ${cons:.2f} | standard ${std:.2f} | aggressive ${agg:.2f}\n"
                f"      {reasoning}"
            )
        print("-" * 80)


def run_daily_pipeline(force_refresh: bool = False) -> dict[str, Any]:
    """
    Daily pipeline:
    1) Fetch the universe (the first run, including Prophet, can take about 5-10 minutes)
    2) Score everything and pick the top 3
    3) Save JSON
    4) Print the report
    """
    _LOG.info("Starting the daily pipeline. A first full run with Prophet can take about 5-10 minutes; the cache speeds later runs.")
    est_minutes = max(4, int(round(len(TICKER_LIST) * 0.1)))
    _LOG.info("Starting news sentiment. Expect about %s minutes, including FinBERT and cache reads.", est_minutes)
    all_data = collect_all_stocks(tickers=TICKER_LIST, force_refresh=force_refresh)
    result = pick_top_stocks(all_data, top_n=3)
    out_path = save_daily_picks(result)
    _LOG.info("Saved daily picks: %s", out_path)

    try:
        try:
            from track_record import (  # type: ignore
                record_daily_picks,
                save_performance_snapshot,
                update_actuals,
            )
        except ModuleNotFoundError:
            from src.track_record import (  # type: ignore
                record_daily_picks,
                save_performance_snapshot,
                update_actuals,
            )
        record_daily_picks(result)
        update_stats = update_actuals()
        _LOG.info("Track record updated: %s", update_stats)
        save_performance_snapshot()
    except Exception as e:  # noqa: BLE001
        _LOG.warning("Track record update failed (the main pipeline still finished): %s", e)

    _print_top3_report(result)
    return result


if __name__ == "__main__":
    run_daily_pipeline(force_refresh=False)
