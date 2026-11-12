#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a public demo snapshot from this repository's pipeline output.

The demo repository hosts a read-only Streamlit app that cannot install torch,
transformers or prophet, so it never runs the pipeline. This script turns one
day of pipeline output into the six files that app reads.

Inputs, all produced by src/daily_picker.py:
    data/daily_picks/<date>.json
    data/track_record/recommendations.csv
    data/track_record/performance.json
    data/cache/<date>/<TICKER>.parquet        price history
    data/cache/<date>/<TICKER>_info.json      yfinance info
    data/cache/<date>/sentiment/<TICKER>.json FinBERT scores and the summary
    data/cache/profiles/<TICKER>.json         company profile

Output, written to --out:
    manifest.json  daily.json  universe.json
    prices.csv     recommendations.csv  performance.json

Run from the project root:
    python scripts/export_to_demo.py
    python scripts/export_to_demo.py --date 2026-11-07 --out build/demo_snapshot
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

_log = logging.getLogger("export_to_demo")

# config.AI_LAYERS uses snake_case keys. The dashboard, the README and the demo
# all use these display names, so the export has to translate.
LAYER_DISPLAY = {
    "layer1_infrastructure": "Layer 1 Infrastructure",
    "layer2_cloud": "Layer 2 Cloud & compute",
    "layer3_foundation": "Layer 3 Foundation models",
    "layer4_platform": "Layer 4 Data platforms",
    "layer5_applications": "Layer 5 AI applications",
    "layer6_consumer": "Layer 6 Consumer & content",
}

# Only these keys are copied out of the raw yfinance info blob. Everything else
# is dropped rather than published.
SAFE_INFO = (
    "sector", "industry", "marketCap", "trailingPE", "forwardPE", "priceToBook",
    "returnOnEquity", "profitMargins", "grossMargins", "revenueGrowth",
    "earningsGrowth", "debtToEquity", "currentRatio", "targetMeanPrice",
    "recommendationMean",
)

PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]
PRICE_DAYS = 252

DATA = ROOT / "data"
PICKS_DIR = DATA / "daily_picks"
TRACK_DIR = DATA / "track_record"
CACHE_DIR = DATA / "cache"
PROFILE_DIR = CACHE_DIR / "profiles"


def _read_json(path: Path) -> Any:
    """Return parsed JSON, or None when the file is absent or unreadable."""
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        _log.debug("Could not read %s: %s", path, exc)
        return None


def resolve_date(requested: str | None) -> str:
    """Return the pick date to export.

    Args:
        requested: An explicit YYYY-MM-DD, or None to take the newest file.

    Returns:
        The date string whose daily_picks file exists.

    Raises:
        SystemExit: When no usable pick file is found.
    """
    if requested:
        if (PICKS_DIR / f"{requested}.json").is_file():
            return requested
        raise SystemExit(f"No pick file for {requested} in {PICKS_DIR}")
    files = sorted(PICKS_DIR.glob("20*.json"))
    if not files:
        raise SystemExit(f"No pick files in {PICKS_DIR}. Run src/daily_picker.py first.")
    return files[-1].stem


def layer_of(ticker: str) -> str | None:
    """Return the display name of the AI layer holding this ticker."""
    for key, members in config.AI_LAYERS.items():
        if ticker in members:
            return LAYER_DISPLAY.get(key, key)
    return None


def peer_group_of(ticker: str) -> str | None:
    """Return the SECTOR_MAP peer group holding this ticker."""
    for key, members in config.SECTOR_MAP.items():
        if ticker in members:
            return key
    return None


def build_prices(date_str: str, out: Path) -> int:
    """Write prices.csv from the cached parquet files.

    Args:
        date_str: Cache day to read.
        out: Snapshot directory.

    Returns:
        How many tickers contributed rows.
    """
    day_cache = CACHE_DIR / date_str
    frames: list[pd.DataFrame] = []
    for ticker in config.ALL_STOCKS:
        path = day_cache / f"{ticker}.parquet"
        if not path.is_file():
            _log.warning("No price cache for %s on %s", ticker, date_str)
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001 - a bad file must not stop the export
            _log.warning("Unreadable price cache for %s: %s", ticker, exc)
            continue
        if df.empty:
            continue
        df = df.tail(PRICE_DAYS).reset_index()
        rename = {c: c.lower() for c in df.columns}
        df = df.rename(columns=rename)
        if "date" not in df.columns:
            first = df.columns[0]
            df = df.rename(columns={first: "date"})
        df["ticker"] = ticker
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        missing = [c for c in PRICE_COLUMNS if c not in df.columns]
        if missing:
            _log.warning("%s is missing price columns %s, skipped", ticker, missing)
            continue
        frames.append(df[PRICE_COLUMNS])

    if not frames:
        raise SystemExit("No price data found. Refusing to write an empty prices.csv.")

    prices = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
    prices.to_csv(out / "prices.csv", index=False)
    n = prices["ticker"].nunique()
    _log.info("prices.csv: %s rows, %s tickers", len(prices), n)
    return int(n)


def build_universe(date_str: str, daily: dict[str, Any], out: Path) -> int:
    """Write universe.json, one entry per ticker in config.ALL_STOCKS.

    Scored tickers get their full pipeline output. Tickers the run could not
    score keep a null value and a note saying so, rather than being dropped.

    Args:
        date_str: Pick date.
        daily: Parsed daily_picks file.
        out: Snapshot directory.

    Returns:
        How many tickers carry a composite score.
    """
    missing_label = f"Not in the {date_str} snapshot"
    day_cache = CACHE_DIR / date_str

    scores = {row["ticker"]: row for row in daily.get("all_scores", []) if row.get("ticker")}
    picks = {p["ticker"]: p for p in daily.get("picks", []) if p.get("ticker")}

    entries: list[dict[str, Any]] = []
    scored = 0

    for ticker in config.ALL_STOCKS:
        score = scores.get(ticker)
        pick = picks.get(ticker)
        in_snapshot = score is not None
        if in_snapshot:
            scored += 1

        raw_info = _read_json(day_cache / f"{ticker}_info.json") or {}
        info = {k: raw_info.get(k) for k in SAFE_INFO if raw_info.get(k) is not None}

        sentiment = _read_json(day_cache / "sentiment" / f"{ticker}.json")
        profile = _read_json(PROFILE_DIR / f"{ticker}.json")

        close = None
        mas: dict[str, float] | None = None
        parquet = day_cache / f"{ticker}.parquet"
        if parquet.is_file():
            try:
                df = pd.read_parquet(parquet)
                col = "Close" if "Close" in df.columns else ("close" if "close" in df.columns else None)
                if col and not df.empty:
                    closes = df[col].dropna()
                    close = float(closes.iloc[-1])
                    mas = {
                        f"ma{w}": float(closes.tail(w).mean())
                        for w in (5, 20, 60, 200)
                        if len(closes) >= w
                    } or None
            except Exception as exc:  # noqa: BLE001
                _log.warning("Could not derive price fields for %s: %s", ticker, exc)

        entries.append({
            "ticker": ticker,
            "name": (pick or {}).get("name") or (score or {}).get("name") or ticker,
            "layer": layer_of(ticker),
            "peer_group": peer_group_of(ticker),
            "in_score_snapshot": in_snapshot,
            "composite_score": (score or {}).get("composite_score"),
            "tech_score": (score or {}).get("tech_score"),
            "fund_score": (score or {}).get("fund_score"),
            "valuation_score": (score or {}).get("valuation_score"),
            "prediction_return_pct": (score or {}).get("prediction_return_pct"),
            "prediction_score": ((pick or {}).get("prediction") or {}).get("score"),
            "volume_score": ((pick or {}).get("technical") or {}).get("volume_score"),
            "news_score": (score or {}).get("news_score"),
            "passes_filter": (score or {}).get("passes_filter"),
            "current_price": (pick or {}).get("current_price", close),
            "price_note": None if close is not None else missing_label,
            "info": info,
            "profile": profile,
            "profile_note": "" if profile else missing_label,
            "news": sentiment,
            "news_note": None if sentiment else missing_label,
            "forecast": (pick or {}).get("prediction"),
            "forecast_note": None if pick else missing_label,
            "entry": (pick or {}).get("entry_price"),
            "entry_note": None if pick else missing_label,
            "reasons": (pick or {}).get("pick_reasons", []),
            "risks": (pick or {}).get("risk_factors", []),
            "technical": (pick or {}).get("technical"),
            "fundamental": (pick or {}).get("fundamental"),
            "valuation": (pick or {}).get("valuation"),
            "score_note": None if in_snapshot else missing_label,
            "carried_forward": [],
            "moving_averages": mas,
        })

    payload = {
        "snapshot_date": date_str,
        "weights": dict(config.SCORE_WEIGHTS),
        "missing_label": missing_label,
        "tickers": entries,
    }
    (out / "universe.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    _log.info("universe.json: %s tickers, %s scored", len(entries), scored)
    return scored


def main() -> int:
    """Build the snapshot and report what was written."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Pick date to export (default: newest)")
    parser.add_argument("--out", default="build/demo_snapshot", help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    date_str = resolve_date(args.date)
    daily = _read_json(PICKS_DIR / f"{date_str}.json")
    if not isinstance(daily, dict):
        raise SystemExit(f"daily_picks/{date_str}.json is missing or unreadable")

    out = (ROOT / args.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    (out / "daily.json").write_text(
        json.dumps(daily, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    for name in ("recommendations.csv", "performance.json"):
        src = TRACK_DIR / name
        if not src.is_file():
            raise SystemExit(f"Missing {src}. Run src/daily_picker.py first.")
        shutil.copy2(src, out / name)

    n_prices = build_prices(date_str, out)
    n_scored = build_universe(date_str, daily, out)

    manifest = {
        "snapshot_id": "latest",
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "pick_date": date_str,
        "source": "Full pipeline run: Yahoo Finance prices and fundamentals, "
                  "FinBERT sentiment, Prophet forecasts, Claude summaries",
        "updated_tickers": n_prices,
        "scored_tickers": n_scored,
        "notes": "Every field is regenerated by the scheduled pipeline. "
                 "Tickers the run could not score carry a note instead of a value.",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    _log.info("Snapshot written to %s", out)
    for path in sorted(out.iterdir()):
        _log.info("  %-22s %8.1f KB", path.name, path.stat().st_size / 1024)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
