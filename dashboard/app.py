# -*- coding: utf-8 -*-
"""
US stock analysis dashboard (Streamlit).

Run from the project root:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import html
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Allow the dashboard to import project modules directly
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import AI_LAYERS, ALL_STOCKS, PICKS_DIR, SCORE_WEIGHTS, SECTOR_MAP, TICKER_LIST  # noqa: E402
from data_collector import fetch_spy_quote, fetch_stock_data  # noqa: E402
from entry_price import calculate_entry_prices, get_status_label_cn  # noqa: E402
from profile_fetcher import get_profile  # noqa: E402
from fundamentals import analyze_fundamentals, get_stock_sector  # noqa: E402
from indicators import calculate_indicators, calculate_tech_score  # noqa: E402
from predictor import predict_price  # noqa: E402
from sentiment import analyze_ticker_sentiment  # noqa: E402
from track_record import PERFORMANCE_JSON, RECOMMENDATIONS_CSV, calculate_performance  # noqa: E402
from valuation import calculate_fair_value  # noqa: E402

st.set_page_config(page_title="US Stock Analysis Dashboard", page_icon="📈", layout="wide")

# Same hex values as the CSS custom properties. Plotly cannot read var(--token).
ACCENT = "#15385C"
TEXT = "#10141A"
TEXT_DIM = "#5B6472"
BORDER = "#E3E6EA"
BAR_TRACK = "#ECEFF2"
POSITIVE = "#0E7C5A"
NEGATIVE = "#B3261E"

# Categorical series colours. Built from the non-directional tokens that stay
# visible on white. Positive and negative are not in this sequence.
CHART_COLORS = (
    ACCENT,
    TEXT_DIM,
    TEXT,
)

# Figures in running text. Applied to the raw string, before HTML escaping.
_NUMBER_RE = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")

st.markdown(
    """
<style>
@import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap");

:root {
    --bg: #FFFFFF;
    --surface: #F7F8FA;
    --border: #E3E6EA;
    --text: #10141A;
    --text-dim: #5B6472;
    --accent: #15385C;
    --positive: #0E7C5A;
    --negative: #B3261E;
    --bar-track: #ECEFF2;
}

html, body, .stApp, .stApp * {
    font-family: "IBM Plex Sans", sans-serif !important;
    font-variant-numeric: tabular-nums;
}
.stApp .num,
.stApp .num *,
.stApp .metric-value,
.stApp .metric-value *,
.stApp [data-testid="stMetricValue"],
.stApp [data-testid="stMetricValue"] *,
.stApp [data-testid="stMetricDelta"],
.stApp [data-testid="stMetricDelta"] *,
.stApp .data-table td.numcol,
.stApp .data-table td.numcol *,
.stApp code, .stApp pre, .stApp kbd, .stApp samp {
    font-family: "IBM Plex Mono", monospace !important;
}

.stApp {
    background: var(--bg);
    color: var(--text);
    font-size: 14px;
    font-weight: 400;
    line-height: 1.45;
}
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
#MainMenu,
[data-testid="stMainMenu"],
footer,
[data-testid="stFooter"] {
    display: none !important;
    height: 0 !important;
}
.block-container {
    padding-top: 16px !important;
    padding-bottom: 16px !important;
    padding-left: 16px !important;
    padding-right: 16px !important;
}
[data-testid="stVerticalBlock"] {gap: 4px !important;}
.element-container,
[data-testid="stElementContainer"],
[data-testid="stMarkdown"] {margin: 0 !important;}
[data-testid="stMarkdown"] p,
[data-testid="stMarkdown"] li,
.body-line {
    font-size: 14px;
    font-weight: 400;
    line-height: 1.45;
    color: var(--text);
    margin: 0 0 4px 0;
}
h1, [data-testid="stHeading"] h1 {
    font-size: 24px !important;
    font-weight: 600 !important;
    line-height: 1.25 !important;
    color: var(--text) !important;
    margin: 0 0 4px 0 !important;
    padding: 0 !important;
}
h2, h3, h4, h5, h6,
.card-title {
    font-size: 16px !important;
    font-weight: 600 !important;
    line-height: 1.3 !important;
    color: var(--text) !important;
    margin: 0 0 4px 0 !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    font-size: 22px !important;
    font-weight: 500 !important;
    line-height: 1.2 !important;
    color: var(--text) !important;
}
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] label,
[data-testid="stMetricLabel"] span,
.metric-label,
.score-caption,
.data-table th {
    font-family: "IBM Plex Sans", sans-serif !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    line-height: 1.3 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: var(--text-dim) !important;
}
.metric-value {
    font-size: 22px !important;
    font-weight: 500 !important;
    line-height: 1.2 !important;
    color: var(--text);
}
.caption,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    font-size: 12px !important;
    font-weight: 400 !important;
    line-height: 1.45 !important;
    color: var(--text-dim) !important;
    text-transform: none !important;
    letter-spacing: normal !important;
    margin: 0 0 4px 0 !important;
}
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    font-size: 14px !important;
    font-weight: 400 !important;
    color: var(--text) !important;
    text-transform: none !important;
    letter-spacing: normal !important;
}
[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Up"]),
[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Up"]) * {
    color: var(--positive) !important;
}
[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Down"]),
[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Down"]) * {
    color: var(--negative) !important;
}
.pos {color: var(--positive) !important;}
.neg {color: var(--negative) !important;}

section[data-testid="stSidebar"] {
    background: var(--bg) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] [data-testid="stSidebarContent"],
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
    background: var(--bg) !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 16px !important;
    background: var(--bg) !important;
    box-shadow: none !important;
    margin: 0 0 12px 0 !important;
}
.card, .callout {
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px;
    background: var(--bg);
    box-shadow: none;
    margin: 0 0 12px 0;
}
.callout {background: var(--surface);}
.dim {color: var(--text-dim); font-weight: 400;}

.score-block {margin: 4px 0 8px 0;}
.score-track {
    height: 4px;
    width: 100%;
    background: var(--bar-track);
    border-radius: 2px;
    overflow: hidden;
}
.score-fill {
    height: 4px;
    background: var(--accent);
    border-radius: 2px;
}
.price-line {
    display: flex;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 8px;
    margin: 0 0 4px 0;
}
.status-badge {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 2px;
    font-size: 12px;
    font-weight: 500;
    line-height: 1.45;
    color: var(--bg);
    background: var(--text-dim);
    text-transform: none;
    letter-spacing: normal;
}
.status-buy_now {background: var(--accent);}
.status-watch {
    background: var(--surface);
    color: var(--accent);
    border: 1px solid var(--accent);
}
.status-wait {background: var(--text-dim);}
.status-avoid {background: var(--text);}
.page-footer {
    color: var(--text-dim);
    font-size: 12px;
    line-height: 1.45;
    margin: 12px 0 0 0;
}
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    height: 0 !important;
    margin: 12px 0 !important;
    background: transparent !important;
}
.table-wrap {overflow-x: auto; margin: 4px 0 8px 0;}
table.data-table {
    width: 100%;
    border-collapse: collapse;
    border: none;
    background: transparent;
}
table.data-table th {
    text-align: left;
    padding: 4px 8px;
    border: none;
    border-bottom: 1px solid var(--border);
    background: transparent;
}
table.data-table th.numcol,
table.data-table td.numcol {text-align: right;}
table.data-table td {
    font-size: 14px;
    font-weight: 400;
    color: var(--text);
    text-align: left;
    padding: 4px 8px;
    border: none;
    border-bottom: 1px solid var(--border);
    background: transparent;
}
.reason-list {margin: 0 0 4px 16px; padding: 0;}
.stButton > button,
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-primary"] {
    border-radius: 6px !important;
    box-shadow: none !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}
[data-testid="stExpander"] details {
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    box-shadow: none !important;
    background: var(--bg) !important;
}
[data-testid="stAlert"] {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    box-shadow: none !important;
}
button[data-baseweb="tab"] {
    font-size: 14px !important;
    font-weight: 500 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# Display-only TWD equivalent (fixed rate, no FX API calls)
USD_TWD = 32.0

# Further-reading groups on the company profile, in display order
ARTICLE_TYPE_ORDER = ("official", "ir", "filing", "reference")
ARTICLE_TYPE_LABEL: dict[str, str] = {
    "official": "Official website",
    "ir": "Investor relations",
    "filing": "SEC and regulatory filings",
    "reference": "References",
}


@st.cache_data(ttl=86400)
def load_profile(ticker: str) -> dict[str, Any] | None:
    """Company profile (1-day dashboard cache; the fetcher caches the file for 90 days)."""
    if not (ticker or str(ticker).strip()):
        return None
    return get_profile(str(ticker).strip().upper(), force_refresh=False)


def _profile_looks_valid(profile: Any) -> bool:
    if not profile or not isinstance(profile, dict):
        return False
    if profile.get("name") is not None and str(profile.get("name", "")).strip():
        return True
    if profile.get("description_short") and str(profile.get("description_short", "")).strip():
        return True
    if profile.get("ticker") and (profile.get("description_long") or str(profile.get("name") or "")):
        return True
    return False


def _format_mcap_usd_twd(mcap: Any) -> tuple[str, str]:
    """Market cap: USD plus a TWD estimate using USD_TWD."""
    if mcap is None:
        return "—", "—"
    try:
        m = float(mcap)
    except (TypeError, ValueError):
        return "—", "—"
    if m <= 0 or m != m:  # nan
        return "—", "—"
    a = abs(m)
    usd = a
    if a >= 1_000_000_000_000.0:
        ustr = f"${usd / 1_000_000_000_000.0:,.2f}T"
    elif a >= 1_000_000_000.0:
        ustr = f"${usd / 1_000_000_000.0:,.2f}B"
    elif a >= 1_000_000.0:
        ustr = f"${usd / 1_000_000.0:,.2f}M"
    else:
        ustr = f"${usd:,.0f}"
    ntd = m * USD_TWD
    if ntd >= 1_000_000_000_000.0:
        tw = f"approx. NT$ {ntd / 1_000_000_000_000.0:,.1f} trillion"
    elif ntd >= 100_000_000.0:
        tw = f"approx. NT$ {ntd / 100_000_000.0:,.0f} hundred million"
    else:
        tw = f"approx. NT$ {ntd / 10_000.0:,.0f} ×10,000" if ntd >= 10_000 else f"approx. NT$ {ntd:,.0f}"
    return ustr, f"{tw} (1 USD ≈ {USD_TWD:.0f} TWD)"


def _competitor_to_ticker(label: str) -> str | None:
    """
    Map a competitor phrase to a project ticker.
    Returns the ticker when it matches the universe, otherwise None.
    """
    if not label or not str(label).strip():
        return None
    s = str(label).strip()
    u = s.upper()
    for tok in s.replace("(", " ").replace(")", " ").replace(",", " ").split():
        t0 = tok.strip("(),.").upper()
        if t0 in ALL_STOCKS:
            return t0
    if u in ALL_STOCKS or u.replace(" ", "") in ALL_STOCKS:
        return u if u in ALL_STOCKS else u.replace(" ", "")
    for t, en_name in ALL_STOCKS.items():
        if not en_name:
            continue
        nlow = en_name.lower()
        slow = s.lower()
        if nlow in slow or (len(s) >= 4 and slow in nlow):
            return t
    for t in ALL_STOCKS:
        if t in u or t in s.upper():
            return t
    return None


@st.cache_data(ttl=3600)
def load_latest_daily_json() -> dict[str, Any] | None:
    """Load the latest daily_picks JSON."""
    picks_dir = Path(PICKS_DIR)
    files = sorted(picks_dir.glob("*.json"))
    if not files:
        return None
    latest = files[-1]
    with open(latest, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=1800)
def load_performance() -> dict[str, Any] | None:
    """
    Load the strategy performance snapshot.
    Prefer performance.json; compute it on the spot when the file is missing.

    Returns:
        Performance dict, or None when both the read and the calculation fail.
    """
    path = Path(PERFORMANCE_JSON)
    if path.is_file():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:  # noqa: BLE001
            pass  # Fall through to a live calculation when the snapshot is corrupt
    try:
        return calculate_performance()
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(ttl=1800)
def load_recommendations() -> pd.DataFrame:
    """Load the recommendations CSV for the full history table on the performance page."""
    path = Path(RECOMMENDATIONS_CSV)
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_stock_bundle(ticker: str) -> dict[str, Any]:
    """Fetch one stock and combine the four analyses."""
    raw = fetch_stock_data(ticker, force_refresh=False)
    if not raw:
        raise ValueError(f"{ticker} has no data")
    ph = raw["price_history"]
    info = raw.get("info", {}) or {}
    current_price = float(raw.get("current_price") or ph["Close"].iloc[-1])
    df_ind = calculate_indicators(ph)
    tech = calculate_tech_score(df_ind)
    fund = analyze_fundamentals(ticker, info)
    val = calculate_fair_value(ticker, info, current_price)
    pred = predict_price(ticker, ph)
    return {
        "raw": raw,
        "df_ind": df_ind,
        "technical": tech,
        "fundamental": fund,
        "valuation": val,
        "prediction": pred,
    }


@st.cache_data(ttl=3600)
def get_cached_spy_quote() -> tuple[float | None, float | None]:
    """Cached SPY close and prior-session percent change. (None, None) when unavailable."""
    try:
        return fetch_spy_quote(force_refresh=False)
    except Exception:  # noqa: BLE001
        return None, None


@st.cache_data(ttl=3600)
def get_cached_current_price(ticker: str) -> float | None:
    """Cached current price via the data_collector same-day cache."""
    raw = fetch_stock_data(ticker, force_refresh=False)
    if not raw:
        return None
    cp = raw.get("current_price")
    if cp is None:
        return None
    try:
        return float(cp)
    except (TypeError, ValueError):
        return None


@st.cache_data(ttl=3600)
def get_sector_fundamental_avg(ticker: str) -> dict[str, float]:
    """Average fundamental sub-scores for the same sector, for the radar chart."""
    sector = get_stock_sector(ticker)
    peers = SECTOR_MAP.get(sector, [])
    if not peers:
        return {"growth": 50, "profitability": 50, "financial_health": 50, "industry_outlook": 50}
    rows = []
    for p in peers[:15]:
        raw = fetch_stock_data(p, force_refresh=False)
        if not raw:
            continue
        info = raw.get("info", {}) or {}
        res = analyze_fundamentals(p, info)
        rows.append(res["sub_scores"])
    if not rows:
        return {"growth": 50, "profitability": 50, "financial_health": 50, "industry_outlook": 50}
    df = pd.DataFrame(rows)
    return {k: float(df[k].mean()) for k in ["growth", "profitability", "financial_health", "industry_outlook"]}


def _style_figure(fig: go.Figure, **layout: Any) -> go.Figure:
    """Apply the shared light template, type, and categorical palette.

    Chart titles are cleared. The page heading above the figure is the title.
    Gridlines stay on the value axis only.

    Args:
        fig: Plotly figure to style.
        **layout: Extra layout settings passed through unchanged.

    Returns:
        The same figure, after the theme is applied.
    """
    layout.pop("title", None)
    layout.pop("title_text", None)
    fig.update_layout(
        template="plotly_white",
        colorway=list(CHART_COLORS),
        font=dict(family="IBM Plex Sans, sans-serif", size=12, color=TEXT),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        title_text="",
        margin=dict(l=48, r=16, t=8, b=32),
        **layout,
    )
    fig.update_layout(title_text="")
    fig.update_annotations(font=dict(family="IBM Plex Sans, sans-serif", size=12, color=TEXT))
    kinds = {getattr(trace, "type", None) for trace in fig.data}
    if "scatterpolar" in kinds:
        fig.update_polars(
            radialaxis=dict(
                gridcolor=BORDER,
                linecolor=BORDER,
                showline=False,
                tickfont=dict(family="IBM Plex Sans, sans-serif", size=12, color=TEXT),
            ),
            angularaxis=dict(showgrid=False, showline=False),
            bgcolor="#FFFFFF",
        )
    elif "indicator" not in kinds:
        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            showline=False,
            mirror=False,
            linewidth=0,
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor=BORDER,
            gridwidth=1,
            zeroline=True,
            zerolinecolor=BORDER,
            zerolinewidth=1,
            showline=False,
            mirror=False,
            linewidth=0,
        )
    return fig


def pct_color(v: float | None) -> str:
    """CSS color for a signed figure. None uses the dim text token."""
    if v is None:
        return "var(--text-dim)"
    return "var(--positive)" if v >= 0 else "var(--negative)"


def get_sentiment_color(label: str) -> str:
    """CSS color for a news-sentiment label. Direction uses the palette tokens."""
    return {
        "Strongly Bullish": "var(--positive)",
        "Bullish": "var(--positive)",
        "Neutral": "var(--text-dim)",
        "Bearish": "var(--negative)",
        "Strongly Bearish": "var(--negative)",
        "No News": "var(--text-dim)",
    }.get(label, "var(--text-dim)")


def _sentiment_hex(label: str) -> str:
    """Hex direction color for Plotly, which cannot read CSS variables.

    Args:
        label: Sentiment label such as Bullish or Bearish.

    Returns:
        Positive, negative, or dim text hex from the palette.
    """
    if label in {"Strongly Bullish", "Bullish"}:
        return POSITIVE
    if label in {"Bearish", "Strongly Bearish"}:
        return NEGATIVE
    return TEXT_DIM


def _truncate_text(text: str, max_len: int = 80) -> str:
    """Truncate overly long text."""
    s = str(text or "")
    return s if len(s) <= max_len else s[:max_len].rstrip() + "..."


def _fmt_news_time(value: Any) -> str:
    """Format a timestamp as YYYY-MM-DD HH:MM."""
    try:
        dt = pd.to_datetime(value)
        if pd.isna(dt):
            return str(value or "")
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(value or "")


def _find_pick_news(daily: dict[str, Any], ticker: str) -> dict[str, Any] | None:
    """Find one ticker's news block inside the daily picks."""
    for p in daily.get("picks", []):
        if str(p.get("ticker", "")).upper() == ticker.upper():
            n = p.get("news")
            return n if isinstance(n, dict) else None
    return None


@st.cache_data(ttl=3600)
def get_live_ticker_sentiment(ticker: str) -> dict[str, Any]:
    """Live sentiment analysis (cached for 1 hour)."""
    return analyze_ticker_sentiment(ticker)


def _volume_score_dashboard(vol_ratio: float | None) -> float:
    """Volume score matching daily_picker, used in the composite."""
    if vol_ratio is None:
        return 35.0
    if vol_ratio > 1.5:
        return 80.0
    if vol_ratio >= 1.0:
        return 70.0
    if vol_ratio > 0.5:
        return 55.0
    return 35.0


def _to_float_safe(value: Any, default: float | None = None) -> float | None:
    """Return a finite float or the supplied fallback.

    Pandas represents missing numeric cells as ``NaN``.  Treating that as a
    value leaks strings such as ``+nan%`` into the dashboard, so only finite
    numbers are valid display inputs.
    """
    try:
        if value is None:
            return default
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


@st.cache_data(ttl=3600)
def compute_entry_analysis(ticker: str) -> dict[str, Any]:
    """
    Build a composite close to daily_picker, then call calculate_entry_prices.
    Used by the stock page and the overview table (result is cached).
    """
    sym = str(ticker).strip().upper()
    try:
        b = get_stock_bundle(sym)
        raw = b["raw"]
        ph = raw["price_history"]
        cp = float(raw.get("current_price") or ph["Close"].iloc[-1])
        tech, fund, val, pred = b["technical"], b["fundamental"], b["valuation"], b["prediction"]
        news = get_live_ticker_sentiment(sym)
        vol_ratio = _to_float_safe(tech.get("current_values", {}).get("vol_ratio"))
        vol_score = _volume_score_dashboard(vol_ratio)
        composite = (
            (_to_float_safe(tech.get("total_score")) or 0.0) * 0.25
            + (_to_float_safe(fund.get("total_score")) or 0.0) * 0.25
            + (_to_float_safe(val.get("score")) or 0.0) * 0.20
            + (_to_float_safe(pred.get("score")) or 0.0) * 0.10
            + vol_score * 0.10
            + (_to_float_safe(news.get("score")) or 50.0) * 0.10
        )
        return calculate_entry_prices(
            {
                "ticker": sym,
                "current_price": cp,
                "price_history": ph,
                "technical": tech,
                "fundamental": fund,
                "valuation": val,
                "prediction": pred,
                "composite_score": round(float(composite), 2),
            }
        )
    except Exception:
        return {}


# One sentence each, taken from the scoring functions. High means a stronger reading.
_SCORE_HELP = {
    "technical": (
        "Weighted blend of RSI, MACD, Bollinger position, moving-average alignment, and volume. "
        "A high value means those signals currently agree on a stronger setup."
    ),
    "fundamental": (
        "Weighted blend of growth, profitability, financial health, and industry outlook. "
        "A high value means the reported fundamentals look stronger."
    ),
    "valuation": (
        "Maps estimated upside versus a blended fair value onto 0-100. "
        "A high value means more estimated upside; an implausible cyclical reading is scored 50."
    ),
    "forecast": (
        "Maps the 30-day expected return and the model's confidence onto 0-100. "
        "A high value means the forecast points to a larger gain."
    ),
    "news": (
        "Maps time-weighted FinBERT headline sentiment from -1 to 1 onto 0-100. "
        "A high value means recent headlines lean positive."
    ),
    "growth": (
        "Scores revenue growth and earnings growth together. "
        "A high value means both are growing quickly."
    ),
    "profitability": (
        "Scores gross margin, net margin, and return on equity. "
        "A high value means margins and return on equity are stronger."
    ),
    "financial_health": (
        "Scores debt-to-equity and the current ratio. "
        "A high value means lower leverage and more liquidity."
    ),
    "industry_outlook": (
        "Starts from the peer-group base score and adds a bonus when forward P/E is below trailing P/E. "
        "A high value means that peer group is favored and earnings are expected to grow."
    ),
}

_WEIGHT_LABELS = {
    "technical": "Technical",
    "fundamental": "Fundamental",
    "valuation": "Valuation",
    "prediction": "Forecast",
    "volume": "Volume",
    "news": "News",
}


def _status_badge(status: str) -> str:
    """HTML badge for an entry status. The text is the existing status label.

    Args:
        status: Entry status key such as buy_now, watch, wait, or avoid.

    Returns:
        A span whose text is the status label and whose class marks the key.
    """
    key = str(status or "watch")
    label = html.escape(get_status_label_cn(key))
    safe_key = html.escape(key)
    return f"<span class='status-badge status-{safe_key}'>{label}</span>"


def _signed_html(value: float | None, digits: int = 1, *, prominent: bool = False) -> str:
    """Color a signed percent by direction. None becomes an em dash.

    Args:
        value: Signed number, already in percent points.
        digits: Decimal places.
        prominent: When True, render at the metric-value size.

    Returns:
        HTML for the signed figure.
    """
    if value is None:
        return "<span class='num'>—</span>"
    klass = "pos" if value >= 0 else "neg"
    size = " metric-value" if prominent else ""
    return f"<span class='num{size} {klass}'>{value:+.{digits}f}%</span>"


def _mono_numbers(text: str) -> str:
    """Escape text and wrap numbers in a monospace span.

    Args:
        text: Plain text that may contain figures.

    Returns:
        HTML with each number inside a span of class ``num``.
        Apostrophes stay intact because the text is escaped around the numbers.
    """
    # Escape text around the numbers. Escaping first would split entities such as &#x27;.
    raw = str(text)
    parts: list[str] = []
    last = 0
    for match in _NUMBER_RE.finditer(raw):
        parts.append(html.escape(raw[last:match.start()]))
        parts.append(f"<span class='num'>{html.escape(match.group(0))}</span>")
        last = match.end()
    parts.append(html.escape(raw[last:]))
    return "".join(parts)


def _caption(text: str, *, sidebar: bool = False, target: Any = None) -> None:
    """Caption at the design-system size, with figures in IBM Plex Mono.

    Args:
        text: Caption copy.
        sidebar: When True, render in the sidebar instead of the main column.
        target: Optional Streamlit container. Overrides ``sidebar``.
    """
    if target is not None:
        host = target
    elif sidebar:
        host = st.sidebar
    else:
        host = st
    host.markdown(
        f"<p class='caption'>{_mono_numbers(text)}</p>",
        unsafe_allow_html=True,
    )


def _callout(text: str) -> None:
    """Quiet note. Keeps red and green reserved for direction.

    Args:
        text: Note copy. Figures are rendered in IBM Plex Mono.
    """
    st.markdown(
        f"<div class='callout'>{_mono_numbers(text)}</div>",
        unsafe_allow_html=True,
    )


def _score_bar(value: Any, label: str) -> None:
    """Draw a flat 4px score bar. The printed score is the raw value.

    Args:
        value: Score, expected on a 0-100 scale.
        label: Short name shown beside the score, for example ``Composite``.
    """
    number = _to_float_safe(value)
    width = 0.0 if number is None else min(max(number, 0.0), 100.0)
    shown = "—" if value is None else str(value)
    st.markdown(
        "<div class='score-block'>"
        f"<div class='score-caption'>{html.escape(label)} "
        f"<span class='num'>{html.escape(shown)}</span> / <span class='num'>100</span></div>"
        "<div class='score-track'>"
        f"<div class='score-fill' style='width:{width:.4f}%;'></div>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def _cell_text(value: Any) -> str:
    """Format one table cell without rounding a numeric value.

    Args:
        value: Cell value from a display dataframe.

    Returns:
        A plain string. Missing values become an em dash.
    """
    if value is None or isinstance(value, bool):
        return "—" if value is None else str(value)
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "—"
    return format(number, ".12g")


def _looks_numeric(text: str) -> bool:
    """Return True when a cell is a figure, price, or percent.

    Args:
        text: Display text for one cell.

    Returns:
        True when the cell parses as a number after currency and percent signs.
    """
    cleaned = text.strip().replace(",", "").replace("$", "").replace("%", "").replace("+", "")
    if cleaned in {"", "—", "-", "–"}:
        return False
    try:
        float(cleaned)
    except ValueError:
        return False
    return True


def _render_table(df: pd.DataFrame) -> None:
    """Render a hairline table. Numeric columns are right-aligned in IBM Plex Mono.

    Args:
        df: Display dataframe. The index is not shown.
    """
    if df is None or df.empty:
        return
    columns = [str(col) for col in df.columns]
    body: list[list[str]] = []
    for row in df.itertuples(index=False):
        body.append([_cell_text(value) for value in row])
    numeric: list[bool] = []
    for col_idx in range(len(columns)):
        usable = [body[r][col_idx] for r in range(len(body)) if body[r][col_idx] not in {"", "—"}]
        if not usable:
            numeric.append(False)
            continue
        hits = sum(1 for value in usable if _looks_numeric(value))
        numeric.append(hits / len(usable) >= 0.6)

    header_cells = []
    for name, is_num in zip(columns, numeric, strict=True):
        klass = " class='numcol'" if is_num else ""
        header_cells.append(f"<th{klass}>{html.escape(name)}</th>")
    rows_html: list[str] = []
    for row in body:
        cells = []
        for value, is_num in zip(row, numeric, strict=True):
            klass = " class='numcol'" if is_num else ""
            cells.append(f"<td{klass}>{html.escape(value)}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
    st.markdown(
        "<div class='table-wrap'><table class='data-table'><thead><tr>"
        + "".join(header_cells)
        + "</tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table></div>",
        unsafe_allow_html=True,
    )


def _weight_html() -> str:
    """Format config.SCORE_WEIGHTS with percentages in IBM Plex Mono.

    Returns:
        HTML labels and percentages in config order, separated by middle dots.
    """
    parts = [
        f"{html.escape(_WEIGHT_LABELS.get(key, key))} <span class='num'>{float(weight):.0%}</span>"
        for key, weight in SCORE_WEIGHTS.items()
    ]
    return " · ".join(parts)


def _page_footer(date_str: str | None) -> None:
    """Quiet research disclaimer. The date is the snapshot date already on the page.

    Args:
        date_str: Snapshot date, or None when no snapshot is loaded.
    """
    shown = date_str or "the date shown"
    st.markdown(
        "<p class='page-footer'>"
        + _mono_numbers(
            f"Research project, not investment advice. Data is a snapshot taken on {shown}."
        )
        + "</p>",
        unsafe_allow_html=True,
    )


def _render_sidebar(daily: dict[str, Any] | None) -> str:
    """Sidebar navigation, a one-line description, and a short glossary.

    Args:
        daily: Latest daily snapshot, used only for the data date.

    Returns:
        The selected page id.
    """
    st.sidebar.markdown("<p class='card-title'>Pages</p>", unsafe_allow_html=True)
    _caption("US equity AI-sector ranking of 60 listed companies.", sidebar=True)
    page_options = ["Daily Picks", "Stock Deep Dive", "Universe Overview", "Strategy Performance"]
    default_page = st.session_state.get("page", "Daily Picks")
    page = st.sidebar.radio(
        "Page",
        page_options,
        index=page_options.index(default_page) if default_page in page_options else 0,
    )
    st.session_state["page"] = page
    with st.sidebar.expander("How to read this"):
        st.markdown(
            "<p class='body-line'>"
            + _mono_numbers(
                "Every score runs from 0 to 100. A higher score is a stronger reading "
                "on that dimension, not a predicted return."
            )
            + "</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p class='body-line'><strong>Composite weights.</strong> {_weight_html()}</p>",
            unsafe_allow_html=True,
        )
        for label, key in (
            ("Technical", "technical"),
            ("Fundamental", "fundamental"),
            ("Valuation", "valuation"),
            ("Forecast", "forecast"),
        ):
            st.markdown(
                f"<p class='body-line'><strong>{label}.</strong> {_mono_numbers(_SCORE_HELP[key])}</p>",
                unsafe_allow_html=True,
            )
        st.markdown(
            "<p class='body-line'><strong>Volume.</strong> "
            + _mono_numbers(
                "Compares today's volume with its recent average and maps that ratio onto 0-100. "
                "A high value means heavier-than-usual trading. This weight is inside the composite; "
                "the pick card does not store it as its own number."
            )
            + "</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p class='body-line'><strong>News.</strong> {_mono_numbers(_SCORE_HELP['news'])}</p>",
            unsafe_allow_html=True,
        )
    date_shown = daily.get("date", "-") if isinstance(daily, dict) else "-"
    _caption(f"Data date: {date_shown}", sidebar=True)
    return page


def render_daily_page(daily: dict[str, Any]) -> None:
    """Page 1: daily picks."""
    st.title("Daily picks")
    date_str = daily.get("date", datetime.now().strftime("%Y-%m-%d"))
    m = daily.get("market_overview", {})
    spy_chg = m.get("spy_change_pct")
    spy_price = m.get("spy_price")
    if spy_price is None or spy_chg is None:
        spy_price, spy_chg = get_cached_spy_quote()
    if spy_price is not None and spy_chg is not None:
        spy_txt = (
            f"SPY <span class='num'>{float(spy_price):.2f}</span> "
            f"({_signed_html(float(spy_chg), digits=2)})"
        )
    else:
        spy_txt = "SPY N/A"
    analyzed = int(m.get("total_analyzed", 0) or 0)
    passed = int(m.get("passed_filter", 0) or 0)
    st.markdown(
        f"<p class='caption'>Date: <span class='num'>{html.escape(str(date_str))}</span> | {spy_txt} | "
        f"analyzed <span class='num'>{analyzed}</span>, passed filter <span class='num'>{passed}</span></p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='caption'>Ranks <span class='num'>60</span> US-listed AI and technology companies "
        "each trading day and surfaces the top <span class='num'>3</span>. "
        "Every score is <span class='num'>0</span>-<span class='num'>100</span>. "
        "The composite is a weighted blend of these dimensions: "
        + _weight_html()
        + ".</p>",
        unsafe_allow_html=True,
    )

    picks = daily.get("picks", [])[:3]
    if not picks:
        _callout("No data yet")
    for i, p in enumerate(picks):
        name = p.get("name", "")
        ticker = p.get("ticker", "")
        curr = p.get("current_price")
        pred_ret = _to_float_safe(p.get("prediction", {}).get("expected_return_pct"))
        news = p.get("news", {}) if isinstance(p.get("news"), dict) else {}
        sent_label = str(news.get("sentiment_label", "No News"))
        news_count = int(news.get("news_count", 0) or 0)
        sent_color = get_sentiment_color(sent_label)
        with st.container(border=True):
            st.markdown(f"### {html.escape(str(ticker))} {html.escape(str(name))}")
            price_txt = f"${float(curr):.2f}" if curr is not None else "—"
            st.markdown(
                "<p class='price-line'><span class='metric-label'>Price</span>"
                f"<span class='metric-value'>{html.escape(price_txt)}</span>"
                "<span class='metric-label'>30-day forecast</span>"
                f"{_signed_html(pred_ret, prominent=True)}</p>",
                unsafe_allow_html=True,
            )
            _score_bar(p.get("composite_score"), "Composite")

            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Technical", p.get("technical", {}).get("total_score"), help=_SCORE_HELP["technical"])
            s2.metric("Fundamental", p.get("fundamental", {}).get("total_score"), help=_SCORE_HELP["fundamental"])
            s3.metric("Valuation", p.get("valuation", {}).get("score"), help=_SCORE_HELP["valuation"])
            s4.metric("Forecast", p.get("prediction", {}).get("score"), help=_SCORE_HELP["forecast"])
            s5.metric("News", news.get("score"), help=_SCORE_HELP["news"])
            card_val = p.get("valuation") if isinstance(p.get("valuation"), dict) else {}
            if card_val.get("valuation_reliable") is False:
                _callout(
                    str(card_val.get("valuation_reason") or "").strip()
                    or "Trailing P/E implies an implausible fair value; earnings are likely at a cyclical peak or trough"
                )

            st.markdown(
                "<p class='body-line'>News sentiment: "
                f"<span style='color:{sent_color};font-weight:500;'>{html.escape(sent_label)}</span>"
                f" · <span class='num'>{news_count}</span> articles</p>",
                unsafe_allow_html=True,
            )

            reasons = p.get("pick_reasons") or []
            if reasons:
                st.markdown("<p class='card-title'>Reasons</p>", unsafe_allow_html=True)
                items = "".join(f"<li>{_mono_numbers(str(reason))}</li>" for reason in reasons)
                st.markdown(f"<ul class='reason-list'>{items}</ul>", unsafe_allow_html=True)
            else:
                _caption("No reasons recorded")

            risks = p.get("risk_factors", [])
            if risks:
                _callout("; ".join(str(item) for item in risks))

            entry = p.get("entry_price") if isinstance(p.get("entry_price"), dict) else {}
            if entry:
                status = str(entry.get("current_status") or "watch")
                st.markdown(f"Entry {_status_badge(status)}", unsafe_allow_html=True)
                ep = entry.get("entry_prices") or {}
                dist = entry.get("distance") or {}
                _render_table(
                    pd.DataFrame(
                        {
                            "Level": ["Conservative", "Standard", "Aggressive"],
                            "Price": [
                                f"${float(ep.get('conservative') or 0):.2f}",
                                f"${float(ep.get('standard') or 0):.2f}",
                                f"${float(ep.get('aggressive') or 0):.2f}",
                            ],
                            "Distance": [
                                f"{float(dist.get('to_conservative_pct') or 0):+.1f}%",
                                f"{float(dist.get('to_standard_pct') or 0):+.1f}%",
                                f"{float(dist.get('to_aggressive_pct') or 0):+.1f}%",
                            ],
                        }
                    )
                )
                _callout(str(entry.get("reasoning") or ""))
            else:
                _caption("No entry-price analysis yet (rerun daily_picker to refresh the report)")

            top_news = news.get("top_headlines", []) if isinstance(news.get("top_headlines"), list) else []
            with st.expander(f"Related news ({news_count})"):
                sum_raw = news.get("summary")
                if sum_raw:
                    s = str(sum_raw)
                    summary_preview = s[:200]
                    if len(s) > 200:
                        summary_preview += "..."
                    _caption(summary_preview)
                    st.markdown("---")
                if not top_news:
                    _caption("No related news yet")
                for one in top_news[:3]:
                    title = _truncate_text(str(one.get("title", "")), 80)
                    sent = str(one.get("sentiment_cn", one.get("sentiment", "Neutral")))
                    conf = float(one.get("confidence", 0) or 0)
                    link = str(one.get("link", "") or "")
                    st.markdown(f"<p class='body-line'><strong>{html.escape(title)}</strong></p>", unsafe_allow_html=True)
                    _caption(f"{sent} · confidence {conf:.2f}")
                    if link:
                        st.link_button("Read original", link, use_container_width=False)

            if st.button("View full analysis", key=f"detail_{ticker}"):
                st.session_state["selected_ticker"] = ticker
                st.session_state["page"] = "Stock Deep Dive"
                st.rerun()
        if i < len(picks) - 1:
            st.divider()


def build_technical_figure(df: pd.DataFrame) -> go.Figure:
    """Multi-panel technical chart."""
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.52, 0.16, 0.16, 0.16],
    )
    inc_color, dec_color = POSITIVE, NEGATIVE
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            increasing_line_color=inc_color,
            decreasing_line_color=dec_color,
            increasing_fillcolor=inc_color,
            decreasing_fillcolor=dec_color,
            name="Candles",
        ),
        row=1,
        col=1,
    )
    for col, c, dash in (
        ("MA5", CHART_COLORS[0], "solid"),
        ("MA20", CHART_COLORS[1], "solid"),
        ("MA60", CHART_COLORS[2], "dash"),
        ("MA200", CHART_COLORS[0], "dot"),
    ):
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[col], mode="lines", name=col, line=dict(color=c, width=1.6, dash=dash)),
                row=1,
                col=1,
            )
    for col, c in (
        ("BB_UPPER", CHART_COLORS[1]),
        ("BB_MIDDLE", CHART_COLORS[2]),
        ("BB_LOWER", CHART_COLORS[1]),
    ):
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col, line=dict(color=c, width=1, dash="dot")), row=1, col=1)

    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI", line=dict(color=CHART_COLORS[0])), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=TEXT_DIM, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=TEXT_DIM, row=2, col=1)

    if {"MACD_DIF", "MACD_DEA", "MACD_HIST"}.issubset(df.columns):
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_DIF"], mode="lines", name="DIF", line=dict(color=CHART_COLORS[0])), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_DEA"], mode="lines", name="DEA", line=dict(color=CHART_COLORS[1])), row=3, col=1)
        colors = [inc_color if x >= 0 else dec_color for x in df["MACD_HIST"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_HIST"], name="HIST", marker_color=colors, opacity=0.5), row=3, col=1)

    vol_colors = [inc_color if c >= o else dec_color for o, c in zip(df["Open"], df["Close"], strict=False)]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=vol_colors, name="Volume"), row=4, col=1)
    _style_figure(fig, height=900, xaxis_rangeslider_visible=False, legend_orientation="h")
    return fig


def _render_company_profile(ticker: str) -> None:
    """Company-profile tab: load the profile and render it as cards and sections."""
    sym = str(ticker).strip().upper()
    profile = load_profile(sym)

    if not _profile_looks_valid(profile) or not isinstance(profile, dict):
        _callout("Company profile failed to load. Try again later or refresh it manually.")
        if st.button("Refresh profile", key=f"force_profile_refresh_{sym}"):
            get_profile(sym, force_refresh=True)
            load_profile.clear()
            st.rerun()
        return

    name = html.escape(str(profile.get("name") or sym))
    t_disp = html.escape(str(profile.get("ticker") or sym))
    sec = html.escape(str(profile.get("sector") or "—"))
    indu = html.escape(str(profile.get("industry") or "—"))
    ctry = html.escape(str(profile.get("country") or "—"))
    emp = profile.get("employees")
    try:
        if emp is None:
            emp_str = "—"
        else:
            n = float(emp)
            if n == int(n) and 0 <= n < 1e12:
                emp_str = f"{int(n):,}"
            else:
                emp_str = f"{n:,.0f}"
    except (TypeError, ValueError):
        emp_str = str(emp) if emp is not None else "—"
    desc_s = profile.get("description_short") or ""
    desc_s_safe = html.escape(str(desc_s))

    st.markdown(
        "<div class='card'>"
        f"<h2 class='card-title'>{name} <span class='dim'>({t_disp})</span></h2>"
        f"<p class='caption'>{sec} · {indu} · {ctry} · Employees <span class='num'>{html.escape(emp_str)}</span></p>"
        f"<p class='body-line'>{desc_s_safe or 'No one-line summary yet.'}</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        usd_m, twd_m = _format_mcap_usd_twd(profile.get("market_cap"))
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                "<p class='metric-label'>Market cap</p>"
                f"<p class='metric-value'>{html.escape(usd_m)}</p>"
                f"<p class='caption'>{_mono_numbers(twd_m)}</p>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                "<p class='metric-label'>Employees</p>"
                f"<p class='metric-value'>{html.escape(emp_str)}</p>",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                "<p class='metric-label'>Headquarters / country</p>"
                f"<p class='body-line'>{ctry}</p>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    with st.container(border=True):
        st.markdown("### Full description")
        st.markdown(
            "<div class='body-line'>"
            + html.escape(str(profile.get("description_long") or "No full description yet.")).replace(chr(10), "<br/>")
            + "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    with st.container(border=True):
        st.markdown("#### Business")
        left, right = st.columns(2)
        with left:
            _caption("Segments / revenue")
            segs = profile.get("business_segments") or []
            if not isinstance(segs, list) or not segs:
                _caption("—")
            else:
                for s in segs:
                    st.markdown(f"- {html.escape(str(s))}", unsafe_allow_html=True)
        with right:
            _caption("Key products / services")
            prods = profile.get("key_products") or []
            if not isinstance(prods, list) or not prods:
                _caption("—")
            else:
                for p0 in prods:
                    st.markdown(f"- {html.escape(str(p0))}", unsafe_allow_html=True)

    st.markdown("---")
    with st.container(border=True):
        st.markdown("#### Competitors")
        _caption("Select a tag. If it maps to a ticker in this universe, the page switches to that stock.")
        competitors = [str(x) for x in (profile.get("competitors") or [])]
        if not competitors:
            _caption("No data yet")
        elif hasattr(st, "pills"):
            pkey = f"competitor_pill_{sym}"

            def _on_pill() -> None:
                p = st.session_state.get(pkey)
                if p is None:
                    return
                t = _competitor_to_ticker(str(p))
                if t and t in ALL_STOCKS and t != sym:
                    st.session_state["selected_ticker"] = t
                    st.rerun()

            st.pills(
                "Select a tag to switch stocks (when it maps to a ticker in this universe)",
                options=competitors,
                on_change=_on_pill,
                key=pkey,
                selection_mode="single",
            )
        else:
            cols = st.columns(min(6, max(1, len(competitors))))
            for i, c in enumerate(competitors):
                t_nav = _competitor_to_ticker(c)
                with cols[i % len(cols)]:
                    if t_nav and t_nav in ALL_STOCKS:
                        if st.button(f"{_truncate_text(c, 20)}", key=f"comp_btn_{sym}_{i}"):
                            st.session_state["selected_ticker"] = t_nav
                            st.rerun()
                    else:
                        _caption(_truncate_text(c, 24))

    st.markdown("---")
    with st.container(border=True):
        st.markdown("### Further reading")
        articles: list[dict[str, Any]] = (
            profile.get("articles") if isinstance(profile.get("articles"), list) else []
        )
        by_t: dict[str, list[dict[str, Any]]] = {k: [] for k in ARTICLE_TYPE_ORDER}
        for a in articles:
            if not isinstance(a, dict):
                continue
            t = str(a.get("type") or "reference")
            if t not in by_t:
                by_t.setdefault("reference", []).append(a)
            else:
                by_t[t].append(a)

        for t in ARTICLE_TYPE_ORDER:
            items = by_t.get(t) or []
            if not items:
                continue
            st.markdown(f"**{ARTICLE_TYPE_LABEL.get(t, t)}**")
            for a in items:
                title = str(a.get("title") or "Link")
                url = str(a.get("url") or "").strip()
                src = str(a.get("source") or "")
                if url and url.startswith("http"):
                    t_short = _truncate_text(title, 100)
                    st.link_button(
                        f"{t_short} · {src or '—'}",
                        url,
                        use_container_width=False,
                    )
                else:
                    _caption(f"{title} (invalid URL)")

    _fa = str(profile.get("fetched_at") or "—")
    _src = ", ".join(str(s) for s in (profile.get("sources_used") or [])) or "—"
    _caption(f"Updated: {_fa} · Sources: {_src}")


def render_deep_page(daily: dict[str, Any]) -> None:
    """Page 2: single-stock analysis."""
    st.title("Stock deep dive")
    default_ticker = st.session_state.get("selected_ticker", TICKER_LIST[0])
    idx = TICKER_LIST.index(default_ticker) if default_ticker in TICKER_LIST else 0
    ticker = st.selectbox("Select a stock", TICKER_LIST, index=idx)
    st.session_state["selected_ticker"] = ticker

    bundle = get_stock_bundle(ticker)
    raw, df = bundle["raw"], bundle["df_ind"]
    tech, fund, val, pred = bundle["technical"], bundle["fundamental"], bundle["valuation"], bundle["prediction"]
    info = raw.get("info", {}) or {}

    pick_news = _find_pick_news(daily, ticker)
    realtime_used = False
    if pick_news is None:
        pick_news = get_live_ticker_sentiment(ticker)
        realtime_used = True

    tab_profile, tab1, tab2, tab3, tab4 = st.tabs(
        ["Company profile", "Technicals", "Fundamentals", "Price forecast", "News sentiment"]
    )

    with tab_profile:
        _render_company_profile(ticker)

    with tab1:
        st.plotly_chart(build_technical_figure(df.tail(260)), use_container_width=True)
        cv = tech.get("current_values", {})
        sig = tech.get("signals", {})
        ind_df = pd.DataFrame(
            [
                {"Indicator": "RSI", "Value": cv.get("rsi"), "Signal": sig.get("rsi")},
                {"Indicator": "MACD_HIST", "Value": cv.get("macd_hist"), "Signal": sig.get("macd")},
                {"Indicator": "KD_K", "Value": cv.get("kd_k"), "Signal": ""},
                {"Indicator": "KD_D", "Value": cv.get("kd_d"), "Signal": ""},
                {"Indicator": "ADX", "Value": cv.get("adx"), "Signal": sig.get("ma_trend")},
                {"Indicator": "ATR", "Value": cv.get("atr"), "Signal": ""},
            ]
        )
        _render_table(ind_df)
        st.markdown("### Technical score")
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=float(tech.get("total_score", 0)),
                number={"font": {"family": "IBM Plex Mono, monospace", "size": 22, "color": TEXT}},
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickfont": {"family": "IBM Plex Sans, sans-serif", "size": 12, "color": TEXT_DIM},
                    },
                    "bar": {"color": ACCENT},
                    "bgcolor": BAR_TRACK,
                },
            )
        )
        _style_figure(gauge, height=220)
        st.plotly_chart(gauge, use_container_width=True)
        _caption(_SCORE_HELP["technical"])

    with tab2:
        sub = fund.get("sub_scores", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Growth", sub.get("growth"), help=_SCORE_HELP["growth"])
        c2.metric("Profitability", sub.get("profitability"), help=_SCORE_HELP["profitability"])
        c3.metric("Financial health", sub.get("financial_health"), help=_SCORE_HELP["financial_health"])
        c4.metric("Industry outlook", sub.get("industry_outlook"), help=_SCORE_HELP["industry_outlook"])

        avg = get_sector_fundamental_avg(ticker)
        cats = ["growth", "profitability", "financial_health", "industry_outlook"]
        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(r=[sub.get(k, 0) for k in cats], theta=cats, fill="toself", name=ticker))
        radar.add_trace(go.Scatterpolar(r=[avg.get(k, 0) for k in cats], theta=cats, fill="toself", name="Sector average"))
        _style_figure(radar, polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=420)
        st.plotly_chart(radar, use_container_width=True)

        fin_df = pd.DataFrame(
            [
                ["revenueGrowth", info.get("revenueGrowth")],
                ["earningsGrowth", info.get("earningsGrowth")],
                ["ROE", info.get("returnOnEquity")],
                ["grossMargins", info.get("grossMargins")],
                ["debtToEquity", info.get("debtToEquity")],
                ["currentRatio", info.get("currentRatio")],
            ],
            columns=["Indicator", "Value"],
        )
        _render_table(fin_df)
        a1, a2 = st.columns(2)
        a1.metric(
            "Analyst rating (recommendationMean)",
            info.get("recommendationMean"),
            help="Yahoo Finance recommendationMean, shown as returned. This dashboard does not rescale it.",
        )
        a2.metric(
            "Analyst target (targetMeanPrice)",
            info.get("targetMeanPrice"),
            help="Yahoo Finance targetMeanPrice, shown as returned. This dashboard does not rescale it.",
        )

    with tab3:
        current_price = float(raw.get("current_price") or raw["price_history"]["Close"].iloc[-1])
        fair_price = float(val.get("fair_value"))
        fair_diff = (fair_price / current_price - 1) * 100 if current_price else 0
        p30 = float(pred.get("predicted_30d"))
        low = float(pred.get("predicted_low"))
        high = float(pred.get("predicted_high"))
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Current price",
            f"${current_price:.2f}",
            help="Latest price used by the valuation and the 30-day forecast.",
        )
        c2.metric(
            "Fair value",
            f"${fair_price:.2f}",
            f"{fair_diff:+.2f}%",
            help=_SCORE_HELP["valuation"],
        )
        c3.metric(
            "30-day forecast",
            f"${p30:.2f}",
            help=_SCORE_HELP["forecast"],
        )
        _caption(f"[{low:.2f}, {high:.2f}]", target=c3)
        if val.get("valuation_reliable") is False:
            _callout(
                str(val.get("valuation_reason") or "").strip()
                or "Trailing P/E implies an implausible fair value; earnings are likely at a cyclical peak or trough"
            )

        entry = compute_entry_analysis(ticker)
        if entry:
            st.markdown("### Staged entry plan")
            batch_plan = entry.get("batch_plan") if isinstance(entry.get("batch_plan"), list) else []
            for i, batch in enumerate(batch_plan, 1):
                if not isinstance(batch, dict):
                    continue
                cols_bp = st.columns([1, 2, 3])
                with cols_bp[0]:
                    st.markdown(f"<p class='body-line'><strong>Tranche <span class='num'>{i}</span></strong></p>", unsafe_allow_html=True)
                with cols_bp[1]:
                    px = _to_float_safe(batch.get("price")) or 0.0
                    ap = batch.get("allocation_pct", "")
                    st.markdown(
                        f"<p class='body-line'><span class='num'>${px:.2f}</span> (<span class='num'>{html.escape(str(ap))}%</span>)</p>",
                        unsafe_allow_html=True,
                    )
                with cols_bp[2]:
                    _caption(str(batch.get("condition") or ""))
            rr = _to_float_safe(entry.get("risk_reward_ratio")) or 0.0
            if rr > 0:
                st.metric(
                    "Risk / reward",
                    f"{rr:.2f} : 1",
                    help="Upside versus downside. 2 or higher is favorable.",
                )
        else:
            _caption("Entry prices and the staged plan could not be calculated")

        hist = raw["price_history"].copy().tail(365)
        pred_dates = pd.to_datetime(pred["prediction_dates"])
        pred_line = pd.Series(pred["prediction_prices"], index=pred_dates)
        pred_up = pd.Series(pred["prediction_upper"], index=pred_dates)
        pred_lo = pd.Series(pred["prediction_lower"], index=pred_dates)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="Close", line=dict(color=CHART_COLORS[0], width=2)))
        fig.add_trace(go.Scatter(x=pred_line.index, y=pred_line.values, mode="lines", name="30-day forecast", line=dict(color=CHART_COLORS[1], dash="dash", width=2)))
        fig.add_trace(go.Scatter(x=pred_up.index, y=pred_up.values, mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(
            go.Scatter(
                x=pred_lo.index,
                y=pred_lo.values,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(21,56,92,0.12)",
                line=dict(width=0),
                name="95% interval",
            )
        )
        fig.add_hline(y=fair_price, line_dash="dot", line_color=CHART_COLORS[2], annotation_text="Fair value")
        if info.get("targetMeanPrice") is not None:
            fig.add_hline(y=float(info.get("targetMeanPrice")), line_dash="dash", line_color=CHART_COLORS[1], annotation_text="Analyst target")
        _style_figure(fig, height=500)
        st.plotly_chart(fig, use_container_width=True)

        methods = val.get("methods", {})
        table = pd.DataFrame(
            [
                ["P/E", methods.get("pe_method", {}).get("fair_price"), methods.get("pe_method", {}).get("weight")],
                ["PEG", methods.get("peg_method", {}).get("fair_price"), methods.get("peg_method", {}).get("weight")],
                ["Analyst target", methods.get("analyst_target", {}).get("fair_price"), methods.get("analyst_target", {}).get("weight")],
            ],
            columns=["Method", "Valuation", "Weight"],
        )
        _render_table(table)
        _caption("This forecast is for reference only and is not investment advice")

    with tab4:
        news = pick_news or {}
        sentiment_label = str(news.get("sentiment_label", "No News"))
        sentiment_score = float(news.get("sentiment_score", 0) or 0)
        news_count = int(news.get("news_count", 0) or 0)
        pos = int(news.get("positive_count", 0) or 0)
        neu = int(news.get("neutral_count", 0) or 0)
        neg = int(news.get("negative_count", 0) or 0)

        if realtime_used:
            _callout("This is a live analysis, not today's scheduled run")

        # Top metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(
            f"<p class='card-title' style='color:{get_sentiment_color(sentiment_label)};'>"
            f"{html.escape(sentiment_label)}</p>",
            unsafe_allow_html=True,
        )
        _caption("Sentiment label", target=c1)
        c2.metric(
            "Sentiment score",
            f"{sentiment_score:.2f}",
            help="Time-weighted FinBERT score from -1 to 1. Positive means recent headlines lean positive. The 0-100 news score is (this value + 1) times 50.",
        )
        c3.metric(
            "News count",
            news_count,
            help="Number of headlines included in the sentiment score.",
        )
        c4.metric("Breakdown", f"positive {pos} / neutral {neu} / negative {neg}")

        # Sentiment gauge
        st.markdown("### News sentiment score")
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=sentiment_score,
                number={
                    "valueformat": ".2f",
                    "font": {"family": "IBM Plex Mono, monospace", "size": 22, "color": TEXT},
                },
                gauge={
                    "axis": {
                        "range": [-1, 1],
                        "tickfont": {"family": "IBM Plex Sans, sans-serif", "size": 12, "color": TEXT_DIM},
                    },
                    "bar": {"color": _sentiment_hex(sentiment_label)},
                    "steps": [
                        {"range": [-1, -0.2], "color": "rgba(179,38,30,0.18)"},
                        {"range": [-0.2, 0.2], "color": BAR_TRACK},
                        {"range": [0.2, 1], "color": "rgba(14,124,90,0.18)"},
                    ],
                    "threshold": {"line": {"color": TEXT, "width": 1}, "value": sentiment_score},
                },
            )
        )
        _style_figure(gauge, height=220)
        st.plotly_chart(gauge, use_container_width=True)

        # AI news summary sits under the gauge and above the article list
        st.markdown("### AI news analysis")
        summary_raw = news.get("summary", "")
        summary = str(summary_raw).strip() if summary_raw is not None else ""

        if not summary:
            _caption("No summary yet")
        elif summary.startswith("[Offline summary]"):
            _caption("Claude API unavailable; showing the template summary")
            _callout(summary.replace("[Offline summary] ", "", 1))
        else:
            st.markdown(
                f"<div class='callout'><div class='body-line' style='white-space:pre-wrap;'>{_mono_numbers(summary)}</div></div>",
                unsafe_allow_html=True,
            )

        _caption("Method: Claude Haiku 4.5 + FinBERT sentiment + time-decay weights")
        st.markdown("---")

        st.subheader("Recent news (open the original)")
        analyzed_news = news.get("analyzed_news", []) if isinstance(news.get("analyzed_news"), list) else []
        analyzed_news = sorted(
            analyzed_news,
            key=lambda x: pd.to_datetime(x.get("published_at"), errors="coerce"),
            reverse=True,
        )
        if news_count == 0 or not analyzed_news:
            _callout("No related news yet")
        else:
            for one in analyzed_news:
                sent = str(one.get("sentiment", "neutral"))
                sent_cn = str(one.get("sentiment_cn", "Neutral"))
                sent_color = (
                    "var(--positive)" if sent == "positive" else "var(--negative)" if sent == "negative" else "var(--text-dim)"
                )
                title = html.escape(_truncate_text(str(one.get("title", "")), 80))
                publisher = html.escape(str(one.get("publisher", "")))
                tstr = html.escape(_fmt_news_time(one.get("published_at")))
                conf = float(one.get("confidence", 0) or 0)
                link = str(one.get("link", "") or "")
                with st.container(border=True):
                    st.markdown(
                        f"<div style='border-left:2px solid {sent_color}; padding-left:8px;'>"
                        f"<div class='body-line' style='font-weight:600;'>{title}</div>"
                        f"<p class='caption'>{publisher} · {tstr} · {html.escape(sent_cn)} · "
                        f"confidence <span class='num'>{conf:.2f}</span></p>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    if link:
                        st.link_button("Read original", link, use_container_width=False)


# Published names for the keys in config.AI_LAYERS. The keys stay internal.
_LAYER_NAMES = {
    "layer1_infrastructure": "Layer 1 Infrastructure",
    "layer2_cloud": "Layer 2 Cloud & compute",
    "layer3_foundation": "Layer 3 Foundation models",
    "layer4_platform": "Layer 4 Data platforms",
    "layer5_applications": "Layer 5 AI applications",
    "layer6_consumer": "Layer 6 Consumer & content",
}


def _ai_layer_name(ticker: str) -> str:
    """Look up the AI value-chain layer for one ticker.

    Membership comes from config.AI_LAYERS. The returned name is the
    published label for that key.

    Args:
        ticker: Ticker symbol.

    Returns:
        The layer name, or "—" when the ticker is in no layer.
    """
    sym = str(ticker).strip().upper()
    for key, tickers in AI_LAYERS.items():
        if sym in tickers:
            return _LAYER_NAMES.get(key, key)
    return "—"


def render_overview_page(daily: dict[str, Any]) -> None:
    """Page 3: stock overview."""
    st.title("Universe overview")
    rows = daily.get("all_scores", [])
    if not rows:
        _callout("No overview data")
        return
    df = pd.DataFrame(rows).rename(
        columns={
            "ticker": "Ticker",
            "name": "Company",
            "composite_score": "Composite Score",
            "tech_score": "Technical",
            "fund_score": "Fundamental",
            "valuation_score": "Valuation",
            "prediction_return_pct": "Forecast",
        }
    )
    df["Layer"] = df["Ticker"].map(_ai_layer_name)
    df["Peer group"] = df["Ticker"].apply(get_stock_sector)
    price_map = {}
    for p in daily.get("picks", []):
        price_map[p.get("ticker")] = p.get("current_price")
    df["Price"] = df["Ticker"].map(price_map)

    # Fill prices missing from all_scores via the data_collector cache
    missing_mask = df["Price"].isna()
    if missing_mask.any():
        missing_tickers = df.loc[missing_mask, "Ticker"].astype(str).tolist()
        filled_prices = {t: get_cached_current_price(t) for t in missing_tickers}
        df.loc[missing_mask, "Price"] = df.loc[missing_mask, "Ticker"].map(filled_prices)

    df = df[["Ticker", "Layer", "Peer group", "Company", "Price", "Composite Score", "Technical", "Fundamental", "Valuation", "Forecast"]]
    df = df.sort_values("Composite Score", ascending=False).reset_index(drop=True)

    tickers_list = df["Ticker"].astype(str).tolist()
    entry_by_ticker: dict[str, dict[str, Any]] = {}
    for t_sym in tickers_list:
        er = compute_entry_analysis(t_sym)
        if er:
            entry_by_ticker[t_sym] = er

    def _entry_label(sym: str) -> str:
        e = entry_by_ticker.get(str(sym))
        if not e:
            return "No entry"
        return get_status_label_cn(str(e.get("current_status") or "watch"))

    def _dist_standard_pct(sym: str) -> str:
        e = entry_by_ticker.get(str(sym))
        if not e:
            return "—"
        d = e.get("distance") if isinstance(e.get("distance"), dict) else {}
        v = _to_float_safe(d.get("to_standard_pct"))
        if v is None:
            return "—"
        return f"{v:+.1f}%"

    df["Entry"] = df["Ticker"].map(_entry_label)
    df["Distance to Standard Entry"] = df["Ticker"].map(_dist_standard_pct)
    df["_entry_status"] = df["Ticker"].astype(str).map(
        lambda x: str((entry_by_ticker.get(str(x)) or {}).get("current_status") or "")
    )

    keyword = st.text_input("Search ticker or company")
    sectors = sorted(df["Peer group"].dropna().unique().tolist())
    sector_sel = st.multiselect("Filter by peer group", sectors)
    only_buy_now = st.checkbox("Only show names ready to enter (buy now)", value=False)
    filtered = df.copy()
    if keyword:
        filtered = filtered[filtered["Ticker"].str.contains(keyword, case=False) | filtered["Company"].str.contains(keyword, case=False)]
    if sector_sel:
        filtered = filtered[filtered["Peer group"].isin(sector_sel)]
    if only_buy_now:
        filtered = filtered[filtered["_entry_status"] == "buy_now"]

    display_df = filtered.drop(columns=["_entry_status"], errors="ignore")

    _render_table(display_df)

    sym_opts = display_df["Ticker"].tolist()
    choice: str | None = None
    if sym_opts:
        choice = st.selectbox("Select a stock for the full analysis", sym_opts, index=0)
    else:
        _caption("No stocks match the filter")
    if choice and st.button("Open stock analysis"):
        st.session_state["selected_ticker"] = choice
        st.session_state["page"] = "Stock Deep Dive"
        st.rerun()


def _fmt_pct(value: Any, digits: int = 2) -> str:
    """Format a fraction as a percent string. None becomes an em dash."""
    v = _to_float_safe(value)
    return f"{v:+.{digits}%}" if v is not None else "—"


def _render_perf_metrics(summary: dict[str, Any], cumulative: dict[str, Any] | str) -> None:
    """Performance page, section 1: four headline metric cards."""
    total = int(summary.get("total_recommendations") or 0)
    done = int(summary.get("completed_1m") or 0)
    strat = _to_float_safe(cumulative.get("strategy_cumulative_return")) if isinstance(cumulative, dict) else None
    spy = _to_float_safe(cumulative.get("spy_cumulative_return")) if isinstance(cumulative, dict) else None
    outperf = _to_float_safe(cumulative.get("outperformance")) if isinstance(cumulative, dict) else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Recommendations",
        f"{total}",
        help="Count of logged daily recommendations.",
    )
    c2.metric(
        "1M tracking complete",
        f"{done}",
        help="How many of those recommendations already have a realized 1-month return.",
    )
    if total:
        _caption(f"{done / total:.0%} complete", target=c2)
    # Pass the same number as delta so Streamlit colors it by sign
    c3.metric(
        "Strategy cumulative return",
        _fmt_pct(strat),
        delta=_fmt_pct(strat) if strat is not None else None,
        help="Compounded equal-weight return of the names held over the track-record window.",
    )
    c4.metric(
        "Excess return vs SPY",
        _fmt_pct(outperf),
        delta=_fmt_pct(outperf) if outperf is not None else None,
        help="Strategy cumulative return minus SPY buy-and-hold over the same window.",
    )
    if spy is not None:
        _caption(f"SPY buy-and-hold over the same window: {_fmt_pct(spy)}")


def _render_perf_curve(cumulative: dict[str, Any] | str) -> None:
    """Performance page, section 2: strategy vs SPY cumulative-return chart."""
    st.subheader("Cumulative return")
    if not isinstance(cumulative, dict) or not cumulative.get("dates"):
        _callout("No equity curve yet (more completed tracking records are needed)")
        return

    dates = pd.to_datetime(pd.Series(cumulative["dates"]))
    strat_curve = cumulative.get("strategy_equity_curve") or []
    spy_curve = cumulative.get("spy_equity_curve") or []
    dr = cumulative.get("date_range") if isinstance(cumulative.get("date_range"), dict) else {}
    _caption(
        f"Window: {dr.get('start', '-')} ~ {dr.get('end', '-')}"
        f" | {cumulative.get('trading_days', 0)} trading days"
        f" | each day compounds the equal-weight return of names then held; SPY is buy-and-hold"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=dates, y=strat_curve, mode="lines", name="Strategy", line=dict(color=CHART_COLORS[0], width=2.5))
    )
    if spy_curve:
        fig.add_trace(
            go.Scatter(x=dates, y=spy_curve, mode="lines", name="SPY buy-and-hold", line=dict(color=CHART_COLORS[1], width=2))
        )
    fig.add_hline(y=0, line_dash="dot", line_color=BORDER)
    _style_figure(
        fig,
        height=460,
        hovermode="x unified",
        legend_orientation="h",
        yaxis=dict(title="", tickformat=".0%"),
        xaxis=dict(title=""),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_perf_concentration(concentration: dict[str, Any] | str) -> None:
    """
    Performance page, section 3: leave-one-out attribution.
    Shows whether the return is concentrated in a few names.
    """
    st.subheader("Attribution (leave-one-out)")
    if not isinstance(concentration, dict) or not concentration.get("by_ticker"):
        _callout("No attribution yet (more completed tracking records are needed)")
        return

    by_ticker = concentration["by_ticker"]
    rows = [
        {
            "Ticker": ticker,
            "Times Selected": int(stats.get("picks") or 0),
            "Avg 1M Return": _fmt_pct(stats.get("avg_return_1m")),
            "Contribution": _fmt_pct(stats.get("contribution")),
            "_sort": _to_float_safe(stats.get("contribution"), 0.0),
        }
        for ticker, stats in by_ticker.items()
    ]
    table = pd.DataFrame(rows).sort_values("_sort", ascending=False).drop(columns=["_sort"])
    _render_table(table)

    warning = concentration.get("concentration_warning")
    top = concentration.get("top_contributor") if isinstance(concentration.get("top_contributor"), dict) else {}
    if warning:
        share = _to_float_safe(top.get("share_of_outperformance"))
        pct_txt = f"{share * 100:.1f}" if share is not None else "—"
        _callout(
            f"Performance is concentrated in one name ({top.get('ticker', '—')} contributes {pct_txt}%). "
            "The sample is too narrow to treat the stock-picking result as consistent."
        )
    elif top:
        _caption(f"Largest contributor: {top.get('ticker', '—')} ({_fmt_pct(top.get('contribution'))})")

    _caption(
        "Contribution = full equity-curve return minus the return recomputed without that name. "
        "Equal weight means dropping one name raises the weight of the rest, so contributions "
        "do not sum to the total return. Use the figure to judge concentration, not as an exact split."
    )


def _render_perf_records() -> None:
    """Performance page, section 4: full recommendation history, filterable by tracking status."""
    st.subheader("Recommendation history")
    df = load_recommendations()
    if df.empty:
        _callout("No recommendations yet (run python src/track_record.py first)")
        return

    done_mask = pd.to_numeric(df.get("return_1m"), errors="coerce").notna()
    choice = st.radio("Tracking status", ["All", "Completed", "Tracking"], horizontal=True)
    if choice == "Completed":
        view = df[done_mask]
    elif choice == "Tracking":
        view = df[~done_mask]
    else:
        view = df

    if view.empty:
        _caption("No records match the filter")
        return

    view = view.copy()
    view["_d"] = pd.to_datetime(view["date"], errors="coerce")
    view = view.sort_values(["_d", "top_rank"], ascending=[False, True])

    out = pd.DataFrame(
        {
            "Date": view["date"].astype(str),
            "Ticker": view["ticker"].astype(str),
            "Peer group": view.get("sector", pd.Series(dtype=str)).astype(str),
            "Rank": pd.to_numeric(view.get("top_rank"), errors="coerce"),
            "Entry Price": pd.to_numeric(view.get("entry_price"), errors="coerce").map(
                lambda v: f"${v:.2f}" if pd.notna(v) else "—"
            ),
            "Composite Score": pd.to_numeric(view.get("composite_score"), errors="coerce").round(1),
            "1D": view.get("return_1d").map(_fmt_pct),
            "1W": view.get("return_1w").map(_fmt_pct),
            "2W": view.get("return_2w").map(_fmt_pct),
            "1M": view.get("return_1m").map(_fmt_pct),
            "SPY 1M": view.get("spy_return_1m").map(_fmt_pct),
            "Alpha": view.get("alpha_1m").map(_fmt_pct),
            "Status": view.get("status", pd.Series(dtype=str)).astype(str),
        }
    )
    _render_table(out)
    _caption(f"{len(out)} rows | returns use the entry price on the recommendation date; 1M is 30 calendar days")


def render_performance_page() -> None:
    """Page 4: strategy performance from the track record."""
    st.title("Strategy performance")
    with st.spinner("Loading performance…"):
        perf = load_performance()
    if not perf:
        _callout("Could not load performance. Run python src/track_record.py first.")
        return

    summary = perf.get("summary") if isinstance(perf.get("summary"), dict) else {}
    cumulative = perf.get("cumulative", "insufficient data")
    concentration = perf.get("concentration", "insufficient data")

    snap = perf.get("snapshot_time")
    rng = summary.get("date_range") if isinstance(summary.get("date_range"), dict) else {}
    _caption(
        f"Recommendation window: {rng.get('first', '-')} ~ {rng.get('last', '-')}"
        + (f" | snapshot: {snap}" if snap else " | computed just now")
    )

    _render_perf_metrics(summary, cumulative)
    if perf.get("note"):
        _callout(str(perf["note"]))
    st.divider()
    _render_perf_curve(cumulative)
    st.divider()
    _render_perf_concentration(concentration)
    st.divider()
    _render_perf_records()


def main() -> None:
    """Dashboard entry point."""
    if st.button("Refresh", help="Reload the data"):
        st.cache_data.clear()
        st.rerun()

    daily = load_latest_daily_json()
    page = _render_sidebar(daily)
    if not daily:
        _callout("Run python src/daily_picker.py first to generate today's analysis")
        _page_footer(None)
        return

    if page == "Daily Picks":
        render_daily_page(daily)
    elif page == "Stock Deep Dive":
        render_deep_page(daily)
    elif page == "Strategy Performance":
        render_performance_page()
    else:
        render_overview_page(daily)
    _page_footer(str(daily.get("date") or ""))


if __name__ == "__main__":
    main()
