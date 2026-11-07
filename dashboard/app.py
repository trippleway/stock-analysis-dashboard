# -*- coding: utf-8 -*-
"""
US stock analysis dashboard (Streamlit).

Run from the project root:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import html
import json
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

from config import ALL_STOCKS, PICKS_DIR, SECTOR_MAP, TICKER_LIST  # noqa: E402
from data_collector import fetch_stock_data  # noqa: E402
from entry_price import calculate_entry_prices, get_status_emoji, get_status_label_cn  # noqa: E402
from profile_fetcher import get_profile  # noqa: E402
from fundamentals import analyze_fundamentals, get_stock_sector  # noqa: E402
from indicators import calculate_indicators, calculate_tech_score  # noqa: E402
from predictor import predict_price  # noqa: E402
from sentiment import analyze_ticker_sentiment  # noqa: E402
from track_record import PERFORMANCE_JSON, RECOMMENDATIONS_CSV, calculate_performance  # noqa: E402
from valuation import calculate_fair_value  # noqa: E402

st.set_page_config(page_title="US Stock Analysis Dashboard", page_icon="📈", layout="wide")

# Display-only TWD equivalent (fixed rate, no FX API calls)
USD_TWD = 32.0

# Further-reading groups on the company profile, in display order
ARTICLE_TYPE_ORDER = ("official", "ir", "filing", "reference")
ARTICLE_TYPE_LABEL: dict[str, str] = {
    "official": "🏛️ Official website",
    "ir": "💼 Investor relations",
    "filing": "📑 SEC and regulatory filings",
    "reference": "📌 References",
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


def pct_color(v: float | None) -> str:
    """Up/down color (US convention: green up, red down)."""
    if v is None:
        return "gray"
    return "#00C853" if v >= 0 else "#FF5252"


def get_sentiment_color(label: str) -> str:
    """Color for a news-sentiment label."""
    return {
        "Strongly Bullish": "#00C853",
        "Bullish": "#66BB6A",
        "Neutral": "#9E9E9E",
        "Bearish": "#EF5350",
        "Strongly Bearish": "#D32F2F",
        "No News": "#616161",
    }.get(label, "#9E9E9E")


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
    try:
        if value is None:
            return default
        return float(value)
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


def render_daily_page(daily: dict[str, Any]) -> None:
    """Page 1: daily picks."""
    st.title("US Stock Analysis Dashboard")
    date_str = daily.get("date", datetime.now().strftime("%Y-%m-%d"))
    m = daily.get("market_overview", {})
    spy_chg = m.get("spy_change_pct")
    spy_price = m.get("spy_price")
    spy_txt = f"SPY {spy_price} ({spy_chg:+.2f}%)" if spy_price is not None and spy_chg is not None else "SPY N/A"
    st.caption(
        f"Date: {date_str} | {spy_txt} | analyzed {m.get('total_analyzed', 0)}, passed filter {m.get('passed_filter', 0)}"
    )

    picks = daily.get("picks", [])[:3]
    cols = st.columns(3)
    for i in range(3):
        with cols[i]:
            if i >= len(picks):
                st.info("No data yet")
                continue
            p = picks[i]
            name = p.get("name", "")
            ticker = p.get("ticker", "")
            curr = p.get("current_price")
            pred_ret = p.get("prediction", {}).get("expected_return_pct")
            st.subheader(f"{ticker} {name}")
            if curr is not None:
                color = pct_color(pred_ret if isinstance(pred_ret, (int, float)) else None)
                st.markdown(f"<h3 style='color:{color};'>${float(curr):.2f}</h3>", unsafe_allow_html=True)
            st.progress(min(max(int(float(p.get("composite_score", 0))), 0), 100), text=f"Composite {p.get('composite_score')}")

            news = p.get("news", {}) if isinstance(p.get("news"), dict) else {}
            sent_label = str(news.get("sentiment_label", "No News"))
            news_count = int(news.get("news_count", 0) or 0)
            sent_color = get_sentiment_color(sent_label)
            st.markdown(
                f"<div style='color:{sent_color};font-weight:600;margin:6px 0 8px 0;'>"
                f"📰 News: {sent_label} ({news_count} items)</div>",
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns(2)
            c3, c4 = st.columns(2)
            c1.metric("Technical", p.get("technical", {}).get("total_score"))
            c2.metric("Fundamental", p.get("fundamental", {}).get("total_score"))
            c3.metric("Valuation", p.get("valuation", {}).get("score"))
            c4.metric("Forecast", p.get("prediction", {}).get("score"))

            entry = p.get("entry_price") if isinstance(p.get("entry_price"), dict) else {}
            if entry:
                status = entry.get("current_status") or "watch"
                emoji = get_status_emoji(str(status))
                label = get_status_label_cn(str(status))
                st.markdown(f"### 🎯 Entry  {emoji} **{label}**")
                ep = entry.get("entry_prices") or {}
                dist = entry.get("distance") or {}
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Conservative entry",
                        f"${float(ep.get('conservative') or 0):.2f}",
                        delta=f"{float(dist.get('to_conservative_pct') or 0):+.1f}%",
                        delta_color="inverse",
                    )
                with col2:
                    st.metric(
                        "Standard entry",
                        f"${float(ep.get('standard') or 0):.2f}",
                        delta=f"{float(dist.get('to_standard_pct') or 0):+.1f}%",
                        delta_color="inverse",
                    )
                with col3:
                    st.metric(
                        "Aggressive entry",
                        f"${float(ep.get('aggressive') or 0):.2f}",
                        delta=f"{float(dist.get('to_aggressive_pct') or 0):+.1f}%",
                        delta_color="inverse",
                    )
                st.info(entry.get("reasoning") or "")
            else:
                st.caption("No entry-price analysis yet (rerun daily_picker to refresh the report)")

            with st.expander("Why this pick"):
                for r in p.get("pick_reasons", []):
                    st.markdown(f"- {r}")

            top_news = news.get("top_headlines", []) if isinstance(news.get("top_headlines"), list) else []
            with st.expander(f"📰 Related news ({news_count})"):
                sum_raw = news.get("summary")
                if sum_raw:
                    s = str(sum_raw)
                    summary_preview = s[:200]
                    if len(s) > 200:
                        summary_preview += "..."
                    st.caption(summary_preview)
                    st.markdown("---")
                if not top_news:
                    st.caption("No related news yet")
                for one in top_news[:3]:
                    title = _truncate_text(str(one.get("title", "")), 80)
                    sent = str(one.get("sentiment_cn", one.get("sentiment", "Neutral")))
                    conf = float(one.get("confidence", 0) or 0)
                    link = str(one.get("link", "") or "")
                    st.markdown(f"**{title}**")
                    st.caption(f"{sent} · confidence {conf:.2f}")
                    if link:
                        st.link_button("📖 Read original", link, use_container_width=False)

            risks = p.get("risk_factors", [])
            if risks:
                st.warning("; ".join(risks))

            if st.button("View full analysis", key=f"detail_{ticker}", use_container_width=True):
                st.session_state["selected_ticker"] = ticker
                st.session_state["page"] = "🔬 個股深度分析"
                st.rerun()


def build_technical_figure(df: pd.DataFrame) -> go.Figure:
    """Multi-panel technical chart."""
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.52, 0.16, 0.16, 0.16],
    )
    inc_color, dec_color = "#00C853", "#FF5252"
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            increasing_line_color=inc_color,
            decreasing_line_color=dec_color,
            name="Candles",
        ),
        row=1,
        col=1,
    )
    for col, c in [("MA5", "#FFD166"), ("MA20", "#F4A261"), ("MA60", "#9B5DE5"), ("MA200", "#00BBF9")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col, line=dict(color=c, width=1.6)), row=1, col=1)
    for col, c in [("BB_UPPER", "#8ecae6"), ("BB_MIDDLE", "#adb5bd"), ("BB_LOWER", "#8ecae6")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col, line=dict(color=c, width=1, dash="dot")), row=1, col=1)

    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI", line=dict(color="#00b4d8")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ef476f", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#06d6a0", row=2, col=1)

    if {"MACD_DIF", "MACD_DEA", "MACD_HIST"}.issubset(df.columns):
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_DIF"], mode="lines", name="DIF", line=dict(color="#ffd166")), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_DEA"], mode="lines", name="DEA", line=dict(color="#118ab2")), row=3, col=1)
        colors = [inc_color if x >= 0 else dec_color for x in df["MACD_HIST"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_HIST"], name="HIST", marker_color=colors, opacity=0.5), row=3, col=1)

    vol_colors = [inc_color if c >= o else dec_color for o, c in zip(df["Open"], df["Close"], strict=False)]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=vol_colors, name="Volume"), row=4, col=1)
    fig.update_layout(template="plotly_dark", height=900, xaxis_rangeslider_visible=False, legend_orientation="h")
    return fig


def _render_company_profile(ticker: str) -> None:
    """Company-profile tab: load the profile and render it as cards and sections."""
    sym = str(ticker).strip().upper()
    profile = load_profile(sym)

    if not _profile_looks_valid(profile) or not isinstance(profile, dict):
        st.warning("Company profile failed to load. Try again later or refresh it manually.")
        if st.button("🔄 Refresh profile", key=f"force_profile_refresh_{sym}"):
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
        f"""
<div style="
    background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(147,51,234,0.08));
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
">
    <h2 style="margin:0 0 8px 0;color:#f3f4f6;">{name} <span style="color:#9ca3af;font-size:0.6em">({t_disp})</span></h2>
    <p style="color:#9ca3af;font-size:14px;margin:0 0 12px 0;">{sec} · {indu} · {ctry} · Employees {html.escape(emp_str)}</p>
    <p style="color:#d1d5db;font-size:16px;line-height:1.7;margin:0;">{desc_s_safe or "No one-line summary yet."}</p>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        usd_m, twd_m = _format_mcap_usd_twd(profile.get("market_cap"))
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("##### 💰 Market cap")
            st.markdown(
                f"<p style='margin:0;font-size:15px;color:#e5e7eb'>{html.escape(usd_m)}<br><span style='color:#9ca3af;font-size:13px'>{html.escape(twd_m)}</span></p>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown("##### 👥 Employees")
            st.markdown(
                f"<p style='margin:0;font-size:15px;color:#e5e7eb'>{html.escape(emp_str)}</p>", unsafe_allow_html=True
            )
        with c3:
            st.markdown("##### 🌏 Headquarters / country")
            st.markdown(
                f"<p style='margin:0;font-size:15px;color:#e5e7eb'>{ctry}</p>", unsafe_allow_html=True
            )

    st.markdown("---")
    with st.container(border=True):
        st.markdown("### Full description")
        st.markdown(
            f"""
    <div style="font-size:15px; line-height:1.8; color:#e5e7eb;">
{html.escape(str(profile.get("description_long") or "No full description yet.")).replace(chr(10), "<br/>")}
    </div>
    """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    with st.container(border=True):
        st.markdown("#### Business")
        left, right = st.columns(2)
        with left:
            st.caption("Segments / revenue")
            segs = profile.get("business_segments") or []
            if not isinstance(segs, list) or not segs:
                st.caption("—")
            else:
                for s in segs:
                    st.markdown(f"- {html.escape(str(s))}", unsafe_allow_html=True)
        with right:
            st.caption("Key products / services")
            prods = profile.get("key_products") or []
            if not isinstance(prods, list) or not prods:
                st.caption("—")
            else:
                for p0 in prods:
                    st.markdown(f"- {html.escape(str(p0))}", unsafe_allow_html=True)

    st.markdown("---")
    with st.container(border=True):
        st.markdown("#### Competitors")
        st.caption("Select a tag. If it maps to a ticker in this universe, the page switches to that stock.")
        competitors = [str(x) for x in (profile.get("competitors") or [])]
        if not competitors:
            st.caption("No data yet")
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
                        st.caption(_truncate_text(c, 24))

    st.markdown("---")
    with st.container(border=True):
        st.markdown("### 📚 Further reading")
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
                        f"🔗 {t_short} · {src or '—'}",
                        url,
                        use_container_width=False,
                    )
                else:
                    st.caption(f"{title} (invalid URL)")

    _fa = str(profile.get("fetched_at") or "—")
    _src = ", ".join(str(s) for s in (profile.get("sources_used") or [])) or "—"
    st.caption(f"Updated: {_fa} · Sources: {_src}")


def render_deep_page(daily: dict[str, Any]) -> None:
    """Page 2: single-stock analysis."""
    st.title("🔬 Stock analysis")
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
        ["🏢 Company profile", "📈 Technicals", "💼 Fundamentals", "🔮 Price forecast", "📰 News sentiment"]
    )

    with tab_profile:
        _render_company_profile(ticker)

    with tab1:
        st.plotly_chart(build_technical_figure(df.tail(260)), use_container_width=True)
        cv = tech.get("current_values", {})
        sig = tech.get("signals", {})
        ind_df = pd.DataFrame(
            [
                {"指標": "RSI", "數值": cv.get("rsi"), "信號": sig.get("rsi")},
                {"指標": "MACD_HIST", "數值": cv.get("macd_hist"), "信號": sig.get("macd")},
                {"指標": "KD_K", "數值": cv.get("kd_k"), "信號": ""},
                {"指標": "KD_D", "數值": cv.get("kd_d"), "信號": ""},
                {"指標": "ADX", "數值": cv.get("adx"), "信號": sig.get("ma_trend")},
                {"指標": "ATR", "數值": cv.get("atr"), "信號": ""},
            ]
        )
        st.dataframe(ind_df, use_container_width=True, hide_index=True)
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=float(tech.get("total_score", 0)),
                title={"text": "Technical score"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#4cc9f0"}},
            )
        )
        gauge.update_layout(template="plotly_dark", height=260)
        st.plotly_chart(gauge, use_container_width=True)

    with tab2:
        sub = fund.get("sub_scores", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Growth", sub.get("growth"))
        c2.metric("Profitability", sub.get("profitability"))
        c3.metric("Financial health", sub.get("financial_health"))
        c4.metric("Industry outlook", sub.get("industry_outlook"))

        avg = get_sector_fundamental_avg(ticker)
        cats = ["growth", "profitability", "financial_health", "industry_outlook"]
        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(r=[sub.get(k, 0) for k in cats], theta=cats, fill="toself", name=ticker))
        radar.add_trace(go.Scatterpolar(r=[avg.get(k, 0) for k in cats], theta=cats, fill="toself", name="Sector average"))
        radar.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=420)
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
            columns=["指標", "數值"],
        )
        st.dataframe(fin_df, use_container_width=True, hide_index=True)
        a1, a2 = st.columns(2)
        a1.metric("Analyst rating (recommendationMean)", info.get("recommendationMean"))
        a2.metric("Analyst target (targetMeanPrice)", info.get("targetMeanPrice"))

    with tab3:
        current_price = float(raw.get("current_price") or raw["price_history"]["Close"].iloc[-1])
        fair_price = float(val.get("fair_value"))
        fair_diff = (fair_price / current_price - 1) * 100 if current_price else 0
        p30 = float(pred.get("predicted_30d"))
        low = float(pred.get("predicted_low"))
        high = float(pred.get("predicted_high"))
        c1, c2, c3 = st.columns(3)
        c1.metric("Current price", f"${current_price:.2f}")
        c2.metric("Fair value", f"${fair_price:.2f}", f"{fair_diff:+.2f}%")
        c3.metric("30-day forecast", f"${p30:.2f}", f"[{low:.2f}, {high:.2f}]")

        entry = compute_entry_analysis(ticker)
        if entry:
            st.markdown("### 📋 Staged entry plan")
            batch_plan = entry.get("batch_plan") if isinstance(entry.get("batch_plan"), list) else []
            for i, batch in enumerate(batch_plan, 1):
                if not isinstance(batch, dict):
                    continue
                cols_bp = st.columns([1, 2, 3])
                with cols_bp[0]:
                    st.markdown(f"**Tranche {i}**")
                with cols_bp[1]:
                    px = _to_float_safe(batch.get("price")) or 0.0
                    ap = batch.get("allocation_pct", "")
                    st.markdown(f"${px:.2f} ({ap}%)")
                with cols_bp[2]:
                    st.caption(str(batch.get("condition") or ""))
            rr = _to_float_safe(entry.get("risk_reward_ratio")) or 0.0
            if rr > 0:
                st.metric(
                    "Risk / reward",
                    f"{rr:.2f} : 1",
                    help="Upside versus downside. 2 or higher is favorable.",
                )
        else:
            st.caption("Entry prices and the staged plan could not be calculated")

        hist = raw["price_history"].copy().tail(365)
        pred_dates = pd.to_datetime(pred["prediction_dates"])
        pred_line = pd.Series(pred["prediction_prices"], index=pred_dates)
        pred_up = pd.Series(pred["prediction_upper"], index=pred_dates)
        pred_lo = pd.Series(pred["prediction_lower"], index=pred_dates)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="Close", line=dict(color="#4cc9f0", width=2)))
        fig.add_trace(go.Scatter(x=pred_line.index, y=pred_line.values, mode="lines", name="30-day forecast", line=dict(color="#ffd166", dash="dash", width=2)))
        fig.add_trace(go.Scatter(x=pred_up.index, y=pred_up.values, mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(
            go.Scatter(
                x=pred_lo.index,
                y=pred_lo.values,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(76,201,240,0.2)",
                line=dict(width=0),
                name="95% interval",
            )
        )
        fig.add_hline(y=fair_price, line_dash="dot", line_color="#06d6a0", annotation_text="Fair value")
        if info.get("targetMeanPrice") is not None:
            fig.add_hline(y=float(info.get("targetMeanPrice")), line_dash="dot", line_color="#f4a261", annotation_text="Analyst target")
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)

        methods = val.get("methods", {})
        table = pd.DataFrame(
            [
                ["P/E", methods.get("pe_method", {}).get("fair_price"), methods.get("pe_method", {}).get("weight")],
                ["PEG", methods.get("peg_method", {}).get("fair_price"), methods.get("peg_method", {}).get("weight")],
                ["Analyst target", methods.get("analyst_target", {}).get("fair_price"), methods.get("analyst_target", {}).get("weight")],
            ],
            columns=["方法", "估值", "權重"],
        )
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption("This forecast is for reference only and is not investment advice")

    with tab4:
        news = pick_news or {}
        sentiment_label = str(news.get("sentiment_label", "No News"))
        sentiment_score = float(news.get("sentiment_score", 0) or 0)
        news_count = int(news.get("news_count", 0) or 0)
        pos = int(news.get("positive_count", 0) or 0)
        neu = int(news.get("neutral_count", 0) or 0)
        neg = int(news.get("negative_count", 0) or 0)

        if realtime_used:
            st.info("This is a live analysis, not today's scheduled run")

        # Top metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(
            f"<div style='font-size:16px;color:{get_sentiment_color(sentiment_label)};font-weight:700;'>"
            f"{sentiment_label}</div>",
            unsafe_allow_html=True,
        )
        c1.caption("Sentiment label")
        c2.metric("Sentiment score", f"{sentiment_score:.2f}")
        c3.metric("News count", news_count)
        c4.metric("Breakdown", f"positive {pos} / neutral {neu} / negative {neg}")

        # Sentiment gauge
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=sentiment_score,
                number={"valueformat": ".2f"},
                title={"text": "News sentiment score"},
                gauge={
                    "axis": {"range": [-1, 1]},
                    "bar": {"color": get_sentiment_color(sentiment_label)},
                    "steps": [
                        {"range": [-1, -0.2], "color": "rgba(211,47,47,0.35)"},
                        {"range": [-0.2, 0.2], "color": "rgba(158,158,158,0.35)"},
                        {"range": [0.2, 1], "color": "rgba(0,200,83,0.35)"},
                    ],
                    "threshold": {"line": {"color": "#FFFFFF", "width": 2}, "value": sentiment_score},
                },
            )
        )
        gauge.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(gauge, use_container_width=True)

        # AI news summary sits under the gauge and above the article list
        st.markdown("### 💡 AI news analysis")
        summary_raw = news.get("summary", "")
        summary = str(summary_raw).strip() if summary_raw is not None else ""

        if not summary:
            st.caption("No summary yet")
        elif summary.startswith("[Offline summary]"):
            st.caption("⚠️ Claude API unavailable; showing the template summary")
            st.info(summary.replace("[Offline summary] ", "", 1))
        else:
            summary_safe = html.escape(summary, quote=True)
            st.markdown(
                f"""
    <div style="
        background-color: rgba(59, 130, 246, 0.08);
        border-left: 4px solid #3b82f6;
        padding: 20px 24px;
        border-radius: 8px;
        margin: 16px 0;
    ">
        <div style="
            color: #e5e7eb;
            line-height: 1.8;
            font-size: 15px;
            white-space: pre-wrap;
        ">{summary_safe}</div>
    </div>
    """,
                unsafe_allow_html=True,
            )

        st.caption("📊 Method: Claude Haiku 4.5 + FinBERT sentiment + time-decay weights")
        st.markdown("---")

        st.subheader("Recent news (open the original)")
        analyzed_news = news.get("analyzed_news", []) if isinstance(news.get("analyzed_news"), list) else []
        analyzed_news = sorted(
            analyzed_news,
            key=lambda x: pd.to_datetime(x.get("published_at"), errors="coerce"),
            reverse=True,
        )
        if news_count == 0 or not analyzed_news:
            st.info("No related news yet")
        else:
            for one in analyzed_news:
                sent = str(one.get("sentiment", "neutral"))
                sent_cn = str(one.get("sentiment_cn", "Neutral"))
                sent_color = "#00C853" if sent == "positive" else "#FF5252" if sent == "negative" else "#9E9E9E"
                title = _truncate_text(str(one.get("title", "")), 80)
                publisher = str(one.get("publisher", ""))
                tstr = _fmt_news_time(one.get("published_at"))
                conf = float(one.get("confidence", 0) or 0)
                link = str(one.get("link", "") or "")
                with st.container(border=True):
                    st.markdown(
                        f"<div style='border-left:4px solid {sent_color}; padding-left:10px;'>"
                        f"<div style='font-weight:700;font-size:14px;'>{title}</div>"
                        f"<div style='color:#bdbdbd;font-size:12px;'>{publisher} · {tstr} · {sent_cn} · confidence {conf:.2f}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    if link:
                        st.link_button("📖 Read original", link, use_container_width=False)


def render_overview_page(daily: dict[str, Any]) -> None:
    """Page 3: stock overview."""
    st.title("📋 Stock overview")
    rows = daily.get("all_scores", [])
    if not rows:
        st.info("No overview data")
        return
    df = pd.DataFrame(rows).rename(
        columns={
            "ticker": "股票代號",
            "name": "公司",
            "composite_score": "綜合評分",
            "tech_score": "技術",
            "fund_score": "基本",
            "valuation_score": "估值",
            "prediction_return_pct": "預測",
        }
    )
    df["產業"] = df["股票代號"].apply(get_stock_sector)
    price_map = {}
    for p in daily.get("picks", []):
        price_map[p.get("ticker")] = p.get("current_price")
    df["價格"] = df["股票代號"].map(price_map)

    # Fill prices missing from all_scores via the data_collector cache
    missing_mask = df["價格"].isna()
    if missing_mask.any():
        missing_tickers = df.loc[missing_mask, "股票代號"].astype(str).tolist()
        filled_prices = {t: get_cached_current_price(t) for t in missing_tickers}
        df.loc[missing_mask, "價格"] = df.loc[missing_mask, "股票代號"].map(filled_prices)

    df = df[["股票代號", "公司", "價格", "綜合評分", "技術", "基本", "估值", "預測", "產業"]]
    df = df.sort_values("綜合評分", ascending=False).reset_index(drop=True)

    tickers_list = df["股票代號"].astype(str).tolist()
    entry_by_ticker: dict[str, dict[str, Any]] = {}
    for t_sym in tickers_list:
        er = compute_entry_analysis(t_sym)
        if er:
            entry_by_ticker[t_sym] = er

    def _entry_emoji(sym: str) -> str:
        e = entry_by_ticker.get(str(sym))
        if not e:
            return "⚪"
        return get_status_emoji(str(e.get("current_status") or "watch"))

    def _dist_standard_pct(sym: str) -> str:
        e = entry_by_ticker.get(str(sym))
        if not e:
            return "—"
        d = e.get("distance") if isinstance(e.get("distance"), dict) else {}
        v = _to_float_safe(d.get("to_standard_pct"))
        if v is None:
            return "—"
        return f"{v:+.1f}%"

    df["進場"] = df["股票代號"].map(_entry_emoji)
    df["距標準進場"] = df["股票代號"].map(_dist_standard_pct)
    df["_entry_status"] = df["股票代號"].astype(str).map(
        lambda x: str((entry_by_ticker.get(str(x)) or {}).get("current_status") or "")
    )

    keyword = st.text_input("Search ticker or company")
    sectors = sorted(df["產業"].dropna().unique().tolist())
    sector_sel = st.multiselect("Filter by sector", sectors)
    only_buy_now = st.checkbox("Only show 🟢 ready to enter (buy_now)", value=False)
    filtered = df.copy()
    if keyword:
        filtered = filtered[filtered["股票代號"].str.contains(keyword, case=False) | filtered["公司"].str.contains(keyword, case=False)]
    if sector_sel:
        filtered = filtered[filtered["產業"].isin(sector_sel)]
    if only_buy_now:
        filtered = filtered[filtered["_entry_status"] == "buy_now"]

    display_df = filtered.drop(columns=["_entry_status"], errors="ignore")

    # Prefer row-click navigation; fall back to a plain table plus a dropdown
    selected_ticker = None
    try:
        event = st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )
        if event and event.selection and event.selection.get("rows"):
            i = event.selection["rows"][0]
            selected_ticker = str(display_df.iloc[i]["股票代號"])
    except TypeError:
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    sym_opts = display_df["股票代號"].tolist()
    choice: str | None = None
    if sym_opts:
        choice = st.selectbox("Select a stock for the full analysis", sym_opts, index=0)
    else:
        st.caption("No stocks match the filter")
    if selected_ticker:
        choice = selected_ticker
    if choice and st.button("Open stock analysis"):
        st.session_state["selected_ticker"] = choice
        st.session_state["page"] = "🔬 個股深度分析"
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
    c1.metric("Recommendations", f"{total}")
    # Completion is progress, not performance, so turn off the red/green delta color
    c2.metric(
        "1M tracking complete",
        f"{done}",
        delta=f"{done / total:.0%} complete" if total else None,
        delta_color="off",
    )
    # Pass the same number as delta so Streamlit colors it by sign
    c3.metric("Strategy cumulative return", _fmt_pct(strat), delta=_fmt_pct(strat) if strat is not None else None)
    c4.metric("Excess return vs SPY", _fmt_pct(outperf), delta=_fmt_pct(outperf) if outperf is not None else None)
    if spy is not None:
        st.caption(f"SPY buy-and-hold over the same window: {_fmt_pct(spy)}")


def _render_perf_curve(cumulative: dict[str, Any] | str) -> None:
    """Performance page, section 2: strategy vs SPY cumulative-return chart."""
    st.subheader("Cumulative return")
    if not isinstance(cumulative, dict) or not cumulative.get("dates"):
        st.info("No equity curve yet (more completed tracking records are needed)")
        return

    dates = pd.to_datetime(pd.Series(cumulative["dates"]))
    strat_curve = cumulative.get("strategy_equity_curve") or []
    spy_curve = cumulative.get("spy_equity_curve") or []
    dr = cumulative.get("date_range") if isinstance(cumulative.get("date_range"), dict) else {}
    st.caption(
        f"Window: {dr.get('start', '-')} ~ {dr.get('end', '-')}"
        f" | {cumulative.get('trading_days', 0)} trading days"
        f" | each day compounds the equal-weight return of names then held; SPY is buy-and-hold"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=dates, y=strat_curve, mode="lines", name="Strategy", line=dict(color="#4cc9f0", width=2.5))
    )
    if spy_curve:
        fig.add_trace(
            go.Scatter(x=dates, y=spy_curve, mode="lines", name="SPY buy-and-hold", line=dict(color="#f4a261", width=2))
        )
    fig.add_hline(y=0, line_dash="dot", line_color="#9E9E9E")
    fig.update_layout(
        template="plotly_dark",
        height=460,
        hovermode="x unified",
        legend_orientation="h",
        yaxis=dict(title="Cumulative return", tickformat=".0%"),
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
        st.info("No attribution yet (more completed tracking records are needed)")
        return

    by_ticker = concentration["by_ticker"]
    rows = [
        {
            "標的": ticker,
            "入選次數": int(stats.get("picks") or 0),
            "平均 1M 報酬": _fmt_pct(stats.get("avg_return_1m")),
            "貢獻度": _fmt_pct(stats.get("contribution")),
            "_sort": _to_float_safe(stats.get("contribution"), 0.0),
        }
        for ticker, stats in by_ticker.items()
    ]
    table = pd.DataFrame(rows).sort_values("_sort", ascending=False).drop(columns=["_sort"])
    st.dataframe(table, use_container_width=True, hide_index=True)

    warning = concentration.get("concentration_warning")
    top = concentration.get("top_contributor") if isinstance(concentration.get("top_contributor"), dict) else {}
    if warning:
        share = _to_float_safe(top.get("share_of_outperformance"))
        pct_txt = f"{share * 100:.1f}" if share is not None else "—"
        st.warning(
            f"⚠️ Performance is concentrated in one name ({top.get('ticker', '—')} contributes {pct_txt}%). "
            "The sample is too narrow to treat the stock-picking result as consistent."
        )
    elif top:
        st.caption(f"Largest contributor: {top.get('ticker', '—')} ({_fmt_pct(top.get('contribution'))})")

    st.caption(
        "Contribution = full equity-curve return minus the return recomputed without that name. "
        "Equal weight means dropping one name raises the weight of the rest, so contributions "
        "do not sum to the total return. Use the figure to judge concentration, not as an exact split."
    )


def _render_perf_records() -> None:
    """Performance page, section 4: full recommendation history, filterable by tracking status."""
    st.subheader("Recommendation history")
    df = load_recommendations()
    if df.empty:
        st.info("No recommendations yet (run python src/track_record.py first)")
        return

    done_mask = pd.to_numeric(df.get("return_1m"), errors="coerce").notna()
    choice = st.radio("Tracking status", ["全部", "已完成", "追蹤中"], horizontal=True)
    if choice == "已完成":
        view = df[done_mask]
    elif choice == "追蹤中":
        view = df[~done_mask]
    else:
        view = df

    if view.empty:
        st.caption("No records match the filter")
        return

    view = view.copy()
    view["_d"] = pd.to_datetime(view["date"], errors="coerce")
    view = view.sort_values(["_d", "top_rank"], ascending=[False, True])

    out = pd.DataFrame(
        {
            "日期": view["date"].astype(str),
            "代號": view["ticker"].astype(str),
            "產業": view.get("sector", pd.Series(dtype=str)).astype(str),
            "排名": pd.to_numeric(view.get("top_rank"), errors="coerce"),
            "進場價": pd.to_numeric(view.get("entry_price"), errors="coerce").map(
                lambda v: f"${v:.2f}" if pd.notna(v) else "—"
            ),
            "綜合評分": pd.to_numeric(view.get("composite_score"), errors="coerce").round(1),
            "1D": view.get("return_1d").map(_fmt_pct),
            "1W": view.get("return_1w").map(_fmt_pct),
            "2W": view.get("return_2w").map(_fmt_pct),
            "1M": view.get("return_1m").map(_fmt_pct),
            "SPY 1M": view.get("spy_return_1m").map(_fmt_pct),
            "Alpha": view.get("alpha_1m").map(_fmt_pct),
            "狀態": view.get("status", pd.Series(dtype=str)).astype(str),
        }
    )
    st.dataframe(out, use_container_width=True, hide_index=True)
    st.caption(f"{len(out)} rows | returns use the entry price on the recommendation date; 1M is 30 calendar days")


def render_performance_page() -> None:
    """Page 4: strategy performance from the track record."""
    st.title("📊 Strategy performance")
    with st.spinner("Loading performance…"):
        perf = load_performance()
    if not perf:
        st.error("Could not load performance. Run python src/track_record.py first.")
        return

    summary = perf.get("summary") if isinstance(perf.get("summary"), dict) else {}
    cumulative = perf.get("cumulative", "insufficient data")
    concentration = perf.get("concentration", "insufficient data")

    snap = perf.get("snapshot_time")
    rng = summary.get("date_range") if isinstance(summary.get("date_range"), dict) else {}
    st.caption(
        f"Recommendation window: {rng.get('first', '-')} ~ {rng.get('last', '-')}"
        + (f" | snapshot: {snap}" if snap else " | computed just now")
    )

    _render_perf_metrics(summary, cumulative)
    if perf.get("note"):
        st.info(str(perf["note"]))
    st.divider()
    _render_perf_curve(cumulative)
    st.divider()
    _render_perf_concentration(concentration)
    st.divider()
    _render_perf_records()


def main() -> None:
    """Dashboard entry point."""
    if st.button("🔄 Refresh", help="Reload the data"):
        st.cache_data.clear()
        st.rerun()

    daily = load_latest_daily_json()
    if not daily:
        st.error("Run python src/daily_picker.py first to generate today's analysis")
        return

    st.sidebar.title("Pages")
    page_options = ["📊 每日精選", "🔬 個股深度分析", "📋 股票總覽", "📊 策略績效"]
    default_page = st.session_state.get("page", "📊 每日精選")
    page = st.sidebar.radio("Page", page_options, index=page_options.index(default_page) if default_page in page_options else 0)
    st.session_state["page"] = page
    st.sidebar.caption(f"Last update: {daily.get('date', '-')}")

    if page == "📊 每日精選":
        render_daily_page(daily)
    elif page == "🔬 個股深度分析":
        render_deep_page(daily)
    elif page == "📊 策略績效":
        render_performance_page()
    else:
        render_overview_page(daily)


if __name__ == "__main__":
    main()
