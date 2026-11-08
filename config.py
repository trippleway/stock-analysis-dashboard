# ── 60-name universe, grouped by the six layers of AI commercialization ──

# Layer 1 Infrastructure — 15 names
AI_INFRASTRUCTURE = {
    # GPU / AI chips
    "NVDA":  "NVIDIA",
    "AMD":   "AMD",
    "AVGO":  "Broadcom",
    "MRVL":  "Marvell",
    "ARM":   "Arm Holdings",
    
    # Memory and storage
    "MU":    "Micron",
    "WDC":   "Western Digital",
    
    # Wafer fabrication
    "TSM":   "TSMC",
    "ASML":  "ASML",
    "AMAT":  "Applied Materials",
    "LRCX":  "Lam Research",
    "KLAC":  "KLA Corporation",
    
    # AI servers and cooling
    "SMCI":  "Super Micro Computer",
    "DELL":  "Dell Technologies",
    "VRT":   "Vertiv (liquid cooling)",
}

# Layer 2 Cloud & compute — 8 names
AI_CLOUD = {
    "MSFT":  "Microsoft (Azure)",
    "GOOGL": "Alphabet (GCP)",
    "AMZN":  "Amazon (AWS)",
    "ORCL":  "Oracle Cloud",
    "IBM":   "IBM",
    "CRWV":  "CoreWeave (AI cloud)",
    "NET":   "Cloudflare",
    "AKAM":  "Akamai",
}

# Layer 3 Foundation models — 5 names
AI_FOUNDATION = {
    "META":  "Meta (Llama)",
    "BIDU":  "Baidu (ERNIE)",
    "TSLA":  "Tesla (autonomous-driving AI)",
    "AAPL":  "Apple (on-device AI)",
    "PLTR":  "Palantir (defense AI)",
}

# Layer 4 Data platforms — 10 names
AI_DATA_PLATFORM = {
    "SNOW":  "Snowflake",
    "DDOG":  "Datadog",
    "MDB":   "MongoDB",
    "ESTC":  "Elastic",
    "AI":    "C3.ai",
    "S":     "SentinelOne",
    "PATH":  "UiPath",
    "GTLB":  "GitLab",
    "CFLT":  "Confluent",
    "SOUN":  "SoundHound AI",
}

# Layer 5 AI applications — 12 names
AI_APPLICATIONS = {
    "CRM":   "Salesforce (Einstein)",
    "NOW":   "ServiceNow",
    "ADBE":  "Adobe (Firefly)",
    "INTU":  "Intuit (AI tax)",
    "WDAY":  "Workday",
    "TEAM":  "Atlassian",
    "HUBS":  "HubSpot",
    "SHOP":  "Shopify",
    "TTD":   "The Trade Desk (AI advertising)",
    "APP":   "AppLovin (AI advertising)",
    "U":     "Unity (AI 3D)",
    "ZM":    "Zoom (AI Companion)",
}

# Layer 6 Consumer & content — 10 names
AI_CONSUMER = {
    "SPOT":  "Spotify (AI recommendations)",
    "NFLX":  "Netflix (AI content)",
    "RBLX":  "Roblox (AI UGC)",
    "DUOL":  "Duolingo (AI learning)",
    "DIS":   "Disney",
    "WBD":   "Warner Bros Discovery",
    "SONY":  "Sony Group",   # target company
    "PINS":  "Pinterest",
    "SNAP":  "Snap (AI filters)",
    "RDDT":  "Reddit (AI data licensing)",
}

# Combined universe
ALL_STOCKS = {
    **AI_INFRASTRUCTURE,
    **AI_CLOUD,
    **AI_FOUNDATION,
    **AI_DATA_PLATFORM,
    **AI_APPLICATIONS,
    **AI_CONSUMER,
}
TICKER_LIST = list(ALL_STOCKS.keys())  # 60 names

# AI layer membership
AI_LAYERS = {
    "layer1_infrastructure": list(AI_INFRASTRUCTURE.keys()),
    "layer2_cloud":          list(AI_CLOUD.keys()),
    "layer3_foundation":     list(AI_FOUNDATION.keys()),
    "layer4_platform":       list(AI_DATA_PLATFORM.keys()),
    "layer5_applications":   list(AI_APPLICATIONS.keys()),
    "layer6_consumer":       list(AI_CONSUMER.keys()),
}

# Base points for each layer in the current AI cycle (adjustable)
LAYER_BASE_SCORE = {
    "layer1_infrastructure": 15,   # peak AI CapEx
    "layer2_cloud":          15,
    "layer3_foundation":     10,
    "layer4_platform":       12,
    "layer5_applications":   14,   # inflection starting in 2025
    "layer6_consumer":       10,
}

# ── Sector groups (valuation and fundamentals) ─────────
# AI_LAYERS is the value-chain position used in the investment narrative.
# SECTOR_MAP is the peer group used for P/E comparisons and the sector base score.
# They differ on purpose: MSFT / GOOGL sit in Layer 2 Cloud & compute,
# but for valuation they are compared with software names.
SECTOR_MAP = {
    # Chips, memory, equipment, and AI server hardware
    "semiconductor": [
        "NVDA", "AMD", "AVGO", "MRVL", "ARM",
        "MU", "WDC", "TSM", "ASML", "AMAT",
        "LRCX", "KLAC", "SMCI", "DELL", "VRT",
    ],
    # Cloud, foundation models, data platforms, and AI SaaS (same high-growth software valuation logic)
    "ai_software": [
        "MSFT", "GOOGL", "AMZN", "ORCL", "IBM", "CRWV", "NET", "AKAM",
        "META", "BIDU", "PLTR",
        "SNOW", "DDOG", "MDB", "ESTC", "AI", "S", "PATH", "GTLB", "CFLT", "SOUN",
        "CRM", "NOW", "ADBE", "INTU", "WDAY", "TEAM", "HUBS", "SHOP", "TTD", "APP", "U", "ZM",
    ],
    # Consumer electronics, autos, and consumer content
    "consumer": [
        "AAPL", "TSLA",
        "SPOT", "NFLX", "RBLX", "DUOL", "DIS", "WBD", "SONY", "PINS", "SNAP", "RDDT",
    ],
}

# Sector bonus. fundamentals uses 50 + this value as the industry-outlook base.
SECTOR_BASE_SCORE = {
    "semiconductor": 15,   # most direct beneficiary of peak AI CapEx
    "ai_software":   14,   # AI commercialization is underway; growth is high
    "consumer":      10,   # AI exposure is more indirect
}

# ── Composite score weights and filter thresholds ──────
SCORE_WEIGHTS = {
    "technical":   0.25,
    "fundamental": 0.25,
    "valuation":   0.20,
    "prediction":  0.10,
    "volume":      0.10,
    "news":        0.10,
}

FILTER_THRESHOLDS = {
    "min_tech_score":   50,
    "min_fund_score":   40,
    "min_pred_return":  0.0,
    "min_volume_ratio": 0.5,
}

# ── Fixed backtest window ──────────────────────────────
# The rule was set before any performance was seen and does not move with the run date.
# Re-validate before changing it.
# 36 full months. The first rebalance is the last trading day of
# BACKTEST_FIRST_REBALANCE_MONTH. After that, rebalance on the last trading
# day of each month and hold through the next month-end.
# The last holding day is the last trading day of BACKTEST_LAST_MONTH; there is
# no rebalance that day.
# BACKTEST_START / BACKTEST_END are the calendar bounds of that rule, derived
# from the three constants below. Actual rebalance and holding dates then snap
# to month-end on the trading calendar.
from datetime import date as _date
from datetime import timedelta as _timedelta


def _parse_year_month(token: str) -> tuple[int, int]:
    """Split YYYY-MM into year and month. Reject a bad format or a month outside 1–12."""
    year_text, month_text = str(token).split("-", 1)
    year, month = int(year_text), int(month_text)
    if month < 1 or month > 12:
        raise ValueError(f"Month must be between 1 and 12, got {token}")
    return year, month


def _month_index(year: int, month: int) -> int:
    """Turn a year and month into a month index that can be subtracted."""
    return year * 12 + month - 1


def _calendar_month_end(year: int, month: int) -> _date:
    """Last calendar day of the month. The trading-calendar month-end is chosen by the backtest."""
    if month == 12:
        next_month = _date(year + 1, 1, 1)
    else:
        next_month = _date(year, month + 1, 1)
    return next_month - _timedelta(days=1)


# First rebalance month, last holding month, and the number of full months. Together they fix the window.
BACKTEST_FIRST_REBALANCE_MONTH = "2023-09"
BACKTEST_LAST_MONTH = "2026-09"
BACKTEST_MONTHS = 36

_start_year, _start_month = _parse_year_month(BACKTEST_FIRST_REBALANCE_MONTH)
_end_year, _end_month = _parse_year_month(BACKTEST_LAST_MONTH)
_month_span = _month_index(_end_year, _end_month) - _month_index(_start_year, _start_month)
if _month_span != BACKTEST_MONTHS:
    raise ValueError(
        f"{BACKTEST_FIRST_REBALANCE_MONTH} to {BACKTEST_LAST_MONTH} spans {_month_span} months, "
        f"which does not match BACKTEST_MONTHS={BACKTEST_MONTHS}"
    )

# The bounds include the whole first rebalance month and the whole last holding month.
BACKTEST_START = _date(_start_year, _start_month, 1)
BACKTEST_END = _calendar_month_end(_end_year, _end_month)
# End date for deciding whether a name was ever in the S&P 500 for Group A. This defines the list, not the backtest window.
GROUP_DEFINITION_END = _date(2026, 9, 21)

# ── Technical-indicator parameters and sub-score weights ─
TECH_PARAMS = {
    "ma_periods":        [5, 20, 60, 200],
    "rsi_period":        14,
    "macd_fast":         12,
    "macd_slow":         26,
    "macd_signal":       9,
    "bb_period":         20,
    "bb_std":            2.0,
    "atr_period":        14,
    "kd_period":         9,
    "volume_ma_period":  20,
    "roc_period":        12,
    "adx_period":        14,
}

# Weights for the five technical sub-scores (must sum to 1.0)
TECH_SCORE_WEIGHTS = {
    "macd":      0.25,   # trend-turn signal
    "ma_trend":  0.25,   # moving-average alignment
    "rsi":       0.20,   # overbought / oversold
    "bollinger": 0.15,   # position inside the volatility band
    "volume":    0.15,   # volume confirmation
}

# ── Valuation parameters ───────────────────────────────
# P/E bands by sector (low = cheap, mid = fair, high = expensive)
VALUATION_PARAMS = {
    "pe_benchmarks": {
        "semiconductor": {"low": 18, "mid": 28, "high": 40},
        "ai_software":   {"low": 25, "mid": 38, "high": 55},
        "consumer":      {"low": 15, "mid": 22, "high": 30},
    },
    # Fractions, not percentages. Outside this band the trailing-P/E fair value is not trusted.
    "max_plausible_upside": 1.00,      # +100%
    "max_plausible_downside": -0.50,   # -50%
}

# ── Forecast parameters (shared by Prophet and the linear fallback) ─
PREDICTION_PARAMS = {
    "history_days":              365,
    "forecast_days":             30,
    "confidence_interval":       0.95,
    "prophet_changepoint_prior": 0.05,
}

# ── Paths and data directories ─────────────────────────
import pathlib as _pathlib

_PROJECT_ROOT = _pathlib.Path(__file__).resolve().parent
DATA_DIR = _PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
PICKS_DIR = DATA_DIR / "daily_picks"
LOG_DIR = _PROJECT_ROOT / "logs"
TRACK_RECORD_DIR = DATA_DIR / "track_record"

for _d in (DATA_DIR, CACHE_DIR, PICKS_DIR, LOG_DIR, TRACK_RECORD_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── API keys (read from .env) ──────────────────────────
import os as _os

try:
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv(_PROJECT_ROOT / ".env")
except Exception:  # noqa: BLE001
    pass  # Without python-dotenv, keys can still come from the process environment

# None when unset. sentiment and profile_fetcher then fall back to template output
ANTHROPIC_API_KEY = _os.getenv("ANTHROPIC_API_KEY") or None
ALPHA_VANTAGE_KEY = _os.getenv("ALPHA_VANTAGE_KEY") or None
FMP_API_KEY = _os.getenv("FMP_API_KEY") or None
FRED_API_KEY = _os.getenv("FRED_API_KEY") or None
