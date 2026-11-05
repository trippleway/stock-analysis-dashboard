# ── 60 支股票清單（按 AI 商業化六層次分類） ──

# Layer 1: AI Infrastructure（基礎設施）— 15 支
AI_INFRASTRUCTURE = {
    # GPU / AI 晶片
    "NVDA":  "NVIDIA",
    "AMD":   "AMD",
    "AVGO":  "Broadcom",
    "MRVL":  "Marvell",
    "ARM":   "Arm Holdings",
    
    # 記憶體與儲存
    "MU":    "Micron",
    "WDC":   "Western Digital",
    
    # 晶圓製造
    "TSM":   "TSMC",
    "ASML":  "ASML",
    "AMAT":  "Applied Materials",
    "LRCX":  "Lam Research",
    "KLAC":  "KLA Corporation",
    
    # AI 伺服器與冷卻
    "SMCI":  "Super Micro Computer",
    "DELL":  "Dell Technologies",
    "VRT":   "Vertiv (液冷)",
}

# Layer 2: Cloud & Compute（雲端運算）— 8 支
AI_CLOUD = {
    "MSFT":  "Microsoft (Azure)",
    "GOOGL": "Alphabet (GCP)",
    "AMZN":  "Amazon (AWS)",
    "ORCL":  "Oracle Cloud",
    "IBM":   "IBM",
    "CRWV":  "CoreWeave (AI 雲)",
    "NET":   "Cloudflare",
    "AKAM":  "Akamai",
}

# Layer 3: Foundation Models（基礎模型概念股）— 5 支
AI_FOUNDATION = {
    "META":  "Meta (Llama)",
    "BIDU":  "百度 (文心)",
    "TSLA":  "Tesla (自駕 AI)",
    "AAPL":  "Apple (端側 AI)",
    "PLTR":  "Palantir (國防 AI)",
}

# Layer 4: Data & AI Platforms（資料與 AI 平台）— 10 支
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

# Layer 5: AI Applications（AI 應用軟體）— 12 支
AI_APPLICATIONS = {
    "CRM":   "Salesforce (Einstein)",
    "NOW":   "ServiceNow",
    "ADBE":  "Adobe (Firefly)",
    "INTU":  "Intuit (AI 稅務)",
    "WDAY":  "Workday",
    "TEAM":  "Atlassian",
    "HUBS":  "HubSpot",
    "SHOP":  "Shopify",
    "TTD":   "The Trade Desk (AI 廣告)",
    "APP":   "AppLovin (AI 廣告)",
    "U":     "Unity (AI 3D)",
    "ZM":    "Zoom (AI Companion)",
}

# Layer 6: AI Consumer & Content（AI 消費與內容）— 10 支
AI_CONSUMER = {
    "SPOT":  "Spotify (AI 推薦)",
    "NFLX":  "Netflix (AI 內容)",
    "RBLX":  "Roblox (AI UGC)",
    "DUOL":  "Duolingo (AI 教學)",
    "DIS":   "Disney",
    "WBD":   "Warner Bros Discovery",
    "SONY":  "Sony Group",   # ← 你的目標公司
    "PINS":  "Pinterest",
    "SNAP":  "Snap (AI 濾鏡)",
    "RDDT":  "Reddit (AI 數據授權)",
}

# 合併
ALL_STOCKS = {
    **AI_INFRASTRUCTURE,
    **AI_CLOUD,
    **AI_FOUNDATION,
    **AI_DATA_PLATFORM,
    **AI_APPLICATIONS,
    **AI_CONSUMER,
}
TICKER_LIST = list(ALL_STOCKS.keys())  # 共 60 支

# AI 層次分類
AI_LAYERS = {
    "layer1_infrastructure": list(AI_INFRASTRUCTURE.keys()),
    "layer2_cloud":          list(AI_CLOUD.keys()),
    "layer3_foundation":     list(AI_FOUNDATION.keys()),
    "layer4_platform":       list(AI_DATA_PLATFORM.keys()),
    "layer5_applications":   list(AI_APPLICATIONS.keys()),
    "layer6_consumer":       list(AI_CONSUMER.keys()),
}

# 各層次在當前 AI 週期的基礎分（可調整）
LAYER_BASE_SCORE = {
    "layer1_infrastructure": 15,   # AI CapEx 高峰期
    "layer2_cloud":          15,
    "layer3_foundation":     10,
    "layer4_platform":       12,
    "layer5_applications":   14,   # 2025 開始爆發
    "layer6_consumer":       10,
}

# ── 產業分類（估值與基本面用）───────────────────────────
# 與 AI_LAYERS 的差異：AI_LAYERS 是「價值鏈位置」（投資敘事用），
# SECTOR_MAP 是「同業可比族群」（P/E 比較與產業基礎分用）。
# 兩者刻意不同：MSFT / GOOGL 在價值鏈屬 Layer 2 Cloud，但估值上與軟體股可比。
SECTOR_MAP = {
    # 晶片、記憶體、設備、AI 伺服器硬體
    "semiconductor": [
        "NVDA", "AMD", "AVGO", "MRVL", "ARM",
        "MU", "WDC", "TSM", "ASML", "AMAT",
        "LRCX", "KLAC", "SMCI", "DELL", "VRT",
    ],
    # 雲端、基礎模型、資料平台、AI SaaS（估值邏輯同屬高成長軟體）
    "ai_software": [
        "MSFT", "GOOGL", "AMZN", "ORCL", "IBM", "CRWV", "NET", "AKAM",
        "META", "BIDU", "PLTR",
        "SNOW", "DDOG", "MDB", "ESTC", "AI", "S", "PATH", "GTLB", "CFLT", "SOUN",
        "CRM", "NOW", "ADBE", "INTU", "WDAY", "TEAM", "HUBS", "SHOP", "TTD", "APP", "U", "ZM",
    ],
    # 消費電子、車用與消費內容娛樂
    "consumer": [
        "AAPL", "TSLA",
        "SPOT", "NFLX", "RBLX", "DUOL", "DIS", "WBD", "SONY", "PINS", "SNAP", "RDDT",
    ],
}

# 產業基礎分加分（fundamentals 以 50 + 本值作為產業前景基礎分）
SECTOR_BASE_SCORE = {
    "semiconductor": 15,   # AI CapEx 高峰期最直接受惠
    "ai_software":   14,   # AI 商業化落地中，成長性高
    "consumer":      10,   # AI 影響較間接
}

# ── 綜合評分權重與篩選門檻 ─────────────────────────────
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

# ── 回測固定區間 ───────────────────────────────────────
# 這是履歷引用數字的固定區間，修改前必須重新驗收。
from datetime import date as _date

# 評估起日。第一個換股日是之後的月底 2023-09-29，第一個持有日是 2023-10-02
BACKTEST_START = _date(2023, 9, 22)
BACKTEST_END = _date(2026, 9, 21)
# 組 A 名單判斷「曾經進過 S&P 500」的迄日
GROUP_DEFINITION_END = _date(2026, 9, 21)

# ── 技術指標參數與技術面子分數權重 ─────────────────────
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

# 技術面五項子分數權重（總和需為 1.0）
TECH_SCORE_WEIGHTS = {
    "macd":      0.25,   # 趨勢轉折訊號
    "ma_trend":  0.25,   # 多空排列
    "rsi":       0.20,   # 超買超賣
    "bollinger": 0.15,   # 波動位置
    "volume":    0.15,   # 量能確認
}

# ── 估值參數 ───────────────────────────────────────────
# 各產業的本益比區間（low 便宜 / mid 合理 / high 偏貴）
VALUATION_PARAMS = {
    "pe_benchmarks": {
        "semiconductor": {"low": 18, "mid": 28, "high": 40},
        "ai_software":   {"low": 25, "mid": 38, "high": 55},
        "consumer":      {"low": 15, "mid": 22, "high": 30},
    },
}

# ── 預測參數（Prophet 與線性備援共用）──────────────────
PREDICTION_PARAMS = {
    "history_days":              365,
    "forecast_days":             30,
    "confidence_interval":       0.95,
    "prophet_changepoint_prior": 0.05,
}

# ── 路徑與資料目錄 ─────────────────────────────────────
import pathlib as _pathlib

_PROJECT_ROOT = _pathlib.Path(__file__).resolve().parent
DATA_DIR = _PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
PICKS_DIR = DATA_DIR / "daily_picks"
LOG_DIR = _PROJECT_ROOT / "logs"
TRACK_RECORD_DIR = DATA_DIR / "track_record"

for _d in (DATA_DIR, CACHE_DIR, PICKS_DIR, LOG_DIR, TRACK_RECORD_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── API 金鑰（從 .env 讀取）─────────────────────────────
import os as _os

try:
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv(_PROJECT_ROOT / ".env")
except Exception:  # noqa: BLE001
    pass  # 未安裝 python-dotenv 時仍可從系統環境變數讀取

# 未設定時為 None；sentiment 與 profile_fetcher 會自動退回模板版輸出
ANTHROPIC_API_KEY = _os.getenv("ANTHROPIC_API_KEY") or None
ALPHA_VANTAGE_KEY = _os.getenv("ALPHA_VANTAGE_KEY") or None
FMP_API_KEY = _os.getenv("FMP_API_KEY") or None
FRED_API_KEY = _os.getenv("FRED_API_KEY") or None
