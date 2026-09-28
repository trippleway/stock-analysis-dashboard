# -*- coding: utf-8 -*-
"""
使用 FinBERT 進行新聞情緒分析（含每日快取）。

執行方式（於專案根目錄）：
    python src/sentiment.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

# 抑制 transformers 訊息
logging.getLogger("transformers").setLevel(logging.ERROR)

# 從專案根目錄載入 config（直接執行本檔時 sys.path 預設為 src/）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import ANTHROPIC_API_KEY, CACHE_DIR  # noqa: E402
from news_fetcher import fetch_stock_news  # noqa: E402

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment, misc]

_classifier: Any | None = None
_log = logging.getLogger("sentiment")
if not _log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _get_classifier():
    """延遲初始化 FinBERT。"""
    global _classifier
    if _classifier is None:
        print("[sentiment] 首次使用，正在載入 FinBERT 模型...")
        from transformers import pipeline

        _classifier = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=-1,
        )
        print("[sentiment] 模型載入完成")
    return _classifier


def _cache_path(ticker: str) -> Path:
    """回傳當日情緒快取檔案路徑。"""
    d = datetime.now().strftime("%Y-%m-%d")
    p = Path(CACHE_DIR) / d / "sentiment"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{ticker.upper()}.json"


def _parse_published_at(value: Any) -> datetime | None:
    """解析 published_at（YYYY-MM-DD HH:MM:SS 等格式）到 datetime。"""
    if value is None:
        return None
    try:
        if isinstance(value, str):
            v = value.strip()
            if not v:
                return None
            # 優先解析 fetcher 統一輸出格式
            try:
                dt = datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                if v.endswith("Z"):
                    v = v[:-1] + "+00:00"
                dt = datetime.fromisoformat(v)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
    except Exception:  # noqa: BLE001
        return None
    return None


def _time_weight(days_ago: float) -> float:
    """依新聞距今時間回傳衰減權重。"""
    if days_ago <= 1:
        return 1.0
    if days_ago <= 3:
        return 0.7
    if days_ago <= 7:
        return 0.4
    return 0.2


def _label_cn(label: str) -> str:
    """英文情緒標籤轉中文。"""
    if label == "positive":
        return "正面"
    if label == "negative":
        return "負面"
    return "中性"


def _sentiment_label(score: float) -> str:
    """總體情緒分數轉中文描述。"""
    if score > 0.5:
        return "強烈看漲"
    if score > 0.2:
        return "偏多"
    if score >= -0.2:
        return "中性"
    if score >= -0.5:
        return "偏空"
    return "強烈看空"


def generate_news_summary(sentiment_result: dict[str, Any], ticker: str) -> str:
    """
    模板版新聞摘要（無 LLM 或離線時使用）。
    依情緒分數、標籤與新聞標題組成簡短繁體中文說明。
    """
    sym = str(ticker).upper()
    if int(sentiment_result.get("news_count", 0) or 0) == 0:
        return "近期無相關新聞，建議以基本面和技術面為主要判斷依據。"

    label = str(sentiment_result.get("sentiment_label", "中性"))
    score = float(sentiment_result.get("sentiment_score", 0) or 0)
    pos = int(sentiment_result.get("positive_count", 0) or 0)
    neu = int(sentiment_result.get("neutral_count", 0) or 0)
    neg = int(sentiment_result.get("negative_count", 0) or 0)
    n = int(sentiment_result.get("news_count", 0) or 0)

    tops = sentiment_result.get("top_headlines") or sentiment_result.get("analyzed_news") or []
    titles: list[str] = []
    if isinstance(tops, list):
        for it in tops[:3]:
            if isinstance(it, dict) and it.get("title"):
                titles.append(str(it["title"])[:60])

    title_part = ""
    if titles:
        title_part = "近期標題包含「" + "」「".join(titles) + "」等。"

    return (
        f"{sym} 近一週共 {n} 則新聞，整體情緒為「{label}」（情緒分數 {score:+.2f}）。"
        f"新聞分佈為正面 {pos} 則、中性 {neu} 則、負面 {neg} 則。{title_part}"
        f"建議搭配基本面與技術面綜合評估，並留意單則新聞可能造成的短期波動。"
    )


def generate_news_summary_with_llm(
    sentiment_result: dict[str, Any],
    ticker: str,
    company_name: str = "",
) -> str:
    """使用 Claude Haiku 4.5 生成中文新聞摘要。失敗時 fallback 到模板版。"""

    # 無新聞直接回傳
    if int(sentiment_result.get("news_count", 0) or 0) == 0:
        return "近期無相關新聞，建議以基本面和技術面為主要判斷依據。"

    # 無 API key 用模板版
    key = (ANTHROPIC_API_KEY or "").strip()
    if not key:
        return generate_news_summary(sentiment_result, ticker)

    # 檢查當日快取
    cache_dir = Path(CACHE_DIR) / datetime.now().strftime("%Y-%m-%d") / "summaries"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{ticker.upper()}.txt"
    if cache_file.is_file():
        return cache_file.read_text(encoding="utf-8")

    # 組裝新聞清單
    news_lines: list[str] = []
    analyzed = sentiment_result.get("analyzed_news", [])
    if not isinstance(analyzed, list):
        analyzed = []
    for news in analyzed[:10]:
        if not isinstance(news, dict):
            continue
        sent_cn = str(news.get("sentiment_cn", "中性") or "中性")
        try:
            conf = float(news.get("confidence", 0) or 0)
        except (TypeError, ValueError):
            conf = 0.0
        title = str(news.get("title", "") or "")
        pub = str(news.get("publisher", "") or "")
        pat = str(news.get("published_at", "") or "")
        date_short = pat[:10] if len(pat) >= 10 else pat
        news_lines.append(
            f"- [{sent_cn}，信心 {conf:.2f}] {title} ({pub}, {date_short})"
        )
    news_text = "\n".join(news_lines) if news_lines else "（無可列舉標題）"

    system_prompt = """你是一位專業的金融新聞分析師，擅長將多則新聞整合成簡潔有洞察力的投資觀察。

請根據提供的新聞清單與情緒分析結果，生成一段 150-200 字的中文新聞概覽。

格式要求：
- 第一段：簡述整體情緒傾向與新聞數量
- 第二段：若有正面消息，用一段話綜合 2-3 個主題（不要條列標題）
- 第三段：若有負面消息或風險點，用一段話說明
- 結尾：給出中性的投資觀察

語氣要求：
- 繁體中文
- 專業但易讀
- 避免「應該買進」「強力推薦」這類建議
- 用「可關注」「值得留意」等中性詞彙
- 不要虛構未出現在新聞中的資訊

長度 150-250 字，用空行分段。"""

    sym = ticker.upper()
    name_part = f" ({company_name})" if company_name else ""
    user_message = f"""股票：{sym}{name_part}

整體情緒分數：{float(sentiment_result.get('sentiment_score', 0) or 0):+.2f}（{sentiment_result.get('sentiment_label', '中性')}）
新聞統計：正面 {int(sentiment_result.get('positive_count', 0) or 0)} 則 / 中性 {int(sentiment_result.get('neutral_count', 0) or 0)} 則 / 負面 {int(sentiment_result.get('negative_count', 0) or 0)} 則

新聞清單（按影響力排序）：
{news_text}

請生成新聞概覽。"""

    if anthropic is None:
        return generate_news_summary(sentiment_result, ticker)

    try:
        client = anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        block = response.content[0]
        summary = getattr(block, "text", str(block)).strip()

        cache_file.write_text(summary, encoding="utf-8")
        return summary

    except Exception as e:  # noqa: BLE001
        _log.warning("%s LLM 摘要生成失敗: %s", sym, e)
        return "[離線摘要] " + generate_news_summary(sentiment_result, ticker)


def analyze_news_sentiment(news_items: list[dict[str, Any]]) -> dict[str, Any]:
    """
    對新聞標題做 FinBERT 情緒分析，輸出整體分數與逐條結果。
    """
    if not news_items:
        return {
            "sentiment_score": 0.0,
            "sentiment_label": "中性",
            "news_count": 0,
            "positive_count": 0,
            "neutral_count": 0,
            "negative_count": 0,
            "analyzed_news": [],
            "top_headlines": [],
            "score": 50.0,
        }

    clf = _get_classifier()
    now_utc = datetime.now(timezone.utc)
    analyzed_news: list[dict[str, Any]] = []

    weighted_sum = 0.0
    weight_total = 0.0
    positive_count = 0
    neutral_count = 0
    negative_count = 0

    for item in news_items:
        title = str(item.get("title", "") or "")
        title_for_model = title[:256]  # BERT 輸入長度控制
        link = str(item.get("link", "") or "")
        publisher = str(item.get("publisher", "") or "")
        published_at = str(item.get("published_at", "") or "")

        dt = _parse_published_at(published_at)
        days_ago = (now_utc - dt).total_seconds() / 86400.0 if dt else 99.0
        w = _time_weight(days_ago)

        # 單則分析保護，失敗視為 neutral
        label = "neutral"
        conf = 0.0
        try:
            pred = clf(title_for_model, truncation=True, max_length=256)
            if isinstance(pred, list) and pred:
                one = pred[0]
                label = str(one.get("label", "neutral")).lower()
                conf = float(one.get("score", 0.0))
        except Exception:  # noqa: BLE001
            label = "neutral"
            conf = 0.0

        if label not in ("positive", "neutral", "negative"):
            label = "neutral"

        if label == "positive":
            val = conf
            positive_count += 1
        elif label == "negative":
            val = -conf
            negative_count += 1
        else:
            val = 0.0
            neutral_count += 1

        weighted_sum += val * w
        weight_total += w

        analyzed_news.append(
            {
                "title": title,
                "publisher": publisher,
                "link": link,
                "published_at": published_at,
                "sentiment": label,
                "sentiment_cn": _label_cn(label),
                "confidence": round(conf, 4),
                "weight": round(w, 4),
            }
        )

    sentiment_score = weighted_sum / weight_total if weight_total > 0 else 0.0
    sentiment_score = max(-1.0, min(1.0, sentiment_score))
    overall_label = _sentiment_label(sentiment_score)
    score_0100 = (sentiment_score + 1.0) * 50.0

    top_headlines = sorted(
        analyzed_news,
        key=lambda x: abs(float(x.get("confidence", 0.0)) * float(x.get("weight", 0.0))),
        reverse=True,
    )[:5]

    return {
        "sentiment_score": round(float(sentiment_score), 4),
        "sentiment_label": overall_label,
        "news_count": len(analyzed_news),
        "positive_count": positive_count,
        "neutral_count": neutral_count,
        "negative_count": negative_count,
        "analyzed_news": analyzed_news,
        "top_headlines": top_headlines,
        "score": round(float(score_0100), 2),
    }


def _empty_sentiment_result(ticker: str, label: str = "無新聞") -> dict[str, Any]:
    """無新聞或失敗時的預設回傳。"""
    return {
        "ticker": ticker.upper(),
        "score": 50.0,
        "sentiment_score": 0.0,
        "sentiment_label": label,
        "news_count": 0,
        "positive_count": 0,
        "neutral_count": 0,
        "negative_count": 0,
        "analyzed_news": [],
        "top_headlines": [],
    }


def analyze_ticker_sentiment(ticker: str, max_items: int = 10) -> dict[str, Any]:
    """
    整合新聞抓取與 FinBERT 分析（含當日快取）。
    """
    sym = ticker.upper()
    cache = _cache_path(sym)
    if cache.is_file():
        try:
            with open(cache, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                if "summary" not in data:
                    data["summary"] = generate_news_summary_with_llm(data, sym)
                    try:
                        with open(cache, "w", encoding="utf-8") as fw:
                            json.dump(data, fw, ensure_ascii=False, indent=2)
                    except Exception:  # noqa: BLE001
                        pass
                return data
        except Exception:  # noqa: BLE001
            pass

    news = fetch_stock_news(sym, max_items=max_items, force_refresh=False)
    if not news:
        result = _empty_sentiment_result(sym, label="無新聞")
        result["summary"] = generate_news_summary_with_llm(result, sym)
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    out = analyze_news_sentiment(news)
    out["ticker"] = sym
    out["summary"] = generate_news_summary_with_llm(out, sym)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def analyze_sentiment_batch(tickers: list[str], max_per_ticker: int = 8) -> dict[str, dict[str, Any]]:
    """
    批量情緒分析（失敗股票以預設值補上）。
    """
    out: dict[str, dict[str, Any]] = {}
    for t in tqdm([str(x).upper() for x in tickers], desc="情緒分析", unit="檔"):
        try:
            out[t] = analyze_ticker_sentiment(t, max_items=max_per_ticker)
        except Exception as e:  # noqa: BLE001
            _log.error("%s 情緒分析失敗：%s", t, e)
            fail = _empty_sentiment_result(t, label="分析失敗")
            fail["summary"] = generate_news_summary_with_llm(fail, t)
            out[t] = fail
    return out


if __name__ == "__main__":
    print("FinBERT 模型下載約需 5 分鐘")
    sample = ["NVDA", "TSM"]
    results = analyze_sentiment_batch(sample, max_per_ticker=10)
    for t in sample:
        r = results.get(t, _empty_sentiment_result(t))
        print(f"\n=== {t} ===")
        print(f"情緒分數: {r['sentiment_score']:.4f} | 標籤: {r['sentiment_label']} | score: {r['score']}")
        print(
            f"新聞統計: 正面 {r['positive_count']} / 中性 {r['neutral_count']} / 負面 {r['negative_count']} "
            f"(共 {r['news_count']} 則)"
        )
        print("Top 3 影響新聞:")
        for n in r.get("top_headlines", [])[:3]:
            print(
                f"- {n.get('title','')}\n"
                f"  情緒: {n.get('sentiment_cn')} ({n.get('sentiment')}) | "
                f"信心: {n.get('confidence')} | 連結: {n.get('link')}"
            )
        print("\n📊 LLM 新聞摘要：")
        print(r.get("summary", "無摘要"))
        print("─" * 60)
