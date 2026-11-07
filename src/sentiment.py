# -*- coding: utf-8 -*-
"""
FinBERT news-sentiment analysis, with a same-day cache.

Run from the project root:
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

# Quiet transformers logs
logging.getLogger("transformers").setLevel(logging.ERROR)

# Load config from the project root (running this file directly puts src/ on sys.path)
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
    """Load FinBERT on first use."""
    global _classifier
    if _classifier is None:
        print("[sentiment] First use: loading the FinBERT model...")
        from transformers import pipeline

        _classifier = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=-1,
        )
        print("[sentiment] Model loaded")
    return _classifier


def _cache_path(ticker: str) -> Path:
    """Path of today's sentiment cache file."""
    d = datetime.now().strftime("%Y-%m-%d")
    p = Path(CACHE_DIR) / d / "sentiment"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{ticker.upper()}.json"


def _parse_published_at(value: Any) -> datetime | None:
    """Parse published_at (YYYY-MM-DD HH:MM:SS and similar) into a datetime."""
    if value is None:
        return None
    try:
        if isinstance(value, str):
            v = value.strip()
            if not v:
                return None
            # Prefer the fetcher's unified timestamp format
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
    """Decay weight from how many days ago the article was published."""
    if days_ago <= 1:
        return 1.0
    if days_ago <= 3:
        return 0.7
    if days_ago <= 7:
        return 0.4
    return 0.2


def _label_cn(label: str) -> str:
    """Map a FinBERT label to the per-article display word."""
    if label == "positive":
        return "Positive"
    if label == "negative":
        return "Negative"
    return "Neutral"


def _sentiment_label(score: float) -> str:
    """Map the aggregate sentiment score to the overall label."""
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
    Template news summary, used when there is no LLM or the call fails.
    Built from the sentiment score, the label, and recent headlines.
    """
    sym = str(ticker).upper()
    if int(sentiment_result.get("news_count", 0) or 0) == 0:
        return "No recent news. Use fundamentals and technicals as the main inputs."

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
        title_part = " Recent headlines include " + "; ".join(titles) + "."

    return (
        f"{sym} has {n} articles in the recent window. Overall sentiment is \"{label}\" "
        f"(score {score:+.2f}). "
        f"Breakdown: {pos} positive, {neu} neutral, {neg} negative.{title_part} "
        f"Read this together with fundamentals and technicals. A single article can move the stock in the short term."
    )


def generate_news_summary_with_llm(
    sentiment_result: dict[str, Any],
    ticker: str,
    company_name: str = "",
) -> str:
    """Ask Claude Haiku 4.5 for an English news summary. Fall back to the template on failure."""

    # No articles: return immediately
    if int(sentiment_result.get("news_count", 0) or 0) == 0:
        return "No recent news. Use fundamentals and technicals as the main inputs."

    # No API key: use the template
    key = (ANTHROPIC_API_KEY or "").strip()
    if not key:
        return generate_news_summary(sentiment_result, ticker)

    # Same-day cache
    cache_dir = Path(CACHE_DIR) / datetime.now().strftime("%Y-%m-%d") / "summaries"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{ticker.upper()}.txt"
    if cache_file.is_file():
        return cache_file.read_text(encoding="utf-8")

    # Build the article list
    news_lines: list[str] = []
    analyzed = sentiment_result.get("analyzed_news", [])
    if not isinstance(analyzed, list):
        analyzed = []
    for news in analyzed[:10]:
        if not isinstance(news, dict):
            continue
        sent_cn = str(news.get("sentiment_cn", "Neutral") or "Neutral")
        try:
            conf = float(news.get("confidence", 0) or 0)
        except (TypeError, ValueError):
            conf = 0.0
        title = str(news.get("title", "") or "")
        pub = str(news.get("publisher", "") or "")
        pat = str(news.get("published_at", "") or "")
        date_short = pat[:10] if len(pat) >= 10 else pat
        news_lines.append(
            f"- [{sent_cn}, confidence {conf:.2f}] {title} ({pub}, {date_short})"
        )
    news_text = "\n".join(news_lines) if news_lines else "(no headlines to list)"

    system_prompt = """You are a financial news analyst. You turn a set of articles into a short, useful investment note.

Using the article list and the sentiment results, write an English news overview of about 150-200 words.

Format:
- First paragraph: overall sentiment and how many articles there are
- Second paragraph: if there is positive news, combine two or three themes in prose (do not list headlines)
- Third paragraph: if there is negative news or a risk, explain it in one paragraph
- Close with a neutral observation

Tone:
- English
- Professional and easy to read
- Do not say "buy" or "strong buy"
- Use neutral wording such as "worth watching"
- Do not invent facts that are not in the articles

About 150-250 words, with blank lines between paragraphs."""

    sym = ticker.upper()
    name_part = f" ({company_name})" if company_name else ""
    user_message = f"""Stock: {sym}{name_part}

Overall sentiment score: {float(sentiment_result.get('sentiment_score', 0) or 0):+.2f} ({sentiment_result.get('sentiment_label', '中性')})
Article counts: positive {int(sentiment_result.get('positive_count', 0) or 0)} / neutral {int(sentiment_result.get('neutral_count', 0) or 0)} / negative {int(sentiment_result.get('negative_count', 0) or 0)}

Articles, ordered by impact:
{news_text}

Write the news overview."""

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
        _log.warning("%s LLM summary failed: %s", sym, e)
        return "[離線摘要] " + generate_news_summary(sentiment_result, ticker)


def analyze_news_sentiment(news_items: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Score news headlines with FinBERT and return the aggregate plus each article.
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
        title_for_model = title[:256]  # Keep the BERT input within its length limit
        link = str(item.get("link", "") or "")
        publisher = str(item.get("publisher", "") or "")
        published_at = str(item.get("published_at", "") or "")

        dt = _parse_published_at(published_at)
        days_ago = (now_utc - dt).total_seconds() / 86400.0 if dt else 99.0
        w = _time_weight(days_ago)

        # If one headline fails, treat it as neutral
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
    """Default result when there is no news or the analysis fails."""
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
    Fetch news and run FinBERT, using the same-day cache.
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
    Score many tickers. A failure is filled with the default result.
    """
    out: dict[str, dict[str, Any]] = {}
    for t in tqdm([str(x).upper() for x in tickers], desc="Sentiment", unit="name"):
        try:
            out[t] = analyze_ticker_sentiment(t, max_items=max_per_ticker)
        except Exception as e:  # noqa: BLE001
            _log.error("%s sentiment analysis failed: %s", t, e)
            fail = _empty_sentiment_result(t, label="Analysis failed")
            fail["summary"] = generate_news_summary_with_llm(fail, t)
            out[t] = fail
    return out


if __name__ == "__main__":
    print("Downloading the FinBERT model can take about 5 minutes")
    sample = ["NVDA", "TSM"]
    results = analyze_sentiment_batch(sample, max_per_ticker=10)
    for t in sample:
        r = results.get(t, _empty_sentiment_result(t))
        print(f"\n=== {t} ===")
        print(f"Sentiment score: {r['sentiment_score']:.4f} | label: {r['sentiment_label']} | score: {r['score']}")
        print(
            f"Articles: positive {r['positive_count']} / neutral {r['neutral_count']} / negative {r['negative_count']} "
            f"({r['news_count']} total)"
        )
        print("Top 3 articles by impact:")
        for n in r.get("top_headlines", [])[:3]:
            print(
                f"- {n.get('title','')}\n"
                f"  sentiment: {n.get('sentiment_cn')} ({n.get('sentiment')}) | "
                f"confidence: {n.get('confidence')} | link: {n.get('link')}"
            )
        print("\nLLM news summary:")
        print(r.get("summary", "No summary"))
        print("-" * 60)
