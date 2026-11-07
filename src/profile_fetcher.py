# -*- coding: utf-8 -*-
"""
Company profiles: yfinance, then Wikipedia, then Claude, cached for 90 days.

Run from the project root:
    python src/profile_fetcher.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import requests
import yfinance as yf
from tqdm import tqdm

# Load config from the project root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import ANTHROPIC_API_KEY, CACHE_DIR  # noqa: E402

# data/cache/profiles
PROFILES_DIR = Path(CACHE_DIR) / "profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TTL_DAYS = 90
SHORT_TTL_DAYS = 7

_log = logging.getLogger("profile_fetcher")
if not _log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment, misc]

# Yahoo exchange code (partial) to a Google Finance market code
_EXCHANGE_TO_GOOGLE: dict[str, str] = {
    "NMS": "NASDAQ",
    "NCM": "NASDAQ",
    "NGM": "NASDAQ",
    "NAS": "NASDAQ",
    "NASDAQ": "NASDAQ",
    "NYQ": "NYSE",
    "NYS": "NYSE",
    "NYSE": "NYSE",
    "PCX": "NYSE",
    "BTS": "NYSE",
    "BATS": "NASDAQ",
    "HKG": "HKG",  # keep the original code when there is no better mapping
    "KSC": "NASDAQ",
    "FRA": "FRA",
    "JNB": "NYSE",
    "JKT": "NASDAQ",
}


def _fetch_from_yfinance(ticker: str) -> dict[str, Any]:
    """
    Read yf.Ticker(ticker).info for:
    longName, sector, industry, country, website, fullTimeEmployees, marketCap, longBusinessSummary, exchange, cik
    """
    sym = ticker.strip().upper()
    out: dict[str, Any] = {}
    try:
        t = yf.Ticker(sym)
        info = t.info or {}
        if not isinstance(info, dict) or not info:
            return out
        cik = info.get("cik")
        if cik is None and info.get("cikStr"):
            cik = info.get("cikStr")
        out = {
            "ticker": sym,
            "longName": info.get("longName") or info.get("shortName") or sym,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country") or info.get("region"),
            "website": (info.get("website") or info.get("irWebsite") or "") or "",
            "fullTimeEmployees": info.get("fullTimeEmployees") or info.get("fullTimeEmployee"),
            "marketCap": info.get("marketCap"),
            "longBusinessSummary": info.get("longBusinessSummary") or info.get("description") or "",
            "exchange": info.get("exchange") or "NMS",
            "cik": cik,
        }
    except Exception as e:  # noqa: BLE001
        _log.warning("yfinance read failed for %s: %s", sym, e)
    return out


def _wikipedia_get_summary(
    page_title: str,
    session: requests.Session,
) -> dict[str, Any] | None:
    """
    Call the English Wikipedia page-summary API.
    https://en.wikipedia.org/api/rest_v1/page/summary/{title}
    """
    t = str(page_title or "").strip().replace(" ", "_")
    if not t:
        return None
    enc = quote(t, safe="")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{enc}"
    try:
        r = session.get(url, timeout=12)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        j = r.json()
        if not isinstance(j, dict):
            return None
        return {
            "title": j.get("title", ""),
            "extract": (j.get("extract") or "").strip(),
            "content_url": (j.get("content_urls") or {}).get("desktop", {}).get("page", ""),
        }
    except Exception as e:  # noqa: BLE001
        _log.debug("Wikipedia %s: %s", page_title, e)
        return None


def _fetch_from_wikipedia(company_name: str) -> dict[str, Any]:
    """
    Fetch an English Wikipedia extract. On 404, try Inc. / Corporation variants.
    """
    if not (company_name or "").strip():
        return {}
    name = str(company_name).strip()
    session = requests.Session()
    session.headers["User-Agent"] = "StockAnalysis/1.0 (profile_fetcher; contact: local) Python requests"

    titles = [
        name,
        f"{name} Inc.",
        f"{name} Inc",
        f"{name} Corporation",
        f"{name} (company)",
    ]
    for title in titles:
        data = _wikipedia_get_summary(title, session)
        if data and (data.get("extract") or data.get("content_url")):
            _log.info("Wikipedia hit: %s", title)
            return {
                "title": data.get("title", title),
                "extract": data.get("extract", ""),
                "content_url": data.get("content_url", ""),
            }
    _log.info("Wikipedia has no extract: %s", name)
    return {}


def _guess_ir_url(website: str) -> str:
    """
    Guess an IR URL from the company site, for example
    https://www.nvidia.com -> https://investor.nvidia.com.
    Returns an empty string when the host cannot be parsed.
    """
    w = (website or "").strip()
    if not w.startswith("http"):
        return ""
    try:
        p = urlparse(w)
        host = p.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        if host:
            return f"{p.scheme}://investor.{host}"
    except Exception:  # noqa: BLE001
        return ""
    return ""


def _build_articles_list(
    ticker: str,
    yf_data: dict[str, Any],
    wiki_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Further-reading links: Wikipedia, website, guessed IR, Yahoo, Google, SEC."""
    name = (yf_data.get("longName") or ticker) or "Company"
    sym = str(ticker).upper()
    yf_website = (yf_data.get("website") or "") or ""
    ir = _guess_ir_url(yf_website) if yf_website else ""

    ex = str((yf_data.get("exchange") or "NMS") or "NMS").upper()
    g_ex = _EXCHANGE_TO_GOOGLE.get(ex, "NASDAQ")
    g_url = f"https://www.google.com/finance/quote/{sym}:{g_ex}"

    cik = yf_data.get("cik")
    cik_s = re.sub(r"\D", "", str(cik)) if cik is not None and str(cik).strip() else ""
    if cik_s:
        cik_s = cik_s.zfill(10)
        sec_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik_s}&type=&dateb=&owner=exclude&count=40&search=1"
    else:
        sec_url = f"https://www.sec.gov/edgar/search/#/q={quote(sym)}"

    articles: list[dict[str, Any]] = []
    wu = (wiki_data or {}).get("content_url", "")
    if wu:
        articles.append(
            {"title": f"{name} Wikipedia", "url": wu, "source": "Wikipedia", "type": "reference"},
        )
    if yf_website:
        articles.append(
            {"title": f"{name} official website", "url": yf_website, "source": "Official", "type": "official"},
        )
    if ir and yf_website and ir != yf_website:
        articles.append(
            {
                "title": f"{name} Investor Relations",
                "url": ir,
                "source": "Official",
                "type": "ir",
            },
        )
    articles.append(
        {"title": "Yahoo Finance quote page", "url": f"https://finance.yahoo.com/quote/{sym}", "source": "Yahoo Finance", "type": "reference"},
    )
    articles.append(
        {"title": "Google Finance", "url": g_url, "source": "Google Finance", "type": "reference"},
    )
    articles.append(
        {"title": "SEC filings search", "url": sec_url, "source": "SEC", "type": "filing"},
    )
    return articles


def _parse_claude_json_text(text: str) -> dict[str, Any] | None:
    s = (text or "").strip()
    if not s:
        return None
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    try:
        o = json.loads(s)
        return o if isinstance(o, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            try:
                o = json.loads(m.group(0))
                return o if isinstance(o, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def _fallback_claude_data(yf_data: dict[str, Any], wiki_data: dict[str, Any], ticker: str) -> dict[str, Any]:
    """
    Used when Claude fails or there is no API key.
    Stitches the English yfinance summary and the Wikipedia extract into a template note.
    """
    name = (yf_data.get("longName") or ticker) or "Company"
    yf_s = (yf_data.get("longBusinessSummary") or "")[:1500]
    wik = (wiki_data or {}).get("extract") or ""
    wik = wik[:800] if wik else ""

    long_txt = (
        f"{name} is a publicly listed company. The note below is assembled from the "
        f"English yfinance business description and public extracts, and has not been rewritten by an LLM.\n\n"
    )
    if yf_s:
        long_txt += f"(yfinance excerpt)\n{yf_s}\n\n"
    if wik and not wik.isspace():
        long_txt += f"(Wikipedia extract)\n{wik}\n"
    if len(long_txt) < 80 and not yf_s and not wik:
        long_txt = f"{ticker}: little public narrative is available. Try again later, or read the annual report and the company site."

    return {
        "description_short": f"{name}, {yf_data.get('sector', '')}/{yf_data.get('industry', '')}.".replace("//", "/")[:200],
        "description_long": long_txt[:5000],
        "business_segments": [str(yf_data.get("industry") or "General business")],
        "key_products": ["See the English description and the products section of the annual report"],
        "competitors": ["Industry peers; see the competition discussion in the annual report"],
    }


def _generate_with_claude(
    ticker: str,
    yf_data: dict[str, Any],
    wiki_data: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    """
    Ask Claude Haiku 4.5 for an English JSON profile.
    Success is (dict, ''). Failure is (None, a reason code or message).
    """
    if not (ANTHROPIC_API_KEY or "").strip() or anthropic is None:
        return None, "no_api"

    yf_name = yf_data.get("longName") or ticker
    yf_sector = yf_data.get("sector") or "N/A"
    yf_industry = yf_data.get("industry") or "N/A"
    yf_country = yf_data.get("country") or "N/A"
    yf_employees = yf_data.get("fullTimeEmployees", "N/A")
    yf_summary = (yf_data.get("longBusinessSummary") or "(none)")[:8000]
    wiki_extract = (wiki_data.get("extract") if isinstance(wiki_data, dict) else None) or "(none)"
    wiki_extract = (wiki_extract or "(none)")[:8000]

    system_prompt = """You are an investment researcher. Turn English source material into a clear company profile.
Return one JSON object only. No markdown fences and no text before or after the JSON."""

    user_message = f"""Using the sources below, write an English company profile for {ticker}.

yfinance:
- Name: {yf_name}
- Sector: {yf_sector} / {yf_industry}
- Headquarters: {yf_country}
- Employees: {yf_employees}
- English description: {yf_summary}

Wikipedia extract:
{wiki_extract}

Return JSON only, with no other text:
{{
  "description_short": "one sentence, about 50-80 words",
  "description_long": "about 300-400 words covering history, main business, and market position",
  "business_segments": ["segment 1 (share if known)", "segment 2", "segment 3"],
  "key_products": ["main product or service 1", "product 2", "product 3"],
  "competitors": ["competitor 1", "competitor 2", "competitor 3"]
}}

Write every string in English. Keep proper nouns, including product names, as they appear in the sources."""

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY.strip())
        res = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        block = res.content[0] if res.content else None
        text = getattr(block, "text", str(block)) if block is not None else ""
        parsed = _parse_claude_json_text(text)
        if not parsed:
            return None, "parse"
        for k, default in [
            ("description_short", ""),
            ("description_long", ""),
            ("business_segments", []),
            ("key_products", []),
            ("competitors", []),
        ]:
            if k not in parsed or parsed[k] is None:
                parsed[k] = default
        return parsed, ""
    except Exception as e:  # noqa: BLE001
        _log.warning("Claude profile failed for %s: %s", ticker, e)
        return None, str(e)


def _is_cache_fresh(
    data: dict[str, Any],
    default_ttl: int = DEFAULT_TTL_DAYS,
) -> bool:
    try:
        raw = data.get("fetched_at", "")
        ttl = int(data.get("ttl_days", default_ttl))
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return (now - dt).total_seconds() < max(1, ttl) * 86400
    except Exception:  # noqa: BLE001
        return False


def _profile_path(ticker: str) -> Path:
    return PROFILES_DIR / f"{ticker.strip().upper()}.json"


def get_profile(ticker: str, force_refresh: bool = False) -> dict[str, Any]:
    """
    Main path: cache (ttl_days and fetched_at), then yfinance, Wikipedia, Claude, then write back.
    A badly incomplete yfinance read is stored as a 7-day cache so it can be retried.
    """
    sym = ticker.strip().upper()
    path = _profile_path(sym)

    if not force_refresh and path.is_file():
        try:
            with open(path, encoding="utf-8") as f:
                cached = json.load(f)
            if isinstance(cached, dict) and _is_cache_fresh(cached):
                _log.info("Cache hit: %s", sym)
                return cached
        except Exception as e:  # noqa: BLE001
            _log.debug("Cache read failed: %s", e)

    _log.info("Fetching: %s", sym)

    yf_data = _fetch_from_yfinance(sym)
    company = yf_data.get("longName") or sym
    wiki_data = _fetch_from_wikipedia(str(company))

    yf_strong = bool(
        (yf_data.get("longBusinessSummary") or "").strip()
        or yf_data.get("sector")
        or yf_data.get("industry")
        or yf_data.get("marketCap")
    )

    if not yf_strong:
        # yfinance returned almost nothing: store a short result for 7 days and retry later
        _log.warning("yfinance fields are too thin for %s; writing a 7-day cache", sym)
        mini_yf: dict[str, Any] = {
            "longName": yf_data.get("longName") or sym,
            "ticker": sym,
            "sector": yf_data.get("sector"),
            "industry": yf_data.get("industry"),
            "country": yf_data.get("country"),
            "website": (yf_data.get("website") or "") or "",
            "fullTimeEmployees": yf_data.get("fullTimeEmployees"),
            "marketCap": yf_data.get("marketCap"),
            "longBusinessSummary": yf_data.get("longBusinessSummary") or "",
            "exchange": yf_data.get("exchange") or "NMS",
            "cik": yf_data.get("cik"),
        }
        blurb = _fallback_claude_data(mini_yf, wiki_data, sym)
        out = {
            "ticker": sym,
            "name": mini_yf["longName"],
            "sector": mini_yf["sector"],
            "industry": mini_yf["industry"],
            "country": mini_yf["country"],
            "website": mini_yf["website"] or "",
            "employees": mini_yf["fullTimeEmployees"],
            "market_cap": mini_yf["marketCap"],
            "description_short": blurb["description_short"],
            "description_long": blurb["description_long"],
            "business_segments": blurb["business_segments"],
            "key_products": blurb["key_products"],
            "competitors": blurb["competitors"],
            "articles": _build_articles_list(sym, mini_yf, wiki_data),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "ttl_days": SHORT_TTL_DAYS,
            "sources_used": ["yfinance", "wikipedia", "template"],
        }
        _write_cache(path, out)
        return out

    claude_parsed, err = _generate_with_claude(sym, yf_data, wiki_data)
    if claude_parsed and err == "":
        content = claude_parsed
        src = ["yfinance", "wikipedia", "claude"]
    else:
        _log.info("Using the template profile for %s: err=%s", sym, err or "fallback")
        content = _fallback_claude_data(yf_data, wiki_data, sym)
        if err == "no_api":
            src = ["yfinance", "wikipedia", "template"]
        else:
            src = ["yfinance", "wikipedia", "template", "yfinance_english"]  # the note includes the original English description

    result: dict[str, Any] = {
        "ticker": sym,
        "name": yf_data.get("longName"),
        "sector": yf_data.get("sector"),
        "industry": yf_data.get("industry"),
        "country": yf_data.get("country"),
        "website": (yf_data.get("website") or "") or "",
        "employees": yf_data.get("fullTimeEmployees"),
        "market_cap": yf_data.get("marketCap"),
        "description_short": content.get("description_short", "") or "",
        "description_long": content.get("description_long", "") or "",
        "business_segments": list(content.get("business_segments", []) or []),
        "key_products": list(content.get("key_products", []) or []),
        "competitors": list(content.get("competitors", []) or []),
        "articles": _build_articles_list(sym, yf_data, wiki_data),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "ttl_days": DEFAULT_TTL_DAYS,
        "sources_used": list(dict.fromkeys(src)),
    }
    _write_cache(path, result)
    return result


def _write_cache(path: Path, data: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        _log.info("Wrote cache: %s", path)
    except Exception as e:  # noqa: BLE001
        _log.error("Cache write failed: %s", e)


def get_profiles_batch(
    tickers: list[str],
    show_progress: bool = True,
) -> dict[str, dict[str, Any]]:
    """Fetch many profiles. A failure becomes an empty dict and does not stop the rest. Sleep 0.5s between names."""
    out: dict[str, dict[str, Any]] = {}
    n = len(tickers)
    iterator: Any = range(n)
    if show_progress and n:
        iterator = tqdm(range(n), desc="Profiles", unit="name")

    for j in iterator:
        t = str(tickers[j]).upper()
        try:
            out[t] = get_profile(t, force_refresh=False)
        except Exception as e:  # noqa: BLE001
            _log.error("%s profile failed: %s", t, e)
            out[t] = {}
        if j < n - 1:
            time.sleep(0.5)
    return out


if __name__ == "__main__":
    _sample = ["NVDA", "TSM", "AAPL"]

    print("Round 1: first fetch (an unexpired cache returns immediately)...")
    t_round1 = time.time()
    for x in _sample:
        t0 = time.time()
        r1 = get_profile(x, force_refresh=False)
        print(f"  {x} took {time.time() - t0:.1f}s, sources={r1.get('sources_used')}")
        print(f"  excerpt: {str(r1.get('description_short', ''))[:150]}")
        for a in (r1.get("articles") or [])[:3]:
            print(f"  · {a.get('title')}: {a.get('url')[:70]}…")

    print(f"\nRound 1 total: {time.time() - t_round1:.1f}s")

    print("\nRound 2: should all hit the 90-day cache...")
    t_round2 = time.time()
    for x in _sample:
        t0 = time.time()
        r2 = get_profile(x, force_refresh=False)
        print(f"  {x} took {time.time() - t0:.3f}s, fetched_at={r2.get('fetched_at')}")
        print(f"  summary: {str(r2.get('description_short', 'none'))[:120]}")

    print(f"\nRound 2 total for 3 names: {time.time() - t_round2:.2f}s (should be much faster than round 1)")
