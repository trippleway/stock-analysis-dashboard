# -*- coding: utf-8 -*-
"""
Point-in-time S&P 500 universe for backtests. It covers NYSE and NASDAQ, the universe of mainstream large US stocks.

It sits beside backtest_universe.py (NASDAQ-100) and does not replace that module.
Many of today's hand-picked 60 names are not listed on NASDAQ, so NASDAQ-100 was never the right benchmark.

Data source
-----------
GitHub fja05680/sp500 (MIT). At runtime the repo file list is read first. A
"Historical Components" CSV whose filename carries a date is preferred, and the
newest such date wins. The repo currently has no filename of that form, so the
latest history file named in the README is used:

    S&P 500 Historical Components & Changes (Updated).csv

The raw file is one ``date,tickers`` row at a time. tickers is every constituent
that day, comma-separated, and the whole field is quoted. There is not a row for
every trading day. A full list is recorded when the index is reconstituted.

Known limits (the author's own README; this is not an error-free official source)
---------------------------------------------------------------------------------
- The base file is the appendix of Andreas Clenow's Trading Evolved (1996-2019).
  The author then filled later changes by hand from Wikipedia "Selected Changes".
  That Wikipedia section is not a complete change list, so dates and events can be missing.
- The first years from 1996 often have fewer than 500 names in a row (about 487 at the start).
  The author thinks symbols may be missing before about 2001 and cannot be checked independently.
- Gaps between additions and deletions mean the count is not exactly 500. It is sometimes a bit high (for example 505) and sometimes a bit low.
- Symbols are the ticker that actually traded that day. A rename (FB and META) is not treated as the same company automatically.
- Changes after the last row of the file are not included. A query date after the last row can only reuse that row. It cannot pretend the file covers today.

Run from the project root:
    python src/backtest_universe_sp500.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
import yfinance as yf

_ROOT = Path(__file__).resolve().parent.parent
_SRC = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import DATA_DIR  # noqa: E402

_log = logging.getLogger("backtest_universe_sp500")
if not _log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

_REPO_API = "https://api.github.com/repos/fja05680/sp500/contents/"
_RAW_BASE = "https://raw.githubusercontent.com/fja05680/sp500/master/"
# The maintained history file named in the README. Used when the repo has no filename that carries an update date.
_UPDATED_FILENAME = "S&P 500 Historical Components & Changes (Updated).csv"
_CACHE_DIR = Path(DATA_DIR) / "backtest_cache"
_CACHE_CSV = _CACHE_DIR / "sp500_historical.csv"
_CACHE_META = _CACHE_DIR / "sp500_historical.meta.json"
_CACHE_MAX_AGE = timedelta(days=7)
_HTTP_TIMEOUT_SEC = 60
_HTTP_RETRY_DELAYS = (1.0, 2.0, 4.0)
_YF_CALL_SLEEP_SEC = 0.7
_YF_RETRY_DELAYS = (1.0, 2.0, 4.0)

# Quarterly samples align to the start of the quarter, the same dates as the NASDAQ-100 module.
_FREQ_TO_PANDAS: dict[str, str] = {
    "quarterly": "QS",
    "monthly": "MS",
    "yearly": "YS",
    "daily": "D",
}

# Spellings of yfinance info["country"] that count as a US company.
_US_COUNTRIES = frozenset({"UNITED STATES", "USA", "US", "U.S.", "U.S.A."})
# (MM-DD-YYYY) inside a filename, for example Changes(08-01-2026).csv.
_FILENAME_DATE = re.compile(r"\((\d{2})-(\d{2})-(\d{4})\)")

_HISTORY: pd.DataFrame | None = None
_STALE_WARNED = False


class UniverseCoverageError(RuntimeError):
    """The query date is before the start of the S&P 500 history file, or that snapshot is empty."""


def _coerce_date(value: date | datetime | str, name: str) -> date:
    """
    Turn a date, datetime, or ISO string into a date.

    Args:
        value: Date input.
        name: Parameter name, used in the error message.

    Returns:
        datetime.date.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        try:
            return date.fromisoformat(text[:10])
        except ValueError as exc:
            raise ValueError(f"{name} is not a valid date: {value!r}. Use YYYY-MM-DD") from exc
    raise TypeError(f"{name} must be a date, got {type(value).__name__}")


def _validate_range(start: date, end: date) -> None:
    """The start date cannot be after the end date."""
    if start > end:
        raise ValueError(
            f"start_date ({start.isoformat()}) is after end_date ({end.isoformat()})"
        )


def _http_get(url: str, *, as_json: bool = False) -> bytes | list | dict:
    """
    GET one URL. Retry 3 times on failure.

    Args:
        url: Full URL.
        as_json: When True, return the parsed JSON.

    Returns:
        The response body, or a JSON object.
    """
    last_err: Exception | None = None
    headers = {"User-Agent": "stock-analysis-backtest", "Accept": "application/vnd.github+json"}
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=_HTTP_TIMEOUT_SEC)
            resp.raise_for_status()
            if as_json:
                return resp.json()
            return resp.content
        except Exception as exc:  # noqa: BLE001 — a network failure is retried
            last_err = exc
            _log.warning("Download failed, attempt %s, %s: %s", attempt + 1, url, exc)
            if attempt < 2:
                time.sleep(_HTTP_RETRY_DELAYS[attempt])
    raise RuntimeError(f"Download failed after 3 attempts: {url}. {last_err}") from last_err


def _filename_stamp(name: str) -> date | None:
    """
    Pull (MM-DD-YYYY) out of a filename. Returns None when there is no date.

    Args:
        name: Filename inside the repo.

    Returns:
        The date in the filename, or None.
    """
    match = _FILENAME_DATE.search(name)
    if match is None:
        return None
    month, day, year = (int(part) for part in match.groups())
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _raw_url(filename: str) -> str:
    """Turn a repo filename into a raw.githubusercontent.com URL."""
    return _RAW_BASE + quote(filename)


def _resolve_remote_file() -> tuple[str, str]:
    """
    List files in fja05680/sp500 and decide which historical-constituents CSV to download.

    Preference: the Historical Components CSV whose filename carries a date, and among those the newest date.
    When no such file exists, use the ``(Updated)`` file named in the README.
    A failure to list files is raised. An expired URL is not fetched silently.

    Returns:
        (filename, raw URL).
    """
    payload = _http_get(_REPO_API, as_json=True)
    if not isinstance(payload, list):
        raise RuntimeError(f"GitHub file list has an unexpected shape: {type(payload).__name__}")

    names = [str(item.get("name") or "") for item in payload if isinstance(item, dict)]
    historical = [
        name
        for name in names
        if name.lower().endswith(".csv") and "historical components" in name.lower()
    ]
    dated = [(stamp, name) for name in historical if (stamp := _filename_stamp(name)) is not None]
    if dated:
        dated.sort()
        filename = dated[-1][1]
        _log.info("Using the historical constituents file with the newest filename date: %s", filename)
        return filename, _raw_url(filename)

    if _UPDATED_FILENAME in names:
        _log.info(
            "The repo has no Historical Components CSV whose filename carries a date. "
            "Using the latest history file named in the README: %s",
            _UPDATED_FILENAME,
        )
        return _UPDATED_FILENAME, _raw_url(_UPDATED_FILENAME)

    raise RuntimeError(
        "No S&P 500 historical constituents CSV was found in fja05680/sp500. "
        f"Files seen: {', '.join(names) or '(empty)'}"
    )


def _parse_ticker_cell(value: object) -> set[str]:
    """Split one comma-separated constituents cell into a set of uppercase symbols."""
    if not isinstance(value, str) or not value.strip():
        return set()
    return {part.strip().upper() for part in value.split(",") if part.strip()}


def _parse_history(path: Path) -> pd.DataFrame:
    """
    Read the cached CSV and parse it into dates and ticker sets.

    Args:
        path: Local CSV path.

    Returns:
        A DataFrame sorted by date, with columns date (Timestamp) and tickers (set[str]).
    """
    raw = pd.read_csv(path)
    if "date" not in raw.columns or "tickers" not in raw.columns:
        raise ValueError(f"CSV is missing the date/tickers columns. Columns found: {list(raw.columns)}")

    parsed = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["date"], errors="coerce").dt.normalize(),
            "tickers": raw["tickers"].map(_parse_ticker_cell),
        }
    )
    parsed = parsed.dropna(subset=["date"])
    parsed = parsed.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if parsed.empty:
        raise ValueError(f"{path} has no usable date rows")

    sizes = parsed["tickers"].map(len)
    median = float(sizes.median())
    if median < 450:
        raise ValueError(
            f"The median constituent count is only {median:.0f}. The CSV tickers field may not have been parsed as one column"
        )
    empty_rows = int((sizes == 0).sum())
    if empty_rows:
        _log.warning("%s rows have an empty constituent list and cannot be used as a valid snapshot", empty_rows)
    return parsed


def _read_fetched_at() -> datetime | None:
    """Read the fetch time stored beside the cache. Falls back to the file mtime when there is no meta."""
    if _CACHE_META.is_file():
        try:
            meta = json.loads(_CACHE_META.read_text(encoding="utf-8"))
            text = str(meta.get("fetched_at") or "")
            if text:
                return datetime.fromisoformat(text)
        except Exception as exc:  # noqa: BLE001
            _log.warning("Failed to read the cache timestamp, falling back to the file mtime: %s", exc)
    if _CACHE_CSV.is_file():
        return datetime.fromtimestamp(_CACHE_CSV.stat().st_mtime, tz=timezone.utc)
    return None


def _cache_is_fresh() -> bool:
    """The local CSV exists and was fetched within the last 7 days."""
    if not _CACHE_CSV.is_file():
        return False
    fetched = _read_fetched_at()
    if fetched is None:
        return False
    if fetched.tzinfo is None:
        fetched = fetched.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - fetched
    return age <= _CACHE_MAX_AGE


def _write_meta(filename: str, url: str) -> None:
    """Write the source and time of this fetch so the 7-day cache can be checked."""
    meta = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source_filename": filename,
        "source_url": url,
    }
    _CACHE_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def download_sp500_history() -> pd.DataFrame:
    """
    Download the S&P 500 historical constituents and cache them at data/backtest_cache/sp500_historical.csv.

    A local file fetched within the last 7 days is read as-is and GitHub is not called.
    A damaged cache is downloaded again. A file that failed to parse is not reused.

    Returns:
        A DataFrame with columns date and tickers (set[str]). One row is the full list after one index change.
    """
    global _HISTORY
    if _HISTORY is not None:
        return _HISTORY

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if _cache_is_fresh():
        try:
            parsed = _parse_history(_CACHE_CSV)
        except Exception as exc:  # noqa: BLE001
            _log.warning("The local cache could not be parsed, so it will be downloaded again: %s", exc)
        else:
            _HISTORY = parsed
            first = parsed["date"].iloc[0].date()
            last = parsed["date"].iloc[-1].date()
            _log.info(
                "Using the cache from the last 7 days, %s (%s rows, %s to %s)",
                _CACHE_CSV.name,
                len(parsed),
                first.isoformat(),
                last.isoformat(),
            )
            return parsed

    filename, url = _resolve_remote_file()
    content = _http_get(url)
    if not isinstance(content, (bytes, bytearray)) or not content:
        raise RuntimeError(f"The downloaded content is empty: {url}")
    _CACHE_CSV.write_bytes(content)
    _write_meta(filename, url)
    parsed = _parse_history(_CACHE_CSV)
    _HISTORY = parsed
    first = parsed["date"].iloc[0].date()
    last = parsed["date"].iloc[-1].date()
    _log.info(
        "Downloaded %s -> %s (%s rows, %s to %s)",
        filename,
        _CACHE_CSV,
        len(parsed),
        first.isoformat(),
        last.isoformat(),
    )
    return parsed


def _history_bounds(history: pd.DataFrame) -> tuple[date, date]:
    """Return the dates of the first and last rows in the history file."""
    return history["date"].iloc[0].date(), history["date"].iloc[-1].date()


def _warn_if_past_coverage(history: pd.DataFrame, end: date) -> None:
    """Warn when the query end date is after the last row. Later changes are not in the file. Warn only once per run."""
    global _STALE_WARNED
    _, last = _history_bounds(history)
    if end > last and not _STALE_WARNED:
        _STALE_WARNED = True
        _log.warning(
            "The last row of the S&P 500 source is %s and the query end date is %s. "
            "Constituent changes in that gap are not in the file, so the last valid list is reused.",
            last.isoformat(),
            end.isoformat(),
        )


def _snapshot_as_of(history: pd.DataFrame, as_of: date) -> set[str]:
    """
    Take the constituents from the latest row on or before as_of.

    The source is event-driven, not one row per day. Between two changes, the previous full list stays in force.

    Args:
        history: The result of download_sp500_history.
        as_of: Query date.

    Returns:
        The set of constituents that were valid on that date.
    """
    position = int(history["date"].searchsorted(pd.Timestamp(as_of), side="right")) - 1
    if position < 0:
        first, _last = _history_bounds(history)
        raise UniverseCoverageError(
            f"Cannot load S&P 500 constituents for {as_of.isoformat()}: "
            f"it is before the data start {first.isoformat()}."
        )
    symbols = history["tickers"].iloc[position]
    row_date = history["date"].iloc[position].date()
    if not symbols:
        raise UniverseCoverageError(
            f"Cannot load S&P 500 constituents for {as_of.isoformat()}: "
            f"the latest valid row {row_date.isoformat()} is an empty list."
        )
    return set(symbols)


def _sample_dates(start: date, end: date, freq: str) -> list[date]:
    """
    Build sample dates inside the range. The backtest start is included even when it is not a frequency boundary.

    Args:
        start: Range start (inclusive).
        end: Range end (inclusive).
        freq: quarterly / monthly / yearly / daily.

    Returns:
        A list of dates from earliest to latest.
    """
    key = freq.strip().lower()
    pandas_freq = _FREQ_TO_PANDAS.get(key)
    if pandas_freq is None:
        allowed = ", ".join(_FREQ_TO_PANDAS)
        raise ValueError(f"Unsupported sample frequency {freq!r}. Use {allowed}")

    stamps = pd.date_range(start=start, end=end, freq=pandas_freq)
    dates = [ts.date() for ts in stamps]
    if not dates or dates[0] > start:
        dates.insert(0, start)

    out: list[date] = []
    seen: set[date] = set()
    for sample in dates:
        if start <= sample <= end and sample not in seen:
            seen.add(sample)
            out.append(sample)
    if not out:
        raise ValueError(
            f"No sample dates between {start.isoformat()} and {end.isoformat()} (freq={freq})"
        )
    return out


def get_point_in_time_sp500(
    start_date: date,
    end_date: date,
    freq: str = "quarterly",
) -> dict[date, set[str]]:
    """
    Return the S&P 500 constituents that were in force at each sample date.

    Each sample date uses the latest full list in the CSV on or before that date.
    quarterly aligns to the start of the quarter, the same dates as the NASDAQ-100 module.

    Args:
        start_date: Backtest start (inclusive).
        end_date: Backtest end (inclusive).
        freq: Sample frequency. Defaults to quarterly.

    Returns:
        {sample date: set of constituents valid that day}, ordered from earliest to latest.
    """
    start = _coerce_date(start_date, "start_date")
    end = _coerce_date(end_date, "end_date")
    _validate_range(start, end)
    history = download_sp500_history()
    _warn_if_past_coverage(history, end)

    universe: dict[date, set[str]] = {}
    for sample in _sample_dates(start, end, freq):
        universe[sample] = _snapshot_as_of(history, sample)

    _log.info(
        "S&P 500 point-in-time universe %s to %s (%s): %s dates",
        start.isoformat(),
        end.isoformat(),
        freq,
        len(universe),
    )
    return universe


def get_union_universe_sp500(start_date: date, end_date: date) -> set[str]:
    """
    Return the union of every name that appeared in the S&P 500 during the range.

    It starts from the row that was valid on the start date and includes every event snapshot through the end date.
    Merging only quarterly snapshots would miss a name added mid-quarter and removed before quarter-end.

    Args:
        start_date: Range start (inclusive).
        end_date: Range end (inclusive).

    Returns:
        Symbols that appeared in at least one valid snapshot during the range.
    """
    start = _coerce_date(start_date, "start_date")
    end = _coerce_date(end_date, "end_date")
    _validate_range(start, end)
    history = download_sp500_history()
    _warn_if_past_coverage(history, end)

    start_pos = int(history["date"].searchsorted(pd.Timestamp(start), side="right")) - 1
    if start_pos < 0:
        first, _last = _history_bounds(history)
        raise UniverseCoverageError(
            f"Cannot build an S&P 500 union from {start.isoformat()}: it is before the data start {first.isoformat()}."
        )
    end_pos = int(history["date"].searchsorted(pd.Timestamp(end), side="right"))
    window = history.iloc[start_pos:end_pos]
    if window.empty:
        raise UniverseCoverageError(
            f"No usable S&P 500 snapshot between {start.isoformat()} and {end.isoformat()}."
        )

    union: set[str] = set()
    for symbols in window["tickers"]:
        union.update(symbols)
    if not union:
        raise UniverseCoverageError(
            f"The S&P 500 union from {start.isoformat()} to {end.isoformat()} is empty."
        )

    _log.info(
        "S&P 500 historical union %s to %s: %s names (%s event snapshots)",
        start.isoformat(),
        end.isoformat(),
        len(union),
        len(window),
    )
    return union


def _quote_missing(ticker: str) -> bool:
    """
    Whether recent daily bars are completely missing. Delisted symbols often have no country in info.

    Args:
        ticker: Ticker symbol.

    Returns:
        True when there is no usable quote.
    """
    try:
        hist = yf.Ticker(ticker).history(period="5d")
    except Exception as exc:  # noqa: BLE001
        _log.warning("Failed to read a recent quote for %s: %s", ticker, exc)
        return False
    return hist is None or hist.empty


def _fetch_listing_info(ticker: str) -> tuple[dict[str, str] | None, str]:
    """
    Read country, exchange, and company name from yfinance info. Retry a transient failure 3 times.

    exchange is only the listing venue. TSM and DIS are both NYQ, so exchange alone would
    misread a foreign ADR as a US company. Nationality uses country. exchange is recorded alongside it.

    Args:
        ticker: Ticker symbol.

    Returns:
        (field dict, failure reason). The failure reason is an empty string on success.
    """
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            time.sleep(_YF_CALL_SLEEP_SEC)
            info = yf.Ticker(ticker).info or {}
            if not isinstance(info, dict):
                last_err = ValueError("info is not a dict")
            else:
                country = str(info.get("country") or "").strip()
                exchange = str(info.get("exchange") or "").strip()
                name = str(info.get("longName") or info.get("shortName") or "").strip()
                if country or exchange:
                    return {"country": country, "exchange": exchange, "name": name}, ""
                if info and _quote_missing(ticker):
                    reason = "Yahoo has no quote (it may be delisted), and info has no country / exchange"
                    _log.warning("Cannot tell whether %s is a US company: %s", ticker, reason)
                    return None, reason
                last_err = ValueError("info has no country / exchange")
        except Exception as exc:  # noqa: BLE001 — a single-name failure is retried
            last_err = exc
        if attempt < 2:
            time.sleep(_YF_RETRY_DELAYS[attempt])
    reason = f"Still failing after 3 attempts: {last_err}"
    _log.warning("Cannot tell whether %s is a US company (%s)", ticker, reason)
    return None, reason


def _is_foreign_issuer(country: str, name: str) -> bool | None:
    """
    Decide whether this is a foreign issuer or ADR that the S&P 500 usually does not include.

    Args:
        country: yfinance info['country'].
        name: Company name, used when the name itself says ADR.

    Returns:
        True for a foreign issuer or ADR, False for a US company, None when the data is not enough to guess.
    """
    upper_name = name.upper()
    if "ADR" in upper_name or "DEPOSITARY" in upper_name:
        return True
    if not country:
        return None
    return country.strip().upper() not in _US_COUNTRIES


def _split_only_in_current(
    tickers: list[str],
) -> tuple[list[str], list[str], list[str], dict[str, str]]:
    """
    Split names that are only on the current list into ineligible foreign issuers and US companies that never entered the index.

    Args:
        tickers: Symbols in only_in_current.

    Returns:
        (not_sp500_listed, sp500_but_excluded, eligibility_unknown, unknown_reasons).
    """
    foreign: list[str] = []
    excluded: list[str] = []
    unknown: list[str] = []
    unknown_reasons: dict[str, str] = {}
    for ticker in tickers:
        info, reason = _fetch_listing_info(ticker)
        if info is None:
            unknown.append(ticker)
            unknown_reasons[ticker] = reason or "no country / exchange"
            continue
        verdict = _is_foreign_issuer(info["country"], info["name"])
        label = (
            f"country={info['country'] or '(empty)'} exchange={info['exchange'] or '(empty)'}"
        )
        if verdict is None:
            unknown.append(ticker)
            unknown_reasons[ticker] = f"A listing venue is present but country is missing, so US status cannot be decided ({label})"
            _log.warning("Cannot tell whether %s is a US company: %s", ticker, unknown_reasons[ticker])
            continue
        if verdict:
            foreign.append(ticker)
            _log.info("%s %s -> foreign issuer/ADR, usually ineligible for the S&P 500", ticker, label)
        else:
            excluded.append(ticker)
            _log.info("%s %s -> US company, but it did not enter the S&P 500 during this window", ticker, label)
    return sorted(foreign), sorted(excluded), sorted(unknown), unknown_reasons


def compare_universes_sp500(
    current_list: list[str],
    point_in_time_union: set[str],
) -> dict[str, list[str] | float | dict[str, str]]:
    """
    Compare today's hand-picked list with the historical S&P 500 constituent union.

    overlap_pct is the share of the current list that appeared in this stretch of S&P 500 history (0-100).
    The denominator is the number of names on the current list. Symbols are compared literally.

    only_in_current is split further:
    - not_sp500_listed: yfinance info['country'] is not the United States, or the name says ADR.
      These companies usually fail the S&P 500 "US company" rule. This is not survivorship bias.
      exchange only records the listing venue and is not used for nationality: TSM and DIS are both NYQ.
    - sp500_but_excluded: a US company that did not enter the S&P 500 during this window.
      These are the names screened today that were not in the index in the past.
    Names with no country go in eligibility_unknown and are not merged into the two groups above.

    Args:
        current_list: The current hand-picked list.
        point_in_time_union: The result of get_union_universe_sp500.

    Returns:
        only_in_current, only_in_historical, overlap, overlap_pct,
        not_sp500_listed, sp500_but_excluded, eligibility_unknown,
        eligibility_unknown_reasons.
    """
    current = {str(ticker).upper().strip() for ticker in current_list if str(ticker).strip()}
    historical = {
        str(ticker).upper().strip() for ticker in point_in_time_union if str(ticker).strip()
    }
    overlap = current & historical
    only_current = sorted(current - historical)
    only_historical = historical - current
    overlap_pct = round(100.0 * len(overlap) / len(current), 2) if current else 0.0
    foreign, excluded, unknown, unknown_reasons = _split_only_in_current(only_current)

    return {
        "only_in_current": only_current,
        "only_in_historical": sorted(only_historical),
        "overlap": sorted(overlap),
        "overlap_pct": overlap_pct,
        "not_sp500_listed": foreign,
        "sp500_but_excluded": excluded,
        "eligibility_unknown": unknown,
        "eligibility_unknown_reasons": unknown_reasons,
    }


def _print_ticker_block(title: str, tickers: list[str]) -> None:
    """Print one group of symbols: the count first, then one row at a time."""
    print(f"{title} ({len(tickers)} names)", flush=True)
    if not tickers:
        print("  (none)", flush=True)
        return
    line: list[str] = []
    for ticker in tickers:
        line.append(ticker)
        if len(line) == 12:
            print("  " + ", ".join(line), flush=True)
            line = []
    if line:
        print("  " + ", ".join(line), flush=True)


if __name__ == "__main__":
    from backtest_universe import get_union_universe
    from config import TICKER_LIST

    start = date(2022, 1, 1)
    end = date.today()
    history = download_sp500_history()
    first, last = _history_bounds(history)
    print(
        f"S&P 500 point-in-time universe: {start.isoformat()} to {end.isoformat()}, one date per quarter",
        flush=True,
    )
    print(
        f"Source coverage {first.isoformat()} to {last.isoformat()} ({len(history)} event rows). "
        "This file is a Wikipedia revision. The author notes that the change list can have gaps.",
        flush=True,
    )

    universes = get_point_in_time_sp500(start, end, freq="quarterly")
    print("=== Constituent count by quarter ===", flush=True)
    for sample, members in universes.items():
        flag = ""
        if not 490 <= len(members) <= 520:
            flag = "  <- well away from 500, check the source row"
            _log.warning("%s has %s constituents, well away from about 500", sample.isoformat(), len(members))
        print(f"  {sample.isoformat()}  {len(members):3d} names{flag}", flush=True)

    union = get_union_universe_sp500(start, end)
    quarter_sizes = [len(members) for members in universes.values()]
    print(f"Union size over the window: {len(union)} names", flush=True)
    print(
        f"Single-quarter size: min={min(quarter_sizes)}  max={max(quarter_sizes)}"
        " (the index is nominally 500 names; add/delete gaps move the count by a few)",
        flush=True,
    )
    if len(union) < max(quarter_sizes):
        raise RuntimeError("The union is smaller than a single quarter, so the point-in-time lists are wrong")

    report = compare_universes_sp500(TICKER_LIST, union)
    overlap_pct = float(report["overlap_pct"])
    only_current = list(report["only_in_current"])
    only_historical = list(report["only_in_historical"])
    overlap = list(report["overlap"])
    not_listed = list(report["not_sp500_listed"])
    excluded = list(report["sp500_but_excluded"])
    unknown = list(report["eligibility_unknown"])
    classified = len(not_listed) + len(excluded) + len(unknown)
    if classified != len(only_current):
        raise RuntimeError(
            f"only_in_current split counts do not add up: {classified} != {len(only_current)}"
        )

    print("=== Current 60 names vs the S&P 500 union from 2022 ===", flush=True)
    print(f"Current list: {len(TICKER_LIST)} names", flush=True)
    print(f"Overlap: {len(overlap)} names", flush=True)
    print(
        f"overlap_pct: {overlap_pct:.2f}% (share of the current list that entered this stretch of S&P 500 history)",
        flush=True,
    )
    _print_ticker_block("Only on the current list, never in this S&P 500 history", only_current)
    _print_ticker_block("Overlap", overlap)
    _print_ticker_block("Only in historical S&P 500, not on the current list", only_historical)

    print("=== only_in_current split by whether the name looks S&P 500 eligible ===", flush=True)
    print(
        "not_sp500_listed: country is not the United States, or the name says ADR. "
        "The S&P 500 usually includes only US companies. This is not survivorship bias.",
        flush=True,
    )
    _print_ticker_block("not_sp500_listed", not_listed)
    print(
        "sp500_but_excluded: a US company that did not enter the S&P 500 during this window. "
        "These are the names screened today that were not in the index in the past.",
        flush=True,
    )
    _print_ticker_block("sp500_but_excluded", excluded)
    if unknown:
        reasons = report["eligibility_unknown_reasons"]
        reason_map = reasons if isinstance(reasons, dict) else {}
        print(
            "eligibility_unknown: yfinance did not return enough country data, so these names were not placed in either group above.",
            flush=True,
        )
        print(f"eligibility_unknown ({len(unknown)} names)", flush=True)
        for ticker in unknown:
            print(f"  {ticker} ({reason_map.get(ticker, 'no country')})", flush=True)

    current = {str(ticker).upper() for ticker in TICKER_LIST}
    nasdaq_union = get_union_universe(start, end)
    nasdaq_overlap = current & nasdaq_union
    nasdaq_pct = round(100.0 * len(nasdaq_overlap) / len(current), 2) if current else 0.0
    print("=== overlap_pct comparison (share of the current list that entered that index) ===", flush=True)
    print(f"  NASDAQ-100   {nasdaq_pct:6.2f}%   (overlap {len(nasdaq_overlap)} / {len(current)})", flush=True)
    print(f"  S&P 500      {overlap_pct:6.2f}%   (overlap {len(overlap)} / {len(current)})", flush=True)
    print(
        "Symbols are compared literally. exchange cannot separate an ADR: TSM and DIS are both NYQ, and nationality comes from country.",
        flush=True,
    )
