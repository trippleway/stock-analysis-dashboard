# -*- coding: utf-8 -*-
"""
Point-in-time universe for backtests: the NASDAQ-100 constituents on that date, so the backtest avoids survivorship bias.

Today's hand-picked 60 names were screened by the gains of the last few years. Backtesting that list from 3 years ago
would peek at the future. This module asks nasdaq-100-ticker-history
which stocks were actually in the index on a historical date.

Dependency (PyPI has no release; install from GitHub):
    pip install "nasdaq-100-ticker-history @ git+https://github.com/jmccarrell/n100tickers.git"

Run from the project root:
    python src/backtest_universe.py
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

_ROOT = Path(__file__).resolve().parent.parent
_SRC = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from nasdaq_100_ticker_history import tickers_as_of
except ImportError as exc:  # pragma: no cover - install hint when the package is missing
    raise ImportError(
        "nasdaq-100-ticker-history was not found. Install it:"
        ' pip install "nasdaq-100-ticker-history @ git+https://github.com/jmccarrell/n100tickers.git"'
    ) from exc

_log = logging.getLogger("backtest_universe")
if not _log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# Sample frequency mapped to a pandas offset. Quarterly uses the start of the quarter (1/1, 4/1, 7/1, 10/1),
# so "2023 Q1" is the real constituent list on 2023-01-01, not a look at quarter-end.
_FREQ_TO_PANDAS: dict[str, str] = {
    "quarterly": "QS",
    "monthly": "MS",
    "yearly": "YS",
    "daily": "D",
}

# NASDAQ listing codes in yfinance info["exchange"].
# NMS / NGM / NCM are Nasdaq Global Select / Global / Capital.
_NASDAQ_EXCHANGE_CODES = frozenset({"NMS", "NGM", "NCM", "NAS", "NASDAQ"})
_YF_CALL_SLEEP_SEC = 0.7
_YF_RETRY_DELAYS = (1.0, 2.0, 4.0)


class UniverseCoverageError(RuntimeError):
    """The query date is outside the coverage of nasdaq-100-ticker-history, or the package returned an empty list."""


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


def _membership_as_of(as_of: date) -> set[str]:
    """
    Load the NASDAQ-100 constituents for one date.

    When the package has no data for that year it raises NotImplementedError. This turns that into
    UniverseCoverageError and names the date that is outside coverage.
    An empty set is also a failure, so a backtest does not continue silently with an empty universe.

    Args:
        as_of: Query date (index changes effective that day are included).

    Returns:
        The set of constituent symbols that day (uppercase).
    """
    label = as_of.isoformat()
    try:
        raw = tickers_as_of(as_of.year, as_of.month, as_of.day)
    except NotImplementedError as exc:
        message = (
            f"Cannot load NASDAQ-100 constituents for {label}: "
            f"that date is outside the coverage of nasdaq-100-ticker-history. Original error: {exc}"
        )
        _log.error(message)
        raise UniverseCoverageError(message) from exc
    except Exception as exc:
        message = f"Cannot load NASDAQ-100 constituents for {label}: {exc}"
        _log.error(message)
        raise UniverseCoverageError(message) from exc

    symbols = {str(ticker).upper().strip() for ticker in raw if str(ticker).strip()}
    if not symbols:
        message = (
            f"Cannot load NASDAQ-100 constituents for {label}: "
            "the package returned an empty set. That is treated as outside coverage or bad data, and the run will not continue silently."
        )
        _log.error(message)
        raise UniverseCoverageError(message)
    return symbols


def _sample_dates(start: date, end: date, freq: str) -> list[date]:
    """
    Build sample dates inside the range. The backtest start is included even when it is not a frequency boundary,
    otherwise there is no universe between the start and the first sample date.

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


def get_point_in_time_universe(
    start_date: date,
    end_date: date,
    freq: str = "quarterly",
) -> dict[date, set[str]]:
    """
    Return the NASDAQ-100 constituents that were in the index on each sample date.

    quarterly aligns to the start of the quarter: 2023 Q1 uses the 2023-01-01 list, 2024 Q1 uses 2024-01-01,
    and the whole run does not share one list. A sample date outside the package coverage raises immediately.

    Args:
        start_date: Backtest start (inclusive).
        end_date: Backtest end (inclusive).
        freq: Sample frequency. Defaults to quarterly.

    Returns:
        {sample date: constituent set that day}, ordered from earliest to latest.
    """
    start = _coerce_date(start_date, "start_date")
    end = _coerce_date(end_date, "end_date")
    _validate_range(start, end)

    samples = _sample_dates(start, end, freq)
    universe: dict[date, set[str]] = {}
    for sample in samples:
        universe[sample] = _membership_as_of(sample)

    _log.info(
        "Point-in-time universe %s to %s (%s): %s dates",
        start.isoformat(),
        end.isoformat(),
        freq,
        len(universe),
    )
    return universe


def get_union_universe(start_date: date, end_date: date) -> set[str]:
    """
    Return the union of every name that entered the NASDAQ-100 during the range.

    The union is taken from daily constituents, not from quarterly snapshots only. A name added mid-quarter
    and removed before quarter-end would be missed by a quarterly snapshot, and the price download would be incomplete.

    Args:
        start_date: Range start (inclusive).
        end_date: Range end (inclusive).

    Returns:
        Symbols that appeared on at least one day during the range.
    """
    start = _coerce_date(start_date, "start_date")
    end = _coerce_date(end_date, "end_date")
    _validate_range(start, end)

    union: set[str] = set()
    cursor = start
    step = timedelta(days=1)
    while cursor <= end:
        union.update(_membership_as_of(cursor))
        cursor += step

    _log.info(
        "Historical constituent union %s to %s: %s names",
        start.isoformat(),
        end.isoformat(),
        len(union),
    )
    return union


def _is_nasdaq_listed(exchange: str) -> bool:
    """
    Decide from yfinance info['exchange'] whether the name is listed on NASDAQ.

    Args:
        exchange: Yahoo exchange code, for example NMS or NYQ.

    Returns:
        True when the name is listed on NASDAQ.
    """
    code = exchange.strip().upper()
    if code in _NASDAQ_EXCHANGE_CODES:
        return True
    # A few environments return NasdaqGS / NasdaqGM instead of NMS / NGM.
    return "NASDAQ" in code


def _quote_missing(ticker: str) -> bool:
    """
    Whether recent daily bars are completely missing. Yahoo often returns 404 for a delisted symbol, and info then has no exchange.

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


def _fetch_exchange(ticker: str) -> tuple[str | None, str]:
    """
    Read yfinance info['exchange']. A transient failure is retried 3 times, with a pause between calls.

    Args:
        ticker: Ticker symbol.

    Returns:
        (exchange code, failure reason). The failure reason is an empty string on success.
        A delisted name with a 404 quote is not retried. The reason says info has no exchange.
    """
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            time.sleep(_YF_CALL_SLEEP_SEC)
            info = yf.Ticker(ticker).info or {}
            exchange = info.get("exchange") if isinstance(info, dict) else None
            if isinstance(exchange, str) and exchange.strip():
                return exchange.strip(), ""
            # There was a response but no exchange. Retrying a delisted symbol will not fill this field.
            if isinstance(info, dict) and info and _quote_missing(ticker):
                reason = "Yahoo has no quote (it may be delisted), and info.exchange is empty"
                _log.warning("Cannot tell the exchange for %s: %s", ticker, reason)
                return None, reason
            last_err = ValueError("info.exchange is empty")
        except Exception as exc:  # noqa: BLE001 — a single-name failure is retried and does not stop the batch
            last_err = exc
        if attempt < 2:
            time.sleep(_YF_RETRY_DELAYS[attempt])
    reason = f"Still failing after 3 attempts: {last_err}"
    _log.warning("Cannot tell the exchange for %s (%s)", ticker, reason)
    return None, reason


def _split_only_in_current(
    tickers: list[str],
) -> tuple[list[str], list[str], list[str], dict[str, str]]:
    """
    Split names that are only on the current list into two groups.

    Args:
        tickers: Symbols in only_in_current.

    Returns:
        (not_nasdaq_listed, nasdaq_but_excluded, exchange_unknown, unknown_reasons).
        The first two are "exchange is not NASDAQ" and "listed on NASDAQ but never in the index".
        Names with no exchange go in the third group. They are not guessed into either side.
    """
    not_nasdaq: list[str] = []
    nasdaq_excluded: list[str] = []
    unknown: list[str] = []
    unknown_reasons: dict[str, str] = {}
    for ticker in tickers:
        exchange, reason = _fetch_exchange(ticker)
        if exchange is None:
            unknown.append(ticker)
            unknown_reasons[ticker] = reason or "info.exchange is empty"
            continue
        if _is_nasdaq_listed(exchange):
            nasdaq_excluded.append(ticker)
            _log.info("%s exchange=%s -> listed on NASDAQ but not in the index", ticker, exchange)
        else:
            not_nasdaq.append(ticker)
            _log.info("%s exchange=%s -> not listed on NASDAQ", ticker, exchange)
    return sorted(not_nasdaq), sorted(nasdaq_excluded), sorted(unknown), unknown_reasons


def compare_universes(
    current_list: list[str],
    point_in_time_union: set[str],
) -> dict[str, list[str] | float | dict[str, str]]:
    """
    Compare today's hand-picked list with the historical point-in-time constituent union.

    overlap_pct is the share of the current list that appeared in the historical NASDAQ-100 (0-100).
    The denominator is the number of names on the current list, not the historical union. The question is
    how many of today's picks were actually in the index then, not the Jaccard similarity of the two lists.
    Symbols are compared literally (FB and META are different symbols, because the index used the old symbol at the time).

    only_in_current is split again with yfinance info['exchange']:
    - not_nasdaq_listed: the exchange is not NASDAQ, so the index scope never covered it
    - nasdaq_but_excluded: listed on NASDAQ, but it did not enter the NASDAQ-100 during this history
    Names with no exchange go in exchange_unknown and are not merged into the two groups above.
    exchange_unknown_reasons records why each name was not classified.

    Args:
        current_list: The current hand-picked list.
        point_in_time_union: The result of get_union_universe.

    Returns:
        only_in_current, only_in_historical, overlap, overlap_pct,
        plus not_nasdaq_listed, nasdaq_but_excluded, exchange_unknown,
        exchange_unknown_reasons.
    """
    current = {str(ticker).upper().strip() for ticker in current_list if str(ticker).strip()}
    historical = {
        str(ticker).upper().strip() for ticker in point_in_time_union if str(ticker).strip()
    }
    overlap = current & historical
    only_current = sorted(current - historical)
    only_historical = historical - current
    overlap_pct = round(100.0 * len(overlap) / len(current), 2) if current else 0.0
    not_nasdaq, nasdaq_excluded, unknown, unknown_reasons = _split_only_in_current(only_current)

    return {
        "only_in_current": only_current,
        "only_in_historical": sorted(only_historical),
        "overlap": sorted(overlap),
        "overlap_pct": overlap_pct,
        "not_nasdaq_listed": not_nasdaq,
        "nasdaq_but_excluded": nasdaq_excluded,
        "exchange_unknown": unknown,
        "exchange_unknown_reasons": unknown_reasons,
    }


def _print_ticker_block(title: str, tickers: list[str]) -> None:
    """Print one group of symbols: the count first, then one row at a time, so a line does not get too long."""
    print(f"{title} ({len(tickers)} names)")
    if not tickers:
        print("  (none)")
        return
    line: list[str] = []
    for ticker in tickers:
        line.append(ticker)
        if len(line) == 12:
            print("  " + ", ".join(line))
            line = []
    if line:
        print("  " + ", ".join(line))


if __name__ == "__main__":
    from config import TICKER_LIST

    start = date(2022, 1, 1)
    end = date.today()
    print(
        f"NASDAQ-100 point-in-time universe: {start.isoformat()} to {end.isoformat()}, one date per quarter",
        flush=True,
    )

    universes = get_point_in_time_universe(start, end, freq="quarterly")
    print("=== Constituent count by quarter ===", flush=True)
    for sample, members in universes.items():
        print(f"  {sample.isoformat()}  {len(members):3d} names", flush=True)

    union = get_union_universe(start, end)
    quarter_sizes = [len(members) for members in universes.values()]
    print(f"Union size over the window: {len(union)} names", flush=True)
    print(
        f"Single-quarter size: min={min(quarter_sizes)}  max={max(quarter_sizes)} (the index is about 100 names; mid-quarter changes move the count by a few)",
        flush=True,
    )
    if len(union) < max(quarter_sizes):
        raise RuntimeError("The union is smaller than a single quarter, so the point-in-time lists are wrong")

    report = compare_universes(TICKER_LIST, union)
    overlap_pct = float(report["overlap_pct"])
    only_current = list(report["only_in_current"])
    only_historical = list(report["only_in_historical"])
    overlap = list(report["overlap"])

    print("=== Current 60 names vs the NASDAQ-100 union from 2022 ===", flush=True)
    print(f"Current list: {len(TICKER_LIST)} names", flush=True)
    print(f"Overlap: {len(overlap)} names", flush=True)
    print(
        f"overlap_pct: {overlap_pct:.2f}% (share of the current list that entered this stretch of NASDAQ-100 history)",
        flush=True,
    )
    _print_ticker_block("Only on the current list, never in this NASDAQ-100 history", only_current)
    _print_ticker_block("Overlap", overlap)
    _print_ticker_block("Only in historical NASDAQ-100, not on the current list", only_historical)
    print(
        "Symbols are compared literally: for example the index used FB, not META, in the first half of 2022.",
        flush=True,
    )

    not_nasdaq = list(report["not_nasdaq_listed"])
    nasdaq_excluded = list(report["nasdaq_but_excluded"])
    unknown = list(report["exchange_unknown"])
    classified = len(not_nasdaq) + len(nasdaq_excluded) + len(unknown)
    if classified != len(only_current):
        raise RuntimeError(
            f"only_in_current split counts do not add up: {classified} != {len(only_current)}"
        )

    print("=== only_in_current split by exchange ===", flush=True)
    print(
        "not_nasdaq_listed: the exchange is not NASDAQ. The index scope never covered these names. This is not survivorship bias.",
        flush=True,
    )
    _print_ticker_block("not_nasdaq_listed", not_nasdaq)
    print(
        "nasdaq_but_excluded: listed on NASDAQ, but it did not enter the NASDAQ-100 during this history. "
        "These are the names screened today that were not constituents in the past.",
        flush=True,
    )
    _print_ticker_block("nasdaq_but_excluded", nasdaq_excluded)
    if unknown:
        reasons = report["exchange_unknown_reasons"]
        print(
            "exchange_unknown: yfinance info has no exchange, so these names were not placed in either group above.",
            flush=True,
        )
        print(f"exchange_unknown ({len(unknown)} names)", flush=True)
        reason_map = reasons if isinstance(reasons, dict) else {}
        for ticker in unknown:
            reason = reason_map.get(ticker, "info.exchange is empty")
            print(f"  {ticker} ({reason})", flush=True)
