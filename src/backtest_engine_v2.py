# -*- coding: utf-8 -*-
"""
Event-driven technical backtest. Entries use that day's rank. Exits use take-profit, stop-loss, or signal decay.

Unlike a fixed monthly hold, there is no shared month-end rebalance. On each trading day, if a slot is open
and a name is inside the universe's top n_buy, it is bought at the close. After entry, the exit is checked
every day, so holding length depends on that stock's path.

Lookahead boundary:
- The entry rank uses technical scores only through that trading day, from compute_tech_score_series.
- An exit walks closes and ranks after the entry date through the decision date. It stops when a rule hits and does not read later prices.
- The self-check recomputes prices truncated at the decision date, and replays the exit on a series that ends one day earlier.

Costs match the monthly backtest: 0.15% one way (0.05% commission + 0.10% slippage),
charged on the buy and again on the sell. A position still open has been charged only the buy.

The stop sells only when the close touches it. Intraday lows are ignored. The close can gap through the threshold,
so the realized loss can exceed stop_loss_pct. The baseline stop is -10%; the worst trade was about -23%.
That is a known optimistic bias of the backtest versus live trading: the parameter looks like it caps the loss,
and the fill does not. Do not describe the stop as landing exactly on the setting.

The 10-year run uses the point-in-time S&P 500. A day can buy only names that were in the index that day.
Today's fixed list is not applied back to 2015. Before a large price download, the script prints an estimate and stops.
After confirmation, call run_ten_year_baseline(confirm=True).

Run from the project root:
    python src/backtest_engine_v2.py
    python src/backtest_engine_v2.py --estimate-10y
"""

from __future__ import annotations

import logging
import math
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_SRC = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from backtest_compare_groups import (  # noqa: E402
    _BENCHMARK,
    _DRAWDOWN_THRESHOLD,
    _MIN_HISTORY_BARS,
    _ONE_WAY_COST,
    _TRADING_DAYS,
    _WARMUP_DAYS,
    _as_timestamp_index,
    _cumulative,
    _score_frame,
    _sharpe,
    _spy_drawdown_episodes,
    define_three_groups,
    run_group_backtest,
    run_regime_split_backtest,
)
from backtest_data import build_ohlcv_panel  # noqa: E402
from backtest_signals import compute_tech_score_series  # noqa: E402
from backtest_universe_sp500 import (  # noqa: E402
    get_point_in_time_sp500,
    get_union_universe_sp500,
)
from config import BACKTEST_END, BACKTEST_START, DATA_DIR  # noqa: E402

_log = logging.getLogger("backtest_engine_v2")
if not _log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# Score recompute tolerance, same as backtest_signals.verify_no_lookahead
_SCORE_ATOL = 1e-6
# Performance start for the 10-year point-in-time run. Warmup is fetched earlier so MA200 is valid on day one.
_TEN_YEAR_START = date(2015, 1, 1)
_OHLCV_DIR = Path(DATA_DIR) / "backtest_cache" / "ohlcv"
# On a cache miss, each name is about a 0.7s pause plus one Yahoo request. Failures retry, so the upper bound is loose.
_FETCH_SEC_PER_NEW_SYMBOL = 2.0
_FETCH_SEC_PER_NEW_SYMBOL_SLOW = 4.0
# Headwind Sharpe must beat the monthly version by at least this much before the exit rule is called a clear improvement.
_SHARPE_IMPROVEMENT = 0.20
_EXIT_LABELS = {
    "take_profit": "Take profit",
    "stop_loss": "Stop loss",
    "signal_decay": "Signal decay",
    "still_holding": "Still holding",
}
# Change one exit knob at a time. Case D lengthens decay and widens the stop together. decay_rank stays 15.
_SENSITIVITY_CASES: tuple[dict[str, Any], ...] = (
    {
        "label": "Baseline",
        "note": "take profit 25%, stop -10%, decay 5 days, top 15",
        "take_profit_pct": 0.25,
        "stop_loss_pct": -0.10,
        "score_decay_days": 5,
        "score_decay_rank": 15,
    },
    {
        "label": "A",
        "note": "decay tolerance set to 10 days",
        "take_profit_pct": 0.25,
        "stop_loss_pct": -0.10,
        "score_decay_days": 10,
        "score_decay_rank": 15,
    },
    {
        "label": "B",
        "note": "stop widened to -15%",
        "take_profit_pct": 0.25,
        "stop_loss_pct": -0.15,
        "score_decay_days": 5,
        "score_decay_rank": 15,
    },
    {
        "label": "C",
        "note": "take profit cut to 15%",
        "take_profit_pct": 0.15,
        "stop_loss_pct": -0.10,
        "score_decay_days": 5,
        "score_decay_rank": 15,
    },
    {
        "label": "D",
        "note": "decay 10 days and stop -15%",
        "take_profit_pct": 0.25,
        "stop_loss_pct": -0.15,
        "score_decay_days": 10,
        "score_decay_rank": 15,
    },
)


def _as_stamp(value: date | pd.Timestamp | str) -> pd.Timestamp:
    """Normalize to a timezone-naive midnight timestamp."""
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp.normalize()


def _as_series(values: pd.Series, name: str) -> pd.Series:
    """Turn prices or scores into a float series, oldest first, with duplicate dates removed."""
    if values is None:
        raise ValueError(f"{name} cannot be empty")
    series = pd.to_numeric(values, errors="coerce").astype(float)
    series.index = _as_timestamp_index(series.index)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series


def simulate_position_exit(
    entry_date: date | pd.Timestamp,
    entry_price: float,
    entry_score: float,
    price_series: pd.Series,
    score_series: pd.Series,
    take_profit_pct: float = 0.25,
    stop_loss_pct: float = -0.10,
    score_decay_days: int = 5,
    score_decay_rank: int = 15,
    rank_series: pd.Series | None = None,
) -> dict[str, Any]:
    """
    From the trading day after entry, decide each day whether to exit.

    The day's return is the close versus the entry price, not the one-day change.
    Take-profit and stop-loss use the close only, not the intraday high or low. The close can gap through the stop,
    so the realized loss can exceed stop_loss_pct. Do not treat the setting as the fill.
    If all three rules hit on the same day,
    the order is take-profit, then stop-loss, then signal decay.

    The loop stops at the last bar the caller supplied. Truncating the series on the exit date
    matches the full series. Ending one day earlier means the original exit date is not taken.
    rank_series is that name's rank in the universe that day (1 is the highest score), computed by the caller on the cross-section.
    Exit on signal decay only after score_decay_days consecutive trading days ranked worse than the top score_decay_rank.
    Ranks from before the entry do not count toward that streak.

    Args:
        entry_date: Entry date. The buy is at that close, and that day is not an exit decision.
        entry_price: Entry close.
        entry_score: Technical score on the entry day, used only to reconcile against score_series.
        price_series: Closes. History before the entry may be included; it is not used for the exit.
        score_series: This name's technical score.
        take_profit_pct: Exit when the close is this far above the entry price.
        stop_loss_pct: Exit when the close is this far below the entry price. Must be negative.
        score_decay_days: Consecutive days of a weak rank required to exit.
        score_decay_rank: Ranks inside this top-N still count as a live signal. A worse rank counts as decay.
        rank_series: This name's daily rank in the universe. None skips the decay check.

    Returns:
        exit_date, exit_price, holding_days (calendar days from entry to exit),
        exit_reason, and return_pct (price return, before trading costs).
    """
    if take_profit_pct <= 0:
        raise ValueError(f"take_profit_pct must be positive, got {take_profit_pct}")
    if stop_loss_pct >= 0:
        raise ValueError(f"stop_loss_pct must be negative, got {stop_loss_pct}")
    if score_decay_days < 1 or score_decay_rank < 1:
        raise ValueError("Signal-decay days and rank must both be at least 1")
    if not math.isfinite(entry_price) or entry_price <= 0:
        raise ValueError(f"entry_price must be positive, got {entry_price}")

    prices = _as_series(price_series, "price_series")
    scores = _as_series(score_series, "score_series")
    ranks = _as_series(rank_series, "rank_series") if rank_series is not None else None
    entry_ts = _as_stamp(entry_date)

    if entry_ts in scores.index and math.isfinite(entry_score):
        observed = float(scores.loc[entry_ts])
        if math.isfinite(observed) and abs(observed - float(entry_score)) > 1e-4:
            _log.warning(
                "%s entry score %.4f does not match %.4f on the score series",
                entry_ts.date(),
                float(entry_score),
                observed,
            )

    future = prices.index[prices.index > entry_ts]
    exit_ts = entry_ts
    exit_price = float(entry_price)
    exit_reason = "still_holding"
    streak = 0

    for day in future:
        px = float(prices.loc[day])
        if not math.isfinite(px) or px <= 0:
            # No close today. Do not fill it from the next day, and break the decay streak.
            streak = 0
            continue
        held_return = px / float(entry_price) - 1.0
        if ranks is not None and day in ranks.index and math.isfinite(float(ranks.loc[day])):
            if float(ranks.loc[day]) > score_decay_rank:
                streak += 1
            else:
                streak = 0
        else:
            streak = 0

        exit_ts = day
        exit_price = px
        if held_return >= take_profit_pct:
            exit_reason = "take_profit"
            break
        if held_return <= stop_loss_pct:
            exit_reason = "stop_loss"
            break
        if ranks is not None and streak >= score_decay_days:
            exit_reason = "signal_decay"
            break
        exit_reason = "still_holding"

    holding_days = int((exit_ts.date() - entry_ts.date()).days)
    return {
        "exit_date": exit_ts.date(),
        "exit_price": float(exit_price),
        "holding_days": holding_days,
        "exit_reason": exit_reason,
        "return_pct": float(exit_price / float(entry_price) - 1.0),
    }


def _prepare_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize one name's OHLCV to a timezone-naive table, oldest first."""
    history = frame.copy()
    history.index = _as_timestamp_index(history.index)
    return history[~history.index.duplicated(keep="last")].sort_index()


def _top_names(score_row: pd.Series, n_buy: int) -> list[str]:
    """Top n_buy names by that day's score. Ties break by ticker so the result is reproducible."""
    ranked = score_row.dropna()
    if ranked.empty:
        return []
    table = ranked.rename("score").reset_index()
    table.columns = ["ticker", "score"]
    table = table.sort_values(["score", "ticker"], ascending=[False, True])
    return table["ticker"].head(n_buy).astype(str).tolist()


def _headwind_dates(spy_close: pd.Series) -> set[pd.Timestamp]:
    """
    Collect every headwind trading day using the existing 8% drawdown definition.

    Args:
        spy_close: SPY closes. Warmup before the backtest start may be included.

    Returns:
        Dates that fall inside any drawdown episode.
    """
    dates: set[pd.Timestamp] = set()
    for episode in _spy_drawdown_episodes(spy_close, _DRAWDOWN_THRESHOLD):
        start = _as_stamp(episode["start"])
        end = _as_stamp(episode["end"])
        stamps = pd.date_range(start, end, freq="D")
        dates.update(pd.Timestamp(item).normalize() for item in stamps)
    return dates


def _mark(positions: dict[str, dict[str, Any]], close_row: pd.Series) -> float:
    """Mark positions to that day's close. A missing price keeps the last traded price."""
    value = 0.0
    for ticker, pos in positions.items():
        px = close_row.get(ticker, np.nan)
        if px is None or not math.isfinite(float(px)) or float(px) <= 0:
            px = pos["last_price"]
        else:
            pos["last_price"] = float(px)
            px = float(px)
        value += pos["shares"] * float(px)
    return value


def _cohort_returns(
    trades: list[dict[str, Any]],
    close: pd.DataFrame,
    regime: str,
) -> pd.Series:
    """
    Compound only positions whose entry day belongs to this regime, as daily returns.

    New capital is added on the entry day and is not counted as a return. The buy cost is charged on the entry day
    and the sell cost on the exit day. Cash leaves this sub-portfolio after the exit so idle gaps do not pull Sharpe toward 0.
    On the same day, exits are applied before entries.

    Args:
        trades: Finished simulated trades. Each needs regime, allocated, and shares.
        close: Close panel.
        regime: Tailwind or Headwind.

    Returns:
        Daily returns on the days this regime had capital invested.
    """
    selected = [trade for trade in trades if trade["regime"] == regime]
    if not selected:
        return pd.Series(dtype=float)

    entries: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    exits: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    for trade in selected:
        entries.setdefault(trade["entry_ts"], []).append(trade)
        if trade["exit_reason"] != "still_holding":
            exits.setdefault(trade["exit_ts"], []).append(trade)

    open_shares: dict[str, float] = {}
    prev_value = 0.0
    index: list[pd.Timestamp] = []
    returns: list[float] = []

    for day in close.index:
        mtm = 0.0
        for ticker, shares in open_shares.items():
            px = close.at[day, ticker]
            if not math.isfinite(float(px)) or float(px) <= 0:
                continue
            mtm += shares * float(px)

        exit_cost = 0.0
        for trade in exits.get(day, []):
            ticker = trade["ticker"]
            shares = open_shares.pop(ticker, 0.0)
            px = close.at[day, ticker]
            if not math.isfinite(float(px)) or float(px) <= 0:
                px = trade["exit_price"]
            exit_cost += shares * float(px) * _ONE_WAY_COST

        entry_list = entries.get(day, [])
        new_capital = float(sum(trade["allocated"] for trade in entry_list))
        for trade in entry_list:
            open_shares[trade["ticker"]] = float(trade["shares"])

        base = prev_value + new_capital
        if base > 0:
            # Overnight marks include the sell cost. New capital's end value includes the buy cost.
            overnight_end = mtm - exit_cost
            new_end = new_capital * (1.0 - _ONE_WAY_COST)
            index.append(day)
            returns.append((overnight_end + new_end) / base - 1.0)

        remaining = 0.0
        for ticker, shares in open_shares.items():
            px = close.at[day, ticker]
            if not math.isfinite(float(px)) or float(px) <= 0:
                continue
            remaining += shares * float(px)
        prev_value = remaining

    return pd.Series(returns, index=pd.DatetimeIndex(index), dtype=float)


def _regime_stats(
    trades: list[dict[str, Any]],
    close: pd.DataFrame,
    spy_ret: pd.Series,
    regime: str,
) -> dict[str, Any]:
    """
    Summarize one entry regime: trade count, return after costs, SPY over the same days, and Sharpe.

    Args:
        trades: All trades.
        close: Close panel.
        spy_ret: SPY daily returns aligned to close.
        regime: Tailwind or Headwind.

    Returns:
        Dict with n_trades, strategy_return, spy_return, excess, sharpe, and n_days.
    """
    daily = _cohort_returns(trades, close, regime)
    spy = spy_ret.reindex(daily.index).fillna(0.0)
    strategy_return = _cumulative(daily)
    spy_return = _cumulative(spy)
    return {
        "regime": regime,
        "n_trades": int(sum(1 for trade in trades if trade["regime"] == regime)),
        "n_days": int(len(daily)),
        "strategy_return": strategy_return,
        "spy_return": spy_return,
        "excess": strategy_return - spy_return,
        "sharpe": _sharpe(daily),
        "daily_returns": daily,
    }


def verify_event_chain(
    trades: list[dict[str, Any]],
    frames: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    score: pd.DataFrame,
    ranks: pd.DataFrame,
    top_by_day: dict[pd.Timestamp, list[str]],
    *,
    n_buy: int,
    take_profit_pct: float,
    stop_loss_pct: float,
    score_decay_days: int,
    score_decay_rank: int,
    max_entry_dates: int = 8,
    max_decay_dates: int = 6,
) -> bool:
    """
    Check that the trade chain never uses data from after the decision date.

    Entry: for every trade, recompute the score on prices truncated through the entry date. It must equal the score used to place the order.
    As in verify_no_lookahead, that compares "day t of the full sample" with "the series kept only through day t".
    A sample of entry dates also truncates every name and re-ranks. The top n_buy must match the backtest.

    Exit: replay on prices and ranks truncated through the exit date. The exit date, price, and reason must match.
    Ending the series one trading day earlier must not still exit on the original date.

    Args:
        trades: Trades produced by run_event_driven_backtest.
        frames: Per-name OHLCV, used to recompute truncated scores.
        close: Close panel the backtest actually used. The exit replay follows this calendar.
        score: Score panel used by the backtest.
        ranks: Rank panel used by the backtest.
        top_by_day: The top n_buy names actually used on each trading day.
        n_buy: How many names the entry looks at.
        take_profit_pct, stop_loss_pct, score_decay_days, score_decay_rank: the same exit parameters as the backtest.
        max_entry_dates: How many days to sample for the cross-section re-rank.
        max_decay_dates: How many signal-decay exit days to re-rank across the whole universe.

    Returns:
        True when every check passes.
    """
    if not trades:
        _log.warning("No trades to check for lookahead")
        return False

    failures: list[str] = []
    score_cache: dict[tuple[str, pd.Timestamp], float] = {}

    def _truncated_score(ticker: str, day: pd.Timestamp) -> float:
        key = (ticker, day)
        if key in score_cache:
            return score_cache[key]
        history = frames.get(ticker)
        if history is None or history.empty:
            score_cache[key] = float("nan")
            return float("nan")
        clipped = history.loc[:day]
        if clipped.empty:
            score_cache[key] = float("nan")
            return float("nan")
        try:
            series = compute_tech_score_series(clipped)
        except Exception as exc:  # noqa: BLE001
            _log.warning("%s score recompute truncated at %s failed: %s", ticker, day.date(), exc)
            score_cache[key] = float("nan")
            return float("nan")
        value = float(series.iloc[-1]) if len(series) else float("nan")
        score_cache[key] = value
        return value

    for trade in trades:
        ticker = trade["ticker"]
        entry_ts = trade["entry_ts"]
        recomputed = _truncated_score(ticker, entry_ts)
        used = float(trade["entry_score"])
        if not _scores_close(recomputed, used):
            failures.append(
                f"{ticker} {entry_ts.date()} entry score backtest={used:.4f} truncated recompute={recomputed:.4f}"
            )

        exit_ts = trade["exit_ts"]
        price = _as_series(close[ticker], ticker)
        rank = _as_series(ranks[ticker], ticker) if ticker in ranks.columns else None
        score_row = _as_series(score[ticker], ticker) if ticker in score.columns else price * np.nan
        replay = simulate_position_exit(
            entry_ts,
            trade["entry_price"],
            trade["entry_score"],
            price.loc[:exit_ts],
            score_row.loc[:exit_ts],
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            score_decay_days=score_decay_days,
            score_decay_rank=score_decay_rank,
            rank_series=None if rank is None else rank.loc[:exit_ts],
        )
        if (
            replay["exit_reason"] != trade["exit_reason"]
            or replay["exit_date"] != trade["exit_date"]
            or not _scores_close(replay["exit_price"], trade["exit_price"], atol=1e-6)
        ):
            failures.append(
                f"{ticker} exit replay mismatch: backtest {trade['exit_date']} {trade['exit_reason']}, "
                f"replay {replay['exit_date']} {replay['exit_reason']}"
            )

        earlier = price.index[(price.index > entry_ts) & (price.index < exit_ts)]
        if len(earlier) == 0:
            if trade["exit_reason"] != "still_holding" and trade["exit_ts"] > entry_ts:
                blocked = simulate_position_exit(
                    entry_ts,
                    trade["entry_price"],
                    trade["entry_score"],
                    price.loc[:entry_ts],
                    score_row.loc[:entry_ts],
                    take_profit_pct=take_profit_pct,
                    stop_loss_pct=stop_loss_pct,
                    score_decay_days=score_decay_days,
                    score_decay_rank=score_decay_rank,
                    rank_series=None if rank is None else rank.loc[:entry_ts],
                )
                if blocked["exit_reason"] != "still_holding":
                    failures.append(
                        f"{ticker} exited with {blocked['exit_reason']} when the series ended on the entry date"
                    )
            continue
        prev = earlier[-1]
        early = simulate_position_exit(
            entry_ts,
            trade["entry_price"],
            trade["entry_score"],
            price.loc[:prev],
            score_row.loc[:prev],
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            score_decay_days=score_decay_days,
            score_decay_rank=score_decay_rank,
            rank_series=None if rank is None else rank.loc[:prev],
        )
        if early["exit_date"] == trade["exit_date"]:
            failures.append(
                f"{ticker} still exited on {trade['exit_date']} via {early['exit_reason']} when one day was removed"
            )
        if trade["exit_reason"] != "still_holding" and early["exit_reason"] != "still_holding":
            failures.append(
                f"{ticker} already triggered {early['exit_reason']} the day before the exit, but the backtest waited until {trade['exit_date']}"
            )

    sample_days = _spread_dates(sorted(top_by_day), max_entry_dates)
    decay_days = _spread_dates(
        sorted({trade["exit_ts"] for trade in trades if trade["exit_reason"] == "signal_decay"}),
        max_decay_dates,
    )
    for day in sample_days:
        recomputed_row = {}
        for ticker in score.columns:
            recomputed_row[ticker] = _truncated_score(ticker, day)
        fresh = pd.Series(recomputed_row, dtype=float)
        # Keep only names that had a rank that day, so a short-history name does not suddenly qualify after truncation.
        eligible = fresh.where(ranks.loc[day].notna())
        fresh_top = _top_names(eligible, n_buy)
        used_top = top_by_day.get(day, [])
        if fresh_top != used_top:
            failures.append(
                f"{day.date()} top {n_buy} mismatch: backtest {used_top}, truncated recompute {fresh_top}"
            )

    for day in decay_days:
        for ticker in score.columns:
            _truncated_score(ticker, day)
        for trade in trades:
            if trade["exit_reason"] != "signal_decay" or trade["exit_ts"] != day:
                continue
            ticker = trade["ticker"]
            recomputed = {
                name: _truncated_score(name, day)
                for name in score.columns
            }
            fresh_rank = pd.Series(recomputed, dtype=float)
            fresh_rank = fresh_rank.where(ranks.loc[day].notna()).rank(ascending=False, method="min")
            used_rank = float(ranks.at[day, ticker]) if ticker in ranks.columns else float("nan")
            new_rank = float(fresh_rank.get(ticker, np.nan))
            if not _scores_close(used_rank, new_rank, atol=0.0):
                failures.append(
                    f"{ticker} {day.date()} decay-exit rank backtest={used_rank} truncated recompute={new_rank}"
                )

    if failures:
        _log.warning("Trade-chain lookahead check failed, %s mismatches", len(failures))
        for item in failures[:12]:
            _log.warning("  %s", item)
        return False

    _log.info(
        "Trade-chain lookahead check passed: entry scores and exit replays match for %s trades",
        len(trades),
    )
    return True


def _scores_close(left: float, right: float, atol: float = _SCORE_ATOL) -> bool:
    """Whether two scores match. NaN on both sides counts as a match."""
    if not math.isfinite(left) and not math.isfinite(right):
        return True
    if not math.isfinite(left) or not math.isfinite(right):
        return False
    return abs(left - right) <= atol


def _spread_dates(days: list[pd.Timestamp], limit: int) -> list[pd.Timestamp]:
    """Sample dates evenly from a sorted list. Keep all of them when there are fewer than the limit."""
    if limit < 1 or not days:
        return []
    if len(days) <= limit:
        return list(days)
    picks = [round(i * (len(days) - 1) / (limit - 1)) for i in range(limit)]
    return [days[i] for i in picks]


def _yahoo_symbol(ticker: str) -> str:
    """The S&P history file uses BRK.B. Yahoo quotes need BRK-B."""
    return str(ticker).upper().strip().replace(".", "-")


def _in_index_mask(
    universe: dict[date, set[str]],
    index: pd.DatetimeIndex,
    columns: list[str],
) -> pd.DataFrame:
    """
    Whether each trading day belongs to the latest quarterly constituent list on or before that day.

    From a sample date until the next sample date, that list stays in force.
    A column that is not on the list is False for the whole day. Later index entry is not applied backward.

    Args:
        universe: Result of get_point_in_time_sp500.
        index: Trading days.
        columns: Tickers on the price panel. They must use the same symbols as the constituent list.

    Returns:
        A boolean frame aligned to index and columns.
    """
    mask = pd.DataFrame(False, index=index, columns=columns)
    if not universe or len(index) == 0:
        return mask
    samples = sorted(universe)
    listed = set(columns)
    for pos, sample in enumerate(samples):
        start = pd.Timestamp(sample)
        if pos + 1 < len(samples):
            window = (index >= start) & (index < pd.Timestamp(samples[pos + 1]))
        else:
            window = index >= start
        names = [name for name in universe[sample] if name in listed]
        if names and bool(window.any()):
            mask.loc[window, names] = True
    return mask


def _eligible_mask(
    score: pd.DataFrame,
    ready: pd.DataFrame,
    close: pd.DataFrame,
    membership: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Scores eligible to be candidates that day. Ineligible cells are NaN and are not ranked.

    The original gates still apply: a close, a technical score, and enough history.
    If a point-in-time membership frame is given, the name must be on that day's list.
    This frame only decides who can be bought. It does not force an exit of a name already held.

    Args:
        score: Technical scores.
        ready: Whether the minimum history is met.
        close: Closes.
        membership: Point-in-time constituents. None keeps the caller's fixed list.

    Returns:
        A score frame with the same shape. Ineligible cells are NaN.
    """
    allowed = ready.astype(bool) & score.notna() & close.notna()
    if membership is not None:
        in_index = membership.reindex(index=score.index, columns=score.columns)
        allowed = allowed & in_index.fillna(False).astype(bool)
    return score.where(allowed)


def run_event_driven_backtest(
    tickers: list[str] | None = None,
    years: int | None = None,
    max_positions: int = 5,
    take_profit_pct: float = 0.25,
    stop_loss_pct: float = -0.10,
    score_decay_days: int = 5,
    score_decay_rank: int = 15,
    n_buy: int = 3,
    start: date | None = None,
    end: date | None = None,
    as_of_universe: dict[date, set[str]] | None = None,
) -> dict[str, Any]:
    """
    Run an event-driven backtest on one universe: buy only when a slot is open, sell only when a rule hits.

    At each close, take the top n_buy eligible names by that day's technical score.
    Buy a name that is not already held, and only while a slot is open. At most max_positions names are held.
    A new position is 1/max_positions of equity. If cash is short, use the cash that is left.
    There is no borrowing, and existing positions are not rebalanced.

    Exits call simulate_position_exit. A trade is classified by whether its entry day is tailwind or headwind.
    The hold is not split if it later crosses into the other regime.

    If as_of_universe is passed, the candidate list is the latest quarterly S&P 500 constituents on or before that day.
    Ranks are computed only inside that day's list. Names not yet in the index, or already removed, are not used.
    A name already held that later leaves the index is not forced out. It still waits for take-profit, stop-loss, or signal decay.
    After removal its rank is empty, so signal decay stops accumulating. Take-profit and stop-loss still apply.

    Args:
        tickers: Fixed candidate symbols. Optional when as_of_universe is set; the download set is the union instead.
        years: Legacy "how many years back from today". None uses config.BACKTEST_START through BACKTEST_END.
            A value is used, and the window then moves with the run date, only when start is not also set.
        max_positions: Maximum names held at once.
        take_profit_pct: Take-profit threshold.
        stop_loss_pct: Stop-loss threshold. Negative.
        score_decay_days: Consecutive weak-rank days required to exit.
        score_decay_rank: Ranks inside this top-N still count as a live signal. In a ~500-name point-in-time pool, top 15 is much stricter than in 34 names.
        n_buy: A name is considered for a buy only if its score is inside this top-N.
        start: First performance date, inclusive. None, and years not passed, uses config.BACKTEST_START.
        end: Last performance date, inclusive. None, and years not passed, uses config.BACKTEST_END.
        as_of_universe: Quarterly point-in-time constituents. None uses tickers for the whole window.

    Returns:
        Daily returns after costs, cumulative return, Sharpe, the trade list, holding days,
        exit reasons, annualized one-way fills, and tailwind/headwind performance split by entry day.
    """
    if max_positions < 1 or n_buy < 1:
        raise ValueError("max_positions and n_buy must both be at least 1")

    # "Years back from today" is used only when years is passed and start is not. Otherwise the window is fixed.
    if years is not None and start is None:
        if years < 1:
            raise ValueError(f"years must be at least 1, got {years}")
        end_day = end or date.today()
        eval_start = end_day - timedelta(days=365 * years)
    else:
        end_day = end if end is not None else BACKTEST_END
        eval_start = start if start is not None else BACKTEST_START
    if eval_start > end_day:
        raise ValueError(f"Backtest start {eval_start} is after end {end_day}")
    fetch_start = eval_start - timedelta(days=_WARMUP_DAYS)
    if as_of_universe is not None:
        symbols = sorted({str(ticker).upper().strip() for names in as_of_universe.values() for ticker in names})
    else:
        if not tickers:
            raise ValueError("No fixed list and no point-in-time universe, so the backtest cannot run")
        symbols = [str(ticker).upper().strip() for ticker in tickers if str(ticker).strip()]
    yahoo_symbols = sorted({_yahoo_symbol(ticker) for ticker in symbols})

    panel = build_ohlcv_panel([*yahoo_symbols, _BENCHMARK], fetch_start, end_day)
    spy = panel.get(_BENCHMARK)
    if spy is None or spy.empty:
        raise RuntimeError("No SPY prices, so tailwind and headwind cannot be split")

    master = _as_timestamp_index(spy.index)
    master = master[(master >= pd.Timestamp(fetch_start)) & (master <= pd.Timestamp(end_day))]
    frames = {}
    for ticker in symbols:
        raw = panel.get(_yahoo_symbol(ticker))
        if raw is None or raw.empty:
            continue
        frames[ticker] = _prepare_history(raw)
    dropped = sorted(set(symbols) - set(frames))
    if dropped:
        _log.warning("These symbols have no prices and were dropped from the event-driven backtest: %s", ",".join(dropped))
    if not frames:
        raise RuntimeError("No usable prices, so the backtest cannot run")

    close, score, ready = _score_frame(frames, master)
    membership = None
    if as_of_universe is not None:
        membership = _in_index_mask(as_of_universe, close.index, list(close.columns))
        _log.info(
            "Point-in-time constituents: %s quarterly snapshots, prices available for %s / %s names",
            len(as_of_universe),
            len(frames),
            len(symbols),
        )
    eligible = _eligible_mask(score, ready, close, membership)
    ranks = eligible.rank(axis=1, ascending=False, method="min")

    spy_close = pd.to_numeric(spy["Close"], errors="coerce")
    spy_close.index = _as_timestamp_index(spy_close.index)
    spy_close = spy_close.reindex(master)
    all_episodes = _spy_drawdown_episodes(spy_close.dropna(), _DRAWDOWN_THRESHOLD)
    headwind = _headwind_dates(spy_close.dropna())

    trade_days = master[(master >= pd.Timestamp(eval_start)) & (master <= pd.Timestamp(end_day))]
    if len(trade_days) < 2:
        raise RuntimeError(f"Not enough trading days between {eval_start} and {end_day}")

    cash = 1.0
    positions: dict[str, dict[str, Any]] = {}
    trades: list[dict[str, Any]] = []
    top_by_day: dict[pd.Timestamp, list[str]] = {}
    equity_prev = 1.0
    daily_index: list[pd.Timestamp] = []
    daily_values: list[float] = []
    same_day_reentries = 0

    for day_i, day in enumerate(trade_days):
        close_row = close.loc[day]
        exiting: list[tuple[str, dict[str, Any]]] = []
        for ticker, pos in positions.items():
            result = simulate_position_exit(
                pos["entry_date"],
                pos["entry_price"],
                pos["entry_score"],
                close[ticker].loc[:day],
                score[ticker].loc[:day],
                take_profit_pct=take_profit_pct,
                stop_loss_pct=stop_loss_pct,
                score_decay_days=score_decay_days,
                score_decay_rank=score_decay_rank,
                rank_series=ranks[ticker].loc[:day],
            )
            if result["exit_reason"] == "still_holding":
                continue
            exit_ts = _as_stamp(result["exit_date"])
            if exit_ts != day:
                _log.warning(
                    "%s should have exited on %s, but the simulation day is %s; closing with the trigger-day result",
                    ticker,
                    result["exit_date"],
                    day.date(),
                )
            exiting.append((ticker, result))

        exited_today: set[str] = set()
        for ticker, result in exiting:
            pos = positions.pop(ticker)
            px = result["exit_price"]
            cash += pos["shares"] * px * (1.0 - _ONE_WAY_COST)
            exited_today.add(ticker)
            trades.append(_closed_trade(pos, result, day))

        if day_i < len(trade_days) - 1:
            free_slots = max_positions - len(positions)
            top = _top_names(eligible.loc[day], n_buy)
            top_by_day[day] = top
            candidates = [ticker for ticker in top if ticker not in positions]
            picks = candidates[:free_slots]
            if picks:
                equity_now = cash + _mark(positions, close_row)
                each_target = equity_now / max_positions
                budget = min(cash, each_target * len(picks))
                each = budget / len(picks) if picks else 0.0
                for ticker in picks:
                    px = close_row.get(ticker, np.nan)
                    if not math.isfinite(float(px)) or float(px) <= 0 or each <= 0:
                        continue
                    score_now = score.at[day, ticker]
                    positions[ticker] = {
                        "ticker": ticker,
                        "entry_date": day.date(),
                        "entry_ts": day,
                        "entry_price": float(px),
                        "entry_score": float(score_now),
                        "shares": each * (1.0 - _ONE_WAY_COST) / float(px),
                        "allocated": each,
                        "last_price": float(px),
                        "regime": "Headwind" if day in headwind else "Tailwind",
                    }
                    cash -= each
                    if ticker in exited_today:
                        same_day_reentries += 1

        equity = cash + _mark(positions, close_row)
        daily_index.append(day)
        daily_values.append(equity / equity_prev - 1.0)
        equity_prev = equity

    for ticker, pos in list(positions.items()):
        last_day = trade_days[-1]
        result = simulate_position_exit(
            pos["entry_date"],
            pos["entry_price"],
            pos["entry_score"],
            close[ticker].loc[:last_day],
            score[ticker].loc[:last_day],
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            score_decay_days=score_decay_days,
            score_decay_rank=score_decay_rank,
            rank_series=ranks[ticker].loc[:last_day],
        )
        # Still open when the simulation ends. Record it as still holding so a last-day hit is not counted twice.
        if result["exit_reason"] == "still_holding":
            trades.append(_closed_trade(pos, result, last_day))

    daily = pd.Series(daily_values, index=pd.DatetimeIndex(daily_index), dtype=float)
    if trades:
        first_entry = min(trade["entry_ts"] for trade in trades)
        daily = daily.loc[daily.index >= first_entry]

    spy_ret = spy_close.pct_change().reindex(daily.index).fillna(0.0)
    regime_stats = {
        name: _regime_stats(trades, close.loc[daily.index[0] : daily.index[-1]], spy_ret, name)
        if len(daily)
        else _regime_stats(trades, close, spy_ret, name)
        for name in ("Tailwind", "Headwind")
    }
    holds = [int(trade["holding_days"]) for trade in trades]
    reasons: dict[str, int] = {key: 0 for key in _EXIT_LABELS}
    for trade in trades:
        reasons[trade["exit_reason"]] = reasons.get(trade["exit_reason"], 0) + 1

    n_entries = len(trades)
    n_exits = int(sum(1 for trade in trades if trade["exit_reason"] != "still_holding"))
    if daily.empty:
        span_years = max((end_day - eval_start).days / 365.25, 1 / 365.25)
    else:
        span_years = max((daily.index[-1] - daily.index[0]).days / 365.25, 1 / 365.25)
    annualized_sides = (n_entries + n_exits) / span_years

    lookahead_ok = verify_event_chain(
        trades,
        frames,
        close,
        score,
        ranks,
        top_by_day,
        n_buy=n_buy,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        score_decay_days=score_decay_days,
        score_decay_rank=score_decay_rank,
    )

    result = {
        "tickers": symbols,
        "dropped_tickers": dropped,
        "start": daily.index[0].date().isoformat() if len(daily) else "",
        "end": daily.index[-1].date().isoformat() if len(daily) else "",
        "trades": trades,
        "daily_returns": daily,
        "spy_returns": spy_ret,
        "cumulative_return_net": _cumulative(daily),
        "sharpe": _sharpe(daily),
        "avg_holding_days": float(np.mean(holds)) if holds else float("nan"),
        "holding_min": int(min(holds)) if holds else 0,
        "holding_median": float(np.median(holds)) if holds else float("nan"),
        "holding_max": int(max(holds)) if holds else 0,
        "exit_reasons": reasons,
        "n_entries": n_entries,
        "n_exits": n_exits,
        "n_still_holding": int(reasons.get("still_holding", 0)),
        "n_same_day_reentries": same_day_reentries,
        "span_years": span_years,
        "annualized_sides": annualized_sides,
        "regime": regime_stats,
        "lookahead_ok": lookahead_ok,
        "max_positions": max_positions,
        "n_buy": n_buy,
        "take_profit_pct": take_profit_pct,
        "stop_loss_pct": stop_loss_pct,
        "score_decay_days": score_decay_days,
        "score_decay_rank": score_decay_rank,
        "drawdown_episodes": [
            episode
            for episode in all_episodes
            if pd.Timestamp(episode["end"]) >= pd.Timestamp(eval_start)
            and pd.Timestamp(episode["start"]) <= pd.Timestamp(end_day)
        ],
        "as_of_universe": as_of_universe is not None,
        "n_universe_symbols": len(symbols),
        "n_with_prices": len(frames),
    }
    _log.info(
        "Event-driven backtest %s ~ %s: %.2f%% after costs, Sharpe %.2f, %s trades",
        result["start"],
        result["end"],
        result["cumulative_return_net"] * 100,
        result["sharpe"] if math.isfinite(result["sharpe"]) else float("nan"),
        n_entries,
    )
    return result


def _closed_trade(
    pos: dict[str, Any],
    result: dict[str, Any],
    fallback_day: pd.Timestamp,
) -> dict[str, Any]:
    """Turn a position and an exit result into one trade record."""
    exit_date = result["exit_date"]
    exit_ts = _as_stamp(exit_date) if exit_date is not None else fallback_day
    return {
        "ticker": pos["ticker"],
        "entry_date": pos["entry_date"],
        "entry_ts": pos["entry_ts"],
        "entry_price": pos["entry_price"],
        "entry_score": pos["entry_score"],
        "allocated": pos["allocated"],
        "shares": pos["shares"],
        "regime": pos["regime"],
        "exit_date": result["exit_date"],
        "exit_ts": exit_ts,
        "exit_price": result["exit_price"],
        "holding_days": result["holding_days"],
        "exit_reason": result["exit_reason"],
        "return_pct": result["return_pct"],
    }


def _assert_exit_rules() -> None:
    """Check the three exit rules and the truncated replay on synthetic prices, without market data."""
    index = pd.bdate_range("2024-01-01", periods=12)
    flat = pd.Series(100.0, index=index)
    scores = pd.Series(80.0, index=index)
    ranks = pd.Series(3.0, index=index)

    surged = flat.copy()
    surged.iloc[3] = 130.0
    took = simulate_position_exit(index[0], 100.0, 80.0, surged, scores, rank_series=ranks)
    if took["exit_reason"] != "take_profit" or took["exit_date"] != index[3].date():
        raise AssertionError(f"Take-profit decision was wrong: {took}")

    cut = flat.copy()
    cut.iloc[2] = 88.0
    stopped = simulate_position_exit(index[0], 100.0, 80.0, cut, scores, rank_series=ranks)
    if stopped["exit_reason"] != "stop_loss" or stopped["exit_date"] != index[2].date():
        raise AssertionError(f"Stop-loss decision was wrong: {stopped}")

    weak = ranks.copy()
    weak.iloc[1:6] = 20.0
    decayed = simulate_position_exit(index[0], 100.0, 80.0, flat, scores, rank_series=weak)
    if decayed["exit_reason"] != "signal_decay" or decayed["exit_date"] != index[5].date():
        raise AssertionError(f"Signal-decay decision was wrong: {decayed}")

    reset = weak.copy()
    reset.iloc[4] = 5.0
    held = simulate_position_exit(index[0], 100.0, 80.0, flat, scores, rank_series=reset)
    if held["exit_reason"] == "signal_decay":
        raise AssertionError(f"Rank recovered into the top group but was still treated as decay: {held}")

    early = simulate_position_exit(
        index[0],
        100.0,
        80.0,
        surged.loc[: index[2]],
        scores.loc[: index[2]],
        rank_series=ranks.loc[: index[2]],
    )
    if early["exit_reason"] != "still_holding":
        raise AssertionError(f"Exited before the take-profit price was reached: {early}")
    _log.info("Exit-rule self-check passed")


def _pct(value: float) -> str:
    """Format a ratio as a percent."""
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value * 100:.1f}%"


def _num(value: float) -> str:
    """Format a Sharpe ratio to two decimals."""
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:.2f}"


def _pp(value: float) -> str:
    """Format excess return in percentage points."""
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value * 100:.1f}pp"


def _reason_text(reasons: dict[str, int], total: int) -> str:
    """Counts and shares of exit reasons."""
    if total <= 0:
        return "no trades"
    parts = []
    for key in ("take_profit", "stop_loss", "signal_decay", "still_holding"):
        count = int(reasons.get(key, 0))
        parts.append(f"{_EXIT_LABELS[key]} {count / total * 100:.0f}%")
    return " / ".join(parts)


def _segment(regime: dict[str, Any], name: str) -> dict[str, Any]:
    """Pull the tailwind or headwind row from a monthly regime result."""
    for row in regime["segments"]:
        if row["name"] == name:
            return row
    raise KeyError(name)


def _verdict(monthly_head: dict[str, Any], event_head: dict[str, Any], event: dict[str, Any]) -> str:
    """
    Compare headwind Sharpe for the event-driven run and the fixed monthly run.

    Args:
        monthly_head: The monthly run's headwind segment. The stored name is Headwind.
        event_head: Event-driven trades whose entry day falls in Headwind.
        event: Full event-driven result, so the conclusion also reports total return.

    Returns:
        One conclusion paragraph.
    """
    base = float(monthly_head["sharpe"])
    new = float(event_head["sharpe"])
    total = _pct(float(event["cumulative_return_net"]))
    tail = event["regime"]["Tailwind"]
    shared = (
        f"Event-driven return after costs is {total}, Sharpe {_num(float(event['sharpe']))}. "
        f"Tailwind return {_pct(float(tail['strategy_return']))}, Sharpe {_num(float(tail['sharpe']))}. "
        f"Headwind return {_pct(float(event_head['strategy_return']))}, "
        f"SPY over the same days {_pct(float(event_head['spy_return']))}, "
        f"excess {_pp(float(event_head['excess']))}, Sharpe {_num(new)}. "
    )
    if not math.isfinite(new):
        return shared + "The headwind segment does not have enough invested days, so Sharpe cannot be compared with 0.18."
    gap = new - base
    if gap >= _SHARPE_IMPROVEMENT:
        return (
            shared
            + f"Headwind Sharpe is {gap:.2f} above the monthly run's {_num(base)}. "
            "Take-profit, stop-loss, and signal decay did improve the headwind result."
        )
    if gap > 0:
        return (
            shared
            + f"Headwind Sharpe rose from {_num(base)} to {_num(new)}, "
            f"but the gap {gap:.2f} is below {_SHARPE_IMPROVEMENT:.2f}, so this does not yet fix the headwind result."
        )
    return (
        shared
        + f"Headwind Sharpe {_num(new)} is not above the monthly run's {_num(base)}. "
        "Take-profit, stop-loss, and signal decay did not improve the headwind segment. Keep that number as it is."
    )


def _print_report(
    monthly: dict[str, Any],
    monthly_regime: dict[str, Any],
    event: dict[str, Any],
) -> None:
    """Print the monthly-versus-event comparison, the regime table, and annualized fills."""
    monthly_tail = _segment(monthly_regime, "Tailwind")
    monthly_head = _segment(monthly_regime, "Headwind")
    event_head = event["regime"]["Headwind"]
    holds = (
        f"{event['avg_holding_days']:.1f} days"
        f" (min {event['holding_min']} / median {event['holding_median']:.0f} / max {event['holding_max']})"
    )
    print("\n=== Fixed monthly vs event-driven (Group A, after costs) ===", flush=True)
    print("Rule | Average hold | Exit reasons | Total return (after costs) | Sharpe", flush=True)
    print(
        "Fixed monthly | 30 days (fixed) | N/A | "
        f"{_pct(float(monthly['cumulative_return_net']))} | "
        f"{_num(float(monthly['sharpe_net']))}",
        flush=True,
    )
    print(
        "Event-driven | "
        f"{holds} | "
        f"{_reason_text(event['exit_reasons'], event['n_entries'])} | "
        f"{_pct(float(event['cumulative_return_net']))} | "
        f"{_num(float(event['sharpe']))}",
        flush=True,
    )
    print(
        f"Holding days: min {event['holding_min']}, "
        f"median {event['holding_median']:.0f}, "
        f"max {event['holding_max']}. "
        f"{event['n_still_holding']} positions are still open; their days run through the last date in the data.",
        flush=True,
    )
    if event["holding_min"] != event["holding_max"]:
        print("Min and max differ, so the hold is no longer fixed at one month.", flush=True)
    else:
        print("Min and max are the same, so holds did not spread out in this run.", flush=True)
    gross_sharpe = float(monthly["sharpe"])
    print(
        f"The monthly row's Sharpe is {_num(float(monthly['sharpe_net']))} after costs. "
        f"The earlier summary printed {_num(gross_sharpe)}, which is Sharpe before costs.",
        flush=True,
    )

    print("\n=== Entry-day regime ===", flush=True)
    print(
        "The monthly run keeps its original split: a day is headwind only if that day itself sits in a SPY drawdown. "
        "The event-driven run classifies the whole trade by its entry day. If the hold later crosses regimes, it stays with the entry day's regime.",
        flush=True,
    )
    print("              | Fixed monthly | Event-driven", flush=True)
    print(
        f"Tailwind Sharpe | {_num(float(monthly_tail['sharpe']))} | "
        f"{_num(float(event['regime']['Tailwind']['sharpe']))}",
        flush=True,
    )
    print(
        f"Headwind Sharpe | {_num(float(monthly_head['sharpe']))} | "
        f"{_num(float(event_head['sharpe']))}",
        flush=True,
    )
    print(
        f"Headwind return | {_pct(float(monthly_head['strategy_return']))} | "
        f"{_pct(float(event_head['strategy_return']))}",
        flush=True,
    )
    print(
        f"Headwind vs SPY | {_pp(float(monthly_head['excess']))} | "
        f"{_pp(float(event_head['excess']))}",
        flush=True,
    )
    print(
        f"Event-driven headwind: {event_head['n_trades']} entries, "
        f"{event_head['n_days']} invested days; "
        f"tailwind: {event['regime']['Tailwind']['n_trades']} entries, "
        f"{event['regime']['Tailwind']['n_days']} days.",
        flush=True,
    )
    print(_verdict(monthly_head, event_head, event), flush=True)

    monthly_years = max(
        (
            pd.Timestamp(monthly["end"]) - pd.Timestamp(monthly["start"])
        ).days
        / 365.25,
        1 / 365.25,
    )
    monthly_sides = float(monthly["n_name_changes"]) / monthly_years
    print("\n=== Annualized one-way fills ===", flush=True)
    print(
        "The comparable figure is buys plus sells per year. The monthly run uses rebalance count. "
        "The event-driven run uses entries plus exits. Positions still open have not been sold, so they are not exits.",
        flush=True,
    )
    print(
        f"Fixed monthly: {monthly_sides:.1f} fills/year "
        f"({int(monthly['n_name_changes'])} name changes, window {monthly_years:.2f} years)",
        flush=True,
    )
    print(
        f"Event-driven: {float(event['annualized_sides']):.1f} fills/year "
        f"(entries {event['n_entries']}, exits {event['n_exits']}, "
        f"still holding {event['n_still_holding']}, window {float(event['span_years']):.2f} years)",
        flush=True,
    )
    if event["n_same_day_reentries"]:
        print(
            f"Sold and bought back the same day {event['n_same_day_reentries']} times. Each round trip paid the one-way cost.",
            flush=True,
        )
    if float(event["annualized_sides"]) < monthly_sides:
        print(
            f"The event-driven run is about {float(event['annualized_sides']):.0f} one-way fills a year, "
            f"fewer than the monthly run's {monthly_sides:.0f}.",
            flush=True,
        )
    else:
        print(
            f"The event-driven run is about {float(event['annualized_sides']):.0f} one-way fills a year, "
            f"not below the monthly run's {monthly_sides:.0f}, so activity did not fall.",
            flush=True,
        )


def _same_exit_params(result: dict[str, Any], case: dict[str, Any]) -> bool:
    """Whether a finished backtest used this exact set of exit parameters."""
    return (
        abs(float(result["take_profit_pct"]) - float(case["take_profit_pct"])) < 1e-12
        and abs(float(result["stop_loss_pct"]) - float(case["stop_loss_pct"])) < 1e-12
        and int(result["score_decay_days"]) == int(case["score_decay_days"])
        and int(result["score_decay_rank"]) == int(case["score_decay_rank"])
    )


def _sensitivity_row(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """
    Pull the columns the sensitivity table needs from one full backtest.

    Args:
        case: Parameter set, including label and the four exit parameters.
        result: Return value of run_event_driven_backtest, already split by entry-day regime.

    Returns:
        Annualized one-way fills, tailwind and headwind Sharpe, headwind return, and total return after costs.
    """
    tail = result["regime"]["Tailwind"]
    head = result["regime"]["Headwind"]
    stops = [
        float(trade["return_pct"])
        for trade in result["trades"]
        if trade["exit_reason"] == "stop_loss"
    ]
    return {
        "label": str(case["label"]),
        "note": str(case["note"]),
        "annualized_sides": float(result["annualized_sides"]),
        "tail_sharpe": float(tail["sharpe"]),
        "head_sharpe": float(head["sharpe"]),
        "head_return": float(head["strategy_return"]),
        "total_return": float(result["cumulative_return_net"]),
        "lookahead_ok": bool(result["lookahead_ok"]),
        "worst_stop": min(stops) if stops else float("nan"),
        "stop_loss_pct": float(case["stop_loss_pct"]),
        "score_decay_days": int(result["score_decay_days"]),
        "backtest": result,
    }


def run_parameter_sensitivity(
    tickers: list[str],
    years: int | None = None,
    baseline: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Run the event-driven backtest once for each of five exit-parameter sets, split by entry-day regime.

    Baseline is take-profit 25%, stop -10%, and five days outside the top 15.
    A changes only the decay window to 10 days. B changes only the stop to -15%.
    C changes only take-profit to 15%. D uses 10-day decay and a -15% stop together.
    Entry rules, the position cap, and trading costs stay the same.

    Args:
        tickers: Candidate symbols, usually Group A.
        years: Legacy "years back from today". None keeps the fixed config window. A value overrides it.
        baseline: A finished backtest whose parameters already match the baseline. When passed, the baseline is not rerun.

    Returns:
        Five sensitivity rows, in order: baseline, A, B, C, D.
    """
    rows: list[dict[str, Any]] = []
    for case in _SENSITIVITY_CASES:
        if baseline is not None and _same_exit_params(baseline, case):
            _log.info("Sensitivity %s reuses a finished backtest", case["label"])
            result = baseline
        else:
            _log.info(
                "Sensitivity %s: take-profit %.0f%%, stop %.0f%%, decay %s days, top %s",
                case["label"],
                float(case["take_profit_pct"]) * 100,
                float(case["stop_loss_pct"]) * 100,
                case["score_decay_days"],
                case["score_decay_rank"],
            )
            result = run_event_driven_backtest(
                tickers,
                years=years,
                take_profit_pct=float(case["take_profit_pct"]),
                stop_loss_pct=float(case["stop_loss_pct"]),
                score_decay_days=int(case["score_decay_days"]),
                score_decay_rank=int(case["score_decay_rank"]),
            )
        row = _sensitivity_row(case, result)
        if not row["lookahead_ok"]:
            _log.warning("Sensitivity %s failed the trade-chain lookahead check", case["label"])
        rows.append(row)
    return rows


def _print_sensitivity(rows: list[dict[str, Any]]) -> None:
    """Print the exit-parameter sensitivity table, and note that a close-only stop can gap through its setting."""
    print("\n=== Exit-parameter sensitivity (Group A, after costs) ===", flush=True)
    for row in rows:
        print(f"{row['label']}：{row['note']}", flush=True)
    print(
        "Setting | Annualized one-way fills | Tailwind Sharpe | Headwind Sharpe | Headwind return | Total return (after costs)",
        flush=True,
    )
    for row in rows:
        if not row["lookahead_ok"]:
            print(f"{row['label']} | lookahead check failed, row not used | — | — | — | —", flush=True)
            continue
        print(
            f"{row['label']} | {row['annualized_sides']:.1f} | "
            f"{_num(row['tail_sharpe'])} | {_num(row['head_sharpe'])} | "
            f"{_pct(row['head_return'])} | {_pct(row['total_return'])}",
            flush=True,
        )
    verified = [row for row in rows if row["lookahead_ok"]]
    if not verified:
        return
    base = verified[0]
    print(
        "The stop uses the close only. The baseline setting is "
        f"{base['stop_loss_pct'] * 100:.0f}%，"
        f"The worst price return among stop exits is {_pct(base['worst_stop'])}, "
        "which did not land on the setting. That is a known optimistic bias of the backtest versus live trading. Do not describe the stop as filling exactly at the setting.",
        flush=True,
    )
    for row in verified[1:]:
        print(
            f"{row['label']} annualized one-way fills {row['annualized_sides']:.1f}, "
            f"tailwind Sharpe {_num(row['tail_sharpe'])}, "
            f"headwind Sharpe {_num(row['head_sharpe'])}, "
            f"total return {_pct(row['total_return'])}.",
            flush=True,
        )


# Local grid around case D (decay 10 days, stop -15%). Take-profit and the rank gate stay at the baseline.
_GRID_DECAY_DAYS = (7, 8, 10, 12, 15)
_GRID_STOPS = (-0.12, -0.13, -0.15, -0.17, -0.20)
_GRID_TAKE_PROFIT = 0.25
_GRID_DECAY_RANK = 15
_GRID_CENTER_DAYS = 10
_GRID_CENTER_STOP = -0.15
# A neighbor counts as clearly different from the center only past this gap.
_GRID_TURNOVER_GAP = 0.15
_GRID_SHARPE_GAP = 0.15
_GRID_RETURN_GAP = 0.15


def _grid_index(value: float, options: tuple[float, ...] | tuple[int, ...]) -> int:
    """Index of a parameter on a grid axis. A value that is not on the axis is treated as off the grid."""
    for index, item in enumerate(options):
        if abs(float(item) - float(value)) < 1e-9:
            return index
    return -1


def _grid_distance(left: dict[str, Any], right: dict[str, Any]) -> int:
    """Chebyshev distance between two grid points. One step, including a diagonal neighbor, is 1."""
    day_gap = abs(
        _grid_index(left["score_decay_days"], _GRID_DECAY_DAYS)
        - _grid_index(right["score_decay_days"], _GRID_DECAY_DAYS)
    )
    stop_gap = abs(
        _grid_index(left["stop_loss_pct"], _GRID_STOPS)
        - _grid_index(right["stop_loss_pct"], _GRID_STOPS)
    )
    return max(day_gap, stop_gap)


def _is_center(row: dict[str, Any]) -> bool:
    """Whether this cell is the D center: decay 10 days and stop -15%."""
    return (
        int(row["score_decay_days"]) == _GRID_CENTER_DAYS
        and abs(float(row["stop_loss_pct"]) - _GRID_CENTER_STOP) < 1e-9
    )


def _metric_value(row: dict[str, Any], key: str) -> float:
    """Read one metric. A non-finite value is not eligible to win."""
    value = float(row[key])
    return value if math.isfinite(value) else float("nan")


def _best_rows(rows: list[dict[str, Any]], key: str, higher: bool) -> list[dict[str, Any]]:
    """Find the best cells for one metric. Ties are all kept."""
    usable = [row for row in rows if row["lookahead_ok"] and math.isfinite(_metric_value(row, key))]
    if not usable:
        return []
    target = max(_metric_value(row, key) for row in usable) if higher else min(
        _metric_value(row, key) for row in usable
    )
    return [row for row in usable if abs(_metric_value(row, key) - target) < 1e-9]


def _neighbor_gaps(center: dict[str, Any], other: dict[str, Any]) -> list[str]:
    """
    List metrics where a neighbor differs clearly from case D.

    A relative gap above 15% in annualized fills, an absolute Sharpe gap above 0.15,
    or an absolute return gap above 15 percentage points counts as clearly different.
    """
    gaps: list[str] = []
    base_turn = float(center["annualized_sides"])
    if base_turn > 0 and abs(float(other["annualized_sides"]) - base_turn) / base_turn > _GRID_TURNOVER_GAP:
        gaps.append("annualized fills")
    if abs(float(other["tail_sharpe"]) - float(center["tail_sharpe"])) > _GRID_SHARPE_GAP:
        gaps.append("tailwind Sharpe")
    if abs(float(other["head_sharpe"]) - float(center["head_sharpe"])) > _GRID_SHARPE_GAP:
        gaps.append("headwind Sharpe")
    if abs(float(other["head_return"]) - float(center["head_return"])) > _GRID_RETURN_GAP:
        gaps.append("headwind return")
    if abs(float(other["total_return"]) - float(center["total_return"])) > _GRID_RETURN_GAP:
        gaps.append("total return")
    return gaps


def run_local_grid_search(
    tickers: list[str],
    years: int | None = None,
    reuse: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Run a 5 by 5 local grid of decay days and stop-loss around case D.

    score_decay_days is 7, 8, 10, 12, 15.
    stop_loss_pct is -12%, -13%, -15%, -17%, -20%.
    Take-profit stays at 25% and the rank gate stays at the top 15.
    Every cell runs the full event-driven backtest, split by entry-day regime.

    Args:
        tickers: Candidate symbols, usually Group A.
        years: Legacy "years back from today". None keeps the fixed config window. A value overrides it.
        reuse: Finished backtests. A cell whose parameters already match is reused and not recomputed.

    Returns:
        Twenty-five metric rows, ordered by decay days and then by stop from shallow to deep.
    """
    cached = list(reuse or [])
    rows: list[dict[str, Any]] = []
    total = len(_GRID_DECAY_DAYS) * len(_GRID_STOPS)
    done = 0
    for days in _GRID_DECAY_DAYS:
        for stop in _GRID_STOPS:
            done += 1
            case = {
                "label": f"decay {days}/stop {stop * 100:.0f}%",
                "note": "local grid",
                "take_profit_pct": _GRID_TAKE_PROFIT,
                "stop_loss_pct": stop,
                "score_decay_days": days,
                "score_decay_rank": _GRID_DECAY_RANK,
            }
            matched = next((item for item in cached if _same_exit_params(item, case)), None)
            if matched is not None:
                _log.info("Local grid %s/%s reuses a finished backtest: %s", done, total, case["label"])
                result = matched
            else:
                _log.info(
                    "Local grid %s/%s: decay %s days, stop %.0f%%",
                    done,
                    total,
                    days,
                    stop * 100,
                )
                result = run_event_driven_backtest(
                    tickers,
                    years=years,
                    take_profit_pct=_GRID_TAKE_PROFIT,
                    stop_loss_pct=float(stop),
                    score_decay_days=int(days),
                    score_decay_rank=_GRID_DECAY_RANK,
                )
            row = _sensitivity_row(case, result)
            if not row["lookahead_ok"]:
                _log.warning("Local grid %s failed the trade-chain lookahead check", case["label"])
            rows.append(row)
    return rows


def _format_grid_cell(row: dict[str, Any] | None, key: str, best: bool) -> str:
    """Format one cell to a fixed width. The best cell gets a leading asterisk."""
    if row is None or not row["lookahead_ok"] or not math.isfinite(_metric_value(row, key)):
        text = "—"
    elif key == "annualized_sides":
        text = f"{row[key]:.1f}"
    elif key in {"tail_sharpe", "head_sharpe"}:
        text = _num(row[key])
    else:
        text = _pct(row[key])
    if best:
        text = "*" + text
    return f"{text:>8}"


def _print_grid_matrix(rows: list[dict[str, Any]], key: str, title: str, higher: bool) -> None:
    """Print one metric as a 5 by 5 grid. Rows are decay days and columns are stops."""
    winners = _best_rows(rows, key, higher)
    winner_keys = {
        (int(item["score_decay_days"]), round(float(item["stop_loss_pct"]), 2))
        for item in winners
    }
    direction = "higher is better" if higher else "lower is better"
    print(f"\n{title} ({direction}; * is best)", flush=True)
    axis_label = "decay\\stop"
    header = f"{axis_label:<12}" + "".join(f"{stop * 100:.0f}%".rjust(8) for stop in _GRID_STOPS)
    print(header, flush=True)
    by_key = {
        (int(row["score_decay_days"]), round(float(row["stop_loss_pct"]), 2)): row
        for row in rows
    }
    for days in _GRID_DECAY_DAYS:
        cells = []
        for stop in _GRID_STOPS:
            point = (days, round(float(stop), 2))
            cells.append(_format_grid_cell(by_key.get(point), key, point in winner_keys))
        mark = " D" if days == _GRID_CENTER_DAYS else "  "
        print(f"{days:>2}d{mark:<6}" + "".join(cells), flush=True)


def _grid_conclusion(rows: list[dict[str, Any]]) -> str:
    """
    Decide whether case D sits in a shared best region, or is a single cell whose neighbors differ a lot.

    Args:
        rows: The 25 cell results.

    Returns:
        One conclusion. When the best cells scatter and neighbors differ clearly, it says robustness is limited.
    """
    center_rows = [row for row in rows if _is_center(row) and row["lookahead_ok"]]
    if not center_rows:
        return "Case D (decay 10 days, stop -15%) has no usable result, so it cannot be judged as a local optimum."
    center = center_rows[0]
    metrics = (
        ("annualized_sides", "low turnover", False),
        ("tail_sharpe", "tailwind Sharpe", True),
        ("head_sharpe", "headwind Sharpe", True),
        ("head_return", "headwind return", True),
        ("total_return", "total return", True),
    )
    lines = ["Best cells for the five metrics:"]
    best_points: list[dict[str, Any]] = []
    near_center = 0
    for key, label, higher in metrics:
        winners = _best_rows(rows, key, higher)
        if not winners:
            lines.append(f"  {label}: no valid cell")
            continue
        where = ", ".join(
            f"decay {int(item['score_decay_days'])} days / stop {float(item['stop_loss_pct']) * 100:.0f}%"
            f" ({_num(item[key]) if key.endswith('sharpe') else (_pct(item[key]) if key != 'annualized_sides' else f'{item[key]:.1f}')})"
            for item in winners
        )
        lines.append(f"  {label}: {where}")
        best_points.extend(winners)
        if any(_grid_distance(center, item) <= 1 for item in winners):
            near_center += 1

    scattered = False
    if len(best_points) >= 2:
        farthest = max(_grid_distance(left, right) for left in best_points for right in best_points)
        scattered = farthest >= 3
        lines.append(
            f"The farthest distance among these best cells is {farthest} steps "
            f"(0 is the same cell, 1 is a neighbor, 4 is a corner)."
        )
    lines.append(f"{near_center} metrics have their best cell on D or a neighbor of D.")

    neighbors = [
        row for row in rows
        if row["lookahead_ok"] and not _is_center(row) and _grid_distance(center, row) == 1
    ]
    different = []
    for row in neighbors:
        gaps = _neighbor_gaps(center, row)
        if len(gaps) >= 2:
            different.append(
                f"decay {int(row['score_decay_days'])} days / stop {float(row['stop_loss_pct']) * 100:.0f}%"
                f" ({', '.join(gaps)})"
            )
    if different:
        lines.append("Neighbors that differ clearly from D: " + "; ".join(different) + ".")
    else:
        lines.append("None of the eight neighbors differs from D by the set gap on two or more metrics at once.")

    # Neighbors differ clearly, or the best cells sit at opposite ends and D is not the high ground for most metrics.
    fragile = len(different) >= 3 or (scattered and near_center < 3)
    if fragile:
        lines.append("The result is sensitive to the parameter choice, so this conclusion has limited robustness.")
    elif near_center >= 3:
        lines.append(
            "D sits next to the best cell on several metrics, which supports it as a local optimum on this small grid rather than a one-cell coincidence."
        )
    else:
        lines.append("The best cells do not cluster around D. D is not the high ground for most metrics on this grid.")
    return "\n".join(lines)


def _print_local_grid(rows: list[dict[str, Any]]) -> None:
    """Print the 5 by 5 grid, the best cell for each metric, and whether D is only a one-cell coincidence."""
    print("\n=== Local grid around D (take-profit 25%, top 15, after costs) ===", flush=True)
    print(
        "Rows are signal-decay days and columns are stops. D is the row for 10 decay days and the column for a -15% stop. "
        "An asterisk marks the best cell for that metric.",
        flush=True,
    )
    _print_grid_matrix(rows, "annualized_sides", "Annualized one-way fills", higher=False)
    _print_grid_matrix(rows, "tail_sharpe", "Tailwind Sharpe", higher=True)
    _print_grid_matrix(rows, "head_sharpe", "Headwind Sharpe", higher=True)
    _print_grid_matrix(rows, "head_return", "Headwind return", higher=True)
    _print_grid_matrix(rows, "total_return", "Total return", higher=True)
    print("\nDo the best cells cluster:", flush=True)
    print(_grid_conclusion(rows), flush=True)


def _cache_span(symbol: str) -> tuple[date, date] | None:
    """Read the first and last dates in a local parquet. Returns None when the file is missing or unreadable."""
    path = _OHLCV_DIR / f"{_yahoo_symbol(symbol)}.parquet"
    if not path.is_file():
        return None
    try:
        frame = pd.read_parquet(path, columns=["Close"])
    except Exception as exc:  # noqa: BLE001
        _log.warning("Could not read the %s price cache; treating it as a re-download for the estimate: %s", symbol, exc)
        return None
    if frame.empty:
        return None
    index = _as_timestamp_index(frame.index)
    return index.min().date(), index.max().date()


def estimate_ten_year_fetch(
    start: date = _TEN_YEAR_START,
    end: date | None = None,
) -> dict[str, Any]:
    """
    Estimate the download size for the S&P 500 union from 2015, without fetching prices.

    Membership comes from the local history file's quarterly snapshots and the event union.
    Prices are checked only against parquet files that already exist. Missing files do not hit Yahoo.

    Args:
        start: First performance date.
        end: Last performance date. None means today.

    Returns:
        Union count, already-covered count, quarterly size range, and the download and backtest windows.
    """
    end_day = end or date.today()
    fetch_start = start - timedelta(days=_WARMUP_DAYS)
    union = get_union_universe_sp500(start, end_day)
    snapshots = get_point_in_time_sp500(start, end_day, freq="quarterly")
    sizes = [len(names) for names in snapshots.values()]
    covered: list[str] = []
    partial: list[str] = []
    for ticker in sorted(union):
        span = _cache_span(ticker)
        if span is None:
            continue
        if span[0] <= fetch_start and span[1] >= end_day - timedelta(days=7):
            covered.append(ticker)
        else:
            partial.append(ticker)
    missing = sorted(set(union) - set(covered))
    dotted = sorted(ticker for ticker in union if "." in ticker)
    # Time one cached score, then scale by the 10-year row count and the number of names.
    score_sec, rows_cached = _time_one_score(covered)
    rows_10y = 252 * ((end_day - fetch_start).days / 365.25)
    if score_sec and rows_cached:
        score_each = score_sec * max(rows_10y / rows_cached, 1.0)
    else:
        score_each = 0.15
    n_missing = len(missing)
    download_low = n_missing * _FETCH_SEC_PER_NEW_SYMBOL
    download_high = n_missing * _FETCH_SEC_PER_NEW_SYMBOL_SLOW
    # The lookahead check truncates and rescores every name in the union on sampled days, so it grows with the name count.
    verify_sec = max(len(union), 1) * 8 * score_each
    backtest_sec = len(union) * score_each + verify_sec
    return {
        "start": start,
        "end": end_day,
        "fetch_start": fetch_start,
        "n_quarters": len(snapshots),
        "quarter_min": min(sizes) if sizes else 0,
        "quarter_max": max(sizes) if sizes else 0,
        "quarter_median": int(sorted(sizes)[len(sizes) // 2]) if sizes else 0,
        "n_union": len(union),
        "n_covered": len(covered),
        "n_partial": len(partial),
        "n_missing": n_missing,
        "n_dotted": len(dotted),
        "download_low_sec": download_low,
        "download_high_sec": download_high,
        "backtest_sec": backtest_sec,
        "score_each_sec": score_each,
    }


def _time_one_score(covered: list[str]) -> tuple[float, int]:
    """Time the technical score on one cached file. Returns (seconds, row count). (0, 0) when there is no cache."""
    import time

    candidates = list(covered)
    if not candidates and _OHLCV_DIR.is_dir():
        candidates = [path.stem for path in _OHLCV_DIR.glob("*.parquet")]
    if not candidates:
        return 0.0, 0
    path = _OHLCV_DIR / f"{_yahoo_symbol(candidates[0])}.parquet"
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        _log.warning("Could not time a score from cache: %s", exc)
        return 0.0, 0
    history = _prepare_history(frame)
    started = time.perf_counter()
    compute_tech_score_series(history)
    return max(time.perf_counter() - started, 0.01), len(history)


def _minutes(seconds: float) -> str:
    """Turn a number of seconds into an approximate minute phrase."""
    minutes = seconds / 60.0
    if minutes < 1:
        return f"{seconds:.0f} sec"
    return f"{minutes:.0f} min"


def _print_fetch_estimate(estimate: dict[str, Any]) -> None:
    """Print the size and time of a 10-year price fetch, without starting the download."""
    print("\n=== 10-year point-in-time S&P 500 universe: estimate before download ===", flush=True)
    print(
        f"Performance window: {estimate['start']} ~ {estimate['end']}. "
        f"Prices are fetched from {estimate['fetch_start']}. The first {_WARMUP_DAYS} days are only moving-average warmup.",
        flush=True,
    )
    print(
        f"{estimate['n_quarters']} quarterly snapshots, "
        f"{estimate['quarter_min']}–{estimate['quarter_max']} names in a quarter "
        f"(median {estimate['quarter_median']}).",
        flush=True,
    )
    print(
        f"Union of names that were in the index at least once since 2015: {estimate['n_union']}. "
        f"Of those, {estimate['n_dotted']} have a dot in the symbol and are rewritten to Yahoo's hyphen on download.",
        flush=True,
    )
    print(
        f"Local cache already covers the full window: {estimate['n_covered']}. "
        f"{estimate['n_partial']} files exist but do not cover the pre-2015 warmup, so they still need a refetch and a merge. "
        f"Still to download: {estimate['n_missing']} (plus SPY, if SPY is also missing).",
        flush=True,
    )
    print(
        "Expected download time: "
        f"{_minutes(estimate['download_low_sec'])}–{_minutes(estimate['download_high_sec'])} "
        "(about 2–4 seconds per name, including a 0.7 second pause; failures retry 3 times, so it can take longer).",
        flush=True,
    )
    print(
        "After the download, the baseline (take-profit 25%, stop -10%, decay 5 days, top 15) "
        f"still needs about {_minutes(estimate['backtest_sec'])} for one 10-year backtest. "
        "The lookahead check grows with the size of the union.",
        flush=True,
    )
    print(
        "The rank is the top 15 among about 500 names in that quarter, not the top 15 of the earlier 34 names. "
        "The same decay_rank=15 is much stricter.",
        flush=True,
    )
    print(
        "Yahoo may not have a full daily history for delisted or renamed companies, so fewer names than the union will actually enter the backtest. "
        "This still uses the constituents of the time, rather than mapping today's 34 names back to 2015.",
        flush=True,
    )


def _print_ten_year_baseline(result: dict[str, Any]) -> None:
    """Print trade count, regime Sharpe, and the number of SPY drawdowns for the 10-year baseline."""
    head = result["regime"]["Headwind"]
    tail = result["regime"]["Tailwind"]
    episodes = result.get("drawdown_episodes") or []
    print("\n=== 10-year baseline (point-in-time S&P 500, after costs) ===", flush=True)
    print(
        f"Window {result['start']} ~ {result['end']}. "
        f"Union {result['n_universe_symbols']} names, prices for {result['n_with_prices']}.",
        flush=True,
    )
    print(f"Total trades: {result['n_entries']} (the 3-year, 34-name run had 186)", flush=True)
    print(
        f"Tailwind Sharpe {_num(float(tail['sharpe']))}, "
        f"headwind Sharpe {_num(float(head['sharpe']))}, "
        f"headwind return {_pct(float(head['strategy_return']))}, "
        f"total return {_pct(float(result['cumulative_return_net']))}.",
        flush=True,
    )
    print(f"SPY drawdowns of more than 8% from a high: {len(episodes)}.", flush=True)
    for episode in episodes:
        print(
            f"  {episode['start']} ~ {episode['end']}, "
            f"peak {episode['peak_date']}, depth {episode['depth'] * 100:.1f}%",
            flush=True,
        )
    if len(episodes) > 4:
        print(
            f"These {len(episodes)} episodes are more than the 4 after 2023-10, so the 10-year window covers earlier drawdown cycles.",
            flush=True,
        )
    else:
        print(
            f"This window has only {len(episodes)} drawdowns above 8%, which is not clearly more than the 4 after 2023.",
            flush=True,
        )


def run_ten_year_baseline(confirm: bool = False) -> dict[str, Any]:
    """
    Run the point-in-time S&P 500 backtest from 2015 through today with the baseline exit parameters.

    By default it only estimates the download size and prints it. It does not fetch prices or run the backtest.
    Pass confirm=True to continue.

    Args:
        confirm: True downloads the union prices and runs the baseline parameters.

    Returns:
        estimate is the size estimate. ran is False when this call did not run the backtest.
    """
    estimate = estimate_ten_year_fetch()
    _print_fetch_estimate(estimate)
    if not confirm:
        print(
            "This call did not download anything and did not run the 10-year backtest. "
            "To continue, call run_ten_year_baseline(confirm=True).",
            flush=True,
        )
        return {"ran": False, "estimate": estimate}

    universe = get_point_in_time_sp500(_TEN_YEAR_START, date.today(), freq="quarterly")
    result = run_event_driven_backtest(
        start=_TEN_YEAR_START,
        end=date.today(),
        as_of_universe=universe,
        take_profit_pct=0.25,
        stop_loss_pct=-0.10,
        score_decay_days=5,
        score_decay_rank=15,
    )
    if not result["lookahead_ok"]:
        print("The 10-year trade-chain lookahead check failed. The performance below is not used.", flush=True)
        return {"ran": True, "estimate": estimate, "backtest": result}
    _print_ten_year_baseline(result)
    return {"ran": True, "estimate": estimate, "backtest": result}


if __name__ == "__main__":
    if "--estimate-10y" in sys.argv:
        run_ten_year_baseline(confirm=False)
        raise SystemExit(0)
    _assert_exit_rules()
    group_a = define_three_groups()["A"]
    print(f"Group A ({len(group_a)} names): {', '.join(group_a)}", flush=True)
    print(
        "Event-driven: each close looks at the top 3, holds at most 5 names, equal weight, and does not rebalance. "
        "Take-profit +25%, stop -10%, and exit after 5 days outside the top 15. "
        f"One-way cost {_ONE_WAY_COST * 100:.2f}%.",
        flush=True,
    )
    monthly = run_group_backtest(group_a)
    monthly_regime = run_regime_split_backtest(tickers=group_a, backtest=monthly)
    event = run_event_driven_backtest(group_a)
    if not event["lookahead_ok"]:
        print("The trade-chain lookahead check failed. The performance below is not treated as a valid result.", flush=True)
        raise SystemExit(1)
    print(
        "Lookahead check passed: the entry score, recomputed on prices truncated at the entry day, matches the full series. "
        "The exit is replayed only on data from the entry day through the exit day, and dropping one day means it would not exit on the original exit day.",
        flush=True,
    )
    _print_report(monthly, monthly_regime, event)
    sensitivity = run_parameter_sensitivity(group_a, baseline=event)
    _print_sensitivity(sensitivity)
    grid = run_local_grid_search(
        group_a,
        reuse=[row["backtest"] for row in sensitivity if "backtest" in row],
    )
    _print_local_grid(grid)
