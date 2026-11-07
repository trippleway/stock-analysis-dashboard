# -*- coding: utf-8 -*-
"""
Technical-score backtest comparison across three stock universes.

Group A: names that overlapped the S&P 500 after 2022.
Group B: Group A plus large foreign names that cannot enter the S&P 500 because of domicile.
          It excludes the recent US risers in sp500_but_excluded.
Group C: the current full 60-name list, including those newer companies.

All three use the same window and the same rule: at each month-end, pick the top 3
by that day's technical score and hold them equal-weight until the next month-end
(about 30 calendar days). The score is known only at the close, and returns start
the next trading day, so the day's already-finished move is not used.

Costs are charged only on shares that actually trade: 0.05% commission and 0.10%
slippage, each one way. A buy is charged once and a sell is charged once. Replacing
all three names costs 0.30% round trip. Names that stay in the book are not charged again.

If Group C is clearly better than A and B, the extra performance comes from newer
companies that were added to the universe after the fact, not from the technical
signal being especially good at picking mature stocks.

Group A is also split into tailwind and headwind. A headwind episode is a stretch
where SPY falls more than 8% from a closing high before that high is recovered.
Other trading days are tailwind. The stored segment names stay Tailwind and Headwind.
The selection rule does not change. Only the after-cost daily returns inside each
segment are compounded, to see whether the full result depends on the tailwind.
That ex-post label starts the day after the high, before the 8% drop is known, so
a live book would not have known it.

The realtime regime looks only at the highest close through that day. A close more
than 8% below that high is labeled Headwind, and the label clears only when the close
makes a new high. The defensive book holds SPY for a month when the month-end
rebalance day is already in realtime Headwind, and does not pick single names.

Run from the project root:
    python src/backtest_compare_groups.py
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

from backtest_data import build_ohlcv_panel  # noqa: E402
from backtest_signals import compute_tech_score_series  # noqa: E402
from backtest_universe_sp500 import (  # noqa: E402
    compare_universes_sp500,
    get_union_universe_sp500,
)
from config import (  # noqa: E402
    BACKTEST_END,
    BACKTEST_FIRST_REBALANCE_MONTH,
    BACKTEST_LAST_MONTH,
    BACKTEST_MONTHS,
    BACKTEST_START,
    GROUP_DEFINITION_END,
    TICKER_LIST,
)

_log = logging.getLogger("backtest_compare_groups")
if not _log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# History window for the group comparison, the same span as the earlier S&P 500 membership check.
_GROUP_START = date(2022, 1, 1)
# MA200 needs about 200 trading days. Extra calendar days are warmup and are not in the performance.
_WARMUP_DAYS = 420
# Names with fewer bars than this are not candidates that month, so early-listing indicators do not distort the score.
_MIN_HISTORY_BARS = 200
_TOP_N = 3
_BENCHMARK = "SPY"
# Risk-free rate is 0. Sharpe compares return volatility and does not assume an interest rate.
_TRADING_DAYS = 252
# Group C must beat both other groups by this much before the gap is called "from newer companies".
_CLEAR_GAP = 0.05
# Commission and slippage are one way. A buy is charged once and a sell is charged once.
# Replacing the whole book: (0.05% + 0.10%) x 2 = 0.30%.
_COMMISSION_ONE_SIDE = 0.0005
_SLIPPAGE_ONE_SIDE = 0.001
_ONE_WAY_COST = _COMMISSION_ONE_SIDE + _SLIPPAGE_ONE_SIDE
_ROUND_TRIP_COST = _ONE_WAY_COST * 2
# When the gap between gross and net cumulative return exceeds this, print the rebalance count.
_COST_GAP_ALERT = 0.20
# A continuous stretch is headwind only after a drop of more than this from a closing high.
_DRAWDOWN_THRESHOLD = 0.08

_GROUP_ORDER = ("A", "B", "C")
_GROUP_TITLES = {
    "A": "A Mature US",
    "B": "B Plus foreign large caps",
    "C": "C Full list",
}


def define_three_groups(
    start_date: date = _GROUP_START,
    end_date: date = GROUP_DEFINITION_END,
) -> dict[str, list[str]]:
    """
    Build the three universes from the point-in-time S&P 500 comparison.

    Group A is the overlap. Group B is the full list minus sp500_but_excluded,
    so the count is "60 minus the newer companies". It includes not_sp500_listed
    and symbols whose country could not be found, so they were not in those 16 names.
    Group C is config.TICKER_LIST.

    Args:
        start_date: First date used to decide "was ever in the S&P 500".
        end_date: Last date. Defaults to config.GROUP_DEFINITION_END, so the list does not move with the run date.

    Returns:
        {"A": [...], "B": [...], "C": [...]}, symbols sorted.
    """
    end = end_date
    current = [str(ticker).upper().strip() for ticker in TICKER_LIST if str(ticker).strip()]
    union = get_union_universe_sp500(start_date, end)
    report = compare_universes_sp500(current, union)

    overlap = [str(ticker) for ticker in report["overlap"]]
    emerging = {str(ticker) for ticker in report["sp500_but_excluded"]}
    group_c = sorted(dict.fromkeys(current))
    group_a = sorted(dict.fromkeys(overlap))
    group_b = sorted(ticker for ticker in group_c if ticker not in emerging)

    if not set(group_a) <= set(group_b) <= set(group_c):
        raise RuntimeError("The three universes are not nested, so the classification cannot be compared")
    if not group_a or not group_b or not group_c:
        raise RuntimeError("One of the three universes is empty, so the backtest cannot run")

    extra = sorted(set(group_b) - set(group_a))
    _log.info(
        "Three universes: A=%s names, B=%s names (%s more than A), C=%s names (%s more than B)",
        len(group_a),
        len(group_b),
        ",".join(extra) or "none",
        len(group_c),
        ",".join(sorted(set(group_c) - set(group_b))) or "none",
    )
    return {"A": group_a, "B": group_b, "C": group_c}


def _as_timestamp_index(index: pd.Index) -> pd.DatetimeIndex:
    """Normalize an index to timezone-naive midnight dates."""
    stamps = pd.to_datetime(index)
    if getattr(stamps, "tz", None) is not None:
        stamps = stamps.tz_localize(None)
    return pd.DatetimeIndex(stamps.normalize())


def _score_frame(
    panel: dict[str, pd.DataFrame],
    master: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Score every name in the panel and align the result to the master calendar.

    Args:
        panel: Result of build_ohlcv_panel, without the benchmark.
        master: Trading days used for performance, usually SPY's dates.

    Returns:
        (closes, technical scores, whether the minimum history is met). Missing data is NaN / False.
    """
    closes: dict[str, pd.Series] = {}
    scores: dict[str, pd.Series] = {}
    ready: dict[str, pd.Series] = {}
    for ticker, frame in panel.items():
        history = frame.copy()
        history.index = _as_timestamp_index(history.index)
        history = history[~history.index.duplicated(keep="last")].sort_index()
        try:
            score = compute_tech_score_series(history)
        except Exception as exc:  # noqa: BLE001
            _log.warning("%s technical score failed, so this name is not a candidate: %s", ticker, exc)
            continue
        if score.empty:
            _log.warning("%s technical score is empty, so this name is not a candidate", ticker)
            continue
        score.index = _as_timestamp_index(score.index)
        close = pd.to_numeric(history["Close"], errors="coerce")
        enough = close.rolling(_MIN_HISTORY_BARS, min_periods=_MIN_HISTORY_BARS).count()
        closes[ticker] = close.reindex(master)
        scores[ticker] = score.reindex(master)
        ready[ticker] = enough.reindex(master).ge(_MIN_HISTORY_BARS).fillna(False)
    if not closes:
        return (
            pd.DataFrame(index=master),
            pd.DataFrame(index=master),
            pd.DataFrame(index=master),
        )
    return (
        pd.DataFrame(closes, index=master),
        pd.DataFrame(scores, index=master),
        pd.DataFrame(ready, index=master),
    )


def _month_end_signals(index: pd.DatetimeIndex, start: date, end: date) -> list[pd.Timestamp]:
    """
    Take the last trading day of each month inside the performance window as a selection day.

    The final trading day itself is not a signal day: the pick is known only at that close, and there is no next day to hold.

    Args:
        index: Master calendar.
        start: First performance date, inclusive.
        end: Last performance date, inclusive.

    Returns:
        Signal days from earliest to latest.
    """
    window = index[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))]
    if len(window) < 2:
        return []
    month_ends = pd.Series(window, index=window).groupby(window.to_period("M")).max()
    last_day = window[-1]
    return [pd.Timestamp(stamp) for stamp in month_ends.tolist() if pd.Timestamp(stamp) < last_day]


def _target_weights(picked: list[str]) -> pd.Series:
    """Equal-weight target book. An empty list means the month is held in cash."""
    if not picked:
        return pd.Series(dtype=float)
    weight = 1.0 / len(picked)
    return pd.Series(weight, index=picked, dtype=float)


def _schedule_trade_costs(
    books: list[dict[str, Any]],
) -> tuple[dict[pd.Timestamp, float], dict[str, float | int]]:
    """
    Compute each rebalance cost from the previous and next target weights.

    Cost = (0.05% commission + 0.1% slippage) x (buy weight + sell weight).
    Replacing all three names buys 100% and sells 100%, for 0.30% total.
    The first month is built from cash, so only the buy is charged, at 0.15%. Weights that do not change are not charged.

    Args:
        books: Each period's first_hold_day and weights.

    Returns:
        (cost rate charged on each first holding day, rebalance statistics).
    """
    previous = pd.Series(dtype=float)
    cost_on_day: dict[pd.Timestamp, float] = {}
    n_buys = 0
    n_sells = 0
    one_way_turnover = 0.0
    for book in books:
        new = book["weights"]
        names = previous.index.union(new.index)
        old_w = previous.reindex(names).fillna(0.0)
        new_w = new.reindex(names).fillna(0.0)
        delta = new_w - old_w
        traded = float(delta.abs().sum())
        day = pd.Timestamp(book["first_hold_day"])
        cost_on_day[day] = cost_on_day.get(day, 0.0) + _ONE_WAY_COST * traded
        n_buys += int((old_w.eq(0.0) & new_w.gt(0.0)).sum())
        n_sells += int((old_w.gt(0.0) & new_w.eq(0.0)).sum())
        one_way_turnover += traded / 2.0
        previous = new_w[new_w.gt(0.0)]

    n_books = len(books)
    stats: dict[str, float | int] = {
        "n_rebalances": n_books,
        "n_buys": n_buys,
        "n_sells": n_sells,
        "n_name_changes": n_buys + n_sells,
        "avg_one_way_turnover": one_way_turnover / n_books if n_books else 0.0,
        "total_cost_rate": float(sum(cost_on_day.values())),
    }
    return cost_on_day, stats


def _portfolio_returns(
    close: pd.DataFrame,
    score: pd.DataFrame,
    ready: pd.DataFrame,
    signals: list[pd.Timestamp],
    defend_regime: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series, dict[str, float | int]]:
    """
    Hold the month-end top 3 and return gross daily returns plus the cost charged on rebalance days.

    The score on signal day t uses only the close through t. The holding window is the daily
    returns after t through the next signal day, inclusive: the stretch from the close on t
    to the next rebalance close. Cost is charged on the first day the new book is held:
    pay the cost first, then take that day's move.

    Args:
        close: Aligned closes. The defensive book needs a SPY column.
        score: Aligned technical scores.
        ready: Whether the minimum history is met.
        signals: Selection days.
        defend_regime: Realtime labels, stored as Tailwind or Headwind. A rebalance day labeled Headwind holds SPY for that month.

    Returns:
        (gross daily returns, same-day cost rates, rebalance statistics).
    """
    daily_ret = close.pct_change()
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    books: list[dict[str, Any]] = []
    n_headwind_signals = 0
    headwind_dates: list[str] = []
    for i, signal_day in enumerate(signals):
        if signal_day not in close.index:
            continue
        nxt = signals[i + 1] if i + 1 < len(signals) else close.index[-1]
        hold = close.index[(close.index > signal_day) & (close.index <= nxt)]
        if len(hold) == 0:
            continue
        # The realtime label is known only at the rebalance close. This month holds SPY and does not pick single names.
        regime_label = None
        if defend_regime is not None and signal_day in defend_regime.index:
            regime_label = str(defend_regime.loc[signal_day])
        if regime_label == "Headwind":
            n_headwind_signals += 1
            headwind_dates.append(signal_day.date().isoformat())
        if regime_label == "Headwind" and _BENCHMARK in close.columns:
            if pd.notna(close.at[signal_day, _BENCHMARK]):
                weights.loc[hold, _BENCHMARK] = 1.0
                _log.info(
                    "%s is realtime Headwind, so this month holds SPY through %s",
                    signal_day.date().isoformat(),
                    pd.Timestamp(hold[-1]).date().isoformat(),
                )
                cost_day = hold[0] if not books else signal_day
                books.append(
                    {
                        "first_hold_day": cost_day,
                        "weights": _target_weights([_BENCHMARK]),
                        "defended": True,
                        "signal": signal_day.date().isoformat(),
                        "hold_start": pd.Timestamp(hold[0]).date().isoformat(),
                        "hold_end": pd.Timestamp(hold[-1]).date().isoformat(),
                        "picked": [_BENCHMARK],
                    }
                )
                continue
            _log.warning("%s is realtime Headwind, but there is no SPY close, so this month stays in cash", signal_day.date())

        tradable = (
            score.loc[signal_day].notna()
            & close.loc[signal_day].reindex(score.columns).notna()
            & ready.loc[signal_day].fillna(False).astype(bool)
        )
        ranked = score.loc[signal_day].loc[tradable].dropna()
        if ranked.empty:
            _log.warning("%s has no eligible candidate, so this month stays in cash", signal_day.date())
            picked: list[str] = []
        else:
            order = ranked.reset_index()
            order.columns = ["ticker", "score"]
            order = order.sort_values(["score", "ticker"], ascending=[False, True])
            picked = order["ticker"].head(_TOP_N).tolist()
            weights.loc[hold, picked] = 1.0 / len(picked)
            _log.info(
                "%s picked %s, held through %s",
                signal_day.date().isoformat(),
                ",".join(picked),
                pd.Timestamp(hold[-1]).date().isoformat(),
            )
        # The first book is bought from cash, so the cost lands on the first holding day.
        # Later rebalances trade at the signal close, so the cost lands on that close while the old book is still held.
        cost_day = hold[0] if not books else signal_day
        books.append(
            {
                "first_hold_day": cost_day,
                "weights": _target_weights(picked),
                "defended": False,
                "signal": signal_day.date().isoformat(),
                "hold_start": pd.Timestamp(hold[0]).date().isoformat(),
                "hold_end": pd.Timestamp(hold[-1]).date().isoformat(),
                "picked": list(picked),
            }
        )

    valid = daily_ret.notna() & weights.gt(0)
    effective = weights.where(valid, 0.0)
    scale = effective.sum(axis=1).replace(0.0, np.nan)
    # Names that did not trade that day give their weight to the names still held, so the book does not become part cash for no reason.
    renormalized = effective.div(scale, axis=0).fillna(0.0)
    portfolio = (renormalized * daily_ret.fillna(0.0)).sum(axis=1)
    invested = effective.sum(axis=1).gt(0)
    gross = portfolio.where(invested, np.nan).dropna()

    cost_on_day, stats = _schedule_trade_costs(books)
    stats["n_defended_months"] = int(sum(1 for book in books if book.get("defended")))
    stats["n_headwind_signals"] = n_headwind_signals
    stats["headwind_signal_dates"] = headwind_dates  # type: ignore[assignment]
    stats["months"] = [  # type: ignore[assignment]
        {
            "signal": book["signal"],
            "hold_start": book["hold_start"],
            "hold_end": book["hold_end"],
            "picked": list(book["picked"]),
            "defended": bool(book.get("defended")),
        }
        for book in books
    ]
    costs = pd.Series(0.0, index=gross.index, dtype=float)
    for day, cost in cost_on_day.items():
        stamp = pd.Timestamp(day)
        if stamp in costs.index:
            costs.loc[stamp] = cost
        elif cost > 0:
            _log.warning("Rebalance day %s has no matching return, so this %.2f%% cost was not charged", stamp.date(), cost * 100)
    return gross, costs, stats


def _net_returns(gross: pd.Series, costs: pd.Series) -> pd.Series:
    """
    Charge the cost before taking that day's return.

    Args:
        gross: Daily returns before costs.
        costs: Net-asset fraction to subtract on the same day.

    Returns:
        Daily returns after costs.
    """
    aligned = costs.reindex(gross.index).fillna(0.0)
    return (1.0 - aligned) * (1.0 + gross) - 1.0


def _max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown of an equity curve. Zero when there is no drawdown."""
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def _sharpe(daily_returns: pd.Series) -> float:
    """Annualized Sharpe. The risk-free rate is 0. Returns NaN when volatility is 0."""
    if len(daily_returns) < 2:
        return float("nan")
    sigma = float(daily_returns.std(ddof=1))
    if not math.isfinite(sigma) or sigma <= 0:
        return float("nan")
    return float(daily_returns.mean() / sigma * math.sqrt(_TRADING_DAYS))


def _cumulative(daily_returns: pd.Series) -> float:
    """Cumulative return from compounding daily returns."""
    if daily_returns.empty:
        return float("nan")
    equity = (1.0 + daily_returns).cumprod()
    return float(equity.iloc[-1] - 1.0)


def run_group_backtest(
    tickers: list[str],
    years: int | None = None,
    defend_regime: pd.Series | None = None,
    start_date: date = BACKTEST_START,
    end_date: date = BACKTEST_END,
) -> dict[str, Any]:
    """
    Run the month-end top-3 technical backtest, held to the next month-end, on one universe.

    Prices come from backtest_data.build_ohlcv_panel. Scores use
    compute_tech_score_series, so day t does not include prices after t.
    Extra history is fetched before the performance start so MA200 is already valid
    on the first rebalance day. Warmup is not in the cumulative return, and SPY is
    measured only on the same holding days.
    The same run reports cumulative return before and after costs. Cost lands only where weights change.

    Args:
        tickers: Candidate symbols for this group.
        years: Legacy "years back from today". None uses the fixed start_date to end_date window.
            A value moves with the run date and cannot be compared with resume numbers.
        defend_regime: Realtime labels, stored as Tailwind or Headwind. When passed, a month whose rebalance day is Headwind holds SPY.
        start_date: First performance date, inclusive. Defaults to config.BACKTEST_START.
        end_date: Last performance date, inclusive. Defaults to config.BACKTEST_END.

    Returns:
        A performance dict with cumulative_return (before costs), cumulative_return_net (after costs),
        cost_drag, rebalance stats, excess_vs_spy, sharpe, and max_drawdown.
        Also net_returns, spy_returns, and spy_close, so the regime split can reuse the same path.
        The stored regime names stay Tailwind and Headwind.
    """
    if years is not None:
        if years < 1:
            raise ValueError(f"years must be at least 1, got {years}")
        end = date.today()
        eval_start = end - timedelta(days=365 * years)
    else:
        end = end_date
        eval_start = start_date
    if eval_start >= end:
        raise ValueError(f"Backtest start {eval_start} must be before end {end}")
    fetch_start = eval_start - timedelta(days=_WARMUP_DAYS)
    symbols = [str(ticker).upper().strip() for ticker in tickers if str(ticker).strip()]

    panel = build_ohlcv_panel([*symbols, _BENCHMARK], fetch_start, end)
    spy = panel.get(_BENCHMARK)
    if spy is None or spy.empty:
        raise RuntimeError("No SPY prices, so excess return cannot be computed")

    master = _as_timestamp_index(spy.index)
    master = master[(master >= pd.Timestamp(fetch_start)) & (master <= pd.Timestamp(end))]
    stock_panel = {ticker: frame for ticker, frame in panel.items() if ticker != _BENCHMARK}
    dropped = sorted(set(symbols) - set(stock_panel))
    if dropped:
        _log.warning("These symbols have no prices and were dropped from this group: %s", ",".join(dropped))
    if not stock_panel:
        raise RuntimeError("This group has no usable prices, so the backtest cannot run")

    close, score, ready = _score_frame(stock_panel, master)
    spy_close = pd.to_numeric(spy["Close"], errors="coerce")
    spy_close.index = _as_timestamp_index(spy_close.index)
    spy_close = spy_close.reindex(master)

    signals = _month_end_signals(master, eval_start, end)
    if not signals:
        raise RuntimeError(f"No month-end rebalance day between {eval_start} and {end}")

    if defend_regime is not None:
        close = close.copy()
        close[_BENCHMARK] = spy_close.reindex(close.index)
    strategy_ret, trade_costs, trade_stats = _portfolio_returns(
        close,
        score,
        ready,
        signals,
        defend_regime=defend_regime,
    )
    if strategy_ret.empty or int(trade_stats["n_rebalances"]) == 0:
        raise RuntimeError("No month picked any stock, so performance cannot be computed")

    net_ret = _net_returns(strategy_ret, trade_costs)
    spy_ret = spy_close.pct_change().reindex(strategy_ret.index)
    if spy_ret.isna().all():
        raise RuntimeError("SPY has no return over the holding window")
    spy_ret = spy_ret.fillna(0.0)

    cumulative = _cumulative(strategy_ret)
    cumulative_net = _cumulative(net_ret)
    spy_cumulative = _cumulative(spy_ret)
    equity = (1.0 + strategy_ret).cumprod()
    cost_drag = cumulative - cumulative_net
    result = {
        "tickers": symbols,
        "n_tickers": len(symbols),
        "n_with_prices": len(stock_panel),
        "dropped_tickers": dropped,
        "start": strategy_ret.index[0].date().isoformat(),
        "end": strategy_ret.index[-1].date().isoformat(),
        "n_rebalances": int(trade_stats["n_rebalances"]),
        "n_buys": int(trade_stats["n_buys"]),
        "n_sells": int(trade_stats["n_sells"]),
        "n_name_changes": int(trade_stats["n_name_changes"]),
        "n_defended_months": int(trade_stats.get("n_defended_months", 0)),
        "n_headwind_signals": int(trade_stats.get("n_headwind_signals", 0)),
        "headwind_signal_dates": list(trade_stats.get("headwind_signal_dates", [])),
        "months": list(trade_stats.get("months", [])),
        "gross_returns": strategy_ret,
        "avg_one_way_turnover": float(trade_stats["avg_one_way_turnover"]),
        "total_cost_rate": float(trade_stats["total_cost_rate"]),
        "cumulative_return": cumulative,
        "cumulative_return_net": cumulative_net,
        "cost_drag": cost_drag,
        "spy_cumulative_return": spy_cumulative,
        "excess_vs_spy": cumulative - spy_cumulative,
        "excess_vs_spy_net": cumulative_net - spy_cumulative,
        "sharpe": _sharpe(strategy_ret),
        "sharpe_net": _sharpe(net_ret),
        "max_drawdown": _max_drawdown(equity),
        "max_drawdown_net": _max_drawdown((1.0 + net_ret).cumprod()),
        # Used by the regime split. spy_close includes warmup so a drawdown that started before the backtest can be recognized.
        "net_returns": net_ret,
        "spy_returns": spy_ret,
        "spy_close": spy_close.dropna(),
    }
    _log.info(
        "Backtest finished %s ~ %s: %.2f%% before costs, %.2f%% after costs, gap %.2f percentage points",
        result["start"],
        result["end"],
        cumulative * 100,
        cumulative_net * 100,
        cost_drag * 100,
    )
    return result


def realtime_regime(
    spy_close: pd.Series,
    threshold: float = _DRAWDOWN_THRESHOLD,
) -> pd.Series:
    """
    Label each day Tailwind or Headwind using only data through that day.

    The running high is the highest close through today, inclusive. Once the close
    is more than `threshold` below that high, the day is labeled Headwind. A later
    bounce that has not made a new high stays Headwind. A close at or above the prior
    high clears the label and the day is Tailwind again.

    This is not the ex-post drawdown window. The ex-post version labels Headwind from
    the day after the high, even when the drop has not yet reached 8%. Here the
    label waits until the drop has already happened.

    Args:
        spy_close: SPY closes.
        threshold: Drawdown gate. 0.08 means 8%.

    Returns:
        A Series aligned to the valid closes. Values are Tailwind or Headwind.
    """
    prices = pd.to_numeric(spy_close, errors="coerce").dropna()
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    if prices.empty:
        return pd.Series(dtype=object, name="regime")

    labels: list[str] = []
    peak = float(prices.iloc[0])
    headwind = False
    for px in prices.astype(float):
        price = float(px)
        if price >= peak:
            peak = price
            headwind = False
        elif price <= peak * (1.0 - threshold):
            headwind = True
        labels.append("Headwind" if headwind else "Tailwind")
    return pd.Series(labels, index=prices.index, name="regime")


def verify_realtime_regime(
    spy_close: pd.Series,
    threshold: float = _DRAWDOWN_THRESHOLD,
) -> list[str]:
    """
    Truncation check: the label on day t from data through t must match the full series.

    Args:
        spy_close: SPY closes.
        threshold: Same drawdown gate as realtime_regime.

    Returns:
        Date strings that disagree. An empty list means the check passed. Disagreements are printed.
    """
    prices = pd.to_numeric(spy_close, errors="coerce").dropna()
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    if prices.empty:
        print("Realtime regime truncation check failed: no SPY closes", flush=True)
        return ["<empty>"]
    full = realtime_regime(prices, threshold)
    mismatches: list[str] = []
    for pos in range(len(prices)):
        truncated = realtime_regime(prices.iloc[: pos + 1], threshold)
        if truncated.empty or str(truncated.iloc[-1]) != str(full.iloc[pos]):
            mismatches.append(pd.Timestamp(prices.index[pos]).date().isoformat())
    if mismatches:
        print(f"Realtime regime truncation check disagreed on {len(mismatches)} days:", flush=True)
        for label in mismatches:
            print(f"  {label}", flush=True)
    else:
        print(f"Realtime regime truncation check passed: {len(prices)} trading days", flush=True)
    return mismatches


def _spy_drawdown_episodes(
    close: pd.Series,
    threshold: float = _DRAWDOWN_THRESHOLD,
) -> list[dict[str, Any]]:
    """
    Find continuous stretches where the close falls more than the threshold from a high.

    A peak is a closing high. Before the next close that retakes that high, if the
    lowest point is more than `threshold` below the peak, the whole stretch is one
    drawdown: from the trading day after the peak through the recovery day, inclusive.
    If the series ends before recovery, the end is the last trading day.

    Args:
        close: SPY closes, indexed by trading day.
        threshold: Drawdown gate. 0.08 means 8%.

    Returns:
        One dict per episode, with peak_date, peak, trough_date, trough,
        depth (negative), start, end, and recovered.
    """
    prices = pd.to_numeric(close, errors="coerce").dropna()
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    if len(prices) < 2:
        return []

    episodes: list[dict[str, Any]] = []
    peak_pos = 0
    peak_px = float(prices.iloc[0])

    def _record(peak_at: int, end_at: int, recovered: bool, peak_price: float) -> None:
        """Record a drawdown from the peak through end_at when the drop breaks the threshold."""
        if end_at <= peak_at:
            return
        segment = prices.iloc[peak_at : end_at + 1]
        trough_px = float(segment.min())
        depth = trough_px / peak_price - 1.0
        if depth > -threshold:
            return
        trough_at = peak_at + int(np.argmin(segment.to_numpy()))
        episodes.append(
            {
                "peak_date": prices.index[peak_at].date(),
                "peak": peak_price,
                "trough_date": prices.index[trough_at].date(),
                "trough": trough_px,
                "depth": depth,
                "start": prices.index[peak_at + 1].date(),
                "end": prices.index[end_at].date(),
                "recovered": recovered,
            }
        )

    for i in range(1, len(prices)):
        px = float(prices.iloc[i])
        if px >= peak_px:
            _record(peak_pos, i, True, peak_px)
            peak_pos = i
            peak_px = px
    if peak_pos < len(prices) - 1:
        _record(peak_pos, len(prices) - 1, False, peak_px)
    return episodes


def _segment_performance(
    name: str,
    period: str,
    strategy: pd.Series,
    spy: pd.Series,
) -> dict[str, Any]:
    """
    Turn strategy and SPY returns on the same trading days into one comparison row.

    Args:
        name: Segment name. The stored names are Tailwind and Headwind.
        period: Period text printed in the table.
        strategy: Strategy daily returns for the segment.
        spy: SPY daily returns on the same dates.

    Returns:
        A dict with strategy return, SPY return, excess, and Sharpe.
    """
    aligned_spy = spy.reindex(strategy.index).fillna(0.0)
    strategy_return = _cumulative(strategy)
    spy_return = _cumulative(aligned_spy)
    return {
        "name": name,
        "period": period,
        "n_days": int(len(strategy)),
        "strategy_return": strategy_return,
        "spy_return": spy_return,
        "excess": strategy_return - spy_return,
        "sharpe": _sharpe(strategy),
    }


def run_regime_split_backtest(
    years: int | None = None,
    tickers: list[str] | None = None,
    backtest: dict[str, Any] | None = None,
    start_date: date = BACKTEST_START,
    end_date: date = BACKTEST_END,
) -> dict[str, Any]:
    """
    Split Group A's after-cost daily returns into Tailwind and Headwind.

    Headwind is a continuous stretch where SPY falls more than 8% from a high
    (see _spy_drawdown_episodes). The peak may sit before the backtest start,
    so a decline already underway at the start is not missed. Performance counts
    only trading days that overlap Group A's holding days. Tailwind is every other
    holding day.

    Selection is not rerun. Signals are still the month-end top 3 by that day's
    technical score. The segment numbers only compound daily returns from the same
    equity path, so the two compounded pieces multiply back to the full after-cost return.
    This is an ex-post check, not a filter that could have been used live.

    Args:
        years: Legacy "years back from today". None uses start_date to end_date. A passed backtest is not rerun.
        tickers: Group A symbols. When omitted, the three universes are rebuilt and Group A is taken.
        backtest: A finished run_group_backtest result. It must include net_returns and spy_close.
        start_date: First performance date, inclusive. Defaults to config.BACKTEST_START.
        end_date: Last performance date, inclusive. Defaults to config.BACKTEST_END.

    Returns:
        episodes is the drawdown list. segments is the Tailwind and Headwind performance rows.
        full_net is the full-window after-cost cumulative return.
    """
    if backtest is None:
        if tickers is None:
            tickers = define_three_groups()["A"]
        backtest = run_group_backtest(
            tickers,
            years=years,
            start_date=start_date,
            end_date=end_date,
        )

    net = backtest.get("net_returns")
    spy_ret = backtest.get("spy_returns")
    spy_close = backtest.get("spy_close")
    if not isinstance(net, pd.Series) or net.empty:
        raise RuntimeError("The backtest has no after-cost daily returns, so it cannot be split into Tailwind and Headwind")
    if not isinstance(spy_ret, pd.Series) or not isinstance(spy_close, pd.Series):
        raise RuntimeError("The backtest has no SPY prices, so drawdowns cannot be detected")

    episodes = _spy_drawdown_episodes(spy_close, _DRAWDOWN_THRESHOLD)
    window_start = pd.Timestamp(net.index.min())
    window_end = pd.Timestamp(net.index.max())
    visible: list[dict[str, Any]] = []
    for episode in episodes:
        start = pd.Timestamp(episode["start"])
        end = pd.Timestamp(episode["end"])
        if end < window_start or start > window_end:
            continue
        visible.append(episode)
        _log.info(
            "SPY drawdown %s ~ %s, peak %s %.2f, max drop %.1f%%, %s",
            episode["start"],
            episode["end"],
            episode["peak_date"],
            episode["peak"],
            episode["depth"] * 100,
            "recovered" if episode["recovered"] else "not yet recovered",
        )

    headwind = pd.Series(False, index=net.index)
    period_parts: list[str] = []
    for episode in visible:
        start = pd.Timestamp(episode["start"])
        end = pd.Timestamp(episode["end"])
        headwind |= (net.index >= start) & (net.index <= end)
        overlap_start = max(start, window_start)
        overlap_end = min(end, window_end)
        period_parts.append(
            f"{overlap_start.date()} ~ {overlap_end.date()}"
            f" (peak {episode['peak_date']}, drop {episode['depth'] * 100:.1f}%)"
        )

    head_ret = net.loc[headwind]
    tail_ret = net.loc[~headwind]
    head_spy = spy_ret.reindex(head_ret.index)
    tail_spy = spy_ret.reindex(tail_ret.index)

    # The two segments must partition every holding day, so their product returns to the full net equity.
    combined = float((1.0 + head_ret).prod() * (1.0 + tail_ret).prod() - 1.0)
    full_net = float(backtest["cumulative_return_net"])
    if math.isfinite(full_net) and not math.isclose(combined, full_net, rel_tol=0, abs_tol=1e-8):
        _log.warning(
            "Compounded Tailwind times Headwind is %.4f%%, which does not match the full after-cost %.4f%%",
            combined * 100,
            full_net * 100,
        )

    if visible:
        head_period = "; ".join(period_parts)
    else:
        head_period = "No continuous stretch fell more than 8% from a high in this window"
    tail_period = f"{window_start.date()} ~ {window_end.date()}, excluding drawdown days"

    segments = [
        _segment_performance("Tailwind", tail_period, tail_ret, tail_spy),
        _segment_performance("Headwind", head_period, head_ret, head_spy),
    ]
    return {
        "episodes": visible,
        "segments": segments,
        "full_net": full_net,
        "window": f"{window_start.date()} ~ {window_end.date()}",
    }


def _interpret_regime(regime: dict[str, Any]) -> str:
    """
    Decide whether Headwind lagged SPY, or whether its Sharpe turned negative.

    Args:
        regime: Return value of run_regime_split_backtest.

    Returns:
        A conclusion that can be printed as-is.
    """
    by_name = {row["name"]: row for row in regime["segments"]}
    head = by_name["Headwind"]
    tail = by_name["Tailwind"]
    full_net = float(regime["full_net"])
    if not regime["episodes"] or head["n_days"] == 0:
        return (
            f"Group A full-window after-cost cumulative return is {_pct(full_net)}. "
            "No continuous stretch where SPY fell more than 8% from a high was found, "
            "so the headwind environment cannot be tested separately."
        )

    loses = math.isfinite(head["excess"]) and head["excess"] < 0
    sharpe_negative = math.isfinite(head["sharpe"]) and head["sharpe"] < 0
    if loses or sharpe_negative:
        reasons: list[str] = []
        if loses:
            reasons.append(
                f"Headwind strategy return {_pct(head['strategy_return'])}, "
                f"SPY over the same days {_pct(head['spy_return'])}, excess {_pct(head['excess'])}"
            )
        if sharpe_negative:
            reasons.append(f"Headwind Sharpe {_num(head['sharpe'])} turned negative")
        return (
            f"Group A full-window after-cost cumulative return is {_pct(full_net)}, "
            f"of which Tailwind strategy return is {_pct(tail['strategy_return'])}"
            f" (Sharpe {_num(tail['sharpe'])}). "
            + ". ".join(reasons)
            + ". This total is driven mainly by Tailwind, so reliability in a headwind is in question."
        )
    return (
        f"Headwind strategy return {_pct(head['strategy_return'])}, "
        f"SPY over the same days {_pct(head['spy_return'])}, excess {_pct(head['excess'])}, "
        f"Sharpe {_num(head['sharpe'])}. "
        f"Tailwind strategy return {_pct(tail['strategy_return'])}, Sharpe {_num(tail['sharpe'])}. "
        "Headwind did not lag SPY over the same days, and Sharpe did not turn negative."
    )


def _print_regime(regime: dict[str, Any]) -> None:
    """Print the Tailwind and Headwind comparison table."""
    print(f"\n=== Group A regime split ({regime['window']}, after costs) ===", flush=True)
    print(
        "Drawdown definition: before the close retakes a high, the lowest point falls more than "
        f"{_DRAWDOWN_THRESHOLD * 100:.0f}%. "
        "The strategy is still the full-window month-end top 3. The table only compounds the days that fall in each segment.",
        flush=True,
    )
    if not regime["episodes"]:
        print("SPY had no continuous stretch that fell more than 8% from a high in this window.", flush=True)
    else:
        deepest = min(regime["episodes"], key=lambda item: item["depth"])
        print(
            f"{len(regime['episodes'])} drawdowns exceeded 8%. "
            f"The deepest peaked on {deepest['peak_date']}, down {deepest['depth'] * 100:.1f}%.",
            flush=True,
        )
    print("Segment | Period | Strategy return | SPY return | Excess | Sharpe", flush=True)
    for row in regime["segments"]:
        print(
            f"{row['name']} | {row['period']} ({row['n_days']} trading days) | "
            f"{_pct(row['strategy_return'])} | {_pct(row['spy_return'])} | "
            f"{_pct(row['excess'])} | {_num(row['sharpe'])}",
            flush=True,
        )


def _pct(value: float) -> str:
    """Format a ratio as a percent. A non-finite value is shown as an em dash."""
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value * 100:.1f}%"


def _num(value: float) -> str:
    """Format a Sharpe ratio to two decimals."""
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:.2f}"


def _interpret(results: dict[str, dict[str, Any]]) -> str:
    """
    Use cumulative return to decide whether Group C is clearly better than the mature universes.

    Args:
        results: Performance dicts for the three groups.

    Returns:
        A conclusion that can be printed as-is.
    """
    cum = {key: float(results[key]["cumulative_return"]) for key in _GROUP_ORDER}
    gap_a = cum["C"] - cum["A"]
    gap_b = cum["C"] - cum["B"]
    if gap_a >= _CLEAR_GAP and gap_b >= _CLEAR_GAP:
        return (
            f"Group C's cumulative return is {gap_a * 100:.1f} percentage points above Group A "
            f"and {gap_b * 100:.1f} above Group B (the bar is 5). "
            "The selection rule is identical, so the extra performance comes from newer companies "
            "that were added to the universe after the fact, not from the technical signal being good at picking mature stocks."
        )
    if cum["C"] > cum["A"] and cum["C"] > cum["B"]:
        return (
            f"Group C is above Group A ({gap_a * 100:.1f} percentage points) and Group B ({gap_b * 100:.1f}), "
            "but not by more than 5 points on both. That is not enough to say the result comes mainly from newer companies."
        )
    return (
        "Group C is not better than both Group A and Group B. "
        "Under this month-end top-3 rule, putting newer companies in the universe did not produce a clearly better cumulative return."
    )


def _pp(value: float) -> str:
    """Format a return gap in percentage points. A non-finite value is shown as an em dash."""
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value * 100:.1f}"


def _print_cost_table(results: dict[str, dict[str, Any]]) -> None:
    """Print cumulative return before costs next to cumulative return after costs."""
    header = (
        f"{'Group':<28}{'Gross cumulative':>18}{'Net cumulative':>18}{'Gap (pp)':>12}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for key in _GROUP_ORDER:
        row = results[key]
        print(
            f"{_GROUP_TITLES[key]:<28}"
            f"{_pct(row['cumulative_return']):>16}"
            f"{_pct(row['cumulative_return_net']):>16}"
            f"{_pp(row['cost_drag']):>14}",
            flush=True,
        )


def _print_turnover(results: dict[str, dict[str, Any]]) -> None:
    """For groups whose gap exceeds 20 percentage points, print rebalance and name-change counts."""
    wide = [key for key in _GROUP_ORDER if float(results[key]["cost_drag"]) > _COST_GAP_ALERT]
    if not wide:
        return
    print(
        f"Gap above {_COST_GAP_ALERT * 100:.0f} percentage points. Name changes below "
        "(each buy or sell counts once; names that stay are not counted).",
        flush=True,
    )
    print(
        f"One-way rate {(_COMMISSION_ONE_SIDE + _SLIPPAGE_ONE_SIDE) * 100:.2f}% "
        f"(commission {_COMMISSION_ONE_SIDE * 100:.2f}% + slippage {_SLIPPAGE_ONE_SIDE * 100:.2f}%). "
        f"Replacing all three names costs {_ROUND_TRIP_COST * 100:.2f}% round trip.",
        flush=True,
    )
    for key in wide:
        row = results[key]
        print(
            f"  {_GROUP_TITLES[key]}: {row['n_rebalances']} rebalances, "
            f"{row['n_name_changes']} name changes "
            f"(buys {row['n_buys']}, sells {row['n_sells']}), "
            f"average one-way turnover {row['avg_one_way_turnover'] * 100:.0f}%, "
            f"cost rates sum to {_pct(row['total_cost_rate'])}",
            flush=True,
        )


def _print_table(results: dict[str, dict[str, Any]]) -> None:
    """Print the before-cost performance table for the three groups."""
    header = f"{'Group':<28}{'Names':>6}{'Cumulative':>12}{'Excess vs SPY':>14}{'Sharpe':>8}{'Max DD':>10}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for key in _GROUP_ORDER:
        row = results[key]
        print(
            f"{_GROUP_TITLES[key]:<28}"
            f"{row['n_tickers']:>6}"
            f"{_pct(row['cumulative_return']):>10}"
            f"{_pct(row['excess_vs_spy']):>12}"
            f"{_num(row['sharpe']):>8}"
            f"{_pct(row['max_drawdown']):>10}",
            flush=True,
        )


def _assert_realtime_regime_rules() -> None:
    """Check a hand-computed price path: Headwind starts only after an 8% break, and clears only on a new high."""
    index = pd.bdate_range("2024-01-02", periods=7)
    prices = pd.Series([100.0, 100.0, 95.0, 91.0, 95.0, 100.0, 92.0], index=index)
    labels = realtime_regime(prices).tolist()
    expected = ["Tailwind", "Tailwind", "Tailwind", "Headwind", "Headwind", "Tailwind", "Headwind"]
    if labels != expected:
        raise AssertionError(f"Realtime regime does not match the hand calculation: {labels}")
    still_under = realtime_regime(prices.iloc[:5])
    if str(still_under.iloc[-1]) != "Headwind":
        raise AssertionError("The label cleared once price was within 5% of the high, which breaks the rule")


def _annualized_trades(result: dict[str, Any]) -> float:
    """Annualized trade count. Buys and sells each count once, divided by the years in the holding window."""
    start = date.fromisoformat(str(result["start"]))
    end = date.fromisoformat(str(result["end"]))
    years = (end - start).days / 365.25
    if years <= 0:
        return float("nan")
    return float(result["n_name_changes"]) / years


def run_realtime_defense_compare(
    tickers: list[str],
    years: int | None = None,
    baseline: dict[str, Any] | None = None,
    start_date: date = BACKTEST_START,
    end_date: date = BACKTEST_END,
) -> dict[str, Any]:
    """
    Compare Group A's fixed monthly rule with the realtime headwind defense.

    The baseline still picks the top 3 at each month-end. The defensive book holds SPY
    for a month when the rebalance close is already labeled Headwind. Costs still follow weight changes.
    If the truncation check fails, the defensive book is not run, so a label that can see the future is not treated as a tradable rule.

    Args:
        tickers: Group A candidate symbols.
        years: Legacy "years back from today". None uses start_date to end_date.
        baseline: A finished baseline backtest. When passed it is not recomputed, and its window must match.
        start_date: First performance date, inclusive. Defaults to config.BACKTEST_START.
        end_date: Last performance date, inclusive. Defaults to config.BACKTEST_END.

    Returns:
        A dict with baseline, defensive, and mismatches.
    """
    _assert_realtime_regime_rules()
    window = {"years": years, "start_date": start_date, "end_date": end_date}
    base = baseline if baseline is not None else run_group_backtest(tickers, **window)
    mismatches = verify_realtime_regime(base["spy_close"])
    if mismatches:
        raise RuntimeError("Realtime regime truncation check disagreed, so the defensive comparison did not run")
    regime = realtime_regime(base["spy_close"])
    defensive = run_group_backtest(tickers, defend_regime=regime, **window)
    if (defensive["start"], defensive["end"]) != (base["start"], base["end"]):
        raise RuntimeError(
            f"Defensive window {defensive['start']} ~ {defensive['end']} differs from the baseline "
            f"{base['start']} ~ {base['end']}, so they cannot be compared"
        )
    return {
        "baseline": base,
        "defensive": defensive,
        "mismatches": mismatches,
    }


def _window_return(daily: pd.Series, start: str, end: str) -> pd.Series:
    """Slice daily returns for a holding window. Both endpoints are included."""
    return daily.loc[pd.Timestamp(start) : pd.Timestamp(end)]


def _print_defended_month_gaps(baseline: dict[str, Any], defense: dict[str, Any]) -> None:
    """
    Print each month the defensive book held SPY: Group A's normal pick return, SPY over the same days, and the gap.

    The holding window matches the backtest, from the trading day after the rebalance through the next rebalance close.
    Both sides are price returns for that stretch. The next rebalance's cost is not charged inside the month.

    Args:
        baseline: Full stock-picking backtest. It must include months and gross_returns.
        defense: Backtest that holds SPY in Headwind months. It decides which months are listed.
    """
    stock_by_signal = {month["signal"]: month for month in baseline.get("months", [])}
    gross = baseline.get("gross_returns")
    spy = baseline.get("spy_returns")
    if gross is None or spy is None:
        _log.warning("The baseline backtest has no daily returns, so months cannot be compared")
        return

    defended = [month for month in defense.get("months", []) if month.get("defended")]
    print("\nMonths the defensive book held SPY (Group A stock picks vs SPY)", flush=True)
    header = f"{'Holding window':<27}{'Group A picks':<22}{'Group A':>10}{'SPY':>10}{'Gap':>10}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    stock_factors: list[float] = []
    spy_factors: list[float] = []
    for month in defended:
        stock_month = stock_by_signal.get(month["signal"])
        if stock_month is None:
            _log.warning("Baseline backtest has no rebalance day %s", month["signal"])
            continue
        window = _window_return(gross, stock_month["hold_start"], stock_month["hold_end"])
        spy_window = _window_return(spy, stock_month["hold_start"], stock_month["hold_end"]).reindex(window.index)
        if window.empty or spy_window.isna().all():
            _log.warning("%s has no daily returns to compare", month["signal"])
            continue
        stock_return = _cumulative(window)
        spy_return = _cumulative(spy_window.fillna(0.0))
        names = ",".join(stock_month["picked"]) if stock_month["picked"] else "cash"
        period = f"{stock_month['hold_start']} ~ {stock_month['hold_end']}"
        print(
            f"{period:<27}{names:<22}"
            f"{_pct(stock_return):>10}{_pct(spy_return):>10}{_pct(stock_return - spy_return):>10}",
            flush=True,
        )
        stock_factors.append(1.0 + stock_return)
        spy_factors.append(1.0 + spy_return)
    if stock_factors and spy_factors:
        stock_total = float(math.prod(stock_factors) - 1.0)
        spy_total = float(math.prod(spy_factors) - 1.0)
        print("-" * len(header), flush=True)
        print(
            f"{f'{len(stock_factors)} stretches compounded separately (middle months excluded)':<49}"
            f"{_pct(stock_total):>10}{_pct(spy_total):>10}{_pct(stock_total - spy_total):>10}",
            flush=True,
        )


def _print_realtime_defense(comparison: dict[str, Any]) -> None:
    """Print after-cost performance for the baseline and the realtime headwind defense, and say so when there is no improvement."""
    base = comparison["baseline"]
    defense = comparison["defensive"]
    window = f"{base['start']} ~ {base['end']}"
    print(f"\n=== Realtime headwind defense (Group A monthly, after costs, {window}) ===", flush=True)
    print(
        "When the rebalance close is already 8% below the highest SPY close through that day, that month holds SPY. "
        "Stock picking resumes only after a later rebalance close makes a new high.",
        flush=True,
    )
    header = f"{'Setting':<28}{'Total return':>14}{'Sharpe':>8}{'Max DD':>10}{'Trades/year':>14}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    rows = (
        ("Baseline (keep picking)", base),
        ("Defense (hold SPY in Headwind)", defense),
    )
    for title, row in rows:
        print(
            f"{title:<28}"
            f"{_pct(row['cumulative_return_net']):>10}"
            f"{_num(row['sharpe_net']):>8}"
            f"{_pct(row['max_drawdown_net']):>10}"
            f"{_annualized_trades(row):>12.1f}",
            flush=True,
        )
    dates = defense.get("headwind_signal_dates") or []
    print(
        f"Headwind months: {defense['n_headwind_signals']} (the month-end rebalance day was already down 8%)",
        flush=True,
    )
    if dates:
        print("  Headwind rebalance days: " + ", ".join(str(item) for item in dates), flush=True)
    print(
        f"Defensive book actually held SPY: {defense['n_defended_months']} months",
        flush=True,
    )
    _print_defended_month_gaps(base, defense)
    sharpe_better = float(defense["sharpe_net"]) > float(base["sharpe_net"])
    # Max drawdown is negative. A larger number means a shallower drop.
    drawdown_better = float(defense["max_drawdown_net"]) > float(base["max_drawdown_net"])
    print(
        f"Sharpe {_num(base['sharpe_net'])} -> {_num(defense['sharpe_net'])}, "
        f"max drawdown {_pct(base['max_drawdown_net'])} -> {_pct(defense['max_drawdown_net'])}.",
        flush=True,
    )
    if not sharpe_better or not drawdown_better:
        print("The realtime headwind defense has no edge", flush=True)


if __name__ == "__main__":
    groups = define_three_groups()
    print(
        f"Backtest rule: first rebalance at the end of {BACKTEST_FIRST_REBALANCE_MONTH}, "
        f"held through the end of {BACKTEST_LAST_MONTH}, {BACKTEST_MONTHS} full months. "
        f"Calendar bounds {BACKTEST_START.isoformat()} ~ {BACKTEST_END.isoformat()}, "
        f"group-list end date {GROUP_DEFINITION_END.isoformat()}",
        flush=True,
    )
    print("=== Three universes ===", flush=True)
    for key in _GROUP_ORDER:
        print(f"{_GROUP_TITLES[key]} ({len(groups[key])} names)", flush=True)
        print("  " + ", ".join(groups[key]), flush=True)
    extra_b = sorted(set(groups["B"]) - set(groups["A"]))
    only_c = sorted(set(groups["C"]) - set(groups["B"]))
    print(f"Foreign or unclassified large caps in B but not A ({len(extra_b)}): {', '.join(extra_b)}", flush=True)
    print(f"Newer companies only in C ({len(only_c)}): {', '.join(only_c)}", flush=True)
    print(
        "Rule: on the last trading day of each month, pick the top 3 by that day's technical score and hold them equal-weight to the next month-end. "
        f"Names with fewer than {_MIN_HISTORY_BARS} trading days are not candidates. "
        f"Rebalance cost is {_ONE_WAY_COST * 100:.2f}% one way (commission {_COMMISSION_ONE_SIDE * 100:.2f}% + slippage {_SLIPPAGE_ONE_SIDE * 100:.2f}%). "
        f"Replacing all three names costs {_ROUND_TRIP_COST * 100:.2f}%. Sharpe uses a risk-free rate of 0.",
        flush=True,
    )

    results: dict[str, dict[str, Any]] = {}
    for key in _GROUP_ORDER:
        print(f"\n=== Backtest {_GROUP_TITLES[key]} ===", flush=True)
        results[key] = run_group_backtest(groups[key])

    spy_return = float(results["A"]["spy_cumulative_return"])
    window = f"{results['A']['start']} ~ {results['A']['end']}"
    print(f"\n=== Before costs ({window}, SPY over the same window {_pct(spy_return)}) ===", flush=True)
    _print_table(results)
    print("\n=== Trading-cost comparison ===", flush=True)
    _print_cost_table(results)
    _print_turnover(results)
    for key in _GROUP_ORDER:
        dropped = results[key]["dropped_tickers"]
        if dropped:
            print(f"{_GROUP_TITLES[key]} dropped for missing prices: {', '.join(dropped)}", flush=True)
    print(_interpret(results), flush=True)

    regime = run_regime_split_backtest(tickers=groups["A"], backtest=results["A"])
    _print_regime(regime)
    print(_interpret_regime(regime), flush=True)

    defense = run_realtime_defense_compare(groups["A"], baseline=results["A"])
    _print_realtime_defense(defense)
