# -*- coding: utf-8 -*-
"""
三組股票母體的技術面回測對照。

組 A：2022 年後進過 S&P 500 的重疊名單。
組 B：組 A，再加上國籍不符、因此進不了 S&P 500 的外國大型股；
      不含 sp500_but_excluded 那批近年才竄起的美國公司。
組 C：現在的完整 60 檔，含那批新興公司。

三組用同一段期間、同一套規則：每個月底用當天技術面分數選前 3 名，
等權持有到下一個月底（大約 30 個日曆天）。分數在收盤才知道，
報酬從下一個交易日才開始算，避免用到當天已經走完的漲跌。

交易成本只扣在實際成交的部位：手續費單邊 0.05%，滑價單邊 0.1%。
買進扣一次、賣出再扣一次。三檔全部換掉時，進出合計 0.30%。
留在名單裡的持股不重複扣。

若組 C 明顯好於組 A 與組 B，多出來的績效來自「母體裡有事後才納入的新興公司」，
不是這套技術面訊號在成熟股票上特別會選。

組 A 另外做順風／逆風拆分。SPY 從某個收盤高點起、在再次收復該高點之前，
最低點跌幅超過 8%，整段算逆風；其餘交易日算順風。選股規則不變，
只把落在該區段的扣成本後日報酬分開複利，用來檢查總成績是不是只靠順風段。
這段事後標籤在高點隔天就開始算逆風，當時還沒跌到 8%，實盤不知道。

即時版 realtime_regime 每個交易日只看截至當天的最高收盤價。
收盤價低於這個高點 8% 以上才標逆風，之後要收盤創新高才解除。
防禦版在月底換股日若處在即時逆風，該月改持有 SPY，不選個股。

執行方式（於專案根目錄）：
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

# 組別對照用的歷史窗口，與先前 S&P 500 成分股比較同一段
_GROUP_START = date(2022, 1, 1)
# MA200 需要約 200 個交易日，多抓一段日曆天當暖機，這段不計入績效
_WARMUP_DAYS = 420
# 未滿這麼多個交易日的股票不進當月候選，避免上市初期指標失真
_MIN_HISTORY_BARS = 200
_TOP_N = 3
_BENCHMARK = "SPY"
# 無風險利率設為 0，Sharpe 只比較報酬波動，不引入利率假設
_TRADING_DAYS = 252
# 組 C 比另外兩組都高出這個幅度，才說「明顯來自新興公司」
_CLEAR_GAP = 0.05
# 手續費、滑價都是單邊。買進扣一次，賣出再扣一次。
# 部位全部換掉時：(0.05% + 0.10%) × 2 = 0.30%。
_COMMISSION_ONE_SIDE = 0.0005
_SLIPPAGE_ONE_SIDE = 0.001
_ONE_WAY_COST = _COMMISSION_ONE_SIDE + _SLIPPAGE_ONE_SIDE
_ROUND_TRIP_COST = _ONE_WAY_COST * 2
# 未扣成本與扣成本後的累積報酬差距超過這個幅度，就印出換股次數
_COST_GAP_ALERT = 0.20
# 從收盤高點下跌超過這個幅度，才把該段連續區間視為逆風
_DRAWDOWN_THRESHOLD = 0.08

_GROUP_ORDER = ("A", "B", "C")
_GROUP_TITLES = {
    "A": "A 成熟美股",
    "B": "B 含外國大型股",
    "C": "C 完整清單",
}


def define_three_groups(
    start_date: date = _GROUP_START,
    end_date: date = GROUP_DEFINITION_END,
) -> dict[str, list[str]]:
    """
    用 S&P 500 時點比較結果組出三份母體。

    組 A 是 overlap。組 B 是完整清單去掉 sp500_but_excluded，
    所以檔數是「60 減新興公司」，會包含 not_sp500_listed，
    以及查不到國籍、因此沒被算進那 16 檔的代號。
    組 C 是 config.TICKER_LIST。

    參數:
        start_date: 判斷「曾經進過 S&P 500」的起日。
        end_date: 迄日。預設 config.GROUP_DEFINITION_END，名單不隨執行日變動。

    回傳:
        {"A": [...], "B": [...], "C": [...]}，代號已排序。
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
        raise RuntimeError("三組母體不是巢狀關係，分類結果不能拿來對照")
    if not group_a or not group_b or not group_c:
        raise RuntimeError("三組母體有空清單，無法回測")

    extra = sorted(set(group_b) - set(group_a))
    _log.info(
        "三組母體：A=%s 檔，B=%s 檔（比 A 多 %s），C=%s 檔（比 B 多 %s）",
        len(group_a),
        len(group_b),
        ",".join(extra) or "無",
        len(group_c),
        ",".join(sorted(set(group_c) - set(group_b))) or "無",
    )
    return {"A": group_a, "B": group_b, "C": group_c}


def _as_timestamp_index(index: pd.Index) -> pd.DatetimeIndex:
    """把索引收成無時區、歸一到午夜的日期索引。"""
    stamps = pd.to_datetime(index)
    if getattr(stamps, "tz", None) is not None:
        stamps = stamps.tz_localize(None)
    return pd.DatetimeIndex(stamps.normalize())


def _score_frame(
    panel: dict[str, pd.DataFrame],
    master: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    對面板裡每一檔算技術面分數，並對齊到主日曆。

    參數:
        panel: build_ohlcv_panel 的結果，不含基準。
        master: 績效使用的交易日，通常是 SPY 的日期。

    回傳:
        (收盤價, 技術面分數, 是否已滿最短歷史)。缺資料是 NaN / False。
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
            _log.warning("%s 技術面分數計算失敗，這檔不進候選: %s", ticker, exc)
            continue
        if score.empty:
            _log.warning("%s 技術面分數是空的，這檔不進候選", ticker)
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
    取出績效區間內、每個月最後一個交易日，當作選股日。

    最後一個交易日本身不當訊號日：當天收盤才選完，已經沒有下一天可以持有。

    參數:
        index: 主日曆。
        start: 績效起日（含）。
        end: 績效迄日（含）。

    回傳:
        由早到晚的訊號日。
    """
    window = index[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))]
    if len(window) < 2:
        return []
    month_ends = pd.Series(window, index=window).groupby(window.to_period("M")).max()
    last_day = window[-1]
    return [pd.Timestamp(stamp) for stamp in month_ends.tolist() if pd.Timestamp(stamp) < last_day]


def _target_weights(picked: list[str]) -> pd.Series:
    """等權目標部位。空名單代表這個月轉成現金。"""
    if not picked:
        return pd.Series(dtype=float)
    weight = 1.0 / len(picked)
    return pd.Series(weight, index=picked, dtype=float)


def _schedule_trade_costs(
    books: list[dict[str, Any]],
) -> tuple[dict[pd.Timestamp, float], dict[str, float | int]]:
    """
    依前後兩期的目標權重計算每次調倉成本。

    成本 = (手續費 0.05% + 滑價 0.1%) × (買進權重 + 賣出權重)。
    三檔全部換成另外三檔時，買進 100%、賣出 100%，合計 0.30%。
    第一個月從現金建倉，只有買進，扣 0.15%。繼續持有的部位權重沒變，不扣。

    參數:
        books: 每期的 first_hold_day 與 weights。

    回傳:
        (每個持有首日要扣的成本比例, 換股統計)。
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
    依月底分數持有前 3 名，回傳未扣成本的日報酬，以及扣在換股日的成本。

    訊號日 t 的分數只用到 t 收盤。持有區間是 t 之後、下一個訊號日（含）為止的
    日報酬，也就是吃到「從 t 收盤持有到下一次換股收盤」這一段。
    成本扣在新部位的第一個持有日：先付交易成本，再吃當天漲跌。

    參數:
        close: 對齊後的收盤價。防禦版需含 SPY 欄。
        score: 對齊後的技術面分數。
        ready: 是否已滿最短歷史。
        signals: 選股日。
        defend_regime: 即時順風／逆風。換股日為逆風時，該月改持有 SPY。

    回傳:
        (未扣成本日報酬, 與之對齊的當日成本比例, 換股統計)。
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
        # 換股日收盤才知道即時逆風。這個月不選個股，整個月持有 SPY。
        regime_label = None
        if defend_regime is not None and signal_day in defend_regime.index:
            regime_label = str(defend_regime.loc[signal_day])
        if regime_label == "逆風":
            n_headwind_signals += 1
            headwind_dates.append(signal_day.date().isoformat())
        if regime_label == "逆風" and _BENCHMARK in close.columns:
            if pd.notna(close.at[signal_day, _BENCHMARK]):
                weights.loc[hold, _BENCHMARK] = 1.0
                _log.info(
                    "%s 即時逆風，這個月改持有 SPY，持有到 %s",
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
            _log.warning("%s 即時逆風，但沒有 SPY 收盤價，這個月保持現金", signal_day.date())

        tradable = (
            score.loc[signal_day].notna()
            & close.loc[signal_day].reindex(score.columns).notna()
            & ready.loc[signal_day].fillna(False).astype(bool)
        )
        ranked = score.loc[signal_day].loc[tradable].dropna()
        if ranked.empty:
            _log.warning("%s 沒有合格候選，這個月保持現金", signal_day.date())
            picked: list[str] = []
        else:
            order = ranked.reset_index()
            order.columns = ["ticker", "score"]
            order = order.sort_values(["score", "ticker"], ascending=[False, True])
            picked = order["ticker"].head(_TOP_N).tolist()
            weights.loc[hold, picked] = 1.0 / len(picked)
            _log.info(
                "%s 選出 %s，持有到 %s",
                signal_day.date().isoformat(),
                ",".join(picked),
                pd.Timestamp(hold[-1]).date().isoformat(),
            )
        # 第一筆是從現金買進，成本落在開始持有的那一天。
        # 之後的調倉在訊號日收盤換股，成本落在仍持有舊部位的那個收盤。
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
    # 當天沒成交的持股把權重分給其餘仍持有的名字，避免無故變成部分現金
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
            _log.warning("換股日 %s 沒有對應報酬，這筆 %.2f%% 成本沒扣到", stamp.date(), cost * 100)
    return gross, costs, stats


def _net_returns(gross: pd.Series, costs: pd.Series) -> pd.Series:
    """
    先扣成本再吃當天報酬。

    參數:
        gross: 未扣成本的日報酬。
        costs: 同一天要扣掉的淨值比例。

    回傳:
        扣成本後的日報酬。
    """
    aligned = costs.reindex(gross.index).fillna(0.0)
    return (1.0 - aligned) * (1.0 + gross) - 1.0


def _max_drawdown(equity: pd.Series) -> float:
    """權益曲線的最大回撤。沒有回撤時為 0。"""
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def _sharpe(daily_returns: pd.Series) -> float:
    """年化 Sharpe。無風險利率為 0；波動為 0 時回傳 NaN。"""
    if len(daily_returns) < 2:
        return float("nan")
    sigma = float(daily_returns.std(ddof=1))
    if not math.isfinite(sigma) or sigma <= 0:
        return float("nan")
    return float(daily_returns.mean() / sigma * math.sqrt(_TRADING_DAYS))


def _cumulative(daily_returns: pd.Series) -> float:
    """日報酬複利後的累積報酬。"""
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
    對單一母體跑月底選前 3 名、持有到下月底的技術面回測。

    價量來自 backtest_data.build_ohlcv_panel。分數用
    compute_tech_score_series，第 t 天不含 t 之後的價格。
    績效起點之前多抓一段暖機，讓 MA200 在第一個換股日已經有效，
    暖機期間不計入累積報酬，SPY 也只在相同的持有日上計算。
    同一組同時算出未扣成本與扣成本後的累積報酬。成本只落在權重有變動的部位。

    參數:
        tickers: 這一組的候選代號。
        years: 舊版「從今天往回幾年」。None 時改用 start_date ~ end_date 固定區間；
            有傳入時結果會隨執行日變動，不能拿來對照履歷數字。
        defend_regime: 即時順風／逆風。有傳入時，換股日為逆風的月份改持有 SPY。
        start_date: 績效起日（含），預設 config.BACKTEST_START。
        end_date: 績效迄日（含），預設 config.BACKTEST_END。

    回傳:
        含 cumulative_return（未扣成本）、cumulative_return_net（扣成本後）、
        cost_drag、換股統計、excess_vs_spy、sharpe、max_drawdown 的績效字典。
        另附 net_returns、spy_returns、spy_close，供順風／逆風拆分複用同一條路徑。
    """
    if years is not None:
        if years < 1:
            raise ValueError(f"years 必須至少 1 年，收到 {years}")
        end = date.today()
        eval_start = end - timedelta(days=365 * years)
    else:
        end = end_date
        eval_start = start_date
    if eval_start >= end:
        raise ValueError(f"回測起日 {eval_start} 必須早於迄日 {end}")
    fetch_start = eval_start - timedelta(days=_WARMUP_DAYS)
    symbols = [str(ticker).upper().strip() for ticker in tickers if str(ticker).strip()]

    panel = build_ohlcv_panel([*symbols, _BENCHMARK], fetch_start, end)
    spy = panel.get(_BENCHMARK)
    if spy is None or spy.empty:
        raise RuntimeError("沒有 SPY 價量，無法計算超額報酬")

    master = _as_timestamp_index(spy.index)
    master = master[(master >= pd.Timestamp(fetch_start)) & (master <= pd.Timestamp(end))]
    stock_panel = {ticker: frame for ticker, frame in panel.items() if ticker != _BENCHMARK}
    dropped = sorted(set(symbols) - set(stock_panel))
    if dropped:
        _log.warning("這些代號沒有價量，已從這一組剔除: %s", ",".join(dropped))
    if not stock_panel:
        raise RuntimeError("這一組沒有任何可用價量，無法回測")

    close, score, ready = _score_frame(stock_panel, master)
    spy_close = pd.to_numeric(spy["Close"], errors="coerce")
    spy_close.index = _as_timestamp_index(spy_close.index)
    spy_close = spy_close.reindex(master)

    signals = _month_end_signals(master, eval_start, end)
    if not signals:
        raise RuntimeError(f"{eval_start} ~ {end} 之間沒有月底換股日")

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
        raise RuntimeError("沒有任何一個月選到股票，無法計算績效")

    net_ret = _net_returns(strategy_ret, trade_costs)
    spy_ret = spy_close.pct_change().reindex(strategy_ret.index)
    if spy_ret.isna().all():
        raise RuntimeError("SPY 在持有期間沒有報酬")
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
        # 順風／逆風拆分用。spy_close 含暖機，才能認出回測起點前已開始的回調。
        "net_returns": net_ret,
        "spy_returns": spy_ret,
        "spy_close": spy_close.dropna(),
    }
    _log.info(
        "回測完成 %s ~ %s：未扣成本 %.2f%%，扣成本後 %.2f%%，差距 %.2f 個百分點",
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
    用截至當天的資料判斷順風或逆風。

    歷史最高價是當天（含）以前的最高收盤價。收盤價低於這個高點
    threshold 以上，當天起標成逆風。之後價格回升但還沒創新高，
    仍然是逆風。收盤價大於等於前高，當天解除，改回順風。

    這和事後回調區間不同：事後版從高點隔天就標逆風，
    即使當時跌幅還沒到 8%。這裡要等跌幅已經發生才標。

    參數:
        spy_close: SPY 收盤價。
        threshold: 跌幅門檻，0.08 代表 8%。

    回傳:
        與有效收盤價同索引的 Series，值為「順風」或「逆風」。
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
        labels.append("逆風" if headwind else "順風")
    return pd.Series(labels, index=prices.index, name="regime")


def verify_realtime_regime(
    spy_close: pd.Series,
    threshold: float = _DRAWDOWN_THRESHOLD,
) -> list[str]:
    """
    截斷檢查：只給到第 t 天的資料，第 t 天的順風／逆風必須和完整序列一致。

    參數:
        spy_close: SPY 收盤價。
        threshold: 與 realtime_regime 相同的跌幅門檻。

    回傳:
        不一致的日期字串。空清單表示通過。不一致的日期會直接印出。
    """
    prices = pd.to_numeric(spy_close, errors="coerce").dropna()
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    if prices.empty:
        print("即時逆風截斷檢查失敗：沒有 SPY 收盤價", flush=True)
        return ["<empty>"]
    full = realtime_regime(prices, threshold)
    mismatches: list[str] = []
    for pos in range(len(prices)):
        truncated = realtime_regime(prices.iloc[: pos + 1], threshold)
        if truncated.empty or str(truncated.iloc[-1]) != str(full.iloc[pos]):
            mismatches.append(pd.Timestamp(prices.index[pos]).date().isoformat())
    if mismatches:
        print(f"即時逆風截斷檢查不一致，共 {len(mismatches)} 天：", flush=True)
        for label in mismatches:
            print(f"  {label}", flush=True)
    else:
        print(f"即時逆風截斷檢查通過：{len(prices)} 個交易日", flush=True)
    return mismatches


def _spy_drawdown_episodes(
    close: pd.Series,
    threshold: float = _DRAWDOWN_THRESHOLD,
) -> list[dict[str, Any]]:
    """
    找出收盤價從高點下跌超過門檻的連續區間。

    高點是創新高的收盤。下一次收盤再站上這個高點之前，
    若最低點相對高點的跌幅超過 threshold，整段算一次回調：
    從高點的下一個交易日起，到收復日（含）為止。
    資料結束時仍沒收復，終點就是最後一個交易日。

    參數:
        close: SPY 收盤價，索引為交易日。
        threshold: 跌幅門檻，0.08 代表 8%。

    回傳:
        每段一個字典，含 peak_date、peak、trough_date、trough、
        depth（負數）、start、end、recovered。
    """
    prices = pd.to_numeric(close, errors="coerce").dropna()
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    if len(prices) < 2:
        return []

    episodes: list[dict[str, Any]] = []
    peak_pos = 0
    peak_px = float(prices.iloc[0])

    def _record(peak_at: int, end_at: int, recovered: bool, peak_price: float) -> None:
        """高點到 end_at 之間若跌破門檻，記成一段回調。"""
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
    把同一批交易日的策略報酬與 SPY 報酬做成一列對照。

    參數:
        name: 區段名稱，例如順風段、逆風段。
        period: 印在表格上的期間文字。
        strategy: 該區段的策略日報酬。
        spy: 同一批日期的 SPY 日報酬。

    回傳:
        含策略報酬、SPY 報酬、超額與 Sharpe 的字典。
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
    把組 A 在回測期間的扣成本後日報酬切成順風段與逆風段。

    逆風段是 SPY 從高點下跌超過 8% 的連續區間（見 _spy_drawdown_episodes）。
    高點可以落在回測起點之前，這樣起點時已經在跌的段落不會被漏掉；
    績效只計與組 A 持有日重疊的交易日。順風段是扣掉這些日子之後的其餘交易日。

    選股不重跑。訊號仍是全程每個月底用當天技術面分數選前 3 名，
    區段數字只是把同一條淨值路徑上的日報酬分開複利。
    因此兩段淨值相乘會回到全程扣成本後的累積報酬。
    這是事後檢查，不是可以當下使用的交易濾網。

    參數:
        years: 舊版「從今天往回幾年」。None 時用 start_date ~ end_date。已傳入 backtest 時不再重跑。
        tickers: 組 A 代號。省略時會自己重抓三組母體再取組 A。
        backtest: 已經跑完的 run_group_backtest 結果，需含 net_returns 與 spy_close。
        start_date: 績效起日（含），預設 config.BACKTEST_START。
        end_date: 績效迄日（含），預設 config.BACKTEST_END。

    回傳:
        episodes 為回調清單，segments 為順風、逆風兩列績效，
        full_net 為全程扣成本後累積報酬。
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
        raise RuntimeError("回測結果沒有扣成本後日報酬，無法切順風／逆風")
    if not isinstance(spy_ret, pd.Series) or not isinstance(spy_close, pd.Series):
        raise RuntimeError("回測結果沒有 SPY 價格，無法偵測回調")

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
            "SPY 回調 %s ~ %s，高點 %s %.2f，最大跌幅 %.1f%%，%s",
            episode["start"],
            episode["end"],
            episode["peak_date"],
            episode["peak"],
            episode["depth"] * 100,
            "已收復" if episode["recovered"] else "尚未收復",
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
            f"（高點 {episode['peak_date']}，跌幅 {episode['depth'] * 100:.1f}%）"
        )

    head_ret = net.loc[headwind]
    tail_ret = net.loc[~headwind]
    head_spy = spy_ret.reindex(head_ret.index)
    tail_spy = spy_ret.reindex(tail_ret.index)

    # 兩段必須把每個持有日剛好分到一邊，相乘才會回到全程淨值
    combined = float((1.0 + head_ret).prod() * (1.0 + tail_ret).prod() - 1.0)
    full_net = float(backtest["cumulative_return_net"])
    if math.isfinite(full_net) and not math.isclose(combined, full_net, rel_tol=0, abs_tol=1e-8):
        _log.warning(
            "順風與逆風複利相乘 %.4f%%，與全程扣成本後 %.4f%% 不一致",
            combined * 100,
            full_net * 100,
        )

    if visible:
        head_period = "；".join(period_parts)
    else:
        head_period = "這段期間沒有從高點下跌超過 8% 的連續區間"
    tail_period = f"{window_start.date()} ~ {window_end.date()}，扣除回調日"

    segments = [
        _segment_performance("順風段", tail_period, tail_ret, tail_spy),
        _segment_performance("逆風段", head_period, head_ret, head_spy),
    ]
    return {
        "episodes": visible,
        "segments": segments,
        "full_net": full_net,
        "window": f"{window_start.date()} ~ {window_end.date()}",
    }


def _interpret_regime(regime: dict[str, Any]) -> str:
    """
    判斷逆風段有沒有跑輸 SPY，或 Sharpe 有沒有轉負。

    參數:
        regime: run_regime_split_backtest 的回傳。

    回傳:
        一段可直接印出的結論。
    """
    by_name = {row["name"]: row for row in regime["segments"]}
    head = by_name["逆風段"]
    tail = by_name["順風段"]
    full_net = float(regime["full_net"])
    if not regime["episodes"] or head["n_days"] == 0:
        return (
            f"組 A 全程扣成本後累積報酬 {_pct(full_net)}。"
            "這段期間沒有偵測到 SPY 從高點下跌超過 8% 的連續區間，"
            "無法分開檢驗逆風環境。"
        )

    loses = math.isfinite(head["excess"]) and head["excess"] < 0
    sharpe_negative = math.isfinite(head["sharpe"]) and head["sharpe"] < 0
    if loses or sharpe_negative:
        reasons: list[str] = []
        if loses:
            reasons.append(
                f"逆風段策略報酬 {_pct(head['strategy_return'])}，"
                f"同期 SPY {_pct(head['spy_return'])}，超額 {_pct(head['excess'])}"
            )
        if sharpe_negative:
            reasons.append(f"逆風段 Sharpe {_num(head['sharpe'])}，已轉為負值")
        return (
            f"組 A 全程扣成本後累積報酬 {_pct(full_net)}，"
            f"其中順風段策略報酬 {_pct(tail['strategy_return'])}"
            f"（Sharpe {_num(tail['sharpe'])}）。"
            + "。".join(reasons)
            + "。這段總成績主要由順風段貢獻，策略在逆風環境下的可靠度存疑。"
        )
    return (
        f"逆風段策略報酬 {_pct(head['strategy_return'])}，"
        f"同期 SPY {_pct(head['spy_return'])}，超額 {_pct(head['excess'])}，"
        f"Sharpe {_num(head['sharpe'])}。"
        f"順風段策略報酬 {_pct(tail['strategy_return'])}，Sharpe {_num(tail['sharpe'])}。"
        "逆風段沒有跑輸同期 SPY，Sharpe 也沒有轉負。"
    )


def _print_regime(regime: dict[str, Any]) -> None:
    """印出順風段與逆風段的對照表。"""
    print(f"\n=== 組 A 順風／逆風（{regime['window']}，扣成本後）===", flush=True)
    print(
        "回調定義：收盤價從高點到再次收復該高點之前，最低點跌幅超過 "
        f"{_DRAWDOWN_THRESHOLD * 100:.0f}%。"
        "策略仍是全程月底選前 3 名，下表只把落在該區段的交易日分開複利。",
        flush=True,
    )
    if not regime["episodes"]:
        print("這段期間 SPY 沒有從高點下跌超過 8% 的連續區間。", flush=True)
    else:
        deepest = min(regime["episodes"], key=lambda item: item["depth"])
        print(
            f"超過 8% 的回調共 {len(regime['episodes'])} 段，"
            f"最深的一段高點在 {deepest['peak_date']}，跌幅 {deepest['depth'] * 100:.1f}%。",
            flush=True,
        )
    print("區段 | 期間 | 策略報酬 | SPY 報酬 | 超額 | Sharpe", flush=True)
    for row in regime["segments"]:
        print(
            f"{row['name']} | {row['period']}（{row['n_days']} 個交易日） | "
            f"{_pct(row['strategy_return'])} | {_pct(row['spy_return'])} | "
            f"{_pct(row['excess'])} | {_num(row['sharpe'])}",
            flush=True,
        )


def _pct(value: float) -> str:
    """把比率格式化成百分比。無效值顯示為破折號。"""
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value * 100:.1f}%"


def _num(value: float) -> str:
    """把 Sharpe 格式化成兩位小數。"""
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:.2f}"


def _interpret(results: dict[str, dict[str, Any]]) -> str:
    """
    用累積報酬判斷組 C 是否明顯好於成熟股母體。

    參數:
        results: 三組的績效字典。

    回傳:
        一段可直接印出的結論。
    """
    cum = {key: float(results[key]["cumulative_return"]) for key in _GROUP_ORDER}
    gap_a = cum["C"] - cum["A"]
    gap_b = cum["C"] - cum["B"]
    if gap_a >= _CLEAR_GAP and gap_b >= _CLEAR_GAP:
        return (
            f"組 C 的累積報酬比組 A 高 {gap_a * 100:.1f} 個百分點、"
            f"比組 B 高 {gap_b * 100:.1f} 個百分點（門檻是 5）。"
            "選股規則完全相同，多出來的績效來自母體裡那些事後才放進來的新興公司，"
            "不是技術面訊號在成熟股票上本身會選。"
        )
    if cum["C"] > cum["A"] and cum["C"] > cum["B"]:
        return (
            f"組 C 高於組 A（{gap_a * 100:.1f} 個百分點）與組 B（{gap_b * 100:.1f} 個百分點），"
            "但沒有同時超過 5 個百分點。還不足以單獨說績效主要來自新興公司。"
        )
    return (
        "組 C 並沒有同時好於組 A 與組 B。"
        "在這套月底選前 3 名的規則下，把新興公司放進母體沒有帶來明顯更好的累積報酬。"
    )


def _pp(value: float) -> str:
    """把報酬差距格式化成百分點。無效值顯示為破折號。"""
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value * 100:.1f}"


def _print_cost_table(results: dict[str, dict[str, Any]]) -> None:
    """印出未扣成本與扣成本後的累積報酬對照。"""
    header = (
        f"{'組別':<16}{'未扣成本累積報酬':>16}{'扣成本後累積報酬':>16}{'差距(百分點)':>14}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for key in _GROUP_ORDER:
        row = results[key]
        print(
            f"{_GROUP_TITLES[key]:<16}"
            f"{_pct(row['cumulative_return']):>16}"
            f"{_pct(row['cumulative_return_net']):>16}"
            f"{_pp(row['cost_drag']):>14}",
            flush=True,
        )


def _print_turnover(results: dict[str, dict[str, Any]]) -> None:
    """差距超過 20 個百分點的組別，印出調倉與換股次數。"""
    wide = [key for key in _GROUP_ORDER if float(results[key]["cost_drag"]) > _COST_GAP_ALERT]
    if not wide:
        return
    print(
        f"差距超過 {_COST_GAP_ALERT * 100:.0f} 個百分點，換股次數如下"
        "（每一檔買進或賣出算一次；繼續持有的不計）。",
        flush=True,
    )
    print(
        f"單邊費率 {(_COMMISSION_ONE_SIDE + _SLIPPAGE_ONE_SIDE) * 100:.2f}%"
        f"（手續費 {_COMMISSION_ONE_SIDE * 100:.2f}% + 滑價 {_SLIPPAGE_ONE_SIDE * 100:.2f}%）。"
        f"三檔全部換掉時，買賣合計 {_ROUND_TRIP_COST * 100:.2f}%。",
        flush=True,
    )
    for key in wide:
        row = results[key]
        print(
            f"  {_GROUP_TITLES[key]}：調倉 {row['n_rebalances']} 次，"
            f"換股 {row['n_name_changes']} 次"
            f"（買進 {row['n_buys']}、賣出 {row['n_sells']}），"
            f"平均單邊換手 {row['avg_one_way_turnover'] * 100:.0f}%，"
            f"成本率加總 {_pct(row['total_cost_rate'])}",
            flush=True,
        )


def _print_table(results: dict[str, dict[str, Any]]) -> None:
    """印出三組未扣成本的績效對照表。"""
    header = f"{'組別':<16}{'檔數':>6}{'累積報酬':>10}{'vs SPY 超額':>12}{'Sharpe':>8}{'最大回撤':>10}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for key in _GROUP_ORDER:
        row = results[key]
        print(
            f"{_GROUP_TITLES[key]:<16}"
            f"{row['n_tickers']:>6}"
            f"{_pct(row['cumulative_return']):>10}"
            f"{_pct(row['excess_vs_spy']):>12}"
            f"{_num(row['sharpe']):>8}"
            f"{_pct(row['max_drawdown']):>10}",
            flush=True,
        )


def _assert_realtime_regime_rules() -> None:
    """用一條手算過的價格，確認跌破 8% 才進逆風、創新高才解除。"""
    index = pd.bdate_range("2024-01-02", periods=7)
    prices = pd.Series([100.0, 100.0, 95.0, 91.0, 95.0, 100.0, 92.0], index=index)
    labels = realtime_regime(prices).tolist()
    expected = ["順風", "順風", "順風", "逆風", "逆風", "順風", "逆風"]
    if labels != expected:
        raise AssertionError(f"即時逆風規則與手算不符：{labels}")
    still_under = realtime_regime(prices.iloc[:5])
    if str(still_under.iloc[-1]) != "逆風":
        raise AssertionError("價格回到高點 5% 以內就解除逆風，與規則不符")


def _annualized_trades(result: dict[str, Any]) -> float:
    """年化交易次數。買進與賣出各算一次，除以持有區間的年數。"""
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
    比較組 A 月度固定規則的基準版與即時逆風防禦版。

    基準版每個月底照常選前 3 名。防禦版在換股日收盤已處於即時逆風時，
    該月改持有 SPY，交易成本仍按權重變動計算。
    截斷檢查沒過就不跑防禦版，避免把看得到未來的標籤當成可執行規則。

    參數:
        tickers: 組 A 的候選代號。
        years: 舊版「從今天往回幾年」。None 時用 start_date ~ end_date。
        baseline: 已經跑完的基準回測。有傳入就不再重算，須與這裡的區間相同。
        start_date: 績效起日（含），預設 config.BACKTEST_START。
        end_date: 績效迄日（含），預設 config.BACKTEST_END。

    回傳:
        含 baseline、defensive、mismatches 的字典。
    """
    _assert_realtime_regime_rules()
    window = {"years": years, "start_date": start_date, "end_date": end_date}
    base = baseline if baseline is not None else run_group_backtest(tickers, **window)
    mismatches = verify_realtime_regime(base["spy_close"])
    if mismatches:
        raise RuntimeError("即時逆風截斷檢查不一致，防禦版比較沒有執行")
    regime = realtime_regime(base["spy_close"])
    defensive = run_group_backtest(tickers, defend_regime=regime, **window)
    if (defensive["start"], defensive["end"]) != (base["start"], base["end"]):
        raise RuntimeError(
            f"防禦版區間 {defensive['start']} ~ {defensive['end']} 與基準 "
            f"{base['start']} ~ {base['end']} 不同，無法對照"
        )
    return {
        "baseline": base,
        "defensive": defensive,
        "mismatches": mismatches,
    }


def _window_return(daily: pd.Series, start: str, end: str) -> pd.Series:
    """切出持有區間的日報酬。區間含起日與迄日。"""
    return daily.loc[pd.Timestamp(start) : pd.Timestamp(end)]


def _print_defended_month_gaps(baseline: dict[str, Any], defense: dict[str, Any]) -> None:
    """
    印出防禦版改持 SPY 的每個月：組 A 照常選股報酬、同期 SPY、兩者差距。

    持有區間與回測相同，從換股日的下一個交易日到下一個換股日收盤。
    兩邊都是這段的價格報酬，不把下一次換股的成本算進這個月。

    參數:
        baseline: 全程選股的回測結果，需含 months 與 gross_returns。
        defense: 逆風月改持 SPY 的回測結果，用它決定哪些月份要列。
    """
    stock_by_signal = {month["signal"]: month for month in baseline.get("months", [])}
    gross = baseline.get("gross_returns")
    spy = baseline.get("spy_returns")
    if gross is None or spy is None:
        _log.warning("基準回測沒有日報酬，無法逐月對照")
        return

    defended = [month for month in defense.get("months", []) if month.get("defended")]
    print("\n防禦版改持 SPY 的月份（組 A 照常選股 vs SPY）", flush=True)
    header = f"{'持有期間':<27}{'組 A 選股':<22}{'組 A 報酬':>10}{'SPY 報酬':>10}{'差距':>10}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    stock_factors: list[float] = []
    spy_factors: list[float] = []
    for month in defended:
        stock_month = stock_by_signal.get(month["signal"])
        if stock_month is None:
            _log.warning("基準回測沒有換股日 %s", month["signal"])
            continue
        window = _window_return(gross, stock_month["hold_start"], stock_month["hold_end"])
        spy_window = _window_return(spy, stock_month["hold_start"], stock_month["hold_end"]).reindex(window.index)
        if window.empty or spy_window.isna().all():
            _log.warning("%s 沒有可對照的日報酬", month["signal"])
            continue
        stock_return = _cumulative(window)
        spy_return = _cumulative(spy_window.fillna(0.0))
        names = ",".join(stock_month["picked"]) if stock_month["picked"] else "現金"
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
            f"{f'這 {len(stock_factors)} 段各自複利（不含中間月份）':<49}"
            f"{_pct(stock_total):>10}{_pct(spy_total):>10}{_pct(stock_total - spy_total):>10}",
            flush=True,
        )


def _print_realtime_defense(comparison: dict[str, Any]) -> None:
    """印出基準與即時逆風防禦的扣成本績效，並在沒有改善時如實標註。"""
    base = comparison["baseline"]
    defense = comparison["defensive"]
    window = f"{base['start']} ~ {base['end']}"
    print(f"\n=== 即時逆風防禦（組 A 月度固定，扣成本，{window}）===", flush=True)
    print(
        "換股日收盤價已低於截至當天的 SPY 最高收盤價 8%，該月改持有 SPY，"
        "直到之後某個換股日收盤創新高才恢復選股。",
        flush=True,
    )
    header = f"{'設定':<22}{'總報酬':>10}{'Sharpe':>8}{'最大回撤':>10}{'年化交易次數':>12}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    rows = (
        ("基準（照常選股）", base),
        ("防禦（逆風持有 SPY）", defense),
    )
    for title, row in rows:
        print(
            f"{title:<22}"
            f"{_pct(row['cumulative_return_net']):>10}"
            f"{_num(row['sharpe_net']):>8}"
            f"{_pct(row['max_drawdown_net']):>10}"
            f"{_annualized_trades(row):>12.1f}",
            flush=True,
        )
    dates = defense.get("headwind_signal_dates") or []
    print(
        f"逆風月份數：{defense['n_headwind_signals']}（月底換股日當天已跌破 8%）",
        flush=True,
    )
    if dates:
        print("  逆風換股日：" + ", ".join(str(item) for item in dates), flush=True)
    print(
        f"防禦版實際持有 SPY：{defense['n_defended_months']} 個月",
        flush=True,
    )
    _print_defended_month_gaps(base, defense)
    sharpe_better = float(defense["sharpe_net"]) > float(base["sharpe_net"])
    # 最大回撤是負數，數值變大代表跌得比較淺
    drawdown_better = float(defense["max_drawdown_net"]) > float(base["max_drawdown_net"])
    print(
        f"Sharpe {_num(base['sharpe_net'])} → {_num(defense['sharpe_net'])}，"
        f"最大回撤 {_pct(base['max_drawdown_net'])} → {_pct(defense['max_drawdown_net'])}。",
        flush=True,
    )
    if not sharpe_better or not drawdown_better:
        print("即時版逆風防禦沒有優勢", flush=True)


if __name__ == "__main__":
    groups = define_three_groups()
    print(
        f"回測規則：{BACKTEST_FIRST_REBALANCE_MONTH} 月底第一次換股，"
        f"持有到 {BACKTEST_LAST_MONTH} 月底，共 {BACKTEST_MONTHS} 個完整月。"
        f"日曆邊界 {BACKTEST_START.isoformat()} ~ {BACKTEST_END.isoformat()}，"
        f"組別名單迄日 {GROUP_DEFINITION_END.isoformat()}",
        flush=True,
    )
    print("=== 三組母體 ===", flush=True)
    for key in _GROUP_ORDER:
        print(f"{_GROUP_TITLES[key]}（{len(groups[key])} 檔）", flush=True)
        print("  " + ", ".join(groups[key]), flush=True)
    extra_b = sorted(set(groups["B"]) - set(groups["A"]))
    only_c = sorted(set(groups["C"]) - set(groups["B"]))
    print(f"組 B 比組 A 多出的外國／未歸類大型股（{len(extra_b)} 檔）: {', '.join(extra_b)}", flush=True)
    print(f"只在組 C 的新興公司（{len(only_c)} 檔）: {', '.join(only_c)}", flush=True)
    print(
        "規則：每月最後一個交易日收盤，用當天技術面分數選前 3 名，等權持有到下一個月底。"
        f"未滿 {_MIN_HISTORY_BARS} 個交易日不進候選。"
        f"換股成本為單邊 {_ONE_WAY_COST * 100:.2f}%（手續費 {_COMMISSION_ONE_SIDE * 100:.2f}% + 滑價 {_SLIPPAGE_ONE_SIDE * 100:.2f}%），"
        f"三檔全部換掉合計 {_ROUND_TRIP_COST * 100:.2f}%。Sharpe 的無風險利率為 0。",
        flush=True,
    )

    results: dict[str, dict[str, Any]] = {}
    for key in _GROUP_ORDER:
        print(f"\n=== 回測 {_GROUP_TITLES[key]} ===", flush=True)
        results[key] = run_group_backtest(groups[key])

    spy_return = float(results["A"]["spy_cumulative_return"])
    window = f"{results['A']['start']} ~ {results['A']['end']}"
    print(f"\n=== 未扣成本（{window}，SPY 同期 {_pct(spy_return)}）===", flush=True)
    _print_table(results)
    print("\n=== 交易成本對照 ===", flush=True)
    _print_cost_table(results)
    _print_turnover(results)
    for key in _GROUP_ORDER:
        dropped = results[key]["dropped_tickers"]
        if dropped:
            print(f"{_GROUP_TITLES[key]} 無價量而剔除: {', '.join(dropped)}", flush=True)
    print(_interpret(results), flush=True)

    regime = run_regime_split_backtest(tickers=groups["A"], backtest=results["A"])
    _print_regime(regime)
    print(_interpret_regime(regime), flush=True)

    defense = run_realtime_defense_compare(groups["A"], baseline=results["A"])
    _print_realtime_defense(defense)
