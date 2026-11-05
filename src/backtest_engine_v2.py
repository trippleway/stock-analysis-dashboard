# -*- coding: utf-8 -*-
"""
事件驅動技術面回測：進場看當天排名，出場靠停利、停損或訊號衰退。

和月度固定持有不同，這裡沒有統一的月底換股日。每個交易日若還有空位，
且有股票進了母體前 n_buy 名，就用收盤價買進。買進之後逐日檢查出場，
持有天數會隨個股走勢改變。

前視偏差的界線：
- 進場排名只用該交易日（含）以前的技術面分數。分數來自 compute_tech_score_series。
- 出場只走訪進場日之後、判斷日（含）以前的收盤價與排名，看到條件成立就停，不讀後面的價格。
- 自我檢查會把價量截斷到決策日重算，並把出場重放到「少一天」的資料上。

交易成本沿用月度回測：單邊 0.15%（手續費 0.05% + 滑價 0.10%），
買進扣一次、賣出再扣一次。仍在持有的部位只扣了買進。

停損只在收盤價觸及時才賣，不看日內低點。收盤價可以跳過設定門檻，
實際虧損會大於 stop_loss_pct。基準組停損設在 -10%，實測最差一筆約 -23%。
這是回測相對實盤的已知樂觀偏差：參數看起來把虧損卡在設定值，
成交結果並沒有。報告不要把停損寫成精準落在設定值。

10 年回測改用時點 S&P 500：每個交易日只能買當時成分股，
不能把今天的固定名單套回 2015。大量價量下載前會先印估算並停住，
確認後才呼叫 run_ten_year_baseline(confirm=True)。

執行方式（於專案根目錄）：
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

# 分數重算容差，與 backtest_signals.verify_no_lookahead 相同
_SCORE_ATOL = 1e-6
# 10 年時點回測的績效起點。暖機仍往前多抓，讓 MA200 在第一天已經有效。
_TEN_YEAR_START = date(2015, 1, 1)
_OHLCV_DIR = Path(DATA_DIR) / "backtest_cache" / "ohlcv"
# 未命中快取時，每檔大約是 0.7 秒間隔加上一次 Yahoo 請求。失敗會再試，所以上限抓寬。
_FETCH_SEC_PER_NEW_SYMBOL = 2.0
_FETCH_SEC_PER_NEW_SYMBOL_SLOW = 4.0
# 逆風段 Sharpe 至少高出月度版這個幅度，才說出場規則「明顯改善」
_SHARPE_IMPROVEMENT = 0.20
_EXIT_LABELS = {
    "take_profit": "停利",
    "stop_loss": "停損",
    "signal_decay": "訊號衰退",
    "still_holding": "仍持有",
}
# 一次只改一個出場旋鈕，D 同時拉長衰退天數並放寬停損。decay_rank 全程維持 15。
_SENSITIVITY_CASES: tuple[dict[str, Any], ...] = (
    {
        "label": "基準",
        "note": "停利 25%、停損 -10%、衰退 5 天、前 15 名",
        "take_profit_pct": 0.25,
        "stop_loss_pct": -0.10,
        "score_decay_days": 5,
        "score_decay_rank": 15,
    },
    {
        "label": "A",
        "note": "衰退容忍改 10 天",
        "take_profit_pct": 0.25,
        "stop_loss_pct": -0.10,
        "score_decay_days": 10,
        "score_decay_rank": 15,
    },
    {
        "label": "B",
        "note": "停損放寬到 -15%",
        "take_profit_pct": 0.25,
        "stop_loss_pct": -0.15,
        "score_decay_days": 5,
        "score_decay_rank": 15,
    },
    {
        "label": "C",
        "note": "停利縮小到 15%",
        "take_profit_pct": 0.15,
        "stop_loss_pct": -0.10,
        "score_decay_days": 5,
        "score_decay_rank": 15,
    },
    {
        "label": "D",
        "note": "衰退 10 天且停損 -15%",
        "take_profit_pct": 0.25,
        "stop_loss_pct": -0.15,
        "score_decay_days": 10,
        "score_decay_rank": 15,
    },
)


def _as_stamp(value: date | pd.Timestamp | str) -> pd.Timestamp:
    """收成無時區、午夜的時間戳。"""
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp.normalize()


def _as_series(values: pd.Series, name: str) -> pd.Series:
    """把價格或分數收成由早到晚、無重複日期的 float 序列。"""
    if values is None:
        raise ValueError(f"{name} 不能是空的")
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
    從進場日的下一個交易日起，逐日決定要不要出場。

    當天報酬是收盤價相對進場價的累積漲跌，不是單日漲跌。
    停利、停損都只用收盤價，不看日內高低。收盤可以跳過停損門檻，
    實際虧損會大於 stop_loss_pct，不要把設定值當成實際成交虧損。
    三個條件同一天都成立時，
    優先順序是停利、停損、訊號衰退。

    迴圈只走到呼叫端給的最後一筆資料。把序列截斷在出場日，
    結果會和完整序列相同；少給一天，就不會在原來的出場日出場。
    rank_series 是母體當天的名次（1 為最高分），由呼叫端用當天橫斷面算好。
    連續 score_decay_days 個交易日名次低於前 score_decay_rank 名才算出場。
    進場前的名次不計入這段連續天數。

    參數:
        entry_date: 進場日。當天收盤買進，當天本身不做出場判斷。
        entry_price: 進場收盤價。
        entry_score: 進場當天的技術面分數，只用於和 score_series 對帳。
        price_series: 收盤價。可以含進場日之前的歷史，函式不會拿來判斷出場。
        score_series: 這檔的技術面分數。
        take_profit_pct: 相對進場價上漲達到這個比例就停利。
        stop_loss_pct: 相對進場價下跌達到這個比例就停損，應為負數。
        score_decay_days: 名次連續落後幾天就出場。
        score_decay_rank: 前幾名仍算訊號有效。名次大於這個數字才算跌出。
        rank_series: 這檔每天在母體裡的名次。None 時不檢查訊號衰退。

    回傳:
        exit_date、exit_price、holding_days（進場到出場的日曆天）、
        exit_reason、return_pct（價差報酬，尚未扣交易成本）。
    """
    if take_profit_pct <= 0:
        raise ValueError(f"take_profit_pct 必須是正數，收到 {take_profit_pct}")
    if stop_loss_pct >= 0:
        raise ValueError(f"stop_loss_pct 必須是負數，收到 {stop_loss_pct}")
    if score_decay_days < 1 or score_decay_rank < 1:
        raise ValueError("訊號衰退的天數與名次都必須至少為 1")
    if not math.isfinite(entry_price) or entry_price <= 0:
        raise ValueError(f"entry_price 必須是正數，收到 {entry_price}")

    prices = _as_series(price_series, "price_series")
    scores = _as_series(score_series, "score_series")
    ranks = _as_series(rank_series, "rank_series") if rank_series is not None else None
    entry_ts = _as_stamp(entry_date)

    if entry_ts in scores.index and math.isfinite(entry_score):
        observed = float(scores.loc[entry_ts])
        if math.isfinite(observed) and abs(observed - float(entry_score)) > 1e-4:
            _log.warning(
                "%s 進場分數 %.4f 和分數序列上的 %.4f 不一致",
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
            # 當天沒有成交價，不能用後一天補，連續衰退也中斷
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
    """把單一股票的 OHLCV 收成無時區、由早到晚的表。"""
    history = frame.copy()
    history.index = _as_timestamp_index(history.index)
    return history[~history.index.duplicated(keep="last")].sort_index()


def _top_names(score_row: pd.Series, n_buy: int) -> list[str]:
    """當天分數由高到低取前 n_buy 名。同分用代號字母序，結果可重現。"""
    ranked = score_row.dropna()
    if ranked.empty:
        return []
    table = ranked.rename("score").reset_index()
    table.columns = ["ticker", "score"]
    table = table.sort_values(["score", "ticker"], ascending=[False, True])
    return table["ticker"].head(n_buy).astype(str).tolist()


def _headwind_dates(spy_close: pd.Series) -> set[pd.Timestamp]:
    """
    用既有的 8% 回調定義，收集所有逆風交易日。

    參數:
        spy_close: SPY 收盤價，可含回測起點之前的暖機。

    回傳:
        落在任一段回調區間內的日期。
    """
    dates: set[pd.Timestamp] = set()
    for episode in _spy_drawdown_episodes(spy_close, _DRAWDOWN_THRESHOLD):
        start = _as_stamp(episode["start"])
        end = _as_stamp(episode["end"])
        stamps = pd.date_range(start, end, freq="D")
        dates.update(pd.Timestamp(item).normalize() for item in stamps)
    return dates


def _mark(positions: dict[str, dict[str, Any]], close_row: pd.Series) -> float:
    """用當天收盤價計算持倉市值。缺價格時沿用上一次成交價。"""
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
    只複利「進場日屬於這個環境」的部位，做成日報酬。

    新資金在進場日注入，不計成報酬。買進成本記在進場日，
    賣出成本記在出場日。出場後的現金離開這個子組合，避免空窗把 Sharpe 稀釋成 0。
    同一天先處理出場、再處理進場。

    參數:
        trades: 已完成模擬的交易，需含 regime、allocated、shares。
        close: 收盤價面板。
        regime: 順風段或逆風段。

    回傳:
        這個環境有資金在場的那些交易日的日報酬。
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
            # 隔夜部位的終值含賣出成本；新資金的終值含買進成本
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
    彙總某個進場環境的筆數、扣成本報酬、同期 SPY 與 Sharpe。

    參數:
        trades: 全部交易。
        close: 收盤價面板。
        spy_ret: 與 close 對齊的 SPY 日報酬。
        regime: 順風段或逆風段。

    回傳:
        含 n_trades、strategy_return、spy_return、excess、sharpe、n_days 的字典。
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
    檢查整條交易鏈沒有用到決策日之後的資料。

    進場：每一筆交易把該股價量截斷到進場日（含）重算分數，必須等於當時用來下單的分數。
    這和 verify_no_lookahead 一樣，比的是「完整樣本的第 t 天」對「只留到第 t 天」。
    另抽若干進場日，把母體每一檔都截斷後重排，前 n_buy 名必須和回測當時相同。

    出場：用截斷到出場日的價格與名次重放，出場日、價格、原因必須相同。
    再少給一個交易日，不得仍在原來的出場日出場。

    參數:
        trades: run_event_driven_backtest 產生的交易。
        frames: 每檔的 OHLCV，用來把分數截斷重算。
        close: 回測實際使用的收盤價面板，出場重放沿用這條日曆。
        score: 回測使用的分數面板。
        ranks: 回測使用的名次面板。
        top_by_day: 每個交易日實際採用的前 n_buy 名。
        n_buy: 進場看前幾名。
        take_profit_pct, stop_loss_pct, score_decay_days, score_decay_rank: 與回測相同的出場參數。
        max_entry_dates: 橫斷面重排最多抽幾天。
        max_decay_dates: 訊號衰退出場最多抽幾天做全母體重排。

    回傳:
        True 表示全部通過。
    """
    if not trades:
        _log.warning("沒有交易可做前視偏差檢查")
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
            _log.warning("%s 截斷到 %s 重算分數失敗: %s", ticker, day.date(), exc)
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
                f"{ticker} {entry_ts.date()} 進場分數 回測={used:.4f} 截斷重算={recomputed:.4f}"
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
                f"{ticker} 出場重放不一致：回測 {trade['exit_date']} {trade['exit_reason']}，"
                f"重放 {replay['exit_date']} {replay['exit_reason']}"
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
                        f"{ticker} 只給到進場日就出現出場 {blocked['exit_reason']}"
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
                f"{ticker} 少給一天仍在 {trade['exit_date']} 以 {early['exit_reason']} 出場"
            )
        if trade["exit_reason"] != "still_holding" and early["exit_reason"] != "still_holding":
            failures.append(
                f"{ticker} 出場日前一天已經觸發 {early['exit_reason']}，但回測留到 {trade['exit_date']}"
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
        # 只保留回測當天有排名的股票，避免暖機不足的名字在截斷後突然變合格
        eligible = fresh.where(ranks.loc[day].notna())
        fresh_top = _top_names(eligible, n_buy)
        used_top = top_by_day.get(day, [])
        if fresh_top != used_top:
            failures.append(
                f"{day.date()} 前 {n_buy} 名不一致：回測 {used_top}，截斷重算 {fresh_top}"
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
                    f"{ticker} {day.date()} 衰退出場名次 回測={used_rank} 截斷重算={new_rank}"
                )

    if failures:
        _log.warning("交易鏈前視偏差檢查失敗，共 %s 筆", len(failures))
        for item in failures[:12]:
            _log.warning("  %s", item)
        return False

    _log.info(
        "交易鏈前視偏差檢查通過：%s 筆交易的進場分數與出場重放一致",
        len(trades),
    )
    return True


def _scores_close(left: float, right: float, atol: float = _SCORE_ATOL) -> bool:
    """兩個分數是否相同。兩邊都是 NaN 視為相同。"""
    if not math.isfinite(left) and not math.isfinite(right):
        return True
    if not math.isfinite(left) or not math.isfinite(right):
        return False
    return abs(left - right) <= atol


def _spread_dates(days: list[pd.Timestamp], limit: int) -> list[pd.Timestamp]:
    """從已排序的日期裡均勻抽幾天。天數不足就全部保留。"""
    if limit < 1 or not days:
        return []
    if len(days) <= limit:
        return list(days)
    picks = [round(i * (len(days) - 1) / (limit - 1)) for i in range(limit)]
    return [days[i] for i in picks]


def _yahoo_symbol(ticker: str) -> str:
    """S&P 歷史檔用 BRK.B，Yahoo 報價要用 BRK-B。"""
    return str(ticker).upper().strip().replace(".", "-")


def _in_index_mask(
    universe: dict[date, set[str]],
    index: pd.DatetimeIndex,
    columns: list[str],
) -> pd.DataFrame:
    """
    每個交易日是否屬於該日以前最近一次季度成分股。

    季度取樣日當天起、下一個取樣日之前，沿用這一份名單。
    不在名單裡的欄位整天都是 False，不能因為後來進指數就回溯。

    參數:
        universe: get_point_in_time_sp500 的結果。
        index: 交易日。
        columns: 價量面板上的代號，要和成分股代號同一套寫法。

    回傳:
        與 index、columns 對齊的布林表。
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
    當天可以進候選的分數。不合格的位置是 NaN，不參與排名。

    沿用原本的門檻：要有收盤價、技術面分數，而且已滿最短歷史。
    若給了時點成分股，還必須當天就在那一份名單裡。
    這張表只決定能不能買，不把已經持有的部位提前平掉。

    參數:
        score: 技術面分數。
        ready: 是否已滿最短歷史。
        close: 收盤價。
        membership: 時點成分股。None 代表沿用呼叫端給的固定名單。

    回傳:
        與 score 同形的分數表，不合格為 NaN。
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
    對一份母體跑事件驅動回測：有空位才買，觸發條件才賣。

    每個交易日收盤，用當天技術面分數在合格股票裡取前 n_buy 名。
    還沒持有、而且還有空位的才買進。同時最多持有 max_positions 檔。
    新倉金額是當時淨值的 1/max_positions，現金不夠就用剩下的現金，
    不借錢，也不把已經持有的部位再平衡。

    出場呼叫 simulate_position_exit。交易依進場日屬於順風或逆風歸類，
    不依持有期間橫跨的環境切段。

    若傳入 as_of_universe，候選名單改成該日以前最近一季的 S&P 500 成分股。
    排名也只在這份當時名單裡計算，不會用到尚未進指數、或已經剔除的公司。
    已經持有、後來被剔出指數的部位不強制賣掉，仍等停利、停損或訊號衰退。
    被剔除之後名次是空的，訊號衰退不會再累積，停利和停損照舊。

    參數:
        tickers: 固定候選代號。有 as_of_universe 時可省略，改以下載聯集。
        years: 舊版「從今天往回幾年」。None 時用 config.BACKTEST_START ~ BACKTEST_END；
            有傳入、且沒有另外指定 start 時，才改成隨執行日變動的區間。
        max_positions: 同時最多持有幾檔。
        take_profit_pct: 停利門檻。
        stop_loss_pct: 停損門檻，負數。
        score_decay_days: 名次連續落後幾天出場。
        score_decay_rank: 前幾名仍算訊號有效。時點池約 500 檔時，前 15 名比 34 檔嚴格很多。
        n_buy: 當天分數進前幾名才考慮買進。
        start: 績效起日（含）。None 且沒有傳 years 時用 config.BACKTEST_START。
        end: 績效迄日（含）。None 且沒有傳 years 時用 config.BACKTEST_END。
        as_of_universe: 季度時點成分股。None 則整段都用 tickers。

    回傳:
        扣成本後的日報酬、累積報酬、Sharpe、交易清單、持有天數、
        出場原因、年化單邊成交次數，以及依進場日拆開的順風／逆風績效。
    """
    if max_positions < 1 or n_buy < 1:
        raise ValueError("max_positions 與 n_buy 都必須至少為 1")

    # 明確傳入 years 且沒指定 start，才用「今天往回 N 年」。其餘走固定區間。
    if years is not None and start is None:
        if years < 1:
            raise ValueError(f"years 必須至少 1 年，收到 {years}")
        end_day = end or date.today()
        eval_start = end_day - timedelta(days=365 * years)
    else:
        end_day = end if end is not None else BACKTEST_END
        eval_start = start if start is not None else BACKTEST_START
    if eval_start > end_day:
        raise ValueError(f"回測起日 {eval_start} 晚於迄日 {end_day}")
    fetch_start = eval_start - timedelta(days=_WARMUP_DAYS)
    if as_of_universe is not None:
        symbols = sorted({str(ticker).upper().strip() for names in as_of_universe.values() for ticker in names})
    else:
        if not tickers:
            raise ValueError("沒有固定名單，也沒有時點股票池，無法回測")
        symbols = [str(ticker).upper().strip() for ticker in tickers if str(ticker).strip()]
    yahoo_symbols = sorted({_yahoo_symbol(ticker) for ticker in symbols})

    panel = build_ohlcv_panel([*yahoo_symbols, _BENCHMARK], fetch_start, end_day)
    spy = panel.get(_BENCHMARK)
    if spy is None or spy.empty:
        raise RuntimeError("沒有 SPY 價量，無法劃分順風與逆風")

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
        _log.warning("這些代號沒有價量，已從事件驅動回測剔除: %s", ",".join(dropped))
    if not frames:
        raise RuntimeError("沒有任何可用價量，無法回測")

    close, score, ready = _score_frame(frames, master)
    membership = None
    if as_of_universe is not None:
        membership = _in_index_mask(as_of_universe, close.index, list(close.columns))
        _log.info(
            "時點成分股：%s 個季度快照，價量齊全 %s / %s 檔",
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
        raise RuntimeError(f"{eval_start} ~ {end_day} 之間沒有足夠交易日")

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
                    "%s 應在 %s 出場，模擬日卻是 %s，改以觸發日的結果平倉",
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
                        "regime": "逆風段" if day in headwind else "順風段",
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
        # 模擬結束仍未觸發，覆寫成仍持有，避免最後一天剛好達標時被算兩次
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
        for name in ("順風段", "逆風段")
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
        "事件驅動回測 %s ~ %s：扣成本後 %.2f%%，Sharpe %.2f，交易 %s 筆",
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
    """把持倉和出場結果收成一筆交易紀錄。"""
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
    """用人造價格確認三個出場條件與截斷重放，不碰行情資料。"""
    index = pd.bdate_range("2024-01-01", periods=12)
    flat = pd.Series(100.0, index=index)
    scores = pd.Series(80.0, index=index)
    ranks = pd.Series(3.0, index=index)

    surged = flat.copy()
    surged.iloc[3] = 130.0
    took = simulate_position_exit(index[0], 100.0, 80.0, surged, scores, rank_series=ranks)
    if took["exit_reason"] != "take_profit" or took["exit_date"] != index[3].date():
        raise AssertionError(f"停利判斷錯誤: {took}")

    cut = flat.copy()
    cut.iloc[2] = 88.0
    stopped = simulate_position_exit(index[0], 100.0, 80.0, cut, scores, rank_series=ranks)
    if stopped["exit_reason"] != "stop_loss" or stopped["exit_date"] != index[2].date():
        raise AssertionError(f"停損判斷錯誤: {stopped}")

    weak = ranks.copy()
    weak.iloc[1:6] = 20.0
    decayed = simulate_position_exit(index[0], 100.0, 80.0, flat, scores, rank_series=weak)
    if decayed["exit_reason"] != "signal_decay" or decayed["exit_date"] != index[5].date():
        raise AssertionError(f"訊號衰退判斷錯誤: {decayed}")

    reset = weak.copy()
    reset.iloc[4] = 5.0
    held = simulate_position_exit(index[0], 100.0, 80.0, flat, scores, rank_series=reset)
    if held["exit_reason"] == "signal_decay":
        raise AssertionError(f"名次回到前段仍被判定衰退: {held}")

    early = simulate_position_exit(
        index[0],
        100.0,
        80.0,
        surged.loc[: index[2]],
        scores.loc[: index[2]],
        rank_series=ranks.loc[: index[2]],
    )
    if early["exit_reason"] != "still_holding":
        raise AssertionError(f"尚未碰到停利價就被提前出場: {early}")
    _log.info("出場規則自我檢查通過")


def _pct(value: float) -> str:
    """把比率格式化成百分比。"""
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value * 100:.1f}%"


def _num(value: float) -> str:
    """把 Sharpe 格式化成兩位小數。"""
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:.2f}"


def _pp(value: float) -> str:
    """把超額格式化成百分點。"""
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value * 100:.1f}pp"


def _reason_text(reasons: dict[str, int], total: int) -> str:
    """出場原因的數量與占比。"""
    if total <= 0:
        return "沒有交易"
    parts = []
    for key in ("take_profit", "stop_loss", "signal_decay", "still_holding"):
        count = int(reasons.get(key, 0))
        parts.append(f"{_EXIT_LABELS[key]} {count / total * 100:.0f}%")
    return " / ".join(parts)


def _segment(regime: dict[str, Any], name: str) -> dict[str, Any]:
    """從月度 regime 結果取出順風或逆風那一列。"""
    for row in regime["segments"]:
        if row["name"] == name:
            return row
    raise KeyError(name)


def _verdict(monthly_head: dict[str, Any], event_head: dict[str, Any], event: dict[str, Any]) -> str:
    """
    比較事件驅動與月度固定的逆風段 Sharpe。

    參數:
        monthly_head: 月度固定的逆風段。
        event_head: 事件驅動、進場日落在逆風段的彙總。
        event: 事件驅動全期結果，用來同時報告總報酬，避免只講變好的部分。

    回傳:
        一段結論。
    """
    base = float(monthly_head["sharpe"])
    new = float(event_head["sharpe"])
    total = _pct(float(event["cumulative_return_net"]))
    tail = event["regime"]["順風段"]
    shared = (
        f"事件驅動全期扣成本後報酬 {total}，Sharpe {_num(float(event['sharpe']))}。"
        f"順風段報酬 {_pct(float(tail['strategy_return']))}，Sharpe {_num(float(tail['sharpe']))}。"
        f"逆風段報酬 {_pct(float(event_head['strategy_return']))}，"
        f"同期 SPY {_pct(float(event_head['spy_return']))}，"
        f"超額 {_pp(float(event_head['excess']))}，Sharpe {_num(new)}。"
    )
    if not math.isfinite(new):
        return shared + "逆風段有效交易日不足，Sharpe 無法和 0.18 比較。"
    gap = new - base
    if gap >= _SHARPE_IMPROVEMENT:
        return (
            shared
            + f"逆風段 Sharpe 比月度固定的 {_num(base)} 高出 {gap:.2f}，"
            "停利、停損、訊號衰退這套出場規則確實改善了逆風段的問題。"
        )
    if gap > 0:
        return (
            shared
            + f"逆風段 Sharpe 從 {_num(base)} 升到 {_num(new)}，有提高，"
            f"但差距 {gap:.2f} 小於 {_SHARPE_IMPROVEMENT:.2f}，還不能說已經治好逆風段。"
        )
    return (
        shared
        + f"逆風段 Sharpe {_num(new)} 並沒有高於月度固定的 {_num(base)}。"
        "停利、停損、訊號衰退沒有改善逆風段，這組數字要照實保留。"
    )


def _print_report(
    monthly: dict[str, Any],
    monthly_regime: dict[str, Any],
    event: dict[str, Any],
) -> None:
    """印出月度固定與事件驅動的對照、區段表與年化成交次數。"""
    monthly_tail = _segment(monthly_regime, "順風段")
    monthly_head = _segment(monthly_regime, "逆風段")
    event_head = event["regime"]["逆風段"]
    holds = (
        f"{event['avg_holding_days']:.1f} 天"
        f"（最短 {event['holding_min']} / 中位數 {event['holding_median']:.0f} / 最長 {event['holding_max']}）"
    )
    print("\n=== 月度固定 vs 事件驅動（組 A，扣成本後）===", flush=True)
    print("設定 | 平均持有天數 | 出場原因分布 | 總報酬(扣成本) | Sharpe", flush=True)
    print(
        "月度固定 | 30 天（固定） | N/A | "
        f"{_pct(float(monthly['cumulative_return_net']))} | "
        f"{_num(float(monthly['sharpe_net']))}",
        flush=True,
    )
    print(
        "事件驅動 | "
        f"{holds} | "
        f"{_reason_text(event['exit_reasons'], event['n_entries'])} | "
        f"{_pct(float(event['cumulative_return_net']))} | "
        f"{_num(float(event['sharpe']))}",
        flush=True,
    )
    print(
        f"持有天數分布：最短 {event['holding_min']} 天，"
        f"中位數 {event['holding_median']:.0f} 天，"
        f"最長 {event['holding_max']} 天。"
        f"仍持有 {event['n_still_holding']} 筆，天數算到資料最後一天。",
        flush=True,
    )
    if event["holding_min"] != event["holding_max"]:
        print("最短和最長不同，持有期已不再固定在一個月。", flush=True)
    else:
        print("最短和最長相同，這次持有期仍然沒有散開。", flush=True)
    gross_sharpe = float(monthly["sharpe"])
    print(
        f"月度固定這一列的 Sharpe 是扣成本後 {_num(float(monthly['sharpe_net']))}。"
        f"先前總表印出的 {_num(gross_sharpe)} 是未扣成本 Sharpe。",
        flush=True,
    )

    print("\n=== 進場日環境對照 ===", flush=True)
    print(
        "月度固定沿用原本的切法：交易日本身落在 SPY 回調區間才算逆風。"
        "事件驅動改以進場日歸類整筆交易，持有期若跨到另一個環境，仍算在進場那天的環境。",
        flush=True,
    )
    print("              | 月度固定版 | 事件驅動版", flush=True)
    print(
        f"順風段 Sharpe | {_num(float(monthly_tail['sharpe']))} | "
        f"{_num(float(event['regime']['順風段']['sharpe']))}",
        flush=True,
    )
    print(
        f"逆風段 Sharpe | {_num(float(monthly_head['sharpe']))} | "
        f"{_num(float(event_head['sharpe']))}",
        flush=True,
    )
    print(
        f"逆風段報酬    | {_pct(float(monthly_head['strategy_return']))} | "
        f"{_pct(float(event_head['strategy_return']))}",
        flush=True,
    )
    print(
        f"逆風段 vs SPY | {_pp(float(monthly_head['excess']))} | "
        f"{_pp(float(event_head['excess']))}",
        flush=True,
    )
    print(
        f"事件驅動逆風段 {event_head['n_trades']} 筆進場、"
        f"{event_head['n_days']} 個有部位的交易日；"
        f"順風段 {event['regime']['順風段']['n_trades']} 筆、"
        f"{event['regime']['順風段']['n_days']} 個交易日。",
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
    print("\n=== 年化單邊成交次數 ===", flush=True)
    print(
        "可比指標是一年內買進與賣出的次數合計。月度固定用換股次數，"
        "事件驅動用進場次數加出場次數。仍持有的部位還沒有賣出，不計入出場。",
        flush=True,
    )
    print(
        f"月度固定：{monthly_sides:.1f} 次/年"
        f"（換股 {int(monthly['n_name_changes'])} 次，區間 {monthly_years:.2f} 年）",
        flush=True,
    )
    print(
        f"事件驅動：{float(event['annualized_sides']):.1f} 次/年"
        f"（進場 {event['n_entries']}、出場 {event['n_exits']}、"
        f"仍持有 {event['n_still_holding']}，區間 {float(event['span_years']):.2f} 年）",
        flush=True,
    )
    if event["n_same_day_reentries"]:
        print(
            f"同一天賣出後又買回 {event['n_same_day_reentries']} 次，這些來回都付了單邊成本。",
            flush=True,
        )
    if float(event["annualized_sides"]) < monthly_sides:
        print(
            f"事件驅動一年約 {float(event['annualized_sides']):.0f} 次單邊成交，"
            f"少於月度固定的 {monthly_sides:.0f} 次。",
            flush=True,
        )
    else:
        print(
            f"事件驅動一年約 {float(event['annualized_sides']):.0f} 次單邊成交，"
            f"不少於月度固定的 {monthly_sides:.0f} 次，執行次數並沒有降下來。",
            flush=True,
        )


def _same_exit_params(result: dict[str, Any], case: dict[str, Any]) -> bool:
    """已跑完的回測是否就是這一組出場參數。"""
    return (
        abs(float(result["take_profit_pct"]) - float(case["take_profit_pct"])) < 1e-12
        and abs(float(result["stop_loss_pct"]) - float(case["stop_loss_pct"])) < 1e-12
        and int(result["score_decay_days"]) == int(case["score_decay_days"])
        and int(result["score_decay_rank"]) == int(case["score_decay_rank"])
    )


def _sensitivity_row(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """
    從一次完整回測抽出敏感度表需要的欄位。

    參數:
        case: 參數組合，含 label 與四個出場參數。
        result: run_event_driven_backtest 的回傳，已含進場日 regime split。

    回傳:
        年化單邊成交、順風／逆風 Sharpe、逆風報酬與扣成本總報酬。
    """
    tail = result["regime"]["順風段"]
    head = result["regime"]["逆風段"]
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
    對五組出場參數各跑一次事件驅動回測，含進場日的順風／逆風拆分。

    基準是停利 25%、停損 -10%、連續 5 天跌出前 15 名。
    A 只把衰退天數改成 10。B 只把停損改成 -15%。
    C 只把停利改成 15%。D 同時用衰退 10 天與停損 -15%。
    進場規則、持倉上限與交易成本都不改。

    參數:
        tickers: 候選代號，通常是組 A。
        years: 舊版「從今天往回幾年」。None 時沿用 config 的固定區間；有傳入才覆寫。
        baseline: 已經跑完、且參數與基準相同的回測。有傳入就不再重跑基準。

    回傳:
        五列敏感度結果，順序是基準、A、B、C、D。
    """
    rows: list[dict[str, Any]] = []
    for case in _SENSITIVITY_CASES:
        if baseline is not None and _same_exit_params(baseline, case):
            _log.info("敏感度 %s 沿用已完成的回測", case["label"])
            result = baseline
        else:
            _log.info(
                "敏感度 %s：停利 %.0f%%，停損 %.0f%%，衰退 %s 天，前 %s 名",
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
            _log.warning("敏感度 %s 的交易鏈前視偏差檢查沒有通過", case["label"])
        rows.append(row)
    return rows


def _print_sensitivity(rows: list[dict[str, Any]]) -> None:
    """印出出場參數敏感度對照，並標出收盤價停損可以穿透設定值。"""
    print("\n=== 出場參數敏感度（組 A，扣成本後）===", flush=True)
    for row in rows:
        print(f"{row['label']}：{row['note']}", flush=True)
    print(
        "設定 | 年化單邊成交 | 順風 Sharpe | 逆風 Sharpe | 逆風報酬 | 總報酬(扣成本)",
        flush=True,
    )
    for row in rows:
        if not row["lookahead_ok"]:
            print(f"{row['label']} | 前視偏差檢查未通過，這列不採用 | — | — | — | —", flush=True)
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
        "停損只看收盤價。基準組設定 "
        f"{base['stop_loss_pct'] * 100:.0f}%，"
        f"停損單裡最差的價格報酬是 {_pct(base['worst_stop'])}，"
        "沒有卡在設定值。這是回測相對實盤的已知樂觀偏差，不要把停損寫成剛好落在設定值。",
        flush=True,
    )
    for row in verified[1:]:
        print(
            f"{row['label']} 年化單邊成交 {row['annualized_sides']:.1f}，"
            f"順風 Sharpe {_num(row['tail_sharpe'])}，"
            f"逆風 Sharpe {_num(row['head_sharpe'])}，"
            f"總報酬 {_pct(row['total_return'])}。",
            flush=True,
        )


# D 組（衰退 10 天、停損 -15%）周圍的局部網格。停利與名次門檻維持基準。
_GRID_DECAY_DAYS = (7, 8, 10, 12, 15)
_GRID_STOPS = (-0.12, -0.13, -0.15, -0.17, -0.20)
_GRID_TAKE_PROFIT = 0.25
_GRID_DECAY_RANK = 15
_GRID_CENTER_DAYS = 10
_GRID_CENTER_STOP = -0.15
# 鄰近點和中心差到這個幅度，才算「明顯不同」
_GRID_TURNOVER_GAP = 0.15
_GRID_SHARPE_GAP = 0.15
_GRID_RETURN_GAP = 0.15


def _grid_index(value: float, options: tuple[float, ...] | tuple[int, ...]) -> int:
    """參數在網格軸上的位置。對不到就視為超出網格。"""
    for index, item in enumerate(options):
        if abs(float(item) - float(value)) < 1e-9:
            return index
    return -1


def _grid_distance(left: dict[str, Any], right: dict[str, Any]) -> int:
    """兩個網格點的切比雪夫距離。差一格（含斜鄰）是 1。"""
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
    """是否為 D 組中心：衰退 10 天、停損 -15%。"""
    return (
        int(row["score_decay_days"]) == _GRID_CENTER_DAYS
        and abs(float(row["stop_loss_pct"]) - _GRID_CENTER_STOP) < 1e-9
    )


def _metric_value(row: dict[str, Any], key: str) -> float:
    """讀出一個指標。無效值當成無法參選。"""
    value = float(row[key])
    return value if math.isfinite(value) else float("nan")


def _best_rows(rows: list[dict[str, Any]], key: str, higher: bool) -> list[dict[str, Any]]:
    """找出這個指標的最佳格子。並列時全部保留。"""
    usable = [row for row in rows if row["lookahead_ok"] and math.isfinite(_metric_value(row, key))]
    if not usable:
        return []
    target = max(_metric_value(row, key) for row in usable) if higher else min(
        _metric_value(row, key) for row in usable
    )
    return [row for row in usable if abs(_metric_value(row, key) - target) < 1e-9]


def _neighbor_gaps(center: dict[str, Any], other: dict[str, Any]) -> list[str]:
    """
    列出鄰近點和 D 組差得明顯的指標。

    年化成交相對差超過 15%、Sharpe 絕對差超過 0.15、
    報酬絕對差超過 15 個百分點，才算明顯不同。
    """
    gaps: list[str] = []
    base_turn = float(center["annualized_sides"])
    if base_turn > 0 and abs(float(other["annualized_sides"]) - base_turn) / base_turn > _GRID_TURNOVER_GAP:
        gaps.append("年化成交")
    if abs(float(other["tail_sharpe"]) - float(center["tail_sharpe"])) > _GRID_SHARPE_GAP:
        gaps.append("順風 Sharpe")
    if abs(float(other["head_sharpe"]) - float(center["head_sharpe"])) > _GRID_SHARPE_GAP:
        gaps.append("逆風 Sharpe")
    if abs(float(other["head_return"]) - float(center["head_return"])) > _GRID_RETURN_GAP:
        gaps.append("逆風報酬")
    if abs(float(other["total_return"]) - float(center["total_return"])) > _GRID_RETURN_GAP:
        gaps.append("總報酬")
    return gaps


def run_local_grid_search(
    tickers: list[str],
    years: int | None = None,
    reuse: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    以 D 組為中心，對衰退天數與停損做 5×5 局部網格。

    score_decay_days 取 7、8、10、12、15。
    stop_loss_pct 取 -12%、-13%、-15%、-17%、-20%。
    停利固定 25%，名次門檻固定前 15 名。
    每一格都跑完整的事件驅動回測，含進場日順風／逆風拆分。

    參數:
        tickers: 候選代號，通常是組 A。
        years: 舊版「從今天往回幾年」。None 時沿用 config 的固定區間；有傳入才覆寫。
        reuse: 已經跑完的回測。參數對得上的格子直接沿用，不再重算。

    回傳:
        25 格的指標列，依衰退天數、再依停損由淺到深排列。
    """
    cached = list(reuse or [])
    rows: list[dict[str, Any]] = []
    total = len(_GRID_DECAY_DAYS) * len(_GRID_STOPS)
    done = 0
    for days in _GRID_DECAY_DAYS:
        for stop in _GRID_STOPS:
            done += 1
            case = {
                "label": f"衰退{days}/停損{stop * 100:.0f}%",
                "note": "局部網格",
                "take_profit_pct": _GRID_TAKE_PROFIT,
                "stop_loss_pct": stop,
                "score_decay_days": days,
                "score_decay_rank": _GRID_DECAY_RANK,
            }
            matched = next((item for item in cached if _same_exit_params(item, case)), None)
            if matched is not None:
                _log.info("局部網格 %s/%s 沿用已完成回測：%s", done, total, case["label"])
                result = matched
            else:
                _log.info(
                    "局部網格 %s/%s：衰退 %s 天，停損 %.0f%%",
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
                _log.warning("局部網格 %s 的交易鏈前視偏差檢查沒有通過", case["label"])
            rows.append(row)
    return rows


def _format_grid_cell(row: dict[str, Any] | None, key: str, best: bool) -> str:
    """把一格格式化成固定寬度。最佳格前面加星號。"""
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
    """印出單一指標的 5×5 網格。列是衰退天數，欄是停損。"""
    winners = _best_rows(rows, key, higher)
    winner_keys = {
        (int(item["score_decay_days"]), round(float(item["stop_loss_pct"]), 2))
        for item in winners
    }
    direction = "越高越好" if higher else "越低越好"
    print(f"\n{title}（{direction}；* 為最佳）", flush=True)
    header = f"{'衰退\\\\停損':<12}" + "".join(f"{stop * 100:.0f}%".rjust(8) for stop in _GRID_STOPS)
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
        print(f"{days:>2} 天{mark:<6}" + "".join(cells), flush=True)


def _grid_conclusion(rows: list[dict[str, Any]]) -> str:
    """
    判斷 D 組是落在多個指標的最佳區，還是鄰近點差很多的單點。

    參數:
        rows: 25 格的結果。

    回傳:
        一段結論。最佳點分散且鄰近點明顯不同時，標註穩健性有限。
    """
    center_rows = [row for row in rows if _is_center(row) and row["lookahead_ok"]]
    if not center_rows:
        return "D 組（衰退 10 天、停損 -15%）沒有可用結果，無法判斷是不是局部最優。"
    center = center_rows[0]
    metrics = (
        ("annualized_sides", "低換手", False),
        ("tail_sharpe", "順風 Sharpe", True),
        ("head_sharpe", "逆風 Sharpe", True),
        ("head_return", "逆風報酬", True),
        ("total_return", "總報酬", True),
    )
    lines = ["五個指標的最佳格子："]
    best_points: list[dict[str, Any]] = []
    near_center = 0
    for key, label, higher in metrics:
        winners = _best_rows(rows, key, higher)
        if not winners:
            lines.append(f"  {label}：沒有有效格子")
            continue
        where = "、".join(
            f"衰退 {int(item['score_decay_days'])} 天 / 停損 {float(item['stop_loss_pct']) * 100:.0f}%"
            f"（{_num(item[key]) if key.endswith('sharpe') else (_pct(item[key]) if key != 'annualized_sides' else f'{item[key]:.1f}')}）"
            for item in winners
        )
        lines.append(f"  {label}：{where}")
        best_points.extend(winners)
        if any(_grid_distance(center, item) <= 1 for item in winners):
            near_center += 1

    scattered = False
    if len(best_points) >= 2:
        farthest = max(_grid_distance(left, right) for left in best_points for right in best_points)
        scattered = farthest >= 3
        lines.append(
            f"這些最佳點在網格上的最遠距離是 {farthest} 格"
            f"（0 是同一格，1 是相鄰，4 是對角）。"
        )
    lines.append(f"其中 {near_center} 個指標的最佳點落在 D 組或它的鄰格。")

    neighbors = [
        row for row in rows
        if row["lookahead_ok"] and not _is_center(row) and _grid_distance(center, row) == 1
    ]
    different = []
    for row in neighbors:
        gaps = _neighbor_gaps(center, row)
        if len(gaps) >= 2:
            different.append(
                f"衰退 {int(row['score_decay_days'])} 天 / 停損 {float(row['stop_loss_pct']) * 100:.0f}%"
                f"（{('、'.join(gaps))}）"
            )
    if different:
        lines.append("和 D 組差得明顯的鄰近點：" + "；".join(different) + "。")
    else:
        lines.append("八個鄰近點沒有同時在兩個以上指標和 D 組差出設定的幅度。")

    # 鄰近點明顯不同，或最佳點散到網格兩端且 D 不是多數指標的高地
    fragile = len(different) >= 3 or (scattered and near_center < 3)
    if fragile:
        lines.append("參數選擇對結果敏感，此結論的穩健性有限。")
    elif near_center >= 3:
        lines.append(
            "D 組落在多個指標最佳點的鄰近區域，這支持它是這個小網格裡的局部最優，而不是單獨一格的巧合。"
        )
    else:
        lines.append("最佳點沒有聚在 D 組周圍。D 組在這個網格裡不是多數指標的高地。")
    return "\n".join(lines)


def _print_local_grid(rows: list[dict[str, Any]]) -> None:
    """印出 5×5 網格、各指標最佳點，以及 D 組是否只是單點巧合。"""
    print("\n=== D 組周圍局部網格（停利 25%、前 15 名，扣成本後）===", flush=True)
    print(
        "列是訊號衰退天數，欄是停損。D 標在衰退 10 天那一列、停損 -15% 那一欄。"
        "星號是該指標的最佳格。",
        flush=True,
    )
    _print_grid_matrix(rows, "annualized_sides", "年化單邊成交", higher=False)
    _print_grid_matrix(rows, "tail_sharpe", "順風 Sharpe", higher=True)
    _print_grid_matrix(rows, "head_sharpe", "逆風 Sharpe", higher=True)
    _print_grid_matrix(rows, "head_return", "逆風報酬", higher=True)
    _print_grid_matrix(rows, "total_return", "總報酬", higher=True)
    print("\n最佳點是否聚集：", flush=True)
    print(_grid_conclusion(rows), flush=True)


def _cache_span(symbol: str) -> tuple[date, date] | None:
    """讀本地 parquet 的起迄日。沒有檔案或讀不到則為 None。"""
    path = _OHLCV_DIR / f"{_yahoo_symbol(symbol)}.parquet"
    if not path.is_file():
        return None
    try:
        frame = pd.read_parquet(path, columns=["Close"])
    except Exception as exc:  # noqa: BLE001
        _log.warning("讀取 %s 價量快取失敗，估算時視為需要重抓: %s", symbol, exc)
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
    估算 2015 年起 S&P 500 聯集的下載規模，不抓股價。

    成分股用本地歷史檔的季度時點與事件聯集。
    價量只檢查已經存在的 parquet，缺檔不連 Yahoo。

    參數:
        start: 績效起日。
        end: 績效迄日。None 表示今天。

    回傳:
        聯集檔數、已涵蓋檔數、季度檔數範圍，以及下載與回測的時間區間。
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
    # 用一檔已有快取粗估分數計算時間，再按 10 年列數與檔數放大
    score_sec, rows_cached = _time_one_score(covered)
    rows_10y = 252 * ((end_day - fetch_start).days / 365.25)
    if score_sec and rows_cached:
        score_each = score_sec * max(rows_10y / rows_cached, 1.0)
    else:
        score_each = 0.15
    n_missing = len(missing)
    download_low = n_missing * _FETCH_SEC_PER_NEW_SYMBOL
    download_high = n_missing * _FETCH_SEC_PER_NEW_SYMBOL_SLOW
    # 前視偏差檢查會對抽樣日把聯集裡每一檔截斷重算，這段隨檔數變長
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
    """計時一檔現成快取的技術面分數。回傳（秒數, 列數）。沒有快取則為 (0, 0)。"""
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
        _log.warning("無法用快取估算分數時間: %s", exc)
        return 0.0, 0
    history = _prepare_history(frame)
    started = time.perf_counter()
    compute_tech_score_series(history)
    return max(time.perf_counter() - started, 0.01), len(history)


def _minutes(seconds: float) -> str:
    """把秒數說成大約幾分鐘。"""
    minutes = seconds / 60.0
    if minutes < 1:
        return f"{seconds:.0f} 秒"
    return f"{minutes:.0f} 分鐘"


def _print_fetch_estimate(estimate: dict[str, Any]) -> None:
    """印出 10 年價量抓取的規模與時間，不開始下載。"""
    print("\n=== 10 年 S&P 500 時點股票池：抓取前估算 ===", flush=True)
    print(
        f"績效期間：{estimate['start']} ~ {estimate['end']}。"
        f"價量會從 {estimate['fetch_start']} 開始抓，前面約 {_WARMUP_DAYS} 天只做均線暖機。",
        flush=True,
    )
    print(
        f"季度時點 {estimate['n_quarters']} 個，"
        f"單季成分股 {estimate['quarter_min']}–{estimate['quarter_max']} 檔"
        f"（中位數 {estimate['quarter_median']}）。",
        flush=True,
    )
    print(
        f"2015 以來至少進過一次指數的聯集：{estimate['n_union']} 檔。"
        f"其中代號含小數點、下載時會改成 Yahoo 的連字號：{estimate['n_dotted']} 檔。",
        flush=True,
    )
    print(
        f"本地快取已涵蓋整段區間：{estimate['n_covered']} 檔。"
        f"有檔但沒有涵蓋到 2015 前的暖機：{estimate['n_partial']} 檔，這些仍要重抓並合併。"
        f"還需要下載：{estimate['n_missing']} 檔（另加 SPY，若 SPY 也缺）。",
        flush=True,
    )
    print(
        "預期下載時間："
        f"{_minutes(estimate['download_low_sec'])}–{_minutes(estimate['download_high_sec'])}"
        "（每檔約 2–4 秒，含 0.7 秒間隔；失敗會再試 3 次，所以可能更久）。",
        flush=True,
    )
    print(
        "下載完成後，基準參數（停利 25%、停損 -10%、衰退 5 天、前 15 名）"
        f"再跑一次 10 年回測，大約還要 {_minutes(estimate['backtest_sec'])}。"
        "前視偏差檢查會隨聯集檔數變長。",
        flush=True,
    )
    print(
        "排名是在當季大約 500 檔裡取前 15 名，不是先前那 34 檔裡的前 15 名。"
        "同一個 decay_rank=15 會嚴格很多。",
        flush=True,
    )
    print(
        "已下市或改過代號的公司，Yahoo 可能沒有完整日線，實際進得了回測的檔數會少於聯集。"
        "這仍是用當時成分股，而不是把今天的 34 檔套回 2015。",
        flush=True,
    )


def _print_ten_year_baseline(result: dict[str, Any]) -> None:
    """印出 10 年基準回測的筆數、區段 Sharpe，以及 SPY 回調次數。"""
    head = result["regime"]["逆風段"]
    tail = result["regime"]["順風段"]
    episodes = result.get("drawdown_episodes") or []
    print("\n=== 10 年基準參數（時點 S&P 500，扣成本後）===", flush=True)
    print(
        f"期間 {result['start']} ~ {result['end']}。"
        f"聯集 {result['n_universe_symbols']} 檔，有價量 {result['n_with_prices']} 檔。",
        flush=True,
    )
    print(f"總交易筆數：{result['n_entries']}（3 年、34 檔那次是 186 筆）", flush=True)
    print(
        f"順風 Sharpe {_num(float(tail['sharpe']))}，"
        f"逆風 Sharpe {_num(float(head['sharpe']))}，"
        f"逆風報酬 {_pct(float(head['strategy_return']))}，"
        f"總報酬 {_pct(float(result['cumulative_return_net']))}。",
        flush=True,
    )
    print(f"SPY 從高點回檔超過 8% 的區間：{len(episodes)} 次。", flush=True)
    for episode in episodes:
        print(
            f"  {episode['start']} ~ {episode['end']}，"
            f"高點 {episode['peak_date']}，跌幅 {episode['depth'] * 100:.1f}%",
            flush=True,
        )
    if len(episodes) > 4:
        print(
            f"這 {len(episodes)} 次多於 2023-10 之後的 4 次，10 年窗口有蓋到更早的回調週期。",
            flush=True,
        )
    else:
        print(
            f"這段只數到 {len(episodes)} 次超過 8% 的回調，沒有明顯多於 2023 之後的 4 次。",
            flush=True,
        )


def run_ten_year_baseline(confirm: bool = False) -> dict[str, Any]:
    """
    用基準出場參數跑 2015 至今的時點 S&P 500 回測。

    預設只估算下載規模並印出，不抓股價、不跑回測。
    確認要繼續時才傳 confirm=True。

    參數:
        confirm: True 才會下載聯集價量並跑基準參數。

    回傳:
        estimate 是估算。ran 為 False 表示這次沒有跑回測。
    """
    estimate = estimate_ten_year_fetch()
    _print_fetch_estimate(estimate)
    if not confirm:
        print(
            "這次沒有下載，也沒有跑 10 年回測。"
            "確認要繼續時，再呼叫 run_ten_year_baseline(confirm=True)。",
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
        print("10 年回測的交易鏈前視偏差檢查沒有通過，下面的績效不採用。", flush=True)
        return {"ran": True, "estimate": estimate, "backtest": result}
    _print_ten_year_baseline(result)
    return {"ran": True, "estimate": estimate, "backtest": result}


if __name__ == "__main__":
    if "--estimate-10y" in sys.argv:
        run_ten_year_baseline(confirm=False)
        raise SystemExit(0)
    _assert_exit_rules()
    group_a = define_three_groups()["A"]
    print(f"組 A（{len(group_a)} 檔）: {', '.join(group_a)}", flush=True)
    print(
        "事件驅動：每天收盤看前 3 名，最多同時持有 5 檔，等權且不再平衡。"
        "停利 +25%，停損 -10%，連續 5 天跌出前 15 名則出場。"
        f"單邊成本 {_ONE_WAY_COST * 100:.2f}%。",
        flush=True,
    )
    monthly = run_group_backtest(group_a)
    monthly_regime = run_regime_split_backtest(tickers=group_a, backtest=monthly)
    event = run_event_driven_backtest(group_a)
    if not event["lookahead_ok"]:
        print("交易鏈前視偏差檢查沒有通過，下面不把績效當成有效結果。", flush=True)
        raise SystemExit(1)
    print(
        "前視偏差檢查通過：進場分數用截斷到進場日的價量重算後與完整資料一致；"
        "出場只用進場日到出場日的資料重放，少一天就不會在原出場日出場。",
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
