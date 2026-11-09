# ============================================================
# Stock Analysis Dashboard - Cross-Tool Rules
# ============================================================

## Project Vision
A US-equity AI-sector analysis system that tracks a six-layer AI
commercialisation value chain, produces a daily Top 3 ranking, and logs
realised performance so the strategy can be validated rather than assumed.

## Environment
- Python 3.13, macOS with Anaconda
- `python` points at Anaconda (all packages installed). Never use `python3`.
- Use `python` in shebangs, subprocess calls, and documentation.
- Installed: yfinance, pandas, numpy, streamlit, plotly, prophet, ta,
  anthropic, transformers, torch, tqdm, python-dotenv, pyarrow, requests

## Project Structure
- config.py                      Universe, weights, fixed backtest window
- src/                           Core modules
  - data_collector.py            yfinance ingestion with caching and retries
  - indicators.py                Technical indicators and sub-scores
  - fundamentals.py              Fundamental scoring
  - valuation.py                 Valuation models
  - predictor.py                 Prophet 30-day forecast
  - news_fetcher.py              News retrieval
  - sentiment.py                 FinBERT sentiment + Claude summaries
  - profile_fetcher.py           Company profiles
  - entry_price.py               Entry-level calculation
  - daily_picker.py              Composite scoring and daily selection
  - track_record.py              Recommendation logging and return backfill
  - backtest_*.py                Backtest data, universes, and engines
  - factor_research*.py          Factor IC research
- dashboard/
  - app.py                       Streamlit UI
- data/
  - cache/YYYY-MM-DD/            Daily cache
  - daily_picks/YYYY-MM-DD.json  Daily selection output
  - track_record/                Performance tracking data

## Language
- English for everything: code, comments, docstrings, log messages, UI text,
  and commit messages. No exceptions — this repository is public.

## Code Style Conventions
- Every function needs an English docstring covering purpose, parameters,
  and return value.
- Add inline English comments for non-obvious logic.
- Guard external calls with try/except; log a warning on failure without
  aborting the main flow.
- Every module ends with an `if __name__ == '__main__':` test block.
- Use the `logging` module, not `print`, for real logging.
- Use Python 3.10+ type syntax (e.g. `str | None`).

## API and network usage
- Sleep 0.5-1s between yfinance calls to avoid 429 errors.
- Use the local daily cache (CACHE_DIR / YYYY-MM-DD / *.parquet or *.json).
- Read ANTHROPIC_API_KEY from config.py (loaded from .env).
- Call Claude with model="claude-haiku-4-5-20251001" (cheap and fast).
- Retry failed network requests 3 times with increasing delay.

## Stock universe (60 tickers, six-layer AI value chain)
Layer 1 Infrastructure (15): AI chips, memory, servers
  NVDA, AMD, AVGO, MRVL, ARM, MU, WDC, TSM, ASML, AMAT,
  LRCX, KLAC, SMCI, DELL, VRT
Layer 2 Cloud & compute (8): hyperscale and AI-specialised cloud
  MSFT, GOOGL, AMZN, ORCL, IBM, CRWV, NET, AKAM
Layer 3 Foundation models (5): foundation-model exposure
  META, BIDU, TSLA, AAPL, PLTR
Layer 4 Data platforms (10): warehouses, vector databases, AI platforms
  SNOW, DDOG, MDB, ESTC, AI, S, PATH, GTLB, CFLT, SOUN
Layer 5 AI applications (12): AI SaaS and productivity tools
  CRM, NOW, ADBE, INTU, WDAY, TEAM, HUBS, SHOP, TTD, APP, U, ZM
Layer 6 Consumer & content (10): consumer AI, entertainment, media
  SPOT, NFLX, RBLX, DUOL, DIS, WBD, SONY, PINS, SNAP, RDDT

These English layer names are published in README.md. Keep them identical
everywhere — config.py, the dashboard, and the docs.

## Score weights (in config.SCORE_WEIGHTS)
- Technical: 0.25
- Fundamental: 0.25
- Valuation: 0.20
- Prediction: 0.10
- Volume: 0.10
- News: 0.10

## Filter thresholds (in config.FILTER_THRESHOLDS)
- min_tech_score: 50
- min_fund_score: 40
- min_pred_return: 0.0
- min_volume_ratio: 0.5

## Current status
- Dashboard, scoring pipeline, and track-record logging are complete.
- Backtesting and validation are complete. The headline result (209.9% over
  36 months vs 85.0% for SPY) does not survive its controls: equal-weighting
  the same universe returns 225.7%, the strategy sits at the 65th percentile
  of 1,000 random draws, and rank IC is 0.055 with t = 1.25. On a
  point-in-time S&P 500 cross-section (~486 names/month), no factor reaches
  |t| = 1.5. The conclusion is a null result: the excess return comes from
  sector exposure, not stock selection. See README.md.
- Current work: finding a signal that survives the point-in-time test.

## Research discipline
These constraints are not optional. The project's main finding came from
enforcing them against its own strategy.
1. No look-ahead bias. Any factor must pass a truncation check: the value
   computed from data truncated at month t must equal the value computed
   from the full series at month t.
2. No performance claim without controls. A backtest result means nothing
   until it is compared against equal-weight, random-selection, and rank-IC
   benchmarks on the same universe and window.
3. Report null and negative results as prominently as positive ones,
   including in README.md.
4. Backtest windows, universes, and cost assumptions are fixed constants in
   config.py, never derived from the run date.
5. Treat a result that only holds on a small cross-section as unproven until
   it is re-tested on a point-in-time universe.
6. Preserve existing modules; prefer incremental development over rewrites.

## When generating code
- Produce complete, runnable files. No TODOs or placeholders.
- Import project modules as `from src.xxx import yyy` when running from the
  project root, or handle sys.path with relative imports.
- Start each file with `# -*- coding: utf-8 -*-` and a module-level docstring.
- Document the run command as `python src/xxx.py` (not python3).
- Do not commit. Leave review and commits to the user.
