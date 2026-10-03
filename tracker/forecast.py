"""How long might a price target take - and is the stop likely to be hit first?

Monte Carlo "first-passage" simulation: build thousands of possible future price paths by
re-sampling this stock's real past daily moves (a bootstrap), then count how many paths
touch the target before the stop, and after how many bars. Every sampled day keeps its own
intraday high and low, so a target can be touched intraday exactly as in the backtest.

Caveat: re-sampled history assumes the future behaves like the recent past (same volatility,
same drift). It is an estimate, not a promise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PERCENTILES = (5, 25, 50, 75, 95)


@dataclass
class TargetForecast:
    start: float
    target: float
    stop: float
    horizon: int                 # bars simulated (the strategy's time limit)
    n_paths: int
    p_target: float              # target touched before the stop, within the horizon
    p_stop: float
    p_neither: float
    median_bars: float | None    # among paths that reach the target first
    q25: float | None
    q75: float | None
    baseline_p: float            # same question for a trendless random walk
    hit_times: np.ndarray
    bands: dict[int, np.ndarray]  # percentile -> price path, index 0 = today


def first_passage(df: pd.DataFrame, start: float, target: float, stop: float, horizon: int,
                  lookback: int = 504, n_paths: int = 4000, seed: int = 11) -> TargetForecast:
    horizon = max(int(horizon), 1)
    close, high, low = (df[k].to_numpy(float) for k in ("Close", "High", "Low"))
    with np.errstate(divide="ignore", invalid="ignore"):
        moves = np.column_stack([np.log(close[1:] / close[:-1]),   # close-to-close
                                 np.log(high[1:] / close[1:]),     # how far above the close it traded
                                 np.log(low[1:] / close[1:])])     # how far below
    moves = moves[np.isfinite(moves).all(axis=1)][-lookback:]
    if len(moves) < 20:
        raise ValueError("Not enough price history to simulate future paths.")

    rng = np.random.default_rng(seed)
    pick = moves[rng.integers(0, len(moves), size=(n_paths, horizon))]
    closes = start * np.exp(np.cumsum(pick[:, :, 0], axis=1))
    highs = closes * np.exp(pick[:, :, 1])
    lows = closes * np.exp(pick[:, :, 2])

    never = horizon + 1
    hit_t, hit_s = highs >= target, lows <= stop
    t_first = np.where(hit_t.any(axis=1), hit_t.argmax(axis=1), never)
    s_first = np.where(hit_s.any(axis=1), hit_s.argmax(axis=1), never)
    target_first = t_first < s_first            # same-bar ties count as a stop (conservative)
    stop_first = (s_first <= t_first) & (s_first < never)
    hits = t_first[target_first] + 1.0          # bars until the target is touched

    bands = np.percentile(closes, PERCENTILES, axis=0)
    risk, span = start - stop, target - stop
    return TargetForecast(
        start=start, target=target, stop=stop, horizon=horizon, n_paths=n_paths,
        p_target=float(target_first.mean()), p_stop=float(stop_first.mean()),
        p_neither=float(1 - target_first.mean() - stop_first.mean()),
        median_bars=float(np.median(hits)) if len(hits) else None,
        q25=float(np.percentile(hits, 25)) if len(hits) else None,
        q75=float(np.percentile(hits, 75)) if len(hits) else None,
        baseline_p=float(np.clip(risk / span, 0, 1)) if span > 0 else 0.0,
        hit_times=hits,
        bands={p: np.r_[start, b] for p, b in zip(PERCENTILES, bands)},
    )


def trades_every_day(index: pd.DatetimeIndex) -> bool:
    """Crypto trades 7 days a week; stocks only on weekdays."""
    return bool((index[-60:].dayofweek >= 5).any())


def bar_unit(index: pd.DatetimeIndex) -> str:
    return "days" if trades_every_day(index) else "trading days"


def future_dates(index: pd.DatetimeIndex, bars: int) -> pd.DatetimeIndex:
    """Calendar dates of the next `bars` daily bars after the last one."""
    last = index[-1].normalize()
    if trades_every_day(index):
        return pd.date_range(last + pd.Timedelta(days=1), periods=bars, freq="D")
    return pd.bdate_range(last + pd.offsets.BDay(1), periods=bars)


def date_after(index: pd.DatetimeIndex, bars: float) -> pd.Timestamp:
    k = max(int(round(bars)), 1)
    return future_dates(index, k)[-1]
