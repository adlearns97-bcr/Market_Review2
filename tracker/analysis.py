"""Glue: raw OHLCV -> indicators -> scores -> simulation, plus per-bar snapshots for the UI."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import BacktestResult, Trade, run_backtest
from .data import BARS_PER_YEAR, Timeframe, display_start
from .strategy import ComponentReading, StrategyParams, add_indicators, component_scores, detect_events, explain, score_label


@dataclass
class Snapshot:
    """What the model says as of one bar - using only data up to and including that bar."""
    index: int
    time: pd.Timestamp
    close: float
    score: float
    label: str
    components: list[ComponentReading]
    atr: float
    stop: float                 # suggested stop-loss for a new long entry
    target: float               # suggested profit target: stop distance × params.target_r above the close
    support: float
    resistance: float
    trade: Trade | None         # strategy position held during this bar
    signal: str | None          # "buy"/"sell" if the strategy's signal fired on this bar
    is_latest: bool


@dataclass
class TechnicalAnalysis:
    symbol: str
    timeframe: Timeframe
    params: StrategyParams
    data: pd.DataFrame          # display window: OHLCV + indicators + scores
    backtest: BacktestResult
    events: pd.DataFrame

    def snapshot(self, i: int | None = None) -> Snapshot:
        n = len(self.data)
        i = n - 1 if i is None else int(np.clip(i, 0, n - 1))
        row = self.data.iloc[i]
        p = self.params
        close = float(row["Close"])
        atr = float(row["atr"]) if pd.notna(row["atr"]) else float("nan")
        # Stop-loss: the same distance the simulation's trailing stop starts at. With the ATR stop
        # switched off, fall back to the recent low (support), or 2 ATRs if price is already below it.
        if p.atr_stop_mult > 0:
            stop = close - p.atr_stop_mult * atr
        else:
            stop = float(row["support"]) if row["support"] < close else close - 2 * atr
        risk = close - stop
        return Snapshot(
            index=i,
            time=self.data.index[i],
            close=close,
            score=float(row["score"]),
            label=score_label(float(row["score"]), p),
            components=explain(row, p),
            atr=atr,
            stop=stop,
            target=close + p.target_r * risk if risk > 0 else float("nan"),
            support=float(row["support"]),
            resistance=float(row["resistance"]),
            trade=self.backtest.trade_at(i),
            signal=self.backtest.signals.get(i),
            is_latest=i == n - 1,
        )


def analyze(symbol: str, history: pd.DataFrame, tf: Timeframe, params: StrategyParams,
            capital: float = 10_000.0, commission_pct: float = 0.1) -> TechnicalAnalysis:
    # Indicators are computed on the full download so long averages are warmed up,
    # then everything (chart + simulation) uses only the display window.
    full = add_indicators(history, params)
    full = full.join(component_scores(full, params))
    view = full.iloc[display_start(full.index, tf):]
    if len(view) < 2:
        view = full
    bt = run_backtest(view, params, capital, commission_pct, BARS_PER_YEAR.get(tf.interval, 252))
    return TechnicalAnalysis(symbol, tf, params, view, bt, detect_events(view, params))
