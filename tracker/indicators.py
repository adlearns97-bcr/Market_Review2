"""Classic technical indicators, implemented with plain pandas.

Every function is causal: the value at bar t only uses bars <= t, so nothing here
can leak future prices into a signal or a backtest.
"""
from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, n: int) -> pd.Series:
    """Simple moving average - the plain mean of the last n closes."""
    return series.rolling(n, min_periods=n).mean()


def ema(series: pd.Series, n: int) -> pd.Series:
    """Exponential moving average - recent bars weigh more than old ones."""
    return series.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder). 0-100: size of recent up-moves vs down-moves."""
    delta = close.diff()
    avg_gain = delta.clip(lower=0.0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    out = 100 - 100 / (1 + avg_gain / avg_loss)  # avg_loss == 0 gives inf -> 100
    return out.mask((avg_gain == 0) & (avg_loss == 0), 50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line (fast EMA - slow EMA), its signal line, and the histogram between them."""
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    """Middle band = SMA(n); outer bands sit k standard deviations above and below it."""
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return mid, mid + k * sd, mid - k * sd


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Average True Range - how far price typically moves in one bar (volatility)."""
    prev = close.shift(1)
    true_range = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
