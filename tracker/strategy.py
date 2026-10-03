"""The technical strategy: turn indicators into a single -100..+100 "signal score".

Five components each vote between -1 (bearish) and +1 (bullish):

    trend    price vs. its fast/slow moving averages      (trend following)
    macd     MACD vs. its signal line and zero            (momentum)
    rsi      momentum in the normal range, contrarian at the extremes
    bb       contrarian when price is stretched outside the Bollinger bands
    volume   unusually heavy volume confirms that bar's direction

The score is the weighted average of the votes, scaled to +-100. The strategy
buys when the score crosses up through the buy threshold and sells when it falls
to the sell threshold (see backtest.py for execution details).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import indicators as ind


@dataclass
class StrategyParams:
    sma_fast: int = 50
    sma_slow: int = 200
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    volume_period: int = 20
    level_lookback: int = 20       # bars used for support / resistance
    buy_threshold: float = 35.0
    sell_threshold: float = -20.0
    atr_stop_mult: float = 3.0     # trailing stop distance in ATRs; 0 disables it
    target_r: float = 2.0          # profit target = entry + target_r × (entry − stop), i.e. reward : risk
    w_trend: float = 30.0
    w_macd: float = 25.0
    w_rsi: float = 20.0
    w_bb: float = 15.0
    w_volume: float = 10.0

    def weights(self) -> dict[str, float]:
        return {"trend": self.w_trend, "macd": self.w_macd, "rsi": self.w_rsi,
                "bb": self.w_bb, "volume": self.w_volume}

    def copy(self, **changes) -> "StrategyParams":
        return StrategyParams(**{**asdict(self), **changes})


COMPONENTS = {
    "trend": "Trend (moving averages)",
    "macd": "Momentum (MACD)",
    "rsi": "RSI",
    "bb": "Bollinger bands",
    "volume": "Volume",
}


def add_indicators(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    out = df.copy()
    c = out["Close"]
    out["sma_fast"] = ind.sma(c, p.sma_fast)
    out["sma_slow"] = ind.sma(c, p.sma_slow)
    out["rsi"] = ind.rsi(c, p.rsi_period)
    out["macd"], out["macd_signal"], out["macd_hist"] = ind.macd(c, p.macd_fast, p.macd_slow, p.macd_signal)
    out["bb_mid"], out["bb_upper"], out["bb_lower"] = ind.bollinger(c, p.bb_period, p.bb_std)
    out["atr"] = ind.atr(out["High"], out["Low"], c, p.atr_period)
    out["vol_avg"] = ind.sma(out["Volume"], p.volume_period)
    out["support"] = out["Low"].rolling(p.level_lookback, min_periods=1).min()
    out["resistance"] = out["High"].rolling(p.level_lookback, min_periods=1).max()
    return out


def _weighted_mean(votes: list[pd.Series], weights: list[float]) -> pd.Series:
    """Weighted mean that ignores missing votes (e.g. SMA-200 before 200 bars exist)."""
    num = sum(v.fillna(0) * w for v, w in zip(votes, weights))
    den = sum(v.notna() * w for v, w in zip(votes, weights))
    return (num / den.replace(0, np.nan)).fillna(0.0)


def component_scores(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    """Per-bar component votes (-1..+1) and the composite score (-100..+100)."""
    c = df["Close"]
    sign = lambda s: np.sign(s)  # NaN stays NaN

    trend = _weighted_mean(
        [sign(c - df["sma_slow"]), sign(df["sma_fast"] - df["sma_slow"]), sign(c - df["sma_fast"])],
        [0.40, 0.35, 0.25],
    )

    macd = (0.7 * sign(df["macd"] - df["macd_signal"]) + 0.3 * sign(df["macd"])).fillna(0.0)

    r = df["rsi"]
    rsi = ((r - 50) / 40).clip(-0.5, 0.5)
    rsi = rsi.mask(r <= p.rsi_oversold, 1.0).mask(r >= p.rsi_overbought, -1.0).fillna(0.0)

    width = df["bb_upper"] - df["bb_lower"]
    pct_b = (c - df["bb_lower"]) / width.replace(0, np.nan)
    bb = pd.Series(np.select(
        [pct_b <= 0, pct_b < 0.2, pct_b >= 1, pct_b > 0.8], [1.0, 0.5, -1.0, -0.5], 0.0), index=df.index)

    ratio = df["Volume"] / df["vol_avg"].replace(0, np.nan)
    direction = np.sign(c.diff())
    volume = (direction * (ratio - 1).clip(0, 1)).fillna(0.0)

    out = pd.DataFrame({"s_trend": trend, "s_macd": macd, "s_rsi": rsi, "s_bb": bb, "s_volume": volume,
                        "pct_b": pct_b, "vol_ratio": ratio})
    w = p.weights()
    total = sum(w.values()) or 1.0
    out["score"] = 100 * sum(out[f"s_{k}"] * v for k, v in w.items()) / total
    return out


def score_label(score: float, p: StrategyParams) -> str:
    if score >= p.buy_threshold + 25:
        return "STRONG BUY"
    if score >= p.buy_threshold:
        return "BUY"
    if score <= p.sell_threshold - 25:
        return "STRONG SELL"
    if score <= p.sell_threshold:
        return "SELL"
    return "HOLD"


LABEL_MEANING = {
    "STRONG BUY": "Trend, momentum and the other indicators overwhelmingly point up.",
    "BUY": "The score is above the buy threshold: conditions favour buying.",
    "HOLD": "Mixed signals - neither the buy nor the sell conditions are met.",
    "SELL": "The score is below the sell threshold: conditions favour selling or staying out.",
    "STRONG SELL": "Trend, momentum and the other indicators overwhelmingly point down.",
}


# --------------------------------------------------------------------------- explanations


@dataclass
class ComponentReading:
    key: str
    name: str
    vote: float     # -1..+1
    points: float   # contribution to the composite score
    notes: list[str]


def _fmt(v: float) -> str:
    return f"{v:,.2f}" if abs(v) >= 1 else f"{v:.4f}"


def explain(row: pd.Series, p: StrategyParams) -> list[ComponentReading]:
    """Plain-English reasons behind each component's vote on one bar."""
    w = p.weights()
    total = sum(w.values()) or 1.0
    c = row["Close"]
    readings = []

    notes = []
    if pd.notna(row["sma_slow"]):
        up = c > row["sma_slow"]
        notes.append(f"Price is {'above' if up else 'below'} the {p.sma_slow}-bar average "
                     f"({_fmt(row['sma_slow'])}) → long-term {'uptrend' if up else 'downtrend'}.")
        if pd.notna(row["sma_fast"]):
            gold = row["sma_fast"] > row["sma_slow"]
            notes.append(f"{p.sma_fast}-bar average is {'above' if gold else 'below'} the {p.sma_slow}-bar "
                         f"({'golden-cross' if gold else 'death-cross'} regime).")
    else:
        notes.append(f"Not enough history yet for the {p.sma_slow}-bar average.")
    if pd.notna(row["sma_fast"]):
        up = c > row["sma_fast"]
        notes.append(f"Price is {'above' if up else 'below'} the {p.sma_fast}-bar average "
                     f"→ short-term {'strength' if up else 'weakness'}.")
    readings.append(("trend", notes))

    notes = []
    if pd.notna(row["macd_signal"]):
        rising = row["macd"] > row["macd_signal"]
        notes.append(f"MACD ({row['macd']:.2f}) is {'above' if rising else 'below'} its signal line "
                     f"({row['macd_signal']:.2f}) → momentum is {'improving' if rising else 'fading'}.")
        notes.append(f"MACD is {'above' if row['macd'] > 0 else 'below'} zero → the fast average is "
                     f"{'above' if row['macd'] > 0 else 'below'} the slow one.")
    else:
        notes.append("Not enough history for MACD yet.")
    readings.append(("macd", notes))

    r = row["rsi"]
    if pd.isna(r):
        notes = ["Not enough history for RSI yet."]
    elif r >= p.rsi_overbought:
        notes = [f"RSI {r:.0f} is overbought (≥{p.rsi_overbought:.0f}): the rally may be stretched "
                 "and due a pause - counts against buying."]
    elif r <= p.rsi_oversold:
        notes = [f"RSI {r:.0f} is oversold (≤{p.rsi_oversold:.0f}): selling may be exhausted "
                 "and a bounce is more likely - counts toward buying."]
    else:
        notes = [f"RSI {r:.0f} is in the normal range, leaning {'bullish' if r >= 50 else 'bearish'} "
                 f"(above 50 = buyers have had the upper hand)."]
    readings.append(("rsi", notes))

    b = row["pct_b"]
    if pd.isna(b):
        notes = ["Not enough history for Bollinger bands yet."]
    elif b <= 0:
        notes = [f"Price closed below the lower band (%B {b:.2f}): stretched to the downside, "
                 "snap-backs are common."]
    elif b < 0.2:
        notes = [f"Price is near the lower band (%B {b:.2f}): relatively cheap vs. its recent range."]
    elif b >= 1:
        notes = [f"Price closed above the upper band (%B {b:.2f}): stretched to the upside."]
    elif b > 0.8:
        notes = [f"Price is near the upper band (%B {b:.2f}): relatively expensive vs. its recent range."]
    else:
        notes = [f"Price is inside the bands (%B {b:.2f}) - no stretch either way."]
    readings.append(("bb", notes))

    v = row["vol_ratio"]
    if pd.isna(v):
        notes = ["No volume data for this symbol."]
    elif v >= 1.2:
        up = row["s_volume"] >= 0
        notes = [f"Volume is {v:.1f}× its average on a {'rising' if up else 'falling'} bar → "
                 f"{'buyers' if up else 'sellers'} are acting with conviction."]
    else:
        notes = [f"Volume is {v:.1f}× average - nothing unusual."]
    readings.append(("volume", notes))

    return [ComponentReading(k, COMPONENTS[k], float(row[f"s_{k}"]),
                             100 * float(row[f"s_{k}"]) * w[k] / total, notes)
            for k, notes in readings]


# --------------------------------------------------------------------------- events


def detect_events(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    """Classic textbook indicator events, for the learning log (newest first)."""
    rows = []

    def crosses(a: pd.Series, b) -> tuple[pd.Series, pd.Series]:
        diff = a - b
        prev = diff.shift(1)
        return (prev <= 0) & (diff > 0), (prev >= 0) & (diff < 0)

    def add(mask: pd.Series, event: str, bias: str, detail):
        for ts in df.index[mask.fillna(False).to_numpy(bool)]:
            rows.append((ts, event, bias, detail(df.loc[ts]) if callable(detail) else detail))

    up, down = crosses(df["sma_fast"], df["sma_slow"])
    add(up, "Golden cross", "Bullish", f"{p.sma_fast}-bar average crossed above the {p.sma_slow}-bar")
    add(down, "Death cross", "Bearish", f"{p.sma_fast}-bar average crossed below the {p.sma_slow}-bar")

    up, down = crosses(df["macd"], df["macd_signal"])
    add(up, "MACD bullish cross", "Bullish", lambda r: f"MACD {r['macd']:.2f} crossed above signal")
    add(down, "MACD bearish cross", "Bearish", lambda r: f"MACD {r['macd']:.2f} crossed below signal")

    into_os, out_os = crosses(-df["rsi"], -p.rsi_oversold)
    add(into_os, "RSI oversold", "Bullish", lambda r: f"RSI fell to {r['rsi']:.0f}")
    add(out_os, "RSI leaves oversold", "Bullish", lambda r: f"RSI recovered to {r['rsi']:.0f}")
    into_ob, out_ob = crosses(df["rsi"], p.rsi_overbought)
    add(into_ob, "RSI overbought", "Bearish", lambda r: f"RSI rose to {r['rsi']:.0f}")
    add(out_ob, "RSI leaves overbought", "Bearish", lambda r: f"RSI dropped to {r['rsi']:.0f}")

    add((df["Close"] < df["bb_lower"]) & (df["Close"].shift(1) >= df["bb_lower"].shift(1)),
        "Close below lower band", "Bullish", lambda r: f"Close {_fmt(r['Close'])} < band {_fmt(r['bb_lower'])}")
    add((df["Close"] > df["bb_upper"]) & (df["Close"].shift(1) <= df["bb_upper"].shift(1)),
        "Close above upper band", "Bearish", lambda r: f"Close {_fmt(r['Close'])} > band {_fmt(r['bb_upper'])}")

    spike = df["vol_ratio"] >= 2.0
    rising = df["Close"] >= df["Close"].shift(1)
    add(spike & rising, "Volume spike (up bar)", "Bullish", lambda r: f"{r['vol_ratio']:.1f}× average volume")
    add(spike & ~rising, "Volume spike (down bar)", "Bearish", lambda r: f"{r['vol_ratio']:.1f}× average volume")

    events = pd.DataFrame(rows, columns=["time", "event", "bias", "detail"])
    return events.sort_values("time", ascending=False, kind="stable").reset_index(drop=True)
