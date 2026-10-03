"""Mode 3 - build your own rule-based strategy, then train (optimise) it on history.

Workflow
  1. Build: pick entry rules (ALL ticked rules must be true to buy) and an exit plan: a
     profit target and a stop-loss sized in ATRs, an optional trailing stop, a time limit.
  2. Train: try many parameter combinations on the OLDER part of the history only
     (in-sample / training data) and rank them by the chosen objective.
  3. Test: replay the winner on the most recent part it has never seen (out-of-sample).
     If it only works in training it has learned noise - that is overfitting.
  4. Signal: apply the optimised rules to the latest bar -> buy or not, a target price,
     a stop-loss, and an estimated time to reach the target (see forecast.py).

Execution model (same spirit as backtest.py): rules are checked on a bar's close and a buy
is filled at the next bar's open. Target and stop are touched intraday (high/low); if both
could have happened in the same bar we assume the stop came first (conservative). One trade
per setup: after a trade, the rules must switch off and on again before the next buy.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ind
from .forecast import TargetForecast, first_passage


# --------------------------------------------------------------------------- building blocks


@dataclass(frozen=True)
class ParamSpec:
    key: str
    label: str
    default: float
    choices: tuple          # values the optimiser tries
    lo: float
    hi: float
    step: float = 1.0
    decimals: int = 0
    tip: str = ""


@dataclass(frozen=True)
class RuleSpec:
    key: str
    name: str
    about: str
    params: tuple = ()
    default_on: bool = False


RULES: tuple[RuleSpec, ...] = (
    RuleSpec("trend", "Uptrend filter",
             "Close is above its N-bar moving average - only buy stocks that are already rising.",
             (ParamSpec("trend_len", "SMA length (bars)", 200, (50, 100, 150, 200), 10, 400),), True),
    RuleSpec("ema_cross", "Momentum (EMA cross)",
             "The fast exponential average is above the slow one - short-term momentum points up.",
             (ParamSpec("ema_fast", "Fast EMA", 20, (5, 10, 20), 2, 150),
              ParamSpec("ema_slow", "Slow EMA", 50, (30, 50, 100), 5, 300))),
    RuleSpec("rsi_dip", "Buy the dip (RSI)",
             "RSI(14) is below a level - the price has pulled back or cooled off.",
             (ParamSpec("rsi_max", "RSI below", 40, (30, 35, 40, 45, 50, 55), 5, 95),), True),
    RuleSpec("macd", "MACD turning up",
             "The MACD line is above its signal line - momentum is improving."),
    RuleSpec("bb_dip", "Bollinger stretch",
             "Close is below the lower Bollinger band (20 bars, k standard deviations) - unusually low "
             "versus its recent range.",
             (ParamSpec("bb_k", "Band width (σ)", 2.0, (1.0, 1.5, 2.0, 2.5), 0.5, 4.0, 0.25, 2),)),
    RuleSpec("breakout", "Breakout",
             "Close is above the highest close of the previous N bars - price is breaking out to new highs.",
             (ParamSpec("breakout_len", "Lookback (bars)", 20, (10, 20, 50, 100), 5, 250),)),
    RuleSpec("volume", "Volume confirmation",
             "Volume is at least M × its 20-bar average - big players are active.",
             (ParamSpec("vol_mult", "Volume ×", 1.5, (1.0, 1.25, 1.5, 2.0), 0.5, 5.0, 0.25, 2),)),
)

EXIT_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec("tp_atr", "Target (ATR ×)", 3.0, (1.5, 2.0, 3.0, 4.0, 6.0), 0.5, 20.0, 0.5, 1,
              "Take profit this many ATRs above the entry price."),
    ParamSpec("sl_atr", "Stop-loss (ATR ×)", 2.0, (1.0, 1.5, 2.0, 3.0), 0.5, 10.0, 0.5, 1,
              "Sell if price falls this many ATRs below the entry price."),
    ParamSpec("max_bars", "Time limit (bars)", 30, (10, 20, 40, 60), 2, 250, 1, 0,
              "Sell at the close after this many bars if neither target nor stop was hit."),
    ParamSpec("trailing", "Trailing stop", 0, (0, 1), 0, 1, 1, 0,
              "Ratchet the stop up behind the highest close so gains are protected."),
)

RULE_BY_KEY = {r.key: r for r in RULES}
ALL_PARAMS: dict[str, ParamSpec] = {**{p.key: p for r in RULES for p in r.params}, **{p.key: p for p in EXIT_PARAMS}}
DEFAULT_RULES = tuple(r.key for r in RULES if r.default_on)
DEFAULT_VALUES = {k: float(p.default) for k, p in ALL_PARAMS.items()}

OBJECTIVES = {
    "Sharpe ratio": lambda s: s["sharpe"],
    "Total return": lambda s: s["total_return"],
    "Return ÷ max drawdown": lambda s: s["total_return"] / max(abs(s["max_drawdown"]), 0.02),
    "Profit factor": lambda s: min(s["profit_factor"], 10.0),
    "Win rate": lambda s: s["win_rate"],
}


def fmt_value(key: str, v: float) -> str:
    if key == "trailing":
        return "on" if v else "off"
    p = ALL_PARAMS[key]
    return f"{v:.{p.decimals}f}" if p.decimals else f"{v:.0f}"


# --------------------------------------------------------------------------- features


class FeatureCache:
    """Indicator arrays computed once per parameter value and reused across thousands of trials."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        close = df["Close"]
        self._close = close
        self.o, self.h, self.l, self.c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
        self.atr = ind.atr(df["High"], df["Low"], close, 14).to_numpy(float)
        self.rsi = ind.rsi(close, 14).to_numpy(float)
        macd, sig, _ = ind.macd(close)
        self.macd, self.macd_sig = macd.to_numpy(float), sig.to_numpy(float)
        vol = df["Volume"].astype(float)
        self.vol_ratio = (vol / ind.sma(vol, 20).replace(0, np.nan)).to_numpy(float)
        self.bb_mid = ind.sma(close, 20).to_numpy(float)
        self.bb_sd = close.rolling(20, min_periods=20).std(ddof=0).to_numpy(float)
        self.lists = tuple(a.tolist() for a in (self.o, self.h, self.l, self.c, self.atr))  # fast scalar access
        self._cache: dict = {}

    def _get(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def sma(self, n: int) -> np.ndarray:
        return self._get(("sma", n), lambda: ind.sma(self._close, n).to_numpy(float))

    def ema(self, n: int) -> np.ndarray:
        return self._get(("ema", n), lambda: ind.ema(self._close, n).to_numpy(float))

    def prior_high(self, n: int) -> np.ndarray:
        return self._get(("hh", n), lambda: self._close.rolling(n, min_periods=n).max().shift(1).to_numpy(float))

    def bb_lower(self, k: float) -> np.ndarray:
        return self.bb_mid - k * self.bb_sd

    def condition(self, rule: str, v: dict) -> np.ndarray:
        c = self.c
        with np.errstate(invalid="ignore"):   # NaN (warm-up) compares as False
            if rule == "trend":
                return c > self.sma(int(v["trend_len"]))
            if rule == "ema_cross":
                return self.ema(int(v["ema_fast"])) > self.ema(int(v["ema_slow"]))
            if rule == "rsi_dip":
                return self.rsi < v["rsi_max"]
            if rule == "macd":
                return self.macd > self.macd_sig
            if rule == "bb_dip":
                return c < self.bb_lower(v["bb_k"])
            if rule == "breakout":
                return c > self.prior_high(int(v["breakout_len"]))
            if rule == "volume":
                return self.vol_ratio >= v["vol_mult"]
        raise KeyError(rule)

    def setup(self, rules, v: dict) -> np.ndarray:
        out = np.ones(len(self.c), bool)
        for r in rules:
            out &= self.condition(r, v)
        return out


def _f(x: float) -> str:
    return "n/a" if x != x else (f"{x:,.2f}" if abs(x) >= 1 else f"{x:.4f}")


def describe_conditions(fc: FeatureCache, rules, v: dict, i: int) -> list[tuple[str, bool, str]]:
    """Checklist of each entry rule on bar i: (rule name, passed?, current values)."""
    c = fc.c[i]
    out = []
    for r in rules:
        ok = bool(fc.condition(r, v)[i])
        if r == "trend":
            n = int(v["trend_len"])
            text = f"close {_f(c)} vs {n}-bar average {_f(fc.sma(n)[i])} (needs above)"
        elif r == "ema_cross":
            f, s = int(v["ema_fast"]), int(v["ema_slow"])
            text = f"EMA{f} {_f(fc.ema(f)[i])} vs EMA{s} {_f(fc.ema(s)[i])} (needs fast above)"
        elif r == "rsi_dip":
            text = f"RSI {fc.rsi[i]:.1f} (needs below {v['rsi_max']:.0f})"
        elif r == "macd":
            text = f"MACD {fc.macd[i]:.2f} vs signal {fc.macd_sig[i]:.2f} (needs above)"
        elif r == "bb_dip":
            text = f"close {_f(c)} vs lower band {_f(fc.bb_lower(v['bb_k'])[i])} (needs below)"
        elif r == "breakout":
            n = int(v["breakout_len"])
            text = f"close {_f(c)} vs prior {n}-bar high {_f(fc.prior_high(n)[i])} (needs above)"
        else:
            text = f"volume {fc.vol_ratio[i]:.2f}× average (needs ≥ {v['vol_mult']:.2f}×)"
        out.append((RULE_BY_KEY[r].name, ok, text))
    return out


# --------------------------------------------------------------------------- simulation


@dataclass
class CTrade:
    signal_idx: int
    entry_idx: int
    entry_price: float
    target: float
    stop: float                  # current stop (moves up if trailing)
    initial_stop: float
    exit_idx: int | None = None
    exit_price: float | None = None
    reason: str = ""             # "target" | "stop" | "time" | "" (open)

    def ret(self, fee: float, mark: float | None = None) -> float:
        px = self.exit_price if self.exit_idx is not None else mark
        return (1 - fee) ** 2 * px / self.entry_price - 1


@dataclass
class SimResult:
    start: int
    end: int
    trades: list[CTrade]
    equity: np.ndarray           # account value per bar, starting at 1.0
    in_pos: np.ndarray
    pending: bool                # buy signal fired on the last bar (fills next open)


def simulate(fc: FeatureCache, setup: np.ndarray, v: dict, start: int, end: int, fee: float) -> SimResult:
    o, h, l, c, atr = fc.lists
    su = setup.tolist()
    tp, sl, max_bars, trailing = v["tp_atr"], v["sl_atr"], int(v["max_bars"]), bool(v["trailing"])
    keep = 1.0 - fee
    cash, shares, peak = 1.0, 0.0, 0.0
    pos: CTrade | None = None
    trades: list[CTrade] = []
    signal, armed, pending_last = -1, True, False
    equity = np.empty(end - start)
    in_pos = np.zeros(end - start, bool)

    for i in range(start, end):
        if signal >= 0:                                   # fill yesterday's signal at today's open
            e, a = o[i], atr[signal]
            pos = CTrade(signal, i, e, e + tp * a, e - sl * a, e - sl * a)
            trades.append(pos)
            shares, cash, peak, signal = cash * keep / e, 0.0, e, -1
        if pos is not None:
            ex = None
            if o[i] <= pos.stop:
                ex, why = o[i], "stop"                    # gapped through the stop
            elif o[i] >= pos.target:
                ex, why = o[i], "target"                  # gapped through the target
            elif l[i] <= pos.stop:
                ex, why = pos.stop, "stop"
            elif h[i] >= pos.target:
                ex, why = pos.target, "target"
            elif i - pos.entry_idx + 1 >= max_bars:
                ex, why = c[i], "time"
            if ex is not None:
                pos.exit_idx, pos.exit_price, pos.reason = i, ex, why
                cash, shares, pos = shares * ex * keep, 0.0, None
            elif trailing:
                if c[i] > peak:
                    peak = c[i]
                pos.stop = max(pos.stop, peak - sl * atr[i])
        if not su[i]:
            armed = True
        elif armed and pos is None:
            armed = False
            if i + 1 < end:
                signal = i
            else:
                pending_last = True
        equity[i - start] = cash + shares * c[i]
        in_pos[i - start] = pos is not None
    return SimResult(start, end, trades, equity, in_pos, pending_last)


def sim_stats(fc: FeatureCache, res: SimResult, fee: float, bars_per_year: int = 252) -> dict[str, float]:
    eq = res.equity
    closed = [t for t in res.trades if t.exit_idx is not None]
    rets = np.array([t.ret(fee) for t in closed])
    wins, losses = rets[rets > 0], rets[rets <= 0]
    gross_win, gross_loss = wins.sum(), -losses.sum()
    bar_rets = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.zeros(1)
    sd = bar_rets.std()
    nan = float("nan")
    return {
        "total_return": float(eq[-1] - 1),
        "buy_hold": float((1 - fee) ** 2 * fc.c[res.end - 1] / fc.o[res.start] - 1),
        "trades": len(res.trades),
        "win_rate": len(wins) / len(closed) if closed else nan,
        "avg_trade": float(rets.mean()) if closed else nan,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else (math.inf if gross_win > 0 else nan),
        "max_drawdown": float((eq / np.maximum.accumulate(eq) - 1).min()),
        "sharpe": float(bar_rets.mean() / sd * math.sqrt(bars_per_year)) if sd > 0 else nan,
        "exposure": float(res.in_pos.mean()),
        "target_rate": sum(t.reason == "target" for t in closed) / len(closed) if closed else nan,
        "avg_bars": float(np.mean([t.exit_idx - t.entry_idx + 1 for t in closed])) if closed else nan,
    }


# --------------------------------------------------------------------------- training


@dataclass
class TrainConfig:
    rules: tuple[str, ...] = DEFAULT_RULES
    values: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_VALUES))
    optimise: tuple[str, ...] = tuple(ALL_PARAMS)
    objective: str = "Sharpe ratio"
    iterations: int = 500
    train_frac: float = 0.7
    min_trades: int = 5
    commission_pct: float = 0.1

    @property
    def fee(self) -> float:
        return self.commission_pct / 100

    def active_keys(self) -> list[str]:
        return [p.key for r in RULES if r.key in self.rules for p in r.params] + [p.key for p in EXIT_PARAMS]

    def search_keys(self) -> list[str]:
        return [k for k in self.active_keys() if k in self.optimise]


def space_size(cfg: TrainConfig) -> int:
    return math.prod(len(ALL_PARAMS[k].choices) for k in cfg.search_keys())


def warmup_bars(cfg: TrainConfig) -> int:
    """Bars needed before every indicator any candidate uses is valid."""
    def longest(key: str) -> int:
        vals = [cfg.values[key], *(ALL_PARAMS[key].choices if key in cfg.optimise else ())]
        return int(max(vals))
    need = 40  # MACD (26+9), ATR/RSI (14), Bollinger/volume (20)
    if "trend" in cfg.rules:
        need = max(need, longest("trend_len"))
    if "ema_cross" in cfg.rules:
        need = max(need, longest("ema_slow"))
    if "breakout" in cfg.rules:
        need = max(need, longest("breakout_len") + 1)
    return need


@dataclass
class Trial:
    values: dict[str, float]
    score: float
    train: dict[str, float]
    test: dict[str, float] | None = None


def _candidates(cfg: TrainConfig, seed: int) -> tuple[list[dict], int]:
    keys = cfg.search_keys()
    choices = [ALL_PARAMS[k].choices for k in keys]
    total = math.prod(len(ch) for ch in choices)
    if total <= cfg.iterations:
        combos = list(itertools.product(*choices))
    else:  # random search: sample distinct combinations
        rng = np.random.default_rng(seed)
        picked: set[int] = set()
        while len(picked) < cfg.iterations:
            picked.update(int(x) for x in rng.integers(0, total, cfg.iterations))
        combos = []
        for idx in sorted(picked)[: cfg.iterations]:
            combo = []
            for ch in reversed(choices):
                idx, r = divmod(idx, len(ch))
                combo.append(ch[r])
            combos.append(tuple(reversed(combo)))
    out, seen = [dict(cfg.values)], set()   # the user's own settings always compete too
    for combo in combos:
        vals = {**cfg.values, **dict(zip(keys, map(float, combo)))}
        if "ema_cross" in cfg.rules and vals["ema_fast"] >= vals["ema_slow"]:
            continue
        sig = tuple(sorted(vals.items()))
        if sig not in seen:
            seen.add(sig)
            out.append(vals)
    return out, total


@dataclass
class CurrentSignal:
    action: str                  # "BUY" | "HOLD" | "WAIT" | "NO BUY"
    headline: str
    detail: str
    entry: float                 # entry price (actual if holding, else ~next open)
    target: float
    stop: float
    atr: float
    trade: CTrade | None
    conditions: list[tuple[str, bool, str]]
    bars_left: int               # remaining time limit


@dataclass
class StrategyReport:
    symbol: str
    df: pd.DataFrame
    fc: FeatureCache
    cfg: TrainConfig
    best: dict[str, float]
    start: int
    split: int
    trials: list[Trial]          # leaderboard (best first) with test results
    n_space: int
    n_evaluated: int
    n_valid: int
    seconds: float
    warning: str
    best_train: dict
    best_test: dict
    base_train: dict
    base_test: dict
    full: SimResult              # optimised rules replayed over train + test
    full_stats: dict
    signal: CurrentSignal
    forecast: TargetForecast | None
    history: dict                # how the strategy's own past trades played out

    @property
    def n(self) -> int:
        return len(self.df)


def _split_points(cfg: TrainConfig, n: int) -> tuple[int, int]:
    start = min(warmup_bars(cfg), n // 3)
    if n - start < 150:
        raise ValueError(f"Only {n - start} usable bars after indicator warm-up - choose a longer "
                         "training history or shorter moving averages.")
    return start, start + int((n - start) * cfg.train_frac)


def train_strategy(symbol: str, df: pd.DataFrame, cfg: TrainConfig, progress=None, seed: int = 7) -> StrategyReport:
    if not cfg.rules:
        raise ValueError("Tick at least one entry rule.")
    t0 = time.perf_counter()
    fc = FeatureCache(df)
    n = len(df)
    start, split = _split_points(cfg, n)
    objective = OBJECTIVES[cfg.objective]
    candidates, n_space = _candidates(cfg, seed)
    entry_keys = [p.key for r in RULES if r.key in cfg.rules for p in r.params]
    setups: dict[tuple, np.ndarray] = {}

    def setup_for(vals: dict) -> np.ndarray:
        key = tuple(vals[k] for k in entry_keys)
        if key not in setups:
            setups[key] = fc.setup(cfg.rules, vals)
        return setups[key]

    trials = []
    step = max(len(candidates) // 50, 1)
    for j, vals in enumerate(candidates):
        st = sim_stats(fc, simulate(fc, setup_for(vals), vals, start, split, cfg.fee), cfg.fee)
        score = objective(st) if st["trades"] >= cfg.min_trades else -math.inf
        trials.append(Trial(vals, score if score == score else -math.inf, st))
        if progress and (j % step == 0 or j == len(candidates) - 1):
            progress(j + 1, len(candidates))

    ranked = sorted(trials, key=lambda t: t.score, reverse=True)
    n_valid = sum(math.isfinite(t.score) for t in trials)
    warning = ""
    if n_valid == 0:
        best = dict(cfg.values)
        warning = (f"No combination produced at least {cfg.min_trades} trades in the training period, so your "
                   "own settings are shown. Loosen the rules, lower 'min trades' or use a longer history.")
    else:
        best = ranked[0].values
    board = [t for t in ranked if math.isfinite(t.score)][:20]
    for t in board:
        t.test = sim_stats(fc, simulate(fc, setup_for(t.values), t.values, split, n, cfg.fee), cfg.fee)
    return build_report(symbol, df, cfg, best, board, n_space, len(candidates), n_valid,
                        time.perf_counter() - t0, warning, fc)


def build_report(symbol: str, df: pd.DataFrame, cfg: TrainConfig, best: dict, trials: list[Trial],
                 n_space: int, n_evaluated: int, n_valid: int, seconds: float, warning: str,
                 fc: FeatureCache | None = None) -> StrategyReport:
    """Everything shown on screen for a given set of optimised values (also used for live refreshes)."""
    fc = fc or FeatureCache(df)
    n = len(df)
    start, split = _split_points(cfg, n)
    fee = cfg.fee

    def run(vals, a, b):
        res = simulate(fc, fc.setup(cfg.rules, vals), vals, a, b, fee)
        return res, sim_stats(fc, res, fee)

    full, full_stats = run(best, start, n)
    signal = current_signal(fc, cfg, best, full)
    try:
        horizon = signal.bars_left
        forecast = first_passage(df, float(fc.c[-1]), signal.target, signal.stop, horizon)
    except ValueError:
        forecast = None
    closed = [t for t in full.trades if t.exit_idx is not None]
    hit_bars = [t.exit_idx - t.entry_idx + 1 for t in closed if t.reason == "target"]
    history = {
        "closed": len(closed),
        "targets": len(hit_bars),
        "stops": sum(t.reason == "stop" for t in closed),
        "timeouts": sum(t.reason == "time" for t in closed),
        "hit_bars": hit_bars,
        "median_bars": float(np.median(hit_bars)) if hit_bars else None,
    }
    return StrategyReport(
        symbol=symbol, df=df, fc=fc, cfg=cfg, best=best, start=start, split=split, trials=trials,
        n_space=n_space, n_evaluated=n_evaluated, n_valid=n_valid, seconds=seconds, warning=warning,
        best_train=run(best, start, split)[1], best_test=run(best, split, n)[1],
        base_train=run(cfg.values, start, split)[1], base_test=run(cfg.values, split, n)[1],
        full=full, full_stats=full_stats, signal=signal, forecast=forecast, history=history,
    )


def refresh_report(rep: StrategyReport, df: pd.DataFrame) -> StrategyReport:
    """Re-apply already-optimised values to fresh prices (live mode) without re-training."""
    return build_report(rep.symbol, df, rep.cfg, rep.best, rep.trials, rep.n_space, rep.n_evaluated,
                        rep.n_valid, rep.seconds, rep.warning)


def current_signal(fc: FeatureCache, cfg: TrainConfig, v: dict, full: SimResult) -> CurrentSignal:
    i = len(fc.c) - 1
    c, a = float(fc.c[i]), float(fc.atr[i])
    tp, sl, max_bars = v["tp_atr"], v["sl_atr"], int(v["max_bars"])
    conds = describe_conditions(fc, cfg.rules, v, i)
    last = full.trades[-1] if full.trades else None
    if last is not None and last.exit_idx is None:
        held = i - last.entry_idx + 1
        left = max(max_bars - held, 1)
        return CurrentSignal(
            "HOLD", "In a trade - hold",
            f"The optimised strategy bought {held} bar{'s' if held != 1 else ''} ago. Hold until the price "
            f"hits the target or the stop, or the time limit runs out ({left} bar{'s' if left != 1 else ''} left).",
            last.entry_price, last.target, last.stop, a, last, conds, left)
    target, stop = c + tp * a, c - sl * a
    if full.pending:
        return CurrentSignal(
            "BUY", "Buy signal",
            "Every entry rule turned true on the latest close. The strategy would buy at the next open "
            "with the target and stop below.", c, target, stop, a, None, conds, max_bars)
    if all(ok for _, ok, _ in conds):
        return CurrentSignal(
            "WAIT", "Setup active - no fresh signal",
            "All rules are true, but this setup has already been traded. A new buy needs the rules to "
            "switch off and on again. Levels show what an entry today would use.",
            c, target, stop, a, None, conds, max_bars)
    failing = [name for name, ok, _ in conds if not ok]
    return CurrentSignal(
        "NO BUY", "No buy signal",
        f"Waiting for: {', '.join(failing)}. The levels show what an entry today would use, for reference.",
        c, target, stop, a, None, conds, max_bars)


def overfit_verdict(rep: StrategyReport) -> tuple[str, str]:
    """(tone, text) - does the optimised strategy survive data it has never seen?"""
    tr, te, base = rep.best_train, rep.best_test, rep.base_test
    tail = (f" Test period: strategy {te['total_return']:+.1%} vs buy & hold {te['buy_hold']:+.1%}; "
            f"your own settings made {base['total_return']:+.1%} on the same unseen data.")
    if tr["total_return"] <= 0:
        return "bad", ("No edge found ✗ - even the best combination lost money in the training period. "
                       "These rules don't suit this stock; try different rules or exits." + tail)
    if te["trades"] < 3:
        return "warn", (f"Too few trades to judge - only {te['trades']} in the test period. Try a longer "
                        "history or looser rules." + tail)
    if te["total_return"] <= 0:
        return "bad", ("Fails on unseen data ✗ - it looked good in training but lost money in the test "
                       "period. The optimiser has probably fitted noise (overfitting)." + tail)
    ts, vs = tr["sharpe"], te["sharpe"]
    if ts == ts and vs == vs and vs < 0.5 * ts:
        return "warn", ("Weaker on unseen data - still profitable in the test period, but far less efficient "
                        "than in training. Expect some overfitting." + tail)
    return "good", ("Holds up on unseen data ✓ - test-period results are in line with training. A good "
                    "sign, though never a guarantee." + tail)


def quick_signal(df: pd.DataFrame, cfg: TrainConfig) -> CurrentSignal | None:
    """Apply the rules at the user's own values, without optimising - the screener's fast first pass."""
    try:
        start, _ = _split_points(cfg, len(df))
    except ValueError:
        return None
    fc = FeatureCache(df)
    full = simulate(fc, fc.setup(cfg.rules, cfg.values), cfg.values, start, len(df), cfg.fee)
    return current_signal(fc, cfg, cfg.values, full)
