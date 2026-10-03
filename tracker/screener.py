"""Mode 4 - Screener: run the three evaluation methods over many tickers at once.

Two jobs:
  * Analyse a list (a CSV / Excel file, or pasted text): every ticker gets the full treatment.
  * Scan an exchange, as a funnel (the way professional screeners work):
      1. list the exchange's stocks via Yahoo's equity screener, largest companies first;
      2. quick-screen each one on price data only - the Pure Technical score plus your Modified
         Strat rules at your own values (a few milliseconds per stock);
      3. deep-dive the most promising N with all three methods: fundamentals, and a strategy
         optimised for that stock with a target, a stop and a time-to-target estimate.

The overall call blends the three methods' votes (−1 = sell … +1 = buy):
    ① Pure Technical 30%  ·  ② Technical + Fundamentals 35%  ·  ③ Modified Strat 35%
A Modified Strat buy counts for less when its optimised rules failed on unseen data.
"""
from __future__ import annotations

import io
import math
import re
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import TechnicalAnalysis, analyze
from .combined import CombinedVerdict, combine
from .custom_strategy import (CurrentSignal, StrategyReport, TrainConfig, overfit_verdict, quick_signal,
                              train_strategy)
from .data import TIMEFRAME_BY_LABEL, DataError, fetch_daily_direct
from .fundamentals import FundamentalReport, fetch_fundamentals
from .net import YAHOO_LOCK, get_session
from .strategy import StrategyParams, add_indicators, component_scores, score_label

EXCHANGES: dict[str, tuple[str, ...]] = {    # Yahoo exchange codes
    "NASDAQ (US)": ("NMS", "NGM", "NCM"),
    "NYSE (US)": ("NYQ",),
    "NYSE American (US)": ("ASE",),
    "NSE (India)": ("NSI",),
    "BSE (India)": ("BSE",),
    "London (LSE)": ("LSE",),
    "Toronto (TSX)": ("TOR",),
    "XETRA (Germany)": ("GER",),
    "Euronext Paris": ("PAR",),
    "Tokyo (Japan)": ("JPX",),
    "Hong Kong": ("HKG",),
    "Australia (ASX)": ("ASX",),
}
UNIVERSE_SIZES: dict[str, int | None] = {"Largest 100": 100, "Largest 250": 250, "Largest 500": 500,
                                         "Largest 1,000": 1000, "Entire exchange": None}
SUFFIXES = {"None": "", ".NS  (NSE India)": ".NS", ".BO  (BSE India)": ".BO", ".L  (London)": ".L",
            ".TO  (Toronto)": ".TO", ".DE  (XETRA)": ".DE", ".PA  (Paris)": ".PA", ".T  (Tokyo)": ".T",
            ".HK  (Hong Kong)": ".HK", ".AX  (Australia)": ".AX"}

HISTORY = "5y"                                 # daily bars: enough to train and test a strategy
MIN_SCAN_BARS = 320                            # ~15 months: needed for the 200-day average and a real test
TECH_TF = TIMEFRAME_BY_LABEL["1 Year · 1 day"]
WEIGHTS = {"technical": 0.30, "fundamental": 0.35, "strategy": 0.35}
FUND_VOTE = {"STRONG BUY": 1.0, "ACCUMULATE": 0.6, "BUY (MOMENTUM)": 0.5, "SPECULATIVE": 0.0, "HOLD": 0.0,
             "WAIT": -0.1, "AVOID FOR NOW": -0.5, "AVOID": -0.7, "STRONG AVOID": -1.0,
             "BUY (technical only)": 0.5, "HOLD (technical only)": 0.0, "SELL (technical only)": -0.5}
STRAT_VOTE = {"BUY": 1.0, "HOLD": 0.5, "WAIT": 0.3, "NO BUY": -0.3}
TRUST = {"good": 1.0, "warn": 0.7, "bad": 0.4, "": 0.7}  # how much to believe a Modified Strat buy
METHOD_NAMES = {"technical": "① Pure Technical", "fundamental": "② Technical + Fundamentals",
                "strategy": "③ Modified Strat"}

# --------------------------------------------------------------------------- reading ticker lists

_TICKER = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-=&]{0,14}$")
_HEADERS = {"ticker", "tickers", "symbol", "symbols", "code", "codes", "stock", "stocks", "scrip", "scrip code",
            "nse symbol", "bse code", "instrument", "security", "security id", "ticker symbol"}


def looks_like_ticker(value) -> bool:
    return isinstance(value, str) and bool(_TICKER.match(value.strip().upper()))


def _clean(values) -> list[str]:
    out, seen = [], set()
    for v in values:
        if not isinstance(v, str):
            v = "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)
            v = v[:-2] if v.endswith(".0") else v   # numeric codes read from Excel, e.g. 500325.0
        v = v.strip().upper()
        if looks_like_ticker(v) and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def tickers_from_frame(df: pd.DataFrame) -> list[str]:
    """Find the ticker column in a sheet read with header=None."""
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        return []
    header = [str(v).strip().lower() for v in df.iloc[0].tolist()]
    for j, name in enumerate(header):
        if name in _HEADERS:
            return _clean(df.iloc[1:, j].tolist())
    if len(df) == 1:                                   # everything on one line: AAPL, MSFT, ...
        return _clean(df.iloc[0].tolist())
    scores = [sum(looks_like_ticker(str(v)) for v in df.iloc[:, j]) for j in range(df.shape[1])]
    return _clean(df.iloc[:, int(np.argmax(scores))].tolist())


def read_tickers_file(path: str | Path) -> list[str]:
    """Tickers from a .csv / .txt / .xlsx / .xls file (first sheet that contains any)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        for sheet in pd.read_excel(path, sheet_name=None, header=None, dtype=str).values():
            found = tickers_from_frame(sheet)
            if found:
                return found
        return []
    text = path.read_text(encoding="utf-8-sig", errors="ignore")
    delim = next((d for d in (",", ";", "	", "|") if d in text), None)
    if delim is None:                                   # one ticker per line / space separated
        return parse_ticker_text(text)
    try:
        found = tickers_from_frame(pd.read_csv(io.StringIO(text), header=None, dtype=str, sep=delim,
                                               skip_blank_lines=True, on_bad_lines="skip"))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError):
        found = []
    return found or parse_ticker_text(text)


def parse_ticker_text(text: str) -> list[str]:
    return _clean(re.split(r"[\s,;|]+", text))


def apply_suffix(symbols: list[str], suffix: str) -> list[str]:
    """Add an exchange suffix (e.g. .NS) to bare tickers; leaves indices, crypto and suffixed ones alone."""
    if not suffix:
        return symbols
    return _clean([s if ("." in s or s.startswith("^") or "-" in s or "=" in s) else s + suffix for s in symbols])


# --------------------------------------------------------------------------- per-ticker evaluation


@dataclass
class TickerResult:
    symbol: str
    name: str = ""
    currency: str = "USD"
    price: float = math.nan
    change_pct: float = math.nan
    market_cap: float | None = None
    depth: str = "quick"                 # "quick" | "full" | "error"
    error: str = ""
    tech_score: float = math.nan
    tech_label: str = ""
    ta: TechnicalAnalysis | None = None
    fund: FundamentalReport | None = None
    fund_note: str = ""
    combined: CombinedVerdict | None = None
    strat: StrategyReport | None = None
    strat_note: str = ""
    signal: CurrentSignal | None = None  # quick: your rules as set; full: optimised for this stock
    robust: str = ""                     # out-of-sample verdict tone: good / warn / bad
    votes: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    overall: float = math.nan            # −1 … +1
    label: str = ""
    confidence: str = ""
    warning: str = ""
    pre_score: float = math.nan          # quick-screen ranking

    @property
    def forecast(self):
        return self.strat.forecast if self.strat is not None else None

    @property
    def entry(self) -> float:
        """A new buyer's entry: today's close (≈ next open), even if the strategy bought earlier."""
        return self.price

    @property
    def reward_risk(self) -> float:
        s = self.signal
        if s is None or self.price <= s.stop or s.target <= self.price:
            return math.nan
        return (s.target - self.price) / (self.price - s.stop)


def _header(symbol: str, df: pd.DataFrame, meta: dict) -> TickerResult:
    c = df["Close"].to_numpy(float)
    return TickerResult(symbol=symbol, name=meta.get("longName") or meta.get("shortName") or symbol,
                        currency=meta.get("currency") or "USD", price=float(c[-1]),
                        change_pct=float(c[-1] / c[-2] - 1) if len(c) > 1 else math.nan)


def _tech_vote(score: float) -> float:
    return float(np.clip(score / 60, -1, 1))


def quick_evaluate(symbol: str, df: pd.DataFrame, meta: dict, tech: StrategyParams,
                   strat_cfg: TrainConfig) -> TickerResult:
    if len(df) < MIN_SCAN_BARS:
        raise ValueError(f"only {len(df)} days of history")
    r = _header(symbol, df, meta)
    view = df.iloc[-(tech.sma_slow + 400):]
    ind = add_indicators(view, tech)
    r.tech_score = float(component_scores(ind, tech)["score"].iloc[-1])
    r.tech_label = score_label(r.tech_score, tech)
    r.signal = quick_signal(df, strat_cfg)
    strat_vote = STRAT_VOTE[r.signal.action] if r.signal is not None else 0.0
    r.pre_score = 0.5 * _tech_vote(r.tech_score) + 0.5 * strat_vote
    return r


def full_evaluate(symbol: str, df: pd.DataFrame, meta: dict, tech: StrategyParams, strat_cfg: TrainConfig,
                  iterations: int = 200, fund_weight: float = 0.6) -> TickerResult:
    r = _header(symbol, df, meta)
    r.ta = analyze(symbol, df, TECH_TF, tech)
    snap = r.ta.snapshot()
    r.tech_score, r.tech_label = snap.score, snap.label
    try:
        r.fund = fetch_fundamentals(symbol, r.price)
        r.name = r.fund.name or r.name
    except DataError as exc:
        r.fund_note = str(exc)
    r.combined = combine(r.fund, snap.score, tech, fund_weight)
    try:
        r.strat = train_strategy(symbol, df, replace(strat_cfg, iterations=min(strat_cfg.iterations, iterations)))
        r.signal = r.strat.signal
        r.robust = overfit_verdict(r.strat)[0]
    except ValueError as exc:
        r.strat_note = str(exc)
    r.depth = "full"
    if len(df) < MIN_SCAN_BARS:
        r.warning = (f"Only {len(df)} days of trading history - too short for the 200-day average"
                     + ("" if r.strat is not None else " or to train the Modified Strat")
                     + ", so this call is less reliable.")
    decide(r)
    return r


def decide(r: TickerResult) -> None:
    """Blend the three methods' votes into BUY / HOLD / SELL with a confidence level."""
    votes = {"technical": _tech_vote(r.tech_score)}
    weights = {"technical": WEIGHTS["technical"]}
    if r.combined is not None:
        votes["fundamental"] = FUND_VOTE.get(r.combined.label, 0.0)
        # without company fundamentals (ETF, index...) method ② just repeats the technical view
        weights["fundamental"] = WEIGHTS["fundamental"] if r.combined.fund_tier else 0.10
    if r.signal is not None:
        v = STRAT_VOTE[r.signal.action]
        votes["strategy"] = v * TRUST[r.robust] if v > 0 else v
        weights["strategy"] = WEIGHTS["strategy"]
    r.votes, r.weights = votes, weights
    r.overall = sum(votes[k] * weights[k] for k in votes) / sum(weights.values())
    o = r.overall
    r.label = ("STRONG BUY" if o >= 0.6 else "BUY" if o >= 0.3 else
               "STRONG SELL" if o <= -0.55 else "SELL" if o <= -0.25 else "HOLD")
    side = 1 if "BUY" in r.label else -1 if "SELL" in r.label else 0
    agree = sum((1 if v >= 0.2 else -1 if v <= -0.2 else 0) == side for v in votes.values())
    r.confidence = "High" if agree == len(votes) else "Medium" if agree * 2 >= len(votes) else "Low"
    if r.warning or "strategy" not in votes:   # a method is missing or data is thin: never "High"
        r.confidence = {"High": "Medium"}.get(r.confidence, "Low")


def bottom_line(r: TickerResult, unit: str = "trading days") -> str:
    """The plain-English recommendation at the top of each in-depth report."""
    cur, s, f = r.currency, r.signal, r.forecast
    missing = [name for name, ok, _ in s.conditions if not ok] if s is not None else []
    caution = f" ⚠ {r.warning}" if r.warning else ""
    return _bottom_line(r, unit, cur, s, f, missing) + caution


def _bottom_line(r: TickerResult, unit: str, cur: str, s, f, missing: list[str]) -> str:
    if "BUY" in r.label:
        text = f"Consider buying around {fmt_price(r.entry, cur)}."
        if s is not None:
            text += (f" Put a stop-loss at {fmt_price(s.stop, cur)} ({s.stop / r.entry - 1:+.1%}) and aim for "
                     f"{fmt_price(s.target, cur)} ({s.target / r.entry - 1:+.1%}).")
        if f is not None and f.median_bars:
            text += (f" The simulation expects about {f.median_bars:.0f} {unit} to get there, with a "
                     f"{f.p_target:.0%} chance of hitting the target before the stop.")
        if s is not None and s.trade is not None:
            text += (f" The optimised strategy already bought at {fmt_price(s.trade.entry_price, cur)}, so part "
                     "of the move may be gone.")
        if s is not None and s.action == "NO BUY":
            text += (f" Note: the optimised strategy has no entry trigger yet (waiting for "
                     f"{', '.join(missing)}), so buying now is early - consider buying in smaller pieces.")
        if r.robust == "bad":
            text += " Its optimised rules did not hold up on unseen data, so treat the levels as rough guides."
        if s is None:
            text += " (No target or stop: the Modified Strat couldn't be trained on this stock.)"
        return text
    if "SELL" in r.label:
        stop = f" or at least tighten your stop to {fmt_price(s.stop, cur)}" if s is not None else ""
        return (f"If you own it, consider selling{stop}. Not a buy right now: "
                + "; ".join(_negatives(r)) + ".")
    wait = (f"wait for {', '.join(missing)}" if missing else "wait for the methods to line up")
    stop = f" with a protective stop at {fmt_price(s.stop, cur)}" if s is not None else ""
    return f"If you own it, keep holding{stop}. If you don't, {wait} before buying."


def fmt_price(v: float | None, currency: str = "") -> str:
    if v is None or v != v:
        return "—"
    symbol = {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", "JPY": "¥", "HKD": "HK$", "CAD": "C$",
              "AUD": "A$"}.get(currency, f"{currency} " if currency else "")
    return f"{symbol}{v:,.2f}" if abs(v) >= 1 else f"{symbol}{v:.4f}"


def _negatives(r: TickerResult) -> list[str]:
    out = []
    if r.votes.get("technical", 0) <= -0.2:
        out.append(f"the chart is weak (Pure Technical {r.tech_label}, {r.tech_score:+.0f})")
    if r.fund is not None and r.fund.available and r.fund.tier == "Weak":
        out.append(f"fundamentals are weak ({r.fund.overall:.0f}/100)")
    elif r.combined is not None and r.votes.get("fundamental", 0) <= -0.2:
        out.append(f"the fundamental view says {r.combined.label}")
    if r.signal is not None and r.signal.action == "NO BUY":
        out.append("the optimised strategy has no entry")
    return out or ["the three methods lean negative overall"]


# --------------------------------------------------------------------------- batch jobs


@dataclass
class BatchState:
    """Shared between the worker thread and the GUI: results stream in as they finish."""
    cancel: threading.Event = field(default_factory=threading.Event)
    results: list[TickerResult] = field(default_factory=list)   # append-only; a deep-dive re-appends a symbol
    phase: str = ""
    failed: int = 0
    scanned: int = 0


def _pool(fn, items, workers: int, state: BatchState, progress) -> None:
    ex = ThreadPoolExecutor(workers)
    try:
        futures = [ex.submit(fn, it) for it in items]
        for done, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res is not None:
                state.results.append(res)
            if progress:
                progress(done, len(items))
            if state.cancel.is_set():
                break
    finally:
        ex.shutdown(wait=True, cancel_futures=True)


def _error(symbol: str, exc: Exception) -> TickerResult:
    if not isinstance(exc, (DataError, ValueError)):
        traceback.print_exc()
    return TickerResult(symbol, depth="error", error=str(exc) or type(exc).__name__, label="ERROR")


def analyse_list(symbols: list[str], tech: StrategyParams, strat_cfg: TrainConfig, iterations: int,
                 state: BatchState, progress=None) -> BatchState:
    state.phase = f"Analysing {len(symbols)} ticker{'s' if len(symbols) != 1 else ''} with all three methods…"

    def one(sym: str):
        if state.cancel.is_set():
            return None
        try:
            df, meta = fetch_daily_direct(sym, HISTORY)
            return full_evaluate(sym, df, meta, tech, strat_cfg, iterations)
        except Exception as exc:
            state.failed += 1
            return _error(sym, exc)

    _pool(one, symbols, 4, state, progress)
    state.phase = "Done" if not state.cancel.is_set() else "Cancelled"
    return state


def exchange_universe(exchange: str, limit: int | None, state: BatchState | None = None) -> list[dict]:
    """Stocks listed on an exchange, largest market cap first (Yahoo's equity screener, 250 per page)."""
    import yfinance as yf
    from yfinance import EquityQuery

    query = EquityQuery("is-in", ["exchange", *EXCHANGES[exchange]])
    out, seen, offset = [], set(), 0
    while limit is None or len(out) < limit:
        size = 250 if limit is None else min(250, limit - len(out))
        try:
            with YAHOO_LOCK:
                resp = yf.screen(query, offset=offset, size=size, sortField="intradaymarketcap", sortAsc=False,
                                 session=get_session())
        except Exception as exc:
            if out:
                break  # keep what we have
            raise DataError(f"Couldn't list stocks on {exchange}: {exc}") from exc
        quotes = resp.get("quotes") or []
        for q in quotes:
            sym = q.get("symbol")
            name = q.get("longName") or q.get("shortName") or ""
            if "%" in name or " NOTES" in name.upper():
                continue  # exchange-traded bonds / preferred notes, not stocks
            if sym and sym not in seen:
                seen.add(sym)
                out.append({"symbol": sym, "name": q.get("longName") or q.get("shortName") or sym,
                            "marketCap": q.get("marketCap"), "currency": q.get("currency")})
        offset += len(quotes)
        if state is not None:
            state.phase = f"Listing stocks on {exchange}… {len(out):,} found"
            if state.cancel.is_set():
                break
        if not quotes or offset >= resp.get("total", 0):
            break
    return out[:limit] if limit else out


def scan_exchange(exchange: str, limit: int | None, deep_k: int, tech: StrategyParams, strat_cfg: TrainConfig,
                  iterations: int, state: BatchState, progress=None) -> BatchState:
    universe = exchange_universe(exchange, limit, state)
    if state.cancel.is_set():
        state.phase = "Cancelled"
        return state
    caps = {u["symbol"]: u for u in universe}
    state.phase = f"Quick-screening {len(universe):,} stocks on price data…"

    def quick(u: dict):
        if state.cancel.is_set():
            return None
        try:
            df, meta = fetch_daily_direct(u["symbol"], HISTORY)
            r = quick_evaluate(u["symbol"], df, meta, tech, strat_cfg)
        except Exception:
            state.failed += 1
            return None
        r.market_cap, r.name = u.get("marketCap"), u.get("name") or r.name
        state.scanned += 1
        return r

    _pool(quick, universe, 8, state, progress)
    if state.cancel.is_set():
        state.phase = "Cancelled"
        return state

    ranked = sorted((r for r in state.results if r.depth == "quick" and not math.isnan(r.pre_score)),
                    key=lambda r: r.pre_score, reverse=True)
    picks = [r.symbol for r in ranked[:deep_k]]
    pre = {r.symbol: r.pre_score for r in ranked}
    state.phase = f"Deep-dive: all three methods on the {len(picks)} most promising…"

    def deep(sym: str):
        if state.cancel.is_set():
            return None
        try:
            df, meta = fetch_daily_direct(sym, HISTORY)
            r = full_evaluate(sym, df, meta, tech, strat_cfg, iterations)
            r.market_cap, r.pre_score = caps[sym].get("marketCap"), pre[sym]
            return r
        except Exception as exc:
            return _error(sym, exc)

    _pool(deep, picks, 4, state, progress)
    state.phase = "Done" if not state.cancel.is_set() else "Cancelled"
    return state


def latest_by_symbol(results: list[TickerResult]) -> list[TickerResult]:
    """One result per symbol - a deep-dive supersedes the quick screen of the same stock."""
    latest: dict[str, TickerResult] = {}
    for r in results:
        latest[r.symbol] = r
    return list(latest.values())


def results_frame(results: list[TickerResult]) -> pd.DataFrame:
    """Flat table for exporting to CSV / Excel."""
    rows = []
    for r in latest_by_symbol(results):
        f, s = r.forecast, r.signal
        rows.append({
            "Ticker": r.symbol, "Name": r.name, "Currency": r.currency, "Price": r.price,
            "Change %": r.change_pct * 100, "Market cap": r.market_cap,
            "Overall": r.label or ("(quick screen)" if r.depth == "quick" else ""),
            "Score (-100..100)": r.overall * 100, "Confidence": r.confidence,
            "Pure Technical": f"{r.tech_label} ({r.tech_score:+.0f})" if r.tech_label else "",
            "Technical + Fundamentals": r.combined.label if r.combined else "",
            "Fundamental score": r.fund.overall if r.fund is not None and r.fund.available else None,
            "Modified Strat": s.action if s else "", "Out-of-sample": r.robust,
            "Entry": r.entry if s else None, "Target": s.target if s else None, "Stop-loss": s.stop if s else None,
            "Reward:risk": r.reward_risk,
            "Days to target (median)": f.median_bars if f else None,
            "Chance target first %": f.p_target * 100 if f else None,
            "Depth": r.depth, "Error": r.error,
            "Bottom line": bottom_line(r) if r.depth == "full" else "",
        })
    return pd.DataFrame(rows)
