"""Fundamental scorecard: is this a good *business* at a sensible *price*?

Each metric is mapped to a 0-100 score with a piecewise-linear curve (the points
below are rules of thumb for a typical large company, not laws - a bank, a REIT and
a fast-growing software firm all have different "normal" ranges). Metric scores are
averaged into categories, and categories into an overall score.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .data import fetch_info

CATEGORIES = {               # weight in the overall score
    "Valuation": 0.25,
    "Profitability": 0.25,
    "Growth": 0.20,
    "Financial health": 0.20,
    "Market view": 0.10,
    "Dividends & risk": 0.0,  # shown for context, not scored
}


def _num(info: dict, key: str) -> float | None:
    v = info.get(key)
    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
        return float(v)
    return None


def _price(info: dict) -> float | None:
    return _num(info, "currentPrice") or _num(info, "regularMarketPrice") or _num(info, "previousClose")


def _ratio(a: float | None, b: float | None) -> float | None:
    return None if a is None or not b else a / b


def _pe(key: str) -> Callable[[dict], float | None]:
    def get(info: dict) -> float | None:
        v = _num(info, key)
        eps = _num(info, "trailingEps" if key == "trailingPE" else "forwardEps")
        if v is None and eps is not None and eps < 0:
            return -1.0  # Yahoo omits P/E when earnings are negative
        return v
    return get


def _eps_growth(info: dict) -> float | None:
    t, f = _num(info, "trailingEps"), _num(info, "forwardEps")
    return f / t - 1 if t and t > 0 and f is not None else None


def _net_cash(info: dict) -> float | None:
    cash, debt, cap = _num(info, "totalCash"), _num(info, "totalDebt"), _num(info, "marketCap")
    return None if cash is None or not cap else (cash - (debt or 0.0)) / cap


def _upside(info: dict) -> float | None:
    t, p = _num(info, "targetMeanPrice"), _price(info)
    return t / p - 1 if t and p else None


def _div_yield(info: dict) -> float | None:
    rate, p = _num(info, "dividendRate"), _price(info)
    if rate is None:
        return 0.0 if p else None
    return rate / p if p else None


def _range_pos(info: dict) -> float | None:
    lo, hi, p = _num(info, "fiftyTwoWeekLow"), _num(info, "fiftyTwoWeekHigh"), _price(info)
    return (p - lo) / (hi - lo) if None not in (lo, hi, p) and hi > lo else None


@dataclass(frozen=True)
class MetricSpec:
    key: str
    name: str
    category: str
    fmt: str                                   # "x" multiple, "pct", "ratio", "num"
    curve: tuple[tuple[float, float], ...] | None
    meaning: str
    good: str
    bad: str
    get: Callable[[dict], float | None]
    nonpositive: tuple[float, str] | None = None   # (score, label) when value <= 0


QUOTE_TYPE_NAMES = {"EQUITY": "a stock", "ETF": "an ETF", "INDEX": "a market index", "MUTUALFUND": "a mutual fund",
                    "CRYPTOCURRENCY": "a cryptocurrency", "CURRENCY": "a currency pair", "FUTURE": "a futures contract"}


def _k(key: str) -> Callable[[dict], float | None]:
    return lambda info: _num(info, key)


PE_CURVE = ((8, 100), (15, 85), (25, 55), (40, 25), (60, 5))

METRICS: list[MetricSpec] = [
    # Valuation - what you pay
    MetricSpec("pe", "P/E (trailing)", "Valuation", "x", PE_CURVE,
               "Price ÷ last 12 months' earnings per share. How many years of current profit you pay for.",
               "cheap relative to earnings", "expensive relative to earnings", _pe("trailingPE"),
               nonpositive=(5, "loss-making")),
    MetricSpec("fpe", "P/E (forward)", "Valuation", "x", PE_CURVE,
               "Price ÷ analysts' expected earnings for next year. Lower than trailing P/E = profits expected to grow.",
               "cheap relative to expected earnings", "expensive even on next year's earnings", _pe("forwardPE"),
               nonpositive=(5, "loss expected")),
    MetricSpec("peg", "PEG ratio", "Valuation", "x", ((0.5, 100), (1, 85), (1.5, 60), (2, 40), (3, 10)),
               "P/E ÷ expected growth rate. Around 1 means the price is fair for the growth; above 2 is pricey.",
               "price looks fair for its growth", "price is high for its growth rate",
               lambda i: _num(i, "pegRatio") or _num(i, "trailingPegRatio"), nonpositive=(10, "negative growth")),
    MetricSpec("pb", "Price / Book", "Valuation", "x", ((1, 100), (3, 70), (6, 40), (12, 12), (25, 0)),
               "Price ÷ net assets per share. Low values mean you pay little above what the company owns.",
               "priced close to its net assets", "priced far above its net assets", _k("priceToBook"),
               nonpositive=(10, "negative equity")),
    MetricSpec("ps", "Price / Sales", "Valuation", "x", ((1, 100), (3, 75), (6, 45), (10, 20), (20, 0)),
               "Market value ÷ yearly revenue. Useful when a company has little or no profit yet.",
               "cheap relative to revenue", "expensive relative to revenue", _k("priceToSalesTrailing12Months")),
    MetricSpec("ev_ebitda", "EV / EBITDA", "Valuation", "x", ((6, 100), (10, 80), (15, 55), (25, 25), (40, 5)),
               "Whole-business value (incl. debt) ÷ operating cash profit. Compares companies regardless of debt.",
               "cheap relative to operating profit", "expensive relative to operating profit", _k("enterpriseToEbitda"),
               nonpositive=(5, "negative EBITDA")),
    # Profitability - how good the business is
    MetricSpec("roe", "Return on equity", "Profitability", "pct",
               ((0, 0), (0.05, 25), (0.10, 50), (0.15, 70), (0.20, 85), (0.30, 100)),
               "Profit ÷ shareholders' equity. How well management turns your money into profit (>15% is good).",
               "very profitable use of shareholder money", "weak returns on shareholder money", _k("returnOnEquity")),
    MetricSpec("roa", "Return on assets", "Profitability", "pct",
               ((0, 0), (0.02, 30), (0.05, 60), (0.10, 85), (0.15, 100)),
               "Profit ÷ everything the company owns. Harder to flatter with debt than ROE.",
               "earns a lot on its assets", "earns little on its assets", _k("returnOnAssets")),
    MetricSpec("net_margin", "Net profit margin", "Profitability", "pct",
               ((-0.10, 0), (0, 20), (0.05, 45), (0.10, 65), (0.20, 90), (0.30, 100)),
               "Share of each sale kept as profit after all costs and taxes.",
               "keeps a large share of revenue as profit", "thin or negative profit margin", _k("profitMargins")),
    MetricSpec("op_margin", "Operating margin", "Profitability", "pct",
               ((-0.10, 0), (0, 20), (0.10, 55), (0.20, 80), (0.30, 100)),
               "Profit from core operations ÷ revenue, before interest and taxes.",
               "strong core-business profitability", "weak core-business profitability", _k("operatingMargins")),
    MetricSpec("gross_margin", "Gross margin", "Profitability", "pct",
               ((0.10, 10), (0.25, 35), (0.40, 60), (0.60, 85), (0.75, 100)),
               "Revenue minus direct production cost, ÷ revenue. High = pricing power / a moat.",
               "strong pricing power", "low pricing power", _k("grossMargins")),
    # Growth - is it getting bigger
    MetricSpec("rev_growth", "Revenue growth (YoY)", "Growth", "pct",
               ((-0.10, 0), (0, 30), (0.05, 50), (0.10, 65), (0.20, 85), (0.35, 100)),
               "Latest quarter's sales vs. the same quarter a year ago.",
               "sales growing fast", "sales shrinking or stagnant", _k("revenueGrowth")),
    MetricSpec("eps_growth", "Earnings growth (YoY)", "Growth", "pct",
               ((-0.20, 0), (0, 30), (0.10, 60), (0.25, 85), (0.50, 100)),
               "Latest quarter's earnings vs. the same quarter a year ago.",
               "profits growing fast", "profits falling", _k("earningsGrowth")),
    MetricSpec("fwd_eps_growth", "Expected EPS growth", "Growth", "pct",
               ((-0.20, 0), (0, 30), (0.10, 60), (0.25, 85), (0.50, 100)),
               "Analysts' next-year EPS estimate vs. the last 12 months.",
               "analysts expect strong profit growth", "analysts expect profits to fall", _eps_growth),
    # Financial health - can it survive a bad year
    MetricSpec("de", "Debt / Equity", "Financial health", "ratio",
               ((0, 100), (0.3, 90), (0.8, 70), (1.5, 40), (2.5, 15), (4, 0)),
               "Total debt ÷ shareholders' equity. Above ~2 means the company relies heavily on borrowing.",
               "little debt", "heavily indebted",
               lambda i: None if _num(i, "debtToEquity") is None else _num(i, "debtToEquity") / 100),
    MetricSpec("current_ratio", "Current ratio", "Financial health", "ratio",
               ((0.5, 0), (1, 40), (1.5, 75), (2, 95), (3, 100)),
               "Short-term assets ÷ short-term bills. Below 1 means bills due this year exceed liquid assets.",
               "comfortably covers short-term bills", "tight short-term liquidity", _k("currentRatio")),
    MetricSpec("quick_ratio", "Quick ratio", "Financial health", "ratio",
               ((0.3, 0), (0.7, 40), (1, 70), (1.5, 95), (2, 100)),
               "Like the current ratio but excluding inventory - the strictest liquidity test.",
               "strong liquidity even without inventory", "weak liquidity without inventory", _k("quickRatio")),
    MetricSpec("fcf_yield", "Free-cash-flow yield", "Financial health", "pct",
               ((-0.02, 0), (0, 25), (0.02, 50), (0.04, 75), (0.07, 100)),
               "Cash left after running and investing in the business, ÷ market value. Real cash, hard to fake.",
               "generates lots of spare cash", "burns cash",
               lambda i: _ratio(_num(i, "freeCashflow"), _num(i, "marketCap"))),
    MetricSpec("net_cash", "Net cash / market cap", "Financial health", "pct",
               ((-0.5, 0), (-0.2, 30), (0, 60), (0.1, 85), (0.2, 100)),
               "(Cash - debt) ÷ market value. Positive = more cash than debt.",
               "more cash than debt", "debt far exceeds cash", _net_cash),
    # Market view
    MetricSpec("analyst_rating", "Analyst rating", "Market view", "num",
               ((1, 100), (2, 80), (2.5, 60), (3, 40), (4, 10), (5, 0)),
               "Average analyst recommendation: 1 = strong buy, 3 = hold, 5 = sell.",
               "analysts are positive", "analysts are negative", _k("recommendationMean")),
    MetricSpec("upside", "Analyst target upside", "Market view", "pct",
               ((-0.20, 0), (0, 35), (0.10, 60), (0.20, 80), (0.35, 100)),
               "Average 12-month analyst price target vs. today's price.",
               "analysts see meaningful upside", "trades above analysts' targets", _upside),
    # Context only
    MetricSpec("div_yield", "Dividend yield", "Dividends & risk", "pct", None,
               "Yearly dividend ÷ price. Income you receive just for holding the stock.", "", "", _div_yield),
    MetricSpec("payout", "Payout ratio", "Dividends & risk", "pct", None,
               "Share of profit paid out as dividends. Above ~80% can be hard to sustain.", "", "", _k("payoutRatio")),
    MetricSpec("beta", "Beta", "Dividends & risk", "num", None,
               "Volatility vs. the market: 1 = moves with the market, 1.5 = swings 50% more.", "", "", _k("beta")),
    MetricSpec("range_pos", "Position in 52-week range", "Dividends & risk", "pct", None,
               "Where today's price sits between the 52-week low (0%) and high (100%).", "", "", _range_pos),
]


def curve_score(value: float, curve) -> float:
    xs, ys = zip(*curve)
    return float(np.interp(value, xs, ys))


def format_value(value: float | None, fmt: str) -> str:
    if value is None:
        return "n/a"
    if fmt == "pct":
        return f"{value * 100:.1f}%"
    if fmt in ("x", "ratio"):
        return f"{value:.2f}×"
    return f"{value:.2f}"


@dataclass
class MetricResult:
    spec: MetricSpec
    value: float | None
    display: str
    score: float | None


@dataclass
class FundamentalReport:
    symbol: str
    info: dict
    name: str
    quote_type: str
    sector: str
    industry: str
    currency: str
    summary: str
    metrics: list[MetricResult]
    category_scores: dict[str, float | None]
    overall: float | None
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def available(self) -> bool:
        return self.overall is not None

    @property
    def tier(self) -> str | None:
        if self.overall is None:
            return None
        return "Strong" if self.overall >= 65 else "Fair" if self.overall >= 45 else "Weak"

    def graham_number(self) -> float | None:
        """Benjamin Graham's rough fair-value ceiling for a defensive investor: √(22.5 × EPS × book/share)."""
        eps, bvps = _num(self.info, "trailingEps"), _num(self.info, "bookValue")
        return math.sqrt(22.5 * eps * bvps) if eps and bvps and eps > 0 and bvps > 0 else None


def score_fundamentals(symbol: str, info: dict, price: float | None = None) -> FundamentalReport:
    info = dict(info)
    if price:
        info["currentPrice"] = price  # keep price-based ratios in sync with the live quote
    quote_type = str(info.get("quoteType") or "").upper()
    is_company = quote_type in ("", "EQUITY")  # fund/index ratios describe holdings, not a business

    results: list[MetricResult] = []
    for spec in METRICS:
        value = spec.get(info)
        score, display = None, format_value(value, spec.fmt)
        if value is not None and spec.curve is not None and is_company:
            if spec.nonpositive and value <= 0:
                score, display = spec.nonpositive[0], spec.nonpositive[1]
            else:
                score = curve_score(value, spec.curve)
        results.append(MetricResult(spec, value, display, score))

    categories: dict[str, float | None] = {}
    for cat in CATEGORIES:
        scores = [m.score for m in results if m.spec.category == cat and m.score is not None]
        categories[cat] = float(np.mean(scores)) if scores else None

    scored = [m for m in results if m.score is not None]
    weighted = [(categories[c], w) for c, w in CATEGORIES.items() if w > 0 and categories[c] is not None]
    overall, note = None, ""
    if not is_company:
        kind = QUOTE_TYPE_NAMES.get(quote_type, f"a {quote_type.lower()}")
        note = (f"{symbol} is {kind}, not a single company - it has no business of its own to analyse "
                "(any ratios shown describe its holdings). Only the technical view applies.")
    elif len(scored) < 6:
        note = "Yahoo Finance has too little fundamental data for this symbol to score it reliably."
    else:
        overall = sum(s * w for s, w in weighted) / sum(w for _, w in weighted)

    ranked = sorted(scored, key=lambda m: m.score, reverse=True)
    strengths = [f"{m.spec.name} {m.display} - {m.spec.good}" for m in ranked if m.score >= 75][:5]
    weaknesses = [f"{m.spec.name} {m.display} - {m.spec.bad}" for m in reversed(ranked) if m.score <= 30][:5]

    return FundamentalReport(
        symbol=symbol, info=info,
        name=info.get("longName") or info.get("shortName") or symbol,
        quote_type=quote_type or "UNKNOWN",
        sector=info.get("sector") or "", industry=info.get("industry") or "",
        currency=info.get("currency") or info.get("financialCurrency") or "USD",
        summary=info.get("longBusinessSummary") or "",
        metrics=results, category_scores=categories, overall=overall,
        strengths=strengths, weaknesses=weaknesses, note=note,
    )


def fetch_fundamentals(symbol: str, price: float | None = None) -> FundamentalReport:
    return score_fundamentals(symbol, fetch_info(symbol), price)
