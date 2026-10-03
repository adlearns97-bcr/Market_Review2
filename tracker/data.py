"""Market data: price history, live quotes and company info.

Real data comes from Yahoo Finance through yfinance (quotes can be delayed a few
minutes depending on the exchange). The special symbol ``DEMO`` returns synthetic
data so the app can be explored offline.
"""
from __future__ import annotations

import threading
import time
import zlib
from urllib.parse import quote as url_quote
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .net import YAHOO_LOCK, get_session, thread_session

DEMO_SYMBOL = "DEMO"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


class DataError(RuntimeError):
    """A user-facing problem fetching data (bad symbol, no network...)."""


@dataclass(frozen=True)
class Timeframe:
    label: str
    interval: str       # Yahoo bar size
    fetch_period: str   # how much history to download (extra = indicator warm-up)
    display_kind: str   # "sessions" | "months" | "years"
    display_n: int      # how much of it to show and simulate over


TIMEFRAMES = [
    Timeframe("1 Day · 1 min", "1m", "5d", "sessions", 1),
    Timeframe("5 Days · 5 min", "5m", "1mo", "sessions", 5),
    Timeframe("1 Month · 30 min", "30m", "60d", "months", 1),
    Timeframe("3 Months · 1 hour", "1h", "1y", "months", 3),
    Timeframe("6 Months · 1 day", "1d", "2y", "months", 6),
    Timeframe("1 Year · 1 day", "1d", "2y", "years", 1),
    Timeframe("2 Years · 1 day", "1d", "5y", "years", 2),
    Timeframe("5 Years · 1 week", "1wk", "10y", "years", 5),
]
TIMEFRAME_BY_LABEL = {tf.label: tf for tf in TIMEFRAMES}

# Daily history used to train the custom strategy (mode 3).
TRAIN_HISTORY = {"2 years": "2y", "5 years": "5y", "10 years": "10y", "Max": "max"}


def train_timeframe(label: str) -> Timeframe:
    return Timeframe(f"Daily · {label}", "1d", TRAIN_HISTORY[label], "years", 100)
DEFAULT_TIMEFRAME = "1 Year · 1 day"

# Used to annualise Sharpe ratios. Assumes a regular 6.5h stock session.
BARS_PER_YEAR = {"1m": 252 * 390, "5m": 252 * 78, "30m": 252 * 13, "1h": 252 * 7, "1d": 252, "1wk": 52}
INTRADAY = {"1m", "5m", "30m", "1h"}


@dataclass
class Quote:
    symbol: str
    price: float
    prev_close: float | None
    currency: str
    last_bar: pd.Timestamp

    @property
    def change(self) -> float | None:
        return None if self.prev_close in (None, 0) else self.price - self.prev_close

    @property
    def change_pct(self) -> float | None:
        return None if self.prev_close in (None, 0) else self.price / self.prev_close - 1


@dataclass
class MarketData:
    history: pd.DataFrame
    quote: Quote


def is_demo(symbol: str) -> bool:
    return symbol.upper() == DEMO_SYMBOL


def display_start(index: pd.DatetimeIndex, tf: Timeframe) -> int:
    """Position of the first bar inside the timeframe's display window."""
    if len(index) == 0:
        return 0
    if tf.display_kind == "sessions":
        dates = np.array(index.date)
        unique = pd.unique(dates)
        first = unique[-tf.display_n] if len(unique) >= tf.display_n else unique[0]
        return int(np.argmax(dates >= first))
    offset = pd.DateOffset(months=tf.display_n) if tf.display_kind == "months" else pd.DateOffset(years=tf.display_n)
    return int(index.searchsorted(index[-1] - offset))


# --------------------------------------------------------------------------- Yahoo


def _ticker(symbol: str):
    import yfinance as yf

    return yf.Ticker(symbol, session=get_session())


def fetch_history(symbol: str, tf: Timeframe) -> pd.DataFrame:
    if is_demo(symbol):
        return demo_history(tf)
    try:
        with YAHOO_LOCK:
            df = _ticker(symbol).history(
                period=tf.fetch_period, interval=tf.interval, auto_adjust=True,
                actions=False, timeout=15, raise_errors=True,
            )
    except Exception as exc:  # yfinance raises a zoo of exception types
        raise DataError(_friendly_error(symbol, exc)) from exc
    return _clean(df, symbol)


def fetch_quote(symbol: str, history: pd.DataFrame) -> Quote:
    last_close = float(history["Close"].iloc[-1])
    last_bar = history.index[-1]
    if is_demo(symbol):
        daily_close = history["Close"].groupby(history.index.date).last()
        prev = float(daily_close.iloc[-2]) if len(daily_close) > 1 else None
        return Quote(symbol, last_close, prev, "USD", last_bar)
    price, prev, currency = last_close, None, "USD"
    try:
        with YAHOO_LOCK:
            fi = _ticker(symbol).fast_info
            price = float(fi["lastPrice"] or last_close)
            prev = fi.get("regularMarketPreviousClose") or fi.get("previousClose")
            prev = float(prev) if prev else None
            currency = fi.get("currency") or currency
    except Exception:
        pass  # the chart still works without the quote header
    return Quote(symbol, price, prev, currency, last_bar)


def load_market_data(symbol: str, tf: Timeframe) -> MarketData:
    history = fetch_history(symbol, tf)
    return MarketData(history, fetch_quote(symbol, history))


CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{}"


def fetch_daily_direct(symbol: str, period: str = "5y") -> tuple[pd.DataFrame, dict]:
    """Daily OHLCV straight from Yahoo's chart endpoint, safe to call from many threads at once.

    Used by the screener to download hundreds of symbols in parallel. Prices are adjusted for
    splits and dividends like yfinance's auto_adjust. Returns (history, chart metadata)."""
    if is_demo(symbol):
        return demo_history(Timeframe("demo", "1d", period, "years", 1)), {"currency": "USD", "longName": DEMO_INFO["longName"]}
    for attempt in range(4):
        try:
            resp = thread_session().get(CHART_URL.format(url_quote(symbol, safe="")), timeout=20,
                                        params={"range": period, "interval": "1d", "includeAdjustedClose": "true"})
        except Exception as exc:
            raise DataError(_friendly_error(symbol, exc)) from exc
        if resp.status_code == 429 and attempt < 3:
            time.sleep(2 ** (attempt + 1))  # rate-limited: back off and retry
            continue
        break
    if resp.status_code == 404:
        raise DataError(f"'{symbol}' was not found on Yahoo Finance.")
    if resp.status_code != 200:
        raise DataError(f"Yahoo returned HTTP {resp.status_code} for '{symbol}'.")
    chart = resp.json().get("chart", {})
    result = (chart.get("result") or [None])[0]
    if not result or not result.get("timestamp"):
        raise DataError(f"No price data for '{symbol}'.")
    meta = result.get("meta", {})
    quote = result["indicators"]["quote"][0]
    tz = meta.get("exchangeTimezoneName") or "UTC"
    index = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(tz).normalize()
    df = pd.DataFrame({k.capitalize(): pd.to_numeric(pd.Series(quote.get(k)), errors="coerce").to_numpy(float)
                       for k in ("open", "high", "low", "close", "volume")}, index=index)
    adj = (result["indicators"].get("adjclose") or [{}])[0].get("adjclose")
    if adj is not None:
        with np.errstate(divide="ignore", invalid="ignore"):  # zero/missing closes on thin days
            factor = pd.to_numeric(pd.Series(adj), errors="coerce").to_numpy(float) / df["Close"].to_numpy()
        factor = np.where(np.isfinite(factor), factor, 1.0)
        for col in ("Open", "High", "Low", "Close"):
            df[col] = df[col] * factor
    return _clean(df, symbol), meta


_info_cache: dict[str, tuple[float, dict]] = {}
_info_lock = threading.Lock()
INFO_TTL_SECONDS = 30 * 60


def fetch_info(symbol: str) -> dict:
    """Company profile + fundamentals (Yahoo's quoteSummary), cached for 30 minutes."""
    if is_demo(symbol):
        return dict(DEMO_INFO)
    key = symbol.upper()
    with _info_lock:
        hit = _info_cache.get(key)
        if hit and time.time() - hit[0] < INFO_TTL_SECONDS:
            return dict(hit[1])
    try:
        with YAHOO_LOCK:
            info = _ticker(symbol).info or {}
    except Exception as exc:
        raise DataError(_friendly_error(symbol, exc)) from exc
    with _info_lock:
        _info_cache[key] = (time.time(), info)
    return dict(info)


def _clean(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise DataError(f"No price data for '{symbol}'. Check the symbol uses Yahoo format "
                        "(e.g. AAPL, RELIANCE.NS, BTC-USD, ^GSPC).")
    df = df[OHLCV].astype(float)
    df = df[df["Close"].notna() & (df["Close"] > 0)]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df["Volume"] = df["Volume"].fillna(0.0)
    if len(df) < 30:
        raise DataError(f"Only {len(df)} bars of data for '{symbol}' - not enough to analyse.")
    return df


def _friendly_error(symbol: str, exc: Exception) -> str:
    name, text = type(exc).__name__, str(exc)
    if "RateLimit" in name or "Too Many Requests" in text:
        return "Yahoo Finance is rate-limiting requests. Wait a minute, or slow the live refresh."
    if "Certificate" in name or "SSL" in text:
        return "HTTPS certificate problem reaching Yahoo Finance (proxy/antivirus?)."
    if "delisted" in text or "Missing" in name or "not found" in text.lower():
        return f"'{symbol}' was not found on Yahoo Finance. Check the symbol."
    if "resolve" in text.lower() or "timed out" in text.lower() or "Connection" in name:
        return "Can't reach Yahoo Finance - check your internet connection (or try symbol DEMO)."
    return f"Couldn't load '{symbol}': {text.splitlines()[0] if text else name}"


# --------------------------------------------------------------------------- demo data

_SESSIONS_IN = {"5d": 5, "1mo": 21, "60d": 42, "1y": 252, "2y": 504, "5y": 1260, "10y": 2520, "max": 5040}
_BARS_PER_SESSION = {"1m": 390, "5m": 78, "30m": 13, "1h": 7}


def demo_history(tf: Timeframe) -> pd.DataFrame:
    """Synthetic but realistic-looking OHLCV: a random walk whose drift switches between
    bull, bear and sideways regimes, so every kind of signal shows up."""
    sessions = _SESSIONS_IN[tf.fetch_period]
    tz = "America/New_York"
    today = pd.Timestamp.now(tz=tz).normalize().tz_localize(None)
    days = pd.bdate_range(end=today, periods=sessions)

    if tf.interval in _BARS_PER_SESSION:
        per = _BARS_PER_SESSION[tf.interval]
        offsets = pd.to_timedelta(570 + np.arange(per) * (390 // per), unit="min")  # from 09:30
        stamps = (days.values[:, None] + offsets.values[None, :]).ravel()
        index = pd.DatetimeIndex(stamps).tz_localize(tz)
        sigma = 0.018 / np.sqrt(per)
    elif tf.interval == "1wk":
        index = pd.date_range(end=today, periods=sessions // 5, freq="W-MON").tz_localize(tz)
        sigma = 0.018 * np.sqrt(5)
    else:
        index = days.tz_localize(tz)
        sigma = 0.018

    n = len(index)
    rng = np.random.default_rng(zlib.crc32(tf.interval.encode()))
    regime_len = max(20, n // 12)
    drifts = rng.choice([-0.35, 0.0, 0.45], size=n // regime_len + 1) * sigma
    drift = np.repeat(drifts, regime_len)[:n]
    rets = drift + rng.normal(0, sigma, n)
    close = 100 * np.exp(np.cumsum(rets))

    # Live jiggle: nudge the latest bar every few seconds so "live" mode visibly moves.
    close[-1] *= 1 + np.random.default_rng(int(time.time() // 5)).normal(0, sigma * 0.4)

    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, sigma * 0.15, n))
    wick = np.abs(rng.normal(0, sigma * 0.6, (2, n)))
    high = np.maximum(open_, close) * (1 + wick[0])
    low = np.minimum(open_, close) * (1 - wick[1])
    volume = rng.lognormal(13.5, 0.35, n) * (1 + 25 * np.abs(rets))
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=index)


DEMO_INFO = {
    "quoteType": "EQUITY",
    "longName": "Demo Dynamics Inc. (synthetic)",
    "symbol": DEMO_SYMBOL,
    "sector": "Technology",
    "industry": "Software - Application",
    "exchange": "DEMO",
    "currency": "USD",
    "longBusinessSummary": (
        "Demo Dynamics is a made-up company used to explore the app offline. Its prices "
        "are generated by a random walk and its fundamentals are fixed sample numbers, "
        "chosen to look like a profitable, moderately priced software business."
    ),
    "marketCap": 48_000_000_000,
    "currentPrice": 100.0,
    "trailingPE": 22.4,
    "forwardPE": 19.1,
    "pegRatio": 1.3,
    "priceToBook": 6.2,
    "priceToSalesTrailing12Months": 5.1,
    "enterpriseToEbitda": 15.8,
    "returnOnEquity": 0.27,
    "returnOnAssets": 0.11,
    "profitMargins": 0.21,
    "operatingMargins": 0.26,
    "grossMargins": 0.71,
    "revenueGrowth": 0.12,
    "earningsGrowth": 0.15,
    "trailingEps": 4.46,
    "forwardEps": 5.24,
    "bookValue": 16.1,
    "debtToEquity": 45.0,
    "currentRatio": 1.9,
    "quickRatio": 1.6,
    "freeCashflow": 2_100_000_000,
    "totalCash": 6_000_000_000,
    "totalDebt": 3_500_000_000,
    "recommendationMean": 2.1,
    "recommendationKey": "buy",
    "numberOfAnalystOpinions": 24,
    "targetMeanPrice": 116.0,
    "targetLowPrice": 88.0,
    "targetHighPrice": 140.0,
    "dividendRate": 0.8,
    "payoutRatio": 0.18,
    "beta": 1.15,
    "fiftyTwoWeekLow": 78.0,
    "fiftyTwoWeekHigh": 121.0,
}
