"""Bar-by-bar trading simulation of the signal-score strategy.

Rules (long-only, all-in / all-out):
  * A BUY signal fires when the score crosses up through the buy threshold while flat.
  * A SELL signal fires when the score is at/below the sell threshold while long.
  * Signals are decided on a bar's close and filled at the NEXT bar's open, so the
    simulation never trades on a price it could not have known.
  * Optional ATR trailing stop: starts `atr_stop_mult` ATRs below the entry and only
    ratchets up. If a bar's low touches it we exit at the stop (or at the open, if
    price gapped below it).
  * A percentage commission is charged on every buy and sell.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .strategy import StrategyParams


@dataclass
class Trade:
    entry_idx: int
    entry_time: pd.Timestamp
    entry_price: float
    shares: float
    cost: float                          # cash spent including commission
    exit_idx: int | None = None
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_kind: str = ""                  # "signal" | "stop" | "" (still open)
    exit_reason: str = ""
    proceeds: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.exit_idx is None

    def value_at(self, price: float, fee: float) -> float:
        return self.shares * price * (1 - fee)

    def pnl(self, mark: float | None = None, fee: float = 0.0) -> float:
        end = self.proceeds if not self.is_open else self.value_at(mark, fee)
        return end - self.cost

    def return_pct(self, mark: float | None = None, fee: float = 0.0) -> float:
        return self.pnl(mark, fee) / self.cost

    def bars_held(self, last_idx: int) -> int:
        return (self.exit_idx if not self.is_open else last_idx) - self.entry_idx


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity: pd.Series
    benchmark: pd.Series                 # buy & hold the same capital
    in_position: pd.Series
    stop_line: pd.Series                 # active trailing stop level (NaN when flat)
    signals: dict[int, str]              # bar index -> "buy" / "sell" (decision bars)
    pending: str | None                  # signal fired on the last bar, fills next open
    capital: float
    commission: float                    # fraction per side
    stats: dict[str, float] = field(default_factory=dict)

    @property
    def open_trade(self) -> Trade | None:
        return self.trades[-1] if self.trades and self.trades[-1].is_open else None

    def trade_at(self, i: int) -> Trade | None:
        """The trade that is open during bar i, if any."""
        for t in self.trades:
            if t.entry_idx <= i and (t.is_open or i < t.exit_idx):
                return t
        return None


def run_backtest(df: pd.DataFrame, p: StrategyParams, capital: float = 10_000.0,
                 commission_pct: float = 0.1, bars_per_year: int = 252) -> BacktestResult:
    n = len(df)
    fee = commission_pct / 100
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    score = df["score"].to_numpy(float)
    atr = df["atr"].to_numpy(float)
    times = df.index
    use_stop = p.atr_stop_mult > 0

    cash, trade = capital, None
    trades: list[Trade] = []
    signals: dict[int, str] = {}
    pending: str | None = None
    stop = np.nan
    equity = np.empty(n)
    in_pos = np.zeros(n, bool)
    stop_line = np.full(n, np.nan)

    def close_trade(i: int, price: float, kind: str, reason: str):
        nonlocal cash, trade, stop
        trade.exit_idx, trade.exit_time, trade.exit_price = i, times[i], price
        trade.exit_kind, trade.exit_reason = kind, reason
        trade.proceeds = trade.value_at(price, fee)
        cash, trade, stop = trade.proceeds, None, np.nan

    for i in range(n):
        # 1) Fill yesterday's decision at today's open.
        if pending == "buy" and trade is None:
            shares = cash * (1 - fee) / o[i]
            trade = Trade(i, times[i], o[i], shares, cost=cash)
            trades.append(trade)
            cash = 0.0
            ref_atr = atr[i - 1] if i > 0 else atr[i]
            stop = o[i] - p.atr_stop_mult * ref_atr if use_stop and np.isfinite(ref_atr) else np.nan
        elif pending == "sell" and trade is not None:
            close_trade(i, o[i], "signal", f"Score fell to {score[i - 1]:+.0f} (≤ {p.sell_threshold:+.0f})")
        pending = None

        # 2) Intrabar stop check.
        if trade is not None and np.isfinite(stop) and l[i] <= stop:
            close_trade(i, min(o[i], stop), "stop", f"ATR trailing stop hit ({stop:,.2f})")

        # 3) Decide at the close.
        if trade is not None:
            if use_stop and np.isfinite(atr[i]):
                stop = max(stop, c[i] - p.atr_stop_mult * atr[i]) if np.isfinite(stop) else c[i] - p.atr_stop_mult * atr[i]
            if score[i] <= p.sell_threshold:
                pending, signals[i] = "sell", "sell"
        elif i > 0 and score[i - 1] < p.buy_threshold <= score[i]:
            pending, signals[i] = "buy", "buy"

        in_pos[i] = trade is not None
        stop_line[i] = stop if trade is not None else np.nan
        equity[i] = cash + (trade.shares * c[i] if trade is not None else 0.0)

    result = BacktestResult(
        trades=trades,
        equity=pd.Series(equity, index=times),
        benchmark=pd.Series(capital * (1 - fee) * c / o[0], index=times),
        in_position=pd.Series(in_pos, index=times),
        stop_line=pd.Series(stop_line, index=times),
        signals=signals,
        pending=pending,
        capital=capital,
        commission=fee,
    )
    result.stats = _stats(result, c[-1], bars_per_year)
    return result


def _stats(r: BacktestResult, last_price: float, bars_per_year: int) -> dict[str, float]:
    eq = r.equity
    closed = [t for t in r.trades if not t.is_open]
    rets = np.array([t.return_pct() for t in closed])
    pnls = np.array([t.pnl() for t in closed])
    wins, losses = rets[rets > 0], rets[rets <= 0]
    gross_win, gross_loss = pnls[pnls > 0].sum(), -pnls[pnls <= 0].sum()
    bar_rets = eq.pct_change().dropna()
    sd = bar_rets.std()
    last_idx = len(eq) - 1
    return {
        "final_equity": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] / r.capital - 1),
        "buy_hold_return": float(r.benchmark.iloc[-1] / r.capital - 1),
        "trades": float(len(r.trades)),
        "closed_trades": float(len(closed)),
        "win_rate": float(len(wins) / len(closed)) if closed else np.nan,
        "avg_win": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else (np.inf if gross_win > 0 else np.nan),
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
        "buy_hold_max_drawdown": float((r.benchmark / r.benchmark.cummax() - 1).min()),
        "exposure": float(r.in_position.mean()),
        "sharpe": float(bar_rets.mean() / sd * np.sqrt(bars_per_year)) if sd > 0 else np.nan,
        "avg_bars_held": float(np.mean([t.bars_held(last_idx) for t in r.trades])) if r.trades else np.nan,
        "open_return": float(r.open_trade.return_pct(last_price, r.commission)) if r.open_trade else np.nan,
    }
