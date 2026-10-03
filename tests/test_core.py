"""Core-logic tests (no network, no GUI). Run:  python -m unittest discover tests"""
import unittest

import numpy as np
import pandas as pd

from tracker import indicators as ind
from tracker.analysis import analyze
from tracker.backtest import run_backtest
from tracker.combined import combine
from tracker.data import DEMO_INFO, TIMEFRAME_BY_LABEL, TIMEFRAMES, demo_history, display_start
from tracker.fundamentals import score_fundamentals
from tracker.strategy import StrategyParams, add_indicators, component_scores

TF = TIMEFRAME_BY_LABEL["1 Year · 1 day"]


class IndicatorTests(unittest.TestCase):
    def test_sma_and_rsi_basics(self):
        s = pd.Series(np.arange(1.0, 31.0))
        self.assertEqual(ind.sma(s, 5).iloc[-1], np.mean(np.arange(26.0, 31.0)))
        self.assertAlmostEqual(ind.rsi(s, 14).iloc[-1], 100.0)          # only gains
        self.assertAlmostEqual(ind.rsi(-s, 14).iloc[-1], 0.0)           # only losses
        self.assertAlmostEqual(ind.rsi(pd.Series([5.0] * 30), 14).iloc[-1], 50.0)  # flat

    def test_bollinger_contains_price_mostly(self):
        df = demo_history(TF)
        mid, up, lo = ind.bollinger(df["Close"])
        inside = ((df["Close"] <= up) & (df["Close"] >= lo)).loc[up.notna()].mean()
        self.assertGreater(inside, 0.8)

    def test_atr_positive(self):
        df = demo_history(TF)
        a = ind.atr(df["High"], df["Low"], df["Close"]).dropna()
        self.assertTrue((a > 0).all())


class StrategyTests(unittest.TestCase):
    def test_score_range(self):
        p = StrategyParams()
        df = add_indicators(demo_history(TF), p)
        s = component_scores(df, p)["score"]
        self.assertTrue(s.between(-100, 100).all())

    def test_no_lookahead(self):
        """Scores up to bar k must be identical whether or not later bars exist."""
        p = StrategyParams()
        raw = demo_history(TF)
        full = component_scores(add_indicators(raw, p), p)["score"]
        k = len(raw) - 60
        cut = component_scores(add_indicators(raw.iloc[:k], p), p)["score"]
        pd.testing.assert_series_equal(full.iloc[:k], cut, check_names=False)


class BacktestTests(unittest.TestCase):
    def setUp(self):
        self.ta = analyze("DEMO", demo_history(TF), TF, StrategyParams())

    def test_equity_and_trades_consistent(self):
        bt = self.ta.backtest
        self.assertAlmostEqual(bt.equity.iloc[0], bt.capital, delta=bt.capital * 0.05)
        last = -1
        for t in bt.trades:
            self.assertGreater(t.entry_idx, last)                      # no overlapping trades
            self.assertEqual(bt.signals.get(t.entry_idx - 1), "buy")   # filled the bar after the signal
            if not t.is_open:
                self.assertGreaterEqual(t.exit_idx, t.entry_idx)
                last = t.exit_idx
        closed_pnl = sum(t.pnl() for t in bt.trades if not t.is_open)
        if bt.open_trade is None:
            self.assertAlmostEqual(bt.equity.iloc[-1], bt.capital + closed_pnl, places=6)

    def test_forced_trade_fills_next_open(self):
        df = self.ta.data.copy()
        df["score"] = -50.0
        df.iloc[10, df.columns.get_loc("score")] = 80.0   # cross up on bar 10
        df.iloc[20, df.columns.get_loc("score")] = -50.0
        bt = run_backtest(df, StrategyParams(atr_stop_mult=0), capital=1000, commission_pct=0)
        t = bt.trades[0]
        self.assertEqual(t.entry_idx, 11)
        self.assertEqual(t.entry_price, df["Open"].iloc[11])
        self.assertEqual(t.exit_idx, 12)                    # score back below sell on bar 11
        self.assertAlmostEqual(t.return_pct(), df["Open"].iloc[12] / df["Open"].iloc[11] - 1)

    def test_snapshot_target_and_stop(self):
        snap = self.ta.snapshot()
        p = self.ta.params
        self.assertAlmostEqual(snap.stop, snap.close - p.atr_stop_mult * snap.atr)
        self.assertAlmostEqual((snap.target - snap.close) / (snap.close - snap.stop), p.target_r)
        no_atr_stop = analyze("DEMO", demo_history(TF), TF, StrategyParams(atr_stop_mult=0)).snapshot()
        self.assertLess(no_atr_stop.stop, no_atr_stop.close)       # falls back to support / 2 ATR
        self.assertGreater(no_atr_stop.target, no_atr_stop.close)

    def test_display_window(self):
        for tf in TIMEFRAMES:
            df = demo_history(tf)
            start = display_start(df.index, tf)
            self.assertTrue(0 <= start < len(df), tf.label)


class FundamentalTests(unittest.TestCase):
    def test_demo_company_scores(self):
        r = score_fundamentals("DEMO", DEMO_INFO)
        self.assertTrue(r.available)
        self.assertTrue(0 <= r.overall <= 100)
        self.assertIn(r.tier, ("Strong", "Fair", "Weak"))
        self.assertIsNotNone(r.graham_number())

    def test_etf_is_not_scored(self):
        r = score_fundamentals("SPY", {"quoteType": "ETF", "trailingPE": 25.0, "priceToBook": 4.0})
        self.assertFalse(r.available)
        self.assertTrue(all(m.score is None for m in r.metrics))
        v = combine(r, 60, StrategyParams())
        self.assertIn("technical only", v.label)

    def test_loss_making_pe(self):
        r = score_fundamentals("X", {"quoteType": "EQUITY", "trailingEps": -1.2})
        pe = next(m for m in r.metrics if m.spec.key == "pe")
        self.assertEqual(pe.display, "loss-making")
        self.assertLess(pe.score, 10)

    def test_matrix(self):
        r = score_fundamentals("DEMO", DEMO_INFO)
        p = StrategyParams()
        self.assertEqual(r.tier, "Strong")
        self.assertEqual(combine(r, 80, p).label, "STRONG BUY")
        self.assertEqual(combine(r, 0, p).label, "ACCUMULATE")
        self.assertEqual(combine(r, -80, p).label, "WAIT")


if __name__ == "__main__":
    unittest.main()
