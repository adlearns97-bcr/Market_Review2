"""Tests for the Modified Strat engine and the time-to-target forecast (no network, no GUI)."""
import math
import unittest

import numpy as np
import pandas as pd

from tracker.custom_strategy import DEFAULT_VALUES, FeatureCache, TrainConfig, simulate, space_size, train_strategy
from tracker.data import demo_history, train_timeframe
from tracker.forecast import date_after, first_passage

DF = demo_history(train_timeframe("5 years"))


def bars(rows):
    """Build an OHLC frame from (open, high, low, close) rows."""
    idx = pd.bdate_range("2024-01-01", periods=len(rows), tz="America/New_York")
    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx, dtype=float)
    df["Volume"] = 1e6
    return df


class SimulationTests(unittest.TestCase):
    def _run(self, rows, setup_on, **overrides):
        df = bars(rows)
        fc = FeatureCache(df)
        fc.atr = np.full(len(df), 1.0)                       # fixed 1.00 ATR keeps the arithmetic obvious
        fc.lists = tuple(a.tolist() for a in (fc.o, fc.h, fc.l, fc.c, fc.atr))
        setup = np.zeros(len(df), bool)
        setup[setup_on] = True
        v = {**DEFAULT_VALUES, "tp_atr": 2.0, "sl_atr": 1.0, "max_bars": 5, "trailing": 0.0, **overrides}
        return simulate(fc, setup, v, 0, len(df), fee=0.0)

    def test_target_hit_intraday(self):
        rows = [(10, 10.2, 9.8, 10)] * 3 + [(10, 10.5, 9.9, 10.3), (10.3, 12.5, 10.2, 12.0), (12, 12, 12, 12)]
        res = self._run(rows, [2])
        t = res.trades[0]
        self.assertEqual((t.entry_idx, t.entry_price), (3, 10.0))  # signal on bar 2 -> fill bar 3 open
        self.assertEqual((t.exit_idx, t.exit_price, t.reason), (4, 12.0, "target"))

    def test_stop_before_target_same_bar(self):
        rows = [(10, 10, 10, 10)] * 3 + [(10, 12.5, 8.5, 10)] + [(10, 10, 10, 10)] * 2
        t = self._run(rows, [2]).trades[0]
        self.assertEqual((t.reason, t.exit_price), ("stop", 9.0))    # conservative tie-break

    def test_gap_down_fills_at_open(self):
        rows = [(10, 10, 10, 10)] * 3 + [(10, 10.1, 9.95, 10), (8, 8.2, 7.9, 8)] + [(8, 8, 8, 8)]
        t = self._run(rows, [2]).trades[0]
        self.assertEqual((t.reason, t.exit_price), ("stop", 8.0))

    def test_time_limit_and_one_trade_per_setup(self):
        rows = [(10, 10.1, 9.95, 10)] * 12
        res = self._run(rows, list(range(2, 12)), max_bars=3)
        self.assertEqual(len(res.trades), 1)                          # setup never switched off
        self.assertEqual(res.trades[0].reason, "time")
        self.assertEqual(res.trades[0].exit_idx - res.trades[0].entry_idx + 1, 3)

    def test_pending_signal_on_last_bar(self):
        rows = [(10, 10, 10, 10)] * 5
        res = self._run(rows, [4])
        self.assertTrue(res.pending)
        self.assertEqual(res.trades, [])


class TrainingTests(unittest.TestCase):
    def test_training_and_signal(self):
        cfg = TrainConfig(iterations=60)
        progress = []
        rep = train_strategy("DEMO", DF, cfg, progress=lambda d, t: progress.append((d, t)))
        self.assertLessEqual(rep.n_evaluated, cfg.iterations + 1)
        self.assertEqual(progress[-1][0], progress[-1][1])
        self.assertTrue(rep.start < rep.split < rep.n)
        scores = [t.score for t in rep.trials]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertIn(rep.signal.action, ("BUY", "HOLD", "WAIT", "NO BUY"))
        self.assertGreater(rep.signal.target, rep.signal.stop)
        if rep.signal.trade is None:
            self.assertLess(rep.signal.stop, rep.signal.entry)
            self.assertGreater(rep.signal.target, rep.signal.entry)

    def test_training_never_sees_test_data(self):
        """Scrambling the test period must not change which parameters win in training."""
        cfg = TrainConfig(iterations=40)
        rep = train_strategy("DEMO", DF, cfg)
        scrambled = DF.copy()
        tail = scrambled.iloc[rep.split + 1:]
        scrambled.iloc[rep.split + 1:] = tail.iloc[::-1].to_numpy()
        rep2 = train_strategy("DEMO", scrambled, cfg)
        self.assertEqual(rep.best, rep2.best)

    def test_space_size(self):
        cfg = TrainConfig(rules=("trend",), optimise=("trend_len", "tp_atr"))
        self.assertEqual(space_size(cfg), 4 * 5)


class ForecastTests(unittest.TestCase):
    def test_probabilities_and_baseline(self):
        f = first_passage(DF, 100.0, 106.0, 97.0, horizon=30)
        self.assertAlmostEqual(f.p_target + f.p_stop + f.p_neither, 1.0, places=9)
        self.assertAlmostEqual(f.baseline_p, 3 / 9)
        self.assertEqual(len(f.bands[50]), 31)
        if f.median_bars:
            self.assertTrue(1 <= f.q25 <= f.median_bars <= f.q75 <= 30)

    def test_steady_uptrend_hits_target_on_schedule(self):
        close = 100 * 1.01 ** np.arange(300)                   # +1% every day, no noise
        df = bars([(c, c, c, c) for c in close])
        f = first_passage(df, 100.0, 105.0, 90.0, horizon=20)
        self.assertEqual(f.p_target, 1.0)
        self.assertEqual(f.median_bars, math.ceil(math.log(1.05) / math.log(1.01)))

    def test_future_dates_skip_weekends(self):
        d = date_after(DF.index, 5)
        self.assertLess(d.dayofweek, 5)


if __name__ == "__main__":
    unittest.main()
