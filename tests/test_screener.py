"""Tests for the screener (offline: uses the DEMO symbol and local files)."""
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tracker.custom_strategy import TrainConfig
from tracker.data import fetch_daily_direct
from tracker.screener import (BatchState, TickerResult, analyse_list, apply_suffix, bottom_line, decide,
                              full_evaluate, parse_ticker_text, quick_evaluate, read_tickers_file, results_frame)
from tracker.strategy import StrategyParams


class TickerParsingTests(unittest.TestCase):
    def _file(self, name: str, text: str) -> Path:
        path = Path(tempfile.mkdtemp()) / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_csv_variants(self):
        self.assertEqual(read_tickers_file(self._file("a.csv", "Symbol,Qty\nAAPL,1\nmsft,2\nAAPL,3\n")), ["AAPL", "MSFT"])
        self.assertEqual(read_tickers_file(self._file("b.csv", "AAPL\nMSFT\nAMZN\n")), ["AAPL", "MSFT", "AMZN"])
        self.assertEqual(read_tickers_file(self._file("c.csv", "AAPL, MSFT, GOOGL\n")), ["AAPL", "MSFT", "GOOGL"])
        self.assertEqual(read_tickers_file(self._file("d.txt", "tsla ^gspc btc-usd")), ["TSLA", "^GSPC", "BTC-USD"])

    def test_excel_finds_ticker_column(self):
        path = Path(tempfile.mkdtemp()) / "w.xlsx"
        pd.DataFrame({"Company": ["Apple Inc", "Infosys Ltd"], "Ticker": ["AAPL", "INFY.NS"]}).to_excel(path, index=False)
        self.assertEqual(read_tickers_file(path), ["AAPL", "INFY.NS"])

    def test_suffix(self):
        self.assertEqual(apply_suffix(parse_ticker_text("INFY TCS.NS ^NSEI BTC-USD"), ".NS"),
                         ["INFY.NS", "TCS.NS", "^NSEI", "BTC-USD"])


class EvaluationTests(unittest.TestCase):
    tech, cfg = StrategyParams(), TrainConfig(iterations=40)

    def test_quick_and_full_on_demo(self):
        df, meta = fetch_daily_direct("DEMO")
        q = quick_evaluate("DEMO", df, meta, self.tech, self.cfg)
        self.assertEqual(q.depth, "quick")
        self.assertTrue(-1 <= q.pre_score <= 1)
        r = full_evaluate("DEMO", df, meta, self.tech, self.cfg, iterations=40)
        self.assertEqual(r.depth, "full")
        self.assertEqual(set(r.votes), {"technical", "fundamental", "strategy"})
        self.assertIn(r.label, ("STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"))
        self.assertIn(r.confidence, ("High", "Medium", "Low"))
        self.assertTrue(bottom_line(r))

    def test_too_new_for_quick_screen(self):
        df, meta = fetch_daily_direct("DEMO")
        with self.assertRaises(ValueError):
            quick_evaluate("DEMO", df.iloc[-200:], meta, self.tech, self.cfg)

    def test_list_batch_and_export(self):
        state = analyse_list(["DEMO"], self.tech, self.cfg, 40, BatchState())
        self.assertEqual(len(state.results), 1)
        frame = results_frame(state.results + [TickerResult("DEMO", depth="quick", pre_score=0.1)])
        self.assertEqual(len(frame), 1)                        # one row per symbol, latest wins

    def test_decide_thresholds(self):
        r = TickerResult("X", tech_score=60)
        decide(r)                                              # technical only: +1 vote
        self.assertEqual(r.label, "STRONG BUY")
        self.assertEqual(r.confidence, "Medium")               # a method is missing -> never High
        r = TickerResult("Y", tech_score=-60)
        decide(r)
        self.assertEqual(r.label, "STRONG SELL")
        self.assertTrue(math.isclose(r.overall, -1.0))


if __name__ == "__main__":
    unittest.main()
