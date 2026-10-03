"""Screen 1 - Pure technical: chart signals, the current call, and a trading simulation."""
from __future__ import annotations

import math

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QFormLayout, QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QPushButton, QScrollArea, QSpinBox, QSplitter, QTabWidget, QTextBrowser, QVBoxLayout,
                               QWidget)

from ..analysis import TechnicalAnalysis
from ..strategy import COMPONENTS, LABEL_MEANING, StrategyParams
from .chart import CandleChart, EquityChart
from .theme import LABEL_TONE, TONE, C, fmt_price
from .widgets import ComponentRow, ScoreBar, StatTile, card, fill_table, label, make_table, set_color

HOW_IT_WORKS = f"""
<style> h3 {{ color: {C.text}; margin: 10px 0 4px 0; }} p, li {{ color: {C.text2}; }} b {{ color: {C.text}; }} </style>
<h3>1 · Indicators</h3>
<ul>
<li><b>Moving averages (SMA)</b> - the average closing price over the last N bars. Price above a rising
average = uptrend. The fast average crossing above the slow one is a <i>golden cross</i>; below is a <i>death cross</i>.</li>
<li><b>MACD</b> - the gap between a 12- and 26-bar exponential average. When it rises above its own 9-bar
average (the signal line), momentum is improving.</li>
<li><b>RSI</b> - compares the size of recent up-moves to down-moves on a 0-100 scale. Above 70 is
"overbought" (stretched), below 30 is "oversold".</li>
<li><b>Bollinger bands</b> - a 20-bar average ± 2 standard deviations. Closing outside a band means price is
unusually stretched, and it often snaps back.</li>
<li><b>ATR</b> - average true range, the typical size of one bar's move. Used to place stops a sensible
distance away: volatile stocks need wider stops.</li>
</ul>
<h3>2 · The signal score (−100 … +100)</h3>
<p>Each indicator votes between −1 (bearish) and +1 (bullish). The votes are weighted (defaults: trend 30,
MACD 25, RSI 20, Bollinger 15, volume 10) and scaled to ±100. Trend and MACD are <b>trend-following</b>
(buy strength); RSI and Bollinger are <b>contrarian at the extremes</b> (fade stretched moves). Mixing the two
styles is deliberate - watch how they argue with each other in the breakdown panel.</p>
<h3>3 · The trading rules</h3>
<ul>
<li><b>Buy</b> when the score crosses <i>up through</i> the buy threshold while not holding.</li>
<li><b>Sell</b> when the score falls to the sell threshold, <i>or</i> the ATR trailing stop is hit.</li>
<li>Decisions use the bar's close and are filled at the <b>next bar's open</b> - the simulation never trades
on a price it could not have known. Commission is charged on each side.</li>
<li>Long-only and all-in: the simulation is either fully invested or fully in cash.</li>
</ul>
<h3>4 · Target and stop-loss on a buy</h3>
<p>When the call is BUY or STRONG BUY, the signal card and chart show a trade plan. The <b>stop-loss</b> sits
<i>ATR stop ×</i> ATRs below the close - the same distance the simulation's trailing stop starts at. The
<b>target</b> is <i>Target (× risk)</i> times that distance above the close: at 2, you aim to win twice what you
risk (a 2 : 1 reward-to-risk ratio, a common rule of thumb). The simulation itself doesn't sell at the target -
it lets winners run until a sell signal or the trailing stop - so treat the target as a planning level.</p>
<h3>5 · Reading the simulation honestly</h3>
<p>Always compare against <b>buy &amp; hold</b>. In strong bull markets most timing strategies lag buy &amp; hold
because they sit in cash during pullbacks; their value shows up as <b>smaller drawdowns</b> in bear markets.
A handful of trades is not statistically meaningful, and parameters tuned until the past looks great
(<i>overfitting</i>) rarely work as well on the future. Try the same settings across many symbols and timeframes.</p>
"""


def _pct(v: float, signed: bool = True) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:+.1%}" if signed else f"{v:.1%}"


def _num(v: float, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return "∞" if math.isinf(v) else f"{v:.{digits}f}"


class TechnicalScreen(QWidget):
    params_applied = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ta: TechnicalAnalysis | None = None
        self._selected_ts: pd.Timestamp | None = None
        self._currency = "USD"

        # ---------------- chart column
        self.chart = CandleChart()
        self.chart.bar_clicked.connect(self._on_bar_clicked)
        toggles = QHBoxLayout()
        toggles.setContentsMargins(10, 6, 10, 0)
        self._toggles: dict[str, QCheckBox] = {}
        for key, text in (("bands", "Bollinger bands"), ("volume", "Volume"), ("rsi", "RSI"),
                          ("macd", "MACD"), ("score", "Signal score")):
            cb = QCheckBox(text)
            cb.setChecked(True)
            cb.toggled.connect(lambda _=False: self._redraw_chart(keep_view=True))
            self._toggles[key] = cb
            toggles.addWidget(cb)
        toggles.addStretch(1)
        toggles.addWidget(label("Click a candle to see what the model said then · double-click for latest",
                                name="Muted"))
        chart_box, chart_lay = card(spacing=0)
        chart_lay.setContentsMargins(4, 4, 4, 4)
        chart_lay.addLayout(toggles)
        chart_lay.addWidget(self.chart, 1)

        # ---------------- side column
        side = QWidget()
        side_lay = QVBoxLayout(side)
        side_lay.setContentsMargins(0, 0, 4, 0)
        side_lay.setSpacing(10)
        side_lay.addWidget(self._build_signal_card())
        side_lay.addWidget(self._build_components_card())
        side_lay.addWidget(self._build_levels_card())
        side_lay.addWidget(self._build_settings_card())
        side_lay.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(side)
        scroll.setMinimumWidth(360)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        top = QSplitter(Qt.Horizontal)
        top.addWidget(chart_box)
        top.addWidget(scroll)
        top.setStretchFactor(0, 1)
        top.setStretchFactor(1, 0)
        top.setSizes([1000, 380])

        vsplit = QSplitter(Qt.Vertical)
        vsplit.addWidget(top)
        vsplit.addWidget(self._build_bottom_tabs())
        vsplit.setStretchFactor(0, 3)
        vsplit.setStretchFactor(1, 1)
        vsplit.setSizes([640, 280])

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.addWidget(vsplit)

    # ------------------------------------------------------------------ building

    def _build_signal_card(self):
        frame, lay = card("Technical signal")
        self.asof = label("", name="Muted")
        self.verdict = label("—", size=24, bold=True)
        row = QHBoxLayout()
        row.addWidget(self.verdict)
        row.addStretch(1)
        self.score_value = label("", size=15, bold=True)
        row.addWidget(self.score_value)
        self.score_bar = ScoreBar(-100, 100, height=10)
        self.meaning = label("", name="Secondary", wrap=True)
        self.position = label("", wrap=True)
        lay.addWidget(self.asof)
        lay.addLayout(row)
        lay.addWidget(self.score_bar)
        lay.addWidget(self.meaning)
        lay.addWidget(self.position)

        # Trade plan - shown only when the call is BUY / STRONG BUY
        self.plan_box = QFrame()
        self.plan_box.setObjectName("RuleBox")
        grid = QGridLayout(self.plan_box)
        grid.setContentsMargins(10, 8, 10, 8)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(5)
        grid.addWidget(label("TRADE PLAN", name="CardTitle"), 0, 0, 1, 2)
        self.plan: dict[str, QLabel] = {}
        items = [("entry", "Entry (next open ≈)", "Buys are filled at the next bar's open, so today's close is the "
                                                    "best estimate of the entry price."),
                 ("target", "Target (take profit)", "Close + Target (× risk) × the distance to the stop-loss."),
                 ("stop", "Stop-loss (sell if it falls)", "Close − ATR stop × ATR: the same distance the "
                                                          "simulation's trailing stop starts at."),
                 ("rr", "Reward : risk", "How much you aim to make for every 1 you risk losing.")]
        for r, (key, text, tip) in enumerate(items, start=1):
            name = label(text, name="Secondary")
            name.setToolTip(tip)
            value = label("—", bold=True)
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value.setTextFormat(Qt.RichText)
            grid.addWidget(name, r, 0)
            grid.addWidget(value, r, 1)
            self.plan[key] = value
        self.plan_note = label("", name="Muted", wrap=True)
        self.plan_note.setStyleSheet(f"color: {C.muted}; font-size: 8.5pt;")
        grid.addWidget(self.plan_note, len(items) + 1, 0, 1, 2)
        lay.addWidget(self.plan_box)
        self.plan_hint = label("Target and stop-loss appear here when the call is BUY or STRONG BUY.",
                               name="Muted", wrap=True)
        self.plan_hint.setStyleSheet(f"color: {C.muted}; font-size: 8.5pt;")
        lay.addWidget(self.plan_hint)
        return frame

    def _build_components_card(self):
        frame, lay = card("What's driving the score")
        self.rows = {key: ComponentRow(name) for key, name in COMPONENTS.items()}
        for row in self.rows.values():
            lay.addWidget(row)
        return frame

    def _build_levels_card(self):
        frame, lay = card("Key levels")
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        self.levels = {}
        items = [("close", "Close", "Closing price of the inspected bar."),
                 ("atr", "ATR (volatility)", "Average true range: the typical size of one bar's move."),
                 ("stop", "Suggested stop", "Close minus the ATR stop multiple (or the recent low when the ATR "
                                            "stop is off) - where a new long position would be exited if wrong."),
                 ("support", "Support (recent low)", "Lowest low of the lookback window - buyers stepped in here."),
                 ("resistance", "Resistance (recent high)", "Highest high of the lookback window - sellers "
                                                            "stepped in here.")]
        for r, (key, text, tip) in enumerate(items):
            name = label(text, name="Secondary")
            name.setToolTip(tip)
            value = label("—", bold=True)
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(name, r, 0)
            grid.addWidget(value, r, 1)
            self.levels[key] = value
        lay.addLayout(grid)
        return frame

    def _build_settings_card(self):
        frame, lay = card("Strategy settings")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(6)
        d = StrategyParams()

        def ispin(lo, hi, val, tip=""):
            w = QSpinBox()
            w.setRange(lo, hi)
            w.setValue(val)
            w.setToolTip(tip)
            return w

        def dspin(lo, hi, val, step, tip="", decimals=1):
            w = QDoubleSpinBox()
            w.setRange(lo, hi)
            w.setDecimals(decimals)
            w.setSingleStep(step)
            w.setValue(val)
            w.setToolTip(tip)
            return w

        self.inputs = {
            "sma_fast": ispin(2, 400, d.sma_fast, "Bars in the fast moving average"),
            "sma_slow": ispin(3, 400, d.sma_slow, "Bars in the slow moving average"),
            "rsi_period": ispin(2, 100, d.rsi_period, "Bars used for RSI"),
            "buy_threshold": dspin(-100, 100, d.buy_threshold, 5, "Buy when the score crosses up through this"),
            "sell_threshold": dspin(-100, 100, d.sell_threshold, 5, "Sell when the score falls to this"),
            "atr_stop_mult": dspin(0, 10, d.atr_stop_mult, 0.5, "Trailing stop distance in ATRs (0 = no stop)"),
            "target_r": dspin(0.5, 10, d.target_r, 0.5, "Profit target as a multiple of the risk (entry − stop). "
                                                        "2 = aim to win twice what you risk."),
        }
        names = {"sma_fast": "Fast SMA", "sma_slow": "Slow SMA", "rsi_period": "RSI period",
                 "buy_threshold": "Buy threshold", "sell_threshold": "Sell threshold",
                 "atr_stop_mult": "ATR stop ×", "target_r": "Target (× risk)"}
        for key, w in self.inputs.items():
            form.addRow(names[key], w)

        self.capital = dspin(100, 1e9, 10_000, 1000, "Starting cash for the simulation", decimals=0)
        self.commission = dspin(0, 2, 0.1, 0.05, "Commission charged on each buy and each sell (%)", decimals=2)
        form.addRow("Starting capital", self.capital)
        form.addRow("Commission % / side", self.commission)
        lay.addLayout(form)

        weights_title = label("Score weights (relative)", name="Secondary")
        lay.addWidget(weights_title)
        wform = QGridLayout()
        wform.setHorizontalSpacing(8)
        self.weights = {}
        for i, (key, text) in enumerate((("w_trend", "Trend"), ("w_macd", "MACD"), ("w_rsi", "RSI"),
                                         ("w_bb", "Bollinger"), ("w_volume", "Volume"))):
            w = ispin(0, 100, int(getattr(d, key)), f"How much the {text} vote counts")
            self.weights[key] = w
            wform.addWidget(label(text, name="Muted"), (i // 3) * 2, i % 3)
            wform.addWidget(w, (i // 3) * 2 + 1, i % 3)
        lay.addLayout(wform)

        buttons = QHBoxLayout()
        apply_btn = QPushButton("Apply && re-run")
        apply_btn.setObjectName("Primary")
        apply_btn.clicked.connect(self.params_applied.emit)
        reset_btn = QPushButton("Defaults")
        reset_btn.clicked.connect(self._reset_params)
        buttons.addWidget(apply_btn, 1)
        buttons.addWidget(reset_btn)
        lay.addLayout(buttons)
        return frame

    def _build_bottom_tabs(self):
        tabs = QTabWidget()

        sim = QWidget()
        sim_lay = QHBoxLayout(sim)
        sim_lay.setContentsMargins(10, 10, 10, 10)
        left = QVBoxLayout()
        grid = QGridLayout()
        grid.setSpacing(8)
        self.tiles = {}
        specs = [("total_return", "Strategy return", "Total % gain of the simulated account"),
                 ("buy_hold_return", "Buy & hold", "What simply buying on day one and holding would have made"),
                 ("max_drawdown", "Max drawdown", "Largest peak-to-trough fall of the strategy's account"),
                 ("buy_hold_max_drawdown", "Buy & hold drawdown", "Largest peak-to-trough fall of buy & hold"),
                 ("trades", "Trades", "Number of round trips (including an open one)"),
                 ("win_rate", "Win rate", "Share of closed trades that made money"),
                 ("avg_win", "Avg win", "Average % gain on winning trades"),
                 ("avg_loss", "Avg loss", "Average % loss on losing trades"),
                 ("profit_factor", "Profit factor", "Gross profit ÷ gross loss (>1 = profitable)"),
                 ("exposure", "Time in market", "Share of bars the strategy held a position"),
                 ("sharpe", "Sharpe ratio", "Return per unit of volatility, annualised (>1 is good)"),
                 ("final_equity", "Final equity", "Account value at the end")]
        for i, (key, title, tip) in enumerate(specs):
            tile = StatTile(title, tip)
            self.tiles[key] = tile
            grid.addWidget(tile, i // 4, i % 4)
        left.addLayout(grid)
        self.sim_note = label("", name="Muted", wrap=True)
        left.addWidget(self.sim_note)
        left.addStretch(1)
        sim_lay.addLayout(left, 3)
        self.equity_chart = EquityChart()
        sim_lay.addWidget(self.equity_chart, 2)
        tabs.addTab(sim, "Simulation results")

        self.trades_table = make_table(["#", "Entry", "Entry price", "Exit", "Exit price", "Return", "P&L",
                                        "Bars", "Exit reason"])
        self.trades_table.cellClicked.connect(lambda r, _c: self._jump_from_table(self.trades_table, r))
        tabs.addTab(self.trades_table, "Trades")

        self.events_table = make_table(["Date", "Event", "Bias", "Detail"])
        self.events_table.cellClicked.connect(lambda r, _c: self._jump_from_table(self.events_table, r))
        tabs.addTab(self.events_table, "Indicator events")

        how = QTextBrowser()
        how.setHtml(HOW_IT_WORKS)
        how.setOpenExternalLinks(False)
        tabs.addTab(how, "How it works")
        return tabs

    # ------------------------------------------------------------------ settings

    def settings(self) -> tuple[StrategyParams, float, float]:
        values = {k: w.value() for k, w in self.inputs.items()}
        values.update({k: float(w.value()) for k, w in self.weights.items()})
        if values["sma_fast"] >= values["sma_slow"]:
            values["sma_fast"] = max(2, values["sma_slow"] - 1)
            self.inputs["sma_fast"].setValue(values["sma_fast"])
        if values["sell_threshold"] >= values["buy_threshold"]:
            values["sell_threshold"] = values["buy_threshold"] - 5
            self.inputs["sell_threshold"].setValue(values["sell_threshold"])
        if not any(self.weights[k].value() for k in self.weights):
            self.weights["w_trend"].setValue(1)
            values["w_trend"] = 1.0
        return StrategyParams().copy(**values), self.capital.value(), self.commission.value()

    def _reset_params(self):
        d = StrategyParams()
        for k, w in self.inputs.items():
            w.setValue(getattr(d, k))
        for k, w in self.weights.items():
            w.setValue(int(getattr(d, k)))
        self.capital.setValue(10_000)
        self.commission.setValue(0.1)
        self.params_applied.emit()

    # ------------------------------------------------------------------ updates

    def set_currency(self, currency: str):
        self._currency = currency

    def show_message(self, text: str, color: str = C.text2):
        self.chart.show_message(text, color)

    def set_analysis(self, ta: TechnicalAnalysis, keep_view: bool = False):
        self._ta = ta
        if not keep_view or (self._selected_ts is not None and self._selected_ts not in ta.data.index):
            self._selected_ts = None
        self._redraw_chart(keep_view)
        self._update_side()
        self._update_simulation()
        self._update_tables()

    def _selected_index(self) -> int | None:
        if self._ta is None or self._selected_ts is None:
            return None
        return int(self._ta.data.index.get_loc(self._selected_ts))

    def _redraw_chart(self, keep_view: bool = False):
        if self._ta is None:
            return
        panels = [k for k in ("volume", "rsi", "macd", "score") if self._toggles[k].isChecked()]
        snap = self._ta.snapshot(self._selected_index())
        plan = (snap.index, snap.target, snap.stop) if self._is_buy(snap) else None
        self.chart.plot(self._ta, panels, bands=self._toggles["bands"].isChecked(),
                        selected=self._selected_index(), keep_view=keep_view, plan=plan)

    @staticmethod
    def _is_buy(snap) -> bool:
        return snap.label in ("BUY", "STRONG BUY") and math.isfinite(snap.target) and math.isfinite(snap.stop)

    def _on_bar_clicked(self, i: int):
        if self._ta is None:
            return
        self._selected_ts = None if i < 0 or i == len(self._ta.data) - 1 else self._ta.data.index[i]
        self._redraw_chart(keep_view=True)
        self._update_side()

    def _jump_from_table(self, table, row: int):
        item = table.item(row, 0)
        ts = item.data(Qt.UserRole) if item else None
        if self._ta is not None and ts is not None and ts in self._ta.data.index:
            self._on_bar_clicked(int(self._ta.data.index.get_loc(ts)))

    def _update_side(self):
        ta = self._ta
        snap = ta.snapshot(self._selected_index())
        cur = self._currency
        when = self.chart._labels[snap.index] if self.chart._labels else str(snap.time)
        self.asof.setText(("Latest bar · " if snap.is_latest else "Inspecting past bar · ") + when)
        tone = TONE[LABEL_TONE[snap.label]]
        self.verdict.setText(snap.label)
        set_color(self.verdict, tone, size=24, bold=True)
        self.score_value.setText(f"{snap.score:+.0f}")
        set_color(self.score_value, tone, size=15, bold=True)
        self.score_bar.set_value(snap.score, tone, [(ta.params.buy_threshold, C.up), (ta.params.sell_threshold, C.down)])
        self.meaning.setText(LABEL_MEANING[snap.label])

        bt = ta.backtest
        lines = []
        if snap.trade is not None:
            t = snap.trade
            ret = snap.close * (1 - bt.commission) * t.shares / t.cost - 1
            lines.append(f"<b>Strategy position:</b> <span style='color:{C.up}'>LONG</span> since "
                         f"{self.chart._labels[t.entry_idx]} @ {fmt_price(t.entry_price, cur)} "
                         f"(<span style='color:{C.up if ret >= 0 else C.down}'>{ret:+.1%}</span>)")
        else:
            lines.append(f"<b>Strategy position:</b> flat (in cash) - waiting for the score to cross "
                         f"{ta.params.buy_threshold:+g}")
        if snap.signal:
            color = C.up if snap.signal == "buy" else C.down
            lines.append(f"<span style='color:{color}'><b>{snap.signal.upper()} signal fired on this bar</b></span> "
                         f"→ filled at the next bar's open")
        self.position.setText("<br>".join(lines))
        self.position.setTextFormat(Qt.RichText)

        show_plan = self._is_buy(snap)
        self.plan_box.setVisible(show_plan)
        self.plan_hint.setVisible(not show_plan)
        if show_plan:
            c = snap.close
            self.plan["entry"].setText(fmt_price(c, cur))
            self.plan["target"].setText(f"<span style='color:{C.up}'>{fmt_price(snap.target, cur)}</span>"
                                        f"  ({snap.target / c - 1:+.1%})")
            self.plan["stop"].setText(f"<span style='color:{C.down}'>{fmt_price(snap.stop, cur)}</span>"
                                      f"  ({snap.stop / c - 1:+.1%})")
            self.plan["rr"].setText(f"{(snap.target - c) / (c - snap.stop):.1f} : 1")
            notes = []
            if c < snap.resistance < snap.target:
                notes.append(f"The recent high at {fmt_price(snap.resistance, cur)} may act as a first hurdle "
                             "before the target.")
            if ta.params.atr_stop_mult > 0:
                notes.append("The stop then trails up behind the price, as in the simulation.")
            else:
                notes.append("ATR stop is off, so the stop-loss uses the recent low.")
            self.plan_note.setText(" ".join(notes))

        for reading in snap.components:
            self.rows[reading.key].set_reading(reading.vote, reading.points, reading.notes)

        self.levels["close"].setText(fmt_price(snap.close, cur))
        self.levels["atr"].setText(f"{fmt_price(snap.atr, cur)}  ({snap.atr / snap.close:.1%})"
                                   if snap.atr == snap.atr else "—")
        self.levels["stop"].setText(f"{fmt_price(snap.stop, cur)}  ({snap.stop / snap.close - 1:+.1%})"
                                    if snap.stop == snap.stop else "off")
        self.levels["support"].setText(f"{fmt_price(snap.support, cur)}  ({snap.support / snap.close - 1:+.1%})")
        self.levels["resistance"].setText(f"{fmt_price(snap.resistance, cur)}  "
                                          f"({snap.resistance / snap.close - 1:+.1%})")

    def _update_simulation(self):
        bt = self._ta.backtest
        s = bt.stats
        beat = s["total_return"] >= s["buy_hold_return"]
        self.tiles["total_return"].set(_pct(s["total_return"]), C.up if s["total_return"] >= 0 else C.down)
        self.tiles["buy_hold_return"].set(_pct(s["buy_hold_return"]), C.text2)
        self.tiles["max_drawdown"].set(_pct(s["max_drawdown"]), C.down)
        self.tiles["buy_hold_max_drawdown"].set(_pct(s["buy_hold_max_drawdown"]), C.text2)
        self.tiles["trades"].set(f"{int(s['trades'])}")
        self.tiles["win_rate"].set(_pct(s["win_rate"], signed=False))
        self.tiles["avg_win"].set(_pct(s["avg_win"]), C.up)
        self.tiles["avg_loss"].set(_pct(s["avg_loss"]), C.down)
        self.tiles["profit_factor"].set(_num(s["profit_factor"]))
        self.tiles["exposure"].set(_pct(s["exposure"], signed=False))
        self.tiles["sharpe"].set(_num(s["sharpe"]))
        self.tiles["final_equity"].set(fmt_price(s["final_equity"], self._currency))
        verdict = ("beat" if beat else "lagged") + " buy & hold"
        dd_better = s["max_drawdown"] > s["buy_hold_max_drawdown"]
        self.sim_note.setText(
            f"Over this window the strategy {verdict} by {abs(s['total_return'] - s['buy_hold_return']):.1%}"
            f"{' with a smaller' if dd_better else ' and had a deeper'} worst drawdown. "
            f"Long-only · decided on close, filled next open · {bt.commission * 100:.2f}% commission per side · "
            f"{'ATR trailing stop ×%g' % self._ta.params.atr_stop_mult if self._ta.params.atr_stop_mult else 'no stop'}. "
            "Past results don't predict future ones.")
        self.equity_chart.plot(self._ta)

    def _update_tables(self):
        ta, cur = self._ta, self._currency
        labels = self.chart._labels
        last_close = float(ta.data["Close"].iloc[-1])
        fee = ta.backtest.commission
        rows, colors, data = [], [], []
        for k, t in enumerate(reversed(ta.backtest.trades)):
            ret, pnl = t.return_pct(last_close, fee), t.pnl(last_close, fee)
            col = C.up if ret >= 0 else C.down
            rows.append([str(len(ta.backtest.trades) - k), labels[t.entry_idx], fmt_price(t.entry_price, cur),
                         labels[t.exit_idx] if not t.is_open else "open",
                         fmt_price(t.exit_price, cur) if not t.is_open else fmt_price(last_close, cur),
                         f"{ret:+.2%}", f"{pnl:+,.2f}", str(t.bars_held(len(ta.data) - 1)),
                         t.exit_reason or "Still open (marked to last close)"])
            colors.append([None, None, None, None, None, col, col, None, C.warn if t.exit_kind == "stop" else None])
            data.append(t.entry_time)
        fill_table(self.trades_table, rows, colors, data, align_right={2, 4, 5, 6, 7})

        ev = ta.events.head(400)
        rows = [[labels[ta.data.index.get_loc(r.time)], r.event, r.bias, r.detail] for r in ev.itertuples()]
        colors = [[None, None, C.up if r.bias == "Bullish" else C.down, None] for r in ev.itertuples()]
        fill_table(self.events_table, rows, colors, list(ev["time"]))
