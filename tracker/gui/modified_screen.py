"""Screen 3 - Modified Strat: build your own strategy, train it on history, test it on unseen data,
then get a buy / no-buy signal with a target, a stop-loss and an estimated time to target."""
from __future__ import annotations

import math
import time

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QFrame, QGridLayout, QHBoxLayout,
                               QProgressBar, QPushButton, QScrollArea, QSlider, QSpinBox, QSplitter, QTabWidget,
                               QTextBrowser, QVBoxLayout, QWidget)

from ..custom_strategy import (ALL_PARAMS, DEFAULT_VALUES, EXIT_PARAMS, OBJECTIVES, RULES, ParamSpec,
                               StrategyReport, TrainConfig, fmt_value, overfit_verdict, refresh_report,
                               space_size, train_strategy)
from ..data import TRAIN_HISTORY, fetch_history, train_timeframe
from ..forecast import bar_unit, date_after
from .strategy_chart import StrategyChart, TimeToTargetChart
from .theme import TONE, C, fmt_price
from .widgets import ProbBar, Worker, card, fill_table, label, make_table, set_color

ACCOUNT, RISK_PCT = 10_000, 0.01
ACTION_STYLE = {"BUY": ("BUY", C.up), "HOLD": ("HOLD · IN TRADE", C.accent),
                "WAIT": ("WAIT", C.warn), "NO BUY": ("NO BUY", C.text2)}
DATA_TTL = 10 * 60

HOW_IT_WORKS = f"""
<style> h3 {{ color: {C.text}; margin: 10px 0 4px 0; }} p, li {{ color: {C.text2}; }} b {{ color: {C.text}; }} </style>
<h3>1 · Build</h3>
<p>Tick the entry rules you believe in. The strategy buys only when <b>all</b> ticked rules are true on a
day's close, and fills the order at the next day's open. After a trade, the rules must switch off and on
again before it buys again - no instant re-buying.</p>
<p>The <b>exit plan</b> is sized with ATR (average true range = a typical day's move), so volatile stocks
automatically get wider targets and stops: target = entry + <i>k</i> × ATR, stop-loss = entry − <i>k</i> × ATR.
A trailing stop follows the highest close upward. A time limit sells after N bars if neither is hit.</p>
<h3>2 · Train</h3>
<p>Every parameter with <b>optimise</b> ticked is searched over the listed values. The optimiser only
sees the <b>training</b> period (the older part of the history) and ranks every combination by your chosen
objective. Combinations with too few trades are rejected - a result from 2 trades is luck, not skill.</p>
<h3>3 · Test - the honest part</h3>
<p>The winner is replayed on the <b>test</b> period: the most recent data, which the optimiser never saw.
This is how professionals catch <b>overfitting</b> - rules that memorised the past instead of learning
something real. If test results collapse, the strategy is unlikely to work going forward. Try the same
rules on several stocks; a robust idea works across many.</p>
<h3>4 · Signal, target, stop and time</h3>
<ul>
<li><b>BUY</b> - all rules turned true on the latest close. <b>HOLD</b> - the strategy is already in a trade.
<b>WAIT</b> - the setup is active but was already traded. <b>NO BUY</b> - some rules are not met.</li>
<li><b>Time to target</b> is estimated three ways: (1) 4,000 simulated futures built by re-sampling this
stock's real past daily moves; (2) the coin-flip baseline - how often a trendless random walk would reach the
target before the stop (it depends only on the target/stop distances); (3) how this strategy's own past
trades actually played out.</li>
<li><b>Position size</b> shows how many shares keep the loss at the stop to 1% of a $10,000 account - the
classic way professionals size trades.</li>
</ul>
<p><i>Everything here is an estimate from past data. Markets change; nothing guarantees the future.</i></p>
"""

METRIC_ROWS = [("Return", "total_return", "pct"), ("Buy & hold", "buy_hold", "pct"), ("Trades", "trades", "int"),
               ("Win rate", "win_rate", "pct0"), ("Avg trade", "avg_trade", "pct"),
               ("Profit factor", "profit_factor", "num"), ("Max drawdown", "max_drawdown", "pct"),
               ("Sharpe ratio", "sharpe", "num"), ("Target hit rate", "target_rate", "pct0"),
               ("Time in market", "exposure", "pct0"), ("Avg bars held", "avg_bars", "num1")]


def _fmt(v, kind: str) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, float) and math.isinf(v):
        return "∞"
    return {"pct": f"{v:+.1%}", "pct0": f"{v:.0%}", "int": f"{int(v)}", "num": f"{v:.2f}",
            "num1": f"{v:.1f}"}[kind]


class ModifiedScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._symbol: str | None = None
        self._currency = "USD"
        self._report: StrategyReport | None = None
        self._data: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}
        self._token = 0
        self._workers: dict[int, Worker] = {}
        self._busy = False
        self._auto_train = False
        self._jobs: dict[int, str] = {}

        body = QSplitter(Qt.Horizontal)
        body.addWidget(self._build_builder())
        right = QSplitter(Qt.Vertical)
        right.addWidget(self._build_signal_card())
        chart_box, chart_lay = card(spacing=0)
        chart_lay.setContentsMargins(4, 4, 4, 4)
        self.chart = StrategyChart()
        chart_lay.addWidget(self.chart)
        right.addWidget(chart_box)
        right.addWidget(self._build_tabs())
        right.setStretchFactor(1, 1)
        right.setSizes([285, 410, 235])
        body.addWidget(right)
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        body.setSizes([400, 1100])

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.addWidget(body)
        self._update_space()

    # ------------------------------------------------------------------ builder column

    def _build_builder(self):
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 0, 4, 0)
        lay.setSpacing(10)
        self.rule_boxes: dict[str, QCheckBox] = {}
        self.param_widgets: dict[str, QWidget] = {}
        self.opt_boxes: dict[str, QCheckBox] = {}
        self._rule_params: dict[str, list[str]] = {}
        self._tries: dict[str, QWidget] = {}

        frame, fl = card("1 · Entry rules — buy when ALL ticked rules are true")
        fl.addWidget(label("Your values are the starting point. Tick “optimise” to let training try the listed "
                           "alternatives too.", name="Muted", wrap=True))
        for rule in RULES:
            box = QFrame()
            box.setObjectName("RuleBox")
            bl = QVBoxLayout(box)
            bl.setContentsMargins(10, 8, 10, 8)
            bl.setSpacing(4)
            cb = QCheckBox(rule.name)
            cb.setStyleSheet("font-weight: 700;")
            cb.setChecked(rule.default_on)
            cb.toggled.connect(lambda on, k=rule.key: self._rule_toggled(k, on))
            self.rule_boxes[rule.key] = cb
            bl.addWidget(cb)
            about = label(rule.about, name="Muted", wrap=True)
            about.setStyleSheet(f"color: {C.muted}; font-size: 8.5pt;")
            bl.addWidget(about)
            if rule.params:
                grid = QGridLayout()
                grid.setHorizontalSpacing(8)
                grid.setVerticalSpacing(2)
                for r, spec in enumerate(rule.params):
                    self._param_row(grid, r, spec)
                bl.addLayout(grid)
            self._rule_params[rule.key] = [p.key for p in rule.params]
            fl.addWidget(box)
        lay.addWidget(frame)

        frame, el = card("2 · Exit plan — when to sell")
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)
        for r, spec in enumerate(EXIT_PARAMS):
            self._param_row(grid, r, spec)
        el.addLayout(grid)
        lay.addWidget(frame)

        frame, tl = card("3 · Train & optimise")
        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(6)
        self.history = QComboBox()
        self.history.addItems(list(TRAIN_HISTORY))
        self.history.setCurrentText("5 years")
        self.history.setToolTip("How much daily price history to learn from")
        form.addRow("Price history", self.history)
        split_row = QHBoxLayout()
        self.split = QSlider(Qt.Horizontal)
        self.split.setRange(50, 85)
        self.split.setValue(70)
        self.split_lbl = label("", name="Secondary")
        self.split.valueChanged.connect(self._update_split_label)
        split_row.addWidget(self.split, 1)
        split_row.addWidget(self.split_lbl)
        form.addRow("Train / test", split_row)
        self.objective = QComboBox()
        self.objective.addItems(list(OBJECTIVES))
        self.objective.setToolTip("What “best” means when ranking combinations in training")
        form.addRow("Optimise for", self.objective)
        self.iterations = QSpinBox()
        self.iterations.setRange(10, 5000)
        self.iterations.setSingleStep(100)
        self.iterations.setValue(500)
        self.iterations.setToolTip("How many combinations to try. More = better fit to the past, and more risk "
                                   "of overfitting.")
        self.iterations.valueChanged.connect(self._update_space)
        form.addRow("Max combinations", self.iterations)
        self.min_trades = QSpinBox()
        self.min_trades.setRange(1, 100)
        self.min_trades.setValue(5)
        self.min_trades.setToolTip("Reject combinations with fewer trades than this in training")
        form.addRow("Min trades", self.min_trades)
        self.commission = QDoubleSpinBox()
        self.commission.setRange(0, 2)
        self.commission.setDecimals(2)
        self.commission.setSingleStep(0.05)
        self.commission.setValue(0.10)
        form.addRow("Commission % / side", self.commission)
        tl.addLayout(form)
        self.space_lbl = label("", name="Muted", wrap=True)
        tl.addWidget(self.space_lbl)
        buttons = QHBoxLayout()
        self.train_btn = QPushButton("Train && optimise")
        self.train_btn.setObjectName("Primary")
        self.train_btn.clicked.connect(self.train)
        reset = QPushButton("Defaults")
        reset.clicked.connect(self._reset)
        buttons.addWidget(self.train_btn, 1)
        buttons.addWidget(reset)
        tl.addLayout(buttons)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        tl.addWidget(self.progress)
        self.status = label("", name="Muted", wrap=True)
        tl.addWidget(self.status)
        lay.addWidget(frame)
        lay.addStretch(1)
        self._update_split_label()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        scroll.setMinimumWidth(390)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        for key, cb in self.rule_boxes.items():
            self._rule_toggled(key, cb.isChecked())
        return scroll

    def _param_row(self, grid: QGridLayout, r: int, spec: ParamSpec):
        name = label(spec.label, name="Secondary")
        name.setToolTip(spec.tip)
        if spec.key == "trailing":
            w = QCheckBox("on")
            w.setChecked(bool(spec.default))
            w.toggled.connect(self._update_space)
        elif spec.decimals == 0:
            w = QSpinBox()
            w.setRange(int(spec.lo), int(spec.hi))
            w.setValue(int(spec.default))
            w.valueChanged.connect(self._update_space)
        else:
            w = QDoubleSpinBox()
            w.setRange(spec.lo, spec.hi)
            w.setDecimals(spec.decimals)
            w.setSingleStep(spec.step)
            w.setValue(spec.default)
            w.valueChanged.connect(self._update_space)
        w.setToolTip(spec.tip)
        opt = QCheckBox("optimise")
        opt.setChecked(True)
        opt.toggled.connect(self._update_space)
        tries = label("tries " + " · ".join(fmt_value(spec.key, float(v)) for v in spec.choices), name="Muted")
        tries.setStyleSheet(f"color: {C.muted}; font-size: 8pt;")
        grid.addWidget(name, 2 * r, 0)
        grid.addWidget(w, 2 * r, 1)
        grid.addWidget(opt, 2 * r, 2)
        grid.addWidget(tries, 2 * r + 1, 1, 1, 2)
        grid.setColumnStretch(1, 1)
        self.param_widgets[spec.key] = w
        self.opt_boxes[spec.key] = opt
        self._tries[spec.key] = tries

    def _rule_toggled(self, key: str, on: bool):
        for pk in self._rule_params.get(key, []):
            for w in (self.param_widgets[pk], self.opt_boxes[pk], self._tries[pk]):
                w.setEnabled(on)
        self._update_space()

    def _update_split_label(self):
        v = self.split.value()
        self.split_lbl.setText(f"{v}% / {100 - v}%")

    def config(self) -> TrainConfig:
        values = {k: float(w.isChecked()) if isinstance(w, QCheckBox) else float(w.value())
                  for k, w in self.param_widgets.items()}
        return TrainConfig(
            rules=tuple(k for k, cb in self.rule_boxes.items() if cb.isChecked()),
            values=values,
            optimise=tuple(k for k, cb in self.opt_boxes.items() if cb.isChecked()),
            objective=self.objective.currentText(),
            iterations=self.iterations.value(),
            train_frac=self.split.value() / 100,
            min_trades=self.min_trades.value(),
            commission_pct=self.commission.value(),
        )

    def _update_space(self, *_):
        if not hasattr(self, "train_btn"):
            return  # still building the widgets
        cfg = self.config()
        if not cfg.rules:
            self.space_lbl.setText("Tick at least one entry rule.")
            set_color(self.space_lbl, C.down)
            self.train_btn.setEnabled(False)
            return
        self.train_btn.setEnabled(not self._busy)
        total = space_size(cfg)
        how = "all of them" if total <= cfg.iterations else f"{cfg.iterations:,} at random"
        self.space_lbl.setText(f"Search space: {total:,} combination{'s' if total != 1 else ''} → testing {how}.")
        set_color(self.space_lbl, C.muted)

    def _reset(self):
        for key, cb in self.rule_boxes.items():
            cb.setChecked(any(r.key == key and r.default_on for r in RULES))
        for key, w in self.param_widgets.items():
            if isinstance(w, QCheckBox):
                w.setChecked(bool(DEFAULT_VALUES[key]))
            else:
                w.setValue(DEFAULT_VALUES[key])
            self.opt_boxes[key].setChecked(True)

    def _load_values(self, values: dict):
        for key, w in self.param_widgets.items():
            if isinstance(w, QCheckBox):
                w.setChecked(bool(values[key]))
            else:
                w.setValue(values[key])

    # ------------------------------------------------------------------ signal card

    def _build_signal_card(self):
        frame, lay = card(spacing=4)
        row = QHBoxLayout()
        row.setSpacing(18)

        col = QVBoxLayout()
        col.setSpacing(4)
        col.addWidget(label("OPTIMISED STRATEGY SIGNAL", name="CardTitle"))
        self.asof = label("", name="Muted")
        self.verdict = label("—", size=24, bold=True)
        self.detail = label("", name="Secondary", wrap=True)
        self.conditions = label("", wrap=True)
        self.conditions.setTextFormat(Qt.RichText)
        for w in (self.asof, self.verdict, self.detail, self.conditions):
            col.addWidget(w)
        col.addStretch(1)
        row.addLayout(col, 5)
        row.addWidget(self._vline())

        col = QVBoxLayout()
        col.setSpacing(4)
        col.addWidget(label("TRADE PLAN", name="CardTitle"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(5)
        self.plan: dict[str, tuple] = {}
        items = [("entry", "Entry"), ("target", "Target (take profit)"), ("stop", "Stop-loss (sell if it falls)"),
                 ("trail", "Trailing stop"), ("rr", "Reward : risk"), ("limit", "Time limit"),
                 ("size", "Size for 1% risk")]
        for r, (key, text) in enumerate(items):
            n = label(text, name="Secondary")
            v = label("—", bold=True)
            v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(n, r, 0)
            grid.addWidget(v, r, 1)
            self.plan[key] = (n, v)
        grid.setColumnStretch(1, 1)
        col.addLayout(grid)
        col.addStretch(1)
        row.addLayout(col, 4)
        row.addWidget(self._vline())

        col = QVBoxLayout()
        col.setSpacing(4)
        col.addWidget(label("TIME TO TARGET", name="CardTitle"))
        self.eta = label("—", size=20, bold=True, color=C.up)
        self.eta_range = label("", name="Secondary", wrap=True)
        self.prob = ProbBar()
        self.prob_lbl = label("", name="Muted", wrap=True)
        self.prob_lbl.setTextFormat(Qt.RichText)
        self.prob_lbl.setStyleSheet(f"color: {C.muted}; font-size: 8.5pt;")
        self.baseline = label("", name="Secondary", wrap=True)
        self.baseline.setStyleSheet(f"color: {C.text2}; font-size: 9pt;")
        for w in (self.eta, self.eta_range, self.prob, self.prob_lbl, self.baseline):
            col.addWidget(w)
        col.addStretch(1)
        row.addLayout(col, 4)
        lay.addLayout(row)
        return frame

    @staticmethod
    def _vline():
        line = QFrame()
        line.setObjectName("VLine")
        line.setFixedWidth(1)
        return line

    # ------------------------------------------------------------------ result tabs

    def _build_tabs(self):
        tabs = QTabWidget()

        page = QWidget()
        pl = QVBoxLayout(page)
        pl.setContentsMargins(10, 8, 10, 8)
        self.verdict_lbl = label("", wrap=True)
        self.params_lbl = label("", name="Secondary", wrap=True)
        self.params_lbl.setTextFormat(Qt.RichText)
        pl.addWidget(self.verdict_lbl)
        pl.addWidget(self.params_lbl)
        row = QHBoxLayout()
        self.results = make_table(["Metric", "Your settings · train", "Optimised · train",
                                   "Your settings · TEST", "Optimised · TEST"], stretch_last=False)
        row.addWidget(self.results, 1)
        side = QVBoxLayout()
        copy_btn = QPushButton("Copy optimised values\ninto the builder")
        copy_btn.clicked.connect(lambda: self._report and self._load_values(self._report.best))
        side.addWidget(copy_btn)
        side.addStretch(1)
        row.addLayout(side)
        pl.addLayout(row, 1)
        tabs.addTab(page, "Train vs test")

        self.trades = make_table(["#", "Period", "Entry date", "Entry", "Target", "Stop", "Exit date", "Exit",
                                  "Result", "Return", "Bars"])
        tabs.addTab(self.trades, "Trades")

        self.board = make_table(["#", "Parameters", "Train score", "Train return", "Train trades", "TEST return",
                                 "TEST trades", "TEST Sharpe"], stretch_last=False)
        tabs.addTab(self.board, "Leaderboard")

        self.tt_chart = TimeToTargetChart()
        tabs.addTab(self.tt_chart, "Time-to-target distribution")

        how = QTextBrowser()
        how.setHtml(HOW_IT_WORKS)
        tabs.addTab(how, "How it works")
        return tabs

    # ------------------------------------------------------------------ public API (main window)

    def set_currency(self, currency: str):
        self._currency = currency

    def set_symbol(self, symbol: str):
        if symbol == self._symbol:
            return
        self._symbol = symbol
        self._report = None
        self._token += 1          # ignore any training still running for the old symbol
        self._busy = False
        self._update_space()
        self.progress.setValue(0)
        self.status.setText("")
        self.verdict.setText("—")
        set_color(self.verdict, C.text2, size=24, bold=True)
        for w in (self.detail, self.conditions, self.eta_range, self.prob_lbl, self.baseline):
            w.setText("")
        self.asof.setText(symbol)
        self.eta.setText("—")
        self.prob.set_parts([])
        self.chart.show_message(f"Press “Train & optimise” to build a strategy for {symbol}")
        self._auto_train = True
        if self.isVisible():
            self.train()

    def showEvent(self, event):
        super().showEvent(event)
        if self._auto_train and self._symbol and not self._busy:
            self.train()

    def live_refresh(self):
        """Re-apply the optimised rules to fresh prices (no re-training)."""
        rep = self._report
        if rep is None or self._busy or rep.symbol != self._symbol:
            return
        self._start(self._refresh_job, rep, self.history.currentText(), kind="refresh")

    # ------------------------------------------------------------------ training

    def train(self):
        if not self._symbol or self._busy:
            return
        cfg = self.config()
        if not cfg.rules:
            return
        self._auto_train = False
        period = self.history.currentText()
        cached = self._data.get((self._symbol, period))
        df = cached[1] if cached and time.time() - cached[0] < DATA_TTL else None
        self.progress.setValue(0)
        self.status.setText(f"Downloading {period} of daily prices for {self._symbol}…" if df is None else
                            "Training…")
        set_color(self.status, C.muted)
        if self._report is None:
            self.chart.show_message(f"Training a strategy for {self._symbol}…")
        self._start(self._train_job, self._symbol, period, cfg, df, kind="train", with_progress=True)

    @staticmethod
    def _train_job(symbol, period, cfg, df, progress):
        if df is None:
            df = fetch_history(symbol, train_timeframe(period))
        return df, train_strategy(symbol, df, cfg, progress)

    @staticmethod
    def _refresh_job(rep, period):
        df = fetch_history(rep.symbol, train_timeframe(period))
        return df, refresh_report(rep, df)

    def _start(self, fn, *args, kind: str, with_progress: bool = False):
        self._token += 1
        self._busy = True
        self.train_btn.setEnabled(False)
        worker = Worker(self._token, fn, *args, with_progress=with_progress)
        worker.signals.finished.connect(self._on_finished)
        worker.signals.progress.connect(self._on_progress)
        self._workers[self._token] = worker
        self._jobs[self._token] = kind
        worker.start()

    def _on_progress(self, done: int, total: int):
        self.progress.setValue(int(100 * done / max(total, 1)))
        self.status.setText(f"Training… tested {done:,} of {total:,} combinations")

    def _on_finished(self, token: int):
        worker = self._workers.pop(token, None)
        kind = self._jobs.pop(token, "train")
        if worker is None or token != self._token:
            return
        self._busy = False
        self._update_space()
        if worker.error is not None:
            self.status.setText(worker.error)
            set_color(self.status, C.down)
            if self._report is None:
                self.chart.show_message(worker.error, C.down)
            return
        df, rep = worker.result
        self._data[(rep.symbol, self.history.currentText())] = (time.time(), df)
        self._report = rep
        self.progress.setValue(100)
        tr, te = rep.split - rep.start, rep.n - rep.split
        idx = rep.df.index
        msg = (f"Trained on {tr} bars ({idx[rep.start]:%b %Y} – {idx[rep.split - 1]:%b %Y}), tested on {te} unseen "
               f"bars ({idx[rep.split]:%b %Y} – {idx[-1]:%b %Y}). Evaluated {rep.n_evaluated:,} of "
               f"{rep.n_space:,} combinations in {rep.seconds:.1f}s; {rep.n_valid:,} had ≥ {rep.cfg.min_trades} trades.")
        if kind == "refresh":
            msg = f"Signal refreshed with latest prices at {time.strftime('%H:%M:%S')}. " + msg
        self.status.setText(msg + (f"\n⚠ {rep.warning}" if rep.warning else ""))
        set_color(self.status, C.warn if rep.warning else C.muted)
        self._render(rep, keep_view=kind == "refresh")

    # ------------------------------------------------------------------ rendering

    def _render(self, rep: StrategyReport, keep_view: bool = False):
        self._render_signal(rep)
        self.chart.plot(rep, keep_view=keep_view)
        self._render_results(rep)
        self._render_trades(rep)
        self._render_board(rep)
        self.tt_chart.plot(rep)

    def _render_signal(self, rep: StrategyReport):
        sig, f, cur = rep.signal, rep.forecast, self._currency
        idx = rep.df.index
        unit = bar_unit(idx)
        text, color = ACTION_STYLE[sig.action]
        self.asof.setText(f"{rep.symbol} · as of the close on {idx[-1]:%a %d %b %Y}")
        self.verdict.setText(text)
        set_color(self.verdict, color, size=24, bold=True)
        self.detail.setText(sig.detail)
        rows = [f"<span style='color:{C.up if ok else C.down}'>{'✓' if ok else '✗'}</span> "
                f"<b>{name}</b> <span style='color:{C.text2}'>— {txt}</span>" for name, ok, txt in sig.conditions]
        self.conditions.setText(f"<span style='color:{C.muted}'>Entry rules on the latest bar:</span><br>"
                                + "<br>".join(rows))

        hypo = sig.action in ("NO BUY", "WAIT")
        entry_name = "Bought at" if sig.trade is not None else ("Entry (next open ≈)" if not hypo else
                                                                "Entry if bought now")
        self.plan["entry"][0].setText(entry_name)
        entry_txt = fmt_price(sig.entry, cur)
        if sig.trade is not None:
            entry_txt += f"  on {idx[sig.trade.entry_idx]:%d %b}  (now {fmt_price(float(rep.fc.c[-1]), cur)})"
        self.plan["entry"][1].setText(entry_txt)

        def level(v, colour):
            return f"<span style='color:{colour}'>{fmt_price(v, cur)}</span>  ({v / sig.entry - 1:+.1%})"
        self.plan["target"][1].setText(level(sig.target, C.up))
        self.plan["stop"][1].setText(level(sig.stop, C.down))
        v = rep.best
        if v["trailing"]:
            mid = sig.entry + (sig.target - sig.entry) / 2
            self.plan["trail"][1].setText(f"On · at {fmt_price(mid, cur)} → stop {fmt_price(mid - v['sl_atr'] * sig.atr, cur)}")
            self.plan["trail"][1].setToolTip("The stop follows the highest close upward, "
                                             f"{v['sl_atr']:g} ATR behind it, and never moves down.")
        else:
            self.plan["trail"][1].setText("Off (fixed stop)")
        risk, reward = sig.entry - sig.stop, sig.target - sig.entry
        self.plan["rr"][1].setText(f"{reward / risk:.1f} : 1" if risk > 0 else "—")
        self.plan["limit"][1].setText(f"{sig.bars_left} {unit} (by {date_after(idx, sig.bars_left):%d %b})")
        self.plan["limit"][1].setToolTip("Sell at the close on this day if neither the target nor the stop was hit.")
        shares = int(ACCOUNT * RISK_PCT // risk) if risk > 0 else 0
        self.plan["size"][1].setText(f"{shares:,} sh ≈ {fmt_price(shares * sig.entry, cur)} · risk "
                                     f"{fmt_price(shares * risk, cur)}")
        self.plan["size"][1].setToolTip(f"Shares that keep the loss at the stop to 1% of a "
                                        f"{fmt_price(ACCOUNT, cur)} account - the classic way to size a trade.")

        if f is None:
            self.eta.setText("—")
            self.eta_range.setText("Not enough history to simulate.")
            self.prob.set_parts([])
            self.prob_lbl.setText("")
            self.baseline.setText("")
            return
        if f.median_bars:
            self.eta.setText(f"≈ {f.median_bars:.0f} {unit}")
            set_color(self.eta, C.up, size=20, bold=True)
            self.eta_range.setText(f"Target date ≈ {date_after(idx, f.median_bars):%d %b %Y}  ·  usually "
                                   f"{f.q25:.0f}–{f.q75:.0f} {unit} ({date_after(idx, f.q25):%d %b} – "
                                   f"{date_after(idx, f.q75):%d %b})")
        else:
            self.eta.setText("Unlikely in time")
            set_color(self.eta, C.warn, size=20, bold=True)
            self.eta_range.setText(f"No simulated path reached the target within {f.horizon} {unit}.")
        self.prob.set_parts([(f.p_target, C.up), (f.p_stop, C.down), (f.p_neither, C.muted)])
        sq = lambda color: f"<span style='color:{color}'>■</span>"
        self.prob_lbl.setText(f"{sq(C.up)} target first {f.p_target:.0%} &nbsp; {sq(C.down)} stop first "
                              f"{f.p_stop:.0%} &nbsp; {sq(C.muted)} neither {f.p_neither:.0%}")
        self.prob_lbl.setToolTip(f"Out of {f.n_paths:,} simulated futures built from this stock's real past daily "
                                 f"moves, within the {f.horizon}-{unit[:-1]} time limit"
                                 f"{' (the trailing stop is treated as fixed)' if v['trailing'] else ''}.")
        hist = rep.history
        past = (f"Past trades: {hist['targets']}/{hist['closed']} hit the target"
                + (f" (median {hist['median_bars']:.0f} {unit})" if hist["median_bars"] else "")) if hist["closed"] \
            else "No completed past trades yet"
        self.baseline.setText(f"Coin-flip baseline: {f.baseline_p:.0%} · {past}")
        self.baseline.setToolTip("Coin-flip baseline = how often a trendless random walk would reach the target "
                                 "before the stop; it depends only on the two distances. Past trades = how this "
                                 "strategy's own historical trades actually ended.")

    def _render_results(self, rep: StrategyReport):
        tone, text = overfit_verdict(rep)
        self.verdict_lbl.setText(text)
        set_color(self.verdict_lbl, TONE[tone], bold=True)
        parts = []
        for key in rep.cfg.active_keys():
            spec = ALL_PARAMS[key]
            mine, best = rep.cfg.values[key], rep.best[key]
            if mine != best:
                parts.append(f"{spec.label}: {fmt_value(key, mine)} → <b style='color:{C.accent}'>"
                             f"{fmt_value(key, best)}</b>")
            else:
                parts.append(f"{spec.label}: <b>{fmt_value(key, best)}</b>")
        rules = ", ".join(r.name for r in RULES if r.key in rep.cfg.rules)
        self.params_lbl.setText(f"<b>Rules:</b> {rules}<br><b>Optimised values</b> (yours → optimised): "
                                + "  ·  ".join(parts))
        rows, colors = [], []
        for name, key, kind in METRIC_ROWS:
            vals = [rep.base_train[key], rep.best_train[key], rep.base_test[key], rep.best_test[key]]
            rows.append([name] + [_fmt(v, kind) for v in vals])
            if key == "total_return":
                colors.append([None] + [C.up if v > 0 else C.down for v in vals])
            else:
                colors.append([None] * 5)
        fill_table(self.results, rows, colors, align_right={1, 2, 3, 4})

    def _render_trades(self, rep: StrategyReport):
        idx, fee, cur = rep.df.index, rep.cfg.fee, self._currency
        last = float(rep.fc.c[-1])
        names = {"target": "Target hit", "stop": "Stop-loss", "time": "Time limit", "": "Open"}
        rows, colors = [], []
        for k, t in enumerate(reversed(rep.full.trades)):
            ret = t.ret(fee, last)
            col = C.up if ret >= 0 else C.down
            period = "Train" if t.entry_idx < rep.split else "TEST"
            rows.append([str(len(rep.full.trades) - k), period, f"{idx[t.entry_idx]:%d %b %Y}",
                         fmt_price(t.entry_price, cur), fmt_price(t.target, cur), fmt_price(t.initial_stop, cur),
                         f"{idx[t.exit_idx]:%d %b %Y}" if t.exit_idx is not None else "—",
                         fmt_price(t.exit_price, cur) if t.exit_idx is not None else fmt_price(last, cur),
                         names[t.reason], f"{ret:+.2%}",
                         str((t.exit_idx if t.exit_idx is not None else rep.n - 1) - t.entry_idx + 1)])
            reason_col = {"target": C.up, "stop": C.down, "time": C.text2, "": C.accent}[t.reason]
            colors.append([None, C.accent if period == "TEST" else C.muted, None, None, None, None, None, None,
                           reason_col, col, None])
        fill_table(self.trades, rows, colors, align_right={3, 4, 5, 7, 9, 10})

    def _render_board(self, rep: StrategyReport):
        keys = rep.cfg.search_keys()
        short = {"trend_len": "SMA", "ema_fast": "EMAf", "ema_slow": "EMAs", "rsi_max": "RSI<", "bb_k": "BBσ",
                 "breakout_len": "Brk", "vol_mult": "Vol×", "tp_atr": "TP", "sl_atr": "SL", "max_bars": "Max",
                 "trailing": "Trail"}
        rows, colors = [], []
        for r, t in enumerate(rep.trials, 1):
            params = "  ".join(f"{short[k]} {fmt_value(k, t.values[k])}" for k in keys) or "(your settings)"
            te = t.test or {}
            rows.append([str(r), params, f"{t.score:.2f}", _fmt(t.train["total_return"], "pct"),
                         str(t.train["trades"]), _fmt(te.get("total_return"), "pct"), str(te.get("trades", "—")),
                         _fmt(te.get("sharpe"), "num")])
            tr = te.get("total_return")
            colors.append([None] * 5 + [C.up if tr and tr > 0 else C.down, None, None])
        fill_table(self.board, rows, colors, align_right={2, 3, 4, 5, 6, 7})
