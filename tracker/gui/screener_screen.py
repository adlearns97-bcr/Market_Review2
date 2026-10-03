"""Screen 4 - Screener: analyse your own list of tickers, or scan a whole exchange for opportunities."""
from __future__ import annotations

import html
import math
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QPlainTextEdit, QProgressBar,
                               QPushButton, QScrollArea, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
                               QTextBrowser, QVBoxLayout, QWidget)

from ..custom_strategy import ALL_PARAMS, RULES, fmt_value, overfit_verdict
from ..forecast import bar_unit, date_after
from ..fundamentals import CATEGORIES
from ..screener import (EXCHANGES, METHOD_NAMES, SUFFIXES, UNIVERSE_SIZES, BatchState, TickerResult, analyse_list,
                        apply_suffix, bottom_line, full_evaluate, latest_by_symbol, parse_ticker_text,
                        read_tickers_file, results_frame, scan_exchange, HISTORY)
from ..data import fetch_daily_direct
from .theme import TONE, C, fmt_big, fmt_price
from .widgets import Worker, card, label, set_color

SORT_ROLE = Qt.UserRole + 1
COLUMNS = ["Ticker", "Name", "Price", "Chg %", "Overall", "Score", "Confidence", "① Technical", "② Tech + Fund",
           "③ Modified", "Target", "Stop-loss", "Reward:risk", "Days → target", "Target first", "Fund. score",
           "Unseen-data test"]
FILTERS = ["All results", "Buy signals", "Hold", "Sell signals", "Deep-dived only", "Quick-screened only", "Errors"]
ROBUST_TEXT = {"good": ("holds up ✓", C.up), "warn": ("weaker", C.warn), "bad": ("failed ✗", C.down), "": ("—", C.muted)}


def _label_color(text: str) -> str:
    if "BUY" in text or text in ("ACCUMULATE",):
        return C.up
    if "SELL" in text or "AVOID" in text:
        return C.down
    if text in ("WAIT", "SPECULATIVE"):
        return C.warn
    return C.text2


class SortItem(QTableWidgetItem):
    """Table cell that sorts by a numeric key instead of its text."""

    def __lt__(self, other):
        a, b = self.data(SORT_ROLE), other.data(SORT_ROLE)
        if a is None or b is None:
            return super().__lt__(other)
        try:
            return a < b
        except TypeError:          # e.g. an error row's text next to numbers
            return str(a) < str(b)


class ScreenerScreen(QWidget):
    open_requested = Signal(str, int)   # symbol, screen index (0/1/2)

    def __init__(self, get_tech_params, get_strat_config, parent=None):
        super().__init__(parent)
        self._get_tech, self._get_strat = get_tech_params, get_strat_config
        self._results: dict[str, TickerResult] = {}
        self._items: dict[str, QTableWidgetItem] = {}
        self._state: BatchState | None = None
        self._workers: dict[int, Worker] = {}
        self._token = 0
        self._consumed = 0
        self._busy = False
        self._started = 0.0
        self._dirty = False
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(400)
        self._refresh_timer.timeout.connect(self._drain)

        body = QSplitter(Qt.Horizontal)
        body.addWidget(self._build_controls())
        right = QSplitter(Qt.Vertical)
        right.addWidget(self._build_table_panel())
        right.addWidget(self._build_detail_panel())
        right.setSizes([520, 400])
        body.addWidget(right)
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        body.setSizes([370, 1100])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.addWidget(body)

    # ------------------------------------------------------------------ controls

    def _build_controls(self):
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 0, 4, 0)
        lay.setSpacing(10)

        frame, fl = card("A · Analyse your list")
        fl.addWidget(label("Load a CSV or Excel file with a column of tickers (or paste them below). Every "
                           "ticker gets all three methods and a BUY / HOLD / SELL call.", name="Muted", wrap=True))
        row = QHBoxLayout()
        load = QPushButton("Load CSV / Excel…")
        load.clicked.connect(self._pick_file)
        row.addWidget(load)
        self.file_lbl = label("", name="Muted")
        row.addWidget(self.file_lbl, 1)
        fl.addLayout(row)
        self.tickers = QPlainTextEdit()
        self.tickers.setPlaceholderText("AAPL, MSFT, NVDA\nRELIANCE.NS  TCS.NS\n^NSEI  BTC-USD")
        self.tickers.setFixedHeight(110)
        self.tickers.textChanged.connect(self._update_count)
        fl.addWidget(self.tickers)
        srow = QHBoxLayout()
        srow.addWidget(label("Add suffix to bare tickers", name="Secondary"))
        self.suffix = QComboBox()
        self.suffix.addItems(list(SUFFIXES))
        self.suffix.setToolTip("Yahoo needs an exchange suffix for non-US stocks, e.g. RELIANCE → RELIANCE.NS")
        self.suffix.currentTextChanged.connect(self._update_count)
        srow.addWidget(self.suffix, 1)
        fl.addLayout(srow)
        self.count_lbl = label("No tickers yet", name="Muted", wrap=True)
        fl.addWidget(self.count_lbl)
        self.list_btn = QPushButton("Analyse list")
        self.list_btn.setObjectName("Primary")
        self.list_btn.clicked.connect(self.analyse_list)
        fl.addWidget(self.list_btn)
        lay.addWidget(frame)

        frame, fl = card("B · Scan an exchange")
        fl.addWidget(label("1) list the exchange's stocks, largest first → 2) quick-screen all of them on price "
                           "data → 3) run all three methods on the most promising.", name="Muted", wrap=True))
        grid = QVBoxLayout()
        self.exchange = QComboBox()
        self.exchange.addItems(list(EXCHANGES))
        self.universe = QComboBox()
        self.universe.addItems(list(UNIVERSE_SIZES))
        self.universe.setCurrentText("Largest 500")
        self.deep = QSpinBox()
        self.deep.setRange(5, 100)
        self.deep.setValue(25)
        self.deep.setToolTip("How many of the best quick-screen candidates get the full three-method analysis")
        for text, w in (("Exchange", self.exchange), ("Stocks to scan", self.universe), ("Deep-dive the best", self.deep)):
            r = QHBoxLayout()
            r.addWidget(label(text, name="Secondary"))
            r.addWidget(w, 1)
            grid.addLayout(r)
        fl.addLayout(grid)
        self.estimate = label("", name="Muted", wrap=True)
        fl.addWidget(self.estimate)
        for w in (self.universe, self.deep):
            (w.currentTextChanged if isinstance(w, QComboBox) else w.valueChanged).connect(self._update_estimate)
        self.scan_btn = QPushButton("Scan exchange")
        self.scan_btn.setObjectName("Primary")
        self.scan_btn.clicked.connect(self.scan)
        fl.addWidget(self.scan_btn)
        lay.addWidget(frame)

        frame, fl = card("Settings")
        fl.addWidget(label("① uses your Pure Technical settings, ② adds fundamentals, and ③ trains the rules you "
                           "built in the Modified Strat tab separately for every stock (daily bars, "
                           f"{HISTORY[:-1]} years of history).", name="Muted", wrap=True))
        r = QHBoxLayout()
        r.addWidget(label("Combinations per stock", name="Secondary"))
        self.iterations = QSpinBox()
        self.iterations.setRange(20, 2000)
        self.iterations.setSingleStep(50)
        self.iterations.setValue(200)
        self.iterations.setToolTip("How many parameter combinations the Modified Strat tries for each stock")
        r.addWidget(self.iterations, 1)
        fl.addLayout(r)
        lay.addWidget(frame)

        frame, fl = card("Progress")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        fl.addWidget(self.progress)
        self.status = label("Idle", name="Muted", wrap=True)
        fl.addWidget(self.status)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        fl.addWidget(self.cancel_btn)
        lay.addWidget(frame)
        lay.addStretch(1)
        self._update_estimate()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        scroll.setMinimumWidth(350)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return scroll

    def _build_table_panel(self):
        frame, lay = card(spacing=6)
        top = QHBoxLayout()
        self.summary = label("No results yet", bold=True)
        self.summary.setTextFormat(Qt.RichText)
        top.addWidget(self.summary, 1)
        self.filter = QComboBox()
        self.filter.addItems(FILTERS)
        self.filter.currentTextChanged.connect(lambda _t: self._apply_filter())
        top.addWidget(self.filter)
        export = QPushButton("Export…")
        export.setToolTip("Save the results as Excel or CSV")
        export.clicked.connect(self._export)
        top.addWidget(export)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.clear)
        top.addWidget(clear)
        lay.addLayout(top)

        t = QTableWidget(0, len(COLUMNS))
        t.setHorizontalHeaderLabels(COLUMNS)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QTableWidget.NoEditTriggers)
        t.setSelectionBehavior(QTableWidget.SelectRows)
        t.setSelectionMode(QTableWidget.SingleSelection)
        t.setAlternatingRowColors(True)
        t.setShowGrid(False)
        t.setWordWrap(False)
        hh = t.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Interactive)
        hh.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hh.setStretchLastSection(True)
        widths = [95, 190, 85, 65, 105, 60, 85, 125, 135, 90, 85, 85, 80, 95, 85, 80, 110]
        for i, wdt in enumerate(widths):
            t.setColumnWidth(i, wdt)
        t.setSortingEnabled(True)
        t.itemSelectionChanged.connect(self._show_selected)
        t.cellDoubleClicked.connect(lambda r, _c: self._open(2))
        self.table = t
        lay.addWidget(t, 1)
        return frame

    def _build_detail_panel(self):
        frame, lay = card(spacing=6)
        row = QHBoxLayout()
        row.addWidget(label("IN-DEPTH REPORT", name="CardTitle"))
        row.addStretch(1)
        self.deep_btn = QPushButton("Deep-dive this stock")
        self.deep_btn.setToolTip("Run all three methods on the selected quick-screened stock")
        self.deep_btn.clicked.connect(self._deep_dive_selected)
        row.addWidget(self.deep_btn)
        for i, text in enumerate(("Open in ①", "Open in ②", "Open in ③")):
            b = QPushButton(text)
            b.setToolTip(["Pure Technical", "Technical + Fundamentals", "Modified Strat"][i])
            b.clicked.connect(lambda _c=False, k=i: self._open(k))
            row.addWidget(b)
        lay.addLayout(row)
        self.report = QTextBrowser()
        self.report.setOpenExternalLinks(False)
        self.report.setHtml(f"<p style='color:{C.text2}'>Select a row to see the full reasoning behind its call.</p>")
        lay.addWidget(self.report, 1)
        return frame

    # ------------------------------------------------------------------ inputs

    def _pick_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load tickers", str(Path.home()),
                                              "Spreadsheets (*.csv *.xlsx *.xls *.xlsm *.txt);;All files (*)")
        if path:
            self.load_file(path)

    def load_file(self, path: str):
        try:
            found = read_tickers_file(path)
        except Exception as exc:
            self.file_lbl.setText(f"Couldn't read file: {exc}")
            set_color(self.file_lbl, C.down)
            return
        self.file_lbl.setText(f"{Path(path).name} · {len(found)} tickers")
        set_color(self.file_lbl, C.muted if found else C.down)
        self.tickers.setPlainText(", ".join(found))

    def symbols(self) -> list[str]:
        return apply_suffix(parse_ticker_text(self.tickers.toPlainText()), SUFFIXES[self.suffix.currentText()])

    def _update_count(self, *_):
        syms = self.symbols()
        if not syms:
            self.count_lbl.setText("No tickers yet")
            return
        eta = max(5, round(len(syms) * 0.8))
        preview = ", ".join(syms[:8]) + (" …" if len(syms) > 8 else "")
        self.count_lbl.setText(f"{len(syms)} ticker{'s' if len(syms) != 1 else ''}: {preview}  (≈ {self._fmt_secs(eta)})")

    def _update_estimate(self, *_):
        n = UNIVERSE_SIZES[self.universe.currentText()]
        secs = (n or 3000) * 0.035 + self.deep.value() * 1.0 + 3
        self.estimate.setText(f"Roughly {self._fmt_secs(secs)}"
                              + ("" if n else " (depends on how many stocks the exchange lists)") + ".")

    @staticmethod
    def _fmt_secs(secs: float) -> str:
        return f"{secs:.0f} s" if secs < 90 else f"{secs / 60:.0f} min"

    # ------------------------------------------------------------------ running

    def analyse_list(self):
        syms = self.symbols()
        if not syms:
            self.status.setText("Add some tickers first (load a file or paste them).")
            set_color(self.status, C.down)
            return
        self._run(analyse_list, syms, self._get_tech(), self._get_strat(), self.iterations.value(), kind="list")

    def scan(self):
        limit = UNIVERSE_SIZES[self.universe.currentText()]
        self._run(scan_exchange, self.exchange.currentText(), limit, self.deep.value(), self._get_tech(),
                  self._get_strat(), self.iterations.value(), kind="scan")

    def _run(self, fn, *args, kind: str):
        if self._busy:
            return
        self.clear()
        self._state = BatchState()
        self._token += 1
        self._busy = True
        self._started = time.time()
        for b in (self.list_btn, self.scan_btn, self.deep_btn):
            b.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress.setValue(0)
        self.table.setSortingEnabled(False)
        worker = Worker(self._token, fn, *args, self._state, with_progress=True)
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(self._on_finished)
        self._workers[self._token] = worker
        worker.start()
        self._refresh_timer.start()

    def cancel(self):
        if self._state is not None:
            self._state.cancel.set()
            self.status.setText("Cancelling… (finishing the stocks already in progress)")

    def _on_progress(self, done: int, total: int):
        self.progress.setValue(int(100 * done / max(total, 1)))
        self._dirty = True
        if self._state is not None:
            self.status.setText(f"{self._state.phase}  {done:,} / {total:,}")

    def _on_finished(self, token: int):
        worker = self._workers.pop(token, None)
        if worker is None or token != self._token:
            return
        self._refresh_timer.stop()
        self._busy = False
        for b in (self.list_btn, self.scan_btn, self.deep_btn):
            b.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._drain(force=True)
        self.table.setSortingEnabled(True)
        self.table.sortItems(5, Qt.DescendingOrder)
        if worker.error:
            self.status.setText(worker.error)
            set_color(self.status, C.down)
            return
        st = self._state
        self.progress.setValue(100)
        full = sum(r.depth == "full" for r in self._results.values())
        msg = f"{st.phase} in {self._fmt_secs(time.time() - self._started)}. "
        if st.scanned:
            msg += f"Quick-screened {st.scanned:,} stocks, deep-dived {full}. "
        if st.failed:
            msg += f"{st.failed} skipped (too little history or no data). "
        self.status.setText(msg + "Double-click a row to open it in ③ Modified Strat.")
        set_color(self.status, C.muted)
        if self.table.rowCount():
            self.table.selectRow(0)

    def clear(self):
        self._results.clear()
        self._items.clear()
        self._consumed = 0
        self.table.setRowCount(0)
        self.summary.setText("No results yet")
        self.report.setHtml("")

    # ------------------------------------------------------------------ table

    def _drain(self, force: bool = False):
        """Move newly finished results from the worker into the table (throttled while running)."""
        if self._state is None or (not self._dirty and not force):
            return
        self._dirty = False
        new = self._state.results[self._consumed:]
        self._consumed += len(new)
        if not new:
            return
        sorting = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        self.table.setUpdatesEnabled(False)
        for r in new:
            old = self._results.get(r.symbol)
            if old is not None and old.depth == "full" and r.depth == "quick":
                continue
            self._results[r.symbol] = r
            self._upsert(r)
        self.table.setUpdatesEnabled(True)
        self.table.setSortingEnabled(sorting)
        self._apply_filter()
        self._update_summary()

    def _upsert(self, r: TickerResult):
        item = self._items.get(r.symbol)
        if item is None:
            row = self.table.rowCount()
            self.table.insertRow(row)
        else:
            row = item.row()
        cells = self._cells(r)
        for col, (text, key, color) in enumerate(cells):
            it = SortItem(text)
            it.setData(SORT_ROLE, key)
            if color:
                it.setForeground(QColor(color))
            if col in (2, 3, 5, 10, 11, 12, 13, 14, 15):
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if col == 0:
                it.setData(Qt.UserRole, r.symbol)
                self._items[r.symbol] = it
            self.table.setItem(row, col, it)

    def _cells(self, r: TickerResult) -> list[tuple[str, object, str | None]]:
        cur = r.currency
        nan = -1e18
        if r.depth == "error":
            return [(r.symbol, r.symbol, None), (r.error, r.error, C.down)] + \
                [("", nan, None)] * 2 + [("ERROR", "", C.down)] + [("", nan, None)] * (len(COLUMNS) - 5)
        s, f = r.signal, r.forecast
        full = r.depth == "full"
        overall = (r.label, r.label, _label_color(r.label)) if full else ("· quick", "", C.muted)
        score = r.overall * 100 if full else r.pre_score * 100
        cells = [
            (r.symbol, r.symbol, None),
            (r.name, r.name, C.text2),
            (fmt_price(r.price, cur), r.price, None),
            (f"{r.change_pct:+.2%}" if r.change_pct == r.change_pct else "—",
             r.change_pct if r.change_pct == r.change_pct else nan,
             C.up if r.change_pct >= 0 else C.down),
            overall,
            (f"{score:+.0f}", score, None if full else C.muted),
            (r.confidence if full else "", r.confidence, None),
            (f"{r.tech_label} {r.tech_score:+.0f}", r.tech_score, _label_color(r.tech_label)),
            ((r.combined.label if r.combined else "—"), r.combined.label if r.combined else "",
             _label_color(r.combined.label) if r.combined else C.muted),
            ((s.action if s else "—") + ("" if full or not s else "*"), s.action if s else "",
             {"BUY": C.up, "HOLD": C.accent, "WAIT": C.warn}.get(s.action, C.text2) if s else C.muted),
            (fmt_price(s.target, cur) if s else "—", s.target if s else nan, C.up),
            (fmt_price(s.stop, cur) if s else "—", s.stop if s else nan, C.down),
            (f"{r.reward_risk:.1f}" if r.reward_risk == r.reward_risk else "—", r.reward_risk
             if r.reward_risk == r.reward_risk else nan, None),
            (f"{f.median_bars:.0f}" if f is not None and f.median_bars else "—",
             f.median_bars if f is not None and f.median_bars else 1e18, None),
            (f"{f.p_target:.0%}" if f is not None else "—", f.p_target if f is not None else nan, None),
            (f"{r.fund.overall:.0f}" if r.fund is not None and r.fund.available else "—",
             r.fund.overall if r.fund is not None and r.fund.available else nan, None),
            (ROBUST_TEXT[r.robust][0] if full else "", r.robust, ROBUST_TEXT[r.robust][1]),
        ]
        return cells

    def _apply_filter(self):
        mode = self.filter.currentText()
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            r = self._results.get(item.data(Qt.UserRole)) if item else None
            show = r is not None and {
                "All results": True,
                "Buy signals": "BUY" in r.label,
                "Hold": r.label == "HOLD",
                "Sell signals": "SELL" in r.label,
                "Deep-dived only": r.depth == "full",
                "Quick-screened only": r.depth == "quick",
                "Errors": r.depth == "error",
            }[mode]
            self.table.setRowHidden(row, not show)

    def _update_summary(self):
        rs = list(self._results.values())
        full = [r for r in rs if r.depth == "full"]
        count = lambda pred: sum(pred(r) for r in full)
        parts = [f"<span style='color:{C.up}'>BUY {count(lambda r: 'BUY' in r.label)}</span>",
                 f"<span style='color:{C.warn}'>HOLD {count(lambda r: r.label == 'HOLD')}</span>",
                 f"<span style='color:{C.down}'>SELL {count(lambda r: 'SELL' in r.label)}</span>"]
        quick = sum(r.depth == "quick" for r in rs)
        errors = sum(r.depth == "error" for r in rs)
        extra = []
        if quick:
            extra.append(f"{quick:,} quick-screened (* = your rules, not yet optimised)")
        if errors:
            extra.append(f"{errors} errors")
        self.summary.setText("  ·  ".join(parts) + (f"<span style='color:{C.muted}; font-weight:400'>  ·  "
                                                    + "  ·  ".join(extra) + "</span>" if extra else ""))

    # ------------------------------------------------------------------ selection / actions

    def _selected(self) -> TickerResult | None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        item = self.table.item(rows[0].row(), 0)
        return self._results.get(item.data(Qt.UserRole)) if item else None

    def _show_selected(self):
        r = self._selected()
        if r is None:
            return
        self.deep_btn.setEnabled(not self._busy and r.depth != "full")
        self.report.setHtml(report_html(r))

    def _open(self, index: int):
        r = self._selected()
        if r is not None and r.depth != "error":
            self.open_requested.emit(r.symbol, index)

    def _deep_dive_selected(self):
        r = self._selected()
        if r is None or self._busy:
            return
        tech, cfg, its = self._get_tech(), self._get_strat(), self.iterations.value()

        def job(progress, sym=r.symbol, pre=r.pre_score, cap=r.market_cap):
            progress(0, 1)
            df, meta = fetch_daily_direct(sym, HISTORY)
            res = full_evaluate(sym, df, meta, tech, cfg, its)
            res.pre_score, res.market_cap = pre, cap
            state.results.append(res)
            progress(1, 1)
            return state

        state = self._state or BatchState()
        self._state = state
        state.phase = f"Deep-dive {r.symbol}…"
        self._token += 1
        self._busy = True
        self._started = time.time()
        for b in (self.list_btn, self.scan_btn, self.deep_btn):
            b.setEnabled(False)
        worker = Worker(self._token, job, with_progress=True)
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(self._on_deep_finished)
        self._workers[self._token] = worker
        worker.start()

    def _on_deep_finished(self, token: int):
        worker = self._workers.pop(token, None)
        if worker is None or token != self._token:
            return
        self._busy = False
        for b in (self.list_btn, self.scan_btn):
            b.setEnabled(True)
        if worker.error:
            self.status.setText(worker.error)
            set_color(self.status, C.down)
            return
        self._drain(force=True)
        self.status.setText(f"Deep-dive finished in {self._fmt_secs(time.time() - self._started)}.")
        set_color(self.status, C.muted)
        self._show_selected()

    def _export(self):
        results = latest_by_symbol(list(self._results.values()))
        if not results:
            self.status.setText("Nothing to export yet.")
            return
        path, chosen = QFileDialog.getSaveFileName(self, "Export results", str(Path.home() / "screener_results.xlsx"),
                                                   "Excel (*.xlsx);;CSV (*.csv)")
        if path:
            self.export_to(path)

    def export_to(self, path: str):
        df = results_frame(list(self._results.values()))
        try:
            if path.lower().endswith(".csv"):
                df.to_csv(path, index=False)
            else:
                df.to_excel(path if path.lower().endswith(".xlsx") else path + ".xlsx", index=False)
        except Exception as exc:
            self.status.setText(f"Export failed: {exc}")
            set_color(self.status, C.down)
            return
        self.status.setText(f"Exported {len(df)} rows to {path}")
        set_color(self.status, C.muted)


# --------------------------------------------------------------------------- in-depth report


def _esc(text) -> str:
    return html.escape(str(text))


def _pct(v, signed=True) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    return f"{v:+.1%}" if signed else f"{v:.0%}"


def _span(text, color, bold=False) -> str:
    return f"<span style='color:{color};{'font-weight:700;' if bold else ''}'>{_esc(text)}</span>"


def report_html(r: TickerResult) -> str:
    style = (f"<style>h2 {{ color:{C.text}; margin: 0 0 4px 0; }} h3 {{ color:{C.text}; margin: 14px 0 4px 0; }}"
             f" p, li, td {{ color:{C.text2}; }} b {{ color:{C.text}; }} td {{ padding: 2px 10px 2px 0; }}</style>")
    if r.depth == "error":
        return style + f"<h2>{_esc(r.symbol)}</h2><p>{_span(r.error, C.down)}</p>"
    cur = r.currency
    h = [style, f"<h2>{_esc(r.name)} <span style='color:{C.text2}'>({_esc(r.symbol)})</span></h2>"]
    meta = [f"{fmt_price(r.price, cur)} ({r.change_pct:+.2%} today)"]
    if r.fund is not None:
        meta += [x for x in (r.fund.sector, r.fund.industry) if x]
    if r.market_cap:
        meta.append(f"market cap {fmt_big(r.market_cap)}")
    h.append(f"<p>{' · '.join(_esc(m) for m in meta)}</p>")
    s = r.signal

    if r.depth == "quick":
        h.append(f"<p>{_span('Quick screen only', C.warn, True)} - price-based checks with your own rule values. "
                 "Click <b>Deep-dive this stock</b> to add fundamentals and a strategy optimised for it.</p>")
        h.append(f"<p><b>① Pure Technical:</b> {_span(r.tech_label, _label_color(r.tech_label), True)} "
                 f"({r.tech_score:+.0f})<br>")
        if s is not None:
            h.append(f"<b>③ Your Modified Strat rules (not optimised):</b> {_span(s.action, C.text, True)} - "
                     f"{_esc(s.detail)}<br>")
            h.append("<br>".join(f"{_span('✓' if ok else '✗', C.up if ok else C.down)} <b>{_esc(n)}</b> — {_esc(t)}"
                                 for n, ok, t in s.conditions))
        h.append(f"</p><p>Quick-screen rank score: <b>{r.pre_score * 100:+.0f}</b></p>")
        return "".join(h)

    tone = _label_color(r.label)
    unit = bar_unit(r.strat.df.index) if r.strat is not None else "trading days"
    side = 1 if "BUY" in r.label else -1 if "SELL" in r.label else 0
    agree = sum((1 if v >= 0.2 else -1 if v <= -0.2 else 0) == side for v in r.votes.values())
    h.append(f"<p><span style='font-size:18pt; font-weight:700; color:{tone}'>{_esc(r.label)}</span> &nbsp; "
             f"overall <b>{r.overall * 100:+.0f}</b> / 100 &nbsp;·&nbsp; confidence <b>{r.confidence}</b> "
             f"({agree} of {len(r.votes)} methods agree)</p>")
    h.append(f"<p><b>Bottom line:</b> {_esc(bottom_line(r, unit))}</p>")

    # ---- trade plan
    if s is not None:
        f = r.forecast
        rows = [("Entry (today's price)", fmt_price(r.price, cur))]
        if s.trade is not None:
            rows.append(("Strategy already in a trade", f"bought {fmt_price(s.trade.entry_price, cur)} on "
                                                         f"{r.strat.df.index[s.trade.entry_idx]:%d %b %Y}"))
        rows += [("Target (take profit)", f"{_span(fmt_price(s.target, cur), C.up, True)} ({s.target / r.price - 1:+.1%})"),
                 ("Stop-loss (sell if it falls)", f"{_span(fmt_price(s.stop, cur), C.down, True)} "
                                                  f"({s.stop / r.price - 1:+.1%})"),
                 ("Reward : risk", f"{r.reward_risk:.1f} : 1" if r.reward_risk == r.reward_risk else "—")]
        if r.strat is not None and r.strat.best.get("trailing"):
            rows.append(("Trailing stop", "on - the stop follows the price up and never moves down"))
        if f is not None and f.median_bars:
            idx = r.strat.df.index
            rows.append(("Time to target", f"≈ {f.median_bars:.0f} {unit} (by ~{date_after(idx, f.median_bars):%d %b %Y}), "
                                           f"usually {f.q25:.0f}–{f.q75:.0f}"))
            rows.append(("Chance target is hit first", f"{f.p_target:.0%} (stop first {f.p_stop:.0%}, neither within "
                                                       f"{f.horizon} {unit} {f.p_neither:.0%}) · coin-flip baseline "
                                                       f"{f.baseline_p:.0%}"))
        risk = r.price - s.stop
        if risk > 0:
            shares = int(100 // risk)
            rows.append(("Size for 1% risk", f"{shares:,} shares ≈ {fmt_price(shares * r.price, cur)} on a "
                                             f"{fmt_price(10_000, cur)} account"))
        h.append("<h3>Trade plan</h3><table>" + "".join(f"<tr><td>{a}</td><td><b>{b}</b></td></tr>" for a, b in rows)
                 + "</table>")

    # ---- ① Pure Technical
    h.append(f"<h3>① Pure Technical — {_span(r.tech_label, _label_color(r.tech_label))} ({r.tech_score:+.0f})</h3>")
    if r.ta is not None:
        snap = r.ta.snapshot()
        h.append("<ul>" + "".join(
            f"<li><b>{_esc(c.name)}</b> {_span(f'{c.points:+.1f} pts', C.up if c.points > 0.05 else C.down if c.points < -0.05 else C.muted)}"
            f" — {_esc(c.notes[0])}</li>" for c in snap.components) + "</ul>")
        st = r.ta.backtest.stats
        h.append(f"<p>Simulating this signal over the last year: {_pct(st['total_return'])} vs buy &amp; hold "
                 f"{_pct(st['buy_hold_return'])} (worst drawdown {_pct(st['max_drawdown'])}, {int(st['trades'])} trades).</p>")

    # ---- ② Technical + Fundamentals
    if r.combined is not None:
        cv = r.combined
        h.append(f"<h3>② Technical + Fundamentals — {_span(cv.label, _label_color(cv.label))}</h3>")
        h.append(f"<p>{_esc(cv.explanation)}</p>")
        fr = r.fund
        if fr is not None and fr.available:
            cats = " · ".join(f"{c} {v:.0f}" for c, v in fr.category_scores.items() if v is not None and CATEGORIES[c] > 0)
            h.append(f"<p>Fundamental score <b>{fr.overall:.0f}/100 ({fr.tier})</b> — {_esc(cats)}</p>")
            if fr.strengths:
                h.append("<p><b>Strengths</b><br>" + "<br>".join(f"{_span('✓', C.up)} {_esc(x)}" for x in fr.strengths[:4]) + "</p>")
            if fr.weaknesses:
                h.append("<p><b>Weaknesses</b><br>" + "<br>".join(f"{_span('✗', C.down)} {_esc(x)}" for x in fr.weaknesses[:4]) + "</p>")
            vals = []
            g = fr.graham_number()
            if g:
                vals.append(f"Graham number {fmt_price(g, cur)} ({g / r.price - 1:+.0%})")
            tgt = fr.info.get("targetMeanPrice")
            if tgt:
                vals.append(f"analyst mean target {fmt_price(tgt, cur)} ({tgt / r.price - 1:+.0%})")
            if vals:
                h.append(f"<p>{_esc(' · '.join(vals))}</p>")
        elif r.fund_note:
            h.append(f"<p>{_esc(r.fund_note)}</p>")

    # ---- ③ Modified Strat
    if r.strat is not None:
        rep = r.strat
        h.append(f"<h3>③ Modified Strat (optimised for {_esc(r.symbol)}) — {_span(s.action, C.text)}</h3>")
        h.append(f"<p>{_esc(s.detail)}</p>")
        rules = ", ".join(x.name for x in RULES if x.key in rep.cfg.rules)
        values = " · ".join(f"{ALL_PARAMS[k].label} {fmt_value(k, rep.best[k])}" for k in rep.cfg.active_keys())
        h.append(f"<p><b>Rules:</b> {_esc(rules)}<br><b>Optimised values:</b> {_esc(values)}</p>")
        h.append("<p>" + "<br>".join(f"{_span('✓' if ok else '✗', C.up if ok else C.down)} <b>{_esc(n)}</b> — {_esc(t)}"
                                     for n, ok, t in s.conditions) + "</p>")
        tone_, text = overfit_verdict(rep)
        h.append(f"<p>{_span(text, TONE[tone_])}</p>")
        h.append(f"<p>Training: {_pct(rep.best_train['total_return'])} over {rep.best_train['trades']} trades · "
                 f"unseen test: {_pct(rep.best_test['total_return'])} over {rep.best_test['trades']} trades "
                 f"(buy &amp; hold {_pct(rep.best_test['buy_hold'])}).</p>")
    elif r.strat_note:
        h.append(f"<h3>③ Modified Strat</h3><p>{_esc(r.strat_note)}</p>")

    # ---- how the call was made
    rows = "".join(f"<tr><td>{METHOD_NAMES[k]}</td><td>{_span(f'{v:+.2f}', C.up if v > 0.05 else C.down if v < -0.05 else C.muted, True)}"
                   f"</td><td>weight {r.weights[k] / sum(r.weights.values()):.0%}</td></tr>" for k, v in r.votes.items())
    h.append("<h3>How the overall call was made</h3>"
             "<p>Each method votes from −1 (sell) to +1 (buy); the weighted average is the overall score. "
             "BUY ≥ +30, SELL ≤ −25, otherwise HOLD. A Modified Strat buy is discounted when its rules failed "
             "on unseen data.</p><table>" + rows + "</table>")
    h.append(f"<p style='color:{C.muted}'>Educational estimate from past data - not financial advice.</p>")
    return "".join(h)
