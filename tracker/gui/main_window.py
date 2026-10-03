"""Main window: ticker bar, live refresh, and navigation between the two analysis modes."""
from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QCompleter, QFrame, QHBoxLayout, QLineEdit,
                               QMainWindow, QPushButton, QStackedWidget, QVBoxLayout, QWidget)

from ..analysis import TechnicalAnalysis, analyze
from ..data import DEFAULT_TIMEFRAME, INTRADAY, TIMEFRAME_BY_LABEL, TIMEFRAMES, MarketData, load_market_data
from ..fundamentals import FundamentalReport, fetch_fundamentals
from .fundamental_screen import FundamentalScreen
from .modified_screen import ModifiedScreen
from .screener_screen import ScreenerScreen
from .technical_screen import TechnicalScreen
from .theme import C, fmt_price
from .widgets import Worker, label, set_color

POPULAR = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "NFLX", "AMD", "INTC", "JPM", "V", "KO",
           "PEP", "WMT", "XOM", "SPY", "QQQ", "^GSPC", "^IXIC", "^NSEI", "^BSESN", "RELIANCE.NS", "TCS.NS",
           "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "TATAMOTORS.NS", "BTC-USD", "ETH-USD", "DEMO"]
REFRESH = {"15 s": 15, "30 s": 30, "1 min": 60, "5 min": 300}


class MainWindow(QMainWindow):
    def __init__(self, symbol: str = "AAPL"):
        super().__init__()
        self.setWindowTitle("Market Tracker · learning edition")
        self.resize(1560, 960)
        self._workers: dict[tuple[str, int], Worker] = {}
        self._callbacks: dict[str, tuple] = {}
        self._hist_token = self._info_token = 0
        self._hist_initial = True
        self._fetching = False
        self._symbol: str | None = None
        self._tf = TIMEFRAME_BY_LABEL[DEFAULT_TIMEFRAME]
        self._market: MarketData | None = None
        self._ta: TechnicalAnalysis | None = None
        self._report: FundamentalReport | None = None

        self.technical = TechnicalScreen()
        self.fundamental = FundamentalScreen()
        self.modified = ModifiedScreen()
        self.screener = ScreenerScreen(lambda: self.technical.settings()[0], self.modified.config)
        self.screener.open_requested.connect(self.open_symbol)
        self.technical.params_applied.connect(lambda: self._reanalyze(keep_view=True))

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_top_bar())
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_side_bar())
        self.stack = QStackedWidget()
        self.stack.addWidget(self.technical)
        self.stack.addWidget(self.fundamental)
        self.stack.addWidget(self.modified)
        self.stack.addWidget(self.screener)
        body.addWidget(self.stack, 1)
        outer.addLayout(body, 1)
        self.setCentralWidget(root)

        self.status = label("", name="Muted")
        self.statusBar().addWidget(self.status, 1)
        self.statusBar().addPermanentWidget(label(
            "Data: Yahoo Finance via yfinance (may be delayed)  ·  Educational tool - not financial advice  ",
            name="Muted"))

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._live_tick)
        self.symbol_edit.setText(symbol)
        QTimer.singleShot(0, self.load)

    # ------------------------------------------------------------------ layout

    def _build_top_bar(self):
        bar = QFrame()
        bar.setObjectName("TopBar")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(10)

        brand = label("◆ Market Tracker", size=12, bold=True, color=C.accent)
        lay.addWidget(brand)
        lay.addSpacing(12)

        self.symbol_edit = QLineEdit()
        self.symbol_edit.setPlaceholderText("Symbol, e.g. AAPL, RELIANCE.NS, BTC-USD, DEMO")
        self.symbol_edit.setFixedWidth(210)
        self.symbol_edit.setToolTip("Any Yahoo Finance symbol. Indian stocks end in .NS (NSE) or .BO (BSE); "
                                    "indices start with ^; crypto looks like BTC-USD. DEMO works offline.")
        completer = QCompleter(POPULAR, self)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.symbol_edit.setCompleter(completer)
        self.symbol_edit.returnPressed.connect(self.load)
        lay.addWidget(self.symbol_edit)

        self.tf_combo = QComboBox()
        self.tf_combo.addItems([tf.label for tf in TIMEFRAMES])
        self.tf_combo.setCurrentText(DEFAULT_TIMEFRAME)
        self.tf_combo.currentTextChanged.connect(lambda _t: self.load())
        lay.addWidget(self.tf_combo)

        load_btn = QPushButton("Load")
        load_btn.setObjectName("Primary")
        load_btn.clicked.connect(self.load)
        lay.addWidget(load_btn)
        lay.addSpacing(10)

        self.live = QCheckBox("Live")
        self.live.setToolTip("Re-download prices on a timer so the last candle updates in real time")
        self.live.setChecked(True)
        self.live.toggled.connect(self._update_timer)
        self.refresh_combo = QComboBox()
        self.refresh_combo.addItems(list(REFRESH))
        self.refresh_combo.setCurrentText("30 s")
        self.refresh_combo.currentTextChanged.connect(lambda _t: self._update_timer())
        lay.addWidget(self.live)
        lay.addWidget(self.refresh_combo)
        self.live_dot = label("●", color=C.muted)
        lay.addWidget(self.live_dot)

        lay.addStretch(1)
        quote = QVBoxLayout()
        quote.setSpacing(0)
        row = QHBoxLayout()
        row.setSpacing(10)
        self.price_lbl = label("—", size=17, bold=True)
        self.change_lbl = label("", size=11, bold=True)
        row.addStretch(1)
        row.addWidget(self.price_lbl)
        row.addWidget(self.change_lbl)
        self.quote_meta = label("", name="Muted")
        self.quote_meta.setAlignment(Qt.AlignRight)
        quote.addLayout(row)
        quote.addWidget(self.quote_meta)
        lay.addLayout(quote)
        return bar

    def _build_side_bar(self):
        side = QFrame()
        side.setObjectName("SideBar")
        side.setFixedWidth(210)
        lay = QVBoxLayout(side)
        lay.setContentsMargins(0, 14, 0, 14)
        lay.setSpacing(4)
        head = label("ANALYSIS MODE", name="CardTitle")
        head.setContentsMargins(16, 0, 0, 6)
        lay.addWidget(head)
        group = QButtonGroup(self)
        group.setExclusive(True)
        modes = [("①  Pure Technical\n      Chart signals + simulation", 0),
                 ("②  Technical + Fundamentals\n      Is it a good business too?", 1),
                 ("③  Modified Strat\n      Build, train && optimise", 2),
                 ("④  Screener\n      Your list or an exchange", 3)]
        self._nav: list[QPushButton] = []
        for text, idx in modes:
            btn = QPushButton(text)
            btn.setObjectName("Nav")
            btn.setCheckable(True)
            btn.setChecked(idx == 0)
            btn.clicked.connect(lambda _c=False, i=idx: self.stack.setCurrentIndex(i))
            group.addButton(btn)
            lay.addWidget(btn)
            self._nav.append(btn)
        lay.addStretch(1)
        tips = label("Tips\n• Hover the chart for OHLC values\n• Click a candle to replay the model's "
                     "call on that day\n• Zoom / pan with the toolbar above the chart\n• Type DEMO to "
                     "use offline sample data", name="Muted", wrap=True)
        tips.setContentsMargins(16, 0, 12, 0)
        tips.setStyleSheet(f"color: {C.muted}; font-size: 8.5pt;")
        lay.addWidget(tips)
        return side

    # ------------------------------------------------------------------ loading

    def _start(self, kind: str, token: int, fn, *args, on_done, on_fail):
        worker = Worker(token, fn, *args)
        self._workers[(kind, token)] = worker   # keep alive until it reports back
        self._callbacks[kind] = (on_done, on_fail)
        worker.signals.finished.connect(self._on_hist_finished if kind == "hist" else self._on_info_finished)
        worker.start()

    def _on_hist_finished(self, token: int):
        self._dispatch("hist", token)

    def _on_info_finished(self, token: int):
        self._dispatch("info", token)

    def _dispatch(self, kind: str, token: int):
        worker = self._workers.pop((kind, token), None)
        if worker is None:
            return
        on_done, on_fail = self._callbacks[kind]
        if worker.error is not None:
            on_fail(token, worker.error)
        else:
            on_done(token, worker.result)

    def load(self):
        symbol = self.symbol_edit.text().strip().upper()
        if not symbol:
            return
        self.symbol_edit.setText(symbol)
        tf = TIMEFRAME_BY_LABEL[self.tf_combo.currentText()]
        symbol_changed = symbol != self._symbol
        self._symbol, self._tf = symbol, tf

        self._fetch_history(initial=True)
        if symbol_changed:
            self.modified.set_symbol(symbol)
            self._report = None
            self.price_lbl.setText("—")
            self.change_lbl.setText("")
            self.fundamental.set_loading(symbol)
            self._info_token += 1
            self._start("info", self._info_token, fetch_fundamentals, symbol,
                        on_done=self._on_info_done, on_fail=self._on_info_failed)
        self._update_timer()

    def open_symbol(self, symbol: str, screen: int):
        """Jump from the screener to one of the analysis screens for a symbol."""
        self.symbol_edit.setText(symbol)
        self.load()
        self.stack.setCurrentIndex(screen)
        self._nav[screen].setChecked(True)

    def _fetch_history(self, initial: bool):
        self._hist_token += 1
        self._hist_initial = initial
        self._fetching = True
        if initial:
            msg = f"Loading {self._symbol} · {self._tf.label}…"
            self.technical.show_message(msg)
            self.fundamental.show_message(msg)
            self._set_status(msg)
        self._start("hist", self._hist_token, load_market_data, self._symbol, self._tf,
                    on_done=self._on_history_done, on_fail=self._on_history_failed)

    def _on_history_done(self, token: int, market: MarketData):
        if token != self._hist_token:
            return  # a newer request superseded this one
        self._fetching = False
        self._market = market
        currency = market.quote.currency
        self.technical.set_currency(currency)
        self.fundamental.set_currency(currency)
        self.modified.set_currency(currency)
        self._reanalyze(keep_view=not self._hist_initial)
        self._update_quote()
        self._set_status(f"{self._symbol}: {len(market.history)} bars downloaded · updated "
                         f"{datetime.now():%H:%M:%S}")
        self._flash_live()

    def _on_history_failed(self, token: int, message: str):
        if token != self._hist_token:
            return
        self._fetching = False
        self._set_status(message, error=True)
        if self._hist_initial or self._market is None:
            text = f"{message}\n\nTip: try the symbol DEMO to explore the app offline."
            self.technical.show_message(text, C.down)
            self.fundamental.show_message(text, C.down)

    def _on_info_done(self, token: int, report: FundamentalReport):
        if token != self._info_token:
            return
        self._report = report
        self.fundamental.set_report(report)
        self._update_quote()

    def _on_info_failed(self, token: int, message: str):
        if token == self._info_token:
            self.fundamental.set_error(message)

    def _reanalyze(self, keep_view: bool):
        if self._market is None:
            return
        params, capital, commission = self.technical.settings()
        try:
            self._ta = analyze(self._symbol, self._market.history, self._tf, params, capital, commission)
        except Exception as exc:
            self._set_status(f"Analysis failed: {exc}", error=True)
            return
        self.technical.set_analysis(self._ta, keep_view)
        self.fundamental.set_analysis(self._ta, keep_view)

    # ------------------------------------------------------------------ live mode

    def _update_timer(self, *_):
        if self.live.isChecked() and self._symbol:
            self.timer.start(REFRESH[self.refresh_combo.currentText()] * 1000)
        else:
            self.timer.stop()
        self.live_dot.setToolTip("Live updates on" if self.timer.isActive() else "Live updates off")
        set_color(self.live_dot, C.up if self.timer.isActive() else C.muted)

    def _live_tick(self):
        if self._symbol and not self._fetching:
            self._fetch_history(initial=False)
        if self.stack.currentWidget() is self.modified:
            self.modified.live_refresh()  # re-apply the trained rules to the latest daily bar

    def _flash_live(self):
        if not self.timer.isActive():
            return
        set_color(self.live_dot, "#7fffd4")
        QTimer.singleShot(600, lambda: set_color(self.live_dot, C.up if self.timer.isActive() else C.muted))

    # ------------------------------------------------------------------ header

    def _update_quote(self):
        if self._market is None:
            return
        q = self._market.quote
        self.price_lbl.setText(fmt_price(q.price, q.currency))
        if q.change is not None:
            color = C.up if q.change >= 0 else C.down
            self.change_lbl.setText(f"{'▲' if q.change >= 0 else '▼'} {q.change:+,.2f} ({q.change_pct:+.2%})")
            set_color(self.change_lbl, color, size=11, bold=True)
        else:
            self.change_lbl.setText("")
        name = self._report.name if self._report is not None else q.symbol
        fmt = "%d %b %Y %H:%M %Z" if self._tf.interval in INTRADAY else "%d %b %Y"
        self.quote_meta.setText(f"{name}  ·  last bar {q.last_bar.strftime(fmt)}  ·  {q.currency}")

    def _set_status(self, text: str, error: bool = False):
        self.status.setText(text)
        set_color(self.status, C.down if error else C.muted)

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)
