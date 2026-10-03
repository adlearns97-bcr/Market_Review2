"""Screen 2 - Technical + fundamentals: is it a good business, at a good price, at a good time?"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QHeaderView, QLabel, QScrollArea, QSlider, QSplitter,
                               QTabWidget, QTextBrowser, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..analysis import TechnicalAnalysis
from ..combined import FUND_TIERS, MATRIX, TECH_TIERS, combine
from ..fundamentals import CATEGORIES, FundamentalReport
from .chart import CandleChart
from .theme import TONE, C, fmt_big, fmt_price, score_color
from .widgets import ScoreBar, ScoreBarDelegate, card, label, set_color

HOW_TO_READ = f"""
<style> h3 {{ color: {C.text}; margin: 10px 0 4px 0; }} p, li {{ color: {C.text2}; }} b {{ color: {C.text}; }} </style>
<h3>Two questions, two tools</h3>
<p><b>Fundamentals answer <i>what</i> to buy</b>: is this a profitable, growing, financially sound business, and
is the price reasonable for what you get? <b>Technicals answer <i>when</i></b>: are buyers or sellers in control
of the price right now?</p>
<h3>How the fundamental score works</h3>
<p>Each metric is mapped to 0-100 using rule-of-thumb ranges for a typical large company, averaged into
categories and then weighted: Valuation 25%, Profitability 25%, Growth 20%, Financial health 20%,
Market view 10%. Dividends and beta are shown for context only.</p>
<h3>Caveats worth knowing</h3>
<ul>
<li><b>Sectors differ.</b> Banks carry lots of "debt" by design, software firms have huge price-to-book
ratios, utilities grow slowly. Compare a company with its peers, not with a universal yardstick.</li>
<li><b>Snapshot, not history.</b> These numbers describe the company <i>today</i>, so they are not used in the
historical simulation - that would use information nobody had back then (look-ahead bias).</li>
<li><b>Buybacks distort ROE.</b> Companies that repurchase lots of stock shrink their equity, which can push
return on equity above 100%.</li>
<li>Analyst ratings lean optimistic - "hold" often means "sell".</li>
</ul>
"""


class FundamentalScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._ta: TechnicalAnalysis | None = None
        self._report: FundamentalReport | None = None
        self._status = "Load a symbol to begin"
        self._currency = "USD"
        self._keep_view = False
        self._last_weak: bool | None = None

        # ---------------- header
        header = QHBoxLayout()
        header.setContentsMargins(4, 0, 4, 4)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        self.name = label("—", size=16, bold=True)
        self.meta = label("", name="Secondary")
        titles.addWidget(self.name)
        titles.addWidget(self.meta)
        header.addLayout(titles, 1)

        # ---------------- left column
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 4, 0)
        left_lay.setSpacing(10)
        left_lay.addWidget(self._build_verdict_card())
        left_lay.addWidget(self._build_matrix_card())
        left_lay.addWidget(self._build_list_card("Strengths", "strengths"))
        left_lay.addWidget(self._build_list_card("Weaknesses", "weaknesses"))
        left_lay.addWidget(self._build_valuation_card())
        left_lay.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        scroll.setMinimumWidth(390)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # ---------------- right column
        self.chart = CandleChart()
        chart_box, chart_lay = card(spacing=0)
        chart_lay.setContentsMargins(4, 4, 4, 4)
        self.chart_note = label("", name="Muted")
        self.chart_note.setContentsMargins(10, 6, 10, 0)
        chart_lay.addWidget(self.chart_note)
        chart_lay.addWidget(self.chart, 1)

        tabs = QTabWidget()
        self.tree = QTreeWidget()
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(["Metric", "Value", "Score", "What it means"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setItemDelegateForColumn(2, ScoreBarDelegate(self.tree))
        hh = self.tree.header()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.Fixed)
        hh.resizeSection(2, 140)
        hh.setStretchLastSection(True)
        tabs.addTab(self.tree, "Fundamental scorecard")
        self.about = QTextBrowser()
        tabs.addTab(self.about, "About the company")
        guide = QTextBrowser()
        guide.setHtml(HOW_TO_READ)
        tabs.addTab(guide, "How to read this")

        right = QSplitter(Qt.Vertical)
        right.addWidget(chart_box)
        right.addWidget(tabs)
        right.setSizes([430, 430])

        body = QSplitter(Qt.Horizontal)
        body.addWidget(scroll)
        body.addWidget(right)
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        body.setSizes([420, 1000])

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.addLayout(header)
        lay.addWidget(body, 1)

    # ------------------------------------------------------------------ building

    def _build_verdict_card(self):
        frame, lay = card("Combined verdict")
        self.verdict = label("—", size=22, bold=True)
        self.explanation = label("", name="Secondary", wrap=True)
        lay.addWidget(self.verdict)
        lay.addWidget(self.explanation)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        self.bars, self.bar_values = {}, {}
        for r, (key, text, lo, hi) in enumerate((("combined", "Combined score", 0, 100),
                                                 ("fund", "Fundamentals", 0, 100),
                                                 ("tech", "Technicals", -100, 100))):
            grid.addWidget(label(text, name="Secondary"), r, 0)
            bar = ScoreBar(lo, hi, height=8)
            grid.addWidget(bar, r, 1)
            value = label("—", bold=True)
            value.setMinimumWidth(92)
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(value, r, 2)
            self.bars[key], self.bar_values[key] = bar, value
        grid.setColumnStretch(1, 1)
        lay.addLayout(grid)

        wrow = QHBoxLayout()
        self.weight_label = label("", name="Secondary")
        self.weight = QSlider(Qt.Horizontal)
        self.weight.setRange(0, 100)
        self.weight.setValue(60)
        self.weight.setToolTip("How much the fundamental score counts in the combined score")
        self.weight.valueChanged.connect(lambda _v: self._refresh())
        wrow.addWidget(self.weight_label)
        wrow.addWidget(self.weight, 1)
        lay.addLayout(wrow)
        return frame

    def _build_matrix_card(self):
        frame, lay = card("Decision matrix")
        lay.addWidget(label("Rows: business quality · columns: price trend", name="Muted"))
        grid = QGridLayout()
        grid.setSpacing(4)
        for c, t in enumerate(TECH_TIERS):
            h = label(t, name="Muted")
            h.setAlignment(Qt.AlignCenter)
            grid.addWidget(h, 0, c + 1)
        self.cells = {}
        for r, f in enumerate(FUND_TIERS):
            rh = label(f, name="Muted")
            grid.addWidget(rh, r + 1, 0)
            for c, t in enumerate(TECH_TIERS):
                text, tone, why = MATRIX[(f, t)]
                cell = QLabel(text)
                cell.setAlignment(Qt.AlignCenter)
                cell.setWordWrap(True)
                cell.setMinimumHeight(44)
                cell.setToolTip(why)
                self.cells[(f, t)] = (cell, tone)
                grid.addWidget(cell, r + 1, c + 1)
        for c in range(1, 4):
            grid.setColumnStretch(c, 1)
        lay.addLayout(grid)
        self._paint_matrix(None)
        return frame

    def _build_list_card(self, title: str, attr: str):
        frame, lay = card(title)
        lbl = label("—", name="Secondary", wrap=True)
        setattr(self, f"{attr}_label", lbl)
        lay.addWidget(lbl)
        return frame

    def _build_valuation_card(self):
        frame, lay = card("Price vs. value")
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        self.val = {}
        items = [("price", "Current price", ""),
                 ("graham", "Graham number", "√(22.5 × EPS × book value per share): Benjamin Graham's ceiling "
                                             "for what a defensive investor should pay. Very conservative - "
                                             "most quality growth companies trade far above it."),
                 ("target", "Analyst target (mean)", "Average 12-month price target from analysts."),
                 ("range", "Analyst target range", "Lowest and highest analyst price targets."),
                 ("analysts", "Analysts covering", ""),
                 ("wk52", "52-week range", ""),
                 ("mcap", "Market cap", "Share price × number of shares: what the whole company is worth.")]
        for r, (key, text, tip) in enumerate(items):
            n = label(text, name="Secondary")
            n.setToolTip(tip)
            v = label("—", bold=True)
            v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(n, r, 0)
            grid.addWidget(v, r, 1)
            self.val[key] = v
        lay.addLayout(grid)
        return frame

    # ------------------------------------------------------------------ public

    def set_currency(self, currency: str):
        self._currency = currency

    def show_message(self, text: str, color: str = C.text2):
        self.chart.show_message(text, color)

    def set_loading(self, symbol: str):
        self._report = None
        self._status = f"Loading fundamentals for {symbol}…"
        self.name.setText(symbol)
        self.meta.setText(self._status)
        self.tree.clear()
        self.about.setHtml("")
        self._refresh()

    def set_error(self, message: str):
        self._report = None
        self._status = f"Fundamentals unavailable: {message}"
        self.meta.setText(self._status)
        self._refresh()

    def set_report(self, report: FundamentalReport):
        self._report = report
        info = report.info
        self.name.setText(f"{report.name}  ({report.symbol})")
        kind = {"EQUITY": "Stock", "ETF": "ETF", "INDEX": "Index", "MUTUALFUND": "Mutual fund",
                "CRYPTOCURRENCY": "Crypto", "CURRENCY": "Currency", "FUTURE": "Future"}.get(report.quote_type, "")
        bits = [b for b in (report.sector, report.industry, info.get("exchange"), kind) if b]
        if info.get("marketCap"):
            bits.append(f"Market cap {fmt_big(info['marketCap'])}")
        self.meta.setText("  ·  ".join(bits) or report.note)
        self.about.setHtml(f"<h3 style='color:{C.text}'>{report.name}</h3>"
                           f"<p style='color:{C.text2}'>{report.summary or 'No description available.'}</p>")
        self._fill_tree(report)
        self._refresh()

    def set_analysis(self, ta: TechnicalAnalysis, keep_view: bool = False):
        self._ta = ta
        self._keep_view = keep_view
        self._refresh(redraw_chart=True)

    # ------------------------------------------------------------------ rendering

    def _fill_tree(self, report: FundamentalReport):
        self.tree.clear()
        bold = QFont()
        bold.setBold(True)
        for cat, weight in CATEGORIES.items():
            cat_score = report.category_scores.get(cat)
            suffix = f"  (weight {weight:.0%})" if weight else "  (context only)"
            parent = QTreeWidgetItem([cat + suffix, "", "", ""])
            parent.setFont(0, bold)
            if cat_score is not None:
                parent.setData(2, Qt.UserRole, cat_score)
            self.tree.addTopLevelItem(parent)
            for m in (m for m in report.metrics if m.spec.category == cat):
                item = QTreeWidgetItem([m.spec.name, m.display, "" if m.score is not None else "—", m.spec.meaning])
                if m.score is not None:
                    item.setData(2, Qt.UserRole, m.score)
                    item.setForeground(1, QBrush(QColor(score_color(m.score))))
                else:
                    item.setForeground(1, QBrush(QColor(C.text2)))
                item.setForeground(3, QBrush(QColor(C.text2)))
                item.setToolTip(3, m.spec.meaning)
                parent.addChild(item)
            parent.setExpanded(True)

    def _paint_matrix(self, active: tuple[str, str] | None):
        for key, (cell, tone) in self.cells.items():
            color = TONE[tone]
            if key == active:
                cell.setStyleSheet(f"background: {color}; color: #0e1116; border-radius: 6px; font-weight: 700;"
                                   f"font-size: 8.5pt; padding: 4px;")
            else:
                cell.setStyleSheet(f"background: {C.panel2}; color: {color}; border-radius: 6px; "
                                   f"border: 1px solid {C.border}; font-size: 8.5pt; padding: 4px;")

    def _refresh(self, redraw_chart: bool = False):
        w = self.weight.value() / 100
        self.weight_label.setText(f"Fundamentals weight {w:.0%}")
        report, ta = self._report, self._ta
        if ta is None:
            return
        snap = ta.snapshot()
        verdict = combine(report, snap.score, ta.params, w)

        tone = TONE[verdict.tone]
        self.verdict.setText(verdict.label)
        set_color(self.verdict, tone, size=22, bold=True)
        expl = verdict.explanation
        if report is None and self._status:
            expl = self._status + " " + expl
        self.explanation.setText(expl)

        self.bars["combined"].set_value(verdict.combined_score, score_color(verdict.combined_score))
        self.bar_values["combined"].setText(f"{verdict.combined_score:.0f} / 100")
        if verdict.fund_score is not None:
            self.bars["fund"].set_value(verdict.fund_score, score_color(verdict.fund_score))
            self.bar_values["fund"].setText(f"{verdict.fund_score:.0f} · {verdict.fund_tier}")
        else:
            self.bars["fund"].set_value(None)
            self.bar_values["fund"].setText("n/a")
        tech_color = C.up if verdict.tech_tier == "Bullish" else C.down if verdict.tech_tier == "Bearish" else C.text2
        self.bars["tech"].set_value(snap.score, tech_color,
                                    [(ta.params.buy_threshold, C.up), (ta.params.sell_threshold, C.down)])
        self.bar_values["tech"].setText(f"{snap.score:+.0f} · {verdict.tech_tier}")
        self._paint_matrix((verdict.fund_tier, verdict.tech_tier) if verdict.fund_tier else None)

        if report is not None and report.available:
            self.strengths_label.setText("\n".join(f"✓  {s}" for s in report.strengths) or "No standout strengths.")
            self.weaknesses_label.setText("\n".join(f"✗  {s}" for s in report.weaknesses) or "No major red flags.")
        else:
            msg = report.note if report is not None else self._status
            self.strengths_label.setText(msg)
            self.weaknesses_label.setText("—")
        self._fill_valuation(report, snap.close)

        weak = report is not None and report.available and report.tier == "Weak"
        if report is not None and report.available:
            note = ("▲ = technical buy points.  Fundamentals are weak, so those buys are greyed out - "
                    "they'd be speculative trades." if weak else
                    f"▲ = technical buy points.  Fundamentals are {report.tier.lower()}, so they're confirmed.")
        else:
            note = "▲ = technical buy points (no fundamental confirmation available)."
        self.chart_note.setText(note)
        if redraw_chart:
            self.chart.plot(ta, ("volume",), bands=False, weak_buys=weak,
                            keep_view=self._keep_view)
            self._last_weak = weak
        elif self._last_weak != weak:
            self.chart.plot(ta, ("volume",), bands=False, weak_buys=weak, keep_view=True)
            self._last_weak = weak

    def _fill_valuation(self, report: FundamentalReport | None, price: float):
        cur = self._currency
        self.val["price"].setText(fmt_price(price, cur))
        if report is None:
            for k in ("graham", "target", "range", "analysts", "wk52", "mcap"):
                self.val[k].setText("—")
            return
        info = report.info

        def vs_price(v):
            if not v:
                return "—"
            diff = v / price - 1
            return (f"{fmt_price(v, cur)}  <span style='color:{C.up if diff >= 0 else C.down}'>"
                    f"({diff:+.0%})</span>")

        self.val["graham"].setText(vs_price(report.graham_number()))
        self.val["target"].setText(vs_price(info.get("targetMeanPrice")))
        lo, hi = info.get("targetLowPrice"), info.get("targetHighPrice")
        self.val["range"].setText(f"{fmt_price(lo, cur)} – {fmt_price(hi, cur)}" if lo and hi else "—")
        n, key = info.get("numberOfAnalystOpinions"), info.get("recommendationKey")
        self.val["analysts"].setText(f"{n} · consensus '{str(key).replace('_', ' ')}'" if n else "—")
        lo52, hi52 = info.get("fiftyTwoWeekLow"), info.get("fiftyTwoWeekHigh")
        self.val["wk52"].setText(f"{fmt_price(lo52, cur)} – {fmt_price(hi52, cur)}" if lo52 and hi52 else "—")
        self.val["mcap"].setText(fmt_big(info.get("marketCap")) if info.get("marketCap") else "—")
