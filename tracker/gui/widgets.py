"""Small reusable widgets: background worker, score gauges, cards and table helpers."""
from __future__ import annotations

import threading
import traceback

from PySide6.QtCore import QObject, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QHeaderView, QLabel, QStyledItemDelegate, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from ..data import DataError
from .theme import C


# --------------------------------------------------------------------------- threading


class WorkerSignals(QObject):
    finished = Signal(int)         # token
    progress = Signal(int, int)    # done, total


class Worker:
    """Runs fn(*args) on a background Python thread and signals the GUI thread when done.

    A plain Python thread is used rather than QThreadPool: curl_cffi (yfinance's HTTP
    client) corrupts the heap when its C callbacks run on Qt-created threads. The result
    stays on the worker (``result`` / ``error``) and only the integer token crosses
    threads; the token also lets the receiver ignore superseded requests."""

    def __init__(self, token: int, fn, *args, with_progress: bool = False):
        self.token, self.fn, self.args = token, fn, args
        self.with_progress = with_progress   # pass fn a progress(done, total) callback
        self.result = None
        self.error: str | None = None
        self.signals = WorkerSignals()

    def start(self):
        threading.Thread(target=self._run, daemon=True, name=f"worker-{self.token}").start()

    def _run(self):
        try:
            kwargs = {"progress": self.signals.progress.emit} if self.with_progress else {}
            self.result = self.fn(*self.args, **kwargs)
        except Exception as exc:  # surfaced in the status bar, never crashes the app
            if not isinstance(exc, (DataError, ValueError)):  # expected, user-facing problems
                traceback.print_exc()
            self.error = str(exc) or type(exc).__name__
        self.signals.finished.emit(self.token)


# --------------------------------------------------------------------------- layout helpers


def card(title: str | None = None, spacing: int = 8) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("Card")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(14, 12, 14, 14)
    lay.setSpacing(spacing)
    if title:
        t = QLabel(title.upper())
        t.setObjectName("CardTitle")
        lay.addWidget(t)
    return frame, lay


def label(text: str = "", *, name: str | None = None, size: float | None = None, bold: bool = False,
          wrap: bool = False, color: str | None = None) -> QLabel:
    lbl = QLabel(text)
    if name:
        lbl.setObjectName(name)
    style = []
    if size:
        style.append(f"font-size: {size}pt;")
    if bold:
        style.append("font-weight: 700;")
    if color:
        style.append(f"color: {color};")
    if style:
        lbl.setStyleSheet(" ".join(style))
    lbl.setWordWrap(wrap)
    lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return lbl


def set_color(lbl: QLabel, color: str, size: float | None = None, bold: bool = False) -> None:
    style = f"color: {color};"
    if size:
        style += f" font-size: {size}pt;"
    if bold:
        style += " font-weight: 700;"
    lbl.setStyleSheet(style)


# --------------------------------------------------------------------------- gauges


class ScoreBar(QWidget):
    """Horizontal gauge. Diverging (fills from zero) when lo < 0 < hi, otherwise fills from lo."""

    def __init__(self, lo: float = -100, hi: float = 100, height: int = 10, parent=None):
        super().__init__(parent)
        self.lo, self.hi = lo, hi
        self.value: float | None = None
        self.color = C.accent
        self.ticks: list[tuple[float, str]] = []
        self.setFixedHeight(height + 6)
        self._h = height

    def set_value(self, value: float | None, color: str | None = None, ticks: list[tuple[float, str]] | None = None):
        self.value = value
        if color:
            self.color = color
        if ticks is not None:
            self.ticks = ticks
        self.update()

    def sizeHint(self):
        return QSize(160, self._h + 6)

    def _x(self, v: float, w: float) -> float:
        return (min(max(v, self.lo), self.hi) - self.lo) / (self.hi - self.lo) * w

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, top = self.width(), 3
        track = QRectF(0, top, w, self._h)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(C.panel2))
        p.drawRoundedRect(track, self._h / 2, self._h / 2)
        if self.value is not None:
            origin = 0 if self.lo < 0 < self.hi else self.lo
            x0, x1 = sorted((self._x(origin, w), self._x(self.value, w)))
            p.setBrush(QColor(self.color))
            p.drawRoundedRect(QRectF(x0, top, max(x1 - x0, 3), self._h), self._h / 2, self._h / 2)
        for v, color in self.ticks:
            x = self._x(v, w)
            p.setPen(QPen(QColor(color), 2))
            p.drawLine(int(x), 0, int(x), self._h + 6)
        if self.lo < 0 < self.hi:
            p.setPen(QPen(QColor(C.muted), 1))
            x = self._x(0, w)
            p.drawLine(int(x), top, int(x), top + self._h)
        p.end()


class ProbBar(QWidget):
    """A single stacked bar split into coloured segments that sum to 100%."""

    def __init__(self, height: int = 18, parent=None):
        super().__init__(parent)
        self.parts: list[tuple[float, str]] = []
        self.setFixedHeight(height)

    def set_parts(self, parts: list[tuple[float, str]]):
        self.parts = parts
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(C.panel2))
        p.drawRoundedRect(QRectF(0, 0, w, h), 4, 4)
        x = 0.0
        for frac, color in self.parts:
            seg = w * max(frac, 0.0)
            if seg < 0.5:
                continue
            p.setBrush(QColor(color))
            p.drawRect(QRectF(x, 0, seg, h))
            if seg > 34:
                p.setPen(QColor("#0e1116"))
                p.drawText(QRectF(x, 0, seg, h), Qt.AlignCenter, f"{frac:.0%}")
                p.setPen(Qt.NoPen)
            x += seg + 2  # 2px gap between segments
        p.end()


class ComponentRow(QWidget):
    """One component of the technical score: name, points, a diverging bar and the reasons."""

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 4)
        lay.setSpacing(4)
        head = QHBoxLayout()
        self.name = label(name, bold=True)
        self.points = label("", name="Secondary")
        head.addWidget(self.name)
        head.addStretch(1)
        head.addWidget(self.points)
        lay.addLayout(head)
        self.bar = ScoreBar(-1, 1, height=6)
        lay.addWidget(self.bar)
        self.notes = label("", name="Secondary", wrap=True)
        self.notes.setStyleSheet(f"color: {C.text2}; font-size: 9pt;")
        lay.addWidget(self.notes)

    def set_reading(self, vote: float, points: float, notes: list[str]):
        color = C.up if vote > 0.05 else C.down if vote < -0.05 else C.muted
        self.bar.set_value(vote, color)
        set_color(self.points, color, bold=True)
        self.points.setText(f"{points:+.1f} pts")
        self.notes.setText("\n".join(f"• {n}" for n in notes))


class StatTile(QFrame):
    def __init__(self, title: str, tip: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(2)
        self.title = label(title, name="Muted")
        self.title.setStyleSheet(f"color: {C.muted}; font-size: 8.5pt;")
        self.value = label("—", size=13, bold=True)
        lay.addWidget(self.title)
        lay.addWidget(self.value)
        if tip:
            self.setToolTip(tip)

    def set(self, text: str, color: str | None = None):
        self.value.setText(text)
        set_color(self.value, color or C.text, size=13, bold=True)


# --------------------------------------------------------------------------- tables


def make_table(headers: list[str], stretch_last: bool = True) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QTableWidget.NoEditTriggers)
    t.setSelectionBehavior(QTableWidget.SelectRows)
    t.setSelectionMode(QTableWidget.SingleSelection)
    t.setAlternatingRowColors(True)
    t.setShowGrid(False)
    t.setWordWrap(False)
    hh = t.horizontalHeader()
    hh.setSectionResizeMode(QHeaderView.ResizeToContents)
    hh.setStretchLastSection(stretch_last)
    hh.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    return t


def fill_table(t: QTableWidget, rows: list[list[str]], colors: list[list[str | None]] | None = None,
               data: list[object] | None = None, align_right: set[int] = frozenset()):
    t.setUpdatesEnabled(False)
    t.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            item = QTableWidgetItem(text)
            if c in align_right:
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if colors and colors[r][c]:
                item.setForeground(QColor(colors[r][c]))
            if data is not None and c == 0:
                item.setData(Qt.UserRole, data[r])
            t.setItem(r, c, item)
    t.setUpdatesEnabled(True)


class ScoreBarDelegate(QStyledItemDelegate):
    """Paints a 0-100 score (stored in Qt.UserRole) as a small coloured bar with the number."""

    def paint(self, painter, option, index):
        score = index.data(Qt.UserRole)
        if score is None:
            super().paint(painter, option, index)
            return
        from .theme import score_color
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        r = option.rect.adjusted(6, 0, -6, 0)
        bar_w = max(r.width() - 34, 20)
        bar = QRectF(r.left(), r.center().y() - 3, bar_w, 6)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(C.panel2))
        painter.drawRoundedRect(bar, 3, 3)
        painter.setBrush(QColor(score_color(score)))
        painter.drawRoundedRect(QRectF(bar.left(), bar.top(), max(bar_w * score / 100, 3), 6), 3, 3)
        painter.setPen(QColor(C.text))
        painter.drawText(QRectF(r.left() + bar_w + 4, r.top(), 30, r.height()),
                         Qt.AlignRight | Qt.AlignVCenter, f"{score:.0f}")
        painter.restore()

    def sizeHint(self, option, index):
        return QSize(130, 24)
