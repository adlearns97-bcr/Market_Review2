"""Dark trading-terminal theme shared by Qt widgets and matplotlib charts."""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication


class C:
    bg = "#0e1116"
    panel = "#151a21"
    panel2 = "#1c222b"
    border = "#29313c"
    grid = "#1f262f"
    text = "#e6e9ee"
    text2 = "#a7afba"
    muted = "#6f7885"
    accent = "#3987e5"
    up = "#26a69a"
    down = "#ef5350"
    warn = "#e0a030"
    # chart series (validated dark-mode categorical steps)
    sma_fast = "#3987e5"
    sma_slow = "#9085e9"
    band = "#6b7c95"
    rsi = "#9085e9"
    macd = "#3987e5"
    macd_signal = "#d95926"
    score = "#3987e5"
    bench = "#8a929e"


TONE = {"good": C.up, "ok": "#5cb85c", "neutral": C.text2, "warn": C.warn, "bad": C.down}
LABEL_TONE = {"STRONG BUY": "good", "BUY": "good", "HOLD": "neutral", "SELL": "bad", "STRONG SELL": "bad"}
CURRENCY = {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", "JPY": "¥", "CNY": "¥", "HKD": "HK$",
            "CAD": "C$", "AUD": "A$", "CHF": "CHF ", "KRW": "₩"}


def score_color(score: float | None) -> str:
    """0-100 score -> red / amber / green."""
    if score is None:
        return C.muted
    return C.down if score < 40 else C.warn if score < 60 else C.up


def fmt_price(v: float | None, currency: str = "") -> str:
    if v is None or v != v:
        return "—"
    prefix = CURRENCY.get(currency, f"{currency} " if currency else "")
    digits = 2 if abs(v) >= 1 else 4
    return f"{prefix}{v:,.{digits}f}"


def fmt_big(v: float | None) -> str:
    if v is None or v != v:
        return "—"
    for div, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div:
            return f"{v / div:.2f}{suffix}"
    return f"{v:.0f}"


STYLESHEET = f"""
QWidget {{ background: {C.bg}; color: {C.text}; font-size: 10pt; }}
QLabel {{ background: transparent; }}
QToolTip {{ background: {C.panel2}; color: {C.text}; border: 1px solid {C.border}; padding: 6px; }}
QFrame#Card {{ background: {C.panel}; border: 1px solid {C.border}; border-radius: 10px; }}
QFrame#Card QLabel, QFrame#Card QCheckBox {{ background: transparent; }}
QFrame#TopBar, QFrame#SideBar {{ background: {C.panel}; border: none; }}
QFrame#TopBar {{ border-bottom: 1px solid {C.border}; }}
QFrame#SideBar {{ border-right: 1px solid {C.border}; }}
QLabel#Muted {{ color: {C.muted}; }}
QLabel#Secondary {{ color: {C.text2}; }}
QLabel#CardTitle {{ color: {C.text2}; font-size: 9pt; font-weight: 600; letter-spacing: 0.5px; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {C.panel2}; border: 1px solid {C.border}; border-radius: 6px; padding: 5px 8px;
    selection-background-color: {C.accent};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {C.accent}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{ background: {C.panel2}; border: 1px solid {C.border};
    selection-background-color: {C.accent}; outline: none; }}
QPushButton {{ background: {C.panel2}; border: 1px solid {C.border}; border-radius: 6px; padding: 6px 14px; }}
QPushButton:hover {{ border-color: {C.accent}; }}
QPushButton:pressed {{ background: {C.border}; }}
QPushButton#Primary {{ background: {C.accent}; border: 1px solid {C.accent}; color: white; font-weight: 600; }}
QPushButton#Primary:hover {{ background: #4a95ee; }}
QPushButton#Nav {{ text-align: left; padding: 12px 14px; border: none; border-left: 3px solid transparent;
    border-radius: 0; background: transparent; color: {C.text2}; }}
QPushButton#Nav:hover {{ background: {C.panel2}; color: {C.text}; }}
QPushButton#Nav:checked {{ background: {C.panel2}; color: {C.text}; border-left: 3px solid {C.accent}; }}
QTabWidget::pane {{ border: 1px solid {C.border}; border-radius: 8px; background: {C.panel}; top: -1px; }}
QTabWidget::pane QWidget {{ background: {C.panel}; }}
QTabBar::tab {{ background: transparent; color: {C.text2}; padding: 7px 16px; border: none;
    border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {C.text}; border-bottom: 2px solid {C.accent}; }}
QTabBar::tab:hover {{ color: {C.text}; }}
QTableWidget, QTreeWidget, QTextBrowser {{ background: {C.panel}; alternate-background-color: {C.panel2};
    gridline-color: {C.border}; border: none; selection-background-color: #24344a; selection-color: {C.text}; }}
QHeaderView::section {{ background: {C.panel2}; color: {C.text2}; border: none;
    border-bottom: 1px solid {C.border}; padding: 6px 8px; font-weight: 600; }}
QTableCornerButton::section {{ background: {C.panel2}; border: none; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {C.border}; border-radius: 4px; min-height: 30px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {C.border}; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QSplitter::handle {{ background: {C.bg}; }}
QSplitter::handle:horizontal {{ width: 6px; }}
QSplitter::handle:vertical {{ height: 6px; }}
QStatusBar {{ background: {C.panel}; color: {C.muted}; border-top: 1px solid {C.border}; }}
QStatusBar QLabel {{ color: {C.muted}; }}
QToolBar {{ background: transparent; border: none; spacing: 2px; }}
QToolButton {{ background: transparent; border: none; border-radius: 4px; padding: 3px; }}
QToolButton:hover {{ background: {C.panel2}; }}
QToolButton:checked {{ background: #24344a; }}
QFrame#RuleBox {{ background: {C.panel2}; border: 1px solid {C.border}; border-radius: 8px; }}
QFrame#RuleBox QLabel, QFrame#RuleBox QCheckBox {{ background: transparent; }}
QFrame#RuleBox QSpinBox, QFrame#RuleBox QDoubleSpinBox {{ background: {C.panel}; }}
QFrame#VLine {{ background: {C.border}; }}
QProgressBar {{ background: {C.panel2}; border: 1px solid {C.border}; border-radius: 5px; color: {C.text2};
    text-align: center; font-size: 8pt; min-height: 14px; max-height: 14px; }}
QProgressBar::chunk {{ background: {C.accent}; border-radius: 4px; }}
QPushButton:disabled {{ background: {C.panel2}; color: {C.muted}; border-color: {C.border}; }}
QSlider::groove:horizontal {{ height: 4px; background: {C.border}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {C.accent}; width: 14px; margin: -6px 0; border-radius: 7px; }}
QSlider::sub-page:horizontal {{ background: {C.accent}; border-radius: 2px; }}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    roles = {
        QPalette.Window: C.bg, QPalette.WindowText: C.text, QPalette.Base: C.panel2,
        QPalette.AlternateBase: C.panel, QPalette.Text: C.text, QPalette.Button: C.panel2,
        QPalette.ButtonText: C.text, QPalette.Highlight: C.accent, QPalette.HighlightedText: "#ffffff",
        QPalette.ToolTipBase: C.panel2, QPalette.ToolTipText: C.text, QPalette.PlaceholderText: C.muted,
        QPalette.Link: C.accent,
    }
    for role, color in roles.items():
        pal.setColor(role, QColor(color))
    app.setPalette(pal)
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    app.setStyleSheet(STYLESHEET)
