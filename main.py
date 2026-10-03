"""Market Tracker - live candlestick charts, technical signals, a trading simulation,
and a fundamental scorecard. Run:  python main.py [SYMBOL]"""
import logging
import sys

from PySide6.QtWidgets import QApplication

from tracker.gui.main_window import MainWindow
from tracker.gui.theme import apply_theme


def main() -> int:
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # errors are shown in the UI instead
    app = QApplication(sys.argv)
    app.setApplicationName("Market Tracker")
    apply_theme(app)
    window = MainWindow(sys.argv[1] if len(sys.argv) > 1 else "AAPL")
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
