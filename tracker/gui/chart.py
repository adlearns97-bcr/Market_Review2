"""Candlestick chart (matplotlib inside Qt) with indicator panels, trade markers and a crosshair."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from ..analysis import TechnicalAnalysis
from ..data import INTRADAY
from .theme import C, fmt_big

PANEL_RATIOS = {"price": 5.0, "volume": 1.1, "rsi": 1.35, "macd": 1.35, "score": 1.35}
ALL_PANELS = ("volume", "rsi", "macd", "score")


def _price_str(v: float) -> str:
    return f"{v:,.2f}" if abs(v) >= 1 else f"{v:.4f}"


def _style_axis(ax) -> None:
    ax.set_facecolor(C.panel)
    for side, spine in ax.spines.items():
        spine.set_visible(side == "bottom")
        spine.set_color(C.border)
    ax.tick_params(colors=C.text2, labelsize=8, length=0, pad=4)
    ax.yaxis.tick_right()
    ax.grid(True, color=C.grid, linewidth=0.7)
    ax.set_axisbelow(True)


def _bars(ax, x, bottom, height, colors, width: float = 0.7, alpha: float = 1.0, zorder: int = 3,
          edges: bool = False) -> None:
    """Draw many bars as one PolyCollection - ~50x faster than ax.bar() for thousands of candles."""
    colors = np.asarray(colors)
    ok = np.isfinite(bottom) & np.isfinite(height)
    x, bottom, height, colors = x[ok], bottom[ok], height[ok], colors[ok]
    left, right, top = x - width / 2, x + width / 2, bottom + height
    verts = np.stack([np.column_stack(pair) for pair in
                      ((left, bottom), (left, top), (right, top), (right, bottom))], axis=1)
    ax.add_collection(PolyCollection(verts, facecolors=colors, edgecolors=colors if edges else "none",
                                     linewidths=0.3 if edges else 0, alpha=alpha, zorder=zorder))
    ax.autoscale_view()


def _panel_title(ax, text: str) -> None:
    ax.text(0.006, 0.93, text, transform=ax.transAxes, va="top", ha="left", fontsize=8, color=C.text2, zorder=9,
            bbox=dict(boxstyle="square,pad=0.15", fc=C.panel, ec="none", alpha=0.75))


def time_labels(index: pd.DatetimeIndex, interval: str) -> list[str]:
    if interval in INTRADAY:
        one_day = len(set(index.date)) == 1
        fmt = "%H:%M" if one_day else "%d %b %H:%M"
    elif interval == "1wk":
        fmt = "%b %Y"
    else:
        fmt = "%d %b %y"
    return list(index.strftime(fmt))


class _Toolbar(NavigationToolbar2QT):
    # Keep only navigation tools; drop "configure subplots" / "edit axes".
    toolitems = [t for t in NavigationToolbar2QT.toolitems if t[0] in ("Home", "Back", "Forward", "Pan", "Zoom", "Save")]


class CandleChart(QWidget):
    bar_clicked = Signal(int)   # bar index; -1 = back to the latest bar

    def __init__(self, parent=None, toolbar: bool = True):
        super().__init__(parent)
        self.figure = Figure(facecolor=C.panel)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = _Toolbar(self.canvas, self, coordinates=False) if toolbar else None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        if self.toolbar:
            lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, 1)

        self._axes: dict[str, object] = {}
        self._ta: TechnicalAnalysis | None = None
        self._labels: list[str] = []
        self._overlay: list = []
        self._vlines: list = []
        self._hlines: dict = {}
        self._hover_text = None
        self._bg = None
        self._default_xlim = None
        self._index = None
        self._info_idx = None

        self.canvas.mpl_connect("draw_event", self._on_draw)
        self.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.canvas.mpl_connect("axes_leave_event", self._on_leave)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("resize_event", lambda _e: self._apply_margins())
        self.show_message("Load a symbol to begin")

    # ------------------------------------------------------------------ public

    def show_message(self, text: str, color: str = C.text2) -> None:
        self.figure.clear()
        self._axes, self._overlay, self._vlines, self._hlines, self._ta = {}, [], [], {}, None
        self._hover_text = None
        self.figure.text(0.5, 0.5, text, ha="center", va="center", color=color, fontsize=11, wrap=True)
        self.canvas.draw_idle()

    def plot(self, ta: TechnicalAnalysis, panels=ALL_PANELS, *, bands: bool = True, weak_buys: bool = False,
             selected: int | None = None, keep_view: bool = False,
             plan: tuple[int, float, float] | None = None) -> None:
        """Redraw everything. keep_view preserves a zoom/pan the user made (used on live refreshes).
        plan = (bar index, target, stop) draws a buy signal's profit target and stop-loss from that bar."""
        prev_xlim = self._preserved_xlim(ta.data.index) if keep_view else None
        df, p, bt = ta.data, ta.params, ta.backtest
        n = len(df)
        x = np.arange(n)
        o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
        up = c >= o
        colors = np.where(up, C.up, C.down)

        self.figure.clear()
        names = ["price", *[k for k in ALL_PANELS if k in panels]]
        axes = self.figure.subplots(len(names), 1, sharex=True, squeeze=False,
                                    gridspec_kw={"height_ratios": [PANEL_RATIOS[k] for k in names], "hspace": 0.06})[:, 0]
        self._axes = dict(zip(names, axes))
        for ax in axes:
            _style_axis(ax)
        self._ta, self._index = ta, df.index
        self._labels = time_labels(df.index, ta.timeframe.interval)
        self._info_idx = selected

        # ---- price panel
        ax = self._axes["price"]
        thin = n > 400
        ax.vlines(x, l, h, colors=colors, linewidth=0.6 if thin else 0.9, zorder=2)
        body = np.maximum(np.abs(c - o), (np.nanmax(h) - np.nanmin(l)) * 0.0012)
        _bars(ax, x, np.minimum(o, c), body, colors, edges=True)
        if bands:
            ax.fill_between(x, df["bb_lower"], df["bb_upper"], color=C.band, alpha=0.12, linewidth=0, zorder=1,
                            label=f"Bollinger {p.bb_period},{p.bb_std:g}")
            for col in ("bb_lower", "bb_upper"):
                ax.plot(x, df[col], color=C.band, linewidth=0.7, alpha=0.6, zorder=1)
        ax.plot(x, df["sma_fast"], color=C.sma_fast, linewidth=1.3, label=f"SMA {p.sma_fast}", zorder=4)
        ax.plot(x, df["sma_slow"], color=C.sma_slow, linewidth=1.3, label=f"SMA {p.sma_slow}", zorder=4)

        pad = np.nan_to_num(df["atr"].to_numpy(float), nan=float(np.nanmean(h - l))) * 0.7
        buy_color = C.muted if weak_buys else C.up
        entries = [t.entry_idx for t in bt.trades]
        sig_exits = [t.exit_idx for t in bt.trades if t.exit_kind == "signal"]
        stop_exits = [t.exit_idx for t in bt.trades if t.exit_kind == "stop"]
        marker = dict(s=80, edgecolors=C.panel, linewidths=1.4, zorder=6)
        if entries:
            ax.scatter(entries, l[entries] - pad[entries], marker="^", color=buy_color,
                       label="Buy (fundamentals weak)" if weak_buys else "Buy", **marker)
        if sig_exits:
            ax.scatter(sig_exits, h[sig_exits] + pad[sig_exits], marker="v", color=C.down, label="Sell (signal)", **marker)
        if stop_exits:
            ax.scatter(stop_exits, h[stop_exits] + pad[stop_exits], marker="v", color=C.warn, label="Sell (stop)", **marker)
        if bt.pending:
            is_buy = bt.pending == "buy"
            y = l[-1] - pad[-1] if is_buy else h[-1] + pad[-1]
            ax.scatter([n], [y], marker="^" if is_buy else "v", facecolors="none",
                       edgecolors=C.up if is_buy else C.down, s=80, linewidths=1.6, zorder=6)
            ax.annotate(f"{bt.pending.upper()} next open", (n, y), xytext=(0, -14 if is_buy else 10),
                        textcoords="offset points", ha="center", fontsize=7.5,
                        color=C.up if is_buy else C.down)

        open_trade = bt.open_trade
        if open_trade is not None:
            span = [open_trade.entry_idx, n - 1]
            ax.hlines(open_trade.entry_price, *span, colors=C.up, linestyles="--", linewidth=0.9, alpha=0.8, zorder=4)
            stop_vals = bt.stop_line.to_numpy(float)
            ax.plot(x[open_trade.entry_idx:], stop_vals[open_trade.entry_idx:], color=C.warn, linewidth=1.0,
                    linestyle=(0, (4, 3)), zorder=4, label="Trailing stop")

        # last-price tag on the right axis
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.axhline(c[-1], color=colors[-1], linewidth=0.7, linestyle=":", alpha=0.9, zorder=2)
        ax.text(1.0, c[-1], f" {_price_str(c[-1])} ", transform=trans, va="center", ha="left", fontsize=8,
                color="white", fontweight="bold", zorder=10, clip_on=False,
                bbox=dict(boxstyle="square,pad=0.25", fc=colors[-1], ec="none"))

        # buy signal's trade plan: target and stop-loss from the signal bar to the right edge
        if plan is not None:
            x0, target, stop = plan
            x_end = n - 1 + max(3.0, n * 0.035)
            for y, color, name in ((target, C.up, "TARGET"), (stop, C.down, "STOP-LOSS")):
                ax.hlines(y, x0, x_end, colors=color, linestyles="--", linewidth=1.2, zorder=5)
                ax.text(x_end, y, f"{name} ", ha="right", va="bottom" if name == "TARGET" else "top",
                        fontsize=7.5, color=color, zorder=7,
                        bbox=dict(boxstyle="square,pad=0.15", fc=C.panel, ec="none", alpha=0.75))

        # Header block (title / hover info / legend) is offset in points so it never overlaps itself.
        at = lambda dy: transforms.offset_copy(ax.transAxes, fig=self.figure, x=4, y=dy, units="points")
        ax.text(0, 1, f"{ta.symbol}  ·  {ta.timeframe.label}", transform=at(-4), va="top",
                ha="left", fontsize=10.5, fontweight="bold", color=C.text)
        leg = ax.legend(loc="upper left", bbox_to_anchor=(0, 1), bbox_transform=at(-30), ncol=7, fontsize=7.5,
                        frameon=True, facecolor=C.panel, edgecolor=C.panel, framealpha=0.85, labelcolor=C.text2,
                        handlelength=1.6, columnspacing=1.2, borderaxespad=0, borderpad=0.2)
        leg.set_zorder(8)
        ax.margins(y=0.06)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * 0.16)  # headroom so candles stay clear of the header block

        # target / stop price tags on the right axis, nudged apart from the last-price tag if they'd overlap
        if plan is not None:
            lo, hi = ax.get_ylim()
            dpi = self.figure.dpi
            px_per_unit = self.figure.get_figheight() * dpi * ax.get_position().height / (hi - lo)
            for y, color in ((plan[1], C.up), (plan[2], C.down)):
                away = 1 if y >= c[-1] else -1
                shift_pt = away * max(0.0, 16 - abs(y - c[-1]) * px_per_unit) * 72 / dpi
                tag = transforms.offset_copy(trans, fig=self.figure, y=shift_pt, units="points")
                ax.text(1.0, y, f" {_price_str(y)} ", transform=tag, va="center", ha="left", fontsize=8,
                        color="#0e1116", fontweight="bold", zorder=10, clip_on=False,
                        bbox=dict(boxstyle="square,pad=0.25", fc=color, ec="none"))

        # ---- lower panels
        if "volume" in self._axes:
            vx = self._axes["volume"]
            v = df["Volume"].to_numpy(float)
            if np.nansum(v) > 0:
                _bars(vx, x, np.zeros(n), v, colors, alpha=0.5)
                vx.plot(x, df["vol_avg"], color=C.text2, linewidth=0.9, alpha=0.8)
                vx.yaxis.set_major_formatter(FuncFormatter(lambda y, _: fmt_big(y)))
                vx.set_ylim(0, np.nanmax(v) * 1.15)
            else:
                vx.text(0.5, 0.5, "No volume data", transform=vx.transAxes, ha="center", va="center",
                        color=C.muted, fontsize=8)
            vx.yaxis.set_major_locator(MaxNLocator(2))
            _panel_title(vx, f"Volume · {p.volume_period}-bar avg")

        if "rsi" in self._axes:
            rx = self._axes["rsi"]
            rx.axhspan(p.rsi_oversold, p.rsi_overbought, color=C.band, alpha=0.08, linewidth=0)
            rx.axhline(p.rsi_overbought, color=C.down, linewidth=0.8, linestyle="--", alpha=0.8)
            rx.axhline(p.rsi_oversold, color=C.up, linewidth=0.8, linestyle="--", alpha=0.8)
            rx.plot(x, df["rsi"], color=C.rsi, linewidth=1.2)
            rx.set_ylim(0, 100)
            rx.set_yticks([p.rsi_oversold, p.rsi_overbought])
            _panel_title(rx, f"RSI {p.rsi_period}  (above {p.rsi_overbought:g} overbought · below {p.rsi_oversold:g} oversold)")

        if "macd" in self._axes:
            mx = self._axes["macd"]
            hist = df["macd_hist"].to_numpy(float)
            _bars(mx, x, np.zeros(n), hist, np.where(hist >= 0, C.up, C.down), alpha=0.45)
            mx.plot(x, df["macd"], color=C.macd, linewidth=1.2, label="MACD")
            mx.plot(x, df["macd_signal"], color=C.macd_signal, linewidth=1.2, label="Signal")
            mx.axhline(0, color=C.border, linewidth=0.8)
            mx.yaxis.set_major_locator(MaxNLocator(3))
            mx.legend(loc="upper right", bbox_to_anchor=(1.0, 1.02), ncol=2, fontsize=7, frameon=False,
                      labelcolor=C.text2, handlelength=1.4)
            _panel_title(mx, f"MACD {p.macd_fast},{p.macd_slow},{p.macd_signal}")

        if "score" in self._axes:
            sx = self._axes["score"]
            s = df["score"].to_numpy(float)
            sx.fill_between(x, s, p.buy_threshold, where=s >= p.buy_threshold, color=C.up, alpha=0.22,
                            interpolate=True, linewidth=0)
            sx.fill_between(x, s, p.sell_threshold, where=s <= p.sell_threshold, color=C.down, alpha=0.22,
                            interpolate=True, linewidth=0)
            sx.axhline(p.buy_threshold, color=C.up, linewidth=0.8, linestyle="--")
            sx.axhline(p.sell_threshold, color=C.down, linewidth=0.8, linestyle="--")
            sx.axhline(0, color=C.border, linewidth=0.8)
            sx.plot(x, s, color=C.score, linewidth=1.2)
            sx.set_ylim(-100, 100)
            sx.set_yticks([p.sell_threshold, p.buy_threshold])
            _panel_title(sx, f"Signal score  (buy ≥ {p.buy_threshold:+g} · sell ≤ {p.sell_threshold:+g})")

        # ---- shared x axis
        labels = self._labels
        bottom = axes[-1]
        bottom.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        bottom.xaxis.set_major_formatter(FuncFormatter(lambda v, _: labels[int(round(v))] if 0 <= round(v) < n else ""))
        right_pad = max(3.0, n * 0.035) + (1 if bt.pending else 0)
        self._default_xlim = (-1.0, n - 1 + right_pad)
        ax.set_xlim(*(prev_xlim or self._default_xlim))

        if selected is not None:
            for a in axes:
                a.axvline(selected, color=C.accent, linewidth=1.0, alpha=0.55, zorder=1)

        # ---- crosshair overlay (animated artists, redrawn by blitting)
        # (added with add_artist so the hidden lines don't stretch autoscaled limits)
        style = dict(color=C.text2, linewidth=0.6, linestyle="--", alpha=0.7, visible=False, animated=True)
        self._vlines = [a.add_artist(Line2D([0, 0], [0, 1], transform=a.get_xaxis_transform(), **style))
                        for a in axes]
        self._hlines = {a: a.add_artist(Line2D([0, 1], [0, 0], transform=a.get_yaxis_transform(), **style))
                        for a in axes}
        self._hover_text = ax.text(0, 1, self._bar_info(selected if selected is not None else n - 1),
                                   transform=at(-19), va="top", ha="left", fontsize=8.2,
                                   color=C.text2, family="Consolas", animated=True, zorder=9)
        self._overlay = [*self._vlines, *self._hlines.values(), self._hover_text]
        self._apply_margins(draw=False)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ internals

    def _preserved_xlim(self, new_index: pd.DatetimeIndex):
        """If the user had zoomed/panned, return that view shifted onto the new data's positions."""
        ax = self._axes.get("price")
        if ax is None or self._default_xlim is None or self._index is None:
            return None
        cur = ax.get_xlim()
        if np.allclose(cur, self._default_xlim):
            return None
        shift = int(self._index.searchsorted(new_index[0]))  # bars dropped off the front
        return cur[0] - shift, cur[1] - shift

    def _apply_margins(self, draw: bool = True) -> None:
        if not self._axes:
            return
        w, h = max(self.canvas.width(), 300), max(self.canvas.height(), 200)
        self.figure.subplots_adjust(left=22 / w, right=1 - 62 / w, top=1 - 6 / h, bottom=24 / h)
        if draw:
            self.canvas.draw_idle()

    def _bar_info(self, i: int) -> str:
        if self._ta is None:
            return ""
        df = self._ta.data
        i = int(np.clip(i, 0, len(df) - 1))
        r = df.iloc[i]
        prev = df["Close"].iloc[i - 1] if i > 0 else r["Open"]
        chg = r["Close"] / prev - 1 if prev else 0.0
        vol = f"  Vol {fmt_big(r['Volume'])}" if r["Volume"] > 0 else ""
        return (f"{self._labels[i]}   O {_price_str(r['Open'])}  H {_price_str(r['High'])}  "
                f"L {_price_str(r['Low'])}  C {_price_str(r['Close'])}  ({chg:+.2%}){vol}   "
                f"Score {r['score']:+.0f}")

    def _on_draw(self, _event) -> None:
        if not self._overlay:
            self._bg = None
            return
        self._bg = self.canvas.copy_from_bbox(self.figure.bbox)
        for artist in self._overlay:
            self.figure.draw_artist(artist)

    def _blit(self) -> None:
        if self._bg is None:
            return
        self.canvas.restore_region(self._bg)
        for artist in self._overlay:
            self.figure.draw_artist(artist)
        self.canvas.blit(self.figure.bbox)

    def _on_move(self, event) -> None:
        if self._ta is None or event.inaxes not in self._hlines or event.xdata is None:
            return
        n = len(self._ta.data)
        i = int(np.clip(round(event.xdata), 0, n - 1))
        for line in self._vlines:
            line.set_xdata([i, i])
            line.set_visible(True)
        for ax, line in self._hlines.items():
            line.set_visible(ax is event.inaxes)
            line.set_ydata([event.ydata, event.ydata])
        self._hover_text.set_text(self._bar_info(i))
        self._blit()

    def _on_leave(self, _event) -> None:
        if self._ta is None:
            return
        for line in self._overlay[:-1]:
            line.set_visible(False)
        default = self._info_idx if self._info_idx is not None else len(self._ta.data) - 1
        self._hover_text.set_text(self._bar_info(default))
        self._blit()

    def _on_click(self, event) -> None:
        mode = getattr(self.toolbar, "mode", "") if self.toolbar else ""
        if getattr(mode, "value", mode) or self._ta is None or event.inaxes not in self._hlines or event.button != 1:
            return
        if event.dblclick:
            self.bar_clicked.emit(-1)
        elif event.xdata is not None:
            self.bar_clicked.emit(int(np.clip(round(event.xdata), 0, len(self._ta.data) - 1)))


class EquityChart(QWidget):
    """Strategy equity vs. buy & hold."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(facecolor=C.panel)
        self.canvas = FigureCanvasQTAgg(self.figure)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
        self.canvas.mpl_connect("resize_event", lambda _e: self._margins())

    def _margins(self):
        w, h = max(self.canvas.width(), 200), max(self.canvas.height(), 120)
        self.figure.subplots_adjust(left=22 / w, right=1 - 58 / w, top=1 - 8 / h, bottom=22 / h)
        self.canvas.draw_idle()

    def plot(self, ta: TechnicalAnalysis) -> None:
        bt = ta.backtest
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        _style_axis(ax)
        n = len(bt.equity)
        x = np.arange(n)
        eq = bt.equity.to_numpy(float)
        ax.fill_between(x, eq.min() * 0.98, eq, where=bt.in_position.to_numpy(bool), color=C.accent,
                        alpha=0.08, linewidth=0, step="mid")
        ax.plot(x, bt.benchmark.to_numpy(float), color=C.bench, linewidth=1.3, label="Buy & hold")
        ax.plot(x, eq, color=C.accent, linewidth=1.6, label="Strategy")
        ax.axhline(bt.capital, color=C.border, linewidth=0.8)
        labels = time_labels(bt.equity.index, ta.timeframe.interval)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: labels[int(round(v))] if 0 <= round(v) < n else ""))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: fmt_big(y)))
        ax.set_xlim(-1, n)
        handles = [Line2D([], [], color=C.accent, linewidth=1.6), Line2D([], [], color=C.bench, linewidth=1.3),
                   Patch(color=C.accent, alpha=0.15)]
        ax.legend(handles, ["Strategy", "Buy & hold", "In a trade"], loc="upper left", fontsize=7.5,
                  frameon=True, facecolor=C.panel, edgecolor=C.panel, framealpha=0.85, labelcolor=C.text2, ncol=3)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * 0.15)  # room for the legend
        self._margins()
