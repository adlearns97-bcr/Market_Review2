"""Charts for the Modified Strat screen: history + trades + projected target cone, and time-to-target."""
from __future__ import annotations

import numpy as np
from matplotlib import transforms
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from ..custom_strategy import StrategyReport
from ..forecast import bar_unit, future_dates
from .chart import _Toolbar, _bars, _price_str, _style_axis, time_labels
from .theme import C

DEFAULT_BARS_SHOWN = 130


class StrategyChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(facecolor=C.panel)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = _Toolbar(self.canvas, self, coordinates=False)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, 1)
        self._price_ax = self._eq_ax = None
        self._ylo = self._yhi = None
        self._eq_lines: list = []
        self._default_xlim = None
        self.canvas.mpl_connect("resize_event", lambda _e: self._margins())
        self.show_message("Press “Train & optimise” to build a strategy")

    def show_message(self, text: str, color: str = C.text2) -> None:
        self.figure.clear()
        self._price_ax = None
        self.figure.text(0.5, 0.5, text, ha="center", va="center", color=color, fontsize=11, wrap=True)
        self.canvas.draw_idle()

    def _margins(self, draw: bool = True) -> None:
        if self._price_ax is None:
            return
        w, h = max(self.canvas.width(), 300), max(self.canvas.height(), 200)
        self.figure.subplots_adjust(left=22 / w, right=1 - 62 / w, top=1 - 6 / h, bottom=24 / h)
        if draw:
            self.canvas.draw_idle()

    def _fit_y(self, ax) -> None:
        """Rescale the price axis to whatever bars are visible (history and projection)."""
        if self._ylo is None:
            return
        lo, hi = ax.get_xlim()
        a, b = max(int(np.floor(lo)), 0), min(int(np.ceil(hi)) + 1, len(self._ylo))
        if b - a < 2:
            return
        ymin, ymax = np.nanmin(self._ylo[a:b]), np.nanmax(self._yhi[a:b])
        pad = (ymax - ymin) * 0.06
        ax.set_ylim(ymin - pad, ymax + pad * 3.2)  # extra headroom for the legend
        # Equity panel: rebase both curves to 100 at the left edge of the view so they stay comparable.
        if self._eq_ax is not None and self._eq_lines:
            m = len(self._eq_lines[0][1])
            a0, b0 = min(a, m - 1), min(b, m)
            lows, highs = [], []
            for line, raw in self._eq_lines:
                y = raw / raw[a0] * 100
                line.set_ydata(y)
                lows.append(np.nanmin(y[a0:b0]))
                highs.append(np.nanmax(y[a0:b0]))
            lo_, hi_ = min(lows), max(highs)
            pad = max(hi_ - lo_, 2.0) * 0.1
            self._eq_ax.set_ylim(lo_ - pad, hi_ + pad * 3.5)

    def plot(self, rep: StrategyReport, keep_view: bool = False) -> None:
        prev = None
        if keep_view and self._price_ax is not None and self._default_xlim is not None:
            cur = self._price_ax.get_xlim()
            if not np.allclose(cur, self._default_xlim):
                prev = cur

        df, fc, v, s0 = rep.df, rep.fc, rep.best, rep.start
        m = rep.n - s0
        x = np.arange(m)
        o, h, l, c = (a[s0:] for a in (fc.o, fc.h, fc.l, fc.c))
        colors = np.where(c >= o, C.up, C.down)
        sig, fcst = rep.signal, rep.forecast
        horizon = fcst.horizon if fcst else int(v["max_bars"])
        x_last, x_end = m - 1, m - 1 + horizon
        split_x = rep.split - s0

        self.figure.clear()
        ax, ex = self.figure.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [4, 1.25], "hspace": 0.06})
        self._price_ax, self._eq_ax = ax, ex
        for a in (ax, ex):
            _style_axis(a)
            a.axvspan(split_x - 0.5, x_last + 0.5, color=C.accent, alpha=0.055, linewidth=0, zorder=0,
                      label="Test period (unseen)" if a is ax else None)
            a.axvline(x_last + 0.5, color=C.border, linewidth=0.9, zorder=1)

        # ---- candles + the lines behind the chosen rules
        ax.vlines(x, l, h, colors=colors, linewidth=0.6 if m > 400 else 0.9, zorder=2)
        body = np.maximum(np.abs(c - o), (np.nanmax(h) - np.nanmin(l)) * 0.0008)
        _bars(ax, x, np.minimum(o, c), body, colors, edges=True)
        rules = rep.cfg.rules
        if "trend" in rules:
            n = int(v["trend_len"])
            ax.plot(x, fc.sma(n)[s0:], color=C.sma_slow, linewidth=1.3, label=f"SMA {n}", zorder=4)
        if "ema_cross" in rules:
            f, s = int(v["ema_fast"]), int(v["ema_slow"])
            ax.plot(x, fc.ema(f)[s0:], color=C.sma_fast, linewidth=1.0, label=f"EMA {f}", zorder=4)
            ax.plot(x, fc.ema(s)[s0:], color=C.macd_signal, linewidth=1.0, label=f"EMA {s}", zorder=4)
        if "bb_dip" in rules:
            ax.plot(x, fc.bb_lower(v["bb_k"])[s0:], color=C.band, linewidth=0.9, linestyle="--",
                    label=f"Lower band {v['bb_k']:g}σ", zorder=4)
        if "breakout" in rules:
            n = int(v["breakout_len"])
            ax.plot(x, fc.prior_high(n)[s0:], color=C.warn, linewidth=0.9, linestyle=":", label=f"{n}-bar high", zorder=4)

        # ---- trades
        pad = np.nan_to_num(fc.atr[s0:], nan=float(np.nanmean(h - l))) * 0.7
        mk = dict(s=70, edgecolors=C.panel, linewidths=1.3, zorder=6)
        trades = rep.full.trades
        entries = [t.entry_idx - s0 for t in trades]
        if entries:
            ax.scatter(entries, l[entries] - pad[entries], marker="^", color=C.accent, label="Buy", **mk)
        for reason, color, text in (("target", C.up, "Sell · target"), ("stop", C.down, "Sell · stop-loss"),
                                    ("time", C.muted, "Sell · time limit")):
            xs = [t.exit_idx - s0 for t in trades if t.reason == reason]
            if xs:
                ax.scatter(xs, h[xs] + pad[xs], marker="v", color=color, label=text, **mk)

        # ---- the plan: target / stop lines and the simulated cone
        x0 = sig.trade.entry_idx - s0 if sig.trade is not None else x_last
        hypothetical = sig.action in ("NO BUY", "WAIT")
        style = (0, (2, 3)) if hypothetical else "--"
        ax.hlines(sig.target, x0, x_end, colors=C.up, linestyles=[style], linewidth=1.2, zorder=5)
        ax.hlines(sig.stop, x0, x_end, colors=C.down, linestyles=[style], linewidth=1.2, zorder=5)
        # price tags on the right axis, like the last-price tag
        tag = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        for y, color, name in ((sig.target, C.up, "TARGET"), (sig.stop, C.down, "STOP")):
            ax.text(1.0, y, f" {_price_str(y)} ", transform=tag, va="center", ha="left", fontsize=8, color="#0e1116",
                    fontweight="bold", zorder=10, clip_on=False, bbox=dict(boxstyle="square,pad=0.25", fc=color, ec="none"))
            ax.text(x_end, y, f"{name}{' (if bought now)' if hypothetical else ''} ", ha="right",
                    va="bottom" if name == "TARGET" else "top", fontsize=7.5, color=color, zorder=7,
                    bbox=dict(boxstyle="square,pad=0.15", fc=C.panel, ec="none", alpha=0.75))
        if sig.trade is not None:
            ax.hlines(sig.entry, x0, x_last, colors=C.accent, linestyles=":", linewidth=1.0, zorder=5)
        if sig.action == "BUY":
            ax.scatter([x_last + 1], [l[-1] - pad[-1]], marker="^", facecolors="none", edgecolors=C.accent,
                       s=90, linewidths=1.6, zorder=6)

        fx = x_last + np.arange(horizon + 1)
        if fcst is not None:
            b = fcst.bands
            ax.fill_between(fx, b[5], b[95], color=C.accent, alpha=0.10, linewidth=0, zorder=1,
                            label="Simulated 5–95% range")
            ax.fill_between(fx, b[25], b[75], color=C.accent, alpha=0.20, linewidth=0, zorder=1)
            ax.plot(fx, b[50], color=C.accent, linewidth=1.2, linestyle="--", zorder=3, label="Median path")
            if fcst.median_bars:
                xm = x_last + fcst.median_bars
                ax.plot([xm], [sig.target], marker="D", color=C.up, markersize=7, markeredgecolor=C.panel, zorder=8)
                ax.annotate(f"≈{fcst.median_bars:.0f} {bar_unit(df.index)}", (xm, sig.target), xytext=(0, 9),
                            textcoords="offset points", ha="center", fontsize=8, color=C.up, zorder=8)

        # last price tag
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(1.0, c[-1], f" {_price_str(c[-1])} ", transform=trans, va="center", ha="left", fontsize=8,
                color="white", fontweight="bold", zorder=10, clip_on=False,
                bbox=dict(boxstyle="square,pad=0.25", fc=colors[-1], ec="none"))

        # header + train/test labels
        at = lambda dy: transforms.offset_copy(ax.transAxes, fig=self.figure, x=4, y=dy, units="points")
        ax.text(0, 1, f"{rep.symbol}  ·  optimised strategy  ·  daily bars", transform=at(-4), va="top", ha="left",
                fontsize=10.5, fontweight="bold", color=C.text)
        leg = ax.legend(loc="upper left", bbox_to_anchor=(0, 1), bbox_transform=at(-20), ncol=6, fontsize=7.5,
                        frameon=True, facecolor=C.panel, edgecolor=C.panel, framealpha=0.85, labelcolor=C.text2,
                        handlelength=1.6, columnspacing=1.2, borderaxespad=0, borderpad=0.2)
        leg.set_zorder(9)
        bottom = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(split_x - 1, 0.015, "◀ TRAINING (optimised on this) ", transform=bottom, ha="right", va="bottom",
                fontsize=7.5, color=C.muted, zorder=7)
        ax.text(split_x + 1, 0.015, " TEST (unseen) ▶", transform=bottom, ha="left", va="bottom",
                fontsize=7.5, color=C.accent, zorder=7)
        ax.text(x_last + 1.5, 0.015, "FUTURE ▶", transform=bottom, ha="left", va="bottom", fontsize=7.5,
                color=C.muted, zorder=7)

        # ---- equity panel
        bh_raw, eq_raw = c / o[0], np.asarray(rep.full.equity, float)
        bh_line, = ex.plot(x, bh_raw * 100, color=C.bench, linewidth=1.1, label="Buy & hold")
        eq_line, = ex.plot(x, eq_raw * 100, color=C.accent, linewidth=1.5, label="Optimised strategy")
        self._eq_lines = [(bh_line, bh_raw), (eq_line, eq_raw)]
        ex.yaxis.set_major_locator(MaxNLocator(3))
        ex.legend(loc="upper left", fontsize=7.2, frameon=True, facecolor=C.panel, edgecolor=C.panel,
                  framealpha=0.85, labelcolor=C.text2, ncol=3, borderaxespad=0.3,
                  title=None).set_zorder(9)
        ex.text(0.995, 0.93, "growth of 100 from the left edge of the view", transform=ex.transAxes, ha="right",
                va="top", fontsize=7.2, color=C.muted)

        # ---- axis labels (history dates, then future business days)
        labels = time_labels(df.index[s0:], "1d") + [d.strftime("%d %b %y") for d in future_dates(df.index, horizon + 10)]
        ex.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        ex.xaxis.set_major_formatter(FuncFormatter(lambda val, _: labels[int(round(val))] if 0 <= round(val) < len(labels) else ""))

        # y-range per bar, used to auto-fit the price axis to the visible window
        size = x_end + 1
        ylo = np.full(size, np.nan)
        yhi = np.full(size, np.nan)
        ylo[:m], yhi[:m] = l - pad, h + pad
        ylo[x0:] = np.fmin(ylo[x0:], sig.stop)
        yhi[x0:] = np.fmax(yhi[x0:], sig.target)
        if fcst is not None:
            ylo[x_last:] = np.fmin(ylo[x_last:], fcst.bands[5])
            yhi[x_last:] = np.fmax(yhi[x_last:], fcst.bands[95])
        self._ylo, self._yhi = ylo, yhi

        lo = max(-1.0, x_last - DEFAULT_BARS_SHOWN)
        self._default_xlim = (lo, x_end + 2.0)
        ax.callbacks.connect("xlim_changed", self._fit_y)
        ax.set_xlim(*(prev or self._default_xlim))
        self._fit_y(ax)
        self._margins(draw=False)
        self.canvas.draw_idle()


class TimeToTargetChart(QWidget):
    """Distribution of simulated bars-until-target, with the strategy's real past hits as ticks."""

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
        self.figure.subplots_adjust(left=22 / w, right=1 - 48 / w, top=1 - 10 / h, bottom=40 / h)
        self.canvas.draw_idle()

    def plot(self, rep: StrategyReport) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        _style_axis(ax)
        f = rep.forecast
        unit = bar_unit(rep.df.index)
        if f is None or not len(f.hit_times):
            ax.text(0.5, 0.5, "No simulated path reached the target within the time limit.",
                    transform=ax.transAxes, ha="center", va="center", color=C.text2)
            self._margins()
            return
        step = max(1, f.horizon // 40)
        bins = np.arange(0.5, f.horizon + step + 0.5, step)
        ax.hist(f.hit_times, bins=bins, color=C.up, alpha=0.75, rwidth=0.85,
                label=f"Simulated paths that hit the target first ({f.p_target:.0%} of {f.n_paths:,})")
        ax.axvline(f.median_bars, color=C.text, linewidth=1.1, linestyle="--")
        ax.axvspan(f.q25, f.q75, color=C.up, alpha=0.08, linewidth=0, label="Middle 50% of those paths")
        ax.text(f.median_bars, 0.97, f"  median ≈ {f.median_bars:.0f} {unit}", transform=ax.get_xaxis_transform(),
                va="top", ha="left", fontsize=8, color=C.text)
        hits = rep.history["hit_bars"]
        if hits:
            ax.scatter(hits, np.zeros(len(hits)), marker="|", s=220, color=C.accent, linewidths=1.6, zorder=5,
                       clip_on=False, label=f"This strategy's past target hits ({len(hits)})")
        ax.set_xlim(0, f.horizon + 1)
        ax.set_xlabel(f"{unit.capitalize()} until the target is touched", color=C.text2, fontsize=8)
        ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
        ax.legend(loc="upper right", fontsize=7.5, frameon=True, facecolor=C.panel, edgecolor=C.panel,
                  framealpha=0.85, labelcolor=C.text2)
        self._margins()
