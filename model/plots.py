"""Static figures for docs/ (matplotlib, light theme from the reference palette)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "text.color": TEXT, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "lines.linewidth": 2,
})


def money(x, _=None):
    if x >= 1e6:
        return f"{x / 1e6:.3g}M"
    if x >= 1e3:
        return f"{x / 1e3:.3g}k"
    return f"{x:.3g}"


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def predicted_vs_actual(actual, p50, path, title):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(actual, p50, s=10, alpha=0.35, color=BLUE, edgecolors="none", label="Test sale")
    lo, hi = min(actual.min(), p50.min()) * 0.8, max(actual.max(), p50.max()) * 1.25
    ax.plot([lo, hi], [lo, hi], color=TEXT_2, linewidth=1, linestyle="--", label="Perfect prediction")
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlim(lo, hi), ax.set_ylim(lo, hi)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FuncFormatter(money))
    ax.set_xlabel("Actual sale price"), ax.set_ylabel("Predicted price (p50)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    _save(fig, path)


def error_by_bucket(labels, model_err, baseline_err, counts, path, title):
    fig, ax = plt.subplots(figsize=(6.6, 4))
    x = np.arange(len(labels))
    w = 0.38
    b1 = ax.bar(x - w / 2 - 0.01, baseline_err, w, color=ORANGE, label="Species-median baseline")
    b2 = ax.bar(x + w / 2 + 0.01, model_err, w, color=BLUE, label="Model")
    for bars in (b1, b2):
        ax.bar_label(bars, labels=[f"{v:.0%}" for v in bars.datavalues], padding=2, color=TEXT_2, fontsize=9)
    ax.set_xticks(x, [f"{l}\n({n} test sales)" for l, n in zip(labels, counts)])
    ax.set_ylim(0, max(max(model_err), max(baseline_err)) * 1.3)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Median absolute % error")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncols=2)
    _save(fig, path)


def calibration(table, path, title):
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.plot([0, 1], [0, 1], color=TEXT_2, linewidth=1, linestyle="--", label="Perfectly calibrated")
    ax.plot(table["predicted"], table["actual"], color=BLUE, marker="o", markersize=7, label="Model (test listings)")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0)), ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Predicted chance of selling within 24 h"), ax.set_ylabel("Share that actually sold")
    ax.set_title(title)
    ax.legend(loc="upper left")
    _save(fig, path)


def importance(names, values, path, title, top=20):
    order = np.argsort(values)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.barh(np.array(names)[order], np.array(values)[order], color=BLUE, height=0.7)
    ax.set_xlabel("CatBoost feature importance")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def demand_curve(prices, p_sell, floor, p50, path, title):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(prices, p_sell, color=BLUE, label="Chance of selling within 24 h")
    ax.axvline(floor, color=TEXT_2, linewidth=1, linestyle=":", label=f"GTS minimum ({money(floor)})")
    ax.axvline(p50, color=ORANGE, linewidth=1.5, linestyle="--", label=f"Predicted sale price ({money(p50)})")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(money))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda x, _: money(x) if str(int(round(x)))[0] in "25" else ""))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Asking price"), ax.set_ylabel("Chance of selling")
    ax.set_title(title)
    ax.legend(loc="upper right")
    _save(fig, path)


def line(xs, series: dict, path, title, xlabel, ylabel, percent_y=True, xticks=None):
    """series: label -> (ys, colour). Up to three series."""
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for label, (ys, colour) in series.items():
        ax.plot(xs, ys, color=colour, marker="o", markersize=7, label=label)
    if percent_y:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    if xticks is not None:
        ax.set_xticks(xs, xticks)
    ax.set_xlabel(xlabel), ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(series) > 1:
        ax.legend()
    _save(fig, path)


def bars(labels, values, path, title, xlabel, fmt="{:+.3f}"):
    fig, ax = plt.subplots(figsize=(6.8, 0.45 * len(labels) + 1.4))
    b = ax.barh(labels[::-1], values[::-1], color=BLUE, height=0.65)
    ax.bar_label(b, labels=[fmt.format(v) for v in values[::-1]], padding=3, color=TEXT_2, fontsize=9)
    ax.axvline(0, color=TEXT_2, linewidth=0.8)
    ax.margins(x=0.15)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    _save(fig, path)
