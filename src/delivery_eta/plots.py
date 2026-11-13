"""Chart helpers: one quiet style for every figure in the project."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 10,
        "legend.frameon": False,
    }
)


def save(fig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def target_distribution(y: pd.Series):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(y, bins=30, color=BLUE, edgecolor=SURFACE, linewidth=1.5)
    ax.axvline(y.mean(), color=INK_2, linewidth=1, linestyle="--")
    ax.text(y.mean(), ax.get_ylim()[1] * 0.95, f"  average {y.mean():.0f} min",
            color=INK_2, va="top")
    ax.set(title="Delivery time per order", xlabel="Minutes", ylabel="Orders")
    ax.grid(axis="x", visible=False)
    return fig


def hbar(values: pd.Series, title: str, xlabel: str, errors=None,
         highlight: str | None = None, fmt: str = "{:.1f}"):
    """Horizontal bars, largest on top, value written at the bar end."""
    values = values.sort_values()
    colors = [ORANGE if name == highlight else BLUE for name in values.index]
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(values) + 1.2))
    xerr = None if errors is None else errors.reindex(values.index).values
    ax.barh(values.index, values.values, color=colors, height=0.55, xerr=xerr,
            error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3})
    pad = values.abs().max() * 0.02
    for i, v in enumerate(values.values):
        offset = pad + (0 if xerr is None else xerr[i])
        ax.text(v + offset, i, fmt.format(v), va="center", color=INK)
    ax.set(title=title, xlabel=xlabel)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, values.max() * 1.18 + (0 if xerr is None else np.nanmax(xerr)))
    return fig


def predicted_vs_actual(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(y_true, y_pred, s=14, color=BLUE, alpha=0.55, edgecolor="none")
    lo, hi = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=1, linestyle="--")
    ax.text(hi, hi, "perfect prediction ", ha="right", va="bottom", color=INK_2)
    ax.set(title="Predicted vs actual (out-of-fold)",
           xlabel="Actual minutes", ylabel="Predicted minutes")
    ax.set_aspect("equal")
    return fig


def error_by_segments(tables: dict[str, pd.DataFrame]):
    """Small multiples: MAE by level, one panel per column, shared x-axis."""
    n = len(tables)
    heights = [len(t) for t in tables.values()]
    fig, axes = plt.subplots(n, 1, figsize=(7, 0.42 * sum(heights) + 1.1 * n),
                             sharex=True, gridspec_kw={"height_ratios": heights})
    axes = np.atleast_1d(axes)
    xmax = max(t["MAE_min"].max() for t in tables.values()) * 1.2
    for ax, (col, table) in zip(axes, tables.items()):
        t = table.sort_values("MAE_min")
        ax.barh(t.index.astype(str), t["MAE_min"], color=BLUE, height=0.55)
        for i, (v, n_orders) in enumerate(zip(t["MAE_min"], t["orders"])):
            ax.text(v + xmax * 0.01, i, f"{v:.1f}  (n={n_orders})", va="center", color=INK)
        ax.set_title(col.replace("_", " "), fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, xmax)
    axes[-1].set_xlabel("Mean absolute error (minutes)")
    fig.suptitle("Where the model is least accurate", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


def promise_tradeoff(report: pd.DataFrame):
    """Average promised time per strategy; each bar carries its on-time rate."""
    t = report.sort_values("avg_promised_min", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(t) + 1.3))
    best = t["avg_promised_min"].idxmin()
    colors = [AQUA if i == best else BLUE for i in t.index]
    ax.barh(t["strategy"], t["avg_promised_min"], color=colors, height=0.55)
    xmax = t["avg_promised_min"].max()
    for i, (_, row) in enumerate(t.iterrows()):
        ax.text(row["avg_promised_min"] + xmax * 0.01, i,
                f"{row['avg_promised_min']:.1f} min  ·  {row['on_time_%']:.1f}% on time",
                va="center", color=INK)
    ax.set_xlim(0, xmax * 1.45)
    ax.grid(axis="y", visible=False)
    ax.set(title="Same 90% on-time target, how long each strategy makes customers wait",
           xlabel="Average promised delivery time (minutes)")
    return fig
