"""Plotting utilities for result analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_pareto_plot(summary: pd.DataFrame, path: str | Path) -> None:
    """Save BLEU-vs-CNAL Pareto plot."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(summary["avg_cnal"], summary["BLEU"])
    for _, row in summary.iterrows():
        ax.annotate(str(row["strategy"]), (row["avg_cnal"], row["BLEU"]), fontsize=8)
    ax.set_xlabel("CNAL latency (lower is better)")
    ax.set_ylabel("BLEU (higher is better)")
    ax.set_title("Quality-Latency Pareto Frontier")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(p, dpi=200)
    plt.close(fig)


def save_erasure_barplot(summary: pd.DataFrame, path: str | Path) -> None:
    """Save Detokenized Erasure bar plot."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(summary["strategy"], summary["avg_erasure"])
    ax.set_ylabel("Average Detokenized Erasure / sentence")
    ax.set_title("Visible Stability")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(p, dpi=200)
    plt.close(fig)
