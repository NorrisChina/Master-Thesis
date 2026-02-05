#!/usr/bin/env python3
"""Matplotlib style helpers for thesis figures.

These settings aim to keep axis labels/titles/legends readable after LaTeX
scales figures down to fit the page.
"""

from __future__ import annotations


def apply_thesis_style(*, base_fontsize: int = 24) -> None:
    """Apply large, readable defaults for thesis plots."""
    import matplotlib as mpl

    fs = int(base_fontsize)
    mpl.rcParams.update(
        {
            # Typography
            "font.size": 12,  # base font size for tick labels and annotations
            "axes.titlesize": 24,  # main title font size
            "axes.labelsize": 24,  # axis label font size
            "legend.fontsize": 16,
            "legend.title_fontsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            # Lines/markers
            "lines.linewidth": 2.2,
            "lines.markersize": 7.5,
            "axes.linewidth": 1.4,
            # Output
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            # Grid
            "grid.linewidth": 1.1,
            "grid.alpha": 0.35,
        }
    )
