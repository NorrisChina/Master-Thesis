#!/usr/bin/env python3
from __future__ import annotations
import os
import numpy as np


def add_advantage_shading_by_baseline(
    ax,
    x: np.ndarray,
    baseline: np.ndarray,
    *,
    advantage_color: str = "#c8f7c5",  # light green
    penalty_color: str = "#f7c2c2",    # light red
    alpha: float = 0.22,
    label_regions: bool = True,
) -> None:
    """Shade regions relative to a baseline curve.

    - Green: y < baseline (advantage vs baseline)
    - Red:   y > baseline (penalty vs baseline)

    Works for linear and log y-scales as long as y-limits and baseline are > 0.
    """
    if x.size == 0:
        return
    if baseline.shape != x.shape:
        raise ValueError("baseline must have same shape as x")

    y0, y1 = ax.get_ylim()
    lo = min(y0, y1)
    hi = max(y0, y1)
    y_lo = np.full_like(x, lo, dtype=float)
    y_hi = np.full_like(x, hi, dtype=float)

    # Draw behind lines/grid.
    ax.fill_between(
        x,
        y_lo,
        baseline,
        where=baseline > y_lo,
        facecolor=advantage_color,
        alpha=alpha,
        # Must be above the Axes background patch to be visible.
        zorder=0.5,
        interpolate=True,
    )
    ax.fill_between(
        x,
        baseline,
        y_hi,
        where=y_hi > baseline,
        facecolor=penalty_color,
        alpha=alpha,
        zorder=0.5,
        interpolate=True,
    )

    if label_regions:
        # Put small labels near the left side; works for both linear and log axes.
        x0 = float(x[0])
        y_mid_adv = float(np.clip(np.median(np.minimum(baseline, y_hi)), lo, hi))
        y_mid_pen = float(np.clip(np.median(np.maximum(baseline, y_lo)), lo, hi))

        # Nudge labels away from the baseline.
        y_adv = max(lo, min(hi, y_mid_adv * 0.75))
        y_pen = max(lo, min(hi, y_mid_pen * 1.25))

        ax.text(
            x0,
            y_adv,
            "Advantage",
            fontsize=9,
            color="#1b5e20",
            alpha=0.9,
            va="center",
            ha="left",
            zorder=5,
        )
        ax.text(
            x0,
            y_pen,
            "Penalty",
            fontsize=9,
            color="#7f1d1d",
            alpha=0.9,
            va="center",
            ha="left",
            zorder=5,
        )

OUT_DIR = os.path.join("experiments", "results")
os.makedirs(OUT_DIR, exist_ok=True)


def bits(x_bytes: float) -> float:
    return 8.0 * x_bytes


def p_miss_from_nver(nver_bits: float) -> float:
    return 2.0 ** (-nver_bits)


def latency_traditional(B: np.ndarray, n_data_bits: float) -> np.ndarray:
    return n_data_bits / B


def latency_id(B: np.ndarray, n_data_bits: float, n_ver_bits: float, t_ver: float, p_desync: float, p_miss: float) -> np.ndarray:
    return t_ver + (n_ver_bits + p_desync * (1.0 - p_miss) * n_data_bits) / B


def break_even_bandwidth(n_data_bits: float, n_ver_bits: float, t_ver: float, p_desync: float, p_miss: float) -> float | None:
    num = (n_data_bits * (1.0 - p_desync * (1.0 - p_miss)) - n_ver_bits)
    if t_ver <= 0:
        return None
    B = num / t_ver
    if B <= 0:
        return None
    return B
