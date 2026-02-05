#!/usr/bin/env python3
"""Plot Random Walk & Correction sweep results + generate representative time-series.

Typical workflow:
  /path/to/python experiments/random_walk_drift_sweep.py --out-csv experiments/results/random_walk_drift_sweep.csv
  /path/to/python experiments/plot_random_walk_drift_sweep.py --csv experiments/results/random_walk_drift_sweep.csv \
      --out-dir thesis_report/figures/plots

This script produces thesis-ready PNGs.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_thesis_style


@dataclass
class Row:
    step_dist: str
    step_param: float
    ticks: int
    replicates: int
    seed: int

    k_radius: float
    n_ver_bits: int

    value_resolution: float
    danger_threshold: float

    p_miss_event: float

    check_rate: float
    miss_rate_given_check: float
    correction_rate: float

    breach_rate: float
    breach_streak_mean: float
    breach_streak_p95: float
    breach_streak_max: int

    danger_rate: float
    danger_streak_mean: float
    danger_streak_p95: float
    danger_streak_max: int

    max_error_mean: float
    max_error_p95: float


def load_rows(path: Path) -> List[Row]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: List[Row] = []
        for r in reader:
            rows.append(
                Row(
                    step_dist=str(r["step_dist"]),
                    step_param=float(r["step_param"]),
                    ticks=int(r["ticks"]),
                    replicates=int(r["replicates"]),
                    seed=int(r["seed"]),
                    k_radius=float(r["k_radius"]),
                    n_ver_bits=int(r["n_ver_bits"]),
                    value_resolution=float(r["value_resolution"]),
                    danger_threshold=float(r["danger_threshold"]),
                    p_miss_event=float(r["p_miss_event"]),
                    check_rate=float(r["check_rate"]),
                    miss_rate_given_check=float(r["miss_rate_given_check"]),
                    correction_rate=float(r["correction_rate"]),
                    breach_rate=float(r["breach_rate"]),
                    breach_streak_mean=float(r["breach_streak_mean"]),
                    breach_streak_p95=float(r["breach_streak_p95"]),
                    breach_streak_max=int(r["breach_streak_max"]),
                    danger_rate=float(r["danger_rate"]),
                    danger_streak_mean=float(r["danger_streak_mean"]),
                    danger_streak_p95=float(r["danger_streak_p95"]),
                    danger_streak_max=int(r["danger_streak_max"]),
                    max_error_mean=float(r["max_error_mean"]),
                    max_error_p95=float(r["max_error_p95"]),
                )
            )
    return rows


def group_by_nver(rows: List[Row]) -> Dict[int, List[Row]]:
    grouped: Dict[int, List[Row]] = {}
    for r in rows:
        grouped.setdefault(r.n_ver_bits, []).append(r)
    for nv in grouped:
        grouped[nv].sort(key=lambda x: x.k_radius)
    return grouped


def plot_metric_vs_k(
    rows: List[Row],
    *,
    metric: str,
    ylabel: str,
    title: str | None,
    out_path: Path,
    ylog: bool = False,
) -> None:
    grouped = group_by_nver(rows)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for nver, rs in sorted(grouped.items()):
        xs = [r.k_radius for r in rs]
        ys = [getattr(r, metric) for r in rs]
        ax.plot(xs, ys, marker="o", linewidth=2.0, label=f"$n_{{ver}}={nver}$")

    ax.set_xlabel("Tolerance radius $R$")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    if ylog:
        ax.set_yscale("log")
    ax.legend(frameon=True)

    # The thesis style uses large label fonts; explicitly reserve margin so the
    # y-axis label doesn't get cropped in the saved PNG.
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.20, top=0.97)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def sample_step(rng: np.random.Generator, *, step_dist: str, step_param: float) -> float:
    if step_dist == "uniform":
        return float(rng.uniform(-step_param, step_param))
    if step_dist == "gaussian":
        return float(rng.normal(0.0, step_param))
    raise ValueError(step_dist)


def p_miss_from_accept_set(k_radius: float, n_ver_bits: int, value_resolution: float) -> float:
    accept_count = int(2 * math.floor(float(k_radius) / float(value_resolution)) + 1)
    accept_count = max(1, accept_count)
    return max(0.0, min(1.0, accept_count / float(2**int(n_ver_bits))))


def simulate_trace(
    *,
    ticks: int,
    seed: int,
    k_radius: float,
    n_ver_bits: int,
    step_dist: str,
    step_param: float,
    value_resolution: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    robot = 50.0
    dt = 50.0

    p_miss = p_miss_from_accept_set(k_radius, n_ver_bits, value_resolution)

    err = np.zeros(int(ticks), dtype=float)
    corr = np.zeros(int(ticks), dtype=int)
    miss = np.zeros(int(ticks), dtype=int)
    checked = np.zeros(int(ticks), dtype=int)

    for t in range(int(ticks)):
        robot += sample_step(rng, step_dist=step_dist, step_param=step_param)
        e = abs(robot - dt)
        err[t] = e

        if e <= k_radius:
            continue

        checked[t] = 1
        if rng.random() < p_miss:
            miss[t] = 1
            continue

        # correction
        corr[t] = 1
        dt = robot

    return err, checked, miss, corr


def plot_timeseries_two_panel(
    *,
    ticks: int,
    seed: int,
    k_radius: float,
    danger_threshold: float,
    step_dist: str,
    step_param: float,
    value_resolution: float,
    nver_a: int,
    nver_b: int,
    out_path: Path,
) -> None:
    err_a, checked_a, miss_a, corr_a = simulate_trace(
        ticks=ticks,
        seed=seed,
        k_radius=k_radius,
        n_ver_bits=nver_a,
        step_dist=step_dist,
        step_param=step_param,
        value_resolution=value_resolution,
    )
    err_b, checked_b, miss_b, corr_b = simulate_trace(
        ticks=ticks,
        seed=seed,
        k_radius=k_radius,
        n_ver_bits=nver_b,
        step_dist=step_dist,
        step_param=step_param,
        value_resolution=value_resolution,
    )

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.8), sharex=True)

    for ax, err, checked, miss, corr, nver in [
        (axes[0], err_a, checked_a, miss_a, corr_a, nver_a),
        (axes[1], err_b, checked_b, miss_b, corr_b, nver_b),
    ]:
        t = np.arange(len(err))
        ax.plot(t, err, color="black", linewidth=1.3, label=r"$E_t = |S_t-\hat S_t|$")
        ax.axhline(k_radius, color="#d62728", linestyle="--", linewidth=1.2, label=r"$R$")
        ax.axhline(danger_threshold, color="#9467bd", linestyle=":", linewidth=1.3, label=r"$D$ (danger)")

        # Mark correction events
        corr_idx = np.where(corr > 0)[0]
        if len(corr_idx) > 0:
            ax.vlines(corr_idx, ymin=0, ymax=np.maximum(err[corr_idx], k_radius), color="#2ca02c", alpha=0.18)

        # Mark miss events (only when checked)
        miss_idx = np.where(miss > 0)[0]
        if len(miss_idx) > 0:
            ax.scatter(miss_idx, err[miss_idx], s=36, color="#ff7f0e", alpha=0.8, label="miss")

        ax.set_ylabel("Error")
        ax.set_title(f"{step_dist} walk, $n_{{ver}}={nver}$")
        ax.grid(True, which="both", alpha=0.22)

    axes[-1].set_xlabel("Time step $t$")
    handles, labels = axes[0].get_legend_handles_labels()
    # De-duplicate legend entries
    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        h2.append(h)
        l2.append(l)
    axes[0].legend(handles=h2, labels=l2, frameon=True, loc="upper left", fontsize=18)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def generate_escape_trajectory(
    *,
    n_steps: int,
    seed: int,
    target: float,
    k_radius: float,
    step_dist: str,
    step_param: float,
    p_collision: float,
    reset_on_correction: bool,
    clip_state: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, float]], list[tuple[int, float]]]:
    """Generate an illustrative 1D drift trajectory with correction and misses.

    Design choice (for visualization): DT uses a fixed reference estimate (ZOH to target).
    On successful correction we optionally reset the robot to the reference to make the
    "cage" effect visually explicit.
    """

    rng = np.random.default_rng(seed)
    s = float(target)
    dt = float(target)

    true_states = np.zeros(int(n_steps), dtype=float)
    dt_estimates = np.zeros(int(n_steps), dtype=float)
    corrections: list[tuple[int, float]] = []
    misses: list[tuple[int, float]] = []

    true_states[0] = s
    dt_estimates[0] = dt

    for t in range(1, int(n_steps)):
        s += sample_step(rng, step_dist=step_dist, step_param=step_param)
        if clip_state is not None:
            lo, hi = clip_state
            s = float(min(max(s, lo), hi))
        dt_est = dt  # fixed reference estimate

        err = abs(s - dt_est)
        if err > k_radius:
            if rng.random() < float(p_collision):
                misses.append((t, float(s)))
            else:
                # Record the location that triggered the correction (pre-reset).
                # This shows points outside the safety set being corrected.
                s_pre = float(s)
                if reset_on_correction:
                    s = float(target)
                corrections.append((t, s_pre))

        true_states[t] = s
        dt_estimates[t] = dt_est

    return true_states, dt_estimates, corrections, misses


def find_seed_for_escape(
    *,
    n_steps: int,
    seed_start: int,
    seed_limit: int,
    target: float,
    k_radius: float,
    danger_threshold: float,
    step_dist: str,
    step_param: float,
    p_collision: float,
    reset_on_correction: bool,
    clip_state: tuple[float, float] | None = None,
) -> int:
    """Find a seed that yields at least one miss and a clear unsafe excursion."""

    best_seed = int(seed_start)
    best_score = -1.0

    for seed in range(int(seed_start), int(seed_limit)):
        s, dt, corr, miss = generate_escape_trajectory(
            n_steps=n_steps,
            seed=seed,
            target=target,
            k_radius=k_radius,
            step_dist=step_dist,
            step_param=step_param,
            p_collision=p_collision,
            reset_on_correction=reset_on_correction,
            clip_state=clip_state,
        )

        if not miss:
            continue

        err = np.abs(s - dt)
        max_err = float(np.max(err))
        unsafe_run = int(np.max(np.convolve((err > k_radius).astype(int), np.ones(1, dtype=int), mode="same")))
        crosses_danger = bool(np.any(err > float(danger_threshold)))
        # Prefer trajectories that cross danger, and with larger peak drift.
        score = (1000.0 if crosses_danger else 0.0) + max_err + 0.1 * len(miss)

        if score > best_score:
            best_score = score
            best_seed = seed

    return best_seed


def plot_escape_timeseries(
    *,
    n_steps: int,
    seed: int,
    target: float,
    k_radius: float,
    danger_threshold: float,
    step_dist: str,
    step_param: float,
    p_collision: float,
    reset_on_correction: bool,
    clip_state: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    out_path: Path,
) -> None:
    s, dt, corrections, misses = generate_escape_trajectory(
        n_steps=n_steps,
        seed=seed,
        target=target,
        k_radius=k_radius,
        step_dist=step_dist,
        step_param=step_param,
        p_collision=p_collision,
        reset_on_correction=reset_on_correction,
        clip_state=clip_state,
    )

    safe_min = float(target - k_radius)
    safe_max = float(target + k_radius)

    fig, ax = plt.subplots(figsize=(12.0, 6.6))
    fontsize = 24

    # Background bands
    ax.axhspan(safe_min, safe_max, color="#2ca02c", alpha=0.14, label=r"Semantic safety set $\mathcal{A}_R$")
    ax.axhspan(-1e9, safe_min, color="#d62728", alpha=0.06)
    ax.axhspan(safe_max, 1e9, color="#d62728", alpha=0.06, label=r"Unsafe region ($|E_t|>R$)")

    # Lines
    t = np.arange(len(s))
    ax.plot(t, s, color="#1f77b4", linewidth=1.8, label=r"Robot true state $S_t$")
    ax.plot(t, dt, color="#ff7f0e", linestyle="--", linewidth=1.8, alpha=0.8, label=r"DT estimate $\hat{S}_t$ (ZOH)")

    # Event markers
    if corrections:
        cx, cy = zip(*corrections)
        ax.scatter(
            cx,
            cy,
            facecolors="none",
            edgecolors="#66bb6a",
            alpha=0.75,
            marker="o",
            s=28,
            linewidths=1.4,
            zorder=5,
            label="Successful correction",
        )
    if misses:
        mx, my = zip(*misses)
        ax.scatter(mx, my, color="#d62728", marker="x", s=90, linewidths=2.2, zorder=6, label="Hash collision (miss)")

    # Thresholds
    ax.axhline(float(target), color="gray", linestyle=":", linewidth=1.2, alpha=0.6)
    ax.axhline(float(target + danger_threshold), color="#9467bd", linestyle=":", linewidth=1.2, alpha=0.8)
    ax.axhline(float(target - danger_threshold), color="#9467bd", linestyle=":", linewidth=1.2, alpha=0.8, label=r"Physical danger boundary $D$")

    # No title: thesis uses the LaTeX caption.
    ax.set_xlabel(r"Time step $t$", fontsize=fontsize)
    ax.set_ylabel("State", fontsize=fontsize)
    ax.tick_params(axis="both", which="both", labelsize=fontsize)
    ax.grid(True, which="both", linestyle="--", alpha=0.25)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ymin = float(np.min(s))
        ymax = float(np.max(s))
        pad = 0.08 * max(1.0, ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)

    # Legend below: avoids shrinking the plot area horizontally.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        borderaxespad=0.0,
        framealpha=0.95,
        fontsize=20,
        columnspacing=1.2,
        handletextpad=0.6,
    )

    # Reserve space for the legend below the axis.
    fig.subplots_adjust(bottom=0.28)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    apply_thesis_style()
    ap = argparse.ArgumentParser(description="Plot random-walk drift sweep results")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("thesis_report/figures/plots"))
    ap.add_argument("--name-suffix", type=str, default="", help="Optional suffix appended to output filenames")

    ap.add_argument("--demo", action="store_true", help="Also generate a representative time-series figure")
    ap.add_argument("--demo-ticks", type=int, default=1200)
    ap.add_argument("--demo-seed", type=int, default=7)
    ap.add_argument("--demo-k", type=float, default=10.0)
    ap.add_argument("--demo-danger", type=float, default=20.0)
    ap.add_argument("--demo-step-dist", choices=["uniform", "gaussian"], default="uniform")
    ap.add_argument("--demo-step-param", type=float, default=1.0)
    ap.add_argument("--demo-resolution", type=float, default=1.0)
    ap.add_argument("--demo-nver-a", type=int, default=16)
    ap.add_argument("--demo-nver-b", type=int, default=8)

    ap.add_argument("--escape-demo", action="store_true", help="Generate the drift & escape time-series figure (state plot)")
    ap.add_argument("--escape-steps", type=int, default=500)
    ap.add_argument("--escape-seed", type=int, default=0, help="Seed to use; if negative, auto-search")
    ap.add_argument("--escape-seed-limit", type=int, default=2000, help="Upper bound for auto seed search")
    ap.add_argument("--escape-target", type=float, default=50.0)
    ap.add_argument("--escape-k", type=float, default=10.0)
    ap.add_argument("--escape-danger", type=float, default=20.0)
    ap.add_argument("--escape-step-dist", choices=["uniform", "gaussian"], default="uniform")
    ap.add_argument("--escape-step-param", type=float, default=2.0)
    ap.add_argument("--escape-p-collision", type=float, default=0.2)
    ap.add_argument("--escape-reset", action="store_true", default=True, help="Reset robot to target on correction (visualization)")
    ap.add_argument("--escape-no-reset", dest="escape_reset", action="store_false")
    ap.add_argument(
        "--escape-clip",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Optional clamp for the plotted state (useful for large sigma); e.g. --escape-clip 2 98",
    )
    ap.add_argument("--escape-ylim", type=float, nargs=2, default=[20.0, 80.0])
    args = ap.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit(f"No rows found in {args.csv}")

    # Keep plots within a single step distribution (the sweep script emits one).
    step = rows[0].step_dist
    step_param = rows[0].step_param

    suffix = str(args.name_suffix).strip()
    suffix_part = ("_" + suffix) if suffix else ""
    prefix = f"random_walk_drift_{step}{suffix_part}"  # e.g., random_walk_drift_uniform

    out_dir = args.out_dir

    plot_metric_vs_k(
        rows,
        metric="breach_streak_p95",
        ylabel="Undetected Interval",
        title=None,
        out_path=out_dir / f"{prefix}_breach_streak_p95.png",
        ylog=False,
    )

    plot_metric_vs_k(
        rows,
        metric="danger_streak_p95",
        ylabel=r"95th percentile danger duration (steps, $|E_t|>D$)",
        title=f"Physical danger duration vs R (D={rows[0].danger_threshold:g}) ({step} step={step_param:g})",
        out_path=out_dir / f"{prefix}_danger_streak_p95.png",
        ylog=False,
    )

    plot_metric_vs_k(
        rows,
        metric="max_error_p95",
        ylabel="Peak Physical Error",
        title=None,
        out_path=out_dir / f"{prefix}_max_error_p95.png",
        ylog=False,
    )

    plot_metric_vs_k(
        rows,
        metric="p_miss_event",
        ylabel=r"$p_{miss}$ per verification event",
        title=f"Collision risk proxy vs R ({step} step={step_param:g})",
        out_path=out_dir / f"{prefix}_pmiss_event.png",
        ylog=True,
    )

    plot_metric_vs_k(
        rows,
        metric="check_rate",
        ylabel="Check rate (fraction of ticks with |error|>R)",
        title=f"How often R is breached (masking effect) ({step} step={step_param:g})",
        out_path=out_dir / f"{prefix}_check_rate.png",
        ylog=False,
    )

    if args.demo:
        plot_timeseries_two_panel(
            ticks=args.demo_ticks,
            seed=args.demo_seed,
            k_radius=args.demo_k,
            danger_threshold=args.demo_danger,
            step_dist=args.demo_step_dist,
            step_param=args.demo_step_param,
            value_resolution=args.demo_resolution,
            nver_a=args.demo_nver_a,
            nver_b=args.demo_nver_b,
            out_path=out_dir / f"{prefix}_timeseries.png",
        )

    if args.escape_demo:
        clip_state = None
        if args.escape_clip is not None:
            clip_state = (float(args.escape_clip[0]), float(args.escape_clip[1]))

        seed = int(args.escape_seed)
        if seed < 0:
            seed = find_seed_for_escape(
                n_steps=int(args.escape_steps),
                seed_start=0,
                seed_limit=int(args.escape_seed_limit),
                target=float(args.escape_target),
                k_radius=float(args.escape_k),
                danger_threshold=float(args.escape_danger),
                step_dist=str(args.escape_step_dist),
                step_param=float(args.escape_step_param),
                p_collision=float(args.escape_p_collision),
                reset_on_correction=bool(args.escape_reset),
                clip_state=clip_state,
            )

        plot_escape_timeseries(
            n_steps=int(args.escape_steps),
            seed=seed,
            target=float(args.escape_target),
            k_radius=float(args.escape_k),
            danger_threshold=float(args.escape_danger),
            step_dist=str(args.escape_step_dist),
            step_param=float(args.escape_step_param),
            p_collision=float(args.escape_p_collision),
            reset_on_correction=bool(args.escape_reset),
            clip_state=clip_state,
            ylim=(float(args.escape_ylim[0]), float(args.escape_ylim[1])) if args.escape_ylim else None,
            out_path=out_dir / f"{prefix}_escape_timeseries.png",
        )

    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
