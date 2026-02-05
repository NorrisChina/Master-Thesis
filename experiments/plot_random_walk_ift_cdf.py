#!/usr/bin/env python3
"""Plot survival-style CDF of inter-event intervals for the random-walk drift model.

Inputs
------
One or more CSV files produced by experiments/random_walk_drift_sweep.py with
--out-ift-csv. Each file contains per-sample inter-check intervals (IFT):
    ift_steps = time steps between consecutive check events (|E_t| > R).

Outputs
-------
A CDF plot: P(IFT <= x) vs x, with separate curves for each R (and fixed n_ver),
or vice versa.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_thesis_style


@dataclass(frozen=True)
class Sample:
    k_radius: float
    n_ver_bits: int
    ift_steps: int
    ift_type: str | None = None


def _as_int(r: Dict[str, str], k: str) -> int:
    return int(r[k])


def _as_float(r: Dict[str, str], k: str) -> float:
    return float(r[k])


def load_samples(paths: List[Path], *, ift_type: str | None) -> List[Sample]:
    samples: List[Sample] = []
    for p in paths:
        with p.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                # Backwards compatible: older files don't have ift_type.
                row_type = r.get("ift_type")
                if ift_type is not None and row_type is not None and str(row_type) != str(ift_type):
                    continue
                samples.append(
                    Sample(
                        k_radius=_as_float(r, "k_radius"),
                        n_ver_bits=_as_int(r, "n_ver_bits"),
                        ift_steps=_as_int(r, "ift_steps"),
                        ift_type=str(row_type) if row_type is not None else None,
                    )
                )
    return samples


def ecdf(xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.asarray(xs, dtype=float)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return np.array([0.0]), np.array([0.0])
    xs = np.sort(xs)
    ys = np.arange(1, xs.size + 1, dtype=float) / float(xs.size)
    return xs, ys


def _apply_ccdf_transform(y_cdf: np.ndarray) -> np.ndarray:
    # Convert CDF to CCDF (survival function). Avoid exact zeros for log-scale plots.
    y = 1.0 - np.asarray(y_cdf, dtype=float)
    return np.maximum(y, 1e-8)


def _quantile_xmax(all_xs: List[np.ndarray], q: float) -> float | None:
    if not all_xs:
        return None
    xs = np.concatenate([np.asarray(x, dtype=float) for x in all_xs if np.asarray(x).size > 0])
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return None
    q = float(q)
    if not (0.0 < q <= 1.0):
        raise ValueError("x-quantile must be in (0, 1]")
    return float(np.quantile(xs, q))


def plot_cdf(
    samples: List[Sample],
    *,
    out_path: Path,
    mode: str,
    xlabel: str,
    xmax: float | None,
    xquantile: float | None,
    ccdf_logy: bool,
    tick_fontsize: float,
    label_fontsize: float,
    legend_fontsize: float,
) -> None:
    # mode:
    # - by_nver: one subplot per n_ver, curves are R
    # - by_k: one subplot per R, curves are n_ver
    ks = sorted({s.k_radius for s in samples})
    nvers = sorted({s.n_ver_bits for s in samples})

    if mode == "by_nver":
        ncols = min(3, max(1, len(nvers)))
        nrows = int(np.ceil(len(nvers) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 4.1 * nrows), squeeze=False)

        cmap = plt.get_cmap("tab10")
        color_for_k = {k: cmap(i % 10) for i, k in enumerate(ks)}

        for idx, nv in enumerate(nvers):
            ax = axes[idx // ncols][idx % ncols]

            missing: List[float] = []
            xs_for_xlim: List[np.ndarray] = []
            for k in ks:
                xs = np.array([s.ift_steps for s in samples if s.n_ver_bits == nv and s.k_radius == k], dtype=float)
                if xs.size == 0:
                    missing.append(k)
                    continue
                x, y = ecdf(xs)
                if ccdf_logy:
                    y = _apply_ccdf_transform(y)
                ax.plot(x, y, linewidth=2.0, color=color_for_k[k], label=f"R={k:g}")
                xs_for_xlim.append(x)

            ax.text(
                0.02,
                0.98,
                rf"$n_{{ver}}={nv}$",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75),
            )
            ax.set_xlabel(str(xlabel), fontsize=float(label_fontsize))
            ax.set_ylabel("CCDF" if ccdf_logy else "CDF")
            ax.yaxis.label.set_size(float(label_fontsize))
            ax.grid(True, which="both", alpha=0.25)
            if ccdf_logy:
                ax.set_yscale("log")
                ax.set_ylim(1e-3, 1.0)
            else:
                ax.set_ylim(0.0, 1.0)

            # X-axis truncation / zoom.
            xmax_eff = None
            if xquantile is not None:
                xmax_eff = _quantile_xmax(xs_for_xlim, float(xquantile))
            if xmax is not None:
                xmax_eff = float(xmax) if xmax_eff is None else min(float(xmax), float(xmax_eff))
            if xmax_eff is not None and np.isfinite(xmax_eff) and xmax_eff > 0:
                ax.set_xlim(left=0.0, right=float(xmax_eff))

            if missing:
                miss_txt = ", ".join(f"{k:g}" for k in missing[:6])
                if len(missing) > 6:
                    miss_txt += ", …"
                ax.text(
                    0.02,
                    0.04,
                    f"No samples for: R={miss_txt}",
                    transform=ax.transAxes,
                    va="bottom",
                    ha="left",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75),
                )

            ax.tick_params(axis="both", which="major", labelsize=float(tick_fontsize))
            ax.legend(
                frameon=True,
                loc="lower right",
                fontsize=float(legend_fontsize),
            )

        # Hide unused axes
        for j in range(len(nvers), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

    elif mode == "by_k":
        ncols = min(3, max(1, len(ks)))
        nrows = int(np.ceil(len(ks) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 4.1 * nrows), squeeze=False)

        cmap = plt.get_cmap("tab10")
        color_for_nv = {nv: cmap(i % 10) for i, nv in enumerate(nvers)}

        for idx, k in enumerate(ks):
            ax = axes[idx // ncols][idx % ncols]

            missing: List[int] = []
            xs_for_xlim: List[np.ndarray] = []
            for nv in nvers:
                xs = np.array([s.ift_steps for s in samples if s.n_ver_bits == nv and s.k_radius == k], dtype=float)
                if xs.size == 0:
                    missing.append(nv)
                    continue
                x, y = ecdf(xs)
                if ccdf_logy:
                    y = _apply_ccdf_transform(y)
                ax.plot(x, y, linewidth=2.0, color=color_for_nv[nv], label=rf"$n_{{ver}}={nv}$")
                xs_for_xlim.append(x)

            ax.text(
                0.02,
                0.98,
                rf"$R={k:g}$",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75),
            )
            ax.set_xlabel(str(xlabel), fontsize=float(label_fontsize))
            ax.set_ylabel("CCDF" if ccdf_logy else "CDF")
            ax.yaxis.label.set_size(float(label_fontsize))
            ax.grid(True, which="both", alpha=0.25)
            if ccdf_logy:
                ax.set_yscale("log")
                ax.set_ylim(1e-3, 1.0)
            else:
                ax.set_ylim(0.0, 1.0)

            xmax_eff = None
            if xquantile is not None:
                xmax_eff = _quantile_xmax(xs_for_xlim, float(xquantile))
            if xmax is not None:
                xmax_eff = float(xmax) if xmax_eff is None else min(float(xmax), float(xmax_eff))
            if xmax_eff is not None and np.isfinite(xmax_eff) and xmax_eff > 0:
                ax.set_xlim(left=0.0, right=float(xmax_eff))

            if missing:
                miss_txt = ", ".join(str(nv) for nv in missing[:6])
                if len(missing) > 6:
                    miss_txt += ", …"
                ax.text(
                    0.02,
                    0.04,
                    f"No samples for: nver={miss_txt}",
                    transform=ax.transAxes,
                    va="bottom",
                    ha="left",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75),
                )

            ax.tick_params(axis="both", which="major", labelsize=float(tick_fontsize))
            ax.legend(frameon=True, loc="lower right", fontsize=float(legend_fontsize))

        for j in range(len(ks), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

    else:
        raise ValueError(mode)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Reserve space so x-labels/ticks aren't clipped, but keep a consistent canvas
    # across plots to make side-by-side comparisons fair.
    fig.tight_layout(pad=0.9)
    fig.subplots_adjust(bottom=0.20)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    apply_thesis_style()
    ap = argparse.ArgumentParser(description="Plot CDF of inter-transmission times")
    ap.add_argument("--ift-csv", type=Path, nargs="+", required=True, help="One or more IFT CSV files")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["by_nver", "by_k"], default="by_nver")
    ap.add_argument("--nver", type=int, default=None, help="Optional: filter to a single n_ver (bits)")
    ap.add_argument(
        "--ift-type",
        choices=["check", "correction"],
        default=None,
        help="If the CSV contains an ift_type column, filter to this interval type.",
    )
    ap.add_argument("--xmax", type=float, default=None, help="Optional: truncate/zoom x-axis to [0, xmax].")
    ap.add_argument(
        "--xquantile",
        type=float,
        default=None,
        help="Optional: truncate x-axis to the given quantile (e.g. 0.99) of the plotted samples.",
    )
    ap.add_argument(
        "--ccdf-logy",
        action="store_true",
        help="Plot CCDF (=1-CDF) and use log-scale on y-axis to emphasize tail risk.",
    )
    ap.add_argument("--tick-fontsize", type=float, default=12.0, help="Font size for tick labels.")
    ap.add_argument("--label-fontsize", type=float, default=12.0, help="Font size for axis labels.")
    ap.add_argument("--legend-fontsize", type=float, default=10.0, help="Font size for legend text.")
    args = ap.parse_args()

    samples = load_samples([Path(p) for p in args.ift_csv], ift_type=str(args.ift_type) if args.ift_type else None)
    if args.nver is not None:
        samples = [s for s in samples if int(s.n_ver_bits) == int(args.nver)]
    if not samples:
        raise SystemExit("No samples loaded")

    if args.ift_type == "correction":
        xlabel = "Successive recovery interval (steps)"
    elif args.ift_type == "check":
        xlabel = "Inter-check interval (steps)"
    else:
        # Backwards-compatible fallback for older CSVs without type.
        xlabel = "Inter-event interval (steps)"

    plot_cdf(
        samples,
        out_path=args.out,
        mode=str(args.mode),
        xlabel=xlabel,
        xmax=float(args.xmax) if args.xmax is not None else None,
        xquantile=float(args.xquantile) if args.xquantile is not None else None,
        ccdf_logy=bool(args.ccdf_logy),
        tick_fontsize=float(args.tick_fontsize),
        label_fontsize=float(args.label_fontsize),
        legend_fontsize=float(args.legend_fontsize),
    )
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
