#!/usr/bin/env python3
"""Plot K-ID sweep results.

Reads the CSV produced by experiments/kid_parameter_sweep.py and generates plots
into experiments/results/.

Typical workflow:
  python experiments/kid_parameter_sweep.py --ticks 200000 --out-csv experiments/results/kid_sweep.csv
  python experiments/plot_kid_parameter_sweep.py --csv experiments/results/kid_sweep.csv

Plots:
- miss_rate_accept_given_unsafe vs n_ver_bits (per K)
- fallback_rate vs n_ver_bits (per K)
- latency vs n_ver_bits (per K)
- speedup vs n_ver_bits (per K)

Optionally overlays theo_risk on the miss-rate chart.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt


@dataclass
class Row:
    system: str
    n_ver_bits: int
    k_radius: int
    miss_rate: float
    theo_risk: float
    fallback_rate: float
    lat_id_us: float
    lat_traditional_us: float
    speedup: float
    ticks: int
    unsafe_count: int
    miss_count: int

    @property
    def unsafe_rate(self) -> float:
        return (self.unsafe_count / self.ticks) if self.ticks else 0.0

    @property
    def overall_risk(self) -> float:
        """Empirical overall risk P(unsafe ∩ accept) per tick.

        Since miss_rate = miss_count / unsafe_count, we can write:
        P(unsafe ∩ accept) = P(unsafe) * P(accept|unsafe) = miss_count / ticks.
        """
        return (self.miss_count / self.ticks) if self.ticks else 0.0


def _as_int(row: Dict[str, str], key: str) -> int:
    return int(row[key])


def _as_float(row: Dict[str, str], key: str) -> float:
    return float(row[key])


def load_rows(csv_path: Path) -> List[Row]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: List[Row] = []
        for r in reader:
            system = r.get("system", "sha256_trunc")
            rows.append(
                Row(
                    system=str(system),
                    n_ver_bits=_as_int(r, "n_ver_bits"),
                    k_radius=_as_int(r, "k_radius"),
                    miss_rate=_as_float(r, "miss_rate_accept_given_unsafe"),
                    theo_risk=_as_float(r, "theo_risk"),
                    fallback_rate=_as_float(r, "fallback_rate"),
                    lat_id_us=_as_float(r, "lat_id_us"),
                    lat_traditional_us=_as_float(r, "lat_traditional_us"),
                    speedup=_as_float(r, "speedup"),
                    ticks=_as_int(r, "ticks"),
                    unsafe_count=_as_int(r, "unsafe_count"),
                    miss_count=_as_int(r, "miss_count"),
                )
            )
    return rows


def group_by_k(rows: List[Row]) -> Dict[int, List[Row]]:
    grouped: Dict[int, List[Row]] = {}
    for r in rows:
        grouped.setdefault(r.k_radius, []).append(r)
    for k, rs in grouped.items():
        rs.sort(key=lambda x: x.n_ver_bits)
    return grouped


def filter_rows(rows: List[Row], *, system: str) -> List[Row]:
    return [r for r in rows if r.system == system]


def group_by_k_for_system(rows: List[Row], *, system: str) -> Dict[int, List[Row]]:
    return group_by_k(filter_rows(rows, system=system))


def plot_metric_by_system_subplots(
    rows: List[Row],
    *,
    metric_name: str,
    ylabel: str,
    title: str,
    out_path: Path,
    overlay_theo: bool = False,
    ylog: bool = False,
) -> None:
    systems = sorted({r.system for r in rows})
    if len(systems) <= 1:
        grouped = group_by_k(rows)
        plot_metric(
            grouped,
            metric_name=metric_name,
            ylabel=ylabel,
            title=title,
            out_path=out_path,
            overlay_theo=overlay_theo,
            ylog=ylog,
        )
        return

    fig, axes = plt.subplots(1, len(systems), figsize=(7.2 * len(systems), 4.2), sharey=True)
    if len(systems) == 2:
        axes = list(axes)

    for ax, system in zip(axes, systems):
        grouped = group_by_k_for_system(rows, system=system)
        for k, rs in sorted(grouped.items()):
            xs = [r.n_ver_bits for r in rs]
            ys = [getattr(r, metric_name) for r in rs]
            ax.plot(xs, ys, marker="o", linewidth=2.0, label=f"K={k}")
            if overlay_theo and metric_name == "miss_rate":
                theo = [r.theo_risk for r in rs]
                ax.plot(xs, theo, linestyle="--", linewidth=1.6, label=f"TheoRisk (K={k})")

        ax.set_xlabel("$n_{ver}$ (bits)")
        ax.set_title(system)
        ax.grid(True, which="both", alpha=0.25)
        if ylog:
            ax.set_yscale("log")

        if metric_name == "speedup":
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.9)
            y0, y1 = ax.get_ylim()
            lo = min(y0, y1)
            hi = max(y0, y1)
            if hi > 1.0:
                ax.axhspan(1.0, hi, facecolor="#2ca02c", alpha=0.06, zorder=0)
            if lo < 1.0:
                ax.axhspan(lo, 1.0, facecolor="#d62728", alpha=0.06, zorder=0)

    axes[0].set_ylabel(ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    if metric_name == "speedup":
        baseline = plt.Line2D(
            [0],
            [0],
            color="gray",
            linestyle="--",
            linewidth=1.2,
            label="Traditional Baseline (1×)",
        )
        handles = handles + [baseline]
        labels = labels + ["Traditional Baseline (1×)"]
    axes[0].legend(handles=handles, labels=labels, frameon=True)
    fig.suptitle(title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pareto_by_system_subplots(
    rows: List[Row],
    *,
    out_path: Path,
    eps: float,
    xlog: bool,
    use_overall_risk: bool,
) -> None:
    systems = sorted({r.system for r in rows})
    if len(systems) <= 1:
        if use_overall_risk:
            plot_pareto_overall_risk(rows, out_path=out_path, eps=eps, xlog=xlog)
        else:
            plot_pareto(rows, out_path=out_path, eps=eps, xlog=xlog)
        return

    fig, axes = plt.subplots(1, len(systems), figsize=(7.2 * len(systems), 4.6), sharey=True)
    if len(systems) == 2:
        axes = list(axes)

    for ax, system in zip(axes, systems):
        rs = filter_rows(rows, system=system)
        ks = sorted({r.k_radius for r in rs})
        nvers = sorted({r.n_ver_bits for r in rs})
        cmap = plt.get_cmap("tab10")
        color_for_k = {k: cmap(i % 10) for i, k in enumerate(ks)}
        marker_for_nver = {nv: m for nv, m in zip(nvers, ["o", "s", "^", "D", "v", "P", "X"])}

        if use_overall_risk:
            pts = [(max(r.overall_risk, eps), r.speedup) for r in rs]
            xlabel = r"Overall risk $P(\mathrm{unsafe}\cap\mathrm{accept})$"
            title = f"{system}: overall risk"
        else:
            pts = [(max(r.miss_rate, eps), r.speedup) for r in rs]
            xlabel = r"Empirical miss rate $P(\mathrm{accept}\mid\mathrm{unsafe})$"
            title = f"{system}: conditional miss"

        nd = pareto_nondominated(pts)
        frontier = sorted([(x, y) for (x, y), is_nd in zip(pts, nd) if is_nd], key=lambda t: t[0])

        for r, (x, y), is_nd in zip(rs, pts, nd):
            ax.scatter(
                x,
                y,
                s=55 if is_nd else 35,
                marker=marker_for_nver.get(r.n_ver_bits, "o"),
                c=[color_for_k[r.k_radius]],
                edgecolors="black" if is_nd else "none",
                linewidths=0.8 if is_nd else 0.0,
                alpha=0.95 if is_nd else 0.25,
                zorder=3 if is_nd else 2,
            )

        if len(frontier) >= 2:
            fx = [p[0] for p in frontier]
            fy = [p[1] for p in frontier]
            ax.plot(fx, fy, color="black", linewidth=1.4, alpha=0.75, zorder=1)

        # Baseline + background regions for speedup.
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.9)
        y0, y1 = ax.get_ylim()
        lo = min(y0, y1)
        hi = max(y0, y1)
        if hi > 1.0:
            ax.axhspan(1.0, hi, facecolor="#2ca02c", alpha=0.06, zorder=0)
        if lo < 1.0:
            ax.axhspan(lo, 1.0, facecolor="#d62728", alpha=0.06, zorder=0)

        # Legends: color by K, marker by n_ver.
        k_handles = [
            plt.Line2D([0], [0], marker="o", linestyle="", color=color_for_k[k], label=f"K={k}")
            for k in ks
        ]
        nv_handles = [
            plt.Line2D(
                [0],
                [0],
                marker=marker_for_nver[nv],
                linestyle="",
                color="gray",
                label=f"n_ver={nv}",
            )
            for nv in nvers
        ]
        leg1 = ax.legend(handles=nv_handles, title="Verifier", loc="upper left", frameon=True)
        ax.add_artist(leg1)
        ax.legend(handles=k_handles, title="Tolerance", loc="upper right", frameon=True)

        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        if xlog:
            ax.set_xscale("log")

    axes[0].set_ylabel("Speedup over traditional (×)")
    fig.suptitle("K-ID trade-off comparison")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_metric(
    grouped: Dict[int, List[Row]],
    metric_name: str,
    ylabel: str,
    title: str,
    out_path: Path,
    overlay_theo: bool = False,
    ylog: bool = False,
) -> None:
    plt.figure(figsize=(7.2, 4.2))

    for k, rs in sorted(grouped.items()):
        xs = [r.n_ver_bits for r in rs]
        ys = [getattr(r, metric_name) for r in rs]
        plt.plot(xs, ys, marker="o", linewidth=2.0, label=f"K={k}")

        if overlay_theo and metric_name == "miss_rate":
            theo = [r.theo_risk for r in rs]
            plt.plot(xs, theo, linestyle="--", linewidth=1.6, label=f"TheoRisk (K={k})")

    plt.xlabel("$n_{ver}$ (bits)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", alpha=0.25)

    if metric_name == "speedup":
        ax = plt.gca()
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.9, label="Traditional Baseline (1×)")
        y0, y1 = ax.get_ylim()
        lo = min(y0, y1)
        hi = max(y0, y1)
        if hi > 1.0:
            ax.axhspan(1.0, hi, facecolor="#2ca02c", alpha=0.06, zorder=0)
        if lo < 1.0:
            ax.axhspan(lo, 1.0, facecolor="#d62728", alpha=0.06, zorder=0)
    plt.legend(frameon=True)

    if ylog:
        plt.yscale("log")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def pareto_nondominated(points: List[Tuple[float, float]]) -> List[bool]:
    """Return a boolean mask indicating which points are Pareto-nondominated.

    We want: minimize miss_rate (x), maximize speedup (y).
    A point i is dominated if there exists j with x_j <= x_i and y_j >= y_i and at least
    one strict.
    """
    mask = [True] * len(points)
    for i, (xi, yi) in enumerate(points):
        if not mask[i]:
            continue
        for j, (xj, yj) in enumerate(points):
            if i == j:
                continue
            if (xj <= xi and yj >= yi) and (xj < xi or yj > yi):
                mask[i] = False
                break
    return mask


def plot_pareto_overall_risk(
    rows: List[Row],
    out_path: Path,
    eps: float,
    xlog: bool,
) -> None:
    """Pareto plot with x = overall risk P(unsafe ∩ accept)."""
    ks = sorted({r.k_radius for r in rows})
    nvers = sorted({r.n_ver_bits for r in rows})
    cmap = plt.get_cmap("tab10")
    color_for_k = {k: cmap(i % 10) for i, k in enumerate(ks)}
    marker_for_nver = {nv: m for nv, m in zip(nvers, ["o", "s", "^", "D", "v", "P", "X"]) }

    pts = [(max(r.overall_risk, eps), r.speedup) for r in rows]
    nd = pareto_nondominated(pts)
    frontier = sorted([(x, y) for (x, y), is_nd in zip(pts, nd) if is_nd], key=lambda t: t[0])

    plt.figure(figsize=(7.2, 4.6))
    for r, (x, y), is_nd in zip(rows, pts, nd):
        plt.scatter(
            x,
            y,
            s=55 if is_nd else 35,
            marker=marker_for_nver.get(r.n_ver_bits, "o"),
            c=[color_for_k[r.k_radius]],
            edgecolors="black" if is_nd else "none",
            linewidths=0.8 if is_nd else 0.0,
            alpha=0.95 if is_nd else 0.25,
            zorder=3 if is_nd else 2,
        )

    if len(frontier) >= 2:
        fx = [p[0] for p in frontier]
        fy = [p[1] for p in frontier]
        plt.plot(fx, fy, color="black", linewidth=1.4, alpha=0.75, zorder=1)

    k_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color_for_k[k], label=f"K={k}")
        for k in ks
    ]
    nv_handles = [
        plt.Line2D([0], [0], marker=marker_for_nver[nv], linestyle="", color="gray", label=f"n_ver={nv}")
        for nv in nvers
    ]

    plt.xlabel(r"Overall risk $P(\mathrm{unsafe}\cap\mathrm{accept})$")
    plt.ylabel("Speedup over traditional (×)")
    plt.title("K-ID trade-off: speedup vs overall risk")
    plt.grid(True, which="both", alpha=0.25)

    ax = plt.gca()
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.9)
    y0, y1 = ax.get_ylim()
    lo = min(y0, y1)
    hi = max(y0, y1)
    if hi > 1.0:
        ax.axhspan(1.0, hi, facecolor="#2ca02c", alpha=0.06, zorder=0)
    if lo < 1.0:
        ax.axhspan(lo, 1.0, facecolor="#d62728", alpha=0.06, zorder=0)
    if xlog:
        plt.xscale("log")

    leg1 = plt.legend(handles=nv_handles, title="Verifier", loc="upper left", frameon=True)
    plt.gca().add_artist(leg1)
    plt.legend(handles=k_handles, title="Tolerance", loc="upper right", frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_pareto(
    rows: List[Row],
    out_path: Path,
    eps: float,
    xlog: bool,
) -> None:
    # Color by K, marker by n_ver.
    ks = sorted({r.k_radius for r in rows})
    nvers = sorted({r.n_ver_bits for r in rows})
    cmap = plt.get_cmap("tab10")
    color_for_k = {k: cmap(i % 10) for i, k in enumerate(ks)}
    marker_for_nver = {nv: m for nv, m in zip(nvers, ["o", "s", "^", "D", "v", "P", "X"]) }

    # Prepare points and nondominated mask.
    pts = [(max(r.miss_rate, eps), r.speedup) for r in rows]
    nd = pareto_nondominated(pts)

    # Compute Pareto frontier polyline (sorted by miss rate).
    frontier = sorted([(x, y) for (x, y), is_nd in zip(pts, nd) if is_nd], key=lambda t: t[0])

    plt.figure(figsize=(7.2, 4.6))
    for r, (x, y), is_nd in zip(rows, pts, nd):
        plt.scatter(
            x,
            y,
            s=55 if is_nd else 35,
            marker=marker_for_nver.get(r.n_ver_bits, "o"),
            c=[color_for_k[r.k_radius]],
            edgecolors="black" if is_nd else "none",
            linewidths=0.8 if is_nd else 0.0,
            alpha=0.95 if is_nd else 0.25,
            zorder=3 if is_nd else 2,
        )

    # Draw a Pareto frontier line (best trade-off curve).
    if len(frontier) >= 2:
        fx = [p[0] for p in frontier]
        fy = [p[1] for p in frontier]
        plt.plot(fx, fy, color="black", linewidth=1.4, alpha=0.75, zorder=1)

    # Build a clean legend: one for K colors, one for n_ver markers.
    k_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color_for_k[k], label=f"K={k}")
        for k in ks
    ]
    nv_handles = [
        plt.Line2D([0], [0], marker=marker_for_nver[nv], linestyle="", color="gray", label=f"n_ver={nv}")
        for nv in nvers
    ]

    plt.xlabel(r"Empirical miss rate $P(\mathrm{accept}\mid\mathrm{unsafe})$")
    plt.ylabel("Speedup over traditional (×)")
    plt.title("K-ID trade-off: speedup vs safety miss rate")
    plt.grid(True, which="both", alpha=0.25)

    ax = plt.gca()
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.9)
    y0, y1 = ax.get_ylim()
    lo = min(y0, y1)
    hi = max(y0, y1)
    if hi > 1.0:
        ax.axhspan(1.0, hi, facecolor="#2ca02c", alpha=0.06, zorder=0)
    if lo < 1.0:
        ax.axhspan(lo, 1.0, facecolor="#d62728", alpha=0.06, zorder=0)
    if xlog:
        plt.xscale("log")

    # Place legends without covering points.
    leg1 = plt.legend(handles=nv_handles, title="Verifier", loc="upper left", frameon=True)
    plt.gca().add_artist(leg1)
    plt.legend(handles=k_handles, title="Tolerance", loc="upper right", frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot results from K-ID parameter sweep CSV")
    p.add_argument("--csv", type=Path, required=True, help="Input CSV produced by kid_parameter_sweep.py")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: <repo>/experiments/results)",
    )
    p.add_argument("--prefix", type=str, default="kid_sweep", help="Filename prefix (default: kid_sweep)")
    p.add_argument("--overlay-theo", action="store_true", help="Overlay theo_risk on miss-rate plot")
    p.add_argument("--log-miss", action="store_true", help="Use log scale for miss-rate plot")
    p.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Epsilon floor for miss-rate values when using log scale or Pareto plot (default: 1e-12)",
    )
    p.add_argument(
        "--pareto-log-x",
        action="store_true",
        help="Use log x-axis for Pareto plot (miss-rate) (default: false)",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    # Resolve paths relative to the repository root so running from other cwd's
    # (e.g. thesis_report/) doesn't scatter outputs.
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = args.csv
    if not csv_path.is_absolute():
        csv_path = repo_root / csv_path

    out_dir: Path
    if args.out_dir is None:
        out_dir = repo_root / "experiments" / "results"
    else:
        out_dir = args.out_dir
        if not out_dir.is_absolute():
            out_dir = repo_root / out_dir

    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit(f"No rows found in CSV: {csv_path}")

    prefix: str = args.prefix

    plot_metric_by_system_subplots(
        rows,
        metric_name="miss_rate",
        ylabel=r"Empirical miss rate $P(\mathrm{accept}\mid\mathrm{unsafe})$",
        title="K-ID safety miss rate vs verifier length",
        out_path=out_dir / f"{prefix}_miss_rate.png",
        overlay_theo=bool(args.overlay_theo),
        ylog=bool(args.log_miss),
    )

    plot_pareto_by_system_subplots(
        rows,
        out_path=out_dir / f"{prefix}_pareto.png",
        eps=float(args.eps),
        xlog=bool(args.pareto_log_x),
        use_overall_risk=False,
    )

    plot_pareto_by_system_subplots(
        rows,
        out_path=out_dir / f"{prefix}_pareto_overall_risk.png",
        eps=float(args.eps),
        xlog=bool(args.pareto_log_x),
        use_overall_risk=True,
    )

    plot_metric_by_system_subplots(
        rows,
        metric_name="fallback_rate",
        ylabel=r"Fallback rate $P(\mathrm{fallback})$",
        title="K-ID fallback rate vs verifier length",
        out_path=out_dir / f"{prefix}_fallback_rate.png",
    )

    plot_metric_by_system_subplots(
        rows,
        metric_name="lat_id_us",
        ylabel="Average latency (µs)",
        title="K-ID average latency vs verifier length",
        out_path=out_dir / f"{prefix}_latency.png",
    )

    plot_metric_by_system_subplots(
        rows,
        metric_name="speedup",
        ylabel="Speedup over traditional (×)",
        title="K-ID speedup vs verifier length",
        out_path=out_dir / f"{prefix}_speedup.png",
    )

    print(f"Saved plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
