#!/usr/bin/env python3
"""Generate heatmaps for the K-ID sweep.

Purpose (thesis-facing): complement the 1D line plots (Fig. 5.3 style) with a
compact 2D view over the full grid (K × n_ver).

Inputs: CSV from experiments/kid_parameter_sweep.py
Outputs: PNG heatmaps for speedup and overall risk, optionally per-system.

We plot overall risk = miss_count / ticks (same as used in the Pareto plot).
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

from plot_style import apply_thesis_style


@dataclass(frozen=True)
class Row:
    system: str
    n_ver_bits: int
    k_radius: int
    value_resolution: float | None
    r_radius: float | None
    speedup: float
    ticks: int
    miss_count: int

    @property
    def overall_risk(self) -> float:
        return (self.miss_count / self.ticks) if self.ticks else 0.0


def _as_int(r: Dict[str, str], k: str) -> int:
    return int(r[k])


def _as_float(r: Dict[str, str], k: str) -> float:
    return float(r[k])


def load_rows(path: Path) -> List[Row]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out: List[Row] = []
        for r in reader:
            out.append(
                Row(
                    system=str(r.get("system", "sha256_trunc")),
                    n_ver_bits=_as_int(r, "n_ver_bits"),
                    k_radius=_as_int(r, "k_radius"),
                    value_resolution=(float(r["value_resolution"]) if r.get("value_resolution") else None),
                    r_radius=(float(r["r_radius"]) if r.get("r_radius") else None),
                    speedup=_as_float(r, "speedup"),
                    ticks=_as_int(r, "ticks"),
                    miss_count=_as_int(r, "miss_count"),
                )
            )
    return out


def _format_r(v: float) -> str:
    # Compact formatting for axis ticks (e.g., 2, 5, 10, 20, 30).
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:g}"


def _k_to_r_mapping(rows: List[Row]) -> dict[int, float]:
    mapping: dict[int, float] = {}
    for r in rows:
        if r.r_radius is not None and math.isfinite(r.r_radius):
            mapping.setdefault(int(r.k_radius), float(r.r_radius))

    if mapping:
        return mapping

    # Fallback: derive from value_resolution if available.
    resolutions = sorted({float(r.value_resolution) for r in rows if r.value_resolution is not None})
    if len(resolutions) == 1:
        res = float(resolutions[0])
        return {int(r.k_radius): int(r.k_radius) * res for r in rows}

    # Last resort: return identity in K units.
    return {int(r.k_radius): float(r.k_radius) for r in rows}


def _matrix_for(rows: List[Row], *, ks: List[int], nvers: List[int], metric: str) -> np.ndarray:
    idx_k = {k: i for i, k in enumerate(ks)}
    idx_nv = {nv: j for j, nv in enumerate(nvers)}

    mat = np.full((len(ks), len(nvers)), np.nan, dtype=float)

    # If duplicates exist (shouldn't), average them.
    buckets: Dict[Tuple[int, int], List[float]] = {}
    for r in rows:
        key = (r.k_radius, r.n_ver_bits)
        if metric == "speedup":
            v = float(r.speedup)
        elif metric == "overall_risk":
            v = float(r.overall_risk)
        else:
            raise ValueError(metric)
        buckets.setdefault(key, []).append(v)

    for (k, nv), vs in buckets.items():
        mat[idx_k[k], idx_nv[nv]] = float(np.mean(vs))

    return mat


def _plot_heatmap(
    ax,
    mat: np.ndarray,
    *,
    ks: List[int],
    y_ticklabels: List[str],
    nvers: List[int],
    title: str,
    cmap: str,
    log: bool,
) -> None:
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#d9d9d9")

    if log:
        # Treat exact zeros as a distinct "safe" state (shown via the colormap's bad color).
        # This avoids LogNorm issues and makes the zero-risk region visually obvious.
        masked = np.ma.masked_where(~np.isfinite(mat) | (mat <= 0), mat)
        finite_pos = np.asarray(mat[np.isfinite(mat) & (mat > 0)], dtype=float)

        if finite_pos.size == 0:
            norm = Normalize(vmin=0.0, vmax=1.0)
        else:
            vmin = float(np.min(finite_pos))
            vmax = float(np.max(finite_pos))
            if vmax <= vmin:
                vmax = vmin * 10.0
            norm = LogNorm(vmin=vmin, vmax=vmax)
        im = ax.imshow(masked, aspect="auto", origin="lower", cmap=cmap_obj, norm=norm)
    else:
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap=cmap_obj)

    ax.set_xticks(np.arange(len(nvers)))
    ax.set_xticklabels([str(x) for x in nvers])
    ax.set_yticks(np.arange(len(ks)))
    ax.set_yticklabels(list(y_ticklabels))

    ax.set_xlabel(r"$n_{ver}$ (bits)")
    ax.set_ylabel("Tolerance $R$")
    ax.set_title(title)

    # Light cell grid for readability.
    ax.set_xticks(np.arange(-0.5, len(nvers), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ks), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6, alpha=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def _luminance(rgba: Tuple[float, float, float, float]) -> float:
    r, g, b, _a = rgba
    # Perceived luminance (sRGB-ish).
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _annotate_cells(
    ax,
    mat: np.ndarray,
    *,
    cmap,
    norm,
    fmt,
    fontsize: int = 12,
    fontweight: str = "bold",
) -> None:
    rows, cols = mat.shape
    for i in range(rows):
        for j in range(cols):
            v = float(mat[i, j]) if np.isfinite(mat[i, j]) else float("nan")

            if not np.isfinite(v):
                continue

            rgba = cmap(norm(v)) if norm is not None else cmap(v)
            text_color = "white" if _luminance(rgba) < 0.45 else "black"
            ax.text(
                j,
                i,
                fmt(v),
                ha="center",
                va="center",
                color=text_color,
                fontsize=fontsize,
                fontweight=fontweight,
                fontfamily="sans-serif" if fontsize == 14 and fontweight is None else None,
            )


def plot_two_metrics(
    rows: List[Row],
    *,
    out_path: Path,
    system: str,
    title_prefix: str,
):
    rs = [r for r in rows if r.system == system]
    if not rs:
        raise SystemExit(f"No rows for system={system}")

    ks = sorted({r.k_radius for r in rs})
    nvers = sorted({r.n_ver_bits for r in rs})

    k_to_r = _k_to_r_mapping(rs)
    y_labels = [_format_r(float(k_to_r.get(int(k), float(k)))) for k in ks]

    m_speed = _matrix_for(rs, ks=ks, nvers=nvers, metric="speedup")
    m_risk = _matrix_for(rs, ks=ks, nvers=nvers, metric="overall_risk")

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.0), constrained_layout=True)

    # More semantic / vivid colormaps:
    # - speedup: red (bad) -> yellow -> green (good)
    # - risk: yellow (low) -> orange -> red (high)
    cmap_speed = "RdYlGn"
    cmap_risk = "YlOrRd"

    # Left: speedup
    im1 = _plot_heatmap(
        axes[0],
        m_speed,
        ks=ks,
        y_ticklabels=y_labels,
        nvers=nvers,
        title="speedup",
        cmap=cmap_speed,
        log=False,
    )
    c1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    c1.set_label("Speedup (×)")
    _annotate_cells(
        axes[0],
        m_speed,
        cmap=im1.cmap,
        norm=im1.norm,
        fmt=lambda v: f"{v:.1f}×",
        fontsize=14,
        fontweight=None,
    )

    # Right: overall risk
    im2 = _plot_heatmap(
        axes[1],
        m_risk,
        ks=ks,
        y_ticklabels=y_labels,
        nvers=nvers,
        title="overall risk",
        cmap=cmap_risk,
        log=True,
    )
    c2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    c2.set_label("Overall risk")

    def _fmt_risk(v: float) -> str:
        if v == 0.0:
            return "0"
        return f"{v:.0e}"
    _annotate_cells(
        axes[1],
        np.where(np.isfinite(m_risk) & (m_risk > 0), m_risk, np.nan),
        cmap=im2.cmap,
        norm=im2.norm,
        fmt=_fmt_risk,
        fontsize=14,
        fontweight=None,
    )
    zero_mask = np.isfinite(m_risk) & (m_risk == 0)
    for i, j in zip(*np.where(zero_mask)):
        axes[1].text(j, i, "0", ha="center", va="center", color="black", fontsize=14, fontweight=None, fontfamily="sans-serif")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    apply_thesis_style()
    ap = argparse.ArgumentParser(description="Plot K-ID heatmaps (K × n_ver)")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="Output PNG")
    ap.add_argument("--system", type=str, default="sha256_trunc")
    ap.add_argument("--title", type=str, default="K-ID sweep")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    plot_two_metrics(rows, out_path=args.out, system=str(args.system), title_prefix=str(args.title))
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
