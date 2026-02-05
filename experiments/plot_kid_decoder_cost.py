#!/usr/bin/env python3
"""Plot the hidden decoder cost of K-ID (DT-side work).

We reuse the CSV from experiments/kid_parameter_sweep.py.

Idea:
- The DT must consider all acceptable quantized states within a physical tolerance radius R.
- This acceptance-set size is exported as `range_size` (number of quantized states), i.e., K = |A_R|.
- A straightforward implementation precomputes the verifier set V_R by hashing
    each acceptable state -> ~K hash computations per DT update.

This script plots:
- range_size (K = |A_R|) vs R
- unique_valid_hashes (|V_R|) vs R

Both reflect DT-side compute/storage pressure that grows with R (via K = |A_R|) and finer
quantization (smaller value_resolution).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Row:
    system: str
    error_dist: str
    n_ver_bits: int
    k_radius: int
    value_resolution: float
    range_size: int
    unique_valid_hashes: int


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
                    error_dist=str(r.get("error_dist", "normal")),
                    n_ver_bits=_as_int(r, "n_ver_bits"),
                    k_radius=_as_int(r, "k_radius"),
                    value_resolution=_as_float(r, "value_resolution"),
                    range_size=_as_int(r, "range_size"),
                    unique_valid_hashes=_as_int(r, "unique_valid_hashes"),
                )
            )
    return out


def plot_decoder_cost(
    rows: List[Row],
    *,
    out_path: Path,
    system: str,
) -> None:
    rs = [r for r in rows if r.system == system]
    if not rs:
        raise SystemExit(f"No rows for system={system}")

    # For this figure, range_size and unique_valid_hashes are effectively
    # independent of n_ver_bits; pick one representative n_ver for clarity.
    nver = min({r.n_ver_bits for r in rs})
    rs = [r for r in rs if r.n_ver_bits == nver]

    rs.sort(key=lambda r: r.k_radius)

    ks = [r.k_radius for r in rs]
    accept_sizes = [r.range_size for r in rs]
    unique_sizes = [r.unique_valid_hashes for r in rs]

    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))

    ln1 = ax1.plot(ks, accept_sizes, marker="o", linewidth=2.2, label=r"$K=|\mathcal{A}_R|$ (quantized states)")
    ax1.set_xlabel("Tolerance radius $R$")
    ax1.set_ylabel(r"Acceptance-set size $K=|\mathcal{A}_R|$")
    ax1.grid(True, which="both", alpha=0.25)

    ax2 = ax1.twinx()
    ln2 = ax2.plot(ks, unique_sizes, marker="s", linewidth=2.0, color="#ff7f0e", label=r"$|\mathcal{V}_R|$ (distinct verifiers)")
    ax2.set_ylabel(r"Distinct verifier count $|\mathcal{V}_R|$")

    title = f"Decoder-side cost scaling with R (system={system}, Δ={rs[0].value_resolution:g}, n_ver={nver})"
    ax1.set_title(title)

    # One shared legend.
    lines = ln1 + ln2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, frameon=True, loc="upper left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot DT-side decoder cost for K-ID")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--system", type=str, default="sha256_trunc")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    plot_decoder_cost(rows, out_path=args.out, system=str(args.system))
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
