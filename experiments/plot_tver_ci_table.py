#!/usr/bin/env python3
"""Plot t_ver CI table output (error bars).

Reads the CSV produced by experiments/measure_tver_ci_table.py and generates a
log-log plot with 95% CI error bars.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_style import apply_thesis_style


def main() -> None:
    apply_thesis_style()
    p = argparse.ArgumentParser(description="Plot t_ver CI table (error bars)")
    p.add_argument("--csv", type=str, required=True, help="Input CSV from measure_tver_ci_table.py")
    p.add_argument("--out", type=str, required=True, help="Output PNG path")
    p.add_argument("--title", type=str, default="t_ver with 95% CI")
    args = p.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path).sort_values("n_data_bytes")

    x = df["n_data_bytes"].to_numpy()
    sha = df["sha_mean_us"].to_numpy()
    sha_ci = df["sha_ci95_half_us"].to_numpy()
    rs = df["rsid_mean_us"].to_numpy()
    rs_ci = df["rsid_ci95_half_us"].to_numpy()

    plt.figure(figsize=(7.2, 4.2))
    plt.errorbar(x, sha, yerr=sha_ci, marker="o", capsize=3, label="SHA-256 ID (t_ver)")
    plt.errorbar(x, rs, yerr=rs_ci, marker="s", capsize=3, label="RS-ID (t_ver)")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Payload size (bytes)")
    plt.ylabel("t_ver (µs, log scale)")
    plt.title(args.title)
    plt.grid(True, which="both", linestyle=":", linewidth=0.8)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_path, dpi=200)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
