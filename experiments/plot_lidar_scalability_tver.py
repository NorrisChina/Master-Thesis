#!/usr/bin/env python3
"""Plot t_ver scalability results from lidar_scalability_tver.py.

Produces a quick-look log-log plot comparing SHA-256 vs RS-ID verification time
as payload size grows to sensor-like regimes.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_style import apply_thesis_style


def main() -> None:
    apply_thesis_style()
    p = argparse.ArgumentParser(description="Plot LiDAR-scale t_ver scalability")
    p.add_argument(
        "--csv",
        type=str,
        default=os.path.join("experiments", "results", "lidar_tver_scalability.csv"),
        help="Input CSV from lidar_scalability_tver.py",
    )
    p.add_argument(
        "--out",
        type=str,
        default=os.path.join("experiments", "results", "lidar_tver_scalability.png"),
        help="Output plot path",
    )
    p.add_argument("--title", type=str, default="t_ver scalability (LiDAR-scale payloads)")
    args = p.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df.sort_values("n_data_bytes")

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
