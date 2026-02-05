#!/usr/bin/env python3
"""
Compute Overhead Fraction Plot: tver / Latency_Total vs. Bandwidth (for SHA and RS-ID)
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from latency_utils import OUT_DIR
from plot_style import apply_thesis_style
from experiments.latency_vs_ndata_fixed_nver import p_miss_sha, p_miss_rsid_concat

def main():
    apply_thesis_style()
    p = argparse.ArgumentParser(description="Compute Overhead Fraction Plot")
    p.add_argument("--nver-bits", type=int, default=16)
    p.add_argument("--ndata-bits", type=int, default=4001, help="Payload size in bits (default: 4001)")
    p.add_argument("--p-desync", type=float, default=0.1)
    p.add_argument(
        "--tver-ci-csv",
        type=str,
        default=None,
        help="Optional CSV from measure_tver_ci_table.py to source t_ver means (sha_mean_us/rsid_mean_us)",
    )
    p.add_argument(
        "--out",
        type=str,
        default=os.path.join(OUT_DIR, "compute_overhead_fraction_vs_bandwidth.png"),
    )
    args = p.parse_args()

    nver = int(args.nver_bits)
    ndata = int(args.ndata_bits)
    p_desync = float(args.p_desync)

    def _load_tver_from_ci_csv(path: str):
        df = pd.read_csv(path)
        sha_map = {int(r["n_data_bits"]): float(r["sha_mean_us"]) for _, r in df.iterrows()}
        rs_map = {int(r["n_data_bits"]): float(r["rsid_mean_us"]) for _, r in df.iterrows()}
        return sha_map, rs_map

    if args.tver_ci_csv:
        tver_sha_us, tver_rsid_us = _load_tver_from_ci_csv(args.tver_ci_csv)
    else:
        # Table 5.3 data
        tver_sha_us = {
            96: 2.05,
            4001: 14.80,
            819200: 2735.99,
            8388608: 27667.92,
            41943040: 137162.03,
        }
        tver_rsid_us = {
            96: 2.19,
            4001: 13.54,
            819200: 2879.77,
            8388608: 30496.18,
            41943040: 149797.66,
        }

    t_sha = tver_sha_us.get(ndata, np.nan) * 1e-6
    t_rs = tver_rsid_us.get(ndata, np.nan) * 1e-6

    # Sweep bandwidth from 10 kbps to 1 Gbps (log scale)
    B_grid = np.logspace(4, 9, num=200)  # 10^4 to 10^9 bps

    # SHA-256 ID
    p_sha = p_miss_sha(nver)
    L_sha = t_sha + (nver + p_desync * (1.0 - p_sha) * ndata) / B_grid
    frac_sha = t_sha / L_sha

    # RS-ID
    p_rs = p_miss_rsid_concat(nver)
    L_rs = t_rs + (nver + p_desync * (1.0 - p_rs) * ndata) / B_grid
    frac_rs = t_rs / L_rs

    plt.figure(figsize=(8.2, 4.6))
    plt.rc('font', size=24)
    plt.rc('axes', titlesize=28)
    plt.rc('axes', labelsize=26)
    plt.rc('xtick', labelsize=22)
    plt.rc('ytick', labelsize=22)
    plt.rc('legend', fontsize=22)
    plt.rc('figure', titlesize=28)

    plt.plot(B_grid/1e6, frac_sha, "o-", label="SHA-256")
    plt.plot(B_grid/1e6, frac_rs, "s-", label="RS-ID")
    plt.xscale("log")
    plt.xlabel("Bandwidth $B$ (Mbps, log scale)", fontsize=26)
    plt.ylabel(r"Compute Overhead Fraction $t_{ver}/Latency_{Total}$", fontsize=26)
    plt.title(f"Compute Overhead Fraction vs Bandwidth\n($n_{{ver}}={nver}$, $n_{{data}}={ndata}$, $p_{{desync}}={p_desync:.2f}$)", fontsize=24)
    plt.grid(True, which="both", ls=":", alpha=0.6)
    plt.legend(loc="best", fontsize=22, frameon=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print("Saved", args.out)

if __name__ == "__main__":
    main()
