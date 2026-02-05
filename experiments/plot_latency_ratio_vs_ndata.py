#!/usr/bin/env python3
"""Absolute latency vs payload size (fixed n_ver).

Used as Fig 5.5 in the payload scalability (LiDAR-scale) section.

Shows three absolute expected latency curves:
- RS-ID
- SHA256 (idcodes backend, labeled "SHA256")
- SHA256(hashlib)

Latency model (consistent with other Chapter 5 plots):
    L(B) = t_ver + (n_ver + p_desync * n_data) / B

We intentionally do NOT let p_miss reduce expected fallback transmission,
because a miss corresponds to incorrect synchronization (not a performance win).
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from latency_utils import OUT_DIR
from plot_style import apply_thesis_style


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def main():
    apply_thesis_style()
    p = argparse.ArgumentParser(description="Latency vs payload size (SHA backends + RS-ID)")
    p.add_argument("--nver-bits", type=int, default=16)
    p.add_argument("--B", type=float, default=5e6, help="Bandwidth (bits/s)")
    p.add_argument("--p-desync", type=float, default=0.1)
    p.add_argument(
        "--ndata-bits",
        type=int,
        nargs="+",
        default=[96, 4001, 819200, 8_388_608, 41_943_040],
        help="Payload sizes in bits (default: 96, 4001, 100KB, 1MiB, 5MiB)",
    )
    p.add_argument(
        "--sweep-csv",
        type=str,
        default=os.path.join(PROJECT_ROOT, "experiments", "results", "sweep_nver_tver_detail_ci.csv"),
        help=(
            "CSV with columns payload_bits,nver,sha_idcodes_mean_us,sha_hashlib_mean_us,rsid_mean_us "
            "(default: experiments/results/sweep_nver_tver_detail_ci.csv)"
        ),
    )
    p.add_argument(
        "--out",
        type=str,
        default=os.path.join(PROJECT_ROOT, "thesis_report", "figures", "plots", "latency_vs_ndata_backends_nver16.png"),
    )
    args = p.parse_args()

    nver = int(args.nver_bits)
    B = float(args.B)
    p_desync = float(args.p_desync)

    tver_sha_idcodes_us: dict[int, float] = {}
    tver_sha_hashlib_us: dict[int, float] = {}
    tver_rsid_us: dict[int, float] = {}

    if args.sweep_csv and os.path.exists(args.sweep_csv):
        df = pd.read_csv(args.sweep_csv)
        required = {"payload_bits", "nver", "sha_idcodes_mean_us", "sha_hashlib_mean_us", "rsid_mean_us"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Sweep CSV missing columns {sorted(missing)}: {args.sweep_csv}")

        df = df[df["nver"] == nver]
        for _, r in df.iterrows():
            payload = int(r["payload_bits"])
            tver_sha_idcodes_us[payload] = float(r["sha_idcodes_mean_us"])
            tver_sha_hashlib_us[payload] = float(r["sha_hashlib_mean_us"])
            tver_rsid_us[payload] = float(r["rsid_mean_us"])
    else:
        # Fallback: legacy table values (idcodes-only). SHA256(hashlib) curve will be omitted.
        tver_sha_idcodes_us = {
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

    ndata_list = [int(x) for x in args.ndata_bits]
    ndata = np.array(ndata_list, dtype=float)

    t_rs = np.array([tver_rsid_us.get(int(n), np.nan) for n in ndata_list], dtype=float) * 1e-6
    L_rs = t_rs + (nver + p_desync * ndata) / B

    t_sha_id = np.array([tver_sha_idcodes_us.get(int(n), np.nan) for n in ndata_list], dtype=float) * 1e-6
    L_sha_id = t_sha_id + (nver + p_desync * ndata) / B

    L_sha_hl = None
    if tver_sha_hashlib_us:
        t_sha_hl = np.array([tver_sha_hashlib_us.get(int(n), np.nan) for n in ndata_list], dtype=float) * 1e-6
        L_sha_hl = t_sha_hl + (nver + p_desync * ndata) / B

    plt.figure(figsize=(8.2, 4.6))
    plt.rc('font', size=24)
    plt.rc('axes', titlesize=28)
    plt.rc('axes', labelsize=26)
    plt.rc('xtick', labelsize=22)
    plt.rc('ytick', labelsize=22)
    plt.rc('legend', fontsize=22)
    plt.rc('figure', titlesize=28)

    plt.loglog(ndata, L_rs, "D-", color="#d62728", label="RS-ID")
    plt.loglog(ndata, L_sha_id, "o-", color="#1f77b4", label="SHA256")
    if L_sha_hl is not None:
        plt.loglog(ndata, L_sha_hl, "s--", color="#1f77b4", label="SHA256(hashlib)")

    plt.xlabel("Payload size $n_{data}$ (bits, log scale)", fontsize=26)
    plt.ylabel("Expected latency (s)", fontsize=26)
    plt.grid(True, which="both", ls=":", alpha=0.6)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=22, frameon=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight", pad_inches=0.15)
    print("Saved", args.out)

if __name__ == "__main__":
    main()
