#!/usr/bin/env python3
"""Latency vs payload size (fixed n_ver).

This script is meant as a quick-look companion to the thesis latency model.
It fixes:
  - verifier length n_ver
  - bandwidth B
  - desync probability p_desync
and compares expected latency as n_data grows (e.g., 96b / 4001b / 100KB).

Model (hybrid):
  L_ID = t_ver + (n_ver + p_desync*(1-p_miss)*n_data) / B
  L_trad = n_data / B

Where:
  p_miss^SHA = 2^{-n_ver}
  p_miss^RS  = 2^{-n_ver}  (concatenated RS-ID, Chapter 3; independent of n_data)

t_ver sources:
  - 96/4001: thesis table values (Chapter 5)
  - 10KB/100KB: experiments/results/lidar_tver_scalability.csv (measured)

Output:
  experiments/results/latency_vs_ndata_nver16.png
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from latency_utils import OUT_DIR

from plot_style import apply_thesis_style


def p_miss_sha(nver_bits: int) -> float:
    return 2.0 ** (-int(nver_bits))


def p_miss_rsid_concat(nver_bits: int) -> float:
  return 2.0 ** (-int(nver_bits))


def main() -> None:
  apply_thesis_style()
  p = argparse.ArgumentParser(description="Latency vs payload size (fixed n_ver)")
  p.add_argument("--nver-bits", type=int, default=16)
  p.add_argument("--B", type=float, default=5e6, help="Bandwidth (bits/s)")
  p.add_argument("--p-desync", type=float, default=0.1)
  p.add_argument(
    "--ndata-bits",
    type=int,
    nargs="+",
    default=[96, 4001, 819200, 8_388_608, 41_943_040],
    help="Payload sizes in bits (default: 96, 4001, 100KB, 1MiB, 5MiB)"
  )
  p.add_argument(
    "--tver-ci-csv",
    type=str,
    default=None,
    help="Optional CSV from measure_tver_ci_table.py to source t_ver means (sha_mean_us/rsid_mean_us)"
  )
  p.add_argument(
    "--out",
    type=str,
    default=os.path.join(OUT_DIR, "latency_vs_ndata_nver16.png")
  )
  args = p.parse_args()

  nver = int(args.nver_bits)
  B = float(args.B)
  p_desync = float(args.p_desync)

  def _load_tver_from_ci_csv(path: str) -> Tuple[Dict[int, float], Dict[int, float]]:
    df = pd.read_csv(path)
    if "n_data_bits" not in df.columns:
      raise ValueError(f"Missing n_data_bits in t_ver CSV: {path}")
    if "sha_mean_us" not in df.columns or "rsid_mean_us" not in df.columns:
      raise ValueError(f"Missing sha_mean_us/rsid_mean_us in t_ver CSV: {path}")
    sha_map: Dict[int, float] = {}
    rs_map: Dict[int, float] = {}
    for _, r in df.iterrows():
      nbits = int(r["n_data_bits"])
      sha_map[nbits] = float(r["sha_mean_us"])
      rs_map[nbits] = float(r["rsid_mean_us"])
    return sha_map, rs_map

  # t_ver (microseconds)
  # Default: thesis table values for 96/4001 and prior LiDAR measurement for 10KB/100KB.
  tver_sha_us: Dict[int, float]
  tver_rsid_us: Dict[int, float]
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

    ndata_list = [int(x) for x in args.ndata_bits]

    # Validate availability of t_ver for requested sizes
    missing_sha = [n for n in ndata_list if n not in tver_sha_us]
    missing_rs = [n for n in ndata_list if n not in tver_rsid_us]
    if missing_sha or missing_rs:
        raise ValueError(
            "Missing t_ver entries for requested n_data_bits. "
            f"missing_sha={missing_sha}, missing_rsid={missing_rs}. "
            "Add them to the tver_*_us dicts (or extend the measurement scripts)."
        )

    ndata = np.array(ndata_list, dtype=float)

    # Traditional
    L_trad = ndata / B

    # SHA-256 ID
    p_sha = p_miss_sha(nver)
    t_sha = np.array([tver_sha_us[int(n)] for n in ndata_list], dtype=float) * 1e-6
    L_sha = t_sha + (nver + p_desync * (1.0 - p_sha) * ndata) / B

    # RS-ID
    p_rs = np.array([p_miss_rsid_concat(nver) for _ in ndata_list], dtype=float)
    t_rs = np.array([tver_rsid_us[int(n)] for n in ndata_list], dtype=float) * 1e-6
    L_rs = t_rs + (nver + p_desync * (1.0 - p_rs) * ndata) / B

    # Plot
    plt.figure(figsize=(8.2, 4.6))
    # 放大所有字体
    plt.rc('font', size=24)
    plt.rc('axes', titlesize=28)
    plt.rc('axes', labelsize=26)
    plt.rc('xtick', labelsize=22)
    plt.rc('ytick', labelsize=22)
    plt.rc('legend', fontsize=22)
    plt.rc('figure', titlesize=28)

    plt.plot(ndata, L_trad, "k--", label="Traditional")
    plt.plot(ndata, L_sha, "o-", label=f"ID-SHA nver={nver}")
    plt.plot(ndata, L_rs, "s-", label=f"ID-RS nver={nver}")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Payload size n_data (bits, log scale)", fontsize=26)
    plt.ylabel("Expected latency (s, log scale)", fontsize=26)
    plt.title(f"Latency vs payload size (n_ver={nver}, B={B/1e6:.2f} Mbps, p_desync={p_desync:.2f})", fontsize=28)
    plt.grid(True, which="both", ls=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=22, frameon=True)

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
