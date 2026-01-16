#!/usr/bin/env python3
"""Latency vs desync probability at fixed bandwidth (thesis-calibrated).

Uses the same parameter sources as `latency_empirical_bandwidth.py` by default:
    - t_ver values from Chapter 5 table (100,000 samples)
    - theoretical p_miss (SHA: 2^{-n_ver}; RS-ID: ceil(n_data/n_ver)/2^{n_ver}, capped at 1)

Generates one figure per n_data (in bits), with curves for SHA-256 and RS-ID at n_ver in {4,16}.
"""
from __future__ import annotations
import os
import argparse
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

from latency_utils import OUT_DIR, add_advantage_shading_by_baseline


def p_miss_sha(nver_bits: int) -> float:
    return 2.0 ** (-int(nver_bits))


def p_miss_rsid_bound(n_data_bits: int, nver_bits: int) -> float:
    n_data_bits_i = int(n_data_bits)
    nver_bits_i = int(nver_bits)
    bound = float(np.ceil(n_data_bits_i / nver_bits_i) / (2.0 ** nver_bits_i))
    return min(1.0, bound)


# Empirical t_ver (microseconds) from Chapter 5 table (100,000 samples).
THESIS_TVER_US_SHA: Dict[Tuple[int, int], float] = {
    (96, 4): 2.13,
    (96, 16): 2.24,
    (4001, 4): 14.65,
    (4001, 16): 15.41,
}

THESIS_TVER_US_RSID: Dict[Tuple[int, int], float] = {
    (96, 4): 14.55,
    (96, 16): 3.79,
    (4001, 4): 51.78,
    (4001, 16): 38.57,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Latency vs Desync plot')
    p.add_argument('--ndata-list', type=int, nargs='+', default=[96, 4001], help='Data sizes in bits')
    p.add_argument('--nver-list', type=int, nargs='+', default=[4, 16], help='Verifier sizes in bits')
    p.add_argument('--B-fixed', dest='B_fixed', type=float, default=5e6)
    p.add_argument('--p-points', dest='p_points', type=int, default=101)
    p.add_argument('--p-desync-min', type=float, default=0.0)
    p.add_argument('--p-desync-max', type=float, default=1.0)
    p.add_argument('--out-prefix', default=os.path.join(OUT_DIR, 'latency_vs_desync'))
    return p


def main():
    args = build_parser().parse_args()

    B = float(args.B_fixed)
    p_grid = np.linspace(float(args.p_desync_min), float(args.p_desync_max), int(args.p_points))

    for n_data_bits in args.ndata_list:
        n_data_bits_i = int(n_data_bits)

        plt.figure(figsize=(9, 6))
        # Traditional baseline
        L_trad = np.full_like(p_grid, (n_data_bits_i / B), dtype=float)
        plt.plot(p_grid, L_trad, 'k--', label=f'Traditional (ndata={n_data_bits_i} bits)')

        for nver_bits in args.nver_list:
            nver_bits_i = int(nver_bits)

            # SHA-256 curve
            t_sha = THESIS_TVER_US_SHA[(n_data_bits_i, nver_bits_i)] * 1e-6
            p_sha = p_miss_sha(nver_bits_i)
            L_sha = t_sha + (nver_bits_i + p_grid * (1.0 - p_sha) * n_data_bits_i) / B
            plt.plot(p_grid, L_sha, label=f'ID-SHA nver={nver_bits_i} (t_ver={t_sha*1e6:.2f}µs, p_miss={p_sha:.2e})')

            # RS-ID curve
            t_rs = THESIS_TVER_US_RSID[(n_data_bits_i, nver_bits_i)] * 1e-6
            p_rs = p_miss_rsid_bound(n_data_bits_i, nver_bits_i)
            L_rs = t_rs + (nver_bits_i + p_grid * (1.0 - p_rs) * n_data_bits_i) / B
            plt.plot(p_grid, L_rs, label=f'ID-RS-ID nver={nver_bits_i} (t_ver={t_rs*1e6:.2f}µs, p_miss={p_rs:.2e})')

        plt.xlabel('Desync probability $p_{desync}$')
        plt.ylabel('Expected latency (s)')
        plt.title(f'Latency vs Desync Probability (ndata={n_data_bits_i} bits, B={B/1e6:.2f} Mbps)')
        plt.grid(True, ls='--', alpha=0.3)

        # Background shading relative to the traditional baseline (green=advantage, red=penalty).
        ax = plt.gca()
        add_advantage_shading_by_baseline(ax, p_grid, L_trad)

        plt.legend(fontsize='small')
        out_path = f"{args.out_prefix}_{n_data_bits_i}bits.png"
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        print('Saved', out_path)


if __name__ == '__main__':
    main()
