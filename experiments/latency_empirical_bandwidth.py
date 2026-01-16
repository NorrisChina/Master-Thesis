#!/usr/bin/env python3
"""Latency vs bandwidth plots (hybrid model) using thesis-calibrated parameters.

This script generates expected latency curves vs bandwidth using the hybrid model:

    L_trad = n_data_bits / B
    L_id   = t_ver + n_ver_bits / B + p_desync * (1 - p_miss) * n_data_bits / B

By default, it uses:
    - Theoretical p_miss from Chapter 3 bounds (SHA: 2^{-n_ver}; RS-ID: ceil(n_data_bits/n_ver)/2^{n_ver}, capped at 1)
    - Empirical t_ver values reported in Chapter 5 (Table t_ver)

Inputs n_data are interpreted as *bits* (e.g., 96 and 4001).
"""
from __future__ import annotations
import os
import time
import argparse
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

# Ensure project src is on sys.path for local imports
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
import sys
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from idsys.core.idsystems import create_id_system
from idsys.core.common import IDCODES_U8

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


def random_state_bytes(nbits: int | float, rng: np.random.Generator) -> List[int]:
    nbits_i = int(nbits)
    nbytes = (nbits_i + 7) // 8
    arr = rng.integers(0, 256, size=nbytes, dtype=np.uint8)
    return arr.tolist()


def tag_equal(a, b) -> bool:
    if isinstance(a, list):
        a = tuple(a)
    if isinstance(b, list):
        b = tuple(b)
    return a == b


def measure_t_ver(system_type: str, gf_exp: int, ndata_bits: int, N: int = 10000, seed: int = 123) -> float:
    rng = np.random.default_rng(seed)
    params = {"gf_exp": gf_exp, "tag_pos": [2]}
    system = create_id_system(system_type, params)
    t_acc = 0.0
    # warm-up a few calls
    for _ in range(100):
        _ = system.send(random_state_bytes(ndata_bits, rng))
    # measure N calls
    for _ in range(N):
        msg = random_state_bytes(ndata_bits, rng)
        t0 = time.perf_counter()
        _ = system.send(msg)
        t1 = time.perf_counter()
        t_acc += (t1 - t0)
    return t_acc / N


def measure_p_miss(system_type: str, gf_exp: int, ndata_bits: int, N: int = 20000, seed: int = 456) -> float:
    rng = np.random.default_rng(seed)
    params = {"gf_exp": gf_exp, "tag_pos": [2]}
    system = create_id_system(system_type, params)
    misses = 0
    # For empirical miss measurement, always desync: compare tags of two random different states
    for _ in range(N):
        msg_a = random_state_bytes(ndata_bits, rng)
        msg_b = random_state_bytes(ndata_bits, rng)
        tag_a = system.send(msg_a)
        tag_b = system.send(msg_b)
        if tag_equal(tag_a, tag_b):
            misses += 1
    return misses / N


def plot_for_ndata(n_data_bits: int, nver_list: List[int], p_desync: float, B_min: float, B_max: float, B_points: int,
                    sha_stats: Dict[int, Tuple[float, float]], rsid_stats: Dict[int, Tuple[float, float]] | None,
                    out_path: str):
    B = np.logspace(np.log10(B_min), np.log10(B_max), B_points)
    plt.figure(figsize=(9,6))

    # Traditional baseline
    L_trad = n_data_bits / B
    plt.loglog(B, L_trad, 'k--', label=f'Traditional (ndata={n_data_bits} bits)')

    # SHA curves
    for nver in nver_list:
        n_ver_bits = float(nver)
        t_ver, p_miss = sha_stats[nver]
        L_id = t_ver + (n_ver_bits + p_desync * (1.0 - p_miss) * n_data_bits) / B
        plt.loglog(B, L_id, label=f'ID-SHA nver={nver} (t_ver={t_ver*1e6:.1f}µs, p_miss={p_miss:.2e})')

    # RSID curves
    if rsid_stats is not None:
        for nver in nver_list:
            n_ver_bits = float(nver)
            t_ver, p_miss = rsid_stats[nver]
            L_id = t_ver + (n_ver_bits + p_desync * (1.0 - p_miss) * n_data_bits) / B
            plt.loglog(B, L_id, label=f'ID-RSID nver={nver} (t_ver={t_ver*1e6:.1f}µs, p_miss={p_miss:.2e})')
    else:
        plt.text(B_min, L_trad[0]*1.5, 'RSID unavailable (dependency missing)', fontsize=9, color='gray')

    plt.xlabel('Bandwidth B (bits/s)')
    plt.ylabel('Expected latency (s)')
    plt.title(f'Latency vs Bandwidth (ndata={n_data_bits} bits, p_desync={p_desync:.2f})')
    plt.grid(True, which='both', ls='--', alpha=0.3)

    # Background shading relative to the traditional baseline (green=advantage, red=penalty).
    ax = plt.gca()
    add_advantage_shading_by_baseline(ax, B, L_trad)

    plt.legend(fontsize='small')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print('Saved', out_path)


def main():
    parser = argparse.ArgumentParser(description='Empirical latency vs bandwidth (SHA256ID/RSID)')
    parser.add_argument('--nver-list', type=int, nargs='+', default=[4,16])
    parser.add_argument('--ndata-list', type=int, nargs='+', default=[96,4001], help='Data sizes in bits')
    parser.add_argument('--p-desync', type=float, default=0.1)
    parser.add_argument('--B-min', type=float, default=1e4)
    parser.add_argument('--B-max', type=float, default=1e8)
    parser.add_argument('--B-points', type=int, default=200)
    parser.add_argument('--use-thesis-values', action='store_true', default=True,
                        help='Use thesis t_ver (Chapter 5) and theoretical p_miss (Chapter 3) (default)')
    parser.add_argument('--measure', dest='use_thesis_values', action='store_false',
                        help='Measure t_ver/p_miss using idsys instead of using thesis values')
    parser.add_argument('--N-tver', type=int, default=100000, help='Samples for t_ver measurement (only with --measure)')
    parser.add_argument('--N-pmiss', type=int, default=20000, help='Samples for empirical p_miss measurement (only with --measure)')
    parser.add_argument('--sha-pmiss-mode', choices=['empirical','theory','fixed'], default='theory', help='SHA p_miss source (only with --measure)')
    parser.add_argument('--rsid-pmiss-mode', choices=['empirical','theory','fixed'], default='theory', help='RSID p_miss source (only with --measure)')
    parser.add_argument('--sha-pmiss', type=float, default=0.0, help='Fixed p_miss for SHA when mode=fixed (only with --measure)')
    parser.add_argument('--rsid-pmiss', type=float, default=0.0, help='Fixed p_miss for RSID when mode=fixed (only with --measure)')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    # Check RSID availability
    rsid_available = IDCODES_U8 is not None

    for ndata_bits in args.ndata_list:
        print(f"Preparing curves for ndata={ndata_bits} bits...")
        sha_stats: Dict[int, Tuple[float, float]] = {}
        rsid_stats: Dict[int, Tuple[float, float]] | None = {} if rsid_available else None
        for nver in args.nver_list:
            if args.use_thesis_values:
                # Use thesis t_ver table + theoretical p_miss bounds
                t_sha = THESIS_TVER_US_SHA[(int(ndata_bits), int(nver))] * 1e-6
                p_sha = p_miss_sha(nver)
                sha_stats[nver] = (t_sha, p_sha)
                print(f"SHA256ID nver={nver}: t_ver={t_sha*1e6:.2f}µs, p_miss={p_sha:.3e}")

                if rsid_available:
                    t_rs = THESIS_TVER_US_RSID[(int(ndata_bits), int(nver))] * 1e-6
                    p_rs = p_miss_rsid_bound(int(ndata_bits), int(nver))
                    assert rsid_stats is not None
                    rsid_stats[nver] = (t_rs, p_rs)
                    print(f"RSID nver={nver}: t_ver={t_rs*1e6:.2f}µs, p_miss={p_rs:.3e}")
            else:
                gf_exp = nver  # align field size/tag length with verifier bits
                # SHA256ID
                t_sha = measure_t_ver('SHA256ID', gf_exp, int(ndata_bits), N=args.N_tver, seed=args.seed)
                if args.sha_pmiss_mode == 'empirical':
                    p_sha = measure_p_miss('SHA256ID', gf_exp, int(ndata_bits), N=args.N_pmiss, seed=args.seed+1)
                elif args.sha_pmiss_mode == 'theory':
                    p_sha = 2.0 ** (-gf_exp)
                else:
                    p_sha = args.sha_pmiss
                sha_stats[nver] = (t_sha, p_sha)
                print(f"SHA256ID nver={nver}: t_ver={t_sha*1e6:.2f}µs, p_miss={p_sha:.3e}")
                # RSID
                if rsid_available:
                    t_rs = measure_t_ver('RSID', gf_exp, int(ndata_bits), N=args.N_tver, seed=args.seed+2)
                    if args.rsid_pmiss_mode == 'empirical':
                        p_rs = measure_p_miss('RSID', gf_exp, int(ndata_bits), N=args.N_pmiss, seed=args.seed+3)
                    elif args.rsid_pmiss_mode == 'theory':
                        p_rs = p_miss_rsid_bound(int(ndata_bits), int(nver))
                    else:
                        p_rs = args.rsid_pmiss
                    assert rsid_stats is not None
                    rsid_stats[nver] = (t_rs, p_rs)
                    print(f"RSID nver={nver}: t_ver={t_rs*1e6:.2f}µs, p_miss={p_rs:.3e}")

        out_path = os.path.join(OUT_DIR, f'latency_empirical_bandwidth_{int(ndata_bits)}bits.png')
        plot_for_ndata(int(ndata_bits), args.nver_list, args.p_desync, args.B_min, args.B_max, args.B_points,
                       sha_stats, rsid_stats, out_path)


if __name__ == '__main__':
    main()
