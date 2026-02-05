
#!/usr/bin/env python3
"""Latency vs desync probability at fixed bandwidth (thesis-calibrated).

Uses the same parameter sources as `latency_empirical_bandwidth.py` by default:
    - t_ver values from Chapter 5 table (100,000 samples)
    - theoretical p_miss (SHA: 2^{-n_ver}; concatenated RS-ID: 2^{-n_ver})

Generates one figure per n_data (in bits), with curves for SHA-256 and RS-ID at n_ver in {4,16}.
"""
from __future__ import annotations
import os
import argparse
import time
from typing import Dict, Tuple

# Strictly use Table 5.1 values for p_miss
THESIS_PMISS = {
    4: 6.25e-2,
    16: 1.53e-5,
}

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from latency_utils import OUT_DIR, add_advantage_shading_by_baseline

from plot_style import apply_thesis_style

# Local project import (for optional measurement mode)
import sys
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from idsys.core.idsystems import create_id_system


def _load_tver_us_from_sweep(csv_path: str) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    """Load t_ver (microseconds) for SHA256(idcodes) and RS-ID from the sweep CSV.

    Returns:
        (sha_us, rsid_us) where keys are (payload_bits, nver_bits).
    """
    df = pd.read_csv(csv_path)
    required = {"payload_bits", "nver", "sha_idcodes_mean_us", "rsid_mean_us"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Sweep CSV missing columns {sorted(missing)}: {csv_path}")

    sha: dict[tuple[int, int], float] = {}
    rsid: dict[tuple[int, int], float] = {}
    for _, r in df.iterrows():
        key = (int(r["payload_bits"]), int(r["nver"]))
        sha[key] = float(r["sha_idcodes_mean_us"])
        rsid[key] = float(r["rsid_mean_us"])
    return sha, rsid


def p_miss_sha(nver_bits: int) -> float:
    return 2.0 ** (-int(nver_bits))


def p_miss_rsid_concat(nver_bits: int) -> float:
    # Concatenated RS-ID (Chapter 3): collision/miss probability is dominated by 1/q
    # with q = 2^{n_ver}, hence p_miss ≈ 2^{-n_ver} (independent of n_data).
    return 2.0 ** (-int(nver_bits))


# Empirical t_ver (microseconds) from Chapter 5 table (100,000 samples).
THESIS_TVER_US_SHA: Dict[Tuple[int, int], float] = {
    (96, 4): 2.13,
    (96, 16): 2.24,
    (4001, 4): 14.65,
    (4001, 16): 15.41,

    # LiDAR-scale measurements (see experiments/results/lidar_tver_scalability.csv)
    (81920, 16): 272.38,     # 10 KB
    (819200, 16): 2697.38,   # 100 KB
}

THESIS_TVER_US_RSID: Dict[Tuple[int, int], float] = {
    (96, 4): 14.55,
    (96, 16): 3.79,
    (4001, 4): 51.78,
    (4001, 16): 38.57,

    # LiDAR-scale measurements (see experiments/results/lidar_tver_scalability.csv)
    (81920, 16): 627.95,     # 10 KB
    (819200, 16): 4824.12,   # 100 KB
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Latency vs Desync plot')
    p.add_argument('--ndata-list', type=int, nargs='+', default=[96, 4001], help='Data sizes in bits')
    p.add_argument('--nver-list', type=int, nargs='+', default=[4, 16], help='Verifier sizes in bits')
    p.add_argument('--B-fixed', dest='B_fixed', type=float, default=5e6)
    p.add_argument('--p-points', dest='p_points', type=int, default=101)
    p.add_argument('--p-desync-min', type=float, default=0.0)
    p.add_argument('--p-desync-max', type=float, default=1.0)
    p.add_argument('--measure-tver', action='store_true', default=False, help='Measure t_ver via idsys instead of using THESIS_TVER_US_* tables')
    p.add_argument('--N-tver', type=int, default=100000, help='Samples for t_ver measurement (only with --measure-tver)')
    p.add_argument('--seed', type=int, default=123, help='RNG seed for measurement mode')
    p.add_argument('--out-prefix', default=os.path.join(OUT_DIR, 'latency_vs_desync'))
    p.add_argument('--out-file', default=None, help='Exact output file path (only supported when ndata-list has one element).')
    return p


def _random_state_bytes(nbits: int, rng: np.random.Generator) -> list[int]:
    nbytes = (int(nbits) + 7) // 8
    arr = rng.integers(0, 256, size=nbytes, dtype=np.uint8)
    return arr.tolist()


def _random_state_rs2id_symbols(nbits: int, nver_bits: int, rng: np.random.Generator) -> list[int]:
    # Pack into GF(2^{2*n_ver}) symbols (k_i=2)
    symbol_bits = 2 * int(nver_bits)
    k = int(np.ceil(int(nbits) / float(symbol_bits)))
    gf_range = 1 << symbol_bits
    arr = rng.integers(0, gf_range, size=k, dtype=np.uint32)
    return arr.astype(np.int64).tolist()


def _measure_tver_us(system_type: str, gf_exp: int, ndata_bits: int, *, N: int, seed: int) -> float:
    rng = np.random.default_rng(int(seed))
    params = {"gf_exp": int(gf_exp), "tag_pos": [2]}
    if system_type == "RS2ID":
        params["tag_pos_in"] = [2]
    system = create_id_system(system_type, params)

    # warm-up
    for _ in range(200):
        if system_type == "RS2ID":
            _ = system.send(_random_state_rs2id_symbols(ndata_bits, gf_exp, rng))
        else:
            _ = system.send(_random_state_bytes(ndata_bits, rng))

    t_acc = 0.0
    for _ in range(int(N)):
        if system_type == "RS2ID":
            msg = _random_state_rs2id_symbols(ndata_bits, gf_exp, rng)
        else:
            msg = _random_state_bytes(ndata_bits, rng)
        t0 = time.perf_counter()
        _ = system.send(msg)
        t1 = time.perf_counter()
        t_acc += (t1 - t0)
    return (t_acc / float(N)) * 1e6


def main():
    apply_thesis_style(base_fontsize=24)
    args = build_parser().parse_args()

    sweep_csv = os.path.join(PROJECT_ROOT, "experiments", "results", "sweep_nver_tver_detail_ci.csv")
    sweep_sha_us: dict[tuple[int, int], float] = {}
    sweep_rsid_us: dict[tuple[int, int], float] = {}
    if os.path.exists(sweep_csv):
        try:
            sweep_sha_us, sweep_rsid_us = _load_tver_us_from_sweep(sweep_csv)
        except Exception as e:
            print(f"Warning: failed to read sweep CSV for t_ver ({sweep_csv}): {e}")
    else:
        print(f"Warning: sweep CSV not found ({sweep_csv}); falling back to legacy THESIS_TVER tables")

    B = float(args.B_fixed)
    p_grid = np.linspace(float(args.p_desync_min), float(args.p_desync_max), int(args.p_points))

    # Only allow nver=4,16
    for nver_bits in args.nver_list:
        assert nver_bits in (4, 16), f"Only nver=4,16 supported! Got {nver_bits}"

    # Use line style (not light colors) to distinguish n_ver so curves stay visible.
    COLORS = {
        "SHA-256": "#1f77b4",     # blue
        "RS-ID": "#d62728",       # red
        "Traditional": "#222222", # near-black
    }
    LINESTYLES = {
        4: "-",
        16: "--",
    }
    LABELS = {
        ("SHA-256", 4): "SHA-256 $n_{ver}=4$",
        ("SHA-256", 16): "SHA-256 $n_{ver}=16$",
        ("RS-ID", 4): "RS-ID $n_{ver}=4$",
        ("RS-ID", 16): "RS-ID $n_{ver}=16$",
        "Traditional": "Traditional",
    }
    for n_data_bits in args.ndata_list:
        n_data_bits_i = int(n_data_bits)
        fig, ax = plt.subplots(figsize=(6.6, 5.2))
        # Traditional baseline
        L_trad_s = np.full_like(p_grid, (n_data_bits_i / B), dtype=float)
        # Plot in microseconds for readability.
        L_trad = L_trad_s * 1e6
        ax.plot(
            p_grid,
            L_trad,
            linestyle=":",
            color=COLORS["Traditional"],
            linewidth=2.6,
            label=LABELS["Traditional"],
            zorder=2.5,
        )
        for nver_bits in args.nver_list:
            nver_bits_i = int(nver_bits)
            assert nver_bits_i in (4, 16), f"Only nver=4,16 supported! Got {nver_bits_i}"
            if args.measure_tver:
                t_sha_us = _measure_tver_us(
                    "SHA256ID",
                    nver_bits_i,
                    n_data_bits_i,
                    N=int(args.N_tver),
                    seed=int(args.seed) + 10_000 * n_data_bits_i + nver_bits_i,
                )
                t_rs_us = _measure_tver_us(
                    "RS2ID",
                    nver_bits_i,
                    n_data_bits_i,
                    N=int(args.N_tver),
                    seed=int(args.seed) + 20_000 * n_data_bits_i + nver_bits_i,
                )
            else:
                key = (n_data_bits_i, nver_bits_i)
                if key in sweep_sha_us:
                    t_sha_us = sweep_sha_us[key]
                else:
                    t_sha_us = THESIS_TVER_US_SHA[key]

                if key in sweep_rsid_us:
                    t_rs_us = sweep_rsid_us[key]
                else:
                    t_rs_us = THESIS_TVER_US_RSID[key]
            # SHA-256 curve
            t_sha = float(t_sha_us) * 1e-6
            # Do not credit p_miss as a latency reduction: misses are incorrect sync.
            L_sha_s = t_sha + (nver_bits_i + p_grid * n_data_bits_i) / B
            ax.plot(
                p_grid,
                L_sha_s * 1e6,
                color=COLORS["SHA-256"],
                linestyle=LINESTYLES[nver_bits_i],
                linewidth=3.0,
                label=LABELS[("SHA-256", nver_bits_i)],
                zorder=3.0,
            )
            # RS-ID curve
            t_rs = float(t_rs_us) * 1e-6
            # Do not credit p_miss as a latency reduction: misses are incorrect sync.
            L_rs_s = t_rs + (nver_bits_i + p_grid * n_data_bits_i) / B
            ax.plot(
                p_grid,
                L_rs_s * 1e6,
                color=COLORS["RS-ID"],
                linestyle=LINESTYLES[nver_bits_i],
                linewidth=3.0,
                label=LABELS[("RS-ID", nver_bits_i)],
                zorder=3.0,
            )
        ax.set_xlabel('Desync probability $p_{desync}$', fontsize=20)
        ax.set_ylabel(r'Expected latency ($\mu$s)', fontsize=20)
        ax.grid(True, ls='--', alpha=0.3)
        add_advantage_shading_by_baseline(ax, p_grid, L_trad)
        import matplotlib.ticker as mticker
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)
        # 图例放到图外右侧
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=14, frameon=True)
        if args.out_file is not None:
            if len(args.ndata_list) != 1:
                raise ValueError("--out-file requires --ndata-list to have exactly one element")
            out_path = str(args.out_file)
        else:
            out_path = os.path.join(PROJECT_ROOT, 'thesis_report', 'figures', 'plots', f'latency_vs_desync_{n_data_bits_i}bits.png')
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print('Saved', out_path)


if __name__ == '__main__':
    main()
