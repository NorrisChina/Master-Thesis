#!/usr/bin/env python3
"""Run identification experiments (Fig.2 and Fig.3 reproduction)

This script uses the local `idsys` implementation to run simulations
for RSID and SHA256ID systems and produces the plots for Fig.2 and Fig.3
in the paper. It is configurable and writes CSV results under
`experiments/results/`.

Usage examples:
  python experiments/run_experiments.py --mode fig2 --N 20000
  python experiments/run_experiments.py --mode fig3 --N 20000
"""

from __future__ import annotations
import os
import sys
import time
import argparse
import math
import csv
from typing import Any, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Ensure project src is on sys.path for local imports
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from idsys.core.idsystems import create_id_system
from idsys.core.common import IDCODES_U8


OUT_DIR = "experiments/results"
os.makedirs(OUT_DIR, exist_ok=True)


def random_state_bytes(nbits: int, rng: np.random.Generator) -> list[int]:
    nbytes = (nbits + 7) // 8
    arr = rng.integers(0, 256, size=nbytes, dtype=np.uint8)
    return arr.tolist()


def tag_equal(a: Any, b: Any) -> bool:
    # idsys tags may be int or list; normalize
    if isinstance(a, list):
        a = tuple(a)
    if isinstance(b, list):
        b = tuple(b)
    return a == b


def run_simulation(system_type: str, gf_exp: int, nver: int, ndata: int, p_desync: float, N: int, seed: int=123) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    params = {"gf_exp": gf_exp, "tag_pos": [2]}
    system = create_id_system(system_type, params)

    total_traffic = 0
    missed_repairs = 0
    desync_count = 0
    repair_count = 0
    ver_time_acc = 0.0

    for _ in range(N):
        msg_r = random_state_bytes(ndata, rng)
        if rng.random() < p_desync:
            # desynced: random different state
            msg_dt = random_state_bytes(ndata, rng)
            desync = True
        else:
            msg_dt = msg_r
            desync = False

        # compute tags and measure time
        t0 = time.perf_counter()
        tag_r = system.send(msg_r)
        t1 = time.perf_counter()
        tag_dt = system.send(msg_dt)
        t2 = time.perf_counter()

        ver_time_acc += (t1 - t0) + (t2 - t1)
        # traffic always includes verifier
        total_traffic += nver

        if tag_equal(tag_r, tag_dt):
            # DT assumes in-sync -> no repair sent
            if desync:
                missed_repairs += 1
            # else correct behaviour
        else:
            # DT detects mismatch -> sends repair
            total_traffic += ndata
            if desync:
                repair_count += 1

        if desync:
            desync_count += 1

    p_err_emp = missed_repairs / desync_count if desync_count > 0 else 0.0
    # theoretical collision probability assuming uniform tags of 2^gf_exp values
    if system_type in ("RSID", "SHA256ID"):
        p_err_theory = math.pow(2.0, -gf_exp) if gf_exp > 0 else 0.0
    else:
        p_err_theory = 0.0
    avg_traffic = total_traffic / N
    mean_ver_time = ver_time_acc / (2 * N)  # two verifiers per tick

    return {
        "system": system_type,
        "gf_exp": gf_exp,
        "nver": nver,
        "ndata": ndata,
        "p_desync": p_desync,
        "N": N,
        "p_err_emp": p_err_emp,
        "p_err_theory": p_err_theory,
        "avg_traffic_bits": avg_traffic,
        "normalized_traffic": avg_traffic / ndata,
        "mean_ver_time_s": mean_ver_time,
    }


def fig2(args: argparse.Namespace):
    # Fig.2 uses p_desync = 0.1 and plots tuples (repair error prob, normalized traffic)
    p_desync = 0.1
    N = args.N
    # Use the specific six (nver, ndata) tuples requested and run both systems (RS-ID and SHA256)
    pairs = [
        (16, 96),
        (12, 96),
        (8, 96),
        (4, 96),
        (12, 4001),
        (8, 4001),
    ]
    # Select systems, gracefully skip RSID if ecidcodes/idcodes isn't available
    requested_systems = args.systems or ["RSID", "SHA256ID"]
    systems = []
    for s in requested_systems:
        if s == "RSID" and IDCODES_U8 is None:
            print("Warning: RSID requires ecidcodes/idcodes which is not installed; skipping RSID.")
            continue
        systems.append(s)

    results = []
    for system in systems:
        for nver, ndata in pairs:
            # GF exponent selection strategy
            if args.gf_exp_mode == 'match-nver':
                gf_exp = nver
            else:
                gf_exp = args.gf_exp
            print(f"Running {system} nver={nver} ndata={ndata}...")
            res = run_simulation(system, gf_exp, nver, ndata, p_desync, N, seed=42)
            results.append(res)
    # save CSV
    csv_path = os.path.join(OUT_DIR, "fig2_results.csv")
    keys = list(results[0].keys())
    with open(csv_path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    # Plot: x = chosen p_err (empirical or theory), y = normalized_traffic
    plt.figure(figsize=(8,6))
    # plot points: different colors per data size, different markers per system
    markers = {"SHA256ID": 'o', 'RSID': 's'}
    colors = {96: 'C0', 4001: 'C1'}
    x_key = 'p_err_theory' if args.xaxis == 'theory' else 'p_err_emp'
    for r in results:
        plt.scatter(r[x_key], r['normalized_traffic'], marker=markers[r['system']], color=colors[r['ndata']], s=80, label=f"{r['system']} ({r['nver']},{r['ndata']})")

    plt.xscale('log')
    plt.xlabel(f"Repair error probability (p_err, {args.xaxis})")
    plt.ylabel('Normalized traffic (avg bits / ndata)')
    plt.title('Fig.2 reproduction (sampled)')
    plt.grid(True, which='both', ls='--', alpha=0.4)
    # avoid duplicate legend entries by creating unique handles
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best', fontsize='small')
    plt.savefig(os.path.join(OUT_DIR, 'fig2.png'), dpi=200)
    print('Fig.2 saved to', os.path.join(OUT_DIR, 'fig2.png'))


def fig3(args: argparse.Namespace):
    # fig3 normalized traffic vs p_desync for various nver/ndata
    N = args.N
    p_list = np.linspace(0.0, 1.0, 11)
    system = 'RSID'  # paper shows RS-ID, and SHA behaves similarly
    nver_list = [4,8,12,16]
    ndata = 96

    plt.figure(figsize=(8,6))
    for nver in nver_list:
        norm_traffic = []
        for p in p_list:
            # Choose gf_exp based on mode
            gf_exp = nver if args.gf_exp_mode == 'match-nver' else args.gf_exp
            print(f"Running RSID nver={nver} gf_exp={gf_exp} p_desync={p:.2f} ...")
            res = run_simulation(system, gf_exp, nver, ndata, p, N, seed=123)
            norm_traffic.append(res['normalized_traffic'])
        plt.plot(p_list, norm_traffic, marker='o', label=f'nver={nver}')

    # also plot traditional and optimal lines
    plt.plot(p_list, [1.0]*len(p_list), 'k--', label='Traditional traffic (normalized)')
    # optimal traffic normalized: E(Nopt)/ndata = p_desync
    plt.plot(p_list, p_list, 'k:', label='Optimal traffic')

    plt.xlabel('Desynchronisation probability p_desync')
    plt.ylabel('Normalized traffic')
    plt.title('Fig.3 reproduction (sampled)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, 'fig3.png'), dpi=200)
    print('Fig.3 saved to', os.path.join(OUT_DIR, 'fig3.png'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['fig2','fig3'], required=True)
    parser.add_argument('--N', type=int, default=20000, help='Number of ticks per configuration')
    parser.add_argument('--systems', nargs='+', choices=['RSID','SHA256ID'], help='Systems to run (default RSID SHA256ID)')
    parser.add_argument('--gf-exp', type=int, default=8, help='GF exponent when --gf-exp-mode=fixed (default 8)')
    parser.add_argument('--gf-exp-mode', choices=['fixed','match-nver'], default='fixed', help='How to choose GF exponent for each point')
    parser.add_argument('--xaxis', choices=['empirical','theory'], default='theory', help='Use empirical or theoretical p_err on x-axis')
    args = parser.parse_args()

    if args.mode == 'fig2':
        fig2(args)
    elif args.mode == 'fig3':
        fig3(args)


if __name__ == '__main__':
    main()
