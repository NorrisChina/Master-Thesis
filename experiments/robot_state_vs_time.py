#!/usr/bin/env python3
"""Simulate robot state over time and visualize:

- Top: Robot state vs time with accept range shading; mark resets and miss errors
- Bottom: Accumulated miss errors vs time

This is a demo figure similar to the previously used illustration.

Outputs:
- experiments/results/robot_state_vs_time.png
"""

from __future__ import annotations
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)


def simulate(ticks: int = 2000,
             verifier_bits: int = 8,
             accept_low: float = 40.0,
             accept_high: float = 60.0,
             ref_value: float = 50.0,
             p_desync: float = 0.1,
             sigma_sync: float = 2.0,
             sigma_desync: float = 12.0,
             seed: int = 123) -> dict:
    rng = np.random.default_rng(seed)

    # Approximate miss probability from verifier length
    p_miss = 2.0 ** (-verifier_bits)

    states = []
    resets_idx = []
    miss_idx = []
    acc_miss = []

    miss_count = 0
    state = ref_value

    for t in range(ticks):
        # Decide in-sync or desync
        desync = rng.random() < p_desync

        if desync:
            # Larger deviations when desynced
            state = ref_value + rng.normal(0.0, sigma_desync)
        else:
            # Small jitter around reference when in sync
            state = ref_value + rng.normal(0.0, sigma_sync)

        # Detection outcome: if desynced, we detect with prob (1 - p_miss)
        detected_mismatch = desync and (rng.random() < (1.0 - p_miss))

        # If mismatch detected -> repair (reset)
        if detected_mismatch:
            resets_idx.append(t)
            state = ref_value  # reset applied
        else:
            # Potential miss error: dangerous if outside acceptance range
            if desync and (state < accept_low or state > accept_high):
                miss_count += 1
                miss_idx.append(t)

        states.append(state)
        acc_miss.append(miss_count)

    return {
        "states": np.array(states),
        "resets": np.array(resets_idx),
        "misses": np.array(miss_idx),
        "acc_miss": np.array(acc_miss),
    }


def plot(sim: dict, verifier_bits: int, accept_low: float, accept_high: float, ref_value: float):
    ticks = len(sim["states"])
    x = np.arange(ticks)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Top: state vs time
    ax1.plot(x, sim["states"], color='C0', linewidth=0.8, label='Robot State')
    ax1.axhspan(accept_low, accept_high, color='C2', alpha=0.2, label=f'Accept Range {int(accept_low)}-{int(accept_high)}')
    ax1.axhline(ref_value, color='k', linestyle='--', linewidth=1.0, label='Reset to 50')

    if sim["misses"].size > 0:
        ax1.scatter(sim["misses"], sim["states"][sim["misses"]], s=25, marker='o', color='red', label='Miss Error')
    if sim["resets"].size > 0:
        # plot resets at the reset baseline
        ax1.scatter(sim["resets"], np.full_like(sim["resets"], ref_value), s=25, marker='x', color='red', label='Reset to 50')

    ax1.set_ylabel('State Value')
    ax1.set_title(f'Robot State vs Time (Verifier: {verifier_bits} bits)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Bottom: accumulated miss errors
    ax2.plot(x, sim["acc_miss"], color='red', linewidth=1.2, label='Accumulated Miss Errors')
    ax2.set_xlabel('Ticks')
    ax2.set_ylabel('Count')
    ax2.set_title('Accumulated Miss Errors vs Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, 'robot_state_vs_time.png')
    fig.savefig(out_path, dpi=200)
    print('Wrote', out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticks', type=int, default=2000)
    parser.add_argument('--verifier_bits', type=int, default=8)
    parser.add_argument('--accept_low', type=float, default=40.0)
    parser.add_argument('--accept_high', type=float, default=60.0)
    parser.add_argument('--ref_value', type=float, default=50.0)
    parser.add_argument('--p_desync', type=float, default=0.1)
    parser.add_argument('--sigma_sync', type=float, default=2.0)
    parser.add_argument('--sigma_desync', type=float, default=12.0)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    sim = simulate(ticks=args.ticks,
                   verifier_bits=args.verifier_bits,
                   accept_low=args.accept_low,
                   accept_high=args.accept_high,
                   ref_value=args.ref_value,
                   p_desync=args.p_desync,
                   sigma_sync=args.sigma_sync,
                   sigma_desync=args.sigma_desync,
                   seed=args.seed)

    plot(sim, args.verifier_bits, args.accept_low, args.accept_high, args.ref_value)


if __name__ == '__main__':
    main()
