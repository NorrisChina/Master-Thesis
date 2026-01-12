#!/usr/bin/env python3
"""Latency modelling for identification traffic.

Model assumptions (configurable):
- Verification time per tick = `mean_ver_time_s` (from measurement or default)
- Repair transfer time = one RTT + (ndata bits / bandwidth)
- If repair is missed with probability p_err, we model retries geometrically:
  expected repair time = repair_time / (1 - p_err)

Expected latency per tick:
  latency = verification_time + p_desync * expected_repair_time
           = verification_time + p_desync * repair_time / (1 - p_err)

This script reads `experiments/results/fig2_results.csv`, computes latency
for each config given network params, prints a table, and saves a plot of
latency (ms) vs normalized traffic.
"""
from __future__ import annotations
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Resolve paths relative to this script's directory to avoid CWD issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "results", "fig2_results.csv")
OUT_PNG = os.path.join(BASE_DIR, "results", "latency_vs_traffic.png")


def compute_latency(row, bandwidth_mbps: float, rtt_ms: float, repair_overhead_factor: float = 1.0):
    """Compute expected latency in seconds for one tick using the model.

    - bandwidth_mbps: link bandwidth in megabits/sec
    - rtt_ms: round-trip time in milliseconds
    - repair_overhead_factor: multiply repair payload by this factor (optional)
    """
    ndata = float(row["ndata"])
    p_desync = float(row["p_desync"]) if "p_desync" in row else 0.1
    p_err = float(row["p_err_emp"]) if "p_err_emp" in row else 0.0
    ver_time = float(row.get("mean_ver_time_s", 1e-6))

    bandwidth_bps = bandwidth_mbps * 1e6
    rtt_s = rtt_ms / 1000.0

    repair_payload_bits = ndata * 8.0 * repair_overhead_factor
    transfer_time = repair_payload_bits / bandwidth_bps
    repair_time = rtt_s + transfer_time

    # handle p_err very close to 1 to avoid division by zero
    if p_err >= 1.0:
        expected_repair_time = float('inf')
    else:
        expected_repair_time = repair_time / (1.0 - p_err)

    expected_latency = ver_time + p_desync * expected_repair_time
    return expected_latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bandwidth-mbps', type=float, default=10.0, help='Link bandwidth in Mbps')
    parser.add_argument('--rtt-ms', type=float, default=50.0, help='Round-trip time in ms')
    parser.add_argument('--out', type=str, default=OUT_PNG, help='Output PNG path')
    parser.add_argument('--csv', type=str, default=CSV_PATH, help='Input CSV with experiment results')
    args = parser.parse_args()

    # Ensure the CSV exists; provide a helpful message if missing
    if not os.path.exists(args.csv):
        print(f"Error: CSV not found at {args.csv}")
        print("Hint: generate results first or pass --csv to an existing file.")
        print(f"Expected default: {CSV_PATH}")
        raise SystemExit(1)

    df = pd.read_csv(args.csv)

    latencies = []
    for _, row in df.iterrows():
        lat_s = compute_latency(row, args.bandwidth_mbps, args.rtt_ms)
        lat_ms = lat_s * 1000.0
        latencies.append(lat_ms)

    df['latency_ms'] = latencies

    # Print a compact table
    print('system,nver,ndata,p_err_emp,mean_ver_time_s,normalized_traffic,latency_ms')
    for _, r in df.iterrows():
        print(f"{r['system']},{int(r['nver'])},{int(r['ndata'])},{r['p_err_emp']:.6g},{r['mean_ver_time_s']:.6g},{r['normalized_traffic']:.6g},{r['latency_ms']:.3f}")

    # Plot latency vs normalized traffic
    plt.figure(figsize=(8,6))
    markers = ['o','s','D','^','v','P','X','*']
    for i, r in df.iterrows():
        plt.scatter(r['normalized_traffic'], r['latency_ms'], s=100, marker=markers[i % len(markers)], label=f"{r['system']} ({int(r['nver'])},{int(r['ndata'])})")

    plt.xlabel('Normalized traffic (avg bits / ndata)')
    plt.ylabel('Expected latency (ms)')
    plt.title(f'Expected latency vs normalized traffic (bw={args.bandwidth_mbps} Mbps, RTT={args.rtt_ms} ms)')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize='small', loc='best')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print('Saved latency plot to', args.out)


if __name__ == '__main__':
    main()
