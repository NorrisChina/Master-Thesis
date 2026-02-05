#!/usr/bin/env python3
"""Sweep n_ver: analyze t_ver, k, and per-symbol/block timing for RS-ID and SHA-256.

This script writes a CSV consumed by plotting scripts in `experiments/`.
It is designed to be runnable from any working directory.
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure local imports work regardless of CWD.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_src_path = os.path.join(_project_root, "src")
_experiments_path = os.path.join(_project_root, "experiments")
for _p in (_project_root, _src_path, _experiments_path):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from latency_vs_ndata_fixed_nver import p_miss_sha, p_miss_rsid_concat
from idsys.core.idsystems import create_id_system
import time
import csv
import math

# Parameters
payload_bits_list = [96, 4001]
nver_bits_list = [4, 8, 12, 16]
sha_block_size = 64  # bytes

results = []

# 95% CI helper
Z = 1.96
def ci95(samples):
    if len(samples) < 2:
        return float('nan')
    return Z * np.std(samples, ddof=1) / np.sqrt(len(samples))

for nver in nver_bits_list:
    for payload_bits in payload_bits_list:
        n_data_bytes = (payload_bits + 7) // 8
        # SHA256 (OpenSSL-backed hashlib)
        sha_hashlib = create_id_system("SHA256ID", {"gf_exp": nver, "backend": "hashlib"})
        # SHA256 (idcodes backend)
        sha_idcodes = create_id_system("SHA256ID", {"gf_exp": nver, "backend": "idcodes"})
        sha_msg = list(os.urandom(n_data_bytes))
        sha_num_blocks = (n_data_bytes + sha_block_size - 1) // sha_block_size
        sha_iters = 100000 if n_data_bytes < 100 else 10000
        sha_hashlib_samples = []
        for _ in range(50):
            t0 = time.perf_counter()
            for _ in range(sha_iters // 50):
                sha_hashlib.send(sha_msg)
            t1 = time.perf_counter()
            sha_hashlib_samples.append((t1 - t0) * 1e6 / (sha_iters // 50))

        sha_idcodes_samples = []
        for _ in range(50):
            t0 = time.perf_counter()
            for _ in range(sha_iters // 50):
                sha_idcodes.send(sha_msg)
            t1 = time.perf_counter()
            sha_idcodes_samples.append((t1 - t0) * 1e6 / (sha_iters // 50))

        sha_hashlib_mean = np.mean(sha_hashlib_samples)
        sha_hashlib_ci = ci95(sha_hashlib_samples)
        sha_hashlib_per_block_us = sha_hashlib_mean / sha_num_blocks if sha_num_blocks > 0 else float('nan')

        sha_idcodes_mean = np.mean(sha_idcodes_samples)
        sha_idcodes_ci = ci95(sha_idcodes_samples)
        sha_idcodes_per_block_us = sha_idcodes_mean / sha_num_blocks if sha_num_blocks > 0 else float('nan')

        # RSID
        rsid = create_id_system("RS2ID", {"gf_exp": nver, "tag_pos": [2], "tag_pos_in": [2]})
        symbol_bits = 2 * nver
        k = int(np.ceil(8.0 * n_data_bytes / symbol_bits))
        rsid_symbol_bytes = (symbol_bits + 7) // 8
        raw = os.urandom(rsid_symbol_bytes * k)
        rs_msg = [int.from_bytes(raw[i*rsid_symbol_bytes:(i+1)*rsid_symbol_bytes], byteorder="little", signed=False) % (1 << symbol_bits) for i in range(k)]
        rsid_iters = 100000 if n_data_bytes < 100 else 10000
        rsid_samples = []
        for _ in range(50):
            t0 = time.perf_counter()
            for _ in range(rsid_iters // 50):
                rsid.send(rs_msg)
            t1 = time.perf_counter()
            rsid_samples.append((t1 - t0) * 1e6 / (rsid_iters // 50))
        rsid_mean = np.mean(rsid_samples)
        rsid_ci = ci95(rsid_samples)
        rsid_per_k_us = rsid_mean / k if k > 0 else float('nan')

        results.append({
            "nver": nver,
            "payload_bits": payload_bits,
            "n_data_bytes": n_data_bytes,
            "sha_num_blocks": sha_num_blocks,
            # Backwards-compatible aliases: SHA-256 refers to the hashlib baseline.
            "sha_mean_us": sha_hashlib_mean,
            "sha_ci95_half_us": sha_hashlib_ci,
            "sha_per_block_us": sha_hashlib_per_block_us,
            # Explicit backends.
            "sha_hashlib_mean_us": sha_hashlib_mean,
            "sha_hashlib_ci95_half_us": sha_hashlib_ci,
            "sha_hashlib_per_block_us": sha_hashlib_per_block_us,
            "sha_idcodes_mean_us": sha_idcodes_mean,
            "sha_idcodes_ci95_half_us": sha_idcodes_ci,
            "sha_idcodes_per_block_us": sha_idcodes_per_block_us,
            "rsid_k": k,
            "rsid_symbol_bits": symbol_bits,
            "rsid_mean_us": rsid_mean,
            "rsid_ci95_half_us": rsid_ci,
            "rsid_per_k_us": rsid_per_k_us,
        })
        print(
            f"nver={nver:2d}, payload={payload_bits:5d}b | "
            f"SHA(hashlib): {sha_hashlib_mean:.2f}±{sha_hashlib_ci:.2f} us ({sha_hashlib_per_block_us:.3f} us/block, {sha_num_blocks} blocks) | "
            f"SHA(idcodes): {sha_idcodes_mean:.2f}±{sha_idcodes_ci:.2f} us ({sha_idcodes_per_block_us:.3f} us/block) | "
            f"RSID: {rsid_mean:.2f}±{rsid_ci:.2f} us ({rsid_per_k_us:.3f} us/symbol, k={k})"
        )

# Save to CSV
out_csv = os.path.join(_project_root, "experiments", "results", "sweep_nver_tver_detail_ci.csv")
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)
print(f"\nWrote: {out_csv}")
