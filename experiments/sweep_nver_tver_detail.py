#!/usr/bin/env python3
"""
Sweep nver: Analyze t_ver, k, and per-symbol/block timing for RSID and SHA256 at various verifier lengths.
"""
import os
import numpy as np
import pandas as pd
from latency_vs_ndata_fixed_nver import p_miss_sha, p_miss_rsid_concat
from idsys.core.idsystems import create_id_system
import time
import csv

# Parameters
payload_bits_list = [96, 4001]
nver_bits_list = [4, 8, 12, 16]
sha_block_size = 64  # bytes

results = []

for nver in nver_bits_list:
    for payload_bits in payload_bits_list:
        n_data_bytes = (payload_bits + 7) // 8
        # SHA256
        sha = create_id_system("SHA256ID", {"gf_exp": nver})
        sha_msg = list(os.urandom(n_data_bytes))
        sha_num_blocks = (n_data_bytes + sha_block_size - 1) // sha_block_size
        # 计时
        sha_iters = 100000 if n_data_bytes < 100 else 10000
        t0 = time.perf_counter()
        for _ in range(sha_iters):
            sha.send(sha_msg)
        t1 = time.perf_counter()
        sha_total_us = (t1 - t0) * 1e6 / sha_iters
        sha_per_block_us = sha_total_us / sha_num_blocks if sha_num_blocks > 0 else float('nan')

        # RSID
        rsid = create_id_system("RS2ID", {"gf_exp": nver, "tag_pos": [2], "tag_pos_in": [2]})
        symbol_bits = 2 * nver
        k = int(np.ceil(8.0 * n_data_bytes / symbol_bits))
        rsid_symbol_bytes = (symbol_bits + 7) // 8
        raw = os.urandom(rsid_symbol_bytes * k)
        rs_msg = [int.from_bytes(raw[i*rsid_symbol_bytes:(i+1)*rsid_symbol_bytes], byteorder="little", signed=False) % (1 << symbol_bits) for i in range(k)]
        rsid_iters = 100000 if n_data_bytes < 100 else 10000
        t0 = time.perf_counter()
        for _ in range(rsid_iters):
            rsid.send(rs_msg)
        t1 = time.perf_counter()
        rsid_total_us = (t1 - t0) * 1e6 / rsid_iters
        rsid_per_k_us = rsid_total_us / k if k > 0 else float('nan')

        results.append({
            "nver": nver,
            "payload_bits": payload_bits,
            "n_data_bytes": n_data_bytes,
            "sha_num_blocks": sha_num_blocks,
            "sha_total_us": sha_total_us,
            "sha_per_block_us": sha_per_block_us,
            "rsid_k": k,
            "rsid_symbol_bits": symbol_bits,
            "rsid_total_us": rsid_total_us,
            "rsid_per_k_us": rsid_per_k_us,
        })
        print(f"nver={nver:2d}, payload={payload_bits:5d}b | SHA: {sha_total_us:.2f} us ({sha_per_block_us:.3f} us/block, {sha_num_blocks} blocks) | RSID: {rsid_total_us:.2f} us ({rsid_per_k_us:.3f} us/symbol, k={k})")

# Save to CSV
out_csv = "experiments/results/sweep_nver_tver_detail.csv"
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)
print(f"\nWrote: {out_csv}")
