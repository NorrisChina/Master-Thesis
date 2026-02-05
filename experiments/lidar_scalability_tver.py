#!/usr/bin/env python3
"""LiDAR-scale t_ver scalability experiment.

Goal
----
Measure the verifier generation time (t_ver) as payload size increases to
sensor-like regimes (e.g., 10 KB, 100 KB).

We time the *encoder-side* tag generation using the same underlying ID systems
as the rest of the thesis codebase:
  - SHA256ID (truncated output via gf_exp)
  - RSID (Reed-Solomon identification)

Important modeling note
----------------------
For SHA256ID, the input is a byte-like message list of length n_data_bytes.
For concatenated RS-ID (RS2ID), we model packing into GF(2^{2*n_ver}) symbols (k_i=2):
    k = ceil((8*n_data_bytes)/(2*n_ver_bits))
So RS2ID processes k outer-field symbols (each 2*n_ver_bits).

Output
------
Writes a CSV with mean/CI times in microseconds:
  experiments/results/lidar_tver_scalability.csv

Example
-------
/workspaces/Master-Thesis/.venv/bin/python experiments/lidar_scalability_tver.py \
  --nver-bits 16 \
  --sizes-bytes 12 501 10240 102400 \
  --seed 123 \
  --out-csv experiments/results/lidar_tver_scalability.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Iterable


# Add the project src/ to sys.path (match other experiments)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from idsys.core.idsystems import create_id_system


@dataclass(frozen=True)
class TimingStats:
    mean_us: float
    std_us: float
    ci95_half_us: float
    num_batches: int
    total_iterations: int


def _ci95_half_width(samples: list[float]) -> float:
    if len(samples) < 2:
        return float("nan")
    z = 1.96
    return z * stdev(samples) / math.sqrt(len(samples))


def _pick_iterations(n_data_bytes: int, *, ref_bytes: int = 12, ref_iters: int = 200_000, min_iters: int = 50, max_iters: int = 200_000) -> int:
    """Heuristic: keep total processed bytes roughly constant across sizes."""
    if n_data_bytes <= 0:
        return min_iters
    iters = int(ref_iters * (ref_bytes / float(n_data_bytes)))
    return max(min_iters, min(max_iters, iters))


def _make_sha_message(n_data_bytes: int, rng: object) -> list[int]:
    # os.urandom is fast and reproducible is not critical for timing.
    return list(os.urandom(n_data_bytes))


def _make_rsid_symbols(n_data_bytes: int, n_ver_bits: int) -> list[int]:
    # Model packing of payload bits into GF(2^{2*n_ver}) symbols (k_i=2).
    symbol_bits = 2 * int(n_ver_bits)
    k = int(math.ceil((8.0 * n_data_bytes) / float(symbol_bits)))
    gf_range = 1 << symbol_bits

    bytes_per_symbol = int(math.ceil(symbol_bits / 8.0))
    raw = os.urandom(bytes_per_symbol * k)
    out: list[int] = []
    for i in range(k):
        chunk = raw[i * bytes_per_symbol : (i + 1) * bytes_per_symbol]
        out.append(int.from_bytes(chunk, byteorder="little", signed=False) % gf_range)
    return out


def _time_send(id_sys, message: list[int], *, iterations: int, num_batches: int, warmup: int | None = None) -> TimingStats:
    # Warm-up (adaptive to keep large payload runs practical)
    if warmup is None:
        warmup = int(max(10, min(200, iterations // 10)))
    for _ in range(warmup):
        id_sys.send(message)

    num_batches = int(max(5, min(num_batches, iterations)))
    base_batch = iterations // num_batches
    remainder = iterations % num_batches
    if base_batch == 0:
        base_batch = 1
        num_batches = iterations
        remainder = 0

    per_op_s_samples: list[float] = []
    total_done = 0
    for batch_idx in range(num_batches):
        batch_size = base_batch + (1 if batch_idx < remainder else 0)
        if batch_size <= 0:
            continue
        t0 = time.perf_counter()
        for _ in range(batch_size):
            id_sys.send(message)
        t1 = time.perf_counter()
        total_done += batch_size
        per_op_s_samples.append((t1 - t0) / batch_size)

    mu_s = mean(per_op_s_samples)
    s_s = stdev(per_op_s_samples) if len(per_op_s_samples) >= 2 else 0.0
    ci95_half_s = _ci95_half_width(per_op_s_samples)

    return TimingStats(
        mean_us=mu_s * 1e6,
        std_us=s_s * 1e6,
        ci95_half_us=ci95_half_s * 1e6,
        num_batches=len(per_op_s_samples),
        total_iterations=total_done,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="LiDAR-scale t_ver scalability experiment")
    p.add_argument("--nver-bits", type=int, default=16, help="Verifier size (bits); used as gf_exp and RSID symbol width")
    p.add_argument(
        "--sizes-bytes",
        type=int,
        nargs="+",
        default=[12, 501, 10_240, 102_400],
        help="Payload sizes (bytes). Defaults roughly match 96b, 4001b, 10KB, 100KB.",
    )
    p.add_argument("--seed", type=int, default=123, help="Seed (reserved; timing uses os.urandom)")
    p.add_argument("--batches", type=int, default=60, help="Number of timing batches for CI estimation")
    p.add_argument(
        "--ref-iters",
        type=int,
        default=200_000,
        help="Reference iterations for 12B payload; larger payloads scale iterations down",
    )
    p.add_argument("--min-iters", type=int, default=50, help="Minimum iterations per size")
    p.add_argument("--out-csv", type=str, default=os.path.join("experiments", "results", "lidar_tver_scalability.csv"))
    args = p.parse_args()

    out_csv = args.out_csv
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    nver_bits = int(args.nver_bits)

    # Construct systems once per scheme (we reuse per size).
    sha = create_id_system("SHA256ID", {"gf_exp": nver_bits})
    rsid = create_id_system("RS2ID", {"gf_exp": nver_bits, "tag_pos": [2], "tag_pos_in": [2]})

    rows: list[dict[str, object]] = []


    for n_data_bytes in args.sizes_bytes:
        iters = _pick_iterations(
            n_data_bytes,
            ref_bytes=12,
            ref_iters=int(args.ref_iters),
            min_iters=int(args.min_iters),
        )

        sha_msg = _make_sha_message(n_data_bytes, None)
        rs_msg = _make_rsid_symbols(n_data_bytes, nver_bits)

        # 统计 SHA256 分块数（每块 64 字节=512 bit）
        sha_block_size = 64
        sha_num_blocks = int((n_data_bytes + sha_block_size - 1) // sha_block_size)

        # 统计 RSID 符号数
        rsid_k = len(rs_msg)
        rsid_symbol_bits = 2 * nver_bits
        rsid_symbol_bytes = (rsid_symbol_bits + 7) // 8

        sha_stats = _time_send(sha, sha_msg, iterations=iters, num_batches=int(args.batches))
        rs_stats = _time_send(rsid, rs_msg, iterations=iters, num_batches=int(args.batches))

        sha_per_block_us = sha_stats.mean_us / sha_num_blocks if sha_num_blocks > 0 else float('nan')
        rsid_per_k_us = rs_stats.mean_us / rsid_k if rsid_k > 0 else float('nan')

        rows.append(
            {
                "n_ver_bits": nver_bits,
                "n_data_bytes": n_data_bytes,
                "n_data_bits": 8 * n_data_bytes,
                "sha_num_blocks": sha_num_blocks,
                "sha_block_size_bytes": sha_block_size,
                "rsid_k_symbols": rsid_k,
                "rsid_symbol_bits": rsid_symbol_bits,
                "rsid_symbol_bytes": rsid_symbol_bytes,
                "iters": iters,
                "sha_mean_us": sha_stats.mean_us,
                "sha_ci95_half_us": sha_stats.ci95_half_us,
                "rsid_mean_us": rs_stats.mean_us,
                "rsid_ci95_half_us": rs_stats.ci95_half_us,
                "sha_per_block_us": sha_per_block_us,
                "rsid_per_k_us": rsid_per_k_us,
            }
        )

        print(
            f"size={n_data_bytes:>7} B | iters={iters:>6} | "
            f"SHA={sha_stats.mean_us:>8.2f}±{sha_stats.ci95_half_us:.2f} µs | "
            f"RSID={rs_stats.mean_us:>8.2f}±{rs_stats.ci95_half_us:.2f} µs | "
            f"SHA_blocks={sha_num_blocks}({sha_per_block_us:.3f} µs/block) | "
            f"RSID_k={rsid_k}({rsid_per_k_us:.3f} µs/symbol) (symbol {rsid_symbol_bits}b/{rsid_symbol_bytes}B)"
        )

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "n_ver_bits",
                "n_data_bytes",
                "n_data_bits",
                "sha_num_blocks",
                "sha_block_size_bytes",
                "rsid_k_symbols",
                "rsid_symbol_bits",
                "rsid_symbol_bytes",
                "iters",
                "sha_mean_us",
                "sha_ci95_half_us",
                "rsid_mean_us",
                "rsid_ci95_half_us",
                "sha_per_block_us",
                "rsid_per_k_us",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote: {out_csv}")


if __name__ == "__main__":
    main()
