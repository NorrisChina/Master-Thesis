#!/usr/bin/env python3
"""Generate a standalone t_ver CI table for selected (n_data, n_ver).

This is intended for reporting empirical verifier-generation time t_ver with a
95% CI, aligned with the latency model experiments.

- SHA256ID: message is a byte list of length ceil(n_data_bits/8)
- Concatenated RS-ID (RS2ID): message is modeled as packed GF(2^{2*n_ver}) symbols (k_i=2)

Outputs:
  - CSV (default): experiments/results/tver_ci_table.csv
  - Optional LaTeX table (default): experiments/results/tver_ci_table.tex

Example:
  /workspaces/Master-Thesis/.venv/bin/python experiments/measure_tver_ci_table.py \
    --nver-bits 16 --ndata-bits 96 4001 819200 \
    --B-fixed 5e6 --p-desync 0.1 \
    --out-csv experiments/results/tver_ci_table_nver16_96_4001_100KB.csv \
    --out-tex experiments/results/tver_ci_table_nver16_96_4001_100KB.tex
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
from typing import Literal


# Add src/ to path (match other experiments)
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


def _make_sha_message(n_data_bits: int) -> list[int]:
    n_data_bytes = (int(n_data_bits) + 7) // 8
    return list(os.urandom(n_data_bytes))


def _make_rsid_message(n_data_bits: int, n_ver_bits: int) -> list[int]:
    """Model packing for concatenated RS-ID (RS2ID).

    We treat the payload as packed symbols over GF(2^{2*n_ver}) (k_i=2), so the
    number of outer symbols scales as k = ceil(n_data_bits / (2*n_ver_bits)).
    """
    n_data_bits_i = int(n_data_bits)
    n_ver_bits_i = int(n_ver_bits)
    symbol_bits = 2 * n_ver_bits_i
    k = int(math.ceil(n_data_bits_i / float(symbol_bits)))
    gf_range = 1 << symbol_bits

    bytes_per_symbol = int(math.ceil(symbol_bits / 8.0))
    raw = os.urandom(bytes_per_symbol * k)
    out: list[int] = []
    for i in range(k):
        chunk = raw[i * bytes_per_symbol : (i + 1) * bytes_per_symbol]
        out.append(int.from_bytes(chunk, byteorder="little", signed=False) % gf_range)
    return out


def _time_send(id_sys, message: list[int], *, iterations: int, num_batches: int, warmup: int | None = None) -> TimingStats:
    # Warm-up: keep large-payload runs practical.
    if warmup is None:
        warmup = int(max(5, min(50, iterations // 10)))
    for _ in range(warmup):
        id_sys.send(message)

    num_batches = int(max(5, min(num_batches, iterations)))
    # Ensure each batch has enough work to reduce timer/OS jitter impact.
    # (Important for large payloads where iterations is small.)
    min_batch_ops = 5
    num_batches = min(num_batches, max(5, iterations // min_batch_ops))
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


def _pick_iterations(n_data_bits: int, *, ref_bits: int = 96, ref_iters: int = 200_000, min_iters: int = 50, max_iters: int = 200_000) -> int:
    if n_data_bits <= 0:
        return min_iters
    iters = int(ref_iters * (ref_bits / float(n_data_bits)))
    return max(min_iters, min(max_iters, iters))


def p_miss_sha(nver_bits: int) -> float:
    return 2.0 ** (-int(nver_bits))


def p_miss_rsid_concat(nver_bits: int) -> float:
    # Concatenated RS-ID bound is dominated by 1/q with q=2^{n_ver}.
    return 2.0 ** (-int(nver_bits))


def expected_latency(B: float, n_data_bits: int, n_ver_bits: int, t_ver_s: float, p_desync: float, p_miss: float) -> float:
    return t_ver_s + (n_ver_bits + p_desync * (1.0 - p_miss) * n_data_bits) / B


def main() -> None:
    p = argparse.ArgumentParser(description="Measure t_ver with CI for selected parameters")
    p.add_argument("--nver-bits", type=int, default=16)
    p.add_argument("--ndata-bits", type=int, nargs="+", default=[96, 4001, 819200])
    p.add_argument("--batches", type=int, default=60)
    p.add_argument("--ref-iters", type=int, default=200000)
    p.add_argument("--min-iters", type=int, default=30)
    p.add_argument("--max-iters", type=int, default=200000)

    # Optional: also compute the implied latency numbers for a fixed (B, p_desync)
    p.add_argument("--B-fixed", type=float, default=5e6)
    p.add_argument("--p-desync", type=float, default=0.1)

    p.add_argument("--out-csv", type=str, default=os.path.join("experiments", "results", "tver_ci_table.csv"))
    p.add_argument("--out-tex", type=str, default=os.path.join("experiments", "results", "tver_ci_table.tex"))
    args = p.parse_args()

    nver = int(args.nver_bits)
    B = float(args.B_fixed)
    p_desync = float(args.p_desync)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    sha = create_id_system("SHA256ID", {"gf_exp": nver})
    rsid = create_id_system("RS2ID", {"gf_exp": nver, "tag_pos": [2], "tag_pos_in": [2]})

    rows: list[dict[str, object]] = []

    for ndata_bits in [int(x) for x in args.ndata_bits]:
        iters = _pick_iterations(
            ndata_bits,
            ref_bits=96,
            ref_iters=int(args.ref_iters),
            min_iters=int(args.min_iters),
            max_iters=int(args.max_iters),
        )

        sha_msg = _make_sha_message(ndata_bits)
        rs_msg = _make_rsid_message(ndata_bits, nver)

        sha_stats = _time_send(sha, sha_msg, iterations=iters, num_batches=int(args.batches))
        rs_stats = _time_send(rsid, rs_msg, iterations=iters, num_batches=int(args.batches))

        psha = p_miss_sha(nver)
        prs = p_miss_rsid_concat(nver)

        L_sha = expected_latency(B, ndata_bits, nver, sha_stats.mean_us * 1e-6, p_desync, psha)
        L_rs = expected_latency(B, ndata_bits, nver, rs_stats.mean_us * 1e-6, p_desync, prs)
        L_trad = ndata_bits / B

        rows.append(
            {
                "n_ver_bits": nver,
                "n_data_bits": ndata_bits,
                "n_data_bytes": (ndata_bits + 7) // 8,
                "rsid_k_symbols": len(rs_msg),
                "iters": iters,
                "sha_mean_us": sha_stats.mean_us,
                "sha_ci95_half_us": sha_stats.ci95_half_us,
                "rsid_mean_us": rs_stats.mean_us,
                "rsid_ci95_half_us": rs_stats.ci95_half_us,
                "B_bits_per_s": B,
                "p_desync": p_desync,
                "p_miss_sha": psha,
                "p_miss_rsid": prs,
                "L_trad_s": L_trad,
                "L_id_sha_s": L_sha,
                "L_id_rsid_s": L_rs,
            }
        )

        print(
            f"ndata={ndata_bits}b | iters={iters} | "
            f"SHA t_ver={sha_stats.mean_us:.2f}±{sha_stats.ci95_half_us:.2f} µs | "
            f"RSID t_ver={rs_stats.mean_us:.2f}±{rs_stats.ci95_half_us:.2f} µs"
        )

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Simple LaTeX table focusing on t_ver and CI.
    tex_lines = []
    tex_lines.append("% Auto-generated by experiments/measure_tver_ci_table.py")
    tex_lines.append("\\begin{table}[!t]")
    tex_lines.append("\\centering")
    tex_lines.append("\\caption{Empirical verifier generation time $t_{ver}$ (mean $\\pm$ 95\\% CI half-width).}")
    tex_lines.append("\\label{tab:tver_ci_selected}")
    tex_lines.append("\\begin{tabular}{r r r r}")
    tex_lines.append("\\toprule")
    tex_lines.append("$n_{data}$ (bits) & $n_{ver}$ (bits) & SHA-256 $t_{ver}$ ($\\mu s$) & RS-ID $t_{ver}$ ($\\mu s$) \\\\")
    tex_lines.append("\\midrule")
    for r in rows:
        ndata = int(r["n_data_bits"])
        nver_bits = int(r["n_ver_bits"])
        sha_str = f"{float(r['sha_mean_us']):.2f} \\pm {float(r['sha_ci95_half_us']):.2f}"
        rs_str = f"{float(r['rsid_mean_us']):.2f} \\pm {float(r['rsid_ci95_half_us']):.2f}"
        tex_lines.append(f"{ndata} & {nver_bits} & {sha_str} & {rs_str} \\")
    tex_lines.append("\\bottomrule")
    tex_lines.append("\\end{tabular}")
    tex_lines.append("\\end{table}")

    with open(args.out_tex, "w") as f:
        f.write("\n".join(tex_lines) + "\n")

    print(f"Wrote CSV: {args.out_csv}")
    print(f"Wrote TeX: {args.out_tex}")


if __name__ == "__main__":
    main()
