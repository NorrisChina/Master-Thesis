#!/usr/bin/env python3
"""Export a thesis-ready LiDAR t_ver table from the CI CSV.

This script formats the output of `experiments/measure_tver_ci_table.py` into the
LaTeX table consumed by the thesis (payload labels + thesis caption).

Example:
  /workspaces/Master-Thesis/.venv/bin/python experiments/export_tver_lidar_table.py \
    --csv experiments/results/tver_ci_table_nver16_lidar_ci5.csv \
    --out-tex thesis_report/tables/generated/tver_lidar_scalability_ci_nver16.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_CAPTION = (
    "Measured verifier generation time $t_{ver}$ for LiDAR-scale payload sizes ($n_{ver}=16$). "
    "Values are mean $\\pm$ 95\\% CI half-width; CI is controlled to $\\leq 5\\%$ relative error."
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export LiDAR-scale t_ver table (thesis formatting)")
    ap.add_argument("--csv", type=Path, required=True, help="Input CSV from measure_tver_ci_table.py")
    ap.add_argument("--out-tex", type=Path, required=True, help="Output .tex file")
    ap.add_argument("--caption", type=str, default=DEFAULT_CAPTION)
    ap.add_argument("--label", type=str, default="tab:tver_lidar_scalability")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Keep only the columns we need and enforce deterministic ordering.
    df = df[["n_data_bits", "n_data_bytes", "n_ver_bits", "sha_mean_us", "sha_ci95_half_us", "rsid_mean_us", "rsid_ci95_half_us"]]
    df = df.sort_values("n_data_bytes")

    # Map thesis payload labels.
    label_by_bits = {
        96: "12 B",
        4001: "501 B",
        819200: "100 KB",
        8388608: "1 MiB",
        41943040: "5 MiB",
    }

    tex_lines: list[str] = []
    tex_lines.append(f"% Auto-generated from {args.csv.name}")
    tex_lines.append("\\begin{table}[!t]")
    tex_lines.append("\\centering")
    tex_lines.append(f"\\caption{{{args.caption}}}")
    tex_lines.append(f"\\label{{{args.label}}}")
    tex_lines.append("\\begin{tabular}{l r r r}")
    tex_lines.append("\\toprule")
    tex_lines.append("Payload & $n_{data}$ (bits) & SHA-256 $t_{ver}$ ($\\mu s$) & RS-ID $t_{ver}$ ($\\mu s$) \\\\")
    tex_lines.append("\\midrule")

    for _, r in df.iterrows():
        ndata_bits = int(r["n_data_bits"])
        nver_bits = int(r["n_ver_bits"])
        if nver_bits != 16:
            # This exporter is thesis-specific; it is fine to refuse unexpected inputs.
            raise SystemExit(f"Expected n_ver_bits=16, got {nver_bits}")

        payload_label = label_by_bits.get(ndata_bits)
        if payload_label is None:
            # Fall back to bytes.
            payload_label = f"{int(r['n_data_bytes'])} B"

        sha_str = f"${float(r['sha_mean_us']):.2f} \\pm {float(r['sha_ci95_half_us']):.2f}$"
        rs_str = f"${float(r['rsid_mean_us']):.2f} \\pm {float(r['rsid_ci95_half_us']):.2f}$"

        tex_lines.append(f"{payload_label} & {ndata_bits} & {sha_str} & {rs_str} \\\\")

    tex_lines.append("\\bottomrule")
    tex_lines.append("\\end{tabular}")
    tex_lines.append("\\end{table}")
    tex_lines.append("")

    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.write_text("\n".join(tex_lines), encoding="utf-8")
    print(f"Wrote: {args.out_tex}")


if __name__ == "__main__":
    main()
