#!/usr/bin/env python3
"""Figure 5.5: Expected latency vs bandwidth for SHA256 backends and RS-ID.

Generates a two-panel plot (96 bits / 4001 bits) at n_ver=16 with three curves:
- RS-ID
- SHA256 (idcodes backend)  -> label "SHA256"
- SHA256 (hashlib backend)  -> label "SHA256(hashlib)"

Timing (t_ver) values are loaded from:
  experiments/results/sweep_nver_tver_detail_ci.csv

Output:
  thesis_report/figures/plots/latency_vs_bandwidth_backends_nver16.png
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_style import apply_thesis_style


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SWEEP_CSV = os.path.join(PROJECT_ROOT, "experiments", "results", "sweep_nver_tver_detail_ci.csv")
OUT_PATH = os.path.join(
    PROJECT_ROOT,
    "thesis_report",
    "figures",
    "plots",
    "latency_vs_bandwidth_backends_nver16.png",
)


def _get_tver_us(df: pd.DataFrame, *, payload_bits: int, nver_bits: int, col: str) -> float:
    row = df[(df["payload_bits"] == payload_bits) & (df["nver"] == nver_bits)]
    if row.empty:
        raise ValueError(f"Missing row for payload_bits={payload_bits}, nver={nver_bits} in {SWEEP_CSV}")
    return float(row.iloc[0][col])


def main() -> None:
    apply_thesis_style(base_fontsize=20)

    if not os.path.exists(SWEEP_CSV):
        raise FileNotFoundError(f"Sweep CSV not found: {SWEEP_CSV}")

    df = pd.read_csv(SWEEP_CSV)
    required = {
        "payload_bits",
        "nver",
        "sha_idcodes_mean_us",
        "sha_hashlib_mean_us",
        "rsid_mean_us",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Sweep CSV missing columns {sorted(missing)}: {SWEEP_CSV}")

    # Fixed configuration (matches Section 5.4 context)
    nver = 16
    p_desync = 0.1
    payloads = [96, 4001]

    # Bandwidth sweep
    B = np.logspace(4, 8, 200)  # bits/s

    # Styling
    colors = {
        "rsid": "#d62728",  # red
        "sha_idcodes": "#1f77b4",  # blue
        "sha_hashlib": "#1f77b4",  # same hue, different linestyle
    }
    styles = {
        "rsid": "-",
        "sha_idcodes": "-",
        "sha_hashlib": "--",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), sharey=True)

    for ax, ndata_bits in zip(axes, payloads, strict=True):
        t_rsid_s = _get_tver_us(df, payload_bits=ndata_bits, nver_bits=nver, col="rsid_mean_us") * 1e-6
        t_sha_idcodes_s = _get_tver_us(df, payload_bits=ndata_bits, nver_bits=nver, col="sha_idcodes_mean_us") * 1e-6
        t_sha_hashlib_s = _get_tver_us(df, payload_bits=ndata_bits, nver_bits=nver, col="sha_hashlib_mean_us") * 1e-6

        # Correctness-oriented expected latency model (consistent with Figs 5.2/5.3 updates):
        # L = t_ver + (n_ver + p_desync * n_data) / B
        n_ver_bits = float(nver)
        n_data_bits = float(ndata_bits)

        L_rsid = t_rsid_s + (n_ver_bits + p_desync * n_data_bits) / B
        L_sha_idcodes = t_sha_idcodes_s + (n_ver_bits + p_desync * n_data_bits) / B
        L_sha_hashlib = t_sha_hashlib_s + (n_ver_bits + p_desync * n_data_bits) / B

        ax.loglog(B, L_rsid, styles["rsid"], color=colors["rsid"], label="RS-ID")
        ax.loglog(B, L_sha_idcodes, styles["sha_idcodes"], color=colors["sha_idcodes"], label="SHA256")
        ax.loglog(B, L_sha_hashlib, styles["sha_hashlib"], color=colors["sha_hashlib"], label="SHA256(hashlib)")

        ax.set_xlabel("Bandwidth $B$ (bits/s)")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.set_title(f"$n_{{data}}={ndata_bits}$ bits")

    axes[0].set_ylabel("Expected latency (s)")

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.02))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT_PATH)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
