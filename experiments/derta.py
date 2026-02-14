#!/usr/bin/env python3
"""Generate the decoder-cost figure used in the thesis.

Writes by default to `thesis_report/figures/decoder_cost_multiline_annotated.png`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
        repo_root = Path(__file__).resolve().parents[1]

        ap = argparse.ArgumentParser(description="Plot decoder compute latency vs quantization resolution")
        ap.add_argument(
                "--out",
                type=Path,
                default=repo_root / "thesis_report" / "figures" / "decoder_cost_multiline_annotated.png",
                help="Output PNG path",
        )
        ap.add_argument("--t-hash-us", type=float, default=2.13, help="Per-hash compute time (microseconds)")
        ap.add_argument("--deltas", type=int, default=100, help="Number of delta samples")
        ap.add_argument("--show", action="store_true", help="Show the figure interactively")
        args = ap.parse_args()

        radii = [2, 10, 30]
        deltas = np.logspace(0, -2, int(args.deltas))
        t_hash_us = float(args.t_hash_us)

        fig, ax = plt.subplots(figsize=(7, 5))
        font_size = 12
        plt.rc("font", size=font_size)
        plt.rc("axes", titlesize=font_size)
        plt.rc("axes", labelsize=font_size)
        plt.rc("xtick", labelsize=font_size)
        plt.rc("ytick", labelsize=font_size)
        plt.rc("legend", fontsize=font_size)
        plt.rc("figure", titlesize=font_size)

        colors = ["tab:green", "tab:orange", "tab:red"]

        for i, radius in enumerate(radii):
                set_sizes = np.ceil((2 * float(radius)) / deltas) + 1
                latency_ms = (set_sizes * t_hash_us) / 1000.0
                ax.plot(deltas, latency_ms, lw=2.5, color=colors[i], label=rf"$R={radius}$")

                cost_fine = float(latency_ms[-1])
                ax.text(
                        0.01,
                        cost_fine,
                        f"{cost_fine:.1f}",
                        fontsize=font_size,
                        color=colors[i],
                        fontweight="bold",
                        ha="left",
                        va="center",
                )

                idx_01 = int(np.abs(deltas - 0.1).argmin())
                cost_01 = float(latency_ms[idx_01])
                ax.plot(0.1, cost_01, "o", color=colors[i], markersize=8, markeredgecolor="white")
                offset_y = -0.5 if radius == 2 else 0.2
                ax.text(
                        0.1,
                        cost_01 + float(offset_y),
                        f"{cost_01:.1f}",
                        fontsize=font_size,
                        color=colors[i],
                        fontweight="bold",
                        ha="center",
                )

        ax.axvline(0.1, color="gray", linestyle=":", linewidth=1.5, alpha=0.8)
        ax.text(
                0.1,
                ax.get_ylim()[1] * 0.85,
                "Selected\nOperating Point\n($\\Delta=0.1$)",
                color="gray",
                ha="center",
                fontsize=font_size,
                backgroundcolor="white",
        )

        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax.text(0.5, 1.1, "Soft Real-time Limit (~1ms)", color="gray", fontsize=font_size)

        ax.set_xscale("log")
        ax.invert_xaxis()

        ax.set_xlabel(r"Quantization Resolution $\Delta$ (Log Scale)", fontsize=font_size)
        ax.set_ylabel("Decoder Compute Latency (ms)", fontsize=font_size)
        ax.set_title("Computational Scalability: The Cost of Safety", fontsize=font_size)
        ax.tick_params(axis="both", which="major", labelsize=font_size)
        ax.tick_params(axis="both", which="minor", labelsize=font_size)
        ax.legend(loc="upper left", fontsize=font_size, frameon=True)
        ax.grid(True, which="both", ls="-", alpha=0.3)

        plt.tight_layout()
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
        if bool(args.show):
                plt.show()
        plt.close(fig)
        print(f"Wrote: {out_path}")
        return 0


if __name__ == "__main__":
        raise SystemExit(main())