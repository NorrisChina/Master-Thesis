#!/usr/bin/env python3
"""Generate the error-model comparison figure used in the thesis.

Outputs by default to `thesis_report/figures/error_models_sigma16.png`.

This script is intentionally self-contained (no SciPy dependency) so it can be
run in a minimal Python environment with NumPy + Matplotlib.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _gaussian_pdf(x: np.ndarray, *, mu: float, sigma: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    z = (x - float(mu)) / sigma
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * z * z)


def _uniform_pdf(x: np.ndarray, *, low: float, high: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    low = float(low)
    high = float(high)
    if high <= low:
        raise ValueError("high must be > low")
    y = np.zeros_like(x, dtype=float)
    y[(x >= low) & (x <= high)] = 1.0 / (high - low)
    return y


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser(description="Plot Gaussian vs variance-matched Uniform error model")
    ap.add_argument(
        "--out",
        type=Path,
        default=repo_root / "thesis_report" / "figures" / "error_models_sigma16.png",
        help="Output PNG path",
    )
    ap.add_argument("--mu", type=float, default=50.0)
    ap.add_argument("--sigma", type=float, default=16.0)
    ap.add_argument("--xmin", type=float, default=0.0)
    ap.add_argument("--xmax", type=float, default=100.0)
    ap.add_argument("--show", action="store_true", help="Show the figure interactively")
    args = ap.parse_args()

    mu = float(args.mu)
    sigma_gaussian = float(args.sigma)

    # Var(U) = a^2 / 3 = sigma^2  => a = sigma * sqrt(3)
    a_uniform = sigma_gaussian * np.sqrt(3.0)

    x_min, x_max = float(args.xmin), float(args.xmax)
    x_range = np.linspace(x_min, x_max, 2000)

    pdf_gaussian = _gaussian_pdf(x_range, mu=mu, sigma=sigma_gaussian)

    uniform_lower = mu - a_uniform
    uniform_upper = mu + a_uniform
    pdf_uniform = _uniform_pdf(x_range, low=uniform_lower, high=uniform_upper)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

# --- 左图: Gaussian ---
    ax0 = axes[0]
    ax0.plot(x_range, pdf_gaussian, "b-", lw=2.5, label=rf"$\sigma={int(sigma_gaussian)}$")
    ax0.fill_between(x_range, pdf_gaussian, color="blue", alpha=0.15)

# 标注物理边界 (0 和 100)
    ax0.axvline(x_min, color="k", lw=3, linestyle="-", alpha=0.3)
    ax0.axvline(x_max, color="k", lw=3, linestyle="-", alpha=0.3)
    ax0.text(x_min + 1, float(np.max(pdf_gaussian)) * 0.9, f"Wall ({x_min:g})", rotation=90, verticalalignment="top", alpha=0.5)
    ax0.text(x_max - 3, float(np.max(pdf_gaussian)) * 0.9, f"Wall ({x_max:g})", rotation=90, verticalalignment="top", alpha=0.5)

# 标注 3-sigma 范围
    sigma3_lower = mu - 3 * sigma_gaussian
    sigma3_upper = mu + 3 * sigma_gaussian
    ax0.axvline(sigma3_lower, color="r", linestyle=":", alpha=0.8, lw=1.5)
    ax0.axvline(sigma3_upper, color="r", linestyle=":", alpha=0.8, lw=1.5)

    LARGE = 20
    MEDIUM = 20
    SMALL = 20

    ax0.text(
        sigma3_lower + 2,
        float(np.max(pdf_gaussian)) * 0.55,
        rf"$-3\sigma$ ({sigma3_lower:.1f})",
        color="r",
        ha="left",
        fontsize=SMALL,
        fontweight="bold",
    )
    ax0.text(
        sigma3_upper - 2,
        float(np.max(pdf_gaussian)) * 0.55,
        rf"$+3\sigma$ ({sigma3_upper:.1f})",
        color="r",
        ha="right",
        fontsize=SMALL,
        fontweight="bold",
    )

    ax0.set_title("(a) Baseline Gaussian Model", fontsize=LARGE)
    ax0.set_xlabel(r"Robot State $S_t$", fontsize=MEDIUM)
    ax0.set_ylabel("Probability Density", fontsize=MEDIUM)
    ax0.legend(loc="upper right", fontsize=MEDIUM, frameon=True)
    ax0.grid(True, alpha=0.3)
    ax0.set_xlim(x_min, x_max)
    ax0.tick_params(axis="both", which="major", labelsize=MEDIUM)

# --- 右图: Uniform ---
    ax1 = axes[1]
    ax1.plot(x_range, pdf_uniform, "g-", lw=2.5, label=rf"$a={a_uniform:.1f}$")
    ax1.fill_between(x_range, pdf_uniform, color="green", alpha=0.15)

# 标注物理边界
    ax1.axvline(x_min, color="k", lw=3, linestyle="-", alpha=0.3)
    ax1.axvline(x_max, color="k", lw=3, linestyle="-", alpha=0.3)


# 标注 Uniform 边界
    ax1.axvline(uniform_lower, color="g", linestyle="--", lw=2)
    ax1.axvline(uniform_upper, color="g", linestyle="--", lw=2)
    ax1.text(
        uniform_lower,
        float(np.max(pdf_uniform)) * 0.05,
        f"{uniform_lower:.1f}",
        color="g",
        ha="left",
        va="bottom",
        fontsize=SMALL,
        fontweight="bold",
    )
    ax1.text(
        uniform_upper,
        float(np.max(pdf_uniform)) * 0.05,
        f"{uniform_upper:.1f}",
        color="g",
        ha="right",
        va="bottom",
        fontsize=SMALL,
        fontweight="bold",
    )

    ax1.set_title("(b) Variance-Matched Uniform Model", fontsize=LARGE)
    ax1.set_xlabel(r"Robot State $S_t$", fontsize=MEDIUM)
    ax1.legend(loc="upper right", fontsize=MEDIUM, frameon=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(x_min, x_max)
    ax1.tick_params(axis="both", which="major", labelsize=MEDIUM)


    plt.suptitle(
        rf"Error Model Comparison: Optimal Space Utilization ($\sigma={int(sigma_gaussian)}$, $\sigma^2={int(sigma_gaussian**2)}$)",
        fontsize=LARGE,
        y=1.05,
    )

    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if bool(args.show):
        plt.show()
    plt.close(fig)
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())