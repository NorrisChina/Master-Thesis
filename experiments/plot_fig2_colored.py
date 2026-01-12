import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS_CSV = os.path.join("experiments", "results", "fig2_results.csv")
OUT_PNG = os.path.join("experiments", "results", "fig2_colored.png")


def main():
    df = pd.read_csv(RESULTS_CSV)

    # Use a qualitative colormap with enough distinct colors
    cmap = plt.get_cmap("tab20")
    n = len(df)
    colors = [cmap(i % 20) for i in range(n)]

    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "8", "<", ">"]

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, row in df.iterrows():
        x = float(row["p_err_emp"])
        # map exact zeros to a small epsilon so they appear on the log x-axis
        display_x = x if x > 0 else 1e-6
        y = row["normalized_traffic"]
        label = f"{row['system']} ({int(row['nver'])},{int(row['ndata'])})"
        ax.scatter(display_x, y, color=colors[i], s=100, marker=markers[i % len(markers)], edgecolors='k', zorder=3)
        # offset text so labels don't sit directly on top of points
        text_x = display_x * 1.05
        ax.text(text_x, y, label, fontsize=9, va='center', color=colors[i])

    ax.set_xscale("log")
    ax.set_xlabel("Repair error probability (p_err)")
    ax.set_ylabel("Normalized traffic (avg bits / ndata)")
    ax.set_title("Fig.2 reproduction — colored points (one color per row)")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # tighten and save
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200)
    print(f"Saved colored Fig.2 to {OUT_PNG}")


if __name__ == '__main__':
    main()
