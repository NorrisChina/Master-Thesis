import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the sweep results
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
csv_path = os.path.join(_project_root, "experiments", "results", "sweep_nver_tver_detail_ci.csv")
df = pd.read_csv(csv_path)

# Plot settings
plt.figure(figsize=(6.5, 4))
systems = [
    ("sha_idcodes", "sha256", "sha_idcodes_mean_us", "sha_idcodes_ci95_half_us", "solid"),
    ("sha_hashlib", "sha256", "sha_hashlib_mean_us", "sha_hashlib_ci95_half_us", "dashed"),
    ("rsid", "RS-ID", "rsid_mean_us", "rsid_ci95_half_us"),
]

markers = {
    ("sha256", 96): "o",
    ("sha256", 4001): "s",
    ("RS-ID", 96): "D",
    ("RS-ID", 4001): "^",
}
colors = {
    ("sha256", 96): "#1f77b4",   # blue
    ("sha256", 4001): "#aec7e8", # light blue
    ("RS-ID", 96): "#d62728",               # red
    ("RS-ID", 4001): "#ff9896",             # light red
}
linestyles = {
    "sha256_solid": "-",
    "sha256_dashed": "--",
    "RS-ID": "-",
}

# Larger axis/tick fonts for thesis readability.
label_fontsize = 16
tick_fontsize = 14
legend_fontsize = 10

# Plot each curve

for sys in systems:
    if len(sys) == 4:
        _, sys_label, y_col, yerr_col = sys
        style_tag = None
    else:
        _, sys_label, y_col, yerr_col, style_tag = sys

    for payload in [96, 4001]:
        mask = (df["payload_bits"] == payload)
        y = df.loc[mask, y_col]
        yerr = df.loc[mask, yerr_col]
        x = df.loc[mask, "nver"]

        if sys_label == "sha256":
            linestyle = linestyles["sha256_solid" if style_tag == "solid" else "sha256_dashed"]
            if style_tag == "solid":
                label = f"SHA256, $n_{{data}}={payload}$"
            else:
                label = f"SHA256 (hashlib), $n_{{data}}={payload}$"
        else:
            linestyle = linestyles.get(sys_label, "-")
            label = f"{sys_label}, $n_{{data}}={payload}$"

        plt.errorbar(
            x,
            y,
            yerr=yerr,
            label=label,
            marker=markers[(sys_label, payload)],
            color=colors[(sys_label, payload)],
            linestyle=linestyle,
            capsize=3,
        )

plt.xlabel("Verifier Length $n_{ver}$ (bits)", fontsize=label_fontsize)
plt.ylabel(r"Encoding Time $t_{ver}$ ($\mu$s)", fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
leg = plt.legend(
    fontsize=legend_fontsize,
    loc="upper right",
    ncol=1,
    frameon=True,
    framealpha=0.65,
    borderpad=0.4,
    handlelength=2.2,
    labelspacing=0.3,
)
leg.get_frame().set_linewidth(0.0)
leg.get_frame().set_edgecolor("none")
leg.get_frame().set_facecolor("white")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout(pad=0.6)
out_path = os.path.join(_project_root, "thesis_report", "figures", "plots", "tver_sweep_summary.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(
    out_path,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.12,
)
plt.close()
