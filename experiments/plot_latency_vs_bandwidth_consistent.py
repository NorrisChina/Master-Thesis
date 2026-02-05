import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plot_style import apply_thesis_style

# Load the sweep results for color/legend consistency
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sweep_csv = os.path.join(_project_root, "experiments", "results", "sweep_nver_tver_detail_ci.csv")
sweep_df = pd.read_csv(sweep_csv)

# Color and label mapping (same as tver_sweep_summary)
COLORS = {
    ("SHA-256", 4): "#1f77b4",    # deep blue
    ("SHA-256", 16): "#aec7e8",   # light blue
    ("RS-ID", 4): "#d62728",      # deep red
    ("RS-ID", 16): "#ff9896",     # light red
    "Traditional": "#222222"
}
LABELS = {
    ("SHA-256", 4): "SHA-256 $n_{ver}=4$",
    ("SHA-256", 16): "SHA-256 $n_{ver}=16$",
    ("RS-ID", 4): "RS-ID $n_{ver}=4$",
    ("RS-ID", 16): "RS-ID $n_{ver}=16$",
    "Traditional": "Traditional"
}
MARKERS = {
    ("SHA-256", 4): "o",
    ("SHA-256", 16): "s",
    ("RS-ID", 4): "D",
    ("RS-ID", 16): "^",
    "Traditional": None
}

# Load latency vs bandwidth data (reuse script logic)
def plot_latency_vs_bandwidth():
    apply_thesis_style(base_fontsize=20)
    B = np.logspace(4, 8, 200)
    nver_list = [4, 16]
    ndata_list = [96, 4001]
    p_desync = 0.1
    for ndata in ndata_list:
        fig, ax = plt.subplots(figsize=(6, 5))
        # Traditional baseline
        L_trad = ndata / B
        ax.loglog(B, L_trad, '--', color=COLORS["Traditional"], label=LABELS["Traditional"])
        for nver in nver_list:
            # SHA-256
            tver_sha = float(
                sweep_df[(sweep_df["payload_bits"] == ndata) & (sweep_df["nver"] == nver)][
                    "sha_hashlib_mean_us"
                ].values[0]
            ) * 1e-6
            p_miss_sha = 2.0 ** (-nver)
            L_id_sha = tver_sha + (nver + p_desync * (1.0 - p_miss_sha) * ndata) / B
            ax.loglog(B, L_id_sha, color=COLORS[("SHA-256", nver)], label=LABELS[("SHA-256", nver)])
            # RS-ID
            tver_rsid = float(sweep_df[(sweep_df["payload_bits"]==ndata) & (sweep_df["nver"]==nver)]["rsid_mean_us"].values[0]) * 1e-6
            p_miss_rsid = 2.0 ** (-nver)
            L_id_rsid = tver_rsid + (nver + p_desync * (1.0 - p_miss_rsid) * ndata) / B
            ax.loglog(B, L_id_rsid, color=COLORS[("RS-ID", nver)], label=LABELS[("RS-ID", nver)])
        # 添加 advantage/penalty 区域
        from latency_utils import add_advantage_shading_by_baseline
        add_advantage_shading_by_baseline(ax, B, L_trad)
        ax.set_xlabel("Bandwidth $B$ (bits/s)")
        ax.set_ylabel("Expected latency (s)")
        ax.grid(True, which='both', ls='--', alpha=0.3)
        # 图例放到图外右侧
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=14, frameon=True)
        plt.tight_layout()
        out_path = os.path.join(
            _project_root,
            "thesis_report",
            "figures",
            "plots",
            f"latency_empirical_bandwidth_{ndata}bits.png",
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

if __name__ == "__main__":
    plot_latency_vs_bandwidth()
