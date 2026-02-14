#!/usr/bin/env bash
set -euo pipefail

# Regenerates all figures referenced by the thesis LaTeX.
# Run from repo root: ./scripts/reproduce_thesis_figures.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Prefer the repo venv if present, but don't require it.
if [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.venv/bin/activate"
fi

if command -v python >/dev/null 2>&1; then
  PYTHON=python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  echo "ERROR: Could not find python or python3 on PATH." >&2
  exit 127
fi

"$PYTHON" experiments/gaussian_uniform.py
"$PYTHON" experiments/derta.py

"$PYTHON" experiments/plot_tver_sweep_summary.py
"$PYTHON" experiments/plot_latency_vs_bandwidth_consistent.py
"$PYTHON" experiments/latency_vs_desync.py --ndata-list 96 4001
"$PYTHON" experiments/plot_latency_ratio_vs_ndata.py

"$PYTHON" experiments/plot_kid_heatmaps.py \
  --csv experiments/results/kid_sweep_gaussian_both.csv \
  --system sha256_trunc \
  --out thesis_report/figures/plots/kid_heatmap_gaussian_sha256.png \
  --title "K-ID sweep (Gaussian)"

"$PYTHON" experiments/plot_kid_heatmaps.py \
  --csv experiments/results/kid_sweep_uniform_both.csv \
  --system sha256_trunc \
  --out thesis_report/figures/plots/kid_heatmap_uniform_sha256.png \
  --title "K-ID sweep (Uniform)"

"$PYTHON" experiments/plot_kid_parameter_sweep.py \
  --csv experiments/results/kid_sweep_gaussian_both.csv \
  --out-dir thesis_report/figures/plots \
  --prefix kid_sweep_gaussian_both \
  --only-pareto-overall-risk

"$PYTHON" experiments/plot_kid_parameter_sweep.py \
  --csv experiments/results/kid_sweep_uniform_both.csv \
  --out-dir thesis_report/figures/plots \
  --prefix kid_sweep_uniform_both \
  --only-pareto-overall-risk

"$PYTHON" experiments/plot_random_walk_drift_sweep.py \
  --csv experiments/results/random_walk_drift_sweep_gaussian_step16_D15.csv \
  --out-dir thesis_report/figures/plots \
  --name-suffix gaussian_step16_D15

"$PYTHON" experiments/plot_random_walk_drift_sweep.py \
  --csv experiments/results/random_walk_drift_sweep_gaussian_step1_D15.csv \
  --out-dir thesis_report/figures/plots \
  --name-suffix step1_D15_sigma1_p01 \
  --escape-demo \
  --escape-step-dist gaussian \
  --escape-step-param 1 \
  --escape-danger 15 \
  --escape-p-collision 0.1

"$PYTHON" experiments/plot_random_walk_drift_sweep.py \
  --csv experiments/results/random_walk_drift_sweep_gaussian_step1_D15.csv \
  --out-dir thesis_report/figures/plots \
  --name-suffix step1_D15_sigma16_p01 \
  --escape-demo \
  --escape-step-dist gaussian \
  --escape-step-param 16 \
  --escape-danger 15 \
  --escape-p-collision 0.1

"$PYTHON" experiments/plot_random_walk_ift_cdf.py --ift-csv \
  experiments/results/random_walk_intervals_gaussian_step16_D15_res0p1_K2_nver8.csv \
  experiments/results/random_walk_intervals_gaussian_step16_D15_res0p1_K5_nver8.csv \
  experiments/results/random_walk_intervals_gaussian_step16_D15_res0p1_K10_nver8.csv \
  experiments/results/random_walk_intervals_gaussian_step16_D15_res0p1_K20_nver8.csv \
  --ift-type correction --mode by_nver --nver 8 --xquantile 0.99 \
  --out thesis_report/figures/plots/random_walk_drift_gaussian_step16_D15_sri_cdf_res0p1_nver8.png

"$PYTHON" experiments/plot_random_walk_ift_cdf.py --ift-csv \
  experiments/results/random_walk_intervals_gaussian_step16_D15_res0p1_K2_nver16.csv \
  experiments/results/random_walk_intervals_gaussian_step16_D15_res0p1_K5_nver16.csv \
  experiments/results/random_walk_intervals_gaussian_step16_D15_res0p1_K10_nver16.csv \
  experiments/results/random_walk_intervals_gaussian_step16_D15_res0p1_K20_nver16.csv \
  --ift-type correction --mode by_nver --nver 16 --xquantile 0.99 \
  --out thesis_report/figures/plots/random_walk_drift_gaussian_step16_D15_sri_cdf_res0p1_nver16.png

echo "Done. Figures are in thesis_report/figures/."