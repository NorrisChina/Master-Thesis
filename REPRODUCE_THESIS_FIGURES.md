# Reproduce thesis figures

This repo contains both (a) checked-in figure assets under `thesis_report/figures/**` and (b) Python scripts (mostly under `experiments/`) that generated them.

This document maps each figure referenced by the LaTeX sources to the generator code and provides commands to regenerate them.

## Prerequisites

- Python 3.10+ recommended
- Python packages used by the figure pipeline:
  - `numpy`, `matplotlib`
  - `pandas` (for latency/tver plots)

If you use the provided `.venv/`, activate it first.

All commands below assume you run them from the repository root.

## Figures used by the thesis

### Chapter 3 (concept figures)

These are static assets with no generator script in this repo:

- `thesis_report/figures/trad.png`
- `thesis_report/figures/ID.png`
- `thesis_report/figures/k-id.png`

### Chapter 4

- `thesis_report/figures/error_models_sigma16.png`
  - Generator: `experiments/gaussian_uniform.py`
  - Command:
    - `python experiments/gaussian_uniform.py`

- `thesis_report/figures/decoder_cost_multiline_annotated.png`
  - Generator: `experiments/derta.py`
  - Command:
    - `python experiments/derta.py`

### Chapter 5

- `thesis_report/figures/plots/tver_sweep_summary.png`
  - Generator: `experiments/plot_tver_sweep_summary.py`
  - Inputs: `experiments/results/sweep_nver_tver_detail_ci.csv`
  - Command:
    - `python experiments/plot_tver_sweep_summary.py`

- `thesis_report/figures/plots/latency_empirical_bandwidth_96bits.png`
- `thesis_report/figures/plots/latency_empirical_bandwidth_4001bits.png`
  - Generator: `experiments/plot_latency_vs_bandwidth_consistent.py`
  - Inputs: `experiments/results/sweep_nver_tver_detail_ci.csv`
  - Command (generates both figures):
    - `python experiments/plot_latency_vs_bandwidth_consistent.py`

- `thesis_report/figures/plots/latency_vs_desync_96bits.png`
- `thesis_report/figures/plots/latency_vs_desync_4001bits.png`
  - Generator: `experiments/latency_vs_desync.py`
  - Inputs: `experiments/results/sweep_nver_tver_detail_ci.csv`
  - Command (generates both figures):
    - `python experiments/latency_vs_desync.py --ndata-list 96 4001`

- `thesis_report/figures/plots/latency_vs_ndata_backends_nver16.png`
  - Generator: `experiments/plot_latency_ratio_vs_ndata.py`
  - Input: `experiments/results/sweep_nver_tver_detail_ci.csv`
  - Command:
    - `python experiments/plot_latency_ratio_vs_ndata.py`

#### K-ID sweep figures

These plots are generated from K-ID sweep CSVs.

- `thesis_report/figures/plots/kid_heatmap_gaussian_sha256.png`
  - Generator: `experiments/plot_kid_heatmaps.py`
  - Input: `experiments/results/kid_sweep_gaussian_both.csv`
  - Command:
    - `python experiments/plot_kid_heatmaps.py --csv experiments/results/kid_sweep_gaussian_both.csv --system sha256_trunc --out thesis_report/figures/plots/kid_heatmap_gaussian_sha256.png --title "K-ID sweep (Gaussian)"`

- `thesis_report/figures/plots/kid_heatmap_uniform_sha256.png`
  - Generator: `experiments/plot_kid_heatmaps.py`
  - Input: `experiments/results/kid_sweep_uniform_both.csv`
  - Command:
    - `python experiments/plot_kid_heatmaps.py --csv experiments/results/kid_sweep_uniform_both.csv --system sha256_trunc --out thesis_report/figures/plots/kid_heatmap_uniform_sha256.png --title "K-ID sweep (Uniform)"`

- `thesis_report/figures/plots/kid_sweep_gaussian_both_pareto_overall_risk.png`
  - Generator: `experiments/plot_kid_parameter_sweep.py`
  - Input: `experiments/results/kid_sweep_gaussian_both.csv`
  - Command:
    - `python experiments/plot_kid_parameter_sweep.py --csv experiments/results/kid_sweep_gaussian_both.csv --out-dir thesis_report/figures/plots --prefix kid_sweep_gaussian_both --only-pareto-overall-risk`

- `thesis_report/figures/plots/kid_sweep_uniform_both_pareto_overall_risk.png`
  - Generator: `experiments/plot_kid_parameter_sweep.py`
  - Input: `experiments/results/kid_sweep_uniform_both.csv`
  - Command:
    - `python experiments/plot_kid_parameter_sweep.py --csv experiments/results/kid_sweep_uniform_both.csv --out-dir thesis_report/figures/plots --prefix kid_sweep_uniform_both --only-pareto-overall-risk`

To regenerate the CSVs from scratch:

- Generator: `experiments/kid_parameter_sweep.py`
- Example (adjust to match your thesis settings):
  - `python experiments/kid_parameter_sweep.py --ticks 200000 --seed 42 --nver 4 8 12 16 --k 20 50 100 200 300 --value-resolution 0.1 --out-csv experiments/results/kid_sweep_gaussian_both.csv --error-dist normal`

#### Random-walk drift figures

There are two kinds of assets referenced by the thesis:

1) Aggregate sweep plots (breach streak / max error) from a sweep CSV

- `thesis_report/figures/plots/random_walk_drift_gaussian_gaussian_step16_D15_breach_streak_p95.png`
- `thesis_report/figures/plots/random_walk_drift_gaussian_gaussian_step16_D15_max_error_p95.png`
  - Generator: `experiments/plot_random_walk_drift_sweep.py`
  - Input: `experiments/results/random_walk_drift_sweep_gaussian_step16_D15.csv`
  - Command (writes multiple `random_walk_drift_gaussian_gaussian_step16_D15_*.png` files):
    - `python experiments/plot_random_walk_drift_sweep.py --csv experiments/results/random_walk_drift_sweep_gaussian_step16_D15.csv --out-dir thesis_report/figures/plots --name-suffix gaussian_step16_D15`

2) Representative time-series “escape” demos

- `thesis_report/figures/plots/random_walk_drift_gaussian_step1_D15_sigma1_p01_escape_timeseries.png`
  - Generator: `experiments/plot_random_walk_drift_sweep.py`
  - Command:
    - `python experiments/plot_random_walk_drift_sweep.py --csv experiments/results/random_walk_drift_sweep_gaussian_step1_D15.csv --out-dir thesis_report/figures/plots --name-suffix step1_D15_sigma1_p01 --escape-demo --escape-step-dist gaussian --escape-step-param 1 --escape-danger 15 --escape-p-collision 0.1`

- `thesis_report/figures/plots/random_walk_drift_gaussian_step1_D15_sigma16_p01_escape_timeseries.png`
  - Generator: `experiments/plot_random_walk_drift_sweep.py`
  - Command:
    - `python experiments/plot_random_walk_drift_sweep.py --csv experiments/results/random_walk_drift_sweep_gaussian_step1_D15.csv --out-dir thesis_report/figures/plots --name-suffix step1_D15_sigma16_p01 --escape-demo --escape-step-dist gaussian --escape-step-param 16 --escape-danger 15 --escape-p-collision 0.1`

3) SRI CDF plots (successive recovery intervals)

- `thesis_report/figures/plots/random_walk_drift_gaussian_step16_D15_sri_cdf_res0p1_nver8.png`
- `thesis_report/figures/plots/random_walk_drift_gaussian_step16_D15_sri_cdf_res0p1_nver16.png`
  - Generator: `experiments/plot_random_walk_ift_cdf.py`
  - Inputs: the interval sample CSVs under `experiments/results/random_walk_intervals_gaussian_step16_D15_res0p1_K*_nver*.csv`
  - Commands:
    - `python experiments/plot_random_walk_ift_cdf.py --ift-csv \
        experiments/results/random_walk_intervals_gaussian_step16_D15_res0p1_K2_nver8.csv \
        experiments/results/random_walk_intervals_gaussian_step16_D15_res0p1_K5_nver8.csv \
        experiments/results/random_walk_intervals_gaussian_step16_D15_res0p1_K10_nver8.csv \
        experiments/results/random_walk_intervals_gaussian_step16_D15_res0p1_K20_nver8.csv \
        --ift-type correction --mode by_nver --nver 8 --xquantile 0.99 \
        --out thesis_report/figures/plots/random_walk_drift_gaussian_step16_D15_sri_cdf_res0p1_nver8.png`

    - `python experiments/plot_random_walk_ift_cdf.py --ift-csv \
        experiments/results/random_walk_intervals_gaussian_step16_D15_res0p1_K2_nver16.csv \
        experiments/results/random_walk_intervals_gaussian_step16_D15_res0p1_K5_nver16.csv \
        experiments/results/random_walk_intervals_gaussian_step16_D15_res0p1_K10_nver16.csv \
        experiments/results/random_walk_intervals_gaussian_step16_D15_res0p1_K20_nver16.csv \
        --ift-type correction --mode by_nver --nver 16 --xquantile 0.99 \
        --out thesis_report/figures/plots/random_walk_drift_gaussian_step16_D15_sri_cdf_res0p1_nver16.png`

To regenerate the interval CSVs from scratch, use either:

- `experiments/random_walk_drift_sweep.py` (original sweep)
- `experiments/dynamic_kid_evaluation.py` (cleaned-up/extended, records both check + correction intervals)

Both can write IFT/interval CSVs via `--out-ift-csv`.

## Build the thesis PDF

Once the figures are regenerated, build the PDF:

- `cd thesis_report && latexmk -pdf thesis.tex`

## Cleanup candidates (review before deleting)

If your post-submission goal is a minimal repo that can **regenerate the currently included thesis figures**, the scripts below do not appear in the dependency chain for those figures.

Important: some of these may still be useful if you want to regenerate *upstream CSVs from scratch*, produce additional (non-included) plots, or export tables.

Status (2026-02-07)

The following scripts have already been removed from `experiments/` to keep the repo minimal. A backup tarball is available at:

- `archive/prune_2026-02-07/experiments_prune_backup.tgz`

Removed scripts:

- `experiments/latency_empirical_bandwidth.py` (older alternative to `plot_latency_vs_bandwidth_consistent.py`)
- `experiments/plot_latency_vs_bandwidth_backends.py` (not referenced by the current LaTeX)
- `experiments/latency_vs_ndata_fixed_nver.py` (not referenced by the current LaTeX)
- `experiments/plot_compute_overhead_fraction_vs_bandwidth.py` (not referenced by the current LaTeX)
- `experiments/plot_kid_decoder_cost.py` (not referenced by the current LaTeX)
- `experiments/plot_lidar_scalability_tver.py`, `experiments/lidar_scalability_tver.py`, `experiments/export_tver_lidar_table.py` (LiDAR scalability artifacts not referenced by the current LaTeX)
- `experiments/fig2_fig3.py` (not referenced by the current LaTeX)
- `experiments/export_kid_pareto_table.py` (table export helper, not referenced by the current LaTeX)

Also removed (tver measurement/table pipeline; restore from backup if you want full “from scratch” regeneration of `experiments/results/sweep_nver_tver_detail_ci.csv`):

- `experiments/measure_tver.py`
- `experiments/measure_tver_ci_table.py`
- `experiments/plot_tver_ci_table.py`
- `experiments/sweep_nver_tver_detail.py` (older variant; `sweep_nver_tver_detail_ci.py` is the one used by the current figure scripts)

Optional further minimization

If you only want to regenerate thesis figures from the checked-in CSVs under `experiments/results/` (and do not need to regenerate those CSVs from scratch), then the following scripts are optional and can be removed as well:

- `experiments/kid_parameter_sweep.py`
- `experiments/sweep_nver_tver_detail_ci.py`
- `experiments/random_walk_drift_sweep.py`
- `experiments/dynamic_kid_evaluation.py`
