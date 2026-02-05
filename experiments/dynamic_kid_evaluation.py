#!/usr/bin/env python3
r"""Dynamic K-ID evaluation (Random Walk with Drift).

This script is a cleaned-up/extended variant of `experiments/random_walk_drift_sweep.py`.
Key differences:
- Uses physical tolerance radius `R` (same unit as the state) instead of legacy `K` naming.
- Optional miss cooldown to avoid per-tick retry storms when `E_t > R`.
- Records both inter-check intervals and inter-correction intervals.
- Avoids running the simulation twice when writing IFT CSV.

Model
-----
State is a 1D scalar random walk:
    S_t = S_{t-1} + δ_t
DT predictor is zero-order hold (ZOH) unless corrected:
    \hat S_t stays constant unless a correction is triggered.

Protocol gate:
- If E_t = |S_t - \hat S_t| <= R: silent
- If E_t > R: attempt verification event
    - miss with probability p_miss_event(R, n_ver)
    - on success, correct: \hat S_t <- S_t

Collision proxy (ROM-style):
    p_miss_event(R, n_ver) ≈ |A_R| / 2^{n_ver}
    |A_R| ≈ 2*floor(R/Δ) + 1
where Δ is `--value-resolution`.

Outputs
-------
- Summary CSV (one row per (R, n_ver))
- Optional IFT CSV in long format with `ift_type in {check, correction}`.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class SweepConfig:
    ticks: int
    replicates: int
    base_seed: int

    r_values: Tuple[float, ...]
    nver_values: Tuple[int, ...]

    step_dist: str  # 'uniform' or 'gaussian'
    step_param: float  # uniform: halfwidth Δ_step; gaussian: std σ_step

    value_resolution: float  # Δ used in |A_R|
    danger_threshold: float  # D

    miss_cooldown: int  # after a miss, skip checks for this many ticks


@dataclass
class SimStats:
    ticks: int
    r_radius: float
    n_ver_bits: int
    step_dist: str
    step_param: float
    value_resolution: float
    danger_threshold: float
    miss_cooldown: int

    p_miss_event: float

    check_rate: float
    miss_rate_given_check: float
    correction_rate: float

    # Semantic "unsafe" = outside tolerance: |S_t - \hat S_t| > R
    breach_rate: float
    breach_streak_mean: float
    breach_streak_p95: float
    breach_streak_max: int

    # Physical unsafe = outside physical danger boundary: |S_t - \hat S_t| > D
    danger_rate: float
    danger_streak_mean: float
    danger_streak_p95: float
    danger_streak_max: int

    max_error_mean: float
    max_error_p95: float


def p_miss_from_accept_set(r_radius: float, n_ver_bits: int, value_resolution: float) -> float:
    if n_ver_bits <= 0:
        raise ValueError("n_ver_bits must be positive")
    if value_resolution <= 0:
        raise ValueError("value_resolution must be positive")

    accept_count = int(2 * math.floor(float(r_radius) / float(value_resolution)) + 1)
    accept_count = max(1, accept_count)

    denom = float(2**int(n_ver_bits))
    p = accept_count / denom
    # Clamp to probability range; if p>=1 this corresponds to guaranteed collision.
    return max(0.0, min(1.0, p))


def sample_step(rng: np.random.Generator, *, step_dist: str, step_param: float) -> float:
    if step_param < 0:
        raise ValueError("step_param must be non-negative")
    if step_dist == "uniform":
        return float(rng.uniform(-step_param, step_param))
    if step_dist == "gaussian":
        return float(rng.normal(0.0, step_param))
    raise ValueError(f"Unsupported step_dist: {step_dist}")


def simulate_one(
    *,
    ticks: int,
    rng: np.random.Generator,
    r_radius: float,
    n_ver_bits: int,
    step_dist: str,
    step_param: float,
    value_resolution: float,
    danger_threshold: float,
    miss_cooldown: int,
    s0: float = 50.0,
) -> Dict[str, object]:
    robot_state = float(s0)
    dt_state = float(s0)

    p_miss_event = p_miss_from_accept_set(r_radius, n_ver_bits, value_resolution)

    check_count = 0
    miss_count = 0
    correction_count = 0

    # Intervals
    last_check_t: int | None = None
    inter_check_intervals: List[int] = []

    last_correction_t: int | None = None
    inter_correction_intervals: List[int] = []

    breach_count = 0
    breach_streaks: List[int] = []
    current_breach = 0

    danger_count = 0
    danger_streaks: List[int] = []
    current_danger = 0

    max_error = 0.0

    cooldown_remaining = 0

    for t in range(int(ticks)):
        robot_state += sample_step(rng, step_dist=step_dist, step_param=step_param)

        error = abs(robot_state - dt_state)
        if error > max_error:
            max_error = float(error)

        # K-ID logic with ZOH predictor
        if error > r_radius:
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
            else:
                check_count += 1
                if last_check_t is not None:
                    inter_check_intervals.append(int(t) - int(last_check_t))
                last_check_t = int(t)

                is_miss = bool(rng.random() < p_miss_event)
                if is_miss:
                    miss_count += 1
                    if miss_cooldown > 0:
                        cooldown_remaining = int(miss_cooldown)
                else:
                    correction_count += 1
                    dt_state = float(robot_state)
                    if last_correction_t is not None:
                        inter_correction_intervals.append(int(t) - int(last_correction_t))
                    last_correction_t = int(t)
        else:
            # Within tolerance: silent
            pass

        # Semantic breach streaks: |error| > R
        if error > r_radius:
            breach_count += 1
            current_breach += 1
        else:
            if current_breach > 0:
                breach_streaks.append(current_breach)
                current_breach = 0

        # Physical danger streaks: |error| > D
        if error > danger_threshold:
            danger_count += 1
            current_danger += 1
        else:
            if current_danger > 0:
                danger_streaks.append(current_danger)
                current_danger = 0

    if current_breach > 0:
        breach_streaks.append(current_breach)
    if current_danger > 0:
        danger_streaks.append(current_danger)

    return {
        "p_miss_event": p_miss_event,
        "check_count": check_count,
        "miss_count": miss_count,
        "correction_count": correction_count,
        "inter_check_intervals": inter_check_intervals,
        "inter_correction_intervals": inter_correction_intervals,
        "breach_count": breach_count,
        "breach_streaks": breach_streaks,
        "danger_count": danger_count,
        "danger_streaks": danger_streaks,
        "max_error": max_error,
    }


def _mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _pctl(xs: Sequence[float], p: float) -> float:
    return float(np.percentile(xs, p)) if xs else 0.0


def aggregate_stats(
    *,
    cfg: SweepConfig,
    r_radius: float,
    n_ver_bits: int,
    ift_rows_out: List[List[object]] | None,
) -> SimStats:
    max_errors: List[float] = []

    breach_streak_means: List[float] = []
    breach_streak_p95s: List[float] = []
    breach_streak_maxes: List[int] = []

    danger_streak_means: List[float] = []
    danger_streak_p95s: List[float] = []
    danger_streak_maxes: List[int] = []

    check_rates: List[float] = []
    miss_rates_given_check: List[float] = []
    correction_rates: List[float] = []
    breach_rates: List[float] = []
    danger_rates: List[float] = []

    p_miss_event = p_miss_from_accept_set(r_radius, n_ver_bits, cfg.value_resolution)

    for i in range(int(cfg.replicates)):
        seed_i = int(cfg.base_seed) + i
        rng = np.random.default_rng(seed_i)
        out = simulate_one(
            ticks=cfg.ticks,
            rng=rng,
            r_radius=r_radius,
            n_ver_bits=n_ver_bits,
            step_dist=cfg.step_dist,
            step_param=cfg.step_param,
            value_resolution=cfg.value_resolution,
            danger_threshold=cfg.danger_threshold,
            miss_cooldown=cfg.miss_cooldown,
        )

        check_count = int(out["check_count"])
        miss_count = int(out["miss_count"])
        correction_count = int(out["correction_count"])
        breach_count = int(out["breach_count"])
        danger_count = int(out["danger_count"])

        breach_streaks = list(out["breach_streaks"])
        danger_streaks = list(out["danger_streaks"])
        max_error = float(out["max_error"])

        max_errors.append(max_error)

        if breach_streaks:
            breach_streak_means.append(float(np.mean(breach_streaks)))
            breach_streak_p95s.append(float(np.percentile(breach_streaks, 95)))
            breach_streak_maxes.append(int(max(breach_streaks)))
        else:
            breach_streak_means.append(0.0)
            breach_streak_p95s.append(0.0)
            breach_streak_maxes.append(0)

        if danger_streaks:
            danger_streak_means.append(float(np.mean(danger_streaks)))
            danger_streak_p95s.append(float(np.percentile(danger_streaks, 95)))
            danger_streak_maxes.append(int(max(danger_streaks)))
        else:
            danger_streak_means.append(0.0)
            danger_streak_p95s.append(0.0)
            danger_streak_maxes.append(0)

        check_rates.append(check_count / float(cfg.ticks))
        miss_rates_given_check.append((miss_count / float(check_count)) if check_count else 0.0)
        correction_rates.append(correction_count / float(cfg.ticks))
        breach_rates.append(breach_count / float(cfg.ticks))
        danger_rates.append(danger_count / float(cfg.ticks))

        if ift_rows_out is not None:
            for ift in list(out.get("inter_check_intervals", [])):
                ift_rows_out.append(
                    [
                        cfg.step_dist,
                        f"{cfg.step_param:.6g}",
                        int(cfg.ticks),
                        int(cfg.base_seed),
                        int(i),
                        f"{float(r_radius):.6g}",
                        int(n_ver_bits),
                        f"{cfg.value_resolution:.6g}",
                        f"{cfg.danger_threshold:.6g}",
                        int(cfg.miss_cooldown),
                        "check",
                        int(ift),
                    ]
                )
            for ift in list(out.get("inter_correction_intervals", [])):
                ift_rows_out.append(
                    [
                        cfg.step_dist,
                        f"{cfg.step_param:.6g}",
                        int(cfg.ticks),
                        int(cfg.base_seed),
                        int(i),
                        f"{float(r_radius):.6g}",
                        int(n_ver_bits),
                        f"{cfg.value_resolution:.6g}",
                        f"{cfg.danger_threshold:.6g}",
                        int(cfg.miss_cooldown),
                        "correction",
                        int(ift),
                    ]
                )

    return SimStats(
        ticks=int(cfg.ticks),
        r_radius=float(r_radius),
        n_ver_bits=int(n_ver_bits),
        step_dist=str(cfg.step_dist),
        step_param=float(cfg.step_param),
        value_resolution=float(cfg.value_resolution),
        danger_threshold=float(cfg.danger_threshold),
        miss_cooldown=int(cfg.miss_cooldown),
        p_miss_event=float(p_miss_event),
        check_rate=_mean(check_rates),
        miss_rate_given_check=_mean(miss_rates_given_check),
        correction_rate=_mean(correction_rates),
        breach_rate=_mean(breach_rates),
        breach_streak_mean=_mean(breach_streak_means),
        breach_streak_p95=_mean(breach_streak_p95s),
        breach_streak_max=int(max(breach_streak_maxes) if breach_streak_maxes else 0),
        danger_rate=_mean(danger_rates),
        danger_streak_mean=_mean(danger_streak_means),
        danger_streak_p95=_mean(danger_streak_p95s),
        danger_streak_max=int(max(danger_streak_maxes) if danger_streak_maxes else 0),
        max_error_mean=_mean(max_errors),
        max_error_p95=_pctl(max_errors, 95),
    )


def write_summary_csv(*, rows: Sequence[SimStats], out_csv: Path, replicates: int, base_seed: int) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "step_dist",
                "step_param",
                "ticks",
                "replicates",
                "seed",
                "r_radius",
                "n_ver_bits",
                "value_resolution",
                "danger_threshold",
                "miss_cooldown",
                "p_miss_event",
                "check_rate",
                "miss_rate_given_check",
                "correction_rate",
                "breach_rate",
                "breach_streak_mean",
                "breach_streak_p95",
                "breach_streak_max",
                "danger_rate",
                "danger_streak_mean",
                "danger_streak_p95",
                "danger_streak_max",
                "max_error_mean",
                "max_error_p95",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.step_dist,
                    f"{r.step_param:.6g}",
                    r.ticks,
                    int(replicates),
                    int(base_seed),
                    f"{r.r_radius:.6g}",
                    r.n_ver_bits,
                    f"{r.value_resolution:.6g}",
                    f"{r.danger_threshold:.6g}",
                    int(r.miss_cooldown),
                    f"{r.p_miss_event:.6g}",
                    f"{r.check_rate:.6g}",
                    f"{r.miss_rate_given_check:.6g}",
                    f"{r.correction_rate:.6g}",
                    f"{r.breach_rate:.6g}",
                    f"{r.breach_streak_mean:.6g}",
                    f"{r.breach_streak_p95:.6g}",
                    r.breach_streak_max,
                    f"{r.danger_rate:.6g}",
                    f"{r.danger_streak_mean:.6g}",
                    f"{r.danger_streak_p95:.6g}",
                    r.danger_streak_max,
                    f"{r.max_error_mean:.6g}",
                    f"{r.max_error_p95:.6g}",
                ]
            )


def write_ift_csv(*, header: List[str], rows: List[List[object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Dynamic K-ID evaluation (random walk drift), using physical radius R")

    p.add_argument("--ticks", type=int, default=50000)
    p.add_argument("--replicates", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)

    # Use R as the primary parameter. Keep --k as a legacy alias (same numeric meaning).
    p.add_argument("--r", type=float, nargs="+", default=None, help="Physical tolerance radius R (same unit as state)")
    p.add_argument("--k", type=float, nargs="+", default=None, help="Legacy alias for --r (deprecated)")
    p.add_argument("--nver", type=int, nargs="+", default=[12])

    p.add_argument("--step-dist", choices=["uniform", "gaussian"], default="gaussian")
    p.add_argument("--step-param", type=float, default=1.0, help="Uniform halfwidth Δ_step or Gaussian std σ_step")

    p.add_argument("--value-resolution", type=float, default=0.1, help="Quantization step Δ used in |A_R|")
    p.add_argument("--danger-threshold", type=float, default=15.0, help="Physical danger threshold D")

    p.add_argument(
        "--miss-cooldown",
        type=int,
        default=0,
        help="After a miss, skip checks for this many ticks to avoid per-tick retry storms",
    )

    p.add_argument("--out-summary-csv", type=Path, default=Path("experiments/results/dynamic_kid_eval_summary.csv"))
    p.add_argument(
        "--out-ift-csv",
        type=Path,
        default=None,
        help="Optional: write IFT samples (long format, includes check + correction intervals)",
    )

    args = p.parse_args()

    r_values = args.r if args.r is not None else args.k
    if r_values is None:
        r_values = [10.0]

    cfg = SweepConfig(
        ticks=int(args.ticks),
        replicates=int(args.replicates),
        base_seed=int(args.seed),
        r_values=tuple(float(x) for x in r_values),
        nver_values=tuple(int(x) for x in args.nver),
        step_dist=str(args.step_dist),
        step_param=float(args.step_param),
        value_resolution=float(args.value_resolution),
        danger_threshold=float(args.danger_threshold),
        miss_cooldown=int(args.miss_cooldown),
    )

    ift_rows: List[List[object]] | None = [] if args.out_ift_csv is not None else None

    summary_rows: List[SimStats] = []
    for r_radius in cfg.r_values:
        for n_ver_bits in cfg.nver_values:
            # Soft warning when collision is guaranteed under the proxy.
            accept_count = int(2 * math.floor(float(r_radius) / float(cfg.value_resolution)) + 1)
            if accept_count >= 2 ** int(n_ver_bits):
                print(
                    f"WARNING: |A_R|={accept_count} >= 2^n_ver={2**int(n_ver_bits)} => p_miss_event clamps to 1.0 "
                    f"(R={r_radius:g}, n_ver={n_ver_bits})"
                )

            summary_rows.append(
                aggregate_stats(cfg=cfg, r_radius=float(r_radius), n_ver_bits=int(n_ver_bits), ift_rows_out=ift_rows)
            )

    write_summary_csv(rows=summary_rows, out_csv=Path(args.out_summary_csv), replicates=cfg.replicates, base_seed=cfg.base_seed)
    print(f"Wrote summary CSV: {args.out_summary_csv}")

    if args.out_ift_csv is not None and ift_rows is not None:
        header = [
            "step_dist",
            "step_param",
            "ticks",
            "seed",
            "replicate",
            "r_radius",
            "n_ver_bits",
            "value_resolution",
            "danger_threshold",
            "miss_cooldown",
            "ift_type",
            "ift_steps",
        ]
        write_ift_csv(header=header, rows=ift_rows, out_csv=Path(args.out_ift_csv))
        print(f"Wrote IFT CSV: {args.out_ift_csv} ({len(ift_rows)} rows)")

    # Quick stdout summary
    for r in summary_rows:
        print(
            f"step={r.step_dist}({r.step_param:g}) R={r.r_radius:g} nver={r.n_ver_bits}: "
            f"p_miss_event={r.p_miss_event:.3g}, check_rate={r.check_rate:.3g}, "
            f"corr_rate={r.correction_rate:.3g}, danger_p95={r.danger_streak_p95:.2f}, "
            f"max_err_p95={r.max_error_p95:.2f}"
        )


if __name__ == "__main__":
    main()
