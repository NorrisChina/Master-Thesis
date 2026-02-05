#!/usr/bin/env python3
r"""Dynamic K-ID evaluation: Random Walk with Drift (thesis experiment).

Motivation
----------
The static K-ID sweep (i.i.d. states) captures *per-tick* miss probabilities,
but cannot represent accumulation of error over time. This script implements a
minimal dynamic model:

- Robot state follows a random walk: S_t = S_{t-1} + δ_t
- Digital Twin (DT) uses Zero-Order Hold (ZOH): \hat S_t stays constant unless
  a correction is triggered.
- K-ID decision:
    - If |S_t - \hat S_t| <= K : no message (masked drift)
    - Else: attempt verification; with probability p_miss the DT *misses* and
      does not correct; otherwise the DT corrects (fallback) and sets \hat S_t=S_t.

We quantify dynamic risk using streak metrics, e.g. duration of unsafe operation
when the drift exceeds a physical danger threshold D.

Collision / miss model
----------------------
To keep the experiment aligned with the ROM-style analysis in Chapter 3, we
model the miss probability during a verification event as:

    p_miss(K, n_ver) ≈ |A_K| / 2^{n_ver}

where |A_K| is the number of *acceptable quantized states* within ±K.
For a 1D scalar state quantized at resolution Δ, we approximate:

    |A_K| = 2*floor(K/Δ) + 1

This intentionally exposes two distinct dynamic risks:
- Masking risk (large K): drift grows without any checks while within K.
- Collision risk (small n_ver and/or large K): checks happen but miss, allowing
  drift to keep growing beyond K.

Outputs
-------
Writes a CSV summary per (K, n_ver, step_dist) configuration.
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

    k_values: Tuple[float, ...]
    nver_values: Tuple[int, ...]

    step_dist: str  # 'uniform' or 'gaussian'
    step_param: float  # uniform: halfwidth Δ_step; gaussian: std σ_step

    value_resolution: float
    danger_threshold: float


@dataclass
class SimStats:
    ticks: int
    k_radius: float
    n_ver_bits: int
    step_dist: str
    step_param: float
    value_resolution: float
    danger_threshold: float

    p_miss_event: float

    check_rate: float
    miss_rate_given_check: float
    correction_rate: float

    # Semantic "unsafe" = outside tolerance: |S_t - \hat S_t| > K
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


def p_miss_from_accept_set(k_radius: float, n_ver_bits: int, value_resolution: float) -> float:
    if n_ver_bits <= 0:
        raise ValueError("n_ver_bits must be positive")
    if value_resolution <= 0:
        raise ValueError("value_resolution must be positive")

    # Thesis accept-set model:
    #   |A_K| = 2*floor(K/Δ) + 1
    #   p_miss(K, n_ver) ≈ |A_K| / 2^{n_ver}
    accept_count = int(2 * math.floor(float(k_radius) / float(value_resolution)) + 1)
    accept_count = max(1, accept_count)
    p = accept_count / float(2 ** int(n_ver_bits))
    return max(0.0, min(1.0, float(p)))


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
    k_radius: float,
    n_ver_bits: int,
    step_dist: str,
    step_param: float,
    value_resolution: float,
    danger_threshold: float,
    s0: float = 50.0,
) -> Dict[str, object]:
    # States
    robot_state = float(s0)
    dt_state = float(s0)

    # Probabilistic miss model per *verification event*
    p_miss_event = p_miss_from_accept_set(k_radius, n_ver_bits, value_resolution)

    check_count = 0
    miss_count = 0
    correction_count = 0

    # Inter-check (inter-verifier) intervals: time between consecutive check events.
    # In K-ID, a verifier is transmitted exactly when E_t > K.
    last_check_t: int | None = None
    inter_check_intervals: List[int] = []

    # Inter-correction (inter-full-update) intervals: time between consecutive successful recoveries.
    # A full-state update (n_data) is transmitted only when a check occurs and is not missed.
    last_correction_t: int | None = None
    inter_correction_intervals: List[int] = []

    breach_count = 0
    breach_streaks: List[int] = []
    current_breach = 0

    danger_count = 0
    danger_streaks: List[int] = []
    current_danger = 0

    max_error = 0.0

    for _t in range(int(ticks)):
        robot_state += sample_step(rng, step_dist=step_dist, step_param=step_param)

        error = abs(robot_state - dt_state)
        if error > max_error:
            max_error = float(error)

        # K-ID logic with ZOH predictor
        if error <= k_radius:
            # Masked drift: no verification, no correction
            pass
        else:
            check_count += 1
            if last_check_t is not None:
                inter_check_intervals.append(int(_t) - int(last_check_t))
            last_check_t = int(_t)
            is_miss = bool(rng.random() < p_miss_event)
            if is_miss:
                miss_count += 1
            else:
                correction_count += 1
                if last_correction_t is not None:
                    inter_correction_intervals.append(int(_t) - int(last_correction_t))
                last_correction_t = int(_t)
                dt_state = float(robot_state)

        # Semantic breach streaks: |error| > K
        if error > k_radius:
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
    k_radius: float,
    n_ver_bits: int,
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

    # Inter-transmission times (time steps between check events).
    inter_check_all: List[int] = []

    # p_miss_event is deterministic given (K, n_ver, Δ)
    p_miss_event = p_miss_from_accept_set(k_radius, n_ver_bits, cfg.value_resolution)

    for i in range(int(cfg.replicates)):
        seed_i = int(cfg.base_seed) + i
        rng = np.random.default_rng(seed_i)
        out = simulate_one(
            ticks=cfg.ticks,
            rng=rng,
            k_radius=k_radius,
            n_ver_bits=n_ver_bits,
            step_dist=cfg.step_dist,
            step_param=cfg.step_param,
            value_resolution=cfg.value_resolution,
            danger_threshold=cfg.danger_threshold,
        )

        check_count = int(out["check_count"])
        miss_count = int(out["miss_count"])
        correction_count = int(out["correction_count"])
        inter_check_all.extend(int(x) for x in list(out.get("inter_check_intervals", [])))
        breach_count = int(out["breach_count"])
        breach_streaks = list(out["breach_streaks"])
        danger_count = int(out["danger_count"])
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

    return SimStats(
        ticks=int(cfg.ticks),
        k_radius=float(k_radius),
        n_ver_bits=int(n_ver_bits),
        step_dist=str(cfg.step_dist),
        step_param=float(cfg.step_param),
        value_resolution=float(cfg.value_resolution),
        danger_threshold=float(cfg.danger_threshold),
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


def write_ift_csv(
    *,
    cfg: SweepConfig,
    k_radius: float,
    n_ver_bits: int,
    out_csv: Path,
) -> None:
    """Write per-sample intervals for survival/CDF plots."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "step_dist",
                "step_param",
                "ticks",
                "seed",
                "replicate",
                "k_radius",
                "n_ver_bits",
                "value_resolution",
                "danger_threshold",
                "ift_type",
                "ift_steps",
            ]
        )

        for i in range(int(cfg.replicates)):
            seed_i = int(cfg.base_seed) + i
            rng = np.random.default_rng(seed_i)
            out = simulate_one(
                ticks=cfg.ticks,
                rng=rng,
                k_radius=k_radius,
                n_ver_bits=n_ver_bits,
                step_dist=cfg.step_dist,
                step_param=cfg.step_param,
                value_resolution=cfg.value_resolution,
                danger_threshold=cfg.danger_threshold,
            )

            for ift in list(out.get("inter_check_intervals", [])):
                w.writerow(
                    [
                        cfg.step_dist,
                        f"{cfg.step_param:.6g}",
                        int(cfg.ticks),
                        int(cfg.base_seed),
                        int(i),
                        f"{float(k_radius):.6g}",
                        int(n_ver_bits),
                        f"{cfg.value_resolution:.6g}",
                        f"{cfg.danger_threshold:.6g}",
                        "check",
                        int(ift),
                    ]
                )

            for ift in list(out.get("inter_correction_intervals", [])):
                w.writerow(
                    [
                        cfg.step_dist,
                        f"{cfg.step_param:.6g}",
                        int(cfg.ticks),
                        int(cfg.base_seed),
                        int(i),
                        f"{float(k_radius):.6g}",
                        int(n_ver_bits),
                        f"{cfg.value_resolution:.6g}",
                        f"{cfg.danger_threshold:.6g}",
                        "correction",
                        int(ift),
                    ]
                )


def write_csv(*, rows: Sequence[SimStats], out_csv: Path, replicates: int, base_seed: int) -> None:
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
                "k_radius",
                "n_ver_bits",
                "value_resolution",
                "danger_threshold",
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
                    f"{r.k_radius:.6g}",
                    r.n_ver_bits,
                    f"{r.value_resolution:.6g}",
                    f"{r.danger_threshold:.6g}",
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


def main() -> None:
    p = argparse.ArgumentParser(description="Random Walk & Correction (dynamic K-ID) sweep")
    p.add_argument("--ticks", type=int, default=20000)
    p.add_argument("--replicates", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--k", type=float, nargs="+", default=[2, 5, 10, 20])
    p.add_argument("--nver", type=int, nargs="+", default=[8, 12, 16])

    p.add_argument("--step-dist", choices=["uniform", "gaussian"], default="uniform")
    p.add_argument("--step-param", type=float, default=1.0, help="Uniform halfwidth Δ_step or Gaussian std σ_step")

    p.add_argument("--value-resolution", type=float, default=1.0, help="Quantization step Δ used in |A_K|")
    p.add_argument("--danger-threshold", type=float, default=20.0, help="Physical danger threshold D")

    p.add_argument("--out-csv", type=Path, default=Path("experiments/results/random_walk_drift_sweep.csv"))
    p.add_argument(
        "--out-ift-csv",
        type=Path,
        default=None,
        help="Optional: write per-sample inter-check intervals (time between transmissions) to CSV",
    )
    args = p.parse_args()

    cfg = SweepConfig(
        ticks=int(args.ticks),
        replicates=int(args.replicates),
        base_seed=int(args.seed),
        k_values=tuple(float(x) for x in args.k),
        nver_values=tuple(int(x) for x in args.nver),
        step_dist=str(args.step_dist),
        step_param=float(args.step_param),
        value_resolution=float(args.value_resolution),
        danger_threshold=float(args.danger_threshold),
    )

    rows: List[SimStats] = []
    for k_radius in cfg.k_values:
        for n_ver_bits in cfg.nver_values:
            rows.append(aggregate_stats(cfg=cfg, k_radius=k_radius, n_ver_bits=n_ver_bits))

            if args.out_ift_csv is not None:
                # Write a separate file per (K, n_ver) to keep files manageable.
                stem = Path(args.out_ift_csv)
                out_ift = stem
                if stem.suffix.lower() == ".csv":
                    out_ift = stem.with_name(stem.stem + f"_K{float(k_radius):g}_nver{int(n_ver_bits)}" + stem.suffix)
                else:
                    out_ift = stem.with_name(stem.name + f"_K{float(k_radius):g}_nver{int(n_ver_bits)}.csv")
                write_ift_csv(cfg=cfg, k_radius=k_radius, n_ver_bits=n_ver_bits, out_csv=out_ift)

    # Patch header placeholders so we don't repeat on every row.
    # (Keep the CSV schema stable for plotting.)
    write_csv(rows=rows, out_csv=args.out_csv, replicates=cfg.replicates, base_seed=cfg.base_seed)

    # Quick stdout summary
    print(f"Wrote {len(rows)} rows to {args.out_csv}")
    for r in rows:
        print(
            f"step={r.step_dist}({r.step_param:g}) K={r.k_radius:g} nver={r.n_ver_bits}: "
            f"p_miss={r.p_miss_event:.3g}, check={r.check_rate:.3g}, "
            f"breach_p95={r.breach_streak_p95:.2f}, danger_p95={r.danger_streak_p95:.2f}, "
            f"max_err_p95={r.max_error_p95:.2f}"
        )


if __name__ == "__main__":
    main()
