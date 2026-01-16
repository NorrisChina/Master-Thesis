#!/usr/bin/env python3
"""K-ID parameter sweep (thesis experiment).

This script simulates a simple K-ID decision rule. It sweeps verifier length
n_ver and tolerance radius K at fixed n_data and bandwidth, and reports:

- Empirical safety miss rate: P(accept | unsafe)
- Theoretical collision risk proxy: |H(acceptable states)| / 2^{n_ver}
- Fallback rate: P(fallback)
- Average latency under the hybrid model:
    L = t_compute + (n_ver / B) + P(fallback) * (n_data / B)
- Speedup over traditional (always transmit n_data): (n_data / B) / L

Notes / assumptions:
- "Accept" is defined as the verifier being in the set of verifiers of
    acceptable states (range [mean-K, mean+K]).
- "Unsafe" means the true value is outside that acceptable range.
- The theoretical risk proxy assumes truncated hash outputs are uniform.

State-space note:
The simulator models the robot state as a continuous value (Normal) that is
clipped to a physical range (e.g., 0..100). To avoid an artificially tiny
discrete state space (which can make empirical miss-rate exactly zero for
n_ver>=8), the value is quantized at a configurable resolution (e.g. 0.1).
The quantized integer is what is hashed.

Example:
  python experiments/kid_parameter_sweep.py --ticks 200000 --nver 4 8 12 16 --k 2 5

Outputs a human-readable table to stdout; optionally writes CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

# Allow running from repo root without installing the package.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _PROJECT_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from idsys.core.idsystems import create_id_system


@dataclass(frozen=True)
class Config:
    n_data_bits: int
    bandwidth_bps: float
    ticks: int
    seed: int
    robot_mean: int
    robot_std: float
    error_dist: str
    uniform_halfwidth: float | None
    physical_min: float
    physical_max: float
    value_resolution: float
    t_compute_s: float
    reset_seed_per_config: bool
    hash_precompute_threshold: int
    system: str
    rsid_tag_pos: Tuple[int, ...]


def _mask_for_bits(n_bits: int) -> int:
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")
    if n_bits > 256:
        raise ValueError("n_bits must be <= 256 for SHA-256 truncation")
    return (1 << n_bits) - 1


def truncated_sha256_int(value: int, n_bits: int) -> int:
    """Truncated SHA-256 as an integer in [0, 2^n_bits)."""
    payload = str(int(value)).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    digest_int = int.from_bytes(digest, byteorder="big", signed=False)
    return digest_int & _mask_for_bits(n_bits)


def _symbols_for_payload(n_data_bits: int, symbol_bits: int) -> int:
    if symbol_bits <= 0:
        raise ValueError("symbol_bits must be positive")
    return max(1, int(math.ceil(n_data_bits / float(symbol_bits))))


def _message_from_state(state_q: int, n_data_bits: int, symbol_bits: int) -> List[int]:
    """Deterministically maps a quantized state to a payload-like message.

    The message length scales as k = ceil(n_data_bits / symbol_bits), matching the
    RS-ID chunking model used in the thesis.
    """
    gf_range = 1 << int(symbol_bits)
    k = _symbols_for_payload(n_data_bits, symbol_bits)
    base = int(state_q) % gf_range
    return [(base + i) % gf_range for i in range(k)]


Codeword = int | Tuple[int, ...]


class VerifierProvider:
    """Provides verifier codewords either via precompute or via cache.

    Supported systems:
      - sha256_trunc: truncated SHA-256 of the quantized scalar state
      - rsid: RS-ID tags computed via the project idsys library
    """

    def __init__(
        self,
        values: Sequence[int],
        n_bits_list: Sequence[int],
        *,
        precompute: bool,
        n_data_bits: int,
        rsid_tag_pos: Tuple[int, ...],
    ):
        self._n_bits_list = [int(x) for x in n_bits_list]
        self._precompute = bool(precompute)
        self._values = [int(v) for v in values]
        self._n_data_bits = int(n_data_bits)
        self._rsid_tag_pos = tuple(int(x) for x in rsid_tag_pos)

        self._pre: Dict[str, Dict[int, Dict[int, Codeword]]] = {}
        self._cache: Dict[str, Dict[int, Dict[int, Codeword]]] = {
            "sha256_trunc": {n: {} for n in self._n_bits_list},
            "rsid": {n: {} for n in self._n_bits_list},
        }
        self._rsid_systems: Dict[int, object] = {}

        if self._precompute:
            distinct_values = list(dict.fromkeys(self._values))
            self._pre["sha256_trunc"] = {}
            for n_bits in self._n_bits_list:
                per_bits: Dict[int, Codeword] = {}
                for v in distinct_values:
                    per_bits[v] = truncated_sha256_int(v, n_bits)
                self._pre["sha256_trunc"][n_bits] = per_bits

            self._pre["rsid"] = {}
            for n_bits in self._n_bits_list:
                id_sys = create_id_system(
                    "RSID",
                    {
                        "gf_exp": int(n_bits),
                        "tag_pos": list(self._rsid_tag_pos),
                    },
                )
                per_bits: Dict[int, Codeword] = {}
                for v in distinct_values:
                    msg = _message_from_state(v, self._n_data_bits, int(n_bits))
                    tag_list = id_sys.send(msg)
                    per_bits[v] = tuple(int(x) for x in tag_list)
                self._pre["rsid"][n_bits] = per_bits

    @property
    def precompute_enabled(self) -> bool:
        return self._precompute

    def _get_rsid_system(self, n_bits: int):
        try:
            return self._rsid_systems[n_bits]
        except KeyError:
            sys_obj = create_id_system(
                "RSID",
                {
                    "gf_exp": int(n_bits),
                    "tag_pos": list(self._rsid_tag_pos),
                },
            )
            self._rsid_systems[n_bits] = sys_obj
            return sys_obj

    def get(self, system: str, value: int, n_bits: int) -> Codeword:
        system = str(system)
        if system not in ("sha256_trunc", "rsid"):
            raise ValueError(f"Unsupported system: {system}")

        if self._precompute:
            return self._pre[system][n_bits][value]

        per_bits = self._cache[system][n_bits]
        try:
            return per_bits[value]
        except KeyError:
            if system == "sha256_trunc":
                cw: Codeword = truncated_sha256_int(int(value), int(n_bits))
            else:
                id_sys = self._get_rsid_system(int(n_bits))
                msg = _message_from_state(int(value), self._n_data_bits, int(n_bits))
                tag_list = id_sys.send(msg)
                cw = tuple(int(x) for x in tag_list)
            per_bits[value] = cw
            return cw

    def per_bits_mapping(self, system: str, n_bits: int) -> Dict[int, Codeword]:
        """Returns a mapping for (system, n_bits). Only valid when precomputed."""
        if not self._precompute:
            raise RuntimeError("per_bits_mapping requires precompute mode")
        return self._pre[str(system)][int(n_bits)]


def prepare_valid_codewords(
    accept_range: Tuple[int, int],
    *,
    system: str,
    n_bits: int,
    vp: VerifierProvider,
) -> Set[Codeword]:
    low, high = accept_range
    return {vp.get(system, v, n_bits) for v in range(low, high + 1)}


def quantize(value: float, physical_min: float, value_resolution: float) -> int:
    """Quantize a physical value into an integer index."""
    return int(round((value - physical_min) / value_resolution))


def quantize_floor(value: float, physical_min: float, value_resolution: float) -> int:
    return int(math.floor((value - physical_min) / value_resolution))


def quantize_ceil(value: float, physical_min: float, value_resolution: float) -> int:
    return int(math.ceil((value - physical_min) / value_resolution))


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def theoretical_unsafe_rate(k_radius: float, std: float) -> float:
    if std <= 0:
        return 0.0
    z = k_radius / std
    return max(0.0, min(1.0, 2.0 * (1.0 - normal_cdf(z))))


def theoretical_unsafe_rate_uniform(k_radius: float, halfwidth: float) -> float:
    """Unsafe rate for X ~ U[mean-halfwidth, mean+halfwidth] (no clipping).

    Unsafe = |X-mean| > k_radius.
    """
    if halfwidth <= 0:
        return 0.0
    if k_radius >= halfwidth:
        return 0.0
    # P(|X-mean| <= K) = (2K) / (2a) = K/a
    return max(0.0, min(1.0, 1.0 - (k_radius / halfwidth)))


@dataclass
class SimResult:
    system: str
    error_dist: str
    n_ver_bits: int
    k_radius: int
    accept_range: Tuple[int, int]
    range_size: int
    miss_count: int
    unsafe_count: int
    fallback_count: int
    ticks: int
    unique_valid_hashes: int
    codeword_bits: int
    domain_unsafe_size: int
    domain_unsafe_collision_count: int

    @property
    def miss_rate_cond_unsafe(self) -> float:
        return (self.miss_count / self.unsafe_count) if self.unsafe_count else 0.0

    @property
    def unsafe_rate(self) -> float:
        return self.unsafe_count / self.ticks if self.ticks else 0.0

    @property
    def fallback_rate(self) -> float:
        return self.fallback_count / self.ticks if self.ticks else 0.0

    @property
    def domain_unsafe_collision_rate(self) -> float:
        return (
            self.domain_unsafe_collision_count / self.domain_unsafe_size
            if self.domain_unsafe_size
            else 0.0
        )


def run_simulation(cfg: Config, n_ver_bits: int, k_radius: int, vp: VerifierProvider) -> SimResult:
    # Deterministic, reproducible behavior.
    seed = cfg.seed if cfg.reset_seed_per_config else None
    rng = np.random.default_rng(seed=seed)

    accept_physical = (cfg.robot_mean - k_radius, cfg.robot_mean + k_radius)
    accept_physical = (max(cfg.physical_min, accept_physical[0]), min(cfg.physical_max, accept_physical[1]))
    accept_range = (
        quantize_floor(accept_physical[0], cfg.physical_min, cfg.value_resolution),
        quantize_ceil(accept_physical[1], cfg.physical_min, cfg.value_resolution),
    )
    range_size = accept_range[1] - accept_range[0] + 1

    valid_codewords = prepare_valid_codewords(
        accept_range,
        system=cfg.system,
        n_bits=n_ver_bits,
        vp=vp,
    )

    # These are optionally computed in main() (exact only when hashes are precomputed).
    domain_unsafe_size = 0
    domain_unsafe_collision_count = 0

    miss_count = 0
    unsafe_count = 0
    fallback_count = 0

    # Generate all samples in a vectorized way.
    if cfg.error_dist == "normal":
        samples = rng.normal(loc=cfg.robot_mean, scale=cfg.robot_std, size=cfg.ticks)
    elif cfg.error_dist == "uniform":
        if cfg.uniform_halfwidth is not None:
            halfwidth = float(cfg.uniform_halfwidth)
        else:
            # Match variance of Normal(std): Var(U[-a,a]) = a^2/3 => a = sqrt(3)*std
            halfwidth = math.sqrt(3.0) * float(cfg.robot_std)
        samples = rng.uniform(
            low=float(cfg.robot_mean) - halfwidth,
            high=float(cfg.robot_mean) + halfwidth,
            size=cfg.ticks,
        )
    else:
        raise ValueError(f"Unsupported --error-dist: {cfg.error_dist}")
    samples = np.clip(samples, cfg.physical_min, cfg.physical_max)

    for v in samples.tolist():
        q = quantize(float(v), cfg.physical_min, cfg.value_resolution)
        cw = vp.get(cfg.system, q, n_ver_bits)
        dt_accepts = cw in valid_codewords
        is_safe = accept_range[0] <= q <= accept_range[1]

        if not is_safe:
            unsafe_count += 1
            if dt_accepts:
                miss_count += 1

        if not dt_accepts:
            fallback_count += 1

    return SimResult(
        system=cfg.system,
        error_dist=cfg.error_dist,
        n_ver_bits=n_ver_bits,
        k_radius=k_radius,
        accept_range=accept_range,
        range_size=range_size,
        miss_count=miss_count,
        unsafe_count=unsafe_count,
        fallback_count=fallback_count,
        ticks=cfg.ticks,
        unique_valid_hashes=len(valid_codewords),
        codeword_bits=(n_ver_bits * len(cfg.rsid_tag_pos) if cfg.system == "rsid" else n_ver_bits),
        domain_unsafe_size=domain_unsafe_size,
        domain_unsafe_collision_count=domain_unsafe_collision_count,
    )


def compute_domain_collision_stats(
    domain_size: int,
    accept_range: Tuple[int, int],
    system: str,
    n_ver_bits: int,
    vp: VerifierProvider,
    valid_codewords: Set[Codeword],
) -> Tuple[int, int]:
    """Exact collision stats over the whole quantized domain.

    Returns (domain_unsafe_size, domain_unsafe_collision_count).
    Only feasible when vp is in precompute mode.
    """
    if not vp.precompute_enabled:
        raise RuntimeError("compute_domain_collision_stats requires precompute mode")

    low, high = accept_range
    domain_unsafe_size = domain_size - max(0, min(domain_size - 1, high) - max(0, low) + 1)
    per_bits = vp.per_bits_mapping(system, n_ver_bits)
    collision_count = 0
    for q in range(domain_size):
        if low <= q <= high:
            continue
        if per_bits[q] in valid_codewords:
            collision_count += 1
    return domain_unsafe_size, collision_count


def compute_latency_metrics(cfg: Config, res: SimResult) -> Dict[str, float]:
    # Traditional always transmits full data.
    latency_traditional_s = cfg.n_data_bits / cfg.bandwidth_bps

    # ID scheme expected latency.
    t_ver_trans_s = res.n_ver_bits / cfg.bandwidth_bps
    fallback_penalty_s = latency_traditional_s
    latency_id_s = cfg.t_compute_s + t_ver_trans_s + res.fallback_rate * fallback_penalty_s

    speedup = (latency_traditional_s / latency_id_s) if latency_id_s > 0 else float("inf")

    # Theoretical proxy: probability that a random verifier matches any acceptable verifier.
    theo_risk = res.unique_valid_hashes / (2 ** res.codeword_bits)

    return {
        "lat_traditional_us": latency_traditional_s * 1e6,
        "lat_id_us": latency_id_s * 1e6,
        "speedup": speedup,
        "t_compute_us": cfg.t_compute_s * 1e6,
        "t_ver_trans_us": t_ver_trans_s * 1e6,
        "fallback_penalty_us": fallback_penalty_s * 1e6,
        "theo_risk": theo_risk,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="K-ID parameter sweep")

    p.add_argument("--n-data-bits", type=int, default=96, help="Payload size in bits (default: 96)")
    p.add_argument("--bandwidth", type=float, default=1e6, help="Bandwidth in bits/s (default: 1e6 = 1 Mbps)")
    p.add_argument("--ticks", type=int, default=200_000, help="Number of simulation ticks (default: 200000)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")

    p.add_argument("--robot-mean", type=int, default=50, help="Mean of the robot state (default: 50)")
    p.add_argument("--robot-std", type=float, default=12.0, help="Std dev of the robot state (default: 12)")
    p.add_argument(
        "--error-dist",
        type=str,
        default="normal",
        choices=["normal", "uniform"],
        help="Prediction error distribution (default: normal)",
    )
    p.add_argument(
        "--uniform-halfwidth",
        type=float,
        default=None,
        help=(
            "Half-width 'a' for uniform noise U[mean-a, mean+a]. "
            "If omitted, uses a=sqrt(3)*robot_std (variance-matched)."
        ),
    )
    p.add_argument("--physical-min", type=float, default=0.0, help="Minimum physical value (default: 0)")
    p.add_argument("--physical-max", type=float, default=100.0, help="Maximum physical value (default: 100)")
    p.add_argument(
        "--value-resolution",
        type=float,
        default=0.1,
        help="Quantization resolution in physical units (default: 0.1; 0.1 => 1001 discrete states over 0..100)",
    )

    p.add_argument(
        "--t-compute-us",
        type=float,
        default=2.13,
        help="Compute overhead t_compute in microseconds (default: 2.13 for SHA-256 @ 96 bits from your table)",
    )

    p.add_argument(
        "--system",
        type=str,
        default="sha256_trunc",
        choices=["sha256_trunc", "rsid", "both"],
        help="Verifier backend to use (default: sha256_trunc)",
    )
    p.add_argument(
        "--rsid-tag-pos",
        type=int,
        nargs="+",
        default=[2],
        help="RS-ID tag positions (default: 2). Used when system=rsid/both.",
    )

    p.add_argument("--nver", type=int, nargs="+", default=[4, 8, 12, 16], help="Verifier sizes in bits")
    p.add_argument("--k", type=int, nargs="+", default=[2, 5, 10, 20], help="Tolerance radii K (range size = 2K+1)")

    p.add_argument(
        "--reset-seed-per-config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, each (n_ver, K) run uses the same RNG seed for comparability (default: true)",
    )

    p.add_argument("--out-csv", type=Path, default=None, help="Optional CSV output path")

    p.add_argument(
        "--hash-precompute-threshold",
        type=int,
        default=200_000,
        help="If quantized domain size <= this, precompute all hashes (default: 200000)",
    )

    p.add_argument(
        "--suggest-std",
        action="store_true",
        help="Print a helper table to choose robot_std given K list (no simulation run)",
    )
    p.add_argument(
        "--std-candidates",
        type=float,
        nargs="+",
        default=[2, 5, 8, 10, 12, 15, 20],
        help="Candidate std values for --suggest-std",
    )

    p.add_argument(
        "--show-domain-collision-rate",
        action="store_true",
        help=(
            "Print the collision rate over the discrete clipped domain (0..100 by default). "
            "If this is 0, empirical miss rate will remain 0 regardless of ticks."
        ),
    )

    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.suggest_std:
        if args.error_dist == "normal":
            print("Std suggestion helper (theoretical unsafe rate under Normal, before clipping):")
            print(f"Ks = {args.k}")
            header = "std" + "".join([f" | unsafe(K={k})" for k in args.k])
        else:
            print("Std suggestion helper (theoretical unsafe rate under Uniform, before clipping):")
            print(f"Ks = {args.k}")
            header = "halfwidth" + "".join([f" | unsafe(K={k})" for k in args.k])
        print(header)
        print("-" * len(header))
        for s in args.std_candidates:
            if args.error_dist == "normal":
                rates = [theoretical_unsafe_rate(float(k), float(s)) for k in args.k]
            else:
                rates = [theoretical_unsafe_rate_uniform(float(k), float(s)) for k in args.k]
            print(f"{s:<4.1f}" + "".join([f" | {r:<11.3f}" for r in rates]))
        # Heuristic: pick the smallest std that gives >=5% unsafe for the largest K.
        k_max = max(args.k)
        chosen = None
        for s in sorted(args.std_candidates):
            if args.error_dist == "normal":
                ok = theoretical_unsafe_rate(float(k_max), float(s)) >= 0.05
            else:
                ok = theoretical_unsafe_rate_uniform(float(k_max), float(s)) >= 0.05
            if ok:
                chosen = s
                break
        if chosen is not None:
            if args.error_dist == "normal":
                print(f"\nHeuristic pick: robot_std ≈ {chosen} (gives >=5% unsafe for K={k_max}).")
            else:
                print(f"\nHeuristic pick: uniform_halfwidth ≈ {chosen} (gives >=5% unsafe for K={k_max}).")
        else:
            print(f"\nHeuristic pick: increase std candidates; none give >=5% unsafe for K={k_max}.")
        return 0

    cfg = Config(
        n_data_bits=args.n_data_bits,
        bandwidth_bps=float(args.bandwidth),
        ticks=int(args.ticks),
        seed=int(args.seed),
        robot_mean=int(args.robot_mean),
        robot_std=float(args.robot_std),
        error_dist=str(args.error_dist),
        uniform_halfwidth=(float(args.uniform_halfwidth) if args.uniform_halfwidth is not None else None),
        physical_min=float(args.physical_min),
        physical_max=float(args.physical_max),
        value_resolution=float(args.value_resolution),
        t_compute_s=float(args.t_compute_us) * 1e-6,
        reset_seed_per_config=bool(args.reset_seed_per_config),
        hash_precompute_threshold=int(args.hash_precompute_threshold),
        system="sha256_trunc",  # may be overridden per-run
        rsid_tag_pos=tuple(int(x) for x in args.rsid_tag_pos),
    )

    n_ver_list = [int(x) for x in args.nver]
    k_list = [int(x) for x in args.k]

    # Build quantized domain.
    if cfg.value_resolution <= 0:
        raise ValueError("--value-resolution must be > 0")
    if cfg.physical_max <= cfg.physical_min:
        raise ValueError("--physical-max must be > --physical-min")
    domain_size = int(round((cfg.physical_max - cfg.physical_min) / cfg.value_resolution)) + 1
    domain_values = list(range(0, domain_size))
    precompute = domain_size <= cfg.hash_precompute_threshold

    vp = VerifierProvider(
        domain_values,
        n_ver_list,
        precompute=precompute,
        n_data_bits=cfg.n_data_bits,
        rsid_tag_pos=cfg.rsid_tag_pos,
    )

    systems: List[str]
    if args.system == "both":
        systems = ["sha256_trunc", "rsid"]
    else:
        systems = [str(args.system)]

    if args.show_domain_collision_rate:
        print(
            f"{'n_ver':<6} | {'K':<3} | {'MissRate P(accept|unsafe)':<26} | {'TheoRisk':<10} | "
            f"{'DomCollRate':<11} | {'FallbackRate':<12} | {'Latency(us)':<12} | {'Speedup':<8}"
        )
        print("-" * 110)
    else:
        print(
            f"{'n_ver':<6} | {'K':<3} | {'MissRate P(accept|unsafe)':<26} | {'TheoRisk':<10} | "
            f"{'FallbackRate':<12} | {'Latency(us)':<12} | {'Speedup':<8}"
        )
        print("-" * 95)

    rows: List[Dict[str, float | int | str]] = []

    for system in systems:
        cfg_system = Config(**{**cfg.__dict__, "system": system})
        print(f"\n=== System: {system} | error_dist={cfg_system.error_dist} ===")
        for k_radius in k_list:
            quantized_range_est = int(round((2.0 * k_radius) / cfg_system.value_resolution)) + 1
            print(
                f"[ Tolerance Radius K = {k_radius} (physical units), value_resolution={cfg_system.value_resolution} "
                f"(~{quantized_range_est} quantized states) ]"
            )
            for n_ver_bits in n_ver_list:
                res = run_simulation(cfg_system, n_ver_bits=n_ver_bits, k_radius=k_radius, vp=vp)

                if args.show_domain_collision_rate and vp.precompute_enabled:
                    accept_physical = (cfg_system.robot_mean - k_radius, cfg_system.robot_mean + k_radius)
                    accept_physical = (
                        max(cfg_system.physical_min, accept_physical[0]),
                        min(cfg_system.physical_max, accept_physical[1]),
                    )
                    accept_range = (
                        quantize_floor(accept_physical[0], cfg_system.physical_min, cfg_system.value_resolution),
                        quantize_ceil(accept_physical[1], cfg_system.physical_min, cfg_system.value_resolution),
                    )
                    valid_codewords = prepare_valid_codewords(
                        accept_range,
                        system=cfg_system.system,
                        n_bits=n_ver_bits,
                        vp=vp,
                    )
                    domain_unsafe_size, domain_unsafe_collision_count = compute_domain_collision_stats(
                        domain_size=domain_size,
                        accept_range=accept_range,
                        system=cfg_system.system,
                        n_ver_bits=n_ver_bits,
                        vp=vp,
                        valid_codewords=valid_codewords,
                    )
                    res.domain_unsafe_size = domain_unsafe_size
                    res.domain_unsafe_collision_count = domain_unsafe_collision_count

                metrics = compute_latency_metrics(cfg_system, res)

                print(
                    (
                        f"{n_ver_bits:<6} | {k_radius:<3} | {res.miss_rate_cond_unsafe:<26.6f} | "
                        f"{metrics['theo_risk']:<10.6f} | {res.domain_unsafe_collision_rate:<11.6f} | "
                        f"{res.fallback_rate:<12.6f} | {metrics['lat_id_us']:<12.2f} | {metrics['speedup']:<7.2f}x"
                        if args.show_domain_collision_rate
                        else f"{n_ver_bits:<6} | {k_radius:<3} | {res.miss_rate_cond_unsafe:<26.6f} | "
                        f"{metrics['theo_risk']:<10.6f} | {res.fallback_rate:<12.6f} | "
                        f"{metrics['lat_id_us']:<12.2f} | {metrics['speedup']:<7.2f}x"
                    )
                )

                rows.append(
                    {
                        "system": cfg_system.system,
                        "error_dist": cfg_system.error_dist,
                        "n_data_bits": cfg.n_data_bits,
                        "bandwidth_bps": cfg.bandwidth_bps,
                        "ticks": cfg.ticks,
                        "seed": cfg.seed,
                        "robot_mean": cfg.robot_mean,
                        "robot_std": cfg.robot_std,
                        "uniform_halfwidth": cfg.uniform_halfwidth if cfg.uniform_halfwidth is not None else "",
                        "physical_min": cfg.physical_min,
                        "physical_max": cfg.physical_max,
                        "value_resolution": cfg.value_resolution,
                        "t_compute_us": cfg.t_compute_s * 1e6,
                        "n_ver_bits": res.n_ver_bits,
                        "k_radius": res.k_radius,
                        "accept_low": res.accept_range[0],
                        "accept_high": res.accept_range[1],
                        "range_size": res.range_size,
                        "unique_valid_hashes": res.unique_valid_hashes,
                        "codeword_bits": res.codeword_bits,
                        "domain_unsafe_size": res.domain_unsafe_size,
                        "domain_unsafe_collision_count": res.domain_unsafe_collision_count,
                        "domain_unsafe_collision_rate": res.domain_unsafe_collision_rate,
                        "unsafe_count": res.unsafe_count,
                        "miss_count": res.miss_count,
                        "fallback_count": res.fallback_count,
                        "miss_rate_accept_given_unsafe": res.miss_rate_cond_unsafe,
                        "fallback_rate": res.fallback_rate,
                        "theo_risk": metrics["theo_risk"],
                        "lat_traditional_us": metrics["lat_traditional_us"],
                        "lat_id_us": metrics["lat_id_us"],
                        "speedup": metrics["speedup"],
                    }
                )
            print("-" * 95)

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved CSV: {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
