import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
import csv
from statistics import mean, stdev
from typing import Literal

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from idsys.core.idsystems import create_id_system


@dataclass(frozen=True)
class TVerStats:
    mean_us: float
    std_us: float
    ci95_half_us: float
    num_batches: int
    total_iterations: int


def _make_sha_message(n_data_bits: int) -> list[int]:
    """SHA256ID expects a byte-like message list."""
    n_data_bytes = (int(n_data_bits) + 7) // 8
    return list(os.urandom(n_data_bytes))


def _make_rsid_message(n_data_bits: int, n_ver_bits: int) -> list[int]:
    """Model packing of payload bits for concatenated RS-ID (RS2ID).

    We pack into GF(2^{2*n_ver}) symbols (k_i=2), so
    k = ceil(n_data_bits / (2*n_ver_bits)).
    """
    n_data_bits_i = int(n_data_bits)
    n_ver_bits_i = int(n_ver_bits)
    symbol_bits = 2 * n_ver_bits_i
    k = int(math.ceil(n_data_bits_i / float(symbol_bits)))
    gf_range = 1 << symbol_bits

    bytes_per_symbol = int(math.ceil(symbol_bits / 8.0))
    raw = os.urandom(bytes_per_symbol * k)
    out: list[int] = []
    for i in range(k):
        chunk = raw[i * bytes_per_symbol : (i + 1) * bytes_per_symbol]
        out.append(int.from_bytes(chunk, byteorder="little", signed=False) % gf_range)
    return out


def measure_t_ver(
    system_type: str,
    n_data_bits: int,
    n_ver_bits: int,
    N_tver: int,
    *,
    num_batches: int = 100,
    rsid_gf_exp_mult: int = 1,
    rsid_message_mode: Literal["packed", "bytes"] = "packed",
    sha_backend: Literal["idcodes", "hashlib"] = "hashlib",
) -> TVerStats:
    """
    Measures the average verification time for a given ID system and estimates
    a 95% confidence interval for the mean.

    Implementation detail:
    Timing every single call can distort measurements due to timer overhead.
    Instead, we time batches of operations. Each batch yields one sample of
    per-operation runtime, enabling a variance estimate.

    Args:
        system_type (str): The type of the ID system to test (e.g., "RSID").
        n_data_bits (int): The size of the data in bits.
        n_ver_bits (int): The size of the verifier in bits.
        N_tver (int): The number of iterations to average over.

    Returns:
        TVerStats: mean/std and an approximate 95% CI half-width (microseconds).
    """
    n_data_bits_i = int(n_data_bits)
    n_ver_bits_i = int(n_ver_bits)

    params = {}

    if system_type == "SHA256ID":
        # Match thesis notation: tag size is n_ver bits.
        params["gf_exp"] = n_ver_bits_i
        params["backend"] = str(sha_backend)
        data_list = _make_sha_message(n_data_bits_i)
    elif system_type in ("RSID", "RS2ID"):
        # RS-ID runs over GF(2^{gf_exp}); we expose a multiplier for legacy usage.
        params["gf_exp"] = int(rsid_gf_exp_mult) * n_ver_bits_i
        params["tag_pos"] = [2]
        params["tag_pos_in"] = [2]
        if rsid_message_mode == "bytes":
            # Legacy behavior: treat payload as bytes list.
            data_list = _make_sha_message(n_data_bits_i)
        else:
            # Preferred: pack payload into GF(2^{2*n_ver}) symbols.
            data_list = _make_rsid_message(n_data_bits_i, n_ver_bits_i)
    else:
        raise ValueError(f"Unsupported system_type: {system_type}")
    
    id_sys = create_id_system("RS2ID" if system_type in ("RSID", "RS2ID") else system_type, params)
    
    # Warm-up phase
    for _ in range(200):
        id_sys.send(data_list)

    if N_tver <= 1:
        raise ValueError("N_tver must be > 1 to compute a confidence interval")

    num_batches = int(max(5, min(num_batches, N_tver)))
    base_batch = N_tver // num_batches
    remainder = N_tver % num_batches
    if base_batch == 0:
        base_batch = 1
        num_batches = N_tver
        remainder = 0

    per_op_s_samples = []
    total_done = 0
    for batch_idx in range(num_batches):
        batch_size = base_batch + (1 if batch_idx < remainder else 0)
        if batch_size <= 0:
            continue
        t0 = time.perf_counter()
        for _ in range(batch_size):
            id_sys.send(data_list)  # includes verifier generation
        t1 = time.perf_counter()
        total_done += batch_size
        per_op_s_samples.append((t1 - t0) / batch_size)

    mu_s = mean(per_op_s_samples)
    s_s = stdev(per_op_s_samples)

    # Normal approximation; with >=30 batches this is very close to the t critical value.
    z = 1.96
    ci95_half_s = z * s_s / math.sqrt(len(per_op_s_samples))

    return TVerStats(
        mean_us=mu_s * 1e6,
        std_us=s_s * 1e6,
        ci95_half_us=ci95_half_s * 1e6,
        num_batches=len(per_op_s_samples),
        total_iterations=total_done,
    )

def main():
    parser = argparse.ArgumentParser(description="Measure and print t_ver for RSID and SHA256ID.")
    parser.add_argument('--N-tver', type=int, default=100000, help='Number of iterations for t_ver measurement.')
    parser.add_argument('--batches', type=int, default=100, help='Number of timing batches for CI estimation (default: 100).')
    parser.add_argument('--rsid-gf-exp-mult', type=int, default=1, help='Use gf_exp = rsid_gf_exp_mult * n_ver_bits for RSID (default: 1).')
    parser.add_argument('--rsid-message-mode', choices=['packed', 'bytes'], default='packed', help='RSID input modeling: packed GF symbols (preferred) or raw bytes list (legacy).')
    parser.add_argument('--sha-backend', choices=['idcodes', 'hashlib'], default='hashlib', help='SHA256 backend: idcodes (legacy) or hashlib (OpenSSL-backed, typical paper baseline).')
    parser.add_argument('--mode', choices=['fixed-bits', 'fixed-symbols'], default='fixed-bits', help='Whether n_data_bits is fixed per row, or computed as k_symbols*n_ver_bits for monotonic scaling vs n_ver.')
    parser.add_argument('--k-symbols', type=int, default=512, help='For --mode fixed-symbols: use n_data_bits = k_symbols * n_ver_bits.')
    parser.add_argument('--ndata-bits', type=int, nargs='+', default=None, help='Override n_data bit sizes (default: 96 4001).')
    parser.add_argument('--nver-bits', type=int, nargs='+', default=None, help='Override n_ver bit sizes (default: 4 8 12 16).')
    parser.add_argument('--out-csv', type=str, default=None, help='Optional output CSV path (one row per (n_data, n_ver) with SHA and RS-ID stats).')
    args = parser.parse_args()

    N_tver = args.N_tver
    num_batches = args.batches
    
    # Parameters to test
    ndata_bits_list = [96, 4001] if args.ndata_bits is None else [int(x) for x in args.ndata_bits]
    nver_bits_list = [4, 8, 12, 16] if args.nver_bits is None else [int(x) for x in args.nver_bits]

    results: dict[int, dict[int, dict[str, TVerStats]]] = {}

    print(f"Starting t_ver measurements (N={N_tver})...")

    if args.mode == 'fixed-symbols':
        ndata_bits_list = [int(args.k_symbols) * int(n_ver) for n_ver in nver_bits_list]

    for n_data in ndata_bits_list:
        results[n_data] = {}
        for n_ver in nver_bits_list:
            results[n_data][n_ver] = {}
            
            # Measure for SHA256ID
            t_ver_sha = measure_t_ver(
                "SHA256ID",
                n_data,
                n_ver,
                N_tver,
                num_batches=num_batches,
                rsid_gf_exp_mult=int(args.rsid_gf_exp_mult),
                rsid_message_mode=str(args.rsid_message_mode),
                sha_backend=str(args.sha_backend),
            )
            results[n_data][n_ver]['sha256'] = t_ver_sha
            
            # Measure for RSID
            t_ver_rsid = measure_t_ver(
                "RSID",
                n_data,
                n_ver,
                N_tver,
                num_batches=num_batches,
                rsid_gf_exp_mult=int(args.rsid_gf_exp_mult),
                rsid_message_mode=str(args.rsid_message_mode),
            )
            results[n_data][n_ver]['rsid'] = t_ver_rsid

    # Optional CSV export (useful for thesis plot scripts).
    if args.out_csv:
        out_dir = os.path.dirname(str(args.out_csv))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        rows: list[dict[str, object]] = []
        for n_data in ndata_bits_list:
            for n_ver in nver_bits_list:
                sha = results[n_data][n_ver]['sha256']
                rsid = results[n_data][n_ver]['rsid']
                rows.append(
                    {
                        "n_data_bits": int(n_data),
                        "n_data_bytes": (int(n_data) + 7) // 8,
                        "n_ver_bits": int(n_ver),
                        "N_tver": int(N_tver),
                        "batches": int(num_batches),
                        "sha_mean_us": float(sha.mean_us),
                        "sha_ci95_half_us": float(sha.ci95_half_us),
                        "rsid_mean_us": float(rsid.mean_us),
                        "rsid_ci95_half_us": float(rsid.ci95_half_us),
                    }
                )

        with open(str(args.out_csv), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote CSV: {args.out_csv}")

    # Print results in a formatted table
    print("\n--- t_ver Measurement Results (mean ± 95% CI half-width, in microseconds) ---")
    print(
        f"{'n_data (bits)':<15} | {'n_ver (bits)':<15} | "
        f"{'SHA-256 t_ver (us)':<24} | {'RSID t_ver (us)':<24}"
    )
    print("-" * 92)

    for n_data in ndata_bits_list:
        for n_ver in nver_bits_list:
            sha = results[n_data][n_ver]['sha256']
            rsid = results[n_data][n_ver]['rsid']
            sha_str = f"{sha.mean_us:.2f} ± {sha.ci95_half_us:.2f}"
            rsid_str = f"{rsid.mean_us:.2f} ± {rsid.ci95_half_us:.2f}"
            print(f"{n_data:<15} | {n_ver:<15} | {sha_str:<24} | {rsid_str:<24}")

    # Also print a short stability summary (useful for thesis write-up).
    all_ci = []
    for n_data in ndata_bits_list:
        for n_ver in nver_bits_list:
            all_ci.append(results[n_data][n_ver]['sha256'].ci95_half_us)
            all_ci.append(results[n_data][n_ver]['rsid'].ci95_half_us)
    print("\nCI summary:")
    print(f"  batches={num_batches}, N={N_tver}, max CI half-width = {max(all_ci):.3f} µs")

if __name__ == "__main__":
    main()
