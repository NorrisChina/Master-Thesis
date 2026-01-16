
import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from statistics import mean, stdev

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


def measure_t_ver(system_type: str, n_data_bits: int, n_ver_bits: int, N_tver: int, *, num_batches: int = 100) -> TVerStats:
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
    # Convert bits to bytes for idsys library
    n_data_bytes = (n_data_bits + 7) // 8
    
    data_bytes = os.urandom(n_data_bytes)
    data_list = list(data_bytes)
    
    params = {"tag_pos": [2]}
    if system_type == "RSID":
        params["gf_exp"] = n_ver_bits * 2 # A safe value
    
    id_sys = create_id_system(system_type, params)
    
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
    args = parser.parse_args()

    N_tver = args.N_tver
    num_batches = args.batches
    
    # Parameters to test
    ndata_bits_list = [96, 4001]
    nver_bits_list = [4, 16]

    results: dict[int, dict[int, dict[str, TVerStats]]] = {}

    print(f"Starting t_ver measurements (N={N_tver})...")

    for n_data in ndata_bits_list:
        results[n_data] = {}
        for n_ver in nver_bits_list:
            results[n_data][n_ver] = {}
            
            # Measure for SHA256ID
            t_ver_sha = measure_t_ver("SHA256ID", n_data, n_ver, N_tver, num_batches=num_batches)
            results[n_data][n_ver]['sha256'] = t_ver_sha
            
            # Measure for RSID
            t_ver_rsid = measure_t_ver("RSID", n_data, n_ver, N_tver, num_batches=num_batches)
            results[n_data][n_ver]['rsid'] = t_ver_rsid

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
